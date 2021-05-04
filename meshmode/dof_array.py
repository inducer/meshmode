__copyright__ = "Copyright (C) 2020 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import operator as op
import numpy as np
from typing import Optional, Iterable, Any, Tuple, Union
from functools import partial, update_wrapper
from numbers import Number
import threading
from contextlib import contextmanager

from pytools import MovedFunctionDeprecationWrapper
from pytools import single_valued, memoize_in
from pytools.obj_array import obj_array_vectorize

from meshmode.array_context import (
        ArrayContext, make_loopy_program,
        ArrayContainerWithArithmetic)
from meshmode.array_context import (
        thaw as _thaw, freeze as _freeze,
        map_array_container, multimap_array_container)

__doc__ = """
.. autoclass:: DOFArray

.. autofunction:: obj_or_dof_array_vectorize
.. autofunction:: obj_or_dof_array_vectorized
.. autofunction:: obj_or_dof_array_vectorize_n_args
.. autofunction:: obj_or_dof_array_vectorized_n_args

.. autofunction:: thaw
.. autofunction:: freeze

.. autofunction:: flatten
.. autofunction:: unflatten

.. autofunction:: array_context_for_pickling
"""


# {{{ DOFArray

class DOFArray(ArrayContainerWithArithmetic):
    r"""This array type holds degree-of-freedom arrays for use with
    :class:`~meshmode.discretization.Discretization`,
    with one entry in the :class:`DOFArray` for each
    :class:`~meshmode.discretization.ElementGroupBase`.
    The arrays contained within a :class:`DOFArray`
    are expected to be logically two-dimensional, with shape
    ``(nelements, ndofs_per_element)``, where ``nelements`` is the same as
    :attr:`~meshmode.discretization.ElementGroupBase.nelements`
    of the associated group.
    ``ndofs_per_element`` is typically, but not necessarily, the same as
    :attr:`~meshmode.discretization.ElementGroupBase.nunit_dofs`
    of the associated group. The entries in this array are further arrays managed by
    :attr:`array_context`.

    One main purpose of this class is to describe the data structure,
    i.e. when a :class:`DOFArray` occurs inside of further numpy object array,
    the level representing the array of element groups can be recognized (by
    people and programs).

    .. attribute:: array_context

        An :class:`meshmode.array_context.ArrayContext`.

    .. attribute:: entry_dtype

        The (assumed uniform) :class:`numpy.dtype` of the group arrays
        contained in this instance.

    .. automethod:: __len__

    The following methods and attributes are implemented to mimic the
    functionality of :class:`~numpy.ndarray`\ s. They require the
    :class:`DOFArray` to be :func:`thaw`\ ed.

    .. attribute:: shape
    .. attribute:: size
    .. automethod:: copy
    .. automethod:: fill
    .. automethod:: conj
    .. attribute:: real
    .. attribute:: imag

    Inherits from :class:`~meshmode.array_context.ArrayContainerWithArithmetic`.

    Basic in-place operations are also supported. Note that not all array types
    provided by :class:`meshmode.array_context.ArrayContext` implementations
    support in-place operations. Those based on lazy evaluation are a salient
    example.

    .. automethod:: __iadd__
    .. automethod:: __isub__
    .. automethod:: __imul__
    .. automethod:: __itruediv__
    .. automethod:: __iand__
    .. automethod:: __ixor__
    .. automethod:: __ior__

    .. note::

        :class:`DOFArray` instances can be pickled and unpickled while the context
        manager :class:`array_context_for_pickling` is active. If, for an array
        to be pickled, the :class:`~meshmode.array_context.ArrayContext` given to
        :func:`array_context_for_pickling` does not agree with :attr:`array_context`,
        the array is frozen and rethawed. If :attr:`array_context` is *None*,
        the :class:`DOFArray` is :func:`thaw`\ ed into the array context given
        to :func:`array_context_for_pickling`.
    """

    def __init__(self, actx: Optional[ArrayContext], data: Tuple[Any]):
        if not (actx is None or isinstance(actx, ArrayContext)):
            raise TypeError("actx must be of type ArrayContext")

        if not isinstance(data, tuple):
            raise TypeError("'data' argument must be a tuple")

        self._array_context = actx
        self._data = data

    # Tell numpy that we would like to do our own array math, thank you very much.
    # (numpy arrays have priority 0.)
    __array_priority__ = 10

    @property
    def array_context(self):
        return self._array_context

    @property
    def entry_dtype(self):
        return single_valued(subary.dtype for subary in self._data)

    @classmethod
    def from_list(cls, actx: Optional[ArrayContext], res_list) -> "DOFArray":
        r"""Create a :class:`DOFArray` from a list of arrays
        (one per :class:`~meshmode.discretization.ElementGroupBase`).

        :arg actx: If *None*, the arrays in *res_list* must be
            :meth:`~meshmode.array_context.ArrayContext.thaw`\ ed.
        """
        from warnings import warn
        warn("DOFArray.from_list is deprecated and will disappear in 2021.",
                DeprecationWarning, stacklevel=2)
        if not (actx is None or isinstance(actx, ArrayContext)):
            raise TypeError("actx must be of type ArrayContext")

        return cls(actx, tuple(res_list))

    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return f"DOFArray({repr(self._data)})"

    # {{{ array container protocol

    def as_iterable(self):
        return ((i, subary) for i, subary in enumerate(self._data))

    @classmethod
    def from_iterable(cls, actx, iterable):
        iterable = list(iterable)

        result = [None] * len(iterable)
        for i, subary in iterable:
            result[i] = subary

        assert all(subary is not None for subary in result)
        return DOFArray(actx, tuple(result))

    # }}}

    # {{{ sequence protocol

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]

    def __iter__(self):
        return iter(self._data)

    # }}}

    # {{{ ndarray interface

    def _like_me(self, data):
        return DOFArray(self.array_context, tuple(data))

    @property
    def shape(self):
        return (len(self),)

    @property
    def size(self):
        return len(self)

    def copy(self):
        return self._like_me([subary.copy() for subary in self])

    def fill(self, value):
        for subary in self:
            subary.fill(value)

    def conj(self):
        return self._like_me([subary.conj() for subary in self])

    conjugate = conj

    @property
    def real(self):
        return self._like_me([subary.real for subary in self])

    @property
    def imag(self):
        return self._like_me([subary.imag for subary in self])

    # }}}

    # {{{ in-place arithmetic

    def _ibop(self, f, arg):
        """Generic in-place binary operator without any broadcast support."""
        if isinstance(arg, DOFArray):
            if len(self) != len(arg):
                raise ValueError("'DOFArray' objects in binary operator must "
                        "have the same length: {len(self)} != {len(arg)}")

            for i, subary in enumerate(self):
                f(subary, arg[i])
        elif isinstance(arg, Number):
            for subary in self:
                f(subary, arg)
        else:
            raise NotImplementedError(f"operation for type {type(arg).__name__}")

        return self

    def __iadd__(self, arg): return self._ibop(op.iadd, arg)             # noqa: E704
    def __isub__(self, arg): return self._ibop(op.isub, arg)            # noqa: E704
    def __imul__(self, arg): return self._ibop(op.imul, arg)            # noqa: E704
    def __itruediv__(self, arg): return self._ibop(op.itruediv, arg)    # noqa: E704
    def __imod__(self, arg): return self._ibop(op.imod, arg)            # noqa: E704

    def __iand__(self, arg): return self._ibop(op.iand, arg)            # noqa: E704
    def __ixor__(self, arg): return self._ibop(op.ixor, arg)            # noqa: E704
    def __ior__(self, arg): return self._ibop(op.ior, arg)              # noqa: E704

    # }}}

    # {{{ pickling

    def __getstate__(self):
        try:
            actx = _ARRAY_CONTEXT_FOR_PICKLING_TLS.actx
        except AttributeError:
            actx = None

        if actx is None:
            raise RuntimeError("DOFArray instances can only be pickled while "
                    "array_context_for_pickling is active.")

        ary = self

        if self.array_context is not actx:
            ary = thaw(actx, freeze(self))

        return [actx.to_numpy(ary_i) for ary_i in ary._data]

    def __setstate__(self, state):
        try:
            actx = _ARRAY_CONTEXT_FOR_PICKLING_TLS.actx
        except AttributeError:
            actx = None

        if actx is None:
            raise RuntimeError("DOFArray instances can only be unpickled while "
                    "array_context_for_pickling is active.")

        self.array_context = actx
        self._data = tuple([actx.from_numpy(ary_i) for ary_i in state])

    # }}}

# }}}


# {{{ deprecated

def obj_or_dof_array_vectorized(f):
    wrapper = partial(map_array_container, f)
    update_wrapper(wrapper, f)
    return wrapper


def obj_or_dof_array_vectorized_n_args(f):
    # See also obj_array_vectorized_n_args fixes in
    # https://github.com/inducer/pytools/pull/76
    #
    # Unfortunately, this can't use partial(), as the callable returned by it
    # will not be turned into a bound method upon attribute access.
    # This may happen here, because the decorator *could* be used
    # on methods, since it can "look past" the leading `self` argument.
    # Only exactly function objects receive this treatment.
    #
    # Spec link:
    # https://docs.python.org/3/reference/datamodel.html#the-standard-type-hierarchy
    # (under "Instance Methods", quote as of Py3.9.4)
    # > Also notice that this transformation only happens for user-defined functions;
    # > other callable objects (and all non-callable objects) are retrieved
    # > without transformation.

    def wrapper(*args):
        return multimap_array_container(f, *args)

    update_wrapper(wrapper, f)
    return wrapper


obj_or_dof_array_vectorize = \
        MovedFunctionDeprecationWrapper(map_array_container, deadline="2022")
obj_or_dof_array_vectorize_n_args = \
        MovedFunctionDeprecationWrapper(multimap_array_container, deadline="2022")

thaw = MovedFunctionDeprecationWrapper(_thaw, deadline="2022")
freeze = MovedFunctionDeprecationWrapper(_freeze, deadline="2022")

# }}}


# {{{ flatten / unflatten

def flatten(ary: Union[DOFArray, np.ndarray]) -> Any:
    r"""Convert a :class:`DOFArray` into a "flat" array of degrees of freedom,
    where the resulting type of the array is given by the
    :attr:`DOFArray.array_context`.

    Array elements are laid out contiguously, with the element group
    index varying slowest, element index next, and intra-element DOF
    index fastest.

    Vectorizes over object arrays of :class:`DOFArray`\ s.
    """
    if isinstance(ary, np.ndarray):
        return obj_array_vectorize(flatten, ary)

    group_sizes = [grp_ary.shape[0] * grp_ary.shape[1] for grp_ary in ary]
    group_starts = np.cumsum([0] + group_sizes)

    actx = ary.array_context

    @memoize_in(actx, (flatten, "flatten_prg"))
    def prg():
        return make_loopy_program(
            "{[iel,idof]: 0<=iel<nelements and 0<=idof<ndofs_per_element}",
            """result[grp_start + iel*ndofs_per_element + idof] \
                = grp_ary[iel, idof]""",
            name="flatten")

    result = actx.empty(group_starts[-1], dtype=ary.entry_dtype)

    for grp_start, grp_ary in zip(group_starts, ary):
        actx.call_loopy(prg(), grp_ary=grp_ary, result=result, grp_start=grp_start)

    return result


def _unflatten(
        actx: ArrayContext, group_shapes: Iterable[Tuple[int, int]], ary: Any
        ) -> DOFArray:
    @memoize_in(actx, (unflatten, "unflatten_prg"))
    def prg():
        return make_loopy_program(
            "{[iel,idof]: 0<=iel<nelements and 0<=idof<ndofs_per_element}",
            "result[iel, idof] = ary[grp_start + iel*ndofs_per_element + idof]",
            name="unflatten")

    group_sizes = [nel * ndof for nel, ndof in group_shapes]

    if ary.size != sum(group_sizes):
        raise ValueError("array has size %d, expected %d"
                % (ary.size, sum(group_sizes)))

    group_starts = np.cumsum([0] + group_sizes)

    return DOFArray(actx, tuple(
        actx.call_loopy(
            prg(),
            grp_start=grp_start, ary=ary,
            nelements=nel,
            ndofs_per_element=ndof,
            )["result"]
        for grp_start, (nel, ndof) in zip(group_starts, group_shapes)))


def unflatten(actx: ArrayContext, discr, ary: Union[Any, np.ndarray],
        ndofs_per_element_per_group: Optional[Iterable[int]] = None) -> DOFArray:
    r"""Convert a 'flat' array returned by :func:`flatten` back to a
    :class:`DOFArray`.

    :arg ndofs_per_element: Optional. If given, an iterable of numbers representing
        the number of degrees of freedom per element, overriding the numbers
        provided by the element groups in *discr*. May be used (for example)
        to handle :class:`DOFArray`\ s that have only one DOF per element,
        representing some per-element quantity.

    Vectorizes over object arrays.
    """
    if isinstance(ary, np.ndarray):
        return obj_array_vectorize(
                lambda subary: unflatten(
                    actx, discr, subary, ndofs_per_element_per_group),
                ary)

    if ndofs_per_element_per_group is None:
        ndofs_per_element_per_group = [
                grp.nunit_dofs for grp in discr.groups]

    nel_ndof_per_element_per_group = [
            (grp.nelements, ndofs_per_element)
            for grp, ndofs_per_element
            in zip(discr.groups, ndofs_per_element_per_group)]

    return _unflatten(actx, nel_ndof_per_element_per_group, ary)

# }}}


# {{{ pickling

_ARRAY_CONTEXT_FOR_PICKLING_TLS = threading.local()


@contextmanager
def array_context_for_pickling(actx: ArrayContext):
    r"""For the current thread, set the array context to be used for pickling
    and unpickling :class:`DOFArray`\ s to *actx*.

    .. versionadded:: 2021.x
    """
    try:
        existing_pickle_actx = _ARRAY_CONTEXT_FOR_PICKLING_TLS.actx
    except AttributeError:
        existing_pickle_actx = None

    if existing_pickle_actx is not None:
        raise RuntimeError("array_context_for_pickling should not be called "
                "inside the context of its own invocation.")

    _ARRAY_CONTEXT_FOR_PICKLING_TLS.actx = actx
    try:
        yield None
    finally:
        _ARRAY_CONTEXT_FOR_PICKLING_TLS.actx = None

# }}}


def flat_norm(ary: DOFArray, ord=None):
    from warnings import warn
    warn("flat_norm is deprecated. Use array_context.np.linalg.norm instead. "
            "flat_norm will disappear in 2022.",
            DeprecationWarning, stacklevel=2)
    return ary.array_context.np.linalg.norm(ary, ord=ord)


# vim: foldmethod=marker
