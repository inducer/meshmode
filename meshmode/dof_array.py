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

import operator
import numpy as np
from typing import Optional, Iterable, Any, Tuple, Union
from functools import partial
from numbers import Number
import operator as op
import decorator
import threading
from contextlib import contextmanager


from pytools import single_valued, memoize_in
from pytools.obj_array import obj_array_vectorize

from meshmode.array_context import ArrayContext, make_loopy_program


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

class DOFArray:
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
    .. automethod:: __getitem__

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

    This object supports arithmetic, comparisons, and logic operators.

    .. note::

        :class:`DOFArray` instances support elementwise ``<``, ``>``,
        ``<=``, ``>=``. (:mod:`numpy` object arrays containing arrays do not.)

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

        self.array_context = actx
        self._data = data

    # Tell numpy that we would like to do our own array math, thank you very much.
    # (numpy arrays have priority 0.)
    __array_priority__ = 10

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

    # {{{ arithmetic

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

    def _bop(self, f, op1, op2):
        """Broadcasting logic for a generic binary operator."""
        if isinstance(op1, np.ndarray):
            return obj_array_vectorize(lambda op: self._bop(f, op, op2),
                op1.astype(np.object, copy=False))
        if isinstance(op2, np.ndarray):
            return obj_array_vectorize(lambda op: self._bop(f, op1, op),
                op2.astype(np.object, copy=False))
        if isinstance(op1, DOFArray) and isinstance(op2, DOFArray):
            if len(op1) != len(op2):
                raise ValueError("DOFArray objects in binary operator must have "
                        f"same length, got {len(op1)} and {len(op2)}")
            return self._like_me([
                f(op1_i, op2_i)
                for op1_i, op2_i in zip(op1._data, op2._data)])
        elif isinstance(op1, DOFArray) and isinstance(op2, Number):
            return self._like_me([f(op1_i, op2) for op1_i in op1._data])
        elif isinstance(op1, Number) and isinstance(op2, DOFArray):
            return self._like_me([f(op1, op2_i) for op2_i in op2._data])
        else:
            raise NotImplementedError("operation for types "
                f"{type(op1).__name__} and {type(op2).__name__}")

    def __add__(self, arg): return self._bop(op.add, self, arg)  # noqa: E704
    def __radd__(self, arg): return self._bop(op.add, arg, self)  # noqa: E704
    def __sub__(self, arg): return self._bop(op.sub, self, arg)  # noqa: E704
    def __rsub__(self, arg): return self._bop(op.sub, arg, self)  # noqa: E704
    def __mul__(self, arg): return self._bop(op.mul, self, arg)  # noqa: E704
    def __rmul__(self, arg): return self._bop(op.mul, arg, self)  # noqa: E704
    def __truediv__(self, arg): return self._bop(op.truediv, self, arg)  # noqa: E704
    def __rtruediv__(self, arg): return self._bop(op.truediv, arg, self)  # noqa: E704, E501
    def __pow__(self, arg): return self._bop(op.pow, self, arg)  # noqa: E704
    def __rpow__(self, arg): return self._bop(op.pow, arg, self)  # noqa: E704
    def __mod__(self, arg): return self._bop(op.mod, self, arg)  # noqa: E704
    def __rmod__(self, arg): return self._bop(op.mod, arg, self)  # noqa: E704
    def __divmod__(self, arg): return self._bop(divmod, self, arg)  # noqa: E704
    def __rdivmod__(self, arg): return self._bop(divmod, arg, self)  # noqa: E704

    def __pos__(self): return self  # noqa: E704
    def __neg__(self): return self._like_me([-self_i for self_i in self._data])  # noqa: E704, E501
    def __abs__(self): return self._like_me([abs(self_i) for self_i in self._data])  # noqa: E704, E501

    def __iadd__(self, arg):
        """
        :param arg: can be a :class:`~numbers.Number` or another :class:`DOFArray`.
        """
        return self._ibop(op.iadd, arg)

    def __isub__(self, arg): return self._ibop(op.isub, arg)            # noqa: E704
    def __imul__(self, arg): return self._ibop(op.imul, arg)            # noqa: E704
    def __itruediv__(self, arg): return self._ibop(op.itruediv, arg)    # noqa: E704
    def __imod__(self, arg): return self._ibop(op.imod, arg)            # noqa: E704

    # }}}

    # {{{ comparison

    def __eq__(self, arg): return self._bop(op.eq, self, arg)  # noqa: E704
    def __ne__(self, arg): return self._bop(op.ne, self, arg)  # noqa: E704
    def __lt__(self, arg): return self._bop(op.lt, self, arg)  # noqa: E704
    def __gt__(self, arg): return self._bop(op.gt, self, arg)  # noqa: E704
    def __le__(self, arg): return self._bop(op.le, self, arg)  # noqa: E704
    def __ge__(self, arg): return self._bop(op.ge, self, arg)  # noqa: E704

    # }}}

    # {{{ logical

    def __and__(self, arg): return self._bop(operator.and_, self, arg)  # noqa: E704
    def __xor__(self, arg): return self._bop(operator.xor, self, arg)  # noqa: E704
    def __or__(self, arg): return self._bop(operator.or_, self, arg)  # noqa: E704
    def __rand__(self, arg): return self._bop(operator.and_, arg, self)  # noqa: E704
    def __rxor__(self, arg): return self._bop(operator.xor, arg, self)  # noqa: E704
    def __ror__(self, arg): return self._bop(operator.or_, arg, self)  # noqa: E704

    def __iand__(self, arg): return self._ibop(op.iand, arg)        # noqa: E704
    def __ixor__(self, arg): return self._ibop(op.ixor, arg)        # noqa: E704
    def __ior__(self, arg): return self._ibop(op.ior, arg)          # noqa: E704

    # }}}

    # bit shifts unimplemented for now

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


# {{{ obj_or_dof vectorization

def obj_or_dof_array_vectorize(f, ary):
    r"""
    Works like :func:`~pytools.obj_array.obj_array_vectorize`, but recurses
    on object arrays and also tolerates one final 'layer' of :class:`DOFArray`\ s.
    """

    if isinstance(ary, DOFArray):
        return ary._like_me([f(ary_i) for ary_i in ary._data])
    elif isinstance(ary, np.ndarray) and ary.dtype.char == "O":
        return obj_array_vectorize(partial(obj_or_dof_array_vectorize, f), ary)
    else:
        return f(ary)


obj_or_dof_array_vectorized = decorator.decorator(obj_or_dof_array_vectorize)


def obj_or_dof_array_vectorize_n_args(f, *args):
    r"""Apply the function *f* elementwise to all entries of any
    object arrays or :class:`DOFArray`\ s in *args*. All such arrays are expected
    to have the same shape (but this is not checked).
    Equivalent to an appropriately-looped execution of::

        result[idx] = f(obj_array_arg1[idx], arg2, obj_array_arg3[idx])

    Return an array of the same shape as the arguments consisting of the
    return values of *f*.  If the elements of arrays found in *args* are
    further object arrays, recurse.  If a :class:`DOFArray` is found,  apply
    *f* to its entries. If non-object-arrays are found, apply *f* to those.
    """
    dofarray_arg_indices = [
            i for i, arg in enumerate(args)
            if isinstance(arg, DOFArray)]

    if not dofarray_arg_indices:
        if any(isinstance(arg, np.ndarray) and arg.dtype.char == "O"
                for i, arg in enumerate(args)):
            from pytools.obj_array import obj_array_vectorize_n_args
            return obj_array_vectorize_n_args(
                    partial(obj_or_dof_array_vectorize_n_args, f), *args)
        else:
            return f(*args)

    leading_da_index = dofarray_arg_indices[0]

    template_ary = args[leading_da_index]
    result = []
    new_args = list(args)
    for igrp in range(len(template_ary)):
        for arg_i in dofarray_arg_indices:
            new_args[arg_i] = args[arg_i][igrp]
        result.append(f(*new_args))

    return DOFArray(template_ary.array_context, tuple(result))


obj_or_dof_array_vectorized_n_args = decorator.decorator(
        obj_or_dof_array_vectorize_n_args)

# }}}


# {{{ thaw / freeze

def thaw(actx: ArrayContext, ary: Union[DOFArray, np.ndarray]) -> np.ndarray:
    r"""Call :meth:`~meshmode.array_context.ArrayContext.thaw` on the element
    group arrays making up the :class:`DOFArray`, using *actx*.

    Vectorizes over object arrays of :class:`DOFArray`\ s.
    """
    if isinstance(ary, np.ndarray):
        return obj_array_vectorize(partial(thaw, actx), ary)

    if ary.array_context is not None:
        raise ValueError("DOFArray passed to thaw is not frozen")

    return DOFArray(actx, tuple(actx.thaw(subary) for subary in ary))


def freeze(ary: Union[DOFArray, np.ndarray]) -> np.ndarray:
    r"""Call :meth:`~meshmode.array_context.ArrayContext.freeze` on the element
    group arrays making up the :class:`DOFArray`, using the
    :class:`~meshmode.array_context.ArrayContext` in *ary*.

    Vectorizes over object arrays of :class:`DOFArray`\ s.
    """
    if isinstance(ary, np.ndarray):
        return obj_array_vectorize(freeze, ary)

    if ary.array_context is None:
        raise ValueError("DOFArray passed to freeze is already frozen")

    return DOFArray(None, tuple(
        ary.array_context.freeze(subary) for subary in ary))

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
