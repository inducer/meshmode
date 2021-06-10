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

import threading
import operator as op
from numbers import Number
from contextlib import contextmanager
from functools import partial, update_wrapper
from typing import Any, Callable, Iterable, Optional, Tuple

import numpy as np
import loopy as lp

from pytools import MovedFunctionDeprecationWrapper
from pytools import single_valued, memoize_in

from arraycontext import (
        ArrayContext, make_loopy_program,
        ArrayContainer, with_container_arithmetic,
        serialize_container, deserialize_container,
        thaw as _thaw, freeze as _freeze,
        rec_map_array_container, rec_multimap_array_container,
        mapped_over_array_containers, multimapped_over_array_containers)

__doc__ = """
.. autoclass:: DOFArray

.. autofunction:: rec_map_dof_array_container
.. autofunction:: mapped_over_dof_arrays
.. autofunction:: rec_multimap_dof_array_container
.. autofunction:: multimapped_over_dof_arrays

.. autofunction:: flatten
.. autofunction:: unflatten
.. autofunction:: unflatten_like
.. autofunction:: flatten_to_numpy
.. autofunction:: unflatten_from_numpy
.. autofunction:: flat_norm

.. autofunction:: array_context_for_pickling
"""


# {{{ DOFArray

@with_container_arithmetic(
        bcast_obj_array=True,
        bcast_numpy_array=True,
        rel_comparison=True,
        _cls_has_array_context_attr=True)
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

        An :class:`~arraycontext.ArrayContext`.

    .. attribute:: entry_dtype

        The (assumed uniform) :class:`numpy.dtype` of the group arrays
        contained in this instance.

    .. automethod:: __len__

    The following methods and attributes are implemented to mimic the
    functionality of :class:`~numpy.ndarray`\ s. They require the
    :class:`DOFArray` to be :func:`~arraycontext.thaw`\ ed.

    .. attribute:: shape
    .. attribute:: size
    .. automethod:: copy
    .. automethod:: fill
    .. automethod:: conj
    .. attribute:: real
    .. attribute:: imag

    Implements the usual set of arithmetic operations, including broadcasting
    of numbers and over numpy object arrays.

    .. note::

        :class:`DOFArray` instances can be pickled and unpickled while the context
        manager :class:`array_context_for_pickling` is active. If, for an array
        to be pickled, the :class:`~arraycontext.ArrayContext` given to
        :func:`array_context_for_pickling` does not agree with :attr:`array_context`,
        the array is frozen and rethawed. If :attr:`array_context` is *None*,
        the :class:`DOFArray` is :func:`~arraycontext.thaw`\ ed into
        the array context given to :func:`array_context_for_pickling`.
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
            :meth:`~arraycontext.ArrayContext.thaw`\ ed.
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

    # {{{ in-place arithmetic

    def _ibop(self, f, arg):
        """Generic in-place binary operator without any broadcast support."""
        from warnings import warn
        warn("In-place operations on DOFArrays are deprecated. "
                "They will be removed in 2022.", DeprecationWarning, stacklevel=3)

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

    def __iadd__(self, arg): return self._ibop(op.iadd, arg)            # noqa: E704
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
            ary = _thaw(actx, _freeze(self))

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

    @classmethod
    def _serialize_init_arrays_code(cls, instance_name):
        return {"_":
                (f"{instance_name}_i", f"{instance_name}")}

    @classmethod
    def _deserialize_init_arrays_code(cls, template_instance_name, args):
        (_, arg), = args.items()
        # Why tuple([...])? https://stackoverflow.com/a/48592299
        return (f"{template_instance_name}.array_context, tuple([{arg}])")

# }}}


# {{{ ArrayContainer implementation

@serialize_container.register(DOFArray)
def _serialize_dof_container(ary: DOFArray):
    return enumerate(ary._data)


@deserialize_container.register(DOFArray)
def _deserialize_dof_container(
        template: Any, iterable: Iterable[Tuple[Any, Any]]):
    def _raise_index_inconsistency(i, stream_i):
        raise ValueError(
                "out-of-sequence indices supplied in DOFArray deserialization "
                f"(expected {i}, received {stream_i})")

    return type(template)(
            template.array_context,
            data=tuple(
                v if i == stream_i else _raise_index_inconsistency(i, stream_i)
                for i, (stream_i, v) in enumerate(iterable)))


@_freeze.register(DOFArray)
def _freeze_dofarray(ary, actx=None):
    if actx is not None:
        if actx is not ary.array_context:
            raise ValueError("supplied array context does not agree with the one "
                    "in the DOFArray in freeze(DOFArray)")
    return type(ary)(
        None,
        tuple(ary.array_context.freeze(subary) for subary in ary._data))


@_thaw.register(DOFArray)
def _thaw_dofarray(ary, actx):
    if ary.array_context is not None:
        raise ValueError("cannot thaw DOFArray that already has an array context")

    return type(ary)(
        actx,
        tuple(actx.thaw(subary) for subary in ary._data))


def rec_map_dof_array_container(f: Callable[[Any], Any], ary):
    r"""Applies *f* recursively to an :class:`~arraycontext.ArrayContainer`.

    Similar to :func:`~arraycontext.map_array_container`, but
    does not further recurse on :class:`DOFArray`\ s.
    """
    from arraycontext.container.traversal import _map_array_container_impl
    return _map_array_container_impl(f, ary, leaf_cls=DOFArray, recursive=True)


def mapped_over_dof_arrays(f):
    wrapper = partial(rec_map_dof_array_container, f)
    update_wrapper(wrapper, f)
    return wrapper


def rec_multimap_dof_array_container(f: Callable[[Any], Any], *args):
    r"""Applies *f* recursively to multiple :class:`~arraycontext.ArrayContainer`\ s.

    Similar to :func:`~arraycontext.multimap_array_container`, but
    does not further recurse on :class:`DOFArray`\ s.
    """
    from arraycontext.container.traversal import _multimap_array_container_impl
    return _multimap_array_container_impl(
            f, *args, leaf_cls=DOFArray, recursive=True)


def multimapped_over_dof_arrays(f):
    def wrapper(*args):
        return rec_multimap_dof_array_container(f, *args)

    update_wrapper(wrapper, f)
    return wrapper

# }}}


# {{{ flatten / unflatten

def _flatten_dof_array(ary: Any, strict: bool = True):
    if not isinstance(ary, DOFArray):
        if strict:
            raise TypeError(f"non-DOFArray type '{type(ary).__name__}' cannot "
                    "be flattened; use 'strict=False' to allow other types")
        else:
            return ary

    actx = ary.array_context
    if actx is None:
        raise ValueError("cannot flatten frozen DOFArrays")

    @memoize_in(actx, (_flatten_dof_array, "flatten_grp_ary_prg"))
    def prg():
        return make_loopy_program(
            [
                "{[iel]: 0 <= iel < nelements}",
                "{[idof]: 0 <= idof < ndofs_per_element}"
            ],
            """
                result[iel * ndofs_per_element + idof] = grp_ary[iel, idof]
            """,
            [
                lp.GlobalArg("result", None,
                             shape="nelements * ndofs_per_element"),
                lp.GlobalArg("grp_ary", None,
                             shape=("nelements", "ndofs_per_element")),
                lp.ValueArg("nelements", np.int32),
                lp.ValueArg("ndofs_per_element", np.int32),
                "..."
            ],
            name="flatten_grp_ary"
        )

    def _flatten(grp_ary):
        # If array has two axes, assume they are elements/dofs. If C-contiguous
        # in those, "flat" and "unflat" memory layout agree.
        if len(grp_ary.shape) == 2 and grp_ary.flags.c_contiguous:
            return grp_ary.reshape(-1, order="C")
        else:
            # NOTE: array has unsupported strides
            return actx.call_loopy(
                prg(),
                grp_ary=grp_ary
            )["result"]

    if len(ary) == 1:
        # can avoid a copy if reshape succeeds
        return _flatten(ary[0])
    else:
        return actx.np.concatenate([_flatten(grp_ary) for grp_ary in ary])


def flatten(ary: ArrayContainer, *, strict: bool = True) -> ArrayContainer:
    r"""Convert all :class:`DOFArray`\ s into a "flat" array of degrees of
    freedom, where the resulting type of the array is given by the
    :attr:`DOFArray.array_context`.

    Array elements are laid out contiguously, with the element group
    index varying slowest, element index next, and intra-element DOF
    index fastest.

    Recurses into the :class:`~arraycontext.ArrayContainer` for all
    :class:`DOFArray`\ s.

    :param strict: if *True*, only :class:`DOFArray`\ s are allowed as leaves
        in the container *ary*. If *False*, any non-:class:`DOFArray` are
        left as is.
    """

    def _flatten(subary):
        return _flatten_dof_array(subary, strict=strict)

    return rec_map_dof_array_container(_flatten, ary)


def _unflatten_dof_array(actx: ArrayContext, ary: Any,
        group_shapes: Iterable[int], group_starts: np.ndarray,
        strict: bool = True) -> DOFArray:
    if ary.size != group_starts[-1]:
        if strict:
            raise ValueError("cannot unflatten array: "
                    f"has size {ary.size}, expected {group_starts[-1]}; "
                    "use 'strict=False' to leave the array unchanged")
        else:
            return ary

    @memoize_in(actx, (_unflatten_dof_array, "unflatten_prg"))
    def prg():
        return make_loopy_program(
            "{[iel,idof]: 0<=iel<nelements and 0<=idof<ndofs_per_element}",
            "result[iel, idof] = ary[grp_start + iel*ndofs_per_element + idof]",
            name="unflatten")

    return DOFArray(actx, tuple(
        actx.call_loopy(
            prg(),
            grp_start=grp_start, ary=ary,
            nelements=nel,
            ndofs_per_element=ndof,
            )["result"]
        for grp_start, (nel, ndof) in zip(group_starts, group_shapes)))


def _unflatten_group_sizes(discr, ndofs_per_element_per_group):
    if ndofs_per_element_per_group is None:
        ndofs_per_element_per_group = [
                grp.nunit_dofs for grp in discr.groups]

    group_shapes = [
            (grp.nelements, ndofs_per_element)
            for grp, ndofs_per_element
            in zip(discr.groups, ndofs_per_element_per_group)]

    group_sizes = [nel * ndof for nel, ndof in group_shapes]
    group_starts = np.cumsum([0] + group_sizes)

    return group_shapes, group_starts


def unflatten(
        actx: ArrayContext, discr, ary: ArrayContainer,
        ndofs_per_element_per_group: Optional[Iterable[int]] = None, *,
        strict: bool = True,
        ) -> ArrayContainer:
    r"""Convert all "flat" arrays returned by :func:`flatten` back to
    :class:`DOFArray`\ s.

    This function recurses into the :class:`~arraycontext.ArrayContainer` for all
    :class:`DOFArray`\ s. All class:`DOFArray`\ s inside the container
    *ary* must agree on the mapping from element group number to number
    of degrees of freedom, as given by `ndofs_per_element_per_group`
    (or via *discr*).

    :arg ndofs_per_element: if given, an iterable of numbers representing
        the number of degrees of freedom per element, overriding the numbers
        provided by the element groups in *discr*. May be used (for example)
        to handle :class:`DOFArray`\ s that have only one DOF per element,
        representing some per-element quantity.
    :param strict: if *True*, only :class:`DOFArray`\ s are allowed as leaves
        in the container *ary*. If *False*, any non-:class:`DOFArray` are
        left as is.
    """
    group_shapes, group_starts = _unflatten_group_sizes(
            discr, ndofs_per_element_per_group)

    def _unflatten(subary):
        return _unflatten_dof_array(
                actx, subary, group_shapes, group_starts,
                strict=strict)

    return rec_map_dof_array_container(_unflatten, ary)


def unflatten_like(
        actx: ArrayContext, ary: ArrayContainer, prototype: ArrayContainer, *,
        strict: bool = True,
        ) -> ArrayContainer:
    r"""Convert all "flat" arrays returned by :func:`flatten` back to
    :class:`DOFArray`\ s based on a *prototype* container.

    This function allows doing a roundtrip with :func:`flatten` for containers
    which have :class:`DOFArray`\ s with different numbers of degrees of
    freedom. This is unlike :func:`unflatten`, where all the :class:`DOFArray`\ s
    must agree on the number of degrees of freedom per element group.
    For example, this enables "unflattening" of arrays associated with different
    :class:`~meshmode.discretization.Discretization`\ s within the same
    container.

    :param prototype: an array container with the same structure as *ary*,
        whose :class:`DOFArray` leaves are used to get the sizes to
        unflatten *ary*.
    :param strict: if *True*, only :class:`DOFArray`\ s are allowed as leaves
        in the container *ary*. If *False*, any non-:class:`DOFArray` are
        left as is.
    """
    from arraycontext import is_array_container

    def _same_key(key1, key2):
        assert key1 == key2
        return key1

    def _unflatten_like(_ary, _prototype):
        if isinstance(_prototype, DOFArray):
            group_shapes = [subary.shape for subary in _prototype]
            group_sizes = [subary.size for subary in _prototype]
            group_starts = np.cumsum([0] + group_sizes)

            return _unflatten_dof_array(
                    actx, _ary, group_shapes, group_starts,
                    strict=True)
        elif is_array_container(_prototype):
            assert type(_ary) is type(_prototype)

            return deserialize_container(_prototype, [
                (_same_key(key1, key2), _unflatten_like(subary, subprototype))
                for (key1, subary), (key2, subprototype) in zip(
                    serialize_container(_ary),
                    serialize_container(_prototype))
                ])
        else:
            if strict:
                raise ValueError("cannot unflatten array "
                        f"with prototype '{type(_prototype).__name__}'; "
                        "use 'strict=False' to leave the array unchanged")

            assert type(_ary) is type(_prototype)
            return _ary

    return _unflatten_like(ary, prototype)


def flatten_to_numpy(actx: ArrayContext, ary: ArrayContainer, *,
        strict: bool = True) -> ArrayContainer:
    r"""Converts all :class:`DOFArray`\ s into "flat" :class:`numpy.ndarray`\ s
    using :func:`flatten`.
    """
    def _flatten_to_numpy(subary):
        if isinstance(subary, DOFArray) and subary.array_context is None:
            subary = _thaw(subary, actx)

        return actx.to_numpy(_flatten_dof_array(subary, strict=strict))

    return rec_map_dof_array_container(_flatten_to_numpy, ary)


def unflatten_from_numpy(
        actx: ArrayContext, discr, ary: ArrayContainer,
        ndofs_per_element_per_group: Optional[Iterable[int]] = None, *,
        strict: bool = True,
        ) -> ArrayContainer:
    r"""Takes "flat" arrays returned by :func:`flatten_to_numpy` and
    reconstructs the corresponding :class:`DOFArray`\ s using :func:`unflatten`.
    """
    group_shapes, group_starts = _unflatten_group_sizes(
            discr, ndofs_per_element_per_group)

    def _unflatten_from_numpy(subary):
        if isinstance(subary, np.ndarray) and subary.dtype.char != "O":
            subary = actx.from_numpy(subary)

        # FIXME: this is doing the recursion itself instead of just using
        # `rec_map_dof_array_container` like `flatten_to_numpy` to catch
        # non-object ndarrays, which `is_array_container` considers as as
        # containers and tries to serialize.
        from arraycontext import map_array_container, is_array_container
        if is_array_container(subary):
            return map_array_container(_unflatten_from_numpy, subary)
        else:
            return _unflatten_dof_array(
                    actx, subary, group_shapes, group_starts,
                    strict=strict)

    return _unflatten_from_numpy(ary)

# }}}


# {{{ flat_norm

def _flatten_array(ary):
    import pyopencl.array as cl
    assert isinstance(ary, cl.Array)

    if ary.size == 0:
        # Work around https://github.com/inducer/pyopencl/pull/402
        return ary._new_with_changes(
                data=None, offset=0, shape=(0,), strides=(ary.dtype.itemsize,))
    if ary.flags.f_contiguous:
        return ary.reshape(-1, order="F")
    elif ary.flags.c_contiguous:
        return ary.reshape(-1, order="C")
    else:
        raise ValueError("cannot flatten group array of DOFArray for norm, "
                f"with strides {ary.strides} of {ary.dtype}")


def flat_norm(ary, ord=None) -> float:
    r"""Return an element-wise :math:`\ell^{\text{ord}}` norm of *ary*.

    :arg ary: may be a :class:`DOFArray` or a
        :class:`~arraycontext.ArrayContainer` containing them.
    """

    from numbers import Number
    if isinstance(ary, Number):
        return abs(ary)

    if ord is None:
        ord = 2

    from arraycontext import is_array_container

    import numpy.linalg as la
    if isinstance(ary, DOFArray):
        actx = ary.array_context
        return la.norm(
                [
                    actx.np.linalg.norm(_flatten_array(subary), ord=ord)
                    for _, subary in serialize_container(ary)],
                ord=ord)

    elif is_array_container(ary):
        return la.norm(
                [flat_norm(subary, ord=ord)
                    for _, subary in serialize_container(ary)],
                ord=ord)

    raise TypeError(
            f"unsupported array type passed to flat_norm: '{type(ary).__name__}'")

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


# {{{ deprecated

obj_or_dof_array_vectorize = MovedFunctionDeprecationWrapper(
        rec_map_array_container, deadline="2022")
obj_or_dof_array_vectorized = MovedFunctionDeprecationWrapper(
        mapped_over_array_containers, deadline="2022")
obj_or_dof_array_vectorize_n_args = MovedFunctionDeprecationWrapper(
        rec_multimap_array_container, deadline="2022")
obj_or_dof_array_vectorized_n_args = MovedFunctionDeprecationWrapper(
        multimapped_over_array_containers, deadline="2022")


def thaw(actx, ary):
    from warnings import warn
    warn("meshmode.dof_array.thaw is deprecated. Use arraycontext.thaw instead. "
            "WARNING: The argument order is reversed between these two functions. "
            "meshmode.dof_array.thaw will continue to work until 2022.",
            DeprecationWarning, stacklevel=2)

    # /!\ arg order flipped
    return _thaw(ary, actx)


freeze = MovedFunctionDeprecationWrapper(_freeze, deadline="2022")

# }}}

# vim: foldmethod=marker
