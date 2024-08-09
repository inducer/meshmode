from __future__ import annotations


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
import threading
from contextlib import contextmanager
from functools import partial, update_wrapper
from numbers import Number
from typing import Any, Callable, Iterable
from warnings import warn

import numpy as np

from arraycontext import (
    Array,
    ArrayContext,
    NotAnArrayContainerError,
    deserialize_container,
    mapped_over_array_containers,
    multimapped_over_array_containers,
    rec_map_array_container,
    rec_multimap_array_container,
    serialize_container,
    with_array_context,
    with_container_arithmetic,
)
from pytools import MovedFunctionDeprecationWrapper, single_valued


__doc__ = """
.. autoclass:: DOFArray

.. autofunction:: rec_map_dof_array_container
.. autofunction:: mapped_over_dof_arrays
.. autofunction:: rec_multimap_dof_array_container
.. autofunction:: multimapped_over_dof_arrays

.. autofunction:: flat_norm

.. autofunction:: array_context_for_pickling

.. autoexception:: InconsistentDOFArray
.. autofunction:: check_dofarray_against_discr
"""


# {{{ DOFArray

@with_container_arithmetic(
        bcast_obj_array=True,
        bcast_numpy_array=True,
        rel_comparison=True,
        bitwise=True,
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
    of the associated group. The entries in this array are further arrays
    managed by :attr:`array_context`, i.e. :class:`DOFArray` is an
    :class:`~arraycontext.ArrayContainer`.

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
    .. automethod:: astype

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

    def __init__(self, actx: ArrayContext | None, data: tuple[Any, ...]) -> None:
        if __debug__:
            if not (actx is None or isinstance(actx, ArrayContext)):
                raise TypeError("actx must be of type ArrayContext")

            if not isinstance(data, tuple):
                raise TypeError("'data' argument must be a tuple")

        self._array_context = actx
        self._data = data

    # Tell numpy that we would like to do our own array math, thank you very much.
    __array_ufunc__ = None

    @property
    def array_context(self) -> ArrayContext:
        return self._array_context

    @property
    def entry_dtype(self) -> np.dtype:
        return single_valued(subary.dtype for subary in self._data)

    @classmethod
    def from_list(cls, actx: ArrayContext | None, res_list) -> DOFArray:
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

    def __str__(self) -> str:
        return f"DOFArray({self._data})"

    def __repr__(self) -> str:
        return f"DOFArray({self._data!r})"

    # {{{ sequence protocol

    def __bool__(self):
        raise ValueError(
                "The truth value of a DOFArray is not well-defined. "
                "Use actx.np.any(x) or actx.np.all(x)")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]

    def __iter__(self):
        return iter(self._data)

    # }}}

    # {{{ ndarray interface

    def _like_me(self, data: Iterable[Array]) -> DOFArray:
        return DOFArray(self.array_context, tuple(data))

    @property
    def shape(self) -> tuple[int]:
        return (len(self),)

    @property
    def size(self) -> int:
        return len(self)

    def copy(self) -> DOFArray:
        return self._like_me([subary.copy() for subary in self])

    def fill(self, value) -> DOFArray:
        for subary in self:
            subary.fill(value)

    def conj(self) -> DOFArray:
        return self._like_me([subary.conj() for subary in self])

    conjugate = conj

    @property
    def real(self) -> DOFArray:
        return self._like_me([subary.real for subary in self])

    @property
    def imag(self) -> DOFArray:
        return self._like_me([subary.imag for subary in self])

    def astype(self, dtype: np.dtype) -> DOFArray:
        dtype = np.dtype(dtype)
        return self._like_me([subary.astype(dtype) for subary in self])

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

    def __iadd__(self, arg): return self._ibop(op.iadd, arg)
    def __isub__(self, arg): return self._ibop(op.isub, arg)
    def __imul__(self, arg): return self._ibop(op.imul, arg)
    def __itruediv__(self, arg): return self._ibop(op.itruediv, arg)
    def __imod__(self, arg): return self._ibop(op.imod, arg)

    def __iand__(self, arg): return self._ibop(op.iand, arg)
    def __ixor__(self, arg): return self._ibop(op.ixor, arg)
    def __ior__(self, arg): return self._ibop(op.ior, arg)

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

        # Make sure metadata inference has been done
        # https://github.com/inducer/meshmode/pull/318#issuecomment-1088320970
        ary = self.array_context.thaw(self.array_context.freeze(self))

        if self.array_context is not actx:
            ary = actx.thaw(actx.freeze(self))

        d = {}
        d["data"] = [actx.to_numpy(ary_i) for ary_i in ary._data]

        d["tags"] = [getattr(ary_i, "tags", frozenset()) for ary_i in ary]

        if len(ary) > 0 and hasattr(ary._data[0], "axes"):
            d["axes_tags"] = [[ax.tags for ax in ary_i.axes] for ary_i in ary._data]
        else:
            d["axes_tags"] = [[frozenset() for _ in range(leaf_ary.ndim)]
                                         for leaf_ary in ary._data]

        return d

    def __setstate__(self, state):
        try:
            actx = _ARRAY_CONTEXT_FOR_PICKLING_TLS.actx
        except AttributeError:
            actx = None

        if actx is None:
            raise RuntimeError("DOFArray instances can only be unpickled while "
                    "array_context_for_pickling is active.")

        self._array_context = actx

        if isinstance(state, dict):
            data = state["data"]
            tags = state["tags"]
            axes_tags = state["axes_tags"]
        else:
            # For backwards compatibility
            from warnings import warn
            warn("A DOFArray is being unpickled without (tag) metadata. "
                 "Program transformation may fail as a result.",
                 stacklevel=2)

            data = state
            tags = [frozenset() for _ in range(len(data))]
            axes_tags = [[frozenset() for _ in range(leaf_ary.ndim)]
                                         for leaf_ary in data]

        assert len(data) == len(tags) == len(axes_tags)

        self._data = []

        for idx, ary in enumerate(data):
            assert len(axes_tags[idx]) == ary.ndim
            assert isinstance(axes_tags[idx], list)

            d = actx.from_numpy(ary)

            try:
                d = d._with_new_tags(tags[idx])
            except AttributeError:
                # 'actx.from_numpy' might return an array that does not have
                # '_with_new_tags' (e.g., np.ndarray).
                pass

            for ida, ax in enumerate(axes_tags[idx]):
                d = actx.tag_axis(ida, ax, d)

            self._data.append(d)

        self._data = tuple(self._data)

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
        template: Any, iterable: Iterable[tuple[Any, Any]]):
    if __debug__:
        def _raise_index_inconsistency(i, stream_i):
            raise ValueError(
                    "out-of-sequence indices supplied in DOFArray deserialization "
                    f"(expected {i}, received {stream_i})")

        return type(template)(
                template.array_context,
                data=tuple(
                    v if i == stream_i else _raise_index_inconsistency(i, stream_i)
                    for i, (stream_i, v) in enumerate(iterable)))
    else:
        return type(template)(
                template.array_context,
                data=tuple([v for _i, v in iterable]))


@with_array_context.register(DOFArray)
def _with_actx_dofarray(ary, actx):
    assert (actx is None) or all(isinstance(subary, actx.array_types)
                                 for subary in ary._data)
    return type(ary)(actx, ary._data)


def rec_map_dof_array_container(f: Callable[[Any], Any], ary):
    r"""Applies *f* recursively to an :class:`~arraycontext.ArrayContainer`.

    Similar to :func:`~arraycontext.map_array_container`, but
    does not further recurse on :class:`DOFArray`\ s.
    """
    def rec(_ary):
        if isinstance(_ary, DOFArray):
            return f(_ary)

        try:
            iterable = serialize_container(_ary)
        except NotAnArrayContainerError:
            return f(_ary)
        else:
            return deserialize_container(_ary, [
                (key, rec(subary)) for key, subary in iterable
                ])

    return rec(ary)


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


# {{{ flat_norm

def _reduce_norm(actx, arys, ord):
    from functools import reduce
    from numbers import Number

    # NOTE: actx can be None if there are no DOFArrays in the container, in
    # which case all the entries should be Numbers and using numpy is ok
    if actx is None:
        anp = np
    else:
        anp = actx.np

    # NOTE: these are ordered by an expected usage frequency
    if ord == 2:
        return anp.sqrt(sum(subary*subary for subary in arys))
    elif ord == np.inf:
        return reduce(anp.maximum, arys)
    elif ord == -np.inf:
        return reduce(anp.minimum, arys)
    elif isinstance(ord, Number) and ord > 0:
        return sum(subary**ord for subary in arys)**(1/ord)
    else:
        raise NotImplementedError(f"unsupported value of 'ord': {ord}")


def flat_norm(ary, ord=None) -> Any:
    r"""Return an element-wise :math:`\ell^{\text{ord}}` norm of *ary*.

    Unlike :attr:`arraycontext.ArrayContext.np`, this function handles
    :class:`DOFArray`\ s by taking a norm of their flattened values
    (in the sense of :func:`arraycontext.flatten`) regardless of how the
    group arrays are stored.

    :arg ary: may be a :class:`DOFArray` or an
        :class:`~arraycontext.ArrayContainer` containing them.
    """

    if ord is None:
        ord = 2

    actx = None

    def _rec(_ary):
        nonlocal actx

        from numbers import Number
        if isinstance(_ary, Number):
            return abs(_ary)

        if isinstance(_ary, DOFArray):
            if _ary.array_context is None:
                raise ValueError("cannot compute the norm of frozen DOFArrays")

            if actx is None:
                actx = _ary.array_context
            else:
                assert actx is _ary.array_context

            return _reduce_norm(actx, [
                actx.np.linalg.norm(actx.np.ravel(subary, order="A"), ord=ord)
                for _, subary in serialize_container(_ary)
                ], ord=ord)

        try:
            iterable = serialize_container(_ary)
        except NotAnArrayContainerError:
            raise TypeError(
                    f"unsupported array type: '{type(_ary).__name__}'") from None
        else:
            arys = [_rec(subary) for _, subary in iterable]
            return _reduce_norm(actx, arys, ord=ord)

    return _rec(ary)

# }}}


# {{{ pickling

_ARRAY_CONTEXT_FOR_PICKLING_TLS = threading.local()


@contextmanager
def array_context_for_pickling(actx: ArrayContext):
    r"""A context manager that, for the current thread, sets the array
    context to be used for pickling and unpickling :class:`DOFArray`\ s
    to *actx*.

    .. versionadded:: 2021.1
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


# {{{ checking

class InconsistentDOFArray(ValueError):
    pass


def check_dofarray_against_discr(discr, dof_ary: DOFArray):
    """Verify that the :class:`DOFArray` *dof_ary* is consistent with
    the discretization *discr*, in terms of things like group count,
    number of elements, and number of DOFs per element. If a discrepancy is
    detected, :exc:`InconsistentDOFArray` is raised.

    :arg discr: a :class:`~meshmode.discretization.Discretization`
        against which *dof_ary* is to be checked.
    """
    if not isinstance(dof_ary, DOFArray):
        raise TypeError("non-array passed to check_dofarray_against_discr")

    if len(dof_ary) != len(discr.groups):
        raise InconsistentDOFArray(
                "DOFArray has unexpected number of groups "
                f"({len(dof_ary)}, expected: {len(discr.groups)})")

    for i, (grp, grp_ary) in enumerate(zip(discr.groups, dof_ary)):
        expected_shape = (grp.nelements, grp.nunit_dofs)
        if grp_ary.shape != expected_shape:
            raise InconsistentDOFArray(
                    f"DOFArray group {i} array has unexpected shape. "
                    f"(observed: {grp_ary.shape}, expected: {expected_shape})")

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
    warn("meshmode.dof_array.thaw is deprecated. Use arraycontext.thaw instead. "
            "WARNING: The argument order is reversed between these two functions. "
            "meshmode.dof_array.thaw will continue to work until 2022.",
            DeprecationWarning, stacklevel=2)

    return actx.thaw(ary)


def freeze(ary, actx):
    warn("meshmode.dof_array.freeze is deprecated. Use arraycontext.freeze instead. "
            "meshmode.dof_array.freeze will continue to work until 2022.",
            DeprecationWarning, stacklevel=2)

    return actx.freeze(ary)

# }}}


# vim: foldmethod=marker
