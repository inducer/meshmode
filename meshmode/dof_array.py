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
from typing import Optional, Iterable, Any
from functools import partial

from pytools import single_valued, memoize_in
from pytools.obj_array import obj_array_vectorize, obj_array_vectorize_n_args

from meshmode.array_context import ArrayContext, make_loopy_program


__doc__ = """
.. autoclass:: DOFArray
.. autofunction:: thaw
.. autofunction:: freeze
.. autofunction:: flatten
.. autofunction:: unflatten
.. autofunction:: flat_norm
"""


# {{{ DOFArray

class DOFArray(np.ndarray):
    """This array type is a subclass of :class:`numpy.ndarray` intended to hold
    degree-of-freedom arrays for use with
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
    of the associated group.
    This array is derived from :class:`numpy.ndarray` with dtype object ("an
    object array").  The entries in this array are further arrays managed by
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

    .. automethod:: from_list

    .. note::

        :class:`DOFArray` instances support elementwise ``<``, ``>``,
        ``<=``, ``>=``. (:mod:`numpy` object arrays containing arrays do not.)
    """

    # Follows https://numpy.org/devdocs/user/basics.subclassing.html

    def __new__(cls, actx: Optional[ArrayContext], input_array):
        if not (actx is None or isinstance(actx, ArrayContext)):
            raise TypeError("actx must be of type ArrayContext")

        result = np.asarray(input_array).view(cls)
        if len(result.shape) != 1:
            raise ValueError("DOFArray instances must have one-dimensional "
                    "shape, with one entry per element group")

        result.array_context = actx
        return result

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.array_context = getattr(obj, "array_context", None)

    @property
    def entry_dtype(self):
        return single_valued(subary.dtype for subary in self.flat)

    @classmethod
    def from_list(cls, actx: Optional[ArrayContext], res_list) -> "DOFArray":
        r"""Create a :class:`DOFArray` from a list of arrays
        (one per :class:`~meshmode.discretization.ElementGroupBase`).

        :arg actx: If *None*, the arrays in *res_list* must be
            :meth:`~meshmode.array_context.ArrayContext.thaw`\ ed.
        """
        if not (actx is None or isinstance(actx, ArrayContext)):
            raise TypeError("actx must be of type ArrayContext")

        ngroups = len(res_list)

        result = np.empty(ngroups, dtype=object).view(cls)
        result.array_context = actx

        # 'result[:] = res_list' may look tempting, however:
        # https://github.com/numpy/numpy/issues/16564
        for igrp in range(ngroups):
            result[igrp] = res_list[igrp]

        return result

    # {{{ work around numpy failing to compare obj arrays of arrays

    def _comparison(self, operator_func, other):
        from numbers import Number
        if isinstance(other, DOFArray):
            return obj_array_vectorize_n_args(operator_func, self, other)

        elif isinstance(other, Number):
            return obj_array_vectorize(
                    lambda self_entry: operator_func(self_entry, other),
                    self)

        else:
            # fall back to "best effort" (i.e. likley failure)
            return operator_func(self, other)

    def __lt__(self, other):
        return self._comparison(operator.lt, other)

    def __gt__(self, other):
        return self._comparison(operator.gt, other)

    def __le__(self, other):
        return self._comparison(operator.le, other)

    def __ge__(self, other):
        return self._comparison(operator.ge, other)

    # }}}

# }}}


def thaw(actx: ArrayContext, ary: np.ndarray) -> np.ndarray:
    r"""Call :meth:`~meshmode.array_context.ArrayContext.thaw` on the element
    group arrays making up the :class:`DOFArray`, using *actx*.

    Vectorizes over object arrays of :class:`DOFArray`\ s.
    """
    if (isinstance(ary, np.ndarray)
            and ary.dtype.char == "O"
            and not isinstance(ary, DOFArray)):
        return obj_array_vectorize(partial(thaw, actx), ary)

    if ary.array_context is not None:
        raise ValueError("DOFArray passed to thaw is not frozen")

    return DOFArray.from_list(actx, [
        actx.thaw(subary)
        for subary in ary
        ])


def freeze(ary: np.ndarray) -> np.ndarray:
    r"""Call :meth:`~meshmode.array_context.ArrayContext.freeze` on the element
    group arrays making up the :class:`DOFArray`, using the
    :class:`~meshmode.array_context.ArrayContext` in *ary*.

    Vectorizes over object arrays of :class:`DOFArray`\ s.
    """
    if (isinstance(ary, np.ndarray)
            and ary.dtype.char == "O"
            and not isinstance(ary, DOFArray)):
        return obj_array_vectorize(freeze, ary)

    if ary.array_context is None:
        raise ValueError("DOFArray passed to freeze is already frozen")

    return DOFArray.from_list(None, [
        ary.array_context.freeze(subary)
        for subary in ary
        ])


def flatten(ary: np.ndarray) -> Any:
    r"""Convert a :class:`DOFArray` into a "flat" array of degrees of freedom,
    where the resulting type of the array is given by the
    :attr:`DOFArray.array_context`.

    Array elements are laid out contiguously, with the element group
    index varying slowest, element index next, and intra-element DOF
    index fastest.

    Vectorizes over object arrays of :class:`DOFArray`\ s.
    """
    if (isinstance(ary, np.ndarray)
            and ary.dtype.char == "O"
            and not isinstance(ary, DOFArray)):
        return obj_array_vectorize(flatten, ary)

    actx = ary.array_context

    return actx.np.concatenate(actx.np.reshape(grp_ary, (-1,))
                               for grp_ary in ary)


def unflatten(actx: ArrayContext, discr, ary,
        ndofs_per_element_per_group: Optional[Iterable[int]] = None) -> np.ndarray:
    r"""Convert a 'flat' array returned by :func:`flatten` back to a :class:`DOFArray`.

    Vectorizes over object arrays of :class:`DOFArray`\ s.
    """
    if (isinstance(ary, np.ndarray)
            and ary.dtype.char == "O"
            and not isinstance(ary, DOFArray)):
        return obj_array_vectorize(
                lambda subary: unflatten(
                    actx, discr, subary, ndofs_per_element_per_group),
                ary)

    @memoize_in(actx, (unflatten, "unflatten_prg"))
    def prg():
        return make_loopy_program(
            "{[iel,idof]: 0<=iel<nelements and 0<=idof<ndofs_per_element}",
            "result[iel, idof] = ary[grp_start + iel*ndofs_per_element + idof]",
            name="unflatten")

    if ndofs_per_element_per_group is None:
        ndofs_per_element_per_group = [
                grp.nunit_dofs for grp in discr.groups]

    group_sizes = [
            grp.nelements * ndofs_per_element
            for grp, ndofs_per_element
            in zip(discr.groups, ndofs_per_element_per_group)]

    if ary.size != sum(group_sizes):
        raise ValueError("array has size %d, expected %d"
                % (ary.size, sum(group_sizes)))

    group_starts = np.cumsum([0] + group_sizes)

    return DOFArray.from_list(actx, [
        actx.call_loopy(
            prg(),
            grp_start=grp_start, ary=ary,
            nelements=grp.nelements,
            ndofs_per_element=ndofs_per_element,
            )["result"]
        for grp_start, grp, ndofs_per_element in zip(
            group_starts,
            discr.groups,
            ndofs_per_element_per_group)])


def flat_norm(ary: DOFArray, ord=2):
    # FIXME This could be done without flattening and copying
    actx = ary.array_context
    import numpy.linalg as la
    return la.norm(actx.to_numpy(flatten(ary)), ord)


# vim: foldmethod=marker
