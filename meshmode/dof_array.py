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

import numpy as np
from typing import Optional, TYPE_CHECKING
from functools import partial

from pytools import single_valued, memoize_in
from pytools.obj_array import obj_array_vectorize

from meshmode.array_context import ArrayContext, make_loopy_program

if TYPE_CHECKING:
    from meshmode.discretization import Discretization as _Discretization


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
    with one entry in the :class:`DOFArray` per element group.
    It is derived from :class:`numpy.ndarray` with dtype object ("an object array")

    The main purpose of this class is to better describe the data structure,
    i.e. when a :class:`DOFArray` occurs inside of further numpy object array,
    the level representing the array of element groups can be recognized (by
    people and programs).

    .. attribute:: array_context
    .. attribute:: entry_dtype
    .. automethod:: from_list
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
        self.array_context = getattr(obj, 'array_context', None)

    @property
    def entry_dtype(self):
        return single_valued(subary.dtype for subary in self.flat)

    @classmethod
    def from_list(cls, actx: Optional[ArrayContext], res_list):
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
    r"""Call :meth:`~meshmode.array_context.arrayContext.freeze` on the element
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


def flatten(ary: np.ndarray) -> np.ndarray:
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

    group_sizes = [grp_ary.shape[0] * grp_ary.shape[1] for grp_ary in ary]
    group_starts = np.cumsum([0] + group_sizes)

    actx = ary.array_context

    @memoize_in(actx, "flatten_prg")
    def prg():
        return make_loopy_program(
            "{[iel,idof]: 0<=iel<nelements and 0<=idof<nunit_dofs}",
            "result[grp_start + iel*nunit_dofs + idof] = grp_ary[iel, idof]",
            name="flatten")

    result = actx.empty(group_starts[-1], dtype=ary.entry_dtype)

    for grp_start, grp_ary in zip(group_starts, ary):
        actx.call_loopy(prg(), grp_ary=grp_ary, result=result, grp_start=grp_start)

    return result


def unflatten(actx: ArrayContext, discr: "_Discretization", ary) -> np.ndarray:
    r"""Convert a 'flat' array returned by :func:`flatten` back to a :class:`DOFArray`.

    Vectorizes over object arrays of :class:`DOFArray`\ s.
    """
    if (isinstance(ary, np.ndarray)
            and ary.dtype.char == "O"
            and not isinstance(ary, DOFArray)):
        return obj_array_vectorize(
                lambda subary: unflatten(actx, discr, subary),
                ary)

    @memoize_in(actx, "unflatten_prg")
    def prg():
        return make_loopy_program(
            "{[iel,idof]: 0<=iel<nelements and 0<=idof<nunit_dofs}",
            "result[iel, idof] = ary[grp_start + iel*nunit_dofs + idof]",
            name="unflatten")

    group_sizes = [grp.ndofs for grp in discr.groups]
    if ary.size != sum(group_sizes):
        raise ValueError("array has size %d, expected %d"
                % (ary.size, sum(group_sizes)))

    group_starts = np.cumsum([0] + group_sizes)

    return DOFArray.from_list(actx, [
            actx.freeze(
                actx.call_loopy(
                    prg(),
                    grp_start=grp_start, ary=ary,
                    nelements=grp.nelements,
                    nunit_dofs=grp.nunit_dofs,
                    )["result"])
            for grp_start, grp in zip(group_starts, discr.groups)])


def flat_norm(ary: DOFArray, ord=2):
    # FIXME This could be done without flattening and copying
    actx = ary.array_context
    import numpy.linalg as la
    return la.norm(actx.to_numpy(flatten(ary)), ord)


# vim: foldmethod=marker
