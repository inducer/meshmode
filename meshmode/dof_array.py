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
from typing import Optional
from functools import partial

from pytools import single_valued
from pytools.obj_array import obj_array_vectorize

from meshmode.array_context import ArrayContext

__doc__ = """
.. autoclass:: DOFArray
.. autofunction:: thaw
.. autofunction:: freeze
.. autofunction:: flatten
.. autofunction:: unflatten
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

        result = np.empty((len(res_list),), dtype=object).view(cls)
        result[:] = res_list
        result.array_context = actx
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


def flatten_dof_array(ary: DOFArray):
    pass


def unflatten_dof_array(actx: ArrayContext, ary):
    pass


# vim: foldmethod=marker
