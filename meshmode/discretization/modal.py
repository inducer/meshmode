__copyright__ = """
Copyright (C) 2021 University of Illinois Board of Trustees
"""

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

from meshmode.array_context import ArrayContext
from meshmode.discretization import DiscretizationBase
from meshmode.discretization.poly_element import (
        ModalSimplexGroupFactory,
        ModalTensorProductGroupFactory)


class ModalDiscretization(DiscretizationBase):
    """An unstructured composite modal discretization."""

    def __init__(self, actx: ArrayContext, mesh, group_factory,
            real_dtype=np.float64):
        """
        :param actx: A :class:`ArrayContext` used to perform computation needed
            during initial set-up of the mesh.
        :param mesh: A :class:`meshmode.mesh.Mesh` over which the discretization is
            built.
        :param group_factory: An :class:`ElementGroupFactory`. Note that the
            element groups must be subclasses of :class:`ModalElementGroupBase`.
        :param real_dtype: The :mod:`numpy` data type used for representing real
            data, either :class:`numpy.float32` or :class:`numpy.float64`.
        """

        # Some sanity checking so we don't do something horrendous later on
        if not isinstance(group_factory, (ModalSimplexGroupFactory,
                                          ModalTensorProductGroupFactory)):
            raise ValueError("You must use a modal group factory when using "
                             "a ModalDiscretization.")

        super().__init__(actx, mesh, group_factory, real_dtype)
