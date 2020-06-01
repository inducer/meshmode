__copyright__ = "Copyright (C) 2020 Benjamin Sepanski"

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
import numpy.linalg as la

from meshmode.interop import ExternalImportHandler


__doc__ = """
.. autoclass:: FIATSimplexCellImporter
"""


# {{{ Compute an affine mapping from given input/outputs

def get_affine_mapping(reference_vects, vects):
    r"""
    Returns (mat, shift),
    a matrix *mat* and vector *shift* which maps the
    *i*th vector in *reference_vects* to the *i*th vector in *vects by

    ..math::

        A ri + b -> vi, \qquad A = mat, b = shift

    :arg reference_vects: An np.array of *n* vectors of dimension *ref_dim*
    :arg vects: An np.array of *n* vectors of dimension *dim*, with
                *ref_dim* <= *dim*.

        NOTE : Should be shape *(ref_dim, nvectors)*, *(dim, nvectors)* respectively.

    *mat* will have shape *(dim, ref_dim)*, *shift* will have shape *(dim,)*
    """
    # Make sure both have same number of vectors
    ref_dim, num_vects = reference_vects.shape
    assert num_vects == vects.shape[1]

    # Make sure d1 <= d2 (see docstring)
    dim = vects.shape[0]
    assert ref_dim <= dim

    # If there is only one vector, set M = I, b = vect - reference
    if num_vects == 1:
        mat = np.eye(dim, ref_dim)
        shift = vects[:, 0] - np.matmul(mat, reference_vects[:, 0])
    else:
        ref_span_vects = reference_vects[:, 1:] - reference_vects[:, 0, np.newaxis]
        span_vects = vects[:, 1:] - vects[:, 0, np.newaxis]
        mat = la.solve(ref_span_vects, span_vects)
        shift = -np.matmul(mat, reference_vects[:, 0]) + vects[:, 0]

    return mat, shift

# }}}


# {{{ Interoperator for FIAT's reference_element.Simplex

class FIATSimplexCellImporter(ExternalImportHandler):
    """
    Import handler for a :mod:`FIAT` simplex cell.

    .. attribute:: data

        An instance of :class:`fiat.FIAT.reference_element.Simplex`.
    """
    def __init__(self, cell):
        """
        :arg cell: a :class:`fiat.FIAT.reference_element.Simplex`.
        """
        # Ensure this cell is actually a simplex
        from FIAT.reference_element import Simplex
        assert isinstance(cell, Simplex)

        super(FIATSimplexCellImporter, self).__init__(cell)

        # Stored as (dim, nunit_vertices)
        from modepy.tools import unit_vertices
        self._unit_vertices = unit_vertices(cell.get_dimension()).T

        # Maps FIAT reference vertices to :mod:`meshmode`
        # unit vertices by x -> Ax + b, where A is :attr:`_mat`
        # and b is :attr:`_shift`
        reference_vertices = np.array(cell.vertices).T
        self._mat, self._shift = get_affine_mapping(reference_vertices,
                                                    self._unit_vertices)

    def make_points(self, dim, entity_id, order):
        """
        Constructs a lattice of points on the *entity_id*th facet
        of dimension *dim*.

        Args are exactly as in
        :meth:`fiat.FIAT.reference_element.Cell.make_points`
        (see `FIAT docs <fiat.FIAT.reference_element.Cell.make_points>`_),
        but the unit nodes are (affinely) mapped to :mod:`modepy`
        `unit coordinates <https://documen.tician.de/modepy/nodes.html>`_.

        :arg dim: Dimension of the facet we are constructing points on.
        :arg entity_id: identifier to determine which facet of
                        dimension *dim* to construct the points on.
        :arg order: Number of points to include in each direction.

        :return: an *np.array* of shape *(dim, npoints)* holding the
                 coordinates of each of the ver
        """
        points = self.data.make_points(dim, entity_id, order)
        if not points:
            return points
        points = np.array(points)
        # Points is (npoints, dim) so have to transpose
        return (np.matmul(self._mat, points.T) + self._shift[:, np.newaxis]).T

# }}}
