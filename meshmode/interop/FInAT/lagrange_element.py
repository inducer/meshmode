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
from meshmode.interop.fiat import FIATSimplexCellImporter


__doc__ = """
.. autoclass:: FinatLagrangeElementImporter
    :members:
"""


class FinatLagrangeElementImporter(ExternalImportHandler):
    """
    An importer for a FInAT element, usually instantiated from
    ``some_instantiated_firedrake_function_space.finat_element``
    """
    def __init__(self, finat_element):
        """
        :param finat_element: A FInAT element of type
            :class:`finat.fiat_elements.Lagrange` or
            :class:`finat.fiat_elements.DiscontinuousLagrange`
            (or :class:`finat.spectral.GaussLegendre` or
            :class:`finat.spectral.GaussLobattoLegendre` in 1D)
            which uses affine mapping of the basis functions
            (i.e. ``finat_element.mapping`` must be
            ``"affine"``)

        :raises TypeError: If :param:`finat_element` is not of type
            :class:`finat.fiat_elements.Lagrange` or
            :class:`finat.fiat_elements.DiscontinuousLagrange`
            (or the 1D classes listed above)

        :raises ValueError: If :param:`finat_element` does not
            use affine mappings of the basis functions
        """
        # {{{ Check and store input

        # Check types
        from finat.fiat_elements import DiscontinuousLagrange, Lagrange
        from finat.spectral import GaussLegendre, GaussLobattoLegendre
        valid_types = (Lagrange, DiscontinuousLagrange,
                      GaussLegendre, GaussLobattoLegendre)
        if not isinstance(finat_element, valid_types):
            raise TypeError(":param:`finat_element` must be of type"
                            " `finat.fiat_elements.Lagrange`,"
                            " `finat.fiat_elements.DiscontinuousLagrange`,"
                            " or `finat.spectral.GaussLegendre` or"
                            " `finat.spectral.GaussLobattoLegendre` in 1D,"
                            " but is instead of type `%s`" % type(finat_element))

        if finat_element.mapping != 'affine':
            raise ValueError("FInAT element must use affine mappings"
                             " of the bases")
        # }}}

        super(FinatLagrangeElementImporter, self).__init__(finat_element)

        self.cell_importer = FIATSimplexCellImporter(finat_element.cell)

        # computed and stored once :meth:`unit_nodes`, :meth:`unit_vertices`,
        # and :meth:`flip_matrix` are called
        self._unit_nodes = None
        self._unit_vertex_indices = None
        self._flip_matrix = None

    def _compute_unit_vertex_indices_and_nodes(self):
        """
        Compute the unit nodes, as well as the unit vertex indices,
        if they have not already been computed.
        """
        if self._unit_nodes is None or self._unit_vertex_indices is None:
            # FIXME : This should work, but uses some private info
            # {{{ Compute unit nodes

            # each point evaluator is a function p(f) evaluates f at a node,
            # so we need to evaluate each point evaluator at the identity to
            # recover the nodes
            point_evaluators = self.data._element.dual.nodes
            unit_nodes = [p(lambda x: x) for p in point_evaluators]
            unit_nodes = np.array(unit_nodes).T
            self._unit_nodes = \
                self.cell_importer.affinely_map_firedrake_to_meshmode(unit_nodes)

            # Is this safe?, I think so bc on a reference element
            close = 1e-8
            # Get vertices as (dim, nunit_vertices)
            unit_vertices = np.array(self.data.cell.vertices).T
            unit_vertices = \
                self.cell_importer.affinely_map_firedrake_to_meshmode(unit_vertices)
            self._unit_vertex_indices = []
            for n_ndx in range(self._unit_nodes.shape[1]):
                for v_ndx in range(unit_vertices.shape[1]):
                    diff = self._unit_nodes[:, n_ndx] - unit_vertices[:, v_ndx]
                    if np.max(np.abs(diff)) < close:
                        self._unit_vertex_indices.append(n_ndx)
                        break

            self._unit_vertex_indices = np.array(self._unit_vertex_indices)

            # }}}

    def dim(self):
        """
        :return: The dimension of the FInAT element's cell
        """
        return self.cell_importer.data.get_dimension()

    def unit_vertex_indices(self):
        """
        :return: A numpy integer array of indices
                 so that *self.unit_nodes()[self.unit_vertex_indices()]*
                 are the nodes of the reference element which coincide
                 with its vertices (this is possibly empty).
        """
        self._compute_unit_vertex_indices_and_nodes()
        return self._unit_vertex_indices

    def unit_nodes(self):
        """
        :return: The unit nodes used by the FInAT element mapped
                 onto the appropriate :mod:`modepy` `reference
                 element <https://documen.tician.de/modepy/nodes.html>`_
                 as an array of shape *(dim, nunit_nodes)*.
        """
        self._compute_unit_vertex_indices_and_nodes()
        return self._unit_nodes

    def nunit_nodes(self):
        """
        :return: The number of unit nodes.
        """
        return self.unit_nodes().shape[1]

    def flip_matrix(self):
        """
        :return: The matrix which should be applied to the
                 *(dim, nunitnodes)*-shaped array of nodes corresponding to
                 an element in order to change orientation - <-> +.

                 The matrix will be *(dim, dim)* and orthogonal with
                 *np.float64* type entries.
        """
        if self._flip_matrix is None:
            # This is very similar to :mod:`meshmode` in processing.py
            # the function :function:`from_simplex_element_group`, but
            # we needed to use firedrake nodes

            from modepy.tools import barycentric_to_unit, unit_to_barycentric

            # Generate a resampling matrix that corresponds to the
            # first two barycentric coordinates being swapped.

            bary_unit_nodes = unit_to_barycentric(self.unit_nodes())

            flipped_bary_unit_nodes = bary_unit_nodes.copy()
            flipped_bary_unit_nodes[0, :] = bary_unit_nodes[1, :]
            flipped_bary_unit_nodes[1, :] = bary_unit_nodes[0, :]
            flipped_unit_nodes = barycentric_to_unit(flipped_bary_unit_nodes)

            from modepy import resampling_matrix, simplex_best_available_basis

            flip_matrix = resampling_matrix(
                simplex_best_available_basis(self.dim(), self.data.degree),
                flipped_unit_nodes, self.unit_nodes())

            flip_matrix[np.abs(flip_matrix) < 1e-15] = 0

            # Flipping twice should be the identity
            assert la.norm(
                np.dot(flip_matrix, flip_matrix)
                - np.eye(len(flip_matrix))) < 1e-13

            self._flip_matrix = flip_matrix

        return self._flip_matrix

    def make_resampling_matrix(self, element_grp):
        """
        :param element_grp: A
            :class:`meshmode.discretization.InterpolatoryElementGroupBase` whose
            basis functions span the same space as the FInAT element.
        :return: A matrix which resamples a function sampled at
                 the firedrake unit nodes to a function sampled at
                 *element_grp.unit_nodes()* (by matrix multiplication)
        """
        from meshmode.discretization import InterpolatoryElementGroupBase
        assert isinstance(element_grp, InterpolatoryElementGroupBase), \
            "element group must be an interpolatory element group so that" \
            " can redistribute onto its nodes"

        from modepy import resampling_matrix
        return resampling_matrix(element_grp.basis(),
                                 new_nodes=element_grp.unit_nodes,
                                 old_nodes=self.unit_nodes())
