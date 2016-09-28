from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2016 Matt Wala"

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
import modepy as mp

import logging
logger = logging.getLogger(__name__)


# {{{ resampling simplex points for refinement

# NOTE: Class internal to refiner: do not make documentation public.
class SimplexResampler(object):
    """
    Resampling of points on simplex elements for refinement.

    Most methods take a ``tesselation`` parameter.
    The tesselation should follow the format of
    :func:`meshmode.mesh.tesselate.tesselatetri()` or
    :func:`meshmode.mesh.tesselate.tesselatetet()`.
    """

    def get_vertex_pair_to_midpoint_order(self, dim):
        """
        :arg dim: Dimension of the element

        :return: A :class:`dict` mapping the vertex pair :math:`(v1, v2)` (with
            :math:`v1 < v2`) to the number of the midpoint in the tesselation
            ordering (the numbering is restricted to the midpoints, so there
            are no gaps in the numbering)
        """
        nmidpoints = dim * (dim + 1) // 2
        return dict(zip(
            ((i, j) for j in range(dim + 1) for i in range(j)),
            range(nmidpoints)
            ))

    def get_midpoints(self, group, tesselation, elements):
        """
        Compute the midpoints of the vertices of the specified elements.

        :arg group: An instance of :class:`meshmode.mesh.SimplexElementGroup`
        :arg tesselation: With attributes `ref_vertices`, `children`
        :arg elements: A list of (group-relative) element numbers

        :return: A :class:`dict` mapping element numbers to midpoint
            coordinates, with each value in the map having shape
            ``(ambient_dim, nmidpoints)``. The ordering of the midpoints
            follows their ordering in the tesselation (see also
            :meth:`SimplexResampler.get_vertex_pair_to_midpoint_order`)
        """
        assert len(group.vertex_indices[0]) == group.dim + 1

        # Get midpoints, converted to unit coordinates.
        midpoints = -1 + np.array([vertex for vertex in
                tesselation.ref_vertices if 1 in vertex], dtype=float)

        resamp_mat = mp.resampling_matrix(
            mp.simplex_best_available_basis(group.dim, group.order),
            midpoints.T,
            group.unit_nodes)

        resamp_midpoints = np.einsum("mu,deu->edm",
                                     resamp_mat,
                                     group.nodes[:, elements])

        return dict(zip(elements, resamp_midpoints))

    def get_tesselated_nodes(self, group, tesselation, elements):
        """
        Compute the nodes of the child elements according to the tesselation.

        :arg group: An instance of :class:`meshmode.mesh.SimplexElementGroup`
        :arg tesselation: With attributes `ref_vertices`, `children`
        :arg elements: A list of (group-relative) element numbers

        :return: A :class:`dict` mapping element numbers to node
            coordinates, with each value in the map having shape
            ``(ambient_dim, nchildren, nunit_nodes)``.
            The ordering of the child nodes follows the ordering
            of ``tesselation.children.``
        """
        assert len(group.vertex_indices[0]) == group.dim + 1

        from meshmode.mesh.refinement.utils import map_unit_nodes_to_children

        # Get child unit node coordinates.
        child_unit_nodes = np.hstack(list(
            map_unit_nodes_to_children(group.unit_nodes, tesselation)))

        resamp_mat = mp.resampling_matrix(
            mp.simplex_best_available_basis(group.dim, group.order),
            child_unit_nodes,
            group.unit_nodes)

        resamp_unit_nodes = np.einsum("cu,deu->edc",
                                      resamp_mat,
                                      group.nodes[:, elements])

        ambient_dim = len(group.nodes)
        nunit_nodes = len(group.unit_nodes[0])

        return dict((elem,
            resamp_unit_nodes[ielem].reshape(
                 (ambient_dim, -1, nunit_nodes)))
            for ielem, elem in enumerate(elements))

# }}}


# vim: foldmethod=marker
