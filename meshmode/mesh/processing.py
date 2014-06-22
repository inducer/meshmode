from __future__ import division

__copyright__ = "Copyright (C) 2014 Andreas Kloeckner"

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
import modepy as mp


__doc__ = """
.. autofunction:: find_volume_mesh_element_orientations
.. autofunction:: perform_flips
"""


def find_volume_mesh_element_orientations(mesh):
    result = np.empty(mesh.nelements, dtype=np.float64)

    from meshmode.mesh import SimplexElementGroup

    for grp in mesh.groups:
        if not isinstance(grp, SimplexElementGroup):
            raise NotImplementedError(
                    "finding element orientations "
                    "only supported on "
                    "exclusively SimplexElementGroup-based meshes")

        # (ambient_dim, nelements, nvertices)
        vertices = mesh.vertices[:, grp.vertex_indices]

        # (ambient_dim, nelements, nspan_vectors)
        spanning_vectors = (
                vertices[:, :, 1:] - vertices[:, :, 0][:, :, np.newaxis])

        ambient_dim = spanning_vectors.shape[0]
        nspan_vectors = spanning_vectors.shape[-1]

        spanning_object_array = np.empty(
                (nspan_vectors, ambient_dim),
                dtype=np.object)

        for ispan in xrange(nspan_vectors):
            for idim in xrange(ambient_dim):
                spanning_object_array[ispan, idim] = \
                        spanning_vectors[idim, :, ispan]

        from pymbolic.geometric_algebra import MultiVector

        mvs = [MultiVector(vec) for vec in spanning_object_array]

        from operator import xor
        outer_prod = reduce(xor, mvs)

        signed_area_element = (outer_prod.I | outer_prod).as_scalar()

        result[grp.element_nr_base:grp.element_nr_base + grp.nelements] \
                = signed_area_element

    return result


# {{{ flips

def perform_flips(mesh, flip_flags):
    from modepy.tools import barycentric_to_unit, unit_to_barycentric

    flip_flags = flip_flags.astype(np.bool)

    from meshmode.mesh import Mesh, SimplexElementGroup

    new_groups = []
    for grp in mesh.groups:
        if not isinstance(grp, SimplexElementGroup):
            raise NotImplementedError("flips only supported on "
                    "exclusively SimplexElementGroup-based meshes")

        grp_flip_flags = flip_flags[
                grp.element_nr_base:grp.element_nr_base+grp.nelements]

        # Swap the first two vertices on elements to be flipped.

        new_vertex_indices = grp.vertex_indices.copy()
        new_vertex_indices[grp_flip_flags, 0] \
                = grp.vertex_indices[grp_flip_flags, 1]
        new_vertex_indices[grp_flip_flags, 1] \
                = grp.vertex_indices[grp_flip_flags, 0]

        # Generate a resampling matrix that corresponds to the
        # first two barycentric coordinates being swapped.

        bary_unit_nodes = unit_to_barycentric(grp.unit_nodes)

        flipped_bary_unit_nodes = bary_unit_nodes.copy()
        flipped_bary_unit_nodes[0, :] = bary_unit_nodes[1, :]
        flipped_bary_unit_nodes[1, :] = bary_unit_nodes[0, :]
        flipped_unit_nodes = barycentric_to_unit(flipped_bary_unit_nodes)

        flip_matrix = mp.resampling_matrix(
                mp.simplex_onb(grp.dim, grp.order),
                flipped_unit_nodes, grp.unit_nodes)

        flip_matrix[np.abs(flip_matrix) < 1e-15] = 0

        # Flipping twice should be the identity
        assert la.norm(
                np.dot(flip_matrix, flip_matrix)
                - np.eye(len(flip_matrix))) < 1e-13

        # Apply the flip matrix to the nodes.
        new_nodes = grp.nodes.copy()
        new_nodes[:, grp_flip_flags] = np.einsum(
                "ij,dej->dei",
                flip_matrix, grp.nodes[:, grp_flip_flags])

        new_groups.append(SimplexElementGroup(
            grp.order, new_vertex_indices, new_nodes,
            unit_nodes=grp.unit_nodes))

    return Mesh(mesh.vertices, new_groups)

# }}}


# vim: foldmethod=marker
