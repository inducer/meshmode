from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2014-6 Shivam Gupta, Andreas Kloeckner"

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

import logging
logger = logging.getLogger(__name__)


# {{{ map unit nodes to children


def map_unit_nodes_to_children(unit_nodes, tesselation):
    """
    Given a collection of unit nodes, return the coordinates of the
    unit nodes mapped onto each of the children of the reference
    element.

    The tesselation should follow the format of
    :func:`meshmode.mesh.tesselate.tesselatetri()` or
    :func:`meshmode.mesh.tesselate.tesselatetet()`.

    `unit_nodes` should be relative to the unit simplex coordinates in
    :module:`modepy`.

    :arg unit_nodes: shaped `(dim, nunit_nodes)`
    :arg tesselation: With attributes `ref_vertices`, `children`
    """
    ref_vertices = np.array(tesselation.ref_vertices, dtype=np.float)

    assert len(unit_nodes.shape) == 2

    for child_element in tesselation.children:
        center = np.vstack(ref_vertices[child_element[0]])
        # Scale by 1/2 since sides in the tesselation have length 2.
        aff_mat = (ref_vertices.T[:, child_element[1:]] - center) / 2
        # (-1, -1, ...) in unit_nodes = (0, 0, ...) in ref_vertices.
        # Hence the translation by +/- 1.
        yield aff_mat.dot(unit_nodes + 1) + center - 1

# }}}

# {{{ test nodal adjacency against geometry


def is_symmetric(relation, debug=False):
    for a, other_list in enumerate(relation):
        for b in other_list:
            if a not in relation[b]:
                if debug:
                    print("Relation is not symmetric: %s -> %s, but not %s -> %s"
                            % (a, b, b, a))
                return False

    return True


def check_nodal_adj_against_geometry(mesh, tol=1e-12):
    def group_and_iel_to_global_iel(igrp, iel):
        return mesh.groups[igrp].element_nr_base + iel

    logger.debug("nodal adj test: tree build")
    from meshmode.mesh.tools import make_element_lookup_tree
    tree = make_element_lookup_tree(mesh, eps=tol)
    logger.debug("nodal adj test: tree build done")

    from meshmode.mesh.processing import find_bounding_box
    bbox_min, bbox_max = find_bounding_box(mesh)

    nadj = mesh.nodal_adjacency
    nvertices_per_element = len(mesh.groups[0].vertex_indices[0])

    connected_to_element_geometry = [set() for i in range(mesh.nelements)]
    connected_to_element_connectivity = [set() for i in range(mesh.nelements)]

    for igrp, grp in enumerate(mesh.groups):
        for iel_grp in range(grp.nelements):
            iel_g = group_and_iel_to_global_iel(igrp, iel_grp)
            nb_starts = nadj.neighbors_starts
            for nb_iel_g in nadj.neighbors[nb_starts[iel_g]:nb_starts[iel_g+1]]:
                connected_to_element_connectivity[iel_g].add(nb_iel_g)

            for vertex_index in grp.vertex_indices[iel_grp]:
                vertex = mesh.vertices[:, vertex_index]

                # check which elements touch this vertex
                for nearby_igrp, nearby_iel in tree.generate_matches(vertex):
                    if nearby_igrp == igrp and nearby_iel == iel_grp:
                        continue
                    nearby_grp = mesh.groups[nearby_igrp]

                    nearby_origin_vertex = mesh.vertices[
                            :, nearby_grp.vertex_indices[nearby_iel][0]]  # noqa
                    transformation = np.empty(
                            (len(mesh.vertices), nvertices_per_element-1))
                    vertex_transformed = vertex - nearby_origin_vertex

                    for inearby_vertex_index, nearby_vertex_index in enumerate(
                            nearby_grp.vertex_indices[nearby_iel][1:]):
                        nearby_vertex = mesh.vertices[:, nearby_vertex_index]
                        transformation[:, inearby_vertex_index] = \
                                nearby_vertex - nearby_origin_vertex
                    bary_coord, residual = \
                            np.linalg.lstsq(transformation, vertex_transformed)[0:2]

                    is_in_element_span = (
                            residual.size == 0 or
                            np.linalg.norm(vertex_transformed) == 0 or
                            (np.linalg.norm(residual) /
                                np.linalg.norm(vertex_transformed)) <= tol)

                    is_connected = (
                            is_in_element_span
                            and np.sum(bary_coord) <= 1+tol
                            and (bary_coord >= -tol).all())
                    el1 = group_and_iel_to_global_iel(nearby_igrp, nearby_iel)
                    el2 = group_and_iel_to_global_iel(igrp, iel_grp)

                    if is_connected:
                        connected_to_element_geometry[el1].add(el2)
                        connected_to_element_geometry[el2].add(el1)

    assert is_symmetric(connected_to_element_connectivity, debug=True)

    # The geometric adjacency relation isn't necessary symmetric:
    #
    #        /|
    #       / |
    #      /  |\
    #    B \  |/  A
    #       \ |
    #        \|
    #
    # Element A will see element B (its vertices are near B) but not the other
    # way around.

    assert connected_to_element_geometry == connected_to_element_connectivity

# }}}

# vim: foldmethod=marker
