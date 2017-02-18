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
import numpy.linalg as la

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


def _mesh_mapping(dim, node_tuples, all_coeffs, rst):
    xyz = np.zeros(dim)

    for ixyz_axis in range(dim):
        for icoeff in range(len(node_tuples)):
            cur = all_coeffs[ixyz_axis][icoeff]
            for irst_axis in range(len(node_tuples[icoeff])):
                if node_tuples[icoeff][irst_axis] == 1:
                    cur *= rst[irst_axis]
            xyz[ixyz_axis] += cur

    return xyz


def _mesh_mapping_jacobian(dim, node_tuples, all_coeffs, rst):
    jacobian = np.zeros((dim, dim))

    for ixyz_axis in range(dim):
        for irst_axis in range(dim):

            for icoeff, node_tuple in enumerate(node_tuples):
                cur = all_coeffs[ixyz_axis][icoeff]
                for irst_axis_term, power in enumerate(node_tuple):
                    if irst_axis_term == irst_axis:
                        # differentiating by that one, do not multiply
                        if power == 0:
                            cur = 0
                    else:
                        cur *= rst[irst_axis_term]**power

                jacobian[ixyz_axis, irst_axis] += cur

    return jacobian


def check_nodal_adj_against_geometry(mesh, tol=1e-12):
    from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
    from pytools import generate_nonnegative_integer_tuples_below as gnitb
    def group_and_iel_to_global_iel(igrp, iel):
        return mesh.groups[igrp].element_nr_base + iel

    logger.debug("nodal adj test: tree build")
    from meshmode.mesh.tools import make_element_lookup_tree
    tree = make_element_lookup_tree(mesh, eps=tol)
    logger.debug("nodal adj test: tree build done")

    from meshmode.mesh.processing import find_bounding_box
    bbox_min, bbox_max = find_bounding_box(mesh)

    nadj = mesh.nodal_adjacency

    connected_to_element_geometry = [set() for i in range(mesh.nelements)]
    connected_to_element_connectivity = [set() for i in range(mesh.nelements)]

    for igrp, grp in enumerate(mesh.groups):
        for iel_grp in range(grp.nelements):
            iel_g = group_and_iel_to_global_iel(igrp, iel_grp)
            nb_starts = nadj.neighbors_starts
            for nb_iel_g in nadj.neighbors[nb_starts[iel_g]:nb_starts[iel_g+1]]:
                connected_to_element_connectivity[iel_g].add(nb_iel_g)

            for vertex_index in grp.vertex_indices[iel_grp]:
                other_vertex = mesh.vertices[:, vertex_index]

                # check which elements touch this other_vertex
                for nearby_igrp, nearby_iel in tree.generate_matches(other_vertex):
                    if nearby_igrp == igrp and nearby_iel == iel_grp:
                        continue
                    #if igrp == 1 and nearby_igrp == 1 and nearby_iel == 1 and iel_grp == 3:
                        #pu.db


                    nearby_grp = mesh.groups[nearby_igrp]
                    nvertices_per_element = len(nearby_grp.vertex_indices[0])
                    if isinstance(nearby_grp, SimplexElementGroup):
                        nearby_origin_vertex = mesh.vertices[
                                :, nearby_grp.vertex_indices[nearby_iel][0]]  # noqa
                        transformation = np.empty(
                                (len(mesh.vertices), nvertices_per_element-1))
                        vertex_transformed = other_vertex - nearby_origin_vertex

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
                    elif isinstance(nearby_grp, TensorProductElementGroup):
                        from meshmode.mesh.tesselate import tesselatesquare, tesselatecube
                        if nearby_grp.dim == 2:
                            node_tuples = list(gnitb(2, 2))
                        elif nearby_grp.dim == 3:
                            node_tuples = list(gnitb(2, 3))

                        # TODO: Shifting everything by nearby_origin_vertex should not be necessary
                        #nearby_origin_vertex = mesh.vertices[
                                #:, nearby_grp.vertex_indices[nearby_iel][0]]  # noqa

                        my_vertices = mesh.vertices[:, nearby_grp.vertex_indices[nearby_iel, :]]
                        min_vert = np.min(my_vertices, axis=1)
                        max_vert = np.max(my_vertices, axis=1)

                        if (
                                (other_vertex < min_vert - tol).any()
                                or (max_vert + tol < other_vertex).any()):
                            # other_vertex not nearby_iel's bounding box, don't even
                            # attempt Newton (no, really, don't!)
                            continue

                        #vertex_transformed = other_vertex - nearby_origin_vertex
                        all_coeffs = np.ones((nearby_grp.dim, len(node_tuples)))
                        for cur_dim in range(nearby_grp.dim):
                            vandermonde = np.ones(
                                    (len(node_tuples), len(node_tuples)))

                            # i: row of vandermonde, what node
                            for i in range(len(node_tuples)):
                                # j: column of vandermonde, what function
                                for j in range(len(node_tuples)):
                                    for k in range(len(node_tuples[j])):
                                        if node_tuples[j][k] == 1:
                                            vandermonde[i, j] *= node_tuples[i][k]

                            b = np.ones(len(node_tuples))
                            for inearby_vertex_index, nearby_vertex_index in enumerate(
                                    nearby_grp.vertex_indices[nearby_iel, :]):
                                #b[inearby_vertex_index] = mesh.vertices[cur_dim, nearby_vertex_index] - nearby_origin_vertex[cur_dim]
                                b[inearby_vertex_index] = mesh.vertices[cur_dim, nearby_vertex_index]

                            coefficients = np.linalg.solve(vandermonde, b)
                            all_coeffs[cur_dim] = coefficients

                        # {{{ Newton's method to find rst coordinates corresponding
                        # to 'other_vertex'

                        # current rst guess
                        cur_coords = np.zeros(nearby_grp.dim)

                        niterations = 15
                        while True:
                            jacobian = np.zeros((nearby_grp.dim, nearby_grp.dim))
                            # i: row of the jacobian (xyz output component of the mapping)
                            for i in range(nearby_grp.dim):
                                # j: column of the jacobian (rst input component)
                                for j in range(nearby_grp.dim):
                                    for k in range(len(node_tuples)):
                                        my_rst_powers = node_tuples[k]
                                        # node_tuples[k] represents powers of rst
                                        if my_rst_powers[j] == 1:
                                            cur = all_coeffs[i, k]
                                            for l in range(len(my_rst_powers)):
                                                if l != j and my_rst_powers[l] == 1:
                                                    cur *= cur_coords[l]
                                            jacobian[i, j] += cur

                            def fmap(rst):
                                return (
                                        _mesh_mapping(nearby_grp.dim, node_tuples, all_coeffs, rst)
                                        )
                            def f(rst):
                                return (
                                        _mesh_mapping(nearby_grp.dim, node_tuples, all_coeffs, rst)
                                        - other_vertex)

                            def f_jacobian(rst):
                                return (
                                        _mesh_mapping_jacobian(nearby_grp.dim, node_tuples, all_coeffs, rst))
                            f_value = f(cur_coords)
                            f_jacobian_value = f_jacobian(cur_coords)

                            cur_coords = cur_coords - la.solve(f_jacobian_value, f_value)

                            # FIXME: Should be check relative to element size
                            if la.norm(f_value, 2) < tol/2:
                                break

                            niterations -= 1
                            if niterations == 0:

                                raise RuntimeError("Newton's method in in-element "
                                        "test did not converge within iteration "
                                        "limit")

                        # }}}

                        is_connected = (
                                (cur_coords <= 1+tol).all()
                                and (cur_coords>= -tol).all())
                        el1 = group_and_iel_to_global_iel(nearby_igrp, nearby_iel)
                        el2 = group_and_iel_to_global_iel(igrp, iel_grp)

                        if is_connected:
                            connected_to_element_geometry[el1].add(el2)
                            connected_to_element_geometry[el2].add(el1)
                    else:
                        raise TypeError("unexpected el group type: %s"
                                % type(nearby_grp).__name__)

    #assert is_symmetric(connected_to_element_connectivity, debug=True)

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
    for i in range(len(connected_to_element_geometry)):
        if connected_to_element_geometry[i] != connected_to_element_connectivity[i]:
            print (i, connected_to_element_connectivity[i], connected_to_element_geometry[i])

    assert connected_to_element_geometry == connected_to_element_connectivity

# }}}

# vim: foldmethod=marker
