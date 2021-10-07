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

from functools import reduce
from numbers import Real
from typing import Optional, Union

from dataclasses import dataclass

import numpy as np
import numpy.linalg as la

import modepy as mp

from meshmode.mesh import (
    InteriorAdjacencyGroup,
    BoundaryAdjacencyGroup,
    InterPartitionAdjacencyGroup
)

from meshmode.mesh.tools import AffineMap


__doc__ = """
.. autoclass:: BoundaryPairMapping

.. autofunction:: find_group_indices
.. autofunction:: partition_mesh
.. autofunction:: find_volume_mesh_element_orientations
.. autofunction:: flip_simplex_element_group
.. autofunction:: perform_flips
.. autofunction:: find_bounding_box
.. autofunction:: merge_disjoint_meshes
.. autofunction:: split_mesh_groups
.. autofunction:: glue_mesh_boundaries

.. autofunction:: map_mesh
.. autofunction:: affine_map
.. autofunction:: rotate_mesh_around_axis
"""


def find_group_indices(groups, meshwide_elems):
    """
    :arg groups: A list of :class:`~meshmode.mesh.MeshElementGroup` instances
        that contain *meshwide_elems*.
    :arg meshwide_elems: A :class:`numpy.ndarray` of mesh-wide element numbers.
        Usually computed by ``elem + element_nr_base``.
    :returns: A :class:`numpy.ndarray` of group numbers that *meshwide_elem*
        belongs to.
    """
    grps = np.zeros_like(meshwide_elems)
    next_grp_boundary = 0
    for grp in groups:
        next_grp_boundary += grp.nelements
        grps += meshwide_elems >= next_grp_boundary

    return grps


# {{{ partition_mesh

def _compute_global_elem_to_part_elem(part_per_element, parts, element_id_dtype):
    """
    Create a map from global element index to partition-wide element index for
    a set of partitions.

    :arg part_per_element: A :class:`numpy.ndarray` mapping element indices to
        partition numbers.
    :arg parts: A :class:`set` of partition numbers.
    :arg element_id_dtype: The element index data type.
    :returns: A :class:`numpy.ndarray` that maps an element's global index
        to its corresponding partition-wide index if that partition belongs to
        *parts* (and if not, to -1).
    """
    global_elem_to_part_elem = np.full(len(part_per_element), -1,
                dtype=element_id_dtype)
    for ipart in parts:
        belongs_to_part = part_per_element == ipart
        global_elem_to_part_elem[belongs_to_part] = (
            np.cumsum(belongs_to_part)[belongs_to_part]-1)

    return global_elem_to_part_elem


def _filter_mesh_groups(groups, selected_elements, vertex_id_dtype):
    """
    Create new mesh groups containing a selected subset of elements.

    :arg groups: An array of `~meshmode.mesh.ElementGroup` instances.
    :arg selected_elements: A sorted array of indices of elements to be included in
        the filtered groups.
    :returns: A tuple ``(new_groups, group_to_new_group, required_vertex_indices)``,
        where *new_groups* is made up of groups from *groups* with elements not in
        *selected_elements* removed (Note: empty groups are omitted),
        *group_to_new_group* maps groups in *groups* to their corresponding location
        in *new_groups*, and *required_vertex_indices* contains indices of all
        vertices required for elements belonging to *new_groups*.
    """

    # {{{ find n_new_groups, group_to_new_group, filtered_group_elements

    group_elem_starts = [np.searchsorted(selected_elements, grp.element_nr_base)
                for grp in groups] + [len(selected_elements)]

    new_group_to_old_group = []
    filtered_group_elements = []
    for igrp, grp in enumerate(groups):
        start_idx, end_idx = group_elem_starts[igrp:igrp+2]
        if end_idx == start_idx:
            continue

        new_group_to_old_group.append(igrp)
        filtered_group_elements.append(selected_elements[start_idx:end_idx]
                    - grp.element_nr_base)

    n_new_groups = len(new_group_to_old_group)

    group_to_new_group = [None] * len(groups)
    for i_new_grp, i_old_grp in enumerate(new_group_to_old_group):
        group_to_new_group[i_old_grp] = i_new_grp

    del grp

    # }}}

    # {{{ filter vertex indices

    filtered_vertex_indices = [
            groups[i_old_grp].vertex_indices[
                    filtered_group_elements[i_new_grp], :]
            for i_new_grp, i_old_grp in enumerate(new_group_to_old_group)]

    if n_new_groups > 0:
        filtered_vertex_indices_flat = np.concatenate([indices.ravel() for indices
                    in filtered_vertex_indices])
    else:
        filtered_vertex_indices_flat = np.empty(0, dtype=vertex_id_dtype)

    required_vertex_indices, new_vertex_indices_flat = np.unique(
                filtered_vertex_indices_flat, return_inverse=True)

    new_vertex_indices = []
    start_idx = 0
    for filtered_indices in filtered_vertex_indices:
        end_idx = start_idx + filtered_indices.size
        new_vertex_indices.append(new_vertex_indices_flat[start_idx:end_idx]
                    .reshape(filtered_indices.shape).astype(vertex_id_dtype))
        start_idx = end_idx

    # }}}

    new_groups = [
            groups[i_old_grp].copy(
                vertex_indices=new_vertex_indices[i_new_grp],
                nodes=groups[i_old_grp].nodes[
                    :, filtered_group_elements[i_new_grp], :].copy())
            for i_new_grp, i_old_grp in enumerate(new_group_to_old_group)]

    return new_groups, group_to_new_group, required_vertex_indices


def _get_connected_partitions(
        mesh, part_per_element, global_elem_to_part_elem):
    """
    Find the partitions that are connected to the current partition.

    :arg mesh: A :class:`~meshmode.mesh.Mesh` representing the unpartitioned mesh.
    :arg part_per_element: A :class:`numpy.ndarray` mapping element indices to
        partition numbers.
    :arg global_elem_to_part_elem: A :class:`numpy.ndarray` mapping from global
        element index to local partition-wide element index for local elements (and
        -1 otherwise).

    :returns: A :class:`set` of indices of the neighboring partitions.
    """
    connected_parts = set()

    for igrp, facial_adj_list in enumerate(mesh.facial_adjacency_groups):
        int_grps = [
            grp for grp in facial_adj_list
            if isinstance(grp, InteriorAdjacencyGroup)]
        for facial_adj in int_grps:
            jgrp = facial_adj.ineighbor_group

            elem_base_i = mesh.groups[igrp].element_nr_base
            elem_base_j = mesh.groups[jgrp].element_nr_base

            elements_are_local = global_elem_to_part_elem[facial_adj.elements
                        + elem_base_i] >= 0
            neighbors_are_nonlocal = global_elem_to_part_elem[facial_adj.neighbors
                        + elem_base_j] < 0

            connected_parts.update(
                part_per_element[
                    facial_adj.neighbors[
                        elements_are_local & neighbors_are_nonlocal]
                    + elem_base_j])

    return connected_parts


def _create_local_to_local_adjacency_groups(mesh, global_elem_to_part_elem,
            part_mesh_groups, global_group_to_part_group,
            part_mesh_group_elem_base):
    r"""
    Create local-to-local facial adjacency groups for a partitioned mesh.

    :arg mesh: A :class:`~meshmode.mesh.Mesh` representing the unpartitioned mesh.
    :arg global_elem_to_part_elem: A :class:`numpy.ndarray` mapping from global
        element index to local partition-wide element index for local elements (and
        -1 otherwise).
    :arg part_mesh_groups: An array of :class:`~meshmode.mesh.ElementGroup` instances
        representing the partitioned mesh groups.
    :arg global_group_to_part_group: An array mapping groups in *mesh* to groups in
        *part_mesh_groups* (or `None` if the group is not local).
    :arg part_mesh_group_elem_base: An array containing the starting partition-wide
        element index for each group in *part_mesh_groups*.

    :returns: A list of lists of `~meshmode.mesh.InteriorAdjacencyGroup` instances
        corresponding to the entries in *mesh.facial_adjacency_groups* that
        have local-to-local adjacency.
    """
    local_to_local_adjacency_groups = [[] for _ in part_mesh_groups]

    for igrp, facial_adj_list in enumerate(mesh.facial_adjacency_groups):
        i_part_grp = global_group_to_part_group[igrp]
        if i_part_grp is None:
            continue

        int_grps = [
            grp for grp in facial_adj_list
            if isinstance(grp, InteriorAdjacencyGroup)]

        for facial_adj in int_grps:
            jgrp = facial_adj.ineighbor_group

            j_part_grp = global_group_to_part_group[jgrp]
            if j_part_grp is None:
                continue

            elem_base_i = mesh.groups[igrp].element_nr_base
            elem_base_j = mesh.groups[jgrp].element_nr_base

            elements_are_local = global_elem_to_part_elem[facial_adj.elements
                        + elem_base_i] >= 0
            neighbors_are_local = global_elem_to_part_elem[facial_adj.neighbors
                        + elem_base_j] >= 0

            adj_indices, = np.where(elements_are_local & neighbors_are_local)

            if len(adj_indices) > 0:
                part_elem_base_i = part_mesh_group_elem_base[i_part_grp]
                part_elem_base_j = part_mesh_group_elem_base[j_part_grp]

                elements = global_elem_to_part_elem[facial_adj.elements[
                            adj_indices] + elem_base_i] - part_elem_base_i
                element_faces = facial_adj.element_faces[adj_indices]
                neighbors = global_elem_to_part_elem[facial_adj.neighbors[
                            adj_indices] + elem_base_j] - part_elem_base_j
                neighbor_faces = facial_adj.neighbor_faces[adj_indices]

                local_to_local_adjacency_groups[i_part_grp].append(
                    InteriorAdjacencyGroup(
                        igroup=i_part_grp,
                        ineighbor_group=j_part_grp,
                        elements=elements,
                        element_faces=element_faces,
                        neighbors=neighbors,
                        neighbor_faces=neighbor_faces,
                        aff_map=facial_adj.aff_map))

    return local_to_local_adjacency_groups


def _create_nonlocal_adjacency_groups(
        mesh, part_per_element, global_elem_to_part_elem, part_mesh_groups,
        global_group_to_part_group, part_mesh_group_elem_base, connected_parts):
    """
    Create non-local adjacency groups for the partitioned mesh.

    :arg mesh: A :class:`~meshmode.mesh.Mesh` representing the unpartitioned mesh.
    :arg part_per_element: A :class:`numpy.ndarray` mapping element indices to
        partition numbers.
    :arg global_elem_to_part_elem: A :class:`numpy.ndarray` mapping from global
        element index to local partition-wide element index for local elements (and
        -1 otherwise).
    :arg part_mesh_groups: An array of `~meshmode.mesh.ElementGroup` instances
        representing the partitioned mesh groups.
    :arg global_group_to_part_group: An array mapping groups in *mesh* to groups in
        *part_mesh_groups* (or `None` if the group is not local).
    :arg part_mesh_group_elem_base: An array containing the starting partition-wide
        element index for each group in *part_mesh_groups*.
    :arg connected_parts: A :class:`set` containing the partitions connected to
        the current one.

    :returns: A list of lists of `~meshmode.mesh.InterPartitionAdjacencyGroup`
        instances corresponding to the entries in *mesh.facial_adjacency_groups* that
        have non-local adjacency.
    """
    global_elem_to_neighbor_elem = _compute_global_elem_to_part_elem(
        part_per_element, connected_parts, mesh.element_id_dtype)

    nonlocal_adj_groups = [[] for _ in part_mesh_groups]

    for igrp, facial_adj_list in enumerate(mesh.facial_adjacency_groups):
        i_part_grp = global_group_to_part_group[igrp]
        if i_part_grp is None:
            continue

        int_grps = [
            grp for grp in facial_adj_list
            if isinstance(grp, InteriorAdjacencyGroup)]

        for facial_adj in int_grps:
            jgrp = facial_adj.ineighbor_group

            elem_base_i = mesh.groups[igrp].element_nr_base
            elem_base_j = mesh.groups[jgrp].element_nr_base

            global_elements = facial_adj.elements + elem_base_i
            global_neighbors = facial_adj.neighbors + elem_base_j

            elements_are_local = global_elem_to_part_elem[global_elements] >= 0

            neighbor_parts = part_per_element[global_neighbors]

            for i_neighbor_part in connected_parts:
                adj_indices, = np.where(
                    elements_are_local
                    & (neighbor_parts == i_neighbor_part))

                if len(adj_indices) > 0:
                    part_elem_base_i = part_mesh_group_elem_base[i_part_grp]

                    elements = global_elem_to_part_elem[facial_adj.elements[
                                adj_indices] + elem_base_i] - part_elem_base_i
                    element_faces = facial_adj.element_faces[adj_indices]
                    neighbors = global_elem_to_neighbor_elem[
                        global_neighbors[adj_indices]]
                    neighbor_faces = facial_adj.neighbor_faces[adj_indices]

                    nonlocal_adj_groups[i_part_grp].append(
                        InterPartitionAdjacencyGroup(
                            igroup=i_part_grp,
                            ineighbor_partition=i_neighbor_part,
                            elements=elements,
                            element_faces=element_faces,
                            neighbors=neighbors,
                            neighbor_faces=neighbor_faces,
                            aff_map=facial_adj.aff_map))

    return nonlocal_adj_groups


def _create_boundary_groups(mesh, global_elem_to_part_elem, part_mesh_groups,
            global_group_to_part_group, part_mesh_group_elem_base):
    """
    Create boundary groups for partitioned mesh.

    :arg mesh: A :class:`~meshmode.mesh.Mesh` representing the unpartitioned mesh.
    :arg global_elem_to_part_elem: A :class:`numpy.ndarray` mapping from global
        element index to local partition-wide element index for local elements (and
        -1 otherwise).
    :arg part_mesh_groups: An array of `~meshmode.mesh.ElementGroup` instances
        representing the partitioned mesh groups.
    :arg global_group_to_part_group: An array mapping groups in *mesh* to groups in
        *part_mesh_groups* (or `None` if the group is not local).
    :arg part_mesh_group_elem_base: An array containing the starting partition-wide
        element index for each group in *part_mesh_groups*.

    :returns: A list of lists of `~meshmode.mesh.BoundaryAdjacencyGroup` instances
        corresponding to the entries in *mesh.facial_adjacency_groups* that have
        boundary faces.
    """
    bdry_adj_groups = [[] for _ in part_mesh_groups]

    for igrp, facial_adj_list in enumerate(mesh.facial_adjacency_groups):
        i_part_grp = global_group_to_part_group[igrp]
        if i_part_grp is None:
            continue

        bdry_grps = [
            grp for grp in facial_adj_list
            if isinstance(grp, BoundaryAdjacencyGroup)]

        for bdry_grp in bdry_grps:
            elem_base = mesh.groups[igrp].element_nr_base

            adj_indices, = np.where(global_elem_to_part_elem[bdry_grp.elements
                        + elem_base] >= 0)

            if len(adj_indices) > 0:
                part_elem_base = part_mesh_group_elem_base[i_part_grp]

                elements = global_elem_to_part_elem[bdry_grp.elements[adj_indices]
                            + elem_base] - part_elem_base
                element_faces = bdry_grp.element_faces[adj_indices]

                bdry_adj_groups[i_part_grp].append(
                    BoundaryAdjacencyGroup(
                        igroup=i_part_grp,
                        boundary_tag=bdry_grp.boundary_tag,
                        elements=elements,
                        element_faces=element_faces))

    return bdry_adj_groups


def partition_mesh(mesh, part_per_element, part_num):
    """
    :arg mesh: A :class:`~meshmode.mesh.Mesh` to be partitioned.
    :arg part_per_element: A :class:`numpy.ndarray` containing one
        integer per element of *mesh* indicating which part of the
        partitioned mesh the element is to become a part of.
    :arg part_num: The part number of the mesh to return.

    :returns: A tuple ``(part_mesh, part_to_global)``, where *part_mesh*
        is a :class:`~meshmode.mesh.Mesh` that is a partition of mesh, and
        *part_to_global* is a :class:`numpy.ndarray` mapping element
        numbers on *part_mesh* to ones in *mesh*.

    .. versionadded:: 2017.1
    """
    assert len(part_per_element) == mesh.nelements, (
        "part_per_element must have shape (mesh.nelements,)")

    # Contains the indices of the elements requested.
    queried_elems, = np.where(part_per_element == part_num)

    global_elem_to_part_elem = _compute_global_elem_to_part_elem(part_per_element,
                {part_num}, mesh.element_id_dtype)

    # Create new mesh groups that mimic the original mesh's groups but only contain
    # the local partition's elements
    part_mesh_groups, global_group_to_part_group, required_vertex_indices =\
                _filter_mesh_groups(mesh.groups, queried_elems, mesh.vertex_id_dtype)

    part_vertices = np.zeros((mesh.ambient_dim, len(required_vertex_indices)))
    for dim in range(mesh.ambient_dim):
        part_vertices[dim] = mesh.vertices[dim][required_vertex_indices]

    part_mesh_group_elem_base = [0 for _ in part_mesh_groups]
    el_nr = 0
    for i_part_grp, grp in enumerate(part_mesh_groups):
        part_mesh_group_elem_base[i_part_grp] = el_nr
        el_nr += grp.nelements

    connected_parts = _get_connected_partitions(
        mesh, part_per_element, global_elem_to_part_elem)

    local_to_local_adj_groups = _create_local_to_local_adjacency_groups(
                mesh, global_elem_to_part_elem, part_mesh_groups,
                global_group_to_part_group, part_mesh_group_elem_base)

    nonlocal_adj_groups = _create_nonlocal_adjacency_groups(
                mesh, part_per_element, global_elem_to_part_elem,
                part_mesh_groups, global_group_to_part_group,
                part_mesh_group_elem_base, connected_parts)

    boundary_adj_groups = _create_boundary_groups(
                mesh, global_elem_to_part_elem, part_mesh_groups,
                global_group_to_part_group, part_mesh_group_elem_base)

    # Combine local/nonlocal/boundary adjacency groups
    part_facial_adj_groups = [
        local_to_local_adj_groups[i_part_grp]
        + nonlocal_adj_groups[i_part_grp]
        + boundary_adj_groups[i_part_grp]
        for i_part_grp in range(len(part_mesh_groups))]

    from meshmode.mesh import Mesh
    part_mesh = Mesh(
            part_vertices,
            part_mesh_groups,
            facial_adjacency_groups=part_facial_adj_groups,
            is_conforming=mesh.is_conforming)

    return part_mesh, queried_elems

# }}}


# {{{ orientations

def find_volume_mesh_element_group_orientation(vertices, grp):
    """Return a positive floating point number for each positively
    oriented element, and a negative floating point number for
    each negatively oriented element.
    """

    from meshmode.mesh import SimplexElementGroup

    if not isinstance(grp, SimplexElementGroup):
        raise NotImplementedError(
                "finding element orientations "
                "only supported on "
                "exclusively SimplexElementGroup-based meshes")

    # (ambient_dim, nelements, nvertices)
    my_vertices = vertices[:, grp.vertex_indices]

    # (ambient_dim, nelements, nspan_vectors)
    spanning_vectors = (
            my_vertices[:, :, 1:] - my_vertices[:, :, 0][:, :, np.newaxis])

    ambient_dim = spanning_vectors.shape[0]
    nspan_vectors = spanning_vectors.shape[-1]

    if ambient_dim != grp.dim:
        raise ValueError("can only find orientation of volume meshes")

    spanning_object_array = np.empty(
            (nspan_vectors, ambient_dim),
            dtype=object)

    for ispan in range(nspan_vectors):
        for idim in range(ambient_dim):
            spanning_object_array[ispan, idim] = \
                    spanning_vectors[idim, :, ispan]

    from pymbolic.geometric_algebra import MultiVector

    mvs = [MultiVector(vec) for vec in spanning_object_array]

    from operator import xor
    outer_prod = -reduce(xor, mvs)      # pylint: disable=invalid-unary-operand-type

    if grp.dim == 1:
        # FIXME: This is a little weird.
        outer_prod = -outer_prod

    return (outer_prod.I | outer_prod).as_scalar()


def find_volume_mesh_element_orientations(mesh, tolerate_unimplemented_checks=False):
    """Return a positive floating point number for each positively
    oriented element, and a negative floating point number for
    each negatively oriented element.

    :arg tolerate_unimplemented_checks: If *True*, elements for which no
        check is available will return *NaN*.
    """

    result = np.empty(mesh.nelements, dtype=np.float64)

    for grp in mesh.groups:
        result_grp_view = result[
                grp.element_nr_base:grp.element_nr_base + grp.nelements]

        if tolerate_unimplemented_checks:
            try:
                signed_area_elements = \
                        find_volume_mesh_element_group_orientation(
                                mesh.vertices, grp)
            except NotImplementedError:
                result_grp_view[:] = float("nan")
            else:
                assert not np.isnan(signed_area_elements).any()
                result_grp_view[:] = signed_area_elements
        else:
            signed_area_elements = \
                    find_volume_mesh_element_group_orientation(
                            mesh.vertices, grp)
            assert not np.isnan(signed_area_elements).any()
            result_grp_view[:] = signed_area_elements

    return result


def test_volume_mesh_element_orientations(mesh):
    area_elements = find_volume_mesh_element_orientations(
            mesh, tolerate_unimplemented_checks=True)

    valid = ~np.isnan(area_elements)

    return (area_elements[valid] > 0).all()

# }}}


# {{{ flips


def get_simplex_element_flip_matrix(order, unit_nodes, permutation=None):
    """
    Generate a resampling matrix that corresponds to a
    permutation of the barycentric coordinates being applied.
    The default permutation is to swap the
    first two barycentric coordinates.

    :arg order: The order of the function space on the simplex,
        (see second argument in :fun:`modepy.simplex_best_available_basis`).
    :arg unit_nodes: A np array of unit nodes with shape *(dim, nunit_nodes)*.
    :arg permutation: Either *None*, or a tuple of shape storing a permutation:
        the *i*th barycentric coordinate gets mapped to the *permutation[i]*th
        coordinate.

    :return: A numpy array of shape *(nunit_nodes, nunit_nodes)*
        which, when its transpose is right-applied to the matrix of nodes
        (shaped *(dim, nunit_nodes)*), corresponds to the permutation being
        applied.
    """
    from modepy.tools import barycentric_to_unit, unit_to_barycentric

    bary_unit_nodes = unit_to_barycentric(unit_nodes)

    flipped_bary_unit_nodes = bary_unit_nodes.copy()
    if permutation is None:
        flipped_bary_unit_nodes[0, :] = bary_unit_nodes[1, :]
        flipped_bary_unit_nodes[1, :] = bary_unit_nodes[0, :]
    else:
        flipped_bary_unit_nodes[permutation, :] = bary_unit_nodes
    flipped_unit_nodes = barycentric_to_unit(flipped_bary_unit_nodes)

    dim = unit_nodes.shape[0]
    shape = mp.Simplex(dim)
    space = mp.PN(dim, order)
    basis = mp.basis_for_space(space, shape)
    flip_matrix = mp.resampling_matrix(
        basis.functions,
        flipped_unit_nodes,
        unit_nodes
    )

    flip_matrix[np.abs(flip_matrix) < 1e-15] = 0

    # Flipping twice should be the identity
    if permutation is None:
        assert la.norm(
                np.dot(flip_matrix, flip_matrix)
                - np.eye(len(flip_matrix))) < 1e-13

    return flip_matrix


def flip_simplex_element_group(vertices, grp, grp_flip_flags):
    from meshmode.mesh import SimplexElementGroup

    if not isinstance(grp, SimplexElementGroup):
        raise NotImplementedError("flips only supported on "
                "exclusively SimplexElementGroup-based meshes")

    # Swap the first two vertices on elements to be flipped.

    new_vertex_indices = grp.vertex_indices.copy()
    new_vertex_indices[grp_flip_flags, 0] \
            = grp.vertex_indices[grp_flip_flags, 1]
    new_vertex_indices[grp_flip_flags, 1] \
            = grp.vertex_indices[grp_flip_flags, 0]

    # Apply the flip matrix to the nodes.
    flip_matrix = get_simplex_element_flip_matrix(grp.order, grp.unit_nodes)
    new_nodes = grp.nodes.copy()
    new_nodes[:, grp_flip_flags] = np.einsum(
            "ij,dej->dei",
            flip_matrix, grp.nodes[:, grp_flip_flags])

    return SimplexElementGroup(
            grp.order, new_vertex_indices, new_nodes,
            unit_nodes=grp.unit_nodes)


def perform_flips(mesh, flip_flags, skip_tests=False):
    """
    :arg flip_flags: A :class:`numpy.ndarray` with
        :attr:`meshmode.mesh.Mesh.nelements` entries
        indicating by their Boolean value whether the element is to be
        flipped.
    """

    flip_flags = flip_flags.astype(bool)

    from meshmode.mesh import Mesh

    new_groups = []
    for grp in mesh.groups:
        grp_flip_flags = flip_flags[
                grp.element_nr_base:grp.element_nr_base+grp.nelements]

        if grp_flip_flags.any():
            new_grp = flip_simplex_element_group(
                    mesh.vertices, grp, grp_flip_flags)
        else:
            new_grp = grp.copy()

        new_groups.append(new_grp)

    return Mesh(
            mesh.vertices, new_groups, skip_tests=skip_tests,
            is_conforming=mesh.is_conforming,
            )

# }}}


# {{{ bounding box

def find_bounding_box(mesh):
    """
    :return: a tuple *(min, max)*, each consisting of a :class:`numpy.ndarray`
        indicating the minimal and maximal extent of the geometry along each axis.
    """

    return (
            np.min(mesh.vertices, axis=-1),
            np.max(mesh.vertices, axis=-1),
            )

# }}}


# {{{ merging

def merge_disjoint_meshes(meshes, skip_tests=False, single_group=False):
    if not meshes:
        raise ValueError("must pass at least one mesh")

    from pytools import is_single_valued
    if not is_single_valued(mesh.ambient_dim for mesh in meshes):
        raise ValueError("all meshes must share the same ambient dimension")

    # {{{ assemble combined vertex array

    ambient_dim = meshes[0].ambient_dim
    nvertices = sum(
            mesh.vertices.shape[-1]
            for mesh in meshes)

    vert_dtype = np.find_common_type(
            [mesh.vertices.dtype for mesh in meshes],
            [])
    vertices = np.empty(
            (ambient_dim, nvertices), vert_dtype)

    current_vert_base = 0
    vert_bases = []
    for mesh in meshes:
        mesh_nvert = mesh.vertices.shape[-1]
        vertices[:, current_vert_base:current_vert_base+mesh_nvert] = \
                mesh.vertices

        vert_bases.append(current_vert_base)
        current_vert_base += mesh_nvert

    # }}}

    # {{{ assemble new groups list

    nodal_adjacency = None
    facial_adjacency_groups = None

    if single_group:
        grp_cls = None
        order = None
        unit_nodes = None
        nodal_adjacency = None
        facial_adjacency_groups = None

        for mesh in meshes:
            if mesh._nodal_adjacency is not None:
                nodal_adjacency = False
            if mesh._facial_adjacency_groups is not None:
                facial_adjacency_groups = False

            for group in mesh.groups:
                if grp_cls is None:
                    grp_cls = type(group)
                    order = group.order
                    unit_nodes = group.unit_nodes
                else:
                    assert type(group) == grp_cls
                    assert group.order == order
                    assert np.array_equal(unit_nodes, group.unit_nodes)

        vertex_indices = np.vstack([
            group.vertex_indices + vert_base
            for mesh, vert_base in zip(meshes, vert_bases)
            for group in mesh.groups])
        nodes = np.hstack([
            group.nodes
            for mesh in meshes
            for group in mesh.groups])

        if not nodes.flags.c_contiguous:
            # hstack stopped producing C-contiguous arrays in numpy 1.14
            nodes = nodes.copy(order="C")

        new_groups = [
                grp_cls(order, vertex_indices, nodes, unit_nodes=unit_nodes)]

    else:
        new_groups = []
        nodal_adjacency = None
        facial_adjacency_groups = None

        for mesh, vert_base in zip(meshes, vert_bases):
            if mesh._nodal_adjacency is not None:
                nodal_adjacency = False
            if mesh._facial_adjacency_groups is not None:
                facial_adjacency_groups = False

            for group in mesh.groups:
                new_vertex_indices = group.vertex_indices + vert_base
                new_group = group.copy(vertex_indices=new_vertex_indices)
                new_groups.append(new_group)

    # }}}

    from meshmode.mesh import Mesh
    return Mesh(vertices, new_groups, skip_tests=skip_tests,
            nodal_adjacency=nodal_adjacency,
            facial_adjacency_groups=facial_adjacency_groups,
            is_conforming=all(
                mesh.is_conforming
                for mesh in meshes))

# }}}


# {{{ split meshes

def split_mesh_groups(mesh, element_flags, return_subgroup_mapping=False):
    """Split all the groups in *mesh* according to the values of
    *element_flags*. The element flags are expected to be integers
    defining, for each group, how the elements are to be split into
    subgroups. For example, a single-group mesh with flags::

        element_flags = [0, 0, 0, 42, 42, 42, 0, 0, 0, 41, 41, 41]

    will create three subgroups. The integer flags need not be increasing
    or contiguous and can repeat across different groups (i.e. they are
    group-local).

    :arg element_flags: a :class:`numpy.ndarray` with
        :attr:`~meshmode.mesh.Mesh.nelements` entries
        indicating how the elements in a group are to be split.

    :returns: a :class:`~meshmode.mesh.Mesh` where each group has been split
        according to flags in *element_flags*. If *return_subgroup_mapping*
        is *True*, it also returns a mapping of
        ``(group_index, subgroup) -> new_group_index``.

    """
    assert element_flags.shape == (mesh.nelements,)

    new_groups = []
    subgroup_to_group_map = {}

    for igrp, grp in enumerate(mesh.groups):
        grp_flags = element_flags[
                grp.element_nr_base:grp.element_nr_base + grp.nelements]
        unique_grp_flags = np.unique(grp_flags)

        for flag in unique_grp_flags:
            subgroup_to_group_map[igrp, flag] = len(new_groups)

            # NOTE: making copies to maintain contiguity of the arrays
            mask = grp_flags == flag
            new_groups.append(grp.copy(
                vertex_indices=grp.vertex_indices[mask, :].copy(),
                nodes=grp.nodes[:, mask, :].copy()
                ))

    from meshmode.mesh import Mesh
    mesh = Mesh(
            vertices=mesh.vertices,
            groups=new_groups,
            is_conforming=mesh.is_conforming)

    if return_subgroup_mapping:
        return mesh, subgroup_to_group_map
    else:
        return mesh

# }}}


# {{{ vertex matching

# FIXME: This tree-based approach is probably slow; see if there's a way to do
# something like this using numpy constructs
def _match_vertices(
        mesh, src_vertex_indices, tgt_vertex_indices, aff_map=None, tol=1e-12):
    if aff_map is None:
        aff_map = AffineMap()

    src_vertices = mesh.vertices[:, src_vertex_indices]
    tgt_vertices = mesh.vertices[:, tgt_vertex_indices]

    tgt_vertex_bboxes = np.stack((
        tgt_vertices - tol,
        tgt_vertices + tol))

    from pytools.spatial_btree import SpatialBinaryTreeBucket
    tree = SpatialBinaryTreeBucket(
        np.min(tgt_vertex_bboxes[0], axis=1),
        np.max(tgt_vertex_bboxes[1], axis=1))
    for ivertex in range(len(tgt_vertex_indices)):
        tree.insert(ivertex, tgt_vertex_bboxes[:, :, ivertex])

    mapped_src_vertices = aff_map(src_vertices)

    matched_tgt_vertices = np.full(len(src_vertex_indices), -1)
    for ivertex in range(len(src_vertex_indices)):
        mapped_src_vertex = mapped_src_vertices[:, ivertex]
        matches = np.array(list(tree.generate_matches(mapped_src_vertex)))
        match_bboxes = tgt_vertex_bboxes[:, :, matches]
        in_bbox = np.all(
            (mapped_src_vertex[:, np.newaxis] >= match_bboxes[0, :, :])
            & (mapped_src_vertex[:, np.newaxis] <= match_bboxes[1, :, :]),
            axis=0)
        candidate_indices = matches[in_bbox]
        if len(candidate_indices) == 0:
            continue
        displacement = (
            mapped_src_vertex.reshape(-1, 1)
            - tgt_vertices[:, candidate_indices])
        distance_sq = np.sum(displacement**2, axis=0)
        matched_tgt_vertices[ivertex] = (
            tgt_vertex_indices[candidate_indices[np.argmin(distance_sq)]])

    return matched_tgt_vertices

# }}}


# {{{ boundary face matching

@dataclass(frozen=True)
class BoundaryPairMapping:
    """
    Represents an affine mapping from one boundary to another.

    .. attribute:: from_btag

        The tag of one boundary.

    .. attribute:: to_btag

        The tag of the other boundary.

    .. attribute:: aff_map

        An :class:`meshmode.AffineMap` that maps points on boundary *from_btag* into
        points on boundary *to_btag*.
    """
    from_btag: int
    to_btag: int
    aff_map: AffineMap

    def inverted(self):
        return BoundaryPairMapping(
            self.to_btag,
            self.from_btag,
            self.aff_map.inverted())


def _get_boundary_face_ids(mesh, btag):
    from meshmode.mesh import _FaceIDs
    face_ids_per_boundary_group = []
    for igrp, fagrp_list in enumerate(mesh.facial_adjacency_groups):
        matching_bdry_grps = [
            fagrp for fagrp in fagrp_list
            if isinstance(fagrp, BoundaryAdjacencyGroup)
            and fagrp.boundary_tag == btag]
        for bdry_grp in matching_bdry_grps:
            face_ids = _FaceIDs(
                groups=np.full(len(bdry_grp.elements), igrp),
                elements=bdry_grp.elements,
                faces=bdry_grp.element_faces)
            face_ids_per_boundary_group.append(face_ids)

    from meshmode.mesh import _concatenate_face_ids
    return _concatenate_face_ids(face_ids_per_boundary_group)


def _get_face_vertex_indices(mesh, face_ids):
    max_face_vertices = max(
        len(ref_fvi)
        for grp in mesh.groups
        for ref_fvi in grp.face_vertex_indices())

    face_vertex_indices_per_group = []
    for igrp, grp in enumerate(mesh.groups):
        belongs_to_group = face_ids.groups == igrp
        faces = face_ids.faces[belongs_to_group]
        elements = face_ids.elements[belongs_to_group]
        face_vertex_indices = np.full(
            (len(faces), max_face_vertices), -1,
            dtype=mesh.vertex_id_dtype)
        for fid, ref_fvi in enumerate(grp.face_vertex_indices()):
            is_face = faces == fid
            face_vertex_indices[is_face, :len(ref_fvi)] = (
                grp.vertex_indices[elements[is_face], :][:, ref_fvi])
        face_vertex_indices_per_group.append(face_vertex_indices)

    return np.stack(face_vertex_indices_per_group)


def _match_boundary_faces(mesh, bdry_pair_mapping, tol):
    """
    Given a :class:`BoundaryPairMapping` *bdry_pair_mapping*, return the
    correspondence between faces of the two boundaries (expressed as a pair of
    :class:`meshmode.mesh._FaceIDs`).

    :arg mesh: The mesh containing the boundaries.
    :arg bdry_pair_mapping: A :class:`BoundaryPairMapping` specifying the boundaries
        whose faces are to be matched.
    :arg tol: The allowed tolerance between the transformed vertex coordinates of
        the first boundary and the vertex coordinates of the second boundary.
    :returns: A pair of :class:`meshmode.mesh._FaceIDs`, each having a number of
        entries equal to the number of faces in the boundary, that represents the
        correspondence between the two boundaries' faces. The first element in the
        pair contains faces from boundary *bdry_pair_mapping.from_btag*, and the
        second contains faces from boundary *bdry_pair_mapping.to_btag*. The order
        of the faces is unspecified.
    """
    btag_m = bdry_pair_mapping.from_btag
    btag_n = bdry_pair_mapping.to_btag

    bdry_m_face_ids = _get_boundary_face_ids(mesh, btag_m)
    bdry_n_face_ids = _get_boundary_face_ids(mesh, btag_n)

    from pytools import single_valued
    nfaces = single_valued((
        len(bdry_m_face_ids.groups),
        len(bdry_m_face_ids.elements),
        len(bdry_m_face_ids.faces),
        len(bdry_n_face_ids.groups),
        len(bdry_n_face_ids.elements),
        len(bdry_n_face_ids.faces)))

    bdry_m_face_vertex_indices = _get_face_vertex_indices(mesh, bdry_m_face_ids)
    bdry_n_face_vertex_indices = _get_face_vertex_indices(mesh, bdry_n_face_ids)

    bdry_m_vertex_indices = np.unique(bdry_m_face_vertex_indices)
    bdry_m_vertex_indices = bdry_m_vertex_indices[bdry_m_vertex_indices >= 0]
    bdry_n_vertex_indices = np.unique(bdry_n_face_vertex_indices)
    bdry_n_vertex_indices = bdry_n_vertex_indices[bdry_n_vertex_indices >= 0]

    matched_bdry_n_vertex_indices = _match_vertices(
        mesh, bdry_m_vertex_indices, bdry_n_vertex_indices,
        aff_map=bdry_pair_mapping.aff_map, tol=tol)

    unmatched_bdry_m_vertex_indices = bdry_m_vertex_indices[
        np.where(matched_bdry_n_vertex_indices < 0)[0]]
    nunmatched = len(unmatched_bdry_m_vertex_indices)
    if nunmatched > 0:
        vertices = mesh.vertices[:, unmatched_bdry_m_vertex_indices]
        mapped_vertices = bdry_pair_mapping.aff_map(vertices)
        raise RuntimeError(
            f"unable to match vertices between boundaries {btag_m} and {btag_n}.\n"
            + "Unmatched vertices (original -> mapped):\n"
            + "\n".join([
                f"{vertices[:, i]} -> {mapped_vertices[:, i]}"
                for i in range(min(nunmatched, 10))])
            + f"\n...\n({nunmatched-10} more omitted.)" if nunmatched > 10 else "")

    from meshmode.mesh import _concatenate_face_ids
    face_ids = _concatenate_face_ids([bdry_m_face_ids, bdry_n_face_ids])

    max_vertex_index = max([np.max(grp.vertex_indices) for grp in mesh.groups])
    vertex_index_map, = np.indices((max_vertex_index+1,),
        dtype=mesh.element_id_dtype)
    vertex_index_map[bdry_m_vertex_indices] = matched_bdry_n_vertex_indices

    from meshmode.mesh import _match_faces_by_vertices
    face_index_pairs = _match_faces_by_vertices(mesh.groups, face_ids,
        vertex_index_map_func=lambda vs: vertex_index_map[vs])

    assert face_index_pairs.shape[1] == nfaces

    # Since the first boundary's faces come before the second boundary's in
    # face_ids, the first boundary's faces should all be in the first row of the
    # result of _match_faces_by_vertices
    from meshmode.mesh import _FaceIDs
    return (
        _FaceIDs(
            groups=face_ids.groups[face_index_pairs[0, :]],
            elements=face_ids.elements[face_index_pairs[0, :]],
            faces=face_ids.faces[face_index_pairs[0, :]]),
        _FaceIDs(
            groups=face_ids.groups[face_index_pairs[1, :]],
            elements=face_ids.elements[face_index_pairs[1, :]],
            faces=face_ids.faces[face_index_pairs[1, :]]))

# }}}


# {{{ boundary gluing

def _complete_boundary_pairs(partial_bdry_pair_mappings_and_tols):
    partial_btag_pairs = {
        (mapping.from_btag, mapping.to_btag)
        for mapping, _ in partial_bdry_pair_mappings_and_tols}

    bdry_pair_mappings_and_tols = []
    for mapping, tol in partial_bdry_pair_mappings_and_tols:
        bdry_pair_mappings_and_tols.append((mapping, tol))
        if (mapping.to_btag, mapping.from_btag) not in partial_btag_pairs:
            bdry_pair_mappings_and_tols.append((mapping.inverted(), tol))

    return bdry_pair_mappings_and_tols


def glue_mesh_boundaries(mesh, bdry_pair_mappings_and_tols):
    """
    Create a new mesh from *mesh* in which one or more pairs of boundaries are
    "glued" together such that the boundary surfaces become part of the interior
    of the mesh. This can be used to construct, e.g., periodic boundaries.

    Corresponding boundaries' vertices must map into each other via an affine
    transformation (though the vertex ordering need not be the same). Currently
    operates only on facial adjacency; any existing nodal adjacency in *mesh* is
    ignored/invalidated.

    :arg bdry_pair_mappings_and_tols: a :class:`list` of tuples *(mapping, tol)*,
        where *mapping* is a :class:`BoundaryPairMapping` instance that specifies
        a mapping between two boundaries in *mesh* that should be glued together,
        and *tol* is the allowed tolerance between the transformed vertex
        coordinates of the first boundary and the vertex coordinates of the second
        boundary when attempting to match the two.
    """
    bdry_pair_mappings_and_tols = _complete_boundary_pairs(
        bdry_pair_mappings_and_tols)

    glued_btags = {
        btag
        for mapping, _ in bdry_pair_mappings_and_tols
        for btag in (mapping.from_btag, mapping.to_btag)}

    face_id_pairs_for_mapping = [
        _match_boundary_faces(mesh, mapping, tol)
        for mapping, tol in bdry_pair_mappings_and_tols]

    from meshmode.mesh import InteriorAdjacencyGroup, BoundaryAdjacencyGroup

    facial_adjacency_groups = []

    for igrp, old_fagrp_list in enumerate(mesh.facial_adjacency_groups):
        fagrp_list = [
            fagrp for fagrp in old_fagrp_list
            if not isinstance(fagrp, BoundaryAdjacencyGroup)
            or fagrp.boundary_tag not in glued_btags]

        for imap, (mapping, _) in enumerate(bdry_pair_mappings_and_tols):
            face_ids, neighbor_face_ids = face_id_pairs_for_mapping[imap]
            belongs_to_group = face_ids.groups == igrp
            for ineighbor_grp in range(len(mesh.groups)):
                indices, = np.where(
                    belongs_to_group
                    & (neighbor_face_ids.groups == ineighbor_grp))
                if len(indices) > 0:
                    elements = face_ids.elements[indices]
                    element_faces = face_ids.faces[indices]
                    neighbors = neighbor_face_ids.elements[indices]
                    neighbor_faces = neighbor_face_ids.faces[indices]
                    fagrp_list.append(InteriorAdjacencyGroup(
                        igroup=igrp,
                        ineighbor_group=ineighbor_grp,
                        elements=elements,
                        element_faces=element_faces,
                        neighbors=neighbors,
                        neighbor_faces=neighbor_faces,
                        aff_map=mapping.aff_map))

        facial_adjacency_groups.append(fagrp_list)

    return mesh.copy(
        nodal_adjacency=False,
        facial_adjacency_groups=facial_adjacency_groups)

# }}}


# {{{ map

def map_mesh(mesh, f):  # noqa
    """Apply the map *f* to the mesh. *f* needs to accept and return arrays of
    shape ``(ambient_dim, npoints)``."""

    if mesh._facial_adjacency_groups is not None:
        has_adj_maps = any([
            hasattr(fagrp, "aff_map")
            and (fagrp.aff_map.matrix is not None
                or fagrp.aff_map.offset is not None)
            for fagrp_list in mesh.facial_adjacency_groups
            for fagrp in fagrp_list])
        if has_adj_maps:
            raise ValueError("cannot apply a general map to a mesh that has "
                "affine mappings in its facial adjacency. If the map is affine, "
                "use affine_map instead")

    vertices = f(mesh.vertices)
    if not vertices.flags.c_contiguous:
        vertices = np.copy(vertices, order="C")

    # {{{ assemble new groups list

    new_groups = []

    for group in mesh.groups:
        mapped_nodes = f(group.nodes.reshape(mesh.ambient_dim, -1))
        if not mapped_nodes.flags.c_contiguous:
            mapped_nodes = np.copy(mapped_nodes, order="C")

        new_groups.append(group.copy(
            nodes=mapped_nodes.reshape(*group.nodes.shape)))

    # }}}

    return mesh.copy(
            vertices=vertices, groups=new_groups,
            is_conforming=mesh.is_conforming)

# }}}


# {{{ affine map

def affine_map(mesh,
        A: Optional[Union[Real, np.ndarray]] = None,    # noqa: N803
        b: Optional[Union[Real, np.ndarray]] = None):
    """Apply the affine map :math:`f(x) = A x + b` to the geometry of *mesh*."""

    if isinstance(A, Real):
        A = np.diag([A] * mesh.ambient_dim)             # noqa: N806

    if isinstance(b, Real):
        b = np.array([b] * mesh.ambient_dim)

    if A is None and b is None:
        return mesh

    if A is not None and A.shape != (mesh.ambient_dim, mesh.ambient_dim):
        raise ValueError(f"A has shape '{A.shape}' for a {mesh.ambient_dim}d mesh")

    if b is not None and b.shape != (mesh.ambient_dim,):
        raise ValueError(f"b has shape '{b.shape}' for a {mesh.ambient_dim}d mesh")

    f = AffineMap(A, b)

    vertices = f(mesh.vertices)
    if not vertices.flags.c_contiguous:
        vertices = np.copy(vertices, order="C")

    # {{{ assemble new groups list

    new_groups = []

    for group in mesh.groups:
        mapped_nodes = f(group.nodes.reshape(mesh.ambient_dim, -1))
        if not mapped_nodes.flags.c_contiguous:
            mapped_nodes = np.copy(mapped_nodes, order="C")

        new_groups.append(group.copy(
            nodes=mapped_nodes.reshape(*group.nodes.shape)))

    # }}}

    # {{{ assemble new facial adjacency groups

    if mesh._facial_adjacency_groups is not None:
        # For a facial adjacency transform T(x) = Gx + h in the original mesh,
        # its corresponding transform in the new mesh will be (T')(x) = G'x + h',
        # where:
        # G' = G
        # h' = Ah + (I - G)b
        def compute_new_map(old_map):
            if old_map.matrix is not None:
                matrix = old_map.matrix.copy()
            else:
                matrix = None
            if old_map.offset is not None:
                if A is not None:
                    offset = A @ old_map.offset
                else:
                    offset = old_map.offset.copy()
                if matrix is not None and b is not None:
                    offset += b - matrix @ b
            else:
                offset = None
            return AffineMap(matrix, offset)

        facial_adjacency_groups = []
        for old_fagrp_list in mesh.facial_adjacency_groups:
            fagrp_list = []
            for old_fagrp in old_fagrp_list:
                if hasattr(old_fagrp, "aff_map"):
                    aff_map = compute_new_map(old_fagrp.aff_map)
                    fagrp_list.append(
                        old_fagrp.copy(
                            aff_map=aff_map))
                else:
                    fagrp_list.append(old_fagrp.copy())
            facial_adjacency_groups.append(fagrp_list)

    else:
        facial_adjacency_groups = None

    # }}}

    return mesh.copy(
            vertices=vertices, groups=new_groups,
            facial_adjacency_groups=facial_adjacency_groups,
            is_conforming=mesh.is_conforming)


def _get_rotation_matrix_from_angle_and_axis(theta, axis):
    # https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    ux, uy, uz = axis / np.linalg.norm(axis, ord=2)

    return np.array([[
        cos_t + ux**2 * (1 - cos_t),
        ux * uy * (1 - cos_t) - uz * sin_t,
        ux * uz * (1 - cos_t) + uy * sin_t
        ], [
        uy * ux * (1 - cos_t) + uz * sin_t,
        cos_t + uy**2 * (1 - cos_t),
        uy * uz * (1 - cos_t) - ux * sin_t
        ], [
        uz * ux * (1 - cos_t) - uy * sin_t,
        uz * uy * (1 - cos_t) + ux * sin_t,
        cos_t + uz**2 * (1 - cos_t)
        ]])


def rotate_mesh_around_axis(mesh, *,
        theta: Real,
        axis: Optional[np.ndarray] = None):
    """Rotate the mesh by *theta* radians around the axis *axis*.

    :arg axis: a (not necessarily unit) vector. By default, the rotation is
        performed around the :math:`z` axis.
    """
    if mesh.ambient_dim == 1:
        return mesh
    elif mesh.ambient_dim == 2:
        axis = None
    elif mesh.ambient_dim == 3:
        pass
    else:
        raise ValueError(f"unsupported mesh dimension: {mesh.ambient_dim}")

    if axis is None:
        axis = np.array([0, 0, 1])

    mat = _get_rotation_matrix_from_angle_and_axis(theta, axis)
    return affine_map(mesh, A=mat[:mesh.ambient_dim, :mesh.ambient_dim])

# }}}

# vim: foldmethod=marker
