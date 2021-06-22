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

from dataclasses import dataclass
from functools import reduce
from numbers import Real
from typing import Optional, Union

import numpy as np
import numpy.linalg as la

import modepy as mp


__doc__ = """
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

    :returns: A list of :class:`dict`\ s, `local_to_local_adjacency_groups`, that
        maps pairs of partitioned group indices to
        :clas:`~meshmode.mesh.FacialAdjacencyGroup` instances if they have
        local-to-local adjacency.
    """
    local_to_local_adjacency_groups = [{} for _ in part_mesh_groups]

    for igrp, facial_adj_dict in enumerate(mesh.facial_adjacency_groups):
        i_part_grp = global_group_to_part_group[igrp]
        if i_part_grp is None:
            continue

        for jgrp, facial_adj in facial_adj_dict.items():
            if jgrp is None:
                continue

            j_part_grp = global_group_to_part_group[jgrp]
            if j_part_grp is None:
                continue

            elem_base_i = mesh.groups[igrp].element_nr_base
            elem_base_j = mesh.groups[jgrp].element_nr_base

            elements_are_local = global_elem_to_part_elem[facial_adj.elements
                        + elem_base_i] >= 0
            neighbors_are_local = global_elem_to_part_elem[facial_adj.neighbors
                        + elem_base_j] >= 0

            adj_indices = np.where(elements_are_local & neighbors_are_local)[0]

            if len(adj_indices) > 0:
                part_elem_base_i = part_mesh_group_elem_base[i_part_grp]
                part_elem_base_j = part_mesh_group_elem_base[j_part_grp]

                elements = global_elem_to_part_elem[facial_adj.elements[
                            adj_indices] + elem_base_i] - part_elem_base_i
                element_faces = facial_adj.element_faces[adj_indices]
                neighbors = global_elem_to_part_elem[facial_adj.neighbors[
                            adj_indices] + elem_base_j] - part_elem_base_j
                neighbor_faces = facial_adj.neighbor_faces[adj_indices]
                mats = facial_adj.aff_transform_mats[adj_indices, :, :]
                vecs = facial_adj.aff_transform_vecs[adj_indices, :]

                from meshmode.mesh import FacialAdjacencyGroup
                local_to_local_adjacency_groups[i_part_grp][j_part_grp] =\
                            FacialAdjacencyGroup(igroup=i_part_grp,
                                        ineighbor_group=j_part_grp,
                                        elements=elements,
                                        element_faces=element_faces,
                                        neighbors=neighbors,
                                        neighbor_faces=neighbor_faces,
                                        aff_transform_mats=mats,
                                        aff_transform_vecs=vecs)

    return local_to_local_adjacency_groups


@dataclass
class _NonLocalAdjacencyData:
    """
    Data structure for intermediate storage of non-local adjacency data. Each
    attribute is a :class:`numpy.ndarray` and contains an entry for every local
    element face that is shared with a remote element.

    .. attribute:: elements

        The group-relative element index.

    .. attribute:: element_faces

        The index of the shared face inside the local element.

    .. attribute:: neighbor_parts

       The partition containing the remote element.

    .. attribute:: global_neighbors

       The global element index of the remote element.

    .. attribute:: neighbor_faces

       The index of the shared face inside the remote element.

    .. attribute:: aff_transform_mats

       The matrix part of the affine mapping from local to remote element.

    .. attribute:: aff_transform_vecs

       The vector part of the affine mapping from local to remote element.
    """
    elements: np.ndarray
    element_faces: np.ndarray
    neighbor_parts: np.ndarray
    global_neighbors: np.ndarray
    neighbor_faces: np.ndarray
    aff_transform_mats: np.ndarray
    aff_transform_vecs: np.ndarray


def _collect_nonlocal_adjacency_data(mesh, part_per_elem, global_elem_to_part_elem,
            part_mesh_groups, global_group_to_part_group,
            part_mesh_group_elem_base):
    """
    Collect non-local adjacency data for the partitioned mesh, and store it in an
    intermediate data structure to use when constructing inter-partition adjacency
    groups.

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

    :returns: A list of :class:`_NonLocalAdjacencyData` instances, one for each
        group in *part_mesh_groups*, containing non-local adjacency data.
    """
    nonlocal_adj_data = [None for _ in part_mesh_groups]

    for igrp, facial_adj_dict in enumerate(mesh.facial_adjacency_groups):
        i_part_grp = global_group_to_part_group[igrp]
        if i_part_grp is None:
            continue

        pairwise_adj = []

        for jgrp, facial_adj in facial_adj_dict.items():
            if jgrp is None:
                continue

            elem_base_i = mesh.groups[igrp].element_nr_base
            elem_base_j = mesh.groups[jgrp].element_nr_base

            elements_are_local = global_elem_to_part_elem[facial_adj.elements
                        + elem_base_i] >= 0
            neighbors_are_nonlocal = global_elem_to_part_elem[facial_adj.neighbors
                        + elem_base_j] < 0
            adj_indices = np.where(elements_are_local & neighbors_are_nonlocal)[0]

            if len(adj_indices) > 0:
                part_elem_base_i = part_mesh_group_elem_base[i_part_grp]

                elements = global_elem_to_part_elem[facial_adj.elements[
                            adj_indices] + elem_base_i] - part_elem_base_i
                element_faces = facial_adj.element_faces[adj_indices]
                global_neighbors = facial_adj.neighbors[adj_indices] + elem_base_j
                neighbor_parts = part_per_elem[global_neighbors]
                neighbor_faces = facial_adj.neighbor_faces[adj_indices]
                mats = facial_adj.aff_transform_mats[adj_indices]
                vecs = facial_adj.aff_transform_vecs[adj_indices]

                pairwise_adj.append(
                    _NonLocalAdjacencyData(
                        elements, element_faces, neighbor_parts, global_neighbors,
                        neighbor_faces, mats, vecs))

        if pairwise_adj:
            nonlocal_adj_data[i_part_grp] = _NonLocalAdjacencyData(
                np.concatenate([adj.elements for adj in pairwise_adj]),
                np.concatenate([adj.element_faces for adj in pairwise_adj]),
                np.concatenate([adj.neighbor_parts for adj in pairwise_adj]),
                np.concatenate([adj.global_neighbors for adj in pairwise_adj]),
                np.concatenate([adj.neighbor_faces for adj in pairwise_adj]),
                np.concatenate([adj.aff_transform_mats for adj in pairwise_adj]),
                np.concatenate([adj.aff_transform_vecs for adj in pairwise_adj]))

    return nonlocal_adj_data


@dataclass
class _BoundaryData:
    """
    Data structure for intermediate storage of boundary data. Each attribute is a
    :class:`numpy.ndarray` and contains an entry for every local element face that
    is a boundary.

    .. attribute:: elements

        The group-relative element index.

    .. attribute:: element_faces

        The index of the shared face inside the local element.

    .. attribute:: neighbors

        Boundary tag data.

    """
    elements: np.ndarray
    element_faces: np.ndarray
    neighbors: np.ndarray


def _collect_bdry_data(mesh, global_elem_to_part_elem, part_mesh_groups,
            global_group_to_part_group, part_mesh_group_elem_base):
    """
    Collect boundary data for partitioned mesh, and store it in an intermediate
    data structure to use when constructing inter-partition adjacency groups.

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

    :returns: A list of :class:`_BoundaryData` instances, one for each
        group in *part_mesh_groups*, containing boundary data.
    """
    bdry_data = [None for _ in part_mesh_groups]

    for igrp, facial_adj_dict in enumerate(mesh.facial_adjacency_groups):
        i_part_grp = global_group_to_part_group[igrp]
        if i_part_grp is None:
            continue

        facial_adj = facial_adj_dict[None]

        elem_base = mesh.groups[igrp].element_nr_base

        adj_indices = np.where(global_elem_to_part_elem[facial_adj.elements
                    + elem_base] >= 0)[0]

        if len(adj_indices) > 0:
            part_elem_base = part_mesh_group_elem_base[i_part_grp]

            elements = global_elem_to_part_elem[facial_adj.elements[adj_indices]
                        + elem_base] - part_elem_base
            element_faces = facial_adj.element_faces[adj_indices]
            neighbors = facial_adj.neighbors[adj_indices]

            bdry_data[i_part_grp] = _BoundaryData(elements, element_faces, neighbors)

    return bdry_data


def _create_inter_partition_adjacency_groups(mesh, part_per_element,
            part_mesh_groups, all_neighbor_parts, nonlocal_adj_data, bdry_data,
            boundary_tag_bit):
    """
    Combine non-local adjacency data and boundary data into inter-partition
    adjacency groups.

    :arg mesh: A :class:`~meshmode.mesh.Mesh` representing the unpartitioned mesh.
    :arg part_per_element: A :class:`numpy.ndarray` mapping element indices to
        partition numbers.
    :arg part_mesh_groups: An array of `~meshmode.mesh.ElementGroup` instances
        representing the partitioned mesh groups.
    :arg all_neighbor_parts: A `set` containing all partition numbers that neighbor
        the current one.
    :arg nonlocal_adj_data: A list of :class:`_NonLocalAdjacencyData` instances, one
        for each group in *part_mesh_groups*, containing non-local adjacency data.
    :arg bdry_data: A list of :class:`_BoundaryData` instances, one for each group
        in *part_mesh_groups*, containing boundary data.

    :returns: A list of `~meshmode.mesh.InterPartitionAdjacencyGroup` instances, one
        for each group in *part_mesh_groups*, containing the aggregated non-local
        adjacency and boundary information from *nonlocal_adj_data* and *bdry_data*.
    """
    global_elem_to_neighbor_elem = _compute_global_elem_to_part_elem(
                part_per_element, all_neighbor_parts, mesh.element_id_dtype)

    inter_partition_adj_groups = []

    for i_part_grp in range(len(part_mesh_groups)):
        nl = nonlocal_adj_data[i_part_grp]
        bdry = bdry_data[i_part_grp]
        if nl is None and bdry is None:
            # Neither non-local adjacency nor boundary
            elements = np.empty(0, dtype=mesh.element_id_dtype)
            element_faces = np.empty(0, dtype=mesh.face_id_dtype)
            neighbor_parts = np.empty(0, dtype=np.int32)
            neighbors = np.empty(0, dtype=mesh.element_id_dtype)
            neighbor_elements = np.empty(0, dtype=mesh.element_id_dtype)
            neighbor_faces = np.empty(0, dtype=mesh.face_id_dtype)
            mats = np.empty((0, mesh.ambient_dim, mesh.ambient_dim),
                dtype=np.float64)
            vecs = np.empty((0, mesh.ambient_dim), dtype=np.float64)

        elif bdry is None:
            # Non-local adjacency only
            elements = nl.elements
            element_faces = nl.element_faces
            neighbor_parts = nl.neighbor_parts
            from meshmode.mesh import BTAG_REALLY_ALL, BTAG_PARTITION
            flags = np.full_like(elements, boundary_tag_bit(BTAG_REALLY_ALL))
            for i_neighbor_part in all_neighbor_parts:
                flags[neighbor_parts == i_neighbor_part] |= (
                    boundary_tag_bit(BTAG_PARTITION(i_neighbor_part)))
            neighbors = -flags
            neighbor_elements = global_elem_to_neighbor_elem[nl.global_neighbors]
            neighbor_faces = nl.neighbor_faces
            mats = nl.aff_transform_mats
            vecs = nl.aff_transform_vecs

        elif nl is None:
            # Boundary only
            nfaces = len(bdry.elements)
            elements = bdry.elements
            element_faces = bdry.element_faces
            neighbor_parts = np.full(nfaces, -1, dtype=np.int32)
            neighbors = bdry.neighbors
            neighbor_elements = np.full(nfaces, -1, dtype=mesh.element_id_dtype)
            neighbor_faces = np.zeros(nfaces, dtype=mesh.face_id_dtype)
            from meshmode.mesh import _make_affine_identity_transforms
            mats, vecs = _make_affine_identity_transforms(mesh.ambient_dim, nfaces)

        else:
            # Both; need to merge together
            nnonlocal = len(nl.elements)
            nbdry = len(bdry.elements)
            nfaces = nnonlocal + nbdry
            elements = np.empty(nfaces, dtype=mesh.element_id_dtype)
            element_faces = np.empty(nfaces, dtype=mesh.face_id_dtype)
            neighbor_parts = np.empty(nfaces, dtype=np.int32)
            neighbors = np.empty(nfaces, dtype=mesh.element_id_dtype)
            neighbor_elements = np.empty(nfaces, dtype=mesh.element_id_dtype)
            neighbor_faces = np.empty(nfaces, dtype=mesh.face_id_dtype)
            mats = np.empty((nfaces, mesh.ambient_dim, mesh.ambient_dim),
                dtype=np.float64)
            vecs = np.empty((nfaces, mesh.ambient_dim), dtype=np.float64)

            # Combine lists of elements/faces and sort to assist in merging
            combined_elements = np.concatenate((nl.elements, bdry.elements))
            combined_element_faces = np.concatenate((nl.element_faces,
                        bdry.element_faces))
            perm = np.lexsort([combined_element_faces, combined_elements])

            # Merge non-local part
            nonlocal_indices = np.where(perm < nnonlocal)[0]
            elements[nonlocal_indices] = nl.elements
            element_faces[nonlocal_indices] = nl.element_faces
            neighbor_parts[nonlocal_indices] = nl.neighbor_parts
            for imerged in nonlocal_indices:
                i_neighbor_part = neighbor_parts[imerged]
                from meshmode.mesh import BTAG_REALLY_ALL, BTAG_PARTITION
                neighbors[imerged] = -(
                            boundary_tag_bit(BTAG_REALLY_ALL)
                            | boundary_tag_bit(BTAG_PARTITION(i_neighbor_part)))
            neighbor_elements[nonlocal_indices] = global_elem_to_neighbor_elem[
                        nl.global_neighbors]
            neighbor_faces[nonlocal_indices] = nl.neighbor_faces
            mats[nonlocal_indices] = nl.aff_transform_mats
            vecs[nonlocal_indices] = nl.aff_transform_vecs

            # Merge boundary part
            bdry_indices = np.where(perm >= nnonlocal)[0]
            elements[bdry_indices] = bdry.elements
            element_faces[bdry_indices] = bdry.element_faces
            neighbors[bdry_indices] = bdry.neighbors
            neighbor_parts[bdry_indices] = -1
            neighbor_elements[bdry_indices] = -1
            neighbor_faces[bdry_indices] = 0
            from meshmode.mesh import _make_affine_identity_transforms
            bdry_mats, bdry_vecs = _make_affine_identity_transforms(
                mesh.ambient_dim, nbdry)
            mats[bdry_indices, :, :] = bdry_mats
            vecs[bdry_indices, :] = bdry_vecs

        from meshmode.mesh import InterPartitionAdjacencyGroup
        inter_partition_adj_groups.append(InterPartitionAdjacencyGroup(
                    igroup=i_part_grp, ineighbor_group=None, elements=elements,
                    element_faces=element_faces, neighbors=neighbors,
                    neighbor_partitions=neighbor_parts,
                    partition_neighbors=neighbor_elements,
                    neighbor_faces=neighbor_faces,
                    aff_transform_mats=mats,
                    aff_transform_vecs=vecs))

    return inter_partition_adj_groups


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
    queried_elems = np.where(np.array(part_per_element) == part_num)[0]

    global_elem_to_part_elem = _compute_global_elem_to_part_elem(part_per_element,
                {part_num}, mesh.element_id_dtype)

    # Create new mesh groups that mimick the original mesh's groups but only contain
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

    local_to_local_adj_groups = _create_local_to_local_adjacency_groups(mesh,
                global_elem_to_part_elem, part_mesh_groups,
                global_group_to_part_group, part_mesh_group_elem_base)

    nonlocal_adj_data = _collect_nonlocal_adjacency_data(mesh,
                np.array(part_per_element), global_elem_to_part_elem,
                part_mesh_groups, global_group_to_part_group,
                part_mesh_group_elem_base)

    bdry_data = _collect_bdry_data(mesh, global_elem_to_part_elem, part_mesh_groups,
                global_group_to_part_group, part_mesh_group_elem_base)

    group_neighbor_parts = [adj.neighbor_parts for adj in nonlocal_adj_data if
                adj is not None]
    all_neighbor_parts = set(np.unique(np.concatenate(group_neighbor_parts))) if\
                group_neighbor_parts else set()

    boundary_tags = mesh.boundary_tags[:]
    btag_to_index = {tag: i for i, tag in enumerate(boundary_tags)}

    def boundary_tag_bit(boundary_tag):
        from meshmode.mesh import _boundary_tag_bit
        return _boundary_tag_bit(boundary_tags, btag_to_index, boundary_tag)

    from meshmode.mesh import BTAG_PARTITION
    for i_neighbor_part in all_neighbor_parts:
        part_tag = BTAG_PARTITION(i_neighbor_part)
        boundary_tags.append(part_tag)
        btag_to_index[part_tag] = len(boundary_tags)-1

    inter_partition_adj_groups = _create_inter_partition_adjacency_groups(mesh,
                part_per_element, part_mesh_groups, all_neighbor_parts,
                nonlocal_adj_data, bdry_data, boundary_tag_bit)

    # Combine local and inter-partition/boundary adjacency groups
    part_facial_adj_groups = local_to_local_adj_groups
    for igrp, facial_adj in enumerate(inter_partition_adj_groups):
        part_facial_adj_groups[igrp][None] = facial_adj

    from meshmode.mesh import Mesh
    part_mesh = Mesh(
            part_vertices,
            part_mesh_groups,
            facial_adjacency_groups=part_facial_adj_groups,
            boundary_tags=boundary_tags,
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

    :param order: The order of the function space on the simplex,
                 (see second argument in
                  :fun:`modepy.simplex_best_available_basis`)
    :param unit_nodes: A np array of unit nodes with shape
                       *(dim, nunit_nodes)*
    :param permutation: Either *None*, or a tuple of shape
                        storing a permutation:
                        the *i*th barycentric coordinate gets mapped to
                        the *permutation[i]*th coordinate.

    :return: A numpy array of shape *(nunit_nodes, nunit_nodes)*
             which, when its transpose is right-applied
             to the matrix of nodes (shaped *(dim, nunit_nodes)*),
             corresponds to the permutation being applied
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


# {{{ mesh boundary gluing

def _get_bdry_face_ids(mesh, btag):
    btag_bit = mesh.boundary_tag_bit(btag)

    from meshmode.mesh import _FaceIDs
    face_ids_per_group = []
    for igrp, fagrp_map in enumerate(mesh.facial_adjacency_groups):
        bdry_grp = fagrp_map.get(None)
        if bdry_grp is not None:
            bdry_face_indices, = np.where((-bdry_grp.neighbors & btag_bit) != 0)
            grp_face_ids = _FaceIDs(
                groups=np.zeros(len(bdry_face_indices), dtype=int) + igrp,
                elements=bdry_grp.elements[bdry_face_indices],
                faces=bdry_grp.element_faces[bdry_face_indices])
        else:
            grp_face_ids = _FaceIDs(
                groups=np.empty(0, dtype=int),
                elements=np.empty(0, dtype=mesh.element_id_dtype),
                faces=np.empty(0, dtype=mesh.face_id_dtype))
        face_ids_per_group.append(grp_face_ids)

    from meshmode.mesh import _concatenate_face_ids
    return _concatenate_face_ids(face_ids_per_group)


def _get_face_vertex_indices(mesh, face_ids):
    max_face_vertices = max(
        len(ref_fvi)
        for grp in mesh.groups
        for ref_fvi in grp.face_vertex_indices())

    face_vertex_indices_per_group = []
    for igrp, grp in enumerate(mesh.groups):
        is_grp = face_ids.groups == igrp
        face_vertex_indices = np.full(
            (np.count_nonzero(is_grp), max_face_vertices), -1,
            dtype=mesh.vertex_id_dtype)
        for fid, ref_fvi in enumerate(grp.face_vertex_indices()):
            is_face = is_grp & (face_ids.faces == fid)
            is_face_grp = face_ids.faces[is_grp] == fid
            face_vertex_indices[is_face_grp, :len(ref_fvi)] = (
                grp.vertex_indices[face_ids.elements[is_face], :][:, ref_fvi])
        face_vertex_indices_per_group.append(face_vertex_indices)

    return np.stack(face_vertex_indices_per_group)


def _compute_face_indices_from_mask(mask):
    # Order by face, then by element
    indices = np.cumsum(mask).reshape(mask.shape) - 1
    indices[~mask] = -1
    return indices


def _match_boundary_faces(mesh, glued_boundary_mappings, tol):
    face_id_pairs_for_mapping = []

    for btag_m, btag_n, aff_transform in glued_boundary_mappings:
        bdry_m_face_ids = _get_bdry_face_ids(mesh, btag_m)
        bdry_n_face_ids = _get_bdry_face_ids(mesh, btag_n)

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

        nvertices = single_valued((
            len(bdry_m_vertex_indices),
            len(bdry_n_vertex_indices)))

        bdry_m_vertices = mesh.vertices[:, bdry_m_vertex_indices]
        bdry_n_vertices = mesh.vertices[:, bdry_n_vertex_indices]

        # FIXME: This approach is probably slow; see if there's a way to do
        # something like this using numpy constructs
        bdry_n_bbox = (
            np.min(bdry_n_vertices, axis=1),
            np.max(bdry_n_vertices, axis=1))
        from pytools.spatial_btree import SpatialBinaryTreeBucket
        tree = SpatialBinaryTreeBucket(bdry_n_bbox[0], bdry_n_bbox[1])
        bdry_n_vertex_bboxes = np.stack((
            bdry_n_vertices - tol,
            bdry_n_vertices + tol))
        for ivertex in range(nvertices):
            tree.insert(ivertex, bdry_n_vertex_bboxes[:, :, ivertex])

        mat, vec = aff_transform
        mapped_bdry_m_vertices = mat @ bdry_m_vertices + vec.reshape(-1, 1)

        equivalent_vertices = np.empty((2, nvertices), dtype=mesh.element_id_dtype)
        for ivertex in range(nvertices):
            bdry_m_index = bdry_m_vertex_indices[ivertex]
            mapped_bdry_m_vertex = mapped_bdry_m_vertices[:, ivertex]
            matches = np.array(list(tree.generate_matches(mapped_bdry_m_vertex)))
            match_bboxes = bdry_n_vertex_bboxes[:, :, matches]
            in_bbox = np.all(
                (mapped_bdry_m_vertex[:, np.newaxis] >= match_bboxes[0, :, :])
                & (mapped_bdry_m_vertex[:, np.newaxis] <= match_bboxes[1, :, :]),
                axis=0)
            candidate_indices = matches[in_bbox]
            if len(candidate_indices) == 0:
                raise RuntimeError("failed to find a matching vertex for vertex "
                    f"{bdry_m_index} at {mapped_bdry_m_vertex}")
            displacement = (
                mapped_bdry_m_vertex.reshape(-1, 1)
                - bdry_n_vertices[:, candidate_indices])
            distance_sq = np.sum(displacement**2, axis=0)
            bdry_n_index = bdry_n_vertex_indices[
                candidate_indices[np.argmin(distance_sq)]]
            equivalent_vertices[:, ivertex] = [bdry_m_index, bdry_n_index]

        from meshmode.mesh import _concatenate_face_ids
        face_ids = _concatenate_face_ids([bdry_m_face_ids, bdry_n_face_ids])

        max_vertex_index = max([np.max(grp.vertex_indices) for grp in mesh.groups])
        vertex_index_map, = np.indices((max_vertex_index+1,),
            dtype=mesh.element_id_dtype)
        vertex_index_map[equivalent_vertices[0, :]] = equivalent_vertices[1, :]

        from meshmode.mesh import _match_faces_by_vertices
        face_index_pairs = _match_faces_by_vertices(mesh.groups, face_ids,
            vertex_index_map_func=lambda vs: vertex_index_map[vs])

        assert face_index_pairs.shape[1] == nfaces

        from meshmode.mesh import _FaceIDs
        order = np.argsort(face_index_pairs[0, :])
        face_id_pairs_for_mapping.append((
            _FaceIDs(
                groups=face_ids.groups[face_index_pairs[0, order]],
                elements=face_ids.elements[face_index_pairs[0, order]],
                faces=face_ids.faces[face_index_pairs[0, order]]),
            _FaceIDs(
                groups=face_ids.groups[face_index_pairs[1, order]],
                elements=face_ids.elements[face_index_pairs[1, order]],
                faces=face_ids.faces[face_index_pairs[1, order]])))

    return face_id_pairs_for_mapping


def _translate_boundary_pair_adjacency_into_group_pair_adjacency(mesh,
        glued_boundary_mappings, face_id_pairs_for_mapping):
    face_id_pairs_for_group_pair = []
    mapping_indices_for_group_pair = []

    for igrp, grp in enumerate(mesh.groups):
        face_id_pairs_grp_map = {}
        mapping_indices_grp_map = {}

        connected_groups = np.unique(np.concatenate([
            face_id_pairs[1].groups[face_id_pairs[0].groups == igrp]
            for face_id_pairs in face_id_pairs_for_mapping]))

        for ineighbor_grp in connected_groups:
            adj_data_for_mapping = []
            face_has_neighbor = np.full((grp.nfaces, grp.nelements), False)

            for imap in range(len(glued_boundary_mappings)):
                mapping_face_id_pairs = face_id_pairs_for_mapping[imap]
                is_grp_pair = (
                    (mapping_face_id_pairs[0].groups == igrp)
                    & (mapping_face_id_pairs[1].groups == ineighbor_grp))
                elements = mapping_face_id_pairs[0].elements[is_grp_pair]
                element_faces = mapping_face_id_pairs[0].faces[is_grp_pair]
                neighbors = mapping_face_id_pairs[1].elements[is_grp_pair]
                neighbor_faces = mapping_face_id_pairs[1].faces[is_grp_pair]
                adj_data_for_mapping.append(
                    (elements, element_faces, neighbors, neighbor_faces))
                face_has_neighbor[element_faces, elements] = True

            translated_indices = _compute_face_indices_from_mask(face_has_neighbor)
            nfaces = np.max(translated_indices) + 1

            from meshmode.mesh import _FaceIDs
            face_id_pairs = (
                _FaceIDs(
                    groups=np.zeros(nfaces, dtype=int) + igrp,
                    elements=np.empty(nfaces, dtype=mesh.element_id_dtype),
                    faces=np.empty(nfaces, dtype=mesh.face_id_dtype)),
                _FaceIDs(
                    groups=np.zeros(nfaces, dtype=int) + ineighbor_grp,
                    elements=np.empty(nfaces, dtype=mesh.element_id_dtype),
                    faces=np.empty(nfaces, dtype=mesh.face_id_dtype)))

            mapping_indices = np.empty(nfaces, dtype=int)

            for imap, (elements, element_faces, neighbors, neighbor_faces) in (
                    enumerate(adj_data_for_mapping)):
                indices = translated_indices[element_faces, elements]
                face_id_pairs[0].elements[indices] = elements
                face_id_pairs[0].faces[indices] = element_faces
                face_id_pairs[1].elements[indices] = neighbors
                face_id_pairs[1].faces[indices] = neighbor_faces
                mapping_indices[indices] = imap

            face_id_pairs_grp_map[ineighbor_grp] = face_id_pairs
            mapping_indices_grp_map[ineighbor_grp] = mapping_indices

        face_id_pairs_for_group_pair.append(face_id_pairs_grp_map)
        mapping_indices_for_group_pair.append(mapping_indices_grp_map)

    return face_id_pairs_for_group_pair, mapping_indices_for_group_pair


def _construct_glued_mesh(mesh, glued_boundary_mappings,
        face_id_pairs_for_group_pair, mapping_indices_for_group_pair):
    glued_btags = (
        set(btag_m for btag_m, _, _ in glued_boundary_mappings)
        | set(btag_n for _, btag_n, _ in glued_boundary_mappings))

    boundary_tags = [
        btag for btag in mesh.boundary_tags
        if btag not in glued_btags]

    btag_to_index = {btag: i for i, btag in enumerate(boundary_tags)}

    def boundary_tag_bit(btag):
        from meshmode.mesh import _boundary_tag_bit
        return _boundary_tag_bit(boundary_tags, btag_to_index, btag)

    mats_for_mapping = np.stack(
        mat for _, _, (mat, _) in glued_boundary_mappings)
    vecs_for_mapping = np.stack(
        vec for _, _, (_, vec) in glued_boundary_mappings)

    from meshmode.mesh import FacialAdjacencyGroup

    facial_adjacency_groups = []

    for igrp, grp in enumerate(mesh.groups):
        fagrp_map = {}

        old_fagrp_map = mesh.facial_adjacency_groups[igrp]
        face_id_pairs_grp_map = face_id_pairs_for_group_pair[igrp]
        mapping_indices_grp_map = mapping_indices_for_group_pair[igrp]

        connected_groups = (
            set(
                ineighbor_grp for ineighbor_grp in old_fagrp_map.keys()
                if ineighbor_grp is not None)
            | set(ineighbor_grp for ineighbor_grp in face_id_pairs_grp_map.keys()))

        for ineighbor_grp in connected_groups:
            face_has_neighbor = np.full((grp.nfaces, grp.nelements), False)
            old_adj = old_fagrp_map.get(ineighbor_grp)
            if old_adj is not None:
                face_has_neighbor[old_adj.element_faces, old_adj.elements] = True
            grp_pair_face_ids = face_id_pairs_grp_map.get(ineighbor_grp)
            if grp_pair_face_ids is not None:
                face_ids = grp_pair_face_ids[0]
                face_has_neighbor[face_ids.faces, face_ids.elements] = True

            merged_indices = _compute_face_indices_from_mask(face_has_neighbor)
            nfaces = np.max(merged_indices) + 1

            elements = np.empty(nfaces, dtype=mesh.element_id_dtype)
            element_faces = np.empty(nfaces, dtype=mesh.face_id_dtype)
            neighbors = np.empty(nfaces, dtype=mesh.element_id_dtype)
            neighbor_faces = np.empty(nfaces, dtype=mesh.face_id_dtype)
            mats = np.empty((nfaces, mesh.ambient_dim, mesh.ambient_dim),
                dtype=np.float64)
            vecs = np.empty((nfaces, mesh.ambient_dim), dtype=np.float64)

            if old_adj is not None:
                indices = merged_indices[old_adj.element_faces, old_adj.elements]
                elements[indices] = old_adj.elements
                element_faces[indices] = old_adj.element_faces
                neighbors[indices] = old_adj.neighbors
                neighbor_faces[indices] = old_adj.neighbor_faces
                mats[indices, :, :] = old_adj.aff_transform_mats
                vecs[indices, :] = old_adj.aff_transform_vecs

            if grp_pair_face_ids is not None:
                face_ids = grp_pair_face_ids[0]
                neighbor_face_ids = grp_pair_face_ids[1]
                indices = merged_indices[face_ids.faces, face_ids.elements]
                elements[indices] = face_ids.elements
                element_faces[indices] = face_ids.faces
                neighbors[indices] = neighbor_face_ids.elements
                neighbor_faces[indices] = neighbor_face_ids.faces
                mapping_indices = mapping_indices_grp_map[ineighbor_grp]
                mats[indices, :, :] = mats_for_mapping[mapping_indices, :, :]
                vecs[indices, :] = vecs_for_mapping[mapping_indices, :]

            fagrp_map[ineighbor_grp] = FacialAdjacencyGroup(
                igroup=igrp,
                ineighbor_group=ineighbor_grp,
                elements=elements,
                element_faces=element_faces,
                neighbors=neighbors,
                neighbor_faces=neighbor_faces,
                aff_transform_mats=mats,
                aff_transform_vecs=vecs)

        is_bdry = np.full((grp.nfaces, grp.nelements), True)
        for ineighbor_grp in connected_groups:
            adj = fagrp_map.get(ineighbor_grp)
            is_bdry[adj.element_faces, adj.elements] = False

        if np.any(is_bdry):
            old_bdry_grp = old_fagrp_map[None]
            indices, = np.where(is_bdry[
                old_bdry_grp.element_faces,
                old_bdry_grp.elements])

            old_flags = -old_bdry_grp.neighbors[indices]
            flags = old_flags.copy()
            for btag in mesh.boundary_tags:
                old_btag_bit = mesh.boundary_tag_bit(btag)
                flags &= ~old_btag_bit
            for btag in boundary_tags:
                old_btag_bit = mesh.boundary_tag_bit(btag)
                btag_bit = boundary_tag_bit(btag)
                flags[(old_flags & old_btag_bit) != 0] |= btag_bit

            fagrp_map[None] = FacialAdjacencyGroup(
                igroup=igrp,
                ineighbor_group=None,
                elements=old_bdry_grp.elements[indices],
                element_faces=old_bdry_grp.element_faces[indices],
                neighbors=-flags,
                neighbor_faces=old_bdry_grp.neighbor_faces[indices],
                aff_transform_mats=old_bdry_grp.aff_transform_mats[indices, :, :],
                aff_transform_vecs=old_bdry_grp.aff_transform_vecs[indices, :])

        facial_adjacency_groups.append(fagrp_map)

    return mesh.copy(
        boundary_tags=boundary_tags,
        facial_adjacency_groups=facial_adjacency_groups)


def glue_mesh_boundaries(mesh, glued_boundary_mappings, tol=1e-12):
    """
    Create a new mesh from *mesh* in which one or more pairs of boundaries are
    "glued" together such that the boundary surfaces become part of the interior
    of the mesh. This can be used to construct, e.g., periodic boundaries.

    Corresponding boundaries' vertices must map into each other via an affine
    transformation (though the vertex ordering need not be the same).

    :arg glued_boundary_mappings: a :class:`list` of tuples
        (btag_m, btag_n, aff_transform) which each specify a mapping between two
        boundaries in *mesh* that should be glued together. aff_transform is a tuple
        (mat, vec) that represents the affine mapping from the vertices of boundary
        btag_m into the vertices of boundary btag_n.
    :arg tol: tolerance allowed between the vertex coordinates of one boundary and
        the transformed vertex coordinates of another boundary when attempting to
        match the two.
    """
    mapped_btags = set(
        (btag_m, btag_n)
        for btag_m, btag_n, _ in glued_boundary_mappings)

    glued_boundary_mappings_both_ways = []

    for btag_m, btag_n, aff_transform in glued_boundary_mappings:
        aff_transform_np = np.array(aff_transform[0]), np.array(aff_transform[1])
        glued_boundary_mappings_both_ways.append(
            (btag_m, btag_n, aff_transform_np))
        if (btag_n, btag_m) not in mapped_btags:
            transform_mat, transform_vec = aff_transform_np
            inv_transform_mat = la.inv(transform_mat)
            inv_transform_vec = -inv_transform_mat @ transform_vec
            glued_boundary_mappings_both_ways.append(
                (btag_n, btag_m, (inv_transform_mat, inv_transform_vec)))

    face_id_pairs_for_mapping = _match_boundary_faces(mesh,
        glued_boundary_mappings_both_ways, tol)

    face_id_pairs_for_group_pair, mapping_indices_for_group_pair = (
        _translate_boundary_pair_adjacency_into_group_pair_adjacency(
            mesh, glued_boundary_mappings_both_ways, face_id_pairs_for_mapping))

    glued_mesh = _construct_glued_mesh(mesh, glued_boundary_mappings_both_ways,
        face_id_pairs_for_group_pair, mapping_indices_for_group_pair)

    return glued_mesh

# }}}


# {{{ map

def map_mesh(mesh, f):  # noqa
    """Apply the map *f* to the mesh. *f* needs to accept and return arrays of
    shape ``(ambient_dim, npoints)``."""

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

    if b is not None:
        b = b.reshape(-1, 1)

    def f(x):
        z = x
        if A is not None:
            z = A @ z

        if b is not None:
            z = z + b

        return z

    return map_mesh(mesh, f)


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

    :param axis: a (not necessarily unit) vector. By default, the rotation is
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
