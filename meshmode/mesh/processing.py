from __future__ import division, absolute_import, print_function

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

from six.moves import range
from functools import reduce

import numpy as np
import numpy.linalg as la
import modepy as mp


__doc__ = """
.. autofunction:: find_group_indices
.. autofunction:: partition_mesh
.. autofunction:: find_volume_mesh_element_orientations
.. autofunction:: perform_flips
.. autofunction:: find_bounding_box
.. autofunction:: merge_disjoint_meshes
.. autofunction:: split_mesh_groups
.. autofunction:: map_mesh
.. autofunction:: affine_map
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
    for igrp, grp in enumerate(groups):
        next_grp_boundary += grp.nelements
        grps += meshwide_elems >= next_grp_boundary
    return grps


# {{{ partition_mesh

def _create_element_part_map(part_per_element, parts, element_id_dtype):
    """
    Create a map from global elem to partition elem for a set of partitions.

    :arg part_per_element: A :class:`numpy.ndarray` mapping element indices to
        partition numbers.
    :arg parts: A `set` of partition numbers.
    :arg element_id_dtype: The element index data type.
    :returns: A :class:`numpy.ndarray` that maps an element's global index
        to its corresponding partition-wide index if that partition belongs to
        *parts* (and if not, to -1).
    """
    n_elems_in_part = {ipart: 0 for ipart in parts}
    global_elem_to_part_elem = np.empty(len(part_per_element),
                dtype=element_id_dtype)
    global_elem_to_part_elem[:] = -1
    for ielem, ipart in enumerate(part_per_element):
        if ipart not in parts:
            continue
        global_elem_to_part_elem[ielem] = n_elems_in_part[ipart]
        n_elems_in_part[ipart] += 1
    return global_elem_to_part_elem


def _split_by_group(groups, elements):
    """
    Split a sorted array of elements according to group.

    :arg groups: An array of `~meshmode.mesh.ElementGroup` instances.
    :arg elements: A sorted array of element indices.
    :returns: A (CSR-style) compressed :class:`numpy.ndarray` that indicates for
        each group the subrange of *elements* that belongs to it.
    """
    group_elem_start = []
    start_idx = 0
    for grp in groups:
        group_elem_start.append(start_idx)
        # Find the index of first element in the next group.
        end_idx = len(elements)
        for idx in range(start_idx, len(elements)):
            if elements[idx] - grp.element_nr_base >= grp.nelements:
                end_idx = idx
                break
        start_idx = end_idx
    group_elem_start.append(start_idx)
    return group_elem_start


def _filter_mesh_groups(groups, selected_elements):
    """
    Create new mesh groups containing a selected subset of elements.

    :arg groups: An array of `~meshmode.mesh.ElementGroup` instances.
    :arg selected_elements: An array of indices of elements to be included in the
        filtered groups.
    :returns: A tuple ``(new_groups, group_to_new_group, required_vertex_indices)``,
        where *new_groups* is made up of groups from *groups* with elements not in
        *selected_elements* removed (Note: empty groups are omitted),
        *group_to_new_group* maps groups in *groups* to their corresponding location
        in *new_groups*, and *required_vertex_indices* contains indices of all
        vertices required for elements belonging to *new_groups*.
    """
    group_elem_start = _split_by_group(groups, selected_elements)

    n_new_groups = 0
    group_to_new_group = [None for _ in groups]
    for igrp in range(len(groups)):
        start_idx = group_elem_start[igrp]
        end_idx = group_elem_start[igrp+1]
        if end_idx == start_idx:
            continue
        group_to_new_group[igrp] = n_new_groups
        n_new_groups += 1

    new_indices = []
    new_nodes = []

    # The set of vertex indices we need.
    # NOTE: There are two methods for producing required_vertex_indices.
    #   Optimizations may come from further exploring these options.
    #index_set = np.array([], dtype=int)
    index_sets = np.array([], dtype=set)

    for igrp, grp in enumerate(groups):
        if group_to_new_group[igrp] is None:
            continue
        start_idx = group_elem_start[igrp]
        end_idx = group_elem_start[igrp+1]
        group_elements = selected_elements[start_idx:end_idx]
        new_indices.append(grp.vertex_indices[group_elements - grp.element_nr_base])
        ambient_dim = grp.nodes.shape[0]
        new_nodes.append(
            np.zeros(
                (ambient_dim, end_idx - start_idx, grp.nunit_nodes)))
        for i in range(ambient_dim):
            for j in range(start_idx, end_idx):
                new_idx = j - start_idx
                elem = group_elements[new_idx] - grp.element_nr_base
                new_nodes[-1][i, new_idx, :] = grp.nodes[i, elem, :]
        index_sets = np.append(index_sets, set(new_indices[-1].ravel()))

    # A sorted np.array of vertex indices we need (without duplicates).
    #required_vertex_indices = np.unique(np.sort(index_set))
    required_vertex_indices = np.array(list(set.union(*index_sets)))

    # Our indices need to be in range [0, len(mesh.nelements)].
    for indices in new_indices:
        for i in range(len(indices)):
            for j in range(len(indices[0])):
                original_index = indices[i, j]
                indices[i, j] = np.where(required_vertex_indices
                            == original_index)[0]

    new_groups = []
    for igrp, grp in enumerate(groups):
        i_new_group = group_to_new_group[igrp]
        if i_new_group is not None:
            new_groups.append(
                type(grp)(
                    grp.order, new_indices[i_new_group], new_nodes[i_new_group],
                    unit_nodes=grp.unit_nodes))

    return new_groups, group_to_new_group, required_vertex_indices


def _create_local_to_local_adjacency_groups(mesh, global_elem_to_part_elem,
            part_mesh_groups, global_group_to_part_group,
            part_mesh_group_elem_base):
    """
    Create local-to-local facial adjacency groups for partitioned mesh.

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

    :returns: A list of `dict`s, `local_to_local_adjacency_groups`, that maps pairs
        of partitioned group indices to `~meshmode.mesh.FacialAdjacencyGroup`
        instances if they have local-to-local adjacency.
    """
    local_to_local_adjacency_groups = [dict() for _ in part_mesh_groups]

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

            adj_indices = np.where(np.logical_and(elements_are_local,
                        neighbors_are_local))[0]

            if len(adj_indices) > 0:
                part_elem_base_i = part_mesh_group_elem_base[i_part_grp]
                part_elem_base_j = part_mesh_group_elem_base[j_part_grp]

                elements = global_elem_to_part_elem[facial_adj.elements[
                            adj_indices] + elem_base_i] - part_elem_base_i
                element_faces = facial_adj.element_faces[adj_indices]
                neighbors = global_elem_to_part_elem[facial_adj.neighbors[
                            adj_indices] + elem_base_j] - part_elem_base_j
                neighbor_faces = facial_adj.neighbor_faces[adj_indices]

                from meshmode.mesh import FacialAdjacencyGroup
                local_to_local_adjacency_groups[i_part_grp][j_part_grp] =\
                            FacialAdjacencyGroup(igroup=i_part_grp,
                                        ineighbor_group=j_part_grp,
                                        elements=elements,
                                        element_faces=element_faces,
                                        neighbors=neighbors,
                                        neighbor_faces=neighbor_faces)

    return local_to_local_adjacency_groups


def _collect_nonlocal_adjacency_data(mesh, part_per_elem, global_elem_to_part_elem,
            part_mesh_groups, global_group_to_part_group,
            part_mesh_group_elem_base):
    """
    Collect nonlocal adjacency data for partitioned mesh, and store it in an
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

    :returns: A tuple `(elements, element_faces, neighbor_parts, global_neighbors,
        neighbor_faces)`, where each component is a :class:`numpy.ndarray` storing
        an entry for every local element face that is shared with a remote element.
        *elements* stores the group-relative element index, *element_faces* the
        index of the shared face in the local element, *neighbor_parts* the
        partition containing the remote element, *global_neighbors* the global
        element index of the remote element, and *neighbor_faces* the index of the
        shared face in the remote element.
    """
    nonlocal_adj_data = [None for _ in part_mesh_groups]

    for igrp, facial_adj_dict in enumerate(mesh.facial_adjacency_groups):
        i_part_grp = global_group_to_part_group[igrp]
        if i_part_grp is None:
            continue

        for jgrp, facial_adj in facial_adj_dict.items():
            if jgrp is None:
                continue

            elem_base_i = mesh.groups[igrp].element_nr_base
            elem_base_j = mesh.groups[jgrp].element_nr_base

            elements_are_local = global_elem_to_part_elem[facial_adj.elements
                        + elem_base_i] >= 0
            neighbors_are_nonlocal = global_elem_to_part_elem[facial_adj.neighbors
                        + elem_base_j] < 0
            adj_indices = np.where(np.logical_and(elements_are_local,
                        neighbors_are_nonlocal))[0]

            if len(adj_indices) > 0:
                part_elem_base_i = part_mesh_group_elem_base[i_part_grp]
                elements = global_elem_to_part_elem[facial_adj.elements[
                            adj_indices] + elem_base_i] - part_elem_base_i
                element_faces = facial_adj.element_faces[adj_indices]
                global_neighbors = facial_adj.neighbors[adj_indices] + elem_base_j
                neighbor_parts = part_per_elem[global_neighbors]
                neighbor_faces = facial_adj.neighbor_faces[adj_indices]

                if nonlocal_adj_data[i_part_grp] is None:
                    nonlocal_adj_data[i_part_grp] = (elements, element_faces,
                                neighbor_parts, global_neighbors, neighbor_faces)
                else:
                    adj_elements, adj_element_faces, adj_neighbor_parts,\
                                adj_global_neighbors, adj_neighbor_faces =\
                                nonlocal_adj_data[i_part_grp]
                    adj_elements = np.concatenate((adj_elements, elements))
                    adj_element_faces = np.concatenate((adj_element_faces,
                                element_faces))
                    adj_neighbor_parts = np.concatenate((adj_neighbor_parts,
                                neighbor_parts))
                    adj_global_neighbors = np.concatenate((adj_global_neighbors,
                                global_neighbors))
                    adj_neighbor_faces = np.concatenate((adj_neighbor_faces,
                                neighbor_faces))
                    nonlocal_adj_data[i_part_grp] = (adj_elements,
                                adj_element_faces, adj_neighbor_parts,
                                adj_global_neighbors, adj_neighbor_faces)

    return nonlocal_adj_data


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

    :returns: A tuple `(elements, element_faces, neighbors)`, where each component
        is a :class:`numpy.ndarray` storing an entry for every local element face
        that is a boundary. *elements* stores the group-relative element index,
        *element_faces* the index of the boundary face in the element, and
        *neighbors* the inherited `neighbors` value from the global mesh.
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
            bdry_data[i_part_grp] = (elements, element_faces, neighbors)

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
    :arg nonlocal_adj_data: Non-local adjacency data for the current partition (see
        :func:`_collect_nonlocal_adjacency_data`).
    :arg bdry_data: Boundary data for the current partition (see
        :func:`_collect_bdry_data`).

    :returns: A list of `~meshmode.mesh.InterPartitionAdjacencyGroup` instances, one
        for each group in *part_mesh_groups*, containing the aggregated non-local
        adjacency and boundary information from *nonlocal_adj_data* and *bdry_data*.
    """
    global_elem_to_neighbor_elem = _create_element_part_map(part_per_element,
                all_neighbor_parts, element_id_dtype=mesh.element_id_dtype)

    inter_partition_adj_groups = []

    for i_part_grp in range(len(part_mesh_groups)):
        has_nonlocal = nonlocal_adj_data[i_part_grp] is not None
        has_bdry = bdry_data[i_part_grp] is not None
        if not has_nonlocal and not has_bdry:
            elements = np.array([], dtype=mesh.element_id_dtype)
            element_faces = np.array([], dtype=mesh.face_id_dtype)
            neighbor_parts = np.array([], dtype=np.int32)
            neighbors = np.array([], dtype=mesh.element_id_dtype)
            neighbor_elements = np.array([], dtype=mesh.element_id_dtype)
            neighbor_faces = np.array([], dtype=mesh.face_id_dtype)

        elif not has_bdry:
            # Non-local adjacency only
            elements, element_faces, neighbor_parts, global_neighbor_elements,\
                        neighbor_faces = nonlocal_adj_data[i_part_grp]
            neighbors = np.empty_like(elements)
            for inonlocal in range(len(neighbors)):
                i_neighbor_part = neighbor_parts[inonlocal]
                from meshmode.mesh import BTAG_REALLY_ALL, BTAG_PARTITION
                neighbors[inonlocal] = -(
                                boundary_tag_bit(BTAG_REALLY_ALL)
                                | boundary_tag_bit(BTAG_PARTITION(i_neighbor_part)))
            neighbor_elements = np.empty_like(global_neighbor_elements)
            for ielem in range(len(global_neighbor_elements)):
                neighbor_elements[ielem] = global_elem_to_neighbor_elem[
                            global_neighbor_elements[ielem]]

        elif not has_nonlocal:
            # Boundary only
            elements, element_faces, neighbors = bdry_data[i_part_grp]
            nelems = len(elements)
            neighbor_parts = np.empty(nelems, dtype=np.int32)
            neighbor_parts.fill(-1)
            neighbor_elements = np.empty(nelems, dtype=mesh.element_id_dtype)
            neighbor_elements.fill(-1)
            neighbor_faces = np.empty(nelems, dtype=mesh.face_id_dtype)
            neighbor_faces.fill(-1)

        else:
            # Both; need to merge together
            nnonlocal = len(nonlocal_adj_data[i_part_grp][0])
            nbdry = len(bdry_data[i_part_grp][0])
            nelems = nnonlocal + nbdry
            elements = np.empty(nelems, dtype=mesh.element_id_dtype)
            element_faces = np.empty(nelems, dtype=mesh.face_id_dtype)
            neighbor_parts = np.empty(nelems, dtype=np.int32)
            neighbors = np.empty(nelems, dtype=mesh.element_id_dtype)
            neighbor_elements = np.empty(nelems, dtype=mesh.element_id_dtype)
            neighbor_faces = np.empty(nelems, dtype=mesh.face_id_dtype)

            # Combine lists of elements/faces and sort to assist in merging
            nonlocal_elements, nonlocal_element_faces, nonlocal_neighbor_parts,\
                        nonlocal_global_neighbor_elements, nonlocal_neighbor_faces =\
                        nonlocal_adj_data[i_part_grp]
            nonlocal_neighbor_elements = np.empty_like(
                        nonlocal_global_neighbor_elements)
            for ielem in range(len(nonlocal_global_neighbor_elements)):
                nonlocal_neighbor_elements[ielem] = global_elem_to_neighbor_elem[
                            nonlocal_global_neighbor_elements[ielem]]
            bdry_elements, bdry_element_faces, bdry_neighbors =\
                        bdry_data[i_part_grp]
            combined_elements = np.concatenate((nonlocal_elements, bdry_elements))
            combined_element_faces = np.concatenate((nonlocal_element_faces,
                        bdry_element_faces))
            perm = np.lexsort([combined_element_faces, combined_elements])

            # Merge
            imerged = 0
            for icombined in perm:
                if icombined < nnonlocal:
                    # Next entry is a non-local adjacency
                    inonlocal = icombined
                    elements[imerged] = nonlocal_elements[inonlocal]
                    element_faces[imerged] = nonlocal_element_faces[inonlocal]
                    neighbor_parts[imerged] = nonlocal_neighbor_parts[inonlocal]
                    i_neighbor_part = neighbor_parts[imerged]
                    from meshmode.mesh import BTAG_REALLY_ALL, BTAG_PARTITION
                    neighbors[imerged] = -(
                                boundary_tag_bit(BTAG_REALLY_ALL)
                                | boundary_tag_bit(BTAG_PARTITION(i_neighbor_part)))
                    neighbor_elements[imerged] = nonlocal_neighbor_elements[
                                inonlocal]
                    neighbor_faces[imerged] = nonlocal_neighbor_faces[inonlocal]
                else:
                    # Next entry is a boundary
                    ibdry = icombined - nnonlocal
                    elements[imerged] = bdry_elements[ibdry]
                    element_faces[imerged] = bdry_element_faces[ibdry]
                    neighbor_parts[imerged] = -1
                    neighbors[imerged] = bdry_neighbors[ibdry]
                    neighbor_elements[imerged] = -1
                    neighbor_faces[imerged] = -1
                imerged = imerged + 1

        from meshmode.mesh import InterPartitionAdjacencyGroup
        inter_partition_adj_groups.append(InterPartitionAdjacencyGroup(
                    igroup=i_part_grp, ineighbor_group=None, elements=elements,
                    element_faces=element_faces, neighbors=neighbors,
                    neighbor_partitions=neighbor_parts,
                    global_neighbors=neighbor_elements,
                    neighbor_faces=neighbor_faces))

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

    global_elem_to_part_elem = _create_element_part_map(part_per_element,
                {part_num}, element_id_dtype=mesh.element_id_dtype)

    # Create new mesh groups that mimick the original mesh's groups but only contain
    # the local partition's elements
    part_mesh_groups, global_group_to_part_group, required_vertex_indices =\
                _filter_mesh_groups(mesh.groups, queried_elems)

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

    # Assemble full set of neighbors
    all_neighbor_parts = set()
    for adj in nonlocal_adj_data:
        if adj is not None:
            _, _, grp_neighbor_parts, _, _ = adj
            for i_neighbor_part in grp_neighbor_parts:
                all_neighbor_parts.add(i_neighbor_part)

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
            dtype=np.object)

    for ispan in range(nspan_vectors):
        for idim in range(ambient_dim):
            spanning_object_array[ispan, idim] = \
                    spanning_vectors[idim, :, ispan]

    from pymbolic.geometric_algebra import MultiVector

    mvs = [MultiVector(vec) for vec in spanning_object_array]

    from operator import xor
    outer_prod = -reduce(xor, mvs)

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

def flip_simplex_element_group(vertices, grp, grp_flip_flags):
    from modepy.tools import barycentric_to_unit, unit_to_barycentric

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

    # Generate a resampling matrix that corresponds to the
    # first two barycentric coordinates being swapped.

    bary_unit_nodes = unit_to_barycentric(grp.unit_nodes)

    flipped_bary_unit_nodes = bary_unit_nodes.copy()
    flipped_bary_unit_nodes[0, :] = bary_unit_nodes[1, :]
    flipped_bary_unit_nodes[1, :] = bary_unit_nodes[0, :]
    flipped_unit_nodes = barycentric_to_unit(flipped_bary_unit_nodes)

    flip_matrix = mp.resampling_matrix(
            mp.simplex_best_available_basis(grp.dim, grp.order),
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

    flip_flags = flip_flags.astype(np.bool)

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

    from meshmode.mesh import Mesh
    return Mesh(vertices, new_groups, skip_tests=True,
            nodal_adjacency=mesh.nodal_adjacency_init_arg(),
            facial_adjacency_groups=mesh._facial_adjacency_groups,
            is_conforming=mesh.is_conforming)

# }}}


# {{{ affine map

def affine_map(mesh, A=None, b=None):  # noqa
    """Apply the affine map *f(x)=Ax+b* to the geometry of *mesh*."""

    if A is None:
        A = np.eye(mesh.ambient_dim)  # noqa

    if b is None:
        b = np.zeros(A.shape[0])

    def f(x):
        return np.dot(A, x) + b.reshape(-1, 1)

    return map_mesh(mesh, f)

# }}}

# vim: foldmethod=marker
