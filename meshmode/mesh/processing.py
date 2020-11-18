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

import numpy as np
import numpy.linalg as la
import modepy as mp
from dataclasses import dataclass


__doc__ = """
.. autofunction:: find_group_indices
.. autofunction:: partition_mesh
.. autofunction:: find_volume_mesh_element_orientations
.. autofunction:: flip_simplex_element_group
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

                from meshmode.mesh import FacialAdjacencyGroup
                local_to_local_adjacency_groups[i_part_grp][j_part_grp] =\
                            FacialAdjacencyGroup(igroup=i_part_grp,
                                        ineighbor_group=j_part_grp,
                                        elements=elements,
                                        element_faces=element_faces,
                                        neighbors=neighbors,
                                        neighbor_faces=neighbor_faces)

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
    """
    elements: np.ndarray
    element_faces: np.ndarray
    neighbor_parts: np.ndarray
    global_neighbors: np.ndarray
    neighbor_faces: np.ndarray


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

                pairwise_adj.append(_NonLocalAdjacencyData(elements, element_faces,
                            neighbor_parts, global_neighbors, neighbor_faces))

        if pairwise_adj:
            nonlocal_adj_data[i_part_grp] = _NonLocalAdjacencyData(
                np.concatenate([adj.elements for adj in pairwise_adj]),
                np.concatenate([adj.element_faces for adj in pairwise_adj]),
                np.concatenate([adj.neighbor_parts for adj in pairwise_adj]),
                np.concatenate([adj.global_neighbors for adj in pairwise_adj]),
                np.concatenate([adj.neighbor_faces for adj in pairwise_adj]))

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
            elements = np.array([], dtype=mesh.element_id_dtype)
            element_faces = np.array([], dtype=mesh.face_id_dtype)
            neighbor_parts = np.array([], dtype=np.int32)
            neighbors = np.array([], dtype=mesh.element_id_dtype)
            neighbor_elements = np.array([], dtype=mesh.element_id_dtype)
            neighbor_faces = np.array([], dtype=mesh.face_id_dtype)

        elif bdry is None:
            # Non-local adjacency only
            elements = nl.elements
            element_faces = nl.element_faces
            neighbor_parts = nl.neighbor_parts
            neighbors = np.empty_like(elements)
            for inonlocal in range(len(neighbors)):
                i_neighbor_part = neighbor_parts[inonlocal]
                from meshmode.mesh import BTAG_REALLY_ALL, BTAG_PARTITION
                neighbors[inonlocal] = -(
                                boundary_tag_bit(BTAG_REALLY_ALL)
                                | boundary_tag_bit(BTAG_PARTITION(i_neighbor_part)))
            neighbor_elements = global_elem_to_neighbor_elem[nl.global_neighbors]
            neighbor_faces = nl.neighbor_faces

        elif nl is None:
            # Boundary only
            nelems = len(bdry.elements)
            elements = bdry.elements
            element_faces = bdry.element_faces
            neighbor_parts = np.empty(nelems, dtype=np.int32)
            neighbor_parts.fill(-1)
            neighbors = bdry.neighbors
            neighbor_elements = np.empty(nelems, dtype=mesh.element_id_dtype)
            neighbor_elements.fill(-1)
            neighbor_faces = np.empty(nelems, dtype=mesh.face_id_dtype)
            neighbor_faces.fill(-1)

        else:
            # Both; need to merge together
            nnonlocal = len(nl.elements)
            nbdry = len(bdry.elements)
            nelems = nnonlocal + nbdry
            elements = np.empty(nelems, dtype=mesh.element_id_dtype)
            element_faces = np.empty(nelems, dtype=mesh.face_id_dtype)
            neighbor_parts = np.empty(nelems, dtype=np.int32)
            neighbors = np.empty(nelems, dtype=mesh.element_id_dtype)
            neighbor_elements = np.empty(nelems, dtype=mesh.element_id_dtype)
            neighbor_faces = np.empty(nelems, dtype=mesh.face_id_dtype)

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

            # Merge boundary part
            bdry_indices = np.where(perm >= nnonlocal)[0]
            elements[bdry_indices] = bdry.elements
            element_faces[bdry_indices] = bdry.element_faces
            neighbors[bdry_indices] = bdry.neighbors
            neighbor_parts[bdry_indices] = -1
            neighbor_elements[bdry_indices] = -1
            neighbor_faces[bdry_indices] = -1

        from meshmode.mesh import InterPartitionAdjacencyGroup
        inter_partition_adj_groups.append(InterPartitionAdjacencyGroup(
                    igroup=i_part_grp, ineighbor_group=None, elements=elements,
                    element_faces=element_faces, neighbors=neighbors,
                    neighbor_partitions=neighbor_parts,
                    partition_neighbors=neighbor_elements,
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
    flip_matrix = mp.resampling_matrix(
            mp.simplex_best_available_basis(dim, order),
            flipped_unit_nodes, unit_nodes)

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

    return mesh.copy(
            vertices=vertices, groups=new_groups,
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
