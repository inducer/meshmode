# mypy: disallow-untyped-defs
from __future__ import annotations


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

from dataclasses import dataclass, replace
from functools import reduce
from typing import TYPE_CHECKING, Literal
from warnings import warn

import numpy as np
import numpy.linalg as la

import modepy as mp

from meshmode.mesh import (
    BTAG_PARTITION,
    BoundaryAdjacencyGroup,
    FacialAdjacencyGroup,
    InteriorAdjacencyGroup,
    InterPartAdjacencyGroup,
    Mesh,
    MeshElementGroup,
    PartID,
    TensorProductElementGroup,
    _FaceIDs,
    make_mesh,
)
from meshmode.mesh.tools import AffineMap, find_point_to_point_mapping


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


__doc__ = """
.. autoclass:: BoundaryPairMapping

.. autofunction:: find_group_indices
.. autofunction:: partition_mesh
.. autofunction:: find_volume_mesh_element_orientations
.. autofunction:: flip_element_group
.. autofunction:: perform_flips
.. autofunction:: find_bounding_box
.. autofunction:: merge_disjoint_meshes
.. autofunction:: make_mesh_grid
.. autofunction:: split_mesh_groups
.. autofunction:: glue_mesh_boundaries

.. autofunction:: map_mesh
.. autofunction:: affine_map
.. autofunction:: rotate_mesh_around_axis

.. autofunction:: remove_unused_vertices
"""


def find_group_indices(
        groups: Sequence[MeshElementGroup],
        meshwide_elems: np.ndarray) -> np.ndarray:
    """
    :arg groups: A list of :class:`~meshmode.mesh.MeshElementGroup` instances
        that contain *meshwide_elems*.
    :arg meshwide_elems: A :class:`numpy.ndarray` of mesh-wide element numbers.
        Usually computed by ``elem + base_element_nr``.
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

def _compute_global_elem_to_part_elem(
        nelements: int,
        part_id_to_elements: Mapping[PartID, np.ndarray],
        part_id_to_part_index: Mapping[PartID, int],
        element_id_dtype: np.dtype) -> np.ndarray:
    """
    Create a map from global element index to part-wide element index for a set of
    parts.

    :arg nelements: The number of elements in the global mesh.
    :arg part_id_to_elements: A :class:`dict` mapping a part identifier to
        a sorted :class:`numpy.ndarray` of elements.
    :arg part_id_to_part_index: A mapping from part identifiers to indices in
        the range ``[0, num_parts)``.
    :arg element_id_dtype: The element index data type.
    :returns: A :class:`numpy.ndarray` ``global_elem_to_part_elem`` of shape
        ``(nelements, 2)``, where ``global_elem_to_part_elem[ielement, 0]`` gives
        the part index of the element and
        ``global_elem_to_part_elem[ielement, 1]`` gives its part-wide element index.
    """
    global_elem_to_part_elem = np.empty((nelements, 2), dtype=element_id_dtype)
    for part_id in part_id_to_elements.keys():
        elements = part_id_to_elements[part_id]
        global_elem_to_part_elem[elements, 0] = part_id_to_part_index[part_id]
        global_elem_to_part_elem[elements, 1] = np.indices(
            (len(elements),), dtype=element_id_dtype)

    return global_elem_to_part_elem


def _filter_mesh_groups(
        mesh: Mesh,
        selected_elements: np.ndarray,
        vertex_id_dtype: np.dtype) -> tuple[list[MeshElementGroup], np.ndarray]:
    """
    Create new mesh groups containing a selected subset of elements.

    :arg mesh: A `~meshmode.mesh.Mesh` instance.
    :arg selected_elements: A sorted array of indices of elements to be included in
        the filtered groups.
    :arg vertex_id_dtype: The vertex index data type.
    :returns: A tuple ``(new_groups, required_vertex_indices)``, where *new_groups*
        is made up of groups from *mesh* containing only elements from
        *selected_elements* (Note: resulting groups may be empty) and
        *required_vertex_indices* contains indices of all vertices required for
        elements belonging to *new_groups*.
    """

    # {{{ find filtered_group_elements

    group_elem_starts = [
        np.searchsorted(selected_elements, base_element_nr)
        for base_element_nr in mesh.base_element_nrs
        ] + [len(selected_elements)]

    filtered_group_elements = []
    for igrp in range(len(mesh.groups)):
        start_idx, end_idx = group_elem_starts[igrp:igrp+2]

        filtered_group_elements.append(
            selected_elements[start_idx:end_idx] - mesh.base_element_nrs[igrp])

    # }}}

    # {{{ filter vertex indices

    filtered_vertex_indices = [
            grp.vertex_indices[
                    filtered_group_elements[igrp], :]
            for igrp, grp in enumerate(mesh.groups)
            if grp.vertex_indices is not None]

    filtered_vertex_indices_flat = np.concatenate([indices.ravel() for indices
                in filtered_vertex_indices])

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
            replace(grp,
                vertex_indices=new_vertex_indices[igrp],
                nodes=grp.nodes[:, filtered_group_elements[igrp], :].copy())
            for igrp, grp in enumerate(mesh.groups)]

    return new_groups, required_vertex_indices


def _get_connected_parts(
        mesh: Mesh,
        part_id_to_part_index: Mapping[PartID, int],
        global_elem_to_part_elem: np.ndarray,
        self_part_id: PartID) -> Sequence[PartID]:
    """
    Find the parts that are connected to the current part.

    :arg mesh: A :class:`~meshmode.mesh.Mesh` representing the unpartitioned mesh.
    :arg part_id_to_part_index: A mapping from part identifiers to indices in the
        range ``[0, num_parts)``.
    :arg global_elem_to_part_elem: A :class:`numpy.ndarray` that maps global element
        indices to part indices and part-wide element indices. See
        :func:`_compute_global_elem_to_part_elem`` for details.
    :arg self_part_id: The identifier of the part currently being created.

    :returns: A sequence of identifiers of the neighboring parts.
    """
    self_part_index = part_id_to_part_index[self_part_id]

    # This set is not used in a way that will cause nondeterminism.
    connected_part_indices = set()

    for igrp, facial_adj_list in enumerate(mesh.facial_adjacency_groups):
        int_grps = [
            grp for grp in facial_adj_list
            if isinstance(grp, InteriorAdjacencyGroup)]
        for facial_adj in int_grps:
            jgrp = facial_adj.ineighbor_group

            elem_base_i = mesh.base_element_nrs[igrp]
            elem_base_j = mesh.base_element_nrs[jgrp]

            elements_are_self = global_elem_to_part_elem[facial_adj.elements
                        + elem_base_i, 0] == self_part_index
            neighbors_are_other = global_elem_to_part_elem[facial_adj.neighbors
                        + elem_base_j, 0] != self_part_index

            connected_part_indices.update(
                global_elem_to_part_elem[
                    facial_adj.neighbors[
                        elements_are_self & neighbors_are_other]
                    + elem_base_j, 0])

    result = tuple(
        part_id
        for part_id, part_index in part_id_to_part_index.items()
        if part_index in connected_part_indices)
    assert len(set(result)) == len(result)
    return result


def _create_self_to_self_adjacency_groups(
        mesh: Mesh,
        global_elem_to_part_elem: np.ndarray,
        self_part_index: int,
        self_mesh_groups: Sequence[MeshElementGroup],
        self_mesh_group_elem_base: Sequence[int]) -> list[list[InteriorAdjacencyGroup]]:
    r"""
    Create self-to-self facial adjacency groups for a partitioned mesh.

    :arg mesh: A :class:`~meshmode.mesh.Mesh` representing the unpartitioned mesh.
    :arg global_elem_to_part_elem: A :class:`numpy.ndarray` that maps global element
        indices to part indices and part-wide element indices. See
        :func:`_compute_global_elem_to_part_elem`` for details.
    :arg self_part_index: The index of the part currently being created, in the
        range ``[0, num_parts)``.
    :arg self_mesh_groups: A list of :class:`~meshmode.mesh.MeshElementGroup`
        instances representing the partitioned mesh groups.
    :arg self_mesh_group_elem_base: A list containing the starting part-wide
        element index for each group in *self_mesh_groups*.

    :returns: A list of lists of `~meshmode.mesh.InteriorAdjacencyGroup` instances
        corresponding to the entries in *mesh.facial_adjacency_groups* that
        have self-to-self adjacency.
    """
    self_to_self_adjacency_groups: list[list[InteriorAdjacencyGroup]] = [
            [] for _ in self_mesh_groups]

    for igrp, facial_adj_list in enumerate(mesh.facial_adjacency_groups):
        int_grps = [
            grp for grp in facial_adj_list
            if isinstance(grp, InteriorAdjacencyGroup)]

        for facial_adj in int_grps:
            jgrp = facial_adj.ineighbor_group

            elem_base_i = mesh.base_element_nrs[igrp]
            elem_base_j = mesh.base_element_nrs[jgrp]

            elements_are_self = global_elem_to_part_elem[facial_adj.elements
                        + elem_base_i, 0] == self_part_index
            neighbors_are_self = global_elem_to_part_elem[facial_adj.neighbors
                        + elem_base_j, 0] == self_part_index

            adj_indices, = np.where(elements_are_self & neighbors_are_self)

            if len(adj_indices) > 0:
                self_elem_base_i = self_mesh_group_elem_base[igrp]
                self_elem_base_j = self_mesh_group_elem_base[jgrp]

                elements = global_elem_to_part_elem[facial_adj.elements[
                            adj_indices] + elem_base_i, 1] - self_elem_base_i
                element_faces = facial_adj.element_faces[adj_indices]
                neighbors = global_elem_to_part_elem[facial_adj.neighbors[
                            adj_indices] + elem_base_j, 1] - self_elem_base_j
                neighbor_faces = facial_adj.neighbor_faces[adj_indices]

                self_to_self_adjacency_groups[igrp].append(
                    InteriorAdjacencyGroup(
                        igroup=igrp,
                        ineighbor_group=jgrp,
                        elements=elements,
                        element_faces=element_faces,
                        neighbors=neighbors,
                        neighbor_faces=neighbor_faces,
                        aff_map=facial_adj.aff_map))

    return self_to_self_adjacency_groups


def _create_self_to_other_adjacency_groups(
        mesh: Mesh,
        part_id_to_part_index: Mapping[PartID, int],
        global_elem_to_part_elem: np.ndarray,
        self_part_id: PartID,
        self_mesh_groups: Sequence[MeshElementGroup],
        self_mesh_group_elem_base: Sequence[int],
        connected_parts: Sequence[PartID]) -> list[list[InterPartAdjacencyGroup]]:
    """
    Create self-to-other adjacency groups for the partitioned mesh.

    :arg mesh: A :class:`~meshmode.mesh.Mesh` representing the unpartitioned mesh.
    :arg part_id_to_part_index: A mapping from part identifiers to indices in the
        range ``[0, num_parts)``.
    :arg global_elem_to_part_elem: A :class:`numpy.ndarray` that maps global element
        indices to part indices and part-wide element indices. See
        :func:`_compute_global_elem_to_part_elem`` for details.
    :arg self_part_id: The identifier of the part currently being created.
    :arg self_mesh_groups: A list of `~meshmode.mesh.MeshElementGroup` instances
        representing the partitioned mesh groups.
    :arg self_mesh_group_elem_base: A list containing the starting part-wide
        element index for each group in *self_mesh_groups*.
    :arg connected_parts: A :class:`set` containing the parts connected to
        the current one.

    :returns: A list of lists of `~meshmode.mesh.InterPartAdjacencyGroup` instances
        corresponding to the entries in *mesh.facial_adjacency_groups* that
        have self-to-other adjacency.
    """
    self_part_index = part_id_to_part_index[self_part_id]

    self_to_other_adj_groups: list[list[InterPartAdjacencyGroup]] = [
            [] for _ in self_mesh_groups]

    for igrp, facial_adj_list in enumerate(mesh.facial_adjacency_groups):
        int_grps = [
            grp for grp in facial_adj_list
            if isinstance(grp, InteriorAdjacencyGroup)]

        for facial_adj in int_grps:
            jgrp = facial_adj.ineighbor_group

            elem_base_i = mesh.base_element_nrs[igrp]
            elem_base_j = mesh.base_element_nrs[jgrp]

            global_elements = facial_adj.elements + elem_base_i
            global_neighbors = facial_adj.neighbors + elem_base_j

            elements_are_self = (
                global_elem_to_part_elem[global_elements, 0] == self_part_index)

            neighbor_part_indices = global_elem_to_part_elem[global_neighbors, 0]

            for neighbor_part_id in connected_parts:
                neighbor_part_index = part_id_to_part_index[neighbor_part_id]
                adj_indices, = np.where(
                    elements_are_self
                    & (neighbor_part_indices == neighbor_part_index))

                if len(adj_indices) > 0:
                    self_elem_base_i = self_mesh_group_elem_base[igrp]

                    elements = global_elem_to_part_elem[facial_adj.elements[
                                adj_indices] + elem_base_i, 1] - self_elem_base_i
                    element_faces = facial_adj.element_faces[adj_indices]
                    neighbors = global_elem_to_part_elem[
                        global_neighbors[adj_indices], 1]
                    neighbor_faces = facial_adj.neighbor_faces[adj_indices]

                    self_to_other_adj_groups[igrp].append(
                        InterPartAdjacencyGroup(
                            igroup=igrp,
                            boundary_tag=BTAG_PARTITION(neighbor_part_id),
                            part_id=neighbor_part_id,
                            elements=elements,
                            element_faces=element_faces,
                            neighbors=neighbors,
                            neighbor_faces=neighbor_faces,
                            aff_map=facial_adj.aff_map))

    return self_to_other_adj_groups


def _create_boundary_groups(
        mesh: Mesh,
        global_elem_to_part_elem: np.ndarray,
        self_part_index: PartID,
        self_mesh_groups: Sequence[MeshElementGroup],
        self_mesh_group_elem_base: Sequence[int]) -> list[list[BoundaryAdjacencyGroup]]:
    """
    Create boundary groups for partitioned mesh.

    :arg mesh: A :class:`~meshmode.mesh.Mesh` representing the unpartitioned mesh.
    :arg global_elem_to_part_elem: A :class:`numpy.ndarray` that maps global element
        indices to part indices and part-wide element indices. See
        :func:`_compute_global_elem_to_part_elem`` for details.
    :arg self_part_index: The index of the part currently being created, in the
        range ``[0, num_parts)``.
    :arg self_mesh_groups: A list of `~meshmode.mesh.MeshElementGroup` instances
        representing the partitioned mesh groups.
    :arg self_mesh_group_elem_base: A list containing the starting part-wide
        element index for each group in *self_mesh_groups*.

    :returns: A list of lists of `~meshmode.mesh.BoundaryAdjacencyGroup` instances
        corresponding to the entries in *mesh.facial_adjacency_groups* that have
        boundary faces.
    """
    bdry_adj_groups: list[list[BoundaryAdjacencyGroup]] = [
            [] for _ in self_mesh_groups]

    for igrp, facial_adj_list in enumerate(mesh.facial_adjacency_groups):
        bdry_grps = [
            grp for grp in facial_adj_list
            if isinstance(grp, BoundaryAdjacencyGroup)]

        for bdry_grp in bdry_grps:
            elem_base = mesh.base_element_nrs[igrp]

            adj_indices, = np.where(
                global_elem_to_part_elem[bdry_grp.elements + elem_base, 0]
                == self_part_index)

            if len(adj_indices) > 0:
                self_elem_base = self_mesh_group_elem_base[igrp]
                elements = global_elem_to_part_elem[bdry_grp.elements[adj_indices]
                            + elem_base, 1] - self_elem_base
                element_faces = bdry_grp.element_faces[adj_indices]
            else:
                elements = np.empty(0, dtype=mesh.element_id_dtype)
                element_faces = np.empty(0, dtype=mesh.face_id_dtype)

            bdry_adj_groups[igrp].append(
                BoundaryAdjacencyGroup(
                    igroup=igrp,
                    boundary_tag=bdry_grp.boundary_tag,
                    elements=elements,
                    element_faces=element_faces))

    return bdry_adj_groups


def _get_mesh_part(
        mesh: Mesh,
        part_id_to_elements: Mapping[PartID, np.ndarray],
        self_part_id: PartID) -> Mesh:
    """
    :arg mesh: A :class:`~meshmode.mesh.Mesh` to be partitioned.
    :arg part_id_to_elements: A :class:`dict` mapping a part identifier to
        a sorted :class:`numpy.ndarray` of elements.
    :arg self_part_id: The part identifier of the mesh to return.

    :returns: A :class:`~meshmode.mesh.Mesh` containing a part of *mesh*.

    .. versionadded:: 2017.1
    """
    if mesh.vertices is None:
        raise ValueError("Mesh must have vertices")

    element_counts = np.zeros(mesh.nelements)
    for elements in part_id_to_elements.values():
        element_counts[elements] += 1
    if np.any(element_counts > 1):
        raise ValueError("elements cannot belong to multiple parts")
    if np.any(element_counts < 1):
        raise ValueError("partition must contain all elements")

    part_id_to_part_index = {
        part_id: part_index
        for part_index, part_id in enumerate(part_id_to_elements.keys())}

    global_elem_to_part_elem = _compute_global_elem_to_part_elem(
        mesh.nelements, part_id_to_elements, part_id_to_part_index,
        mesh.element_id_dtype)

    # Create new mesh groups that mimic the original mesh's groups but only contain
    # the current part's elements
    self_mesh_groups, required_vertex_indices = _filter_mesh_groups(
        mesh, part_id_to_elements[self_part_id], mesh.vertex_id_dtype)

    self_part_index = part_id_to_part_index[self_part_id]

    self_vertices = np.zeros((mesh.ambient_dim, len(required_vertex_indices)))
    for dim in range(mesh.ambient_dim):
        self_vertices[dim] = mesh.vertices[dim][required_vertex_indices]

    self_mesh_group_elem_base = [0 for _ in self_mesh_groups]
    el_nr = 0
    for igrp, grp in enumerate(self_mesh_groups):
        self_mesh_group_elem_base[igrp] = el_nr
        el_nr += grp.nelements

    connected_parts = _get_connected_parts(
        mesh, part_id_to_part_index, global_elem_to_part_elem,
        self_part_id)

    self_to_self_adj_groups = _create_self_to_self_adjacency_groups(
                mesh, global_elem_to_part_elem, self_part_index, self_mesh_groups,
                self_mesh_group_elem_base)

    self_to_other_adj_groups = _create_self_to_other_adjacency_groups(
                mesh, part_id_to_part_index, global_elem_to_part_elem, self_part_id,
                self_mesh_groups, self_mesh_group_elem_base, connected_parts)

    boundary_adj_groups = _create_boundary_groups(
                mesh, global_elem_to_part_elem, self_part_index, self_mesh_groups,
                self_mesh_group_elem_base)

    def _gather_grps(igrp: int) -> list[FacialAdjacencyGroup]:
        self_grps: Sequence[FacialAdjacencyGroup] = self_to_self_adj_groups[igrp]
        other_grps: Sequence[FacialAdjacencyGroup] = self_to_other_adj_groups[igrp]
        bdry_grps: Sequence[FacialAdjacencyGroup] = boundary_adj_groups[igrp]

        return list(self_grps) + list(other_grps) + list(bdry_grps)

    # Combine adjacency groups
    self_facial_adj_groups = [
            _gather_grps(igrp) for igrp in range(len(self_mesh_groups))]

    return make_mesh(
            self_vertices,
            self_mesh_groups,
            facial_adjacency_groups=self_facial_adj_groups,
            is_conforming=mesh.is_conforming)


def partition_mesh(
        mesh: Mesh,
        part_id_to_elements: Mapping[PartID, np.ndarray],
        return_parts: Sequence[PartID] | None = None) -> Mapping[PartID, Mesh]:
    """
    :arg mesh: A :class:`~meshmode.mesh.Mesh` to be partitioned.
    :arg part_id_to_elements: A :class:`dict` mapping a part identifier to
        a sorted :class:`numpy.ndarray` of elements.
    :arg return_parts: An optional list of parts to return. By default, returns all
        parts.

    :returns: A :class:`dict` mapping part identifiers to instances of
        :class:`~meshmode.mesh.Mesh` that represent the corresponding part of
        *mesh*.
    """
    if return_parts is None:
        return_parts = list(part_id_to_elements.keys())

    return {
        part_id: _get_mesh_part(mesh, part_id_to_elements, part_id)
        for part_id in return_parts}

# }}}


# {{{ orientations

def find_volume_mesh_element_group_orientation(
        vertices: np.ndarray,
        grp: MeshElementGroup) -> np.ndarray:
    """
    :returns: a positive floating point number for each positively
        oriented element, and a negative floating point number for
        each negatively oriented element.
    """

    from meshmode.mesh import ModepyElementGroup

    if not isinstance(grp, ModepyElementGroup):
        raise NotImplementedError(
                "finding element orientations "
                "only supported on "
                "meshes containing element groups described by modepy")

    # (ambient_dim, nelements, nvertices)
    my_vertices = vertices[:, grp.vertex_indices]

    def evec(i: int) -> np.ndarray:
        """Make the i-th unit vector."""
        result = np.zeros(grp.dim)
        result[i] = 1
        return result

    def unpack_single(ary: np.ndarray | None) -> int:
        assert ary is not None
        item, = ary
        return item

    base_vertex_index = unpack_single(find_point_to_point_mapping(
             src_points=-np.ones(grp.dim).reshape(-1, 1),
             tgt_points=grp.vertex_unit_coordinates().T))
    spanning_vertex_indices = [
        unpack_single(find_point_to_point_mapping(
                     src_points=(-np.ones(grp.dim) + 2 * evec(i)).reshape(-1, 1),
                     tgt_points=grp.vertex_unit_coordinates().T))
        for i in range(grp.dim)
    ]

    spanning_vectors = (
                my_vertices[:, :, spanning_vertex_indices]
                - my_vertices[:, :, base_vertex_index][:, :, np.newaxis])

    ambient_dim, _nelements, nspan_vectors = spanning_vectors.shape
    assert nspan_vectors == grp.dim

    if ambient_dim != grp.dim:
        raise ValueError("can only find orientation of volume meshes")

    spanning_object_array: np.ndarray = np.empty(
            (nspan_vectors, ambient_dim),
            dtype=object)

    for ispan in range(nspan_vectors):
        for idim in range(ambient_dim):
            spanning_object_array[ispan, idim] = \
                    spanning_vectors[idim, :, ispan]

    from pymbolic.geometric_algebra import MultiVector

    mvs: list[MultiVector[np.floating]] = (
        [MultiVector(vec) for vec in spanning_object_array])

    from operator import xor
    outer_prod = -reduce(xor, mvs)      # pylint: disable=invalid-unary-operand-type

    if grp.dim == 1:
        # FIXME: This is a little weird.
        outer_prod = -outer_prod

    return (outer_prod.I | outer_prod).as_scalar()


def find_volume_mesh_element_orientations(
        mesh: Mesh, *,
        tolerate_unimplemented_checks: bool = False) -> np.ndarray:
    """Return a positive floating point number for each positively
    oriented element, and a negative floating point number for
    each negatively oriented element.

    :arg tolerate_unimplemented_checks: If *True*, elements for which no
        check is available will return *NaN*.
    """
    if mesh.vertices is None:
        raise ValueError("Mesh must have vertices to check orientation")

    result: np.ndarray = np.empty(mesh.nelements, dtype=np.float64)

    for base_element_nr, grp in zip(mesh.base_element_nrs, mesh.groups, strict=True):
        result_grp_view = result[base_element_nr:base_element_nr + grp.nelements]

        try:
            signed_area_elements = \
                    find_volume_mesh_element_group_orientation(
                            mesh.vertices, grp)
        except NotImplementedError:
            if tolerate_unimplemented_checks:
                result_grp_view[:] = float("nan")
            else:
                raise
        else:
            assert not np.isnan(signed_area_elements).any()
            result_grp_view[:] = signed_area_elements

    return result

# }}}


# {{{ flips

def get_simplex_element_flip_matrix(
            order: int,
            unit_nodes: np.ndarray,
            permutation: tuple[int, ...] | None = None,
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a resampling matrix that corresponds to a
    permutation of the barycentric coordinates being applied.
    The default permutation is to swap the
    first two barycentric coordinates.

    :arg order: The order of the function space on the simplex,
        (see second argument in :func:`modepy.simplex_best_available_basis`).
    :arg unit_nodes: A np array of unit nodes with shape *(dim, nunit_nodes)*.
    :arg permutation: Either *None*, or a tuple of shape storing a permutation:
        the *i*th barycentric coordinate gets mapped to the *permutation[i]*th
        coordinate.

    :return: A numpy array of shape *(nunit_nodes, nunit_nodes)*
        which, when its transpose is right-applied to the matrix of nodes
        (shaped *(dim, nunit_nodes)*), corresponds to the permutation being
        applied. Also, an array of indices to carry out the vertex permutation.
    """
    from modepy.tools import barycentric_to_unit, unit_to_barycentric

    bary_unit_nodes = unit_to_barycentric(unit_nodes)

    dim = unit_nodes.shape[0]

    flipped_bary_unit_nodes = bary_unit_nodes.copy()
    if permutation is None:
        # Swap the first two vertices on elements to be flipped.
        permutation_ary = np.arange(dim + 1)
        permutation_ary[1] = 0
        permutation_ary[0] = 1
    else:
        permutation_ary = np.asarray(permutation)

    flipped_bary_unit_nodes[permutation_ary, :] = bary_unit_nodes
    flipped_unit_nodes = barycentric_to_unit(flipped_bary_unit_nodes)

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

    return flip_matrix, permutation_ary


def _get_tensor_product_element_flip_matrix_and_vertex_permutation(
            grp: TensorProductElementGroup,
        ) -> tuple[np.ndarray, np.ndarray]:
    unit_flip_matrix = np.eye(grp.dim)
    unit_flip_matrix[0, 0] = -1

    flipped_vertices = np.einsum(
                 "ij,jn->in",
                 unit_flip_matrix,
                 grp.vertex_unit_coordinates().T)

    vertex_permutation_to = find_point_to_point_mapping(
        src_points=flipped_vertices,
        tgt_points=grp.vertex_unit_coordinates().T)
    if np.any(vertex_permutation_to < 0):
        raise RuntimeError("flip permutation was not found")

    flipped_unit_nodes = np.einsum("ij,jn->in", unit_flip_matrix, grp.unit_nodes)

    basis = mp.basis_for_space(grp.space, grp.shape)
    flip_matrix = mp.resampling_matrix(
        basis.functions,
        flipped_unit_nodes,
        grp.unit_nodes
    )

    flip_matrix[np.abs(flip_matrix) < 1e-15] = 0

    # Flipping twice should be the identity
    assert la.norm(
            np.dot(flip_matrix, flip_matrix)
            - np.eye(len(flip_matrix))) < 1e-13

    return flip_matrix, vertex_permutation_to


def flip_element_group(
        vertices: np.ndarray,
        grp: MeshElementGroup,
        grp_flip_flags: np.ndarray) -> MeshElementGroup:
    from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup

    if isinstance(grp, SimplexElementGroup):
        flip_matrix, vertex_permutation = get_simplex_element_flip_matrix(
                grp.order, grp.unit_nodes)

    elif isinstance(grp, TensorProductElementGroup):
        flip_matrix, vertex_permutation = \
            _get_tensor_product_element_flip_matrix_and_vertex_permutation(grp)

    else:
        raise NotImplementedError("flips only supported on "
                "simplices and tensor product elements")

    if grp.vertex_indices is not None:
        new_vertex_indices = grp.vertex_indices.copy()
        vertex_indices_to_be_flipped = grp.vertex_indices[grp_flip_flags]
        permuted_vertex_indices = np.empty_like(vertex_indices_to_be_flipped)
        permuted_vertex_indices[:, vertex_permutation] = vertex_indices_to_be_flipped
        new_vertex_indices[grp_flip_flags] = permuted_vertex_indices
    else:
        new_vertex_indices = None

    # Apply the flip matrix to the nodes.
    new_nodes = grp.nodes.copy()
    new_nodes[:, grp_flip_flags] = np.einsum(
            "ij,dej->dei",
            flip_matrix, grp.nodes[:, grp_flip_flags])

    return replace(grp, vertex_indices=new_vertex_indices, nodes=new_nodes)


def perform_flips(
        mesh: Mesh,
        flip_flags: np.ndarray,
        skip_tests: bool = False) -> Mesh:
    """
    :arg flip_flags: A :class:`numpy.ndarray` with
        :attr:`meshmode.mesh.Mesh.nelements` entries
        indicating by their Boolean value whether the element is to be
        flipped.
    """
    if mesh.vertices is None:
        raise ValueError("Mesh must have vertices to perform flips")

    flip_flags = flip_flags.astype(bool)

    new_groups = []
    for base_element_nr, grp in zip(mesh.base_element_nrs, mesh.groups, strict=True):
        grp_flip_flags = flip_flags[base_element_nr:base_element_nr + grp.nelements]

        if grp_flip_flags.any():
            new_grp = flip_element_group(mesh.vertices, grp, grp_flip_flags)
        else:
            new_grp = replace(grp)

        new_groups.append(new_grp)

    return make_mesh(
            mesh.vertices, new_groups, skip_tests=skip_tests,
            is_conforming=mesh.is_conforming,
            )

# }}}


# {{{ bounding box

def find_bounding_box(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """
    :return: a tuple *(min, max)*, each consisting of a :class:`numpy.ndarray`
        indicating the minimal and maximal extent of the geometry along each axis.
    """
    if mesh.vertices is None:
        raise ValueError("Mesh must have vertices to compute bounding box")

    return (
            np.min(mesh.vertices, axis=-1),
            np.max(mesh.vertices, axis=-1),
            )

# }}}


# {{{ merging

def merge_disjoint_meshes(
        meshes: Sequence[Mesh], *,
        skip_tests: bool = False,
        single_group: bool = False) -> Mesh:
    if not meshes:
        raise ValueError("must pass at least one mesh")

    from pytools import is_single_valued
    if not is_single_valued(mesh.ambient_dim for mesh in meshes):
        raise ValueError("all meshes must share the same ambient dimension")

    # {{{ assemble combined vertex array

    if all(mesh.vertices is not None for mesh in meshes):
        ambient_dim = meshes[0].ambient_dim
        nvertices = sum(mesh.nvertices for mesh in meshes)

        vert_dtype = np.result_type(*[mesh.vertex_dtype for mesh in meshes])
        vertices = np.empty((ambient_dim, nvertices), vert_dtype)

        current_vert_base = 0
        vert_bases = []
        for mesh in meshes:
            assert mesh.vertices is not None
            mesh_nvert = mesh.nvertices
            vertices[:, current_vert_base:current_vert_base+mesh_nvert] = \
                    mesh.vertices

            vert_bases.append(current_vert_base)
            current_vert_base += mesh_nvert
    else:
        raise ValueError("All meshes must have vertices to perform merge")

    # }}}

    # {{{ assemble new groups list

    nodal_adjacency: Literal[False] | None = None
    if any(mesh._nodal_adjacency is not None for mesh in meshes):
        nodal_adjacency = False

    facial_adjacency_groups: Literal[False] | None = None
    if any(mesh._facial_adjacency_groups is not None for mesh in meshes):
        facial_adjacency_groups = False

    if single_group:
        from pytools import single_valued
        ref_group = single_valued(
            [group for mesh in meshes for group in mesh.groups],
            lambda x, y: (
                type(x) is type(y)
                and x.order == y.order
                and np.array_equal(x.unit_nodes, y.unit_nodes)
                ))

        group_vertex_indices = []
        group_nodes = []
        for mesh, vert_base in zip(meshes, vert_bases, strict=True):
            for group in mesh.groups:
                assert group.vertex_indices is not None
                group_vertex_indices.append(group.vertex_indices + vert_base)
                group_nodes.append(group.nodes)

        vertex_indices = np.vstack(group_vertex_indices)
        nodes = np.hstack(group_nodes)

        if not nodes.flags.c_contiguous:
            # hstack stopped producing C-contiguous arrays in numpy 1.14
            nodes = nodes.copy(order="C")

        new_groups = [replace(ref_group, vertex_indices=vertex_indices, nodes=nodes)]

    else:
        new_groups = []
        for mesh, vert_base in zip(meshes, vert_bases, strict=True):
            for group in mesh.groups:
                assert group.vertex_indices is not None
                new_vertex_indices = group.vertex_indices + vert_base
                new_group = replace(group, vertex_indices=new_vertex_indices)

                new_groups.append(new_group)

    # }}}

    return make_mesh(
            vertices, new_groups,
            skip_tests=skip_tests,
            nodal_adjacency=nodal_adjacency,
            facial_adjacency_groups=facial_adjacency_groups,
            is_conforming=all(mesh.is_conforming for mesh in meshes))

# }}}


# {{{ split meshes

def split_mesh_groups(
        mesh: Mesh,
        element_flags: np.ndarray,
        return_subgroup_mapping: bool = False,
        ) -> Mesh | tuple[Mesh, dict[tuple[int, int], int]]:
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

    new_groups: list[MeshElementGroup] = []
    subgroup_to_group_map = {}

    for igrp, (base_element_nr, grp) in enumerate(
            zip(mesh.base_element_nrs, mesh.groups, strict=True)
            ):
        assert grp.vertex_indices is not None
        grp_flags = element_flags[base_element_nr:base_element_nr + grp.nelements]
        unique_grp_flags = np.unique(grp_flags)

        for flag in unique_grp_flags:
            subgroup_to_group_map[igrp, flag] = len(new_groups)

            # NOTE: making copies to maintain contiguity of the arrays
            mask = grp_flags == flag
            new_groups.append(replace(grp,
                vertex_indices=grp.vertex_indices[mask, :].copy(),
                nodes=grp.nodes[:, mask, :].copy(),
                ))

    mesh = make_mesh(
            vertices=mesh.vertices,
            groups=new_groups,
            is_conforming=mesh.is_conforming)

    if return_subgroup_mapping:
        return mesh, subgroup_to_group_map
    else:
        return mesh

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


def _get_boundary_face_ids(mesh: Mesh, btag: int) -> _FaceIDs:
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


def _get_face_vertex_indices(mesh: Mesh, face_ids: _FaceIDs) -> np.ndarray:
    max_face_vertices = max(
        len(ref_fvi)
        for grp in mesh.groups
        for ref_fvi in grp.face_vertex_indices())

    face_vertex_indices_per_group = []
    for igrp, grp in enumerate(mesh.groups):
        assert grp.vertex_indices is not None

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


def _match_boundary_faces(
        mesh: Mesh, bdry_pair_mapping: BoundaryPairMapping, tol: float,
        ) -> tuple[_FaceIDs, _FaceIDs]:
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
    if mesh.vertices is None:
        raise ValueError("Mesh must have vertices")

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

    bdry_m_vertices = mesh.vertices[:, bdry_m_vertex_indices]
    bdry_n_vertices = mesh.vertices[:, bdry_n_vertex_indices]

    m_idx_to_n_idx = find_point_to_point_mapping(
        bdry_pair_mapping.aff_map(bdry_m_vertices),
        bdry_n_vertices)

    unmatched_bdry_m_vertex_indices = bdry_m_vertex_indices[
        np.where(m_idx_to_n_idx < 0)[0]]
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

    matched_bdry_n_vertex_indices = bdry_n_vertex_indices[
        m_idx_to_n_idx]

    from meshmode.mesh import _concatenate_face_ids
    face_ids = _concatenate_face_ids([bdry_m_face_ids, bdry_n_face_ids])

    max_vertex_index = 0
    for grp in mesh.groups:
        assert grp.vertex_indices is not None
        max_vertex_index = max(max_vertex_index, np.max(grp.vertex_indices))
    vertex_index_map, = np.indices(
        (max_vertex_index + 1,), dtype=mesh.element_id_dtype)
    vertex_index_map[bdry_m_vertex_indices] = matched_bdry_n_vertex_indices

    from meshmode.mesh import _match_faces_by_vertices
    face_index_pairs = _match_faces_by_vertices(mesh.groups, face_ids,
        vertex_index_map_func=lambda vs: vertex_index_map[vs])

    assert face_index_pairs.shape[1] == nfaces

    # Since the first boundary's faces come before the second boundary's in
    # face_ids, the first boundary's faces should all be in the first row of the
    # result of _match_faces_by_vertices
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

def glue_mesh_boundaries(
        mesh: Mesh,
        bdry_pair_mappings_and_tols: Sequence[tuple[BoundaryPairMapping, float]], *,
        use_tree: bool | None = None) -> Mesh:
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
        boundary when attempting to match the two. Pass at most one mapping for each
        unique (order-independent) pair of boundaries.
    """
    if any(grp.vertex_indices is None for grp in mesh.groups):
        raise ValueError(
            "gluing mesh boundaries requires 'vertex_indices' in all groups")

    if use_tree is not None:
        warn("Passing 'use_tree' is deprecated and will be removed "
             "in Q3 2025.", DeprecationWarning, stacklevel=2)

    glued_btags = {
        btag
        for mapping, _ in bdry_pair_mappings_and_tols
        for btag in (mapping.from_btag, mapping.to_btag)}

    btag_to_index = {btag: i for i, btag in enumerate(glued_btags)}

    glued_btag_pairs = set()
    for mapping, _ in bdry_pair_mappings_and_tols:
        if btag_to_index[mapping.from_btag] < btag_to_index[mapping.to_btag]:
            btag_pair = (mapping.from_btag, mapping.to_btag)
        else:
            btag_pair = (mapping.to_btag, mapping.from_btag)

        if btag_pair in glued_btag_pairs:
            raise ValueError(
                "multiple mappings detected for boundaries "
                f"{btag_pair[0]} and {btag_pair[1]}.")

        glued_btag_pairs.add(btag_pair)

    face_id_pairs_for_mapping = [
        _match_boundary_faces(mesh, mapping, tol)
        for mapping, tol in bdry_pair_mappings_and_tols]

    facial_adjacency_groups = []

    for igrp, old_fagrp_list in enumerate(mesh.facial_adjacency_groups):
        fagrp_list = [
            fagrp for fagrp in old_fagrp_list
            if not isinstance(fagrp, BoundaryAdjacencyGroup)
            or fagrp.boundary_tag not in glued_btags]

        for imap, (mapping, _) in enumerate(bdry_pair_mappings_and_tols):
            bdry_m_face_ids, bdry_n_face_ids = face_id_pairs_for_mapping[imap]
            bdry_m_belongs_to_group = bdry_m_face_ids.groups == igrp
            bdry_n_belongs_to_group = bdry_n_face_ids.groups == igrp
            for ineighbor_grp in range(len(mesh.groups)):
                bdry_m_indices, = np.where(
                    bdry_m_belongs_to_group
                    & (bdry_n_face_ids.groups == ineighbor_grp))
                bdry_n_indices, = np.where(
                    bdry_n_belongs_to_group
                    & (bdry_m_face_ids.groups == ineighbor_grp))
                if len(bdry_m_indices) > 0:
                    elements = bdry_m_face_ids.elements[bdry_m_indices]
                    element_faces = bdry_m_face_ids.faces[bdry_m_indices]
                    neighbors = bdry_n_face_ids.elements[bdry_m_indices]
                    neighbor_faces = bdry_n_face_ids.faces[bdry_m_indices]
                    fagrp_list.append(InteriorAdjacencyGroup(
                        igroup=igrp,
                        ineighbor_group=ineighbor_grp,
                        elements=elements,
                        element_faces=element_faces,
                        neighbors=neighbors,
                        neighbor_faces=neighbor_faces,
                        aff_map=mapping.aff_map))
                if len(bdry_n_indices) > 0:
                    elements = bdry_n_face_ids.elements[bdry_n_indices]
                    element_faces = bdry_n_face_ids.faces[bdry_n_indices]
                    neighbors = bdry_m_face_ids.elements[bdry_n_indices]
                    neighbor_faces = bdry_m_face_ids.faces[bdry_n_indices]
                    fagrp_list.append(InteriorAdjacencyGroup(
                        igroup=igrp,
                        ineighbor_group=ineighbor_grp,
                        elements=elements,
                        element_faces=element_faces,
                        neighbors=neighbors,
                        neighbor_faces=neighbor_faces,
                        aff_map=mapping.aff_map.inverted()))

        facial_adjacency_groups.append(fagrp_list)

    return mesh.copy(
        nodal_adjacency=False,
        _facial_adjacency_groups=tuple(
            tuple(fagrps) for fagrps in facial_adjacency_groups),
        )

# }}}


# {{{ map

def map_mesh(mesh: Mesh, f: Callable[[np.ndarray], np.ndarray]) -> Mesh:
    """Apply the map *f* to the mesh. *f* needs to accept and return arrays of
    shape ``(ambient_dim, npoints)``."""

    if mesh._facial_adjacency_groups is not None:
        has_adj_maps = any(
            fagrp.aff_map.matrix is not None or fagrp.aff_map.offset is not None
            for fagrp_list in mesh.facial_adjacency_groups
            for fagrp in fagrp_list if hasattr(fagrp, "aff_map")
            )
        if has_adj_maps:
            raise ValueError("cannot apply a general map to a mesh that has "
                "affine mappings in its facial adjacency. If the map is affine, "
                "use affine_map instead")

    if mesh.vertices is not None:
        vertices = f(mesh.vertices)
        if not vertices.flags.c_contiguous:
            vertices = np.copy(vertices, order="C")
    else:
        vertices = None

    # {{{ assemble new groups list

    new_groups = []
    for group in mesh.groups:
        mapped_nodes = f(group.nodes.reshape(mesh.ambient_dim, -1))
        if not mapped_nodes.flags.c_contiguous:
            mapped_nodes = np.copy(mapped_nodes, order="C")

        new_groups.append(
            replace(group, nodes=mapped_nodes.reshape(*group.nodes.shape))
            )

    # }}}

    return mesh.copy(
            vertices=vertices,
            groups=tuple(new_groups),
            is_conforming=mesh.is_conforming)

# }}}


# {{{ affine map

def affine_map(
        mesh: Mesh,
        A: np.generic | np.ndarray | None = None,
        b: np.generic | np.ndarray | None = None) -> Mesh:
    """Apply the affine map :math:`f(x) = A x + b` to the geometry of *mesh*."""

    if A is not None and not isinstance(A, np.ndarray):
        A = np.diag([A] * mesh.ambient_dim)

    if b is not None and not isinstance(b, np.ndarray):
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

        new_groups.append(
            replace(group, nodes=mapped_nodes.reshape(*group.nodes.shape))
            )

    # }}}

    # {{{ assemble new facial adjacency groups

    if mesh._facial_adjacency_groups is not None:
        # For a facial adjacency transform T(x) = Gx + h in the original mesh,
        # its corresponding transform in the new mesh will be (T')(x) = G'x + h',
        # where:
        # G' = G
        # h' = Ah + (I - G)b
        def compute_new_map(old_map: AffineMap) -> AffineMap:
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
                if isinstance(old_fagrp,
                              InteriorAdjacencyGroup | InterPartAdjacencyGroup):
                    new_fagrp: FacialAdjacencyGroup = replace(
                        old_fagrp, aff_map=compute_new_map(old_fagrp.aff_map))
                else:
                    assert not hasattr(old_fagrp, "aff_map")
                    new_fagrp = old_fagrp

                fagrp_list.append(new_fagrp)
            facial_adjacency_groups.append(fagrp_list)

    else:
        facial_adjacency_groups = None

    # }}}

    return mesh.copy(
            vertices=vertices,
            groups=tuple(new_groups),
            _facial_adjacency_groups=tuple(
                tuple(fagrps)
                for fagrps in facial_adjacency_groups)
            if facial_adjacency_groups is not None else None,
            is_conforming=mesh.is_conforming)


def _get_rotation_matrix_from_angle_and_axis(
        theta: float, axis: np.ndarray) -> np.ndarray:
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


def rotate_mesh_around_axis(
        mesh: Mesh, *,
        theta: float,
        axis: np.ndarray | None = None) -> Mesh:
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


# {{{ make_mesh_grid

def make_mesh_grid(
        mesh: Mesh, *,
        shape: tuple[int, ...],
        offset: tuple[np.ndarray, ...] | None = None,
        skip_tests: bool = False) -> Mesh:
    """Constructs a grid of copies of *mesh*, with *shape* copies in each
    dimensions at the given *offset*.

    :returns: a merged mesh representing the grid.
    """

    if len(shape) != mesh.ambient_dim:
        raise ValueError("grid shape length must match mesh ambient dimension")

    if offset is None:
        bmin, bmax = find_bounding_box(mesh)

        from pytools import wandering_element
        size = bmax - bmin
        offset = tuple(np.array(e_i) * (size[i] + 0.25 * size[i])
            for i, e_i in enumerate(wandering_element(mesh.ambient_dim)))

    if len(offset) != mesh.ambient_dim:
        raise ValueError("must provide an offset per dimension")

    if not all(o.size == mesh.ambient_dim for o in offset):
        raise ValueError("offsets must have the mesh dimension")

    from itertools import product
    meshes = []

    for index in product(*(range(n) for n in shape)):
        b = sum((i * o for i, o in zip(index, offset, strict=True)), offset[0])
        meshes.append(affine_map(mesh, b=b))

    return merge_disjoint_meshes(meshes, skip_tests=skip_tests)

# }}}


# {{{ remove_unused_vertices

def remove_unused_vertices(mesh: Mesh) -> Mesh:
    if mesh.vertices is None:
        raise ValueError("mesh must have vertices")

    def not_none(vi: np.ndarray | None) -> np.ndarray:
        if vi is None:
            raise ValueError("mesh element groups must have vertex indices")
        return vi

    used_vertices = np.unique(np.sort(np.concatenate([
        not_none(grp.vertex_indices).reshape(-1)
        for grp in mesh.groups
    ])))

    used_flags: np.ndarray = np.zeros(mesh.nvertices, dtype=np.bool_)
    used_flags[used_vertices] = 1
    new_vertex_indices = np.cumsum(used_flags, dtype=mesh.vertex_id_dtype) - 1
    new_vertex_indices[~used_flags] = -1

    return mesh.copy(
        vertices=mesh.vertices[:, used_flags],
        groups=tuple(
            replace(grp, vertex_indices=new_vertex_indices[grp.vertex_indices])
            for grp in mesh.groups
        ))

# }}}

# vim: foldmethod=marker
