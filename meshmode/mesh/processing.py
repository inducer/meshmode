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
.. autofunction:: map_mesh
.. autofunction:: affine_map
"""


def find_group_indices(groups, meshwide_elems):
    """
    :arg groups: A list of :class:``MeshElementGroup`` instances that contain
        ``meshwide_elems``.
    :arg meshwide_elems: A :class:``numpy.ndarray`` of mesh-wide element numbers
        Usually computed by ``elem + element_nr_base``.
    :returns: A :class:``numpy.ndarray`` of group numbers that ``meshwide_elem``
        belongs to.
    """
    grps = np.zeros_like(meshwide_elems)
    next_grp_boundary = 0
    for igrp, grp in enumerate(groups):
        next_grp_boundary += grp.nelements
        grps += meshwide_elems >= next_grp_boundary
    return grps


# {{{ partition_mesh

def partition_mesh(mesh, part_per_element, part_num):
    """
    :arg mesh: A :class:`meshmode.mesh.Mesh` to be partitioned.
    :arg part_per_element: A :class:`numpy.ndarray` containing one
        integer per element of *mesh* indicating which part of the
        partitioned mesh the element is to become a part of.
    :arg part_num: The part number of the mesh to return.

    :returns: A tuple ``(part_mesh, part_to_global)``, where *part_mesh*
        is a :class:`meshmode.mesh.Mesh` that is a partition of mesh, and
        *part_to_global* is a :class:`numpy.ndarray` mapping element
        numbers on *part_mesh* to ones in *mesh*.

    .. versionadded:: 2017.1
    """
    assert len(part_per_element) == mesh.nelements, (
        "part_per_element must have shape (mesh.nelements,)")

    # Contains the indices of the elements requested.
    queried_elems = np.where(np.array(part_per_element) == part_num)[0]

    num_groups = len(mesh.groups)
    new_indices = []
    new_nodes = []

    # The set of vertex indices we need.
    # NOTE: There are two methods for producing required_indices.
    #   Optimizations may come from further exploring these options.
    #index_set = np.array([], dtype=int)
    index_sets = np.array([], dtype=set)

    skip_groups = []
    num_prev_elems = 0
    start_idx = 0
    for group_num in range(num_groups):
        mesh_group = mesh.groups[group_num]

        # Find the index of first element in the next group.
        end_idx = len(queried_elems)
        for idx in range(start_idx, len(queried_elems)):
            if queried_elems[idx] - num_prev_elems >= mesh_group.nelements:
                end_idx = idx
                break

        if start_idx == end_idx:
            skip_groups.append(group_num)
            new_indices.append(np.array([]))
            new_nodes.append(np.array([]))
            num_prev_elems += mesh_group.nelements
            continue

        elems = queried_elems[start_idx:end_idx] - num_prev_elems
        new_indices.append(mesh_group.vertex_indices[elems])

        new_nodes.append(
            np.zeros(
                (mesh.ambient_dim, end_idx - start_idx, mesh_group.nunit_nodes)))
        for i in range(mesh.ambient_dim):
            for j in range(start_idx, end_idx):
                elems = queried_elems[j] - num_prev_elems
                new_idx = j - start_idx
                new_nodes[group_num][i, new_idx, :] = mesh_group.nodes[i, elems, :]

        #index_set = np.append(index_set, new_indices[group_num].ravel())
        index_sets = np.append(index_sets, set(new_indices[group_num].ravel()))

        num_prev_elems += mesh_group.nelements
        start_idx = end_idx

    # A sorted np.array of vertex indices we need (without duplicates).
    #required_indices = np.unique(np.sort(index_set))
    required_indices = np.array(list(set.union(*index_sets)))

    new_vertices = np.zeros((mesh.ambient_dim, len(required_indices)))
    for dim in range(mesh.ambient_dim):
        new_vertices[dim] = mesh.vertices[dim][required_indices]

    # Our indices need to be in range [0, len(mesh.nelements)].
    for group_num in range(num_groups):
        if group_num not in skip_groups:
            for i in range(len(new_indices[group_num])):
                for j in range(len(new_indices[group_num][0])):
                    original_index = new_indices[group_num][i, j]
                    new_indices[group_num][i, j] = np.where(
                            required_indices == original_index)[0]

    new_mesh_groups = []
    for group_num, mesh_group in enumerate(mesh.groups):
        if group_num not in skip_groups:
            new_mesh_groups.append(
                type(mesh_group)(
                    mesh_group.order, new_indices[group_num],
                    new_nodes[group_num],
                    unit_nodes=mesh_group.unit_nodes))

    from meshmode.mesh import BTAG_ALL, BTAG_PARTITION
    boundary_tags = [BTAG_PARTITION(n) for n in np.unique(part_per_element)]

    from meshmode.mesh import Mesh
    part_mesh = Mesh(
            new_vertices,
            new_mesh_groups,
            facial_adjacency_groups=None,
            boundary_tags=boundary_tags,
            is_conforming=mesh.is_conforming)

    adj_data = [[] for _ in range(len(part_mesh.groups))]

    for igrp, grp in enumerate(part_mesh.groups):
        elem_base = grp.element_nr_base
        boundary_adj = part_mesh.facial_adjacency_groups[igrp][None]
        boundary_elems = boundary_adj.elements
        boundary_faces = boundary_adj.element_faces
        p_meshwide_elems = queried_elems[boundary_elems + elem_base]
        parent_igrps = find_group_indices(mesh.groups, p_meshwide_elems)
        for adj_idx, elem in enumerate(boundary_elems):
            face = boundary_faces[adj_idx]
            tag = -boundary_adj.neighbors[adj_idx]
            assert tag >= 0, "Expected boundary tag in adjacency group."

            parent_igrp = parent_igrps[adj_idx]
            parent_elem_base = mesh.groups[parent_igrp].element_nr_base
            parent_elem = p_meshwide_elems[adj_idx] - parent_elem_base

            parent_adj = mesh.facial_adjacency_groups[parent_igrp]

            for parent_facial_group in parent_adj.values():
                indices, = np.nonzero(parent_facial_group.elements == parent_elem)
                for idx in indices:
                    if (parent_facial_group.neighbors[idx] >= 0 and
                               parent_facial_group.element_faces[idx] == face):
                        rank_neighbor = (parent_facial_group.neighbors[idx]
                                            + parent_elem_base)
                        n_face = parent_facial_group.neighbor_faces[idx]

                        n_part_num = part_per_element[rank_neighbor]
                        tag = tag & ~part_mesh.boundary_tag_bit(BTAG_ALL)
                        tag = tag | part_mesh.boundary_tag_bit(
                                                    BTAG_PARTITION(n_part_num))
                        boundary_adj.neighbors[adj_idx] = -tag

                        # Find the neighbor element from the other partition.
                        n_meshwide_elem = np.count_nonzero(
                                    part_per_element[:rank_neighbor] == n_part_num)

                        adj_data[igrp].append((elem, face,
                                               n_part_num, n_meshwide_elem, n_face))

    connected_mesh = part_mesh.copy()

    from meshmode.mesh import InterPartitionAdjacencyGroup
    for igrp, adj in enumerate(adj_data):
        if adj:
            bdry = connected_mesh.facial_adjacency_groups[igrp][None]
            # Initialize connections
            n_parts = np.zeros_like(bdry.elements)
            n_parts.fill(-1)
            global_n_elems = np.copy(n_parts)
            n_faces = np.copy(n_parts)

            # Sort both sets of elements so that we can quickly merge
            # the two data structures
            bdry_perm = np.lexsort([bdry.element_faces, bdry.elements])
            elems = bdry.elements[bdry_perm]
            faces = bdry.element_faces[bdry_perm]
            neighbors = bdry.neighbors[bdry_perm]
            adj_elems, adj_faces, adj_n_parts, adj_gl_n_elems, adj_n_faces =\
                                    np.array(adj).T
            adj_perm = np.lexsort([adj_faces, adj_elems])
            adj_elems = adj_elems[adj_perm]
            adj_faces = adj_faces[adj_perm]
            adj_n_parts = adj_n_parts[adj_perm]
            adj_gl_n_elems = adj_gl_n_elems[adj_perm]
            adj_n_faces = adj_n_faces[adj_perm]

            # Merge interpartition adjacency data with FacialAdjacencyGroup
            adj_idx = 0
            for bdry_idx in range(len(elems)):
                if adj_idx >= len(adj_elems):
                    break
                if (adj_elems[adj_idx] == elems[bdry_idx]
                        and adj_faces[adj_idx] == faces[bdry_idx]):
                    n_parts[bdry_idx] = adj_n_parts[adj_idx]
                    global_n_elems[bdry_idx] = adj_gl_n_elems[adj_idx]
                    n_faces[bdry_idx] = adj_n_faces[adj_idx]
                    adj_idx += 1

            connected_mesh.facial_adjacency_groups[igrp][None] =\
                    InterPartitionAdjacencyGroup(elements=elems,
                                                 element_faces=faces,
                                                 neighbors=neighbors,
                                                 igroup=bdry.igroup,
                                                 ineighbor_group=None,
                                                 neighbor_partitions=n_parts,
                                                 global_neighbors=global_n_elems,
                                                 neighbor_faces=n_faces)

    return connected_mesh, queried_elems

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
        check is available will return NaN.
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
    :arg flip_flags: A :class:`numpy.ndarray` with *mesh.nelements* entries
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
