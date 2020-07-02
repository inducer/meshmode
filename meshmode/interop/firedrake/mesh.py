arg = "Copyright (C) 2020 Benjamin Sepanski"

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

__doc__ = """
.. autofunction:: import_firedrake_mesh
"""

from warnings import warn  # noqa
import numpy as np
import six

from modepy import resampling_matrix

from meshmode.mesh import (BTAG_ALL, BTAG_REALLY_ALL, FacialAdjacencyGroup,
    Mesh, NodalAdjacency, SimplexElementGroup)
from meshmode.interop.firedrake.reference_cell import (
    get_affine_reference_simplex_mapping, get_finat_element_unit_nodes)


# {{{ functions to extract information from Mesh Topology


def _get_firedrake_nodal_info(fdrake_mesh_topology):
    """
    Get nodal adjacency and vertex indices corresponding
    to a firedrake mesh topology. Note that as we do not use
    geometric information, there is no guarantee that elements
    have a positive orientation.

    The elements (in firedrake lingo, the cells)
    are guaranteed to have the same numbering in :mod:`meshmode`
    as :mod:`firedrdake`

    :arg fdrake_mesh_topology: A :mod:`firedrake` instance of class
        :class:`MeshTopology` or :class:`MeshGeometry`.

    :return: Returns *vertex_indices* as a numpy array of shape
        *(nelements, ref_element.nvertices)* (as described by
        the ``vertex_indices`` attribute of a :class:`MeshElementGroup`)
        and a :class:`NodalAdjacency` constructed from
        *fdrake_mesh_topology*
        as a tuple *(vertex_indices, nodal_adjacency)*.
    """
    top = fdrake_mesh_topology.topology

    # If you don't understand dmplex, look at the PETSc reference
    # here: https://cse.buffalo.edu/~knepley/classes/caam519/CSBook.pdf
    # used to get topology info
    # FIXME... not sure how to get around the private access
    dm = top._topology_dm

    # Get range of dmplex ids for cells, facets, and vertices
    c_start, c_end = dm.getHeightStratum(0)
    f_start, f_end = dm.getHeightStratum(1)
    v_start, v_end = dm.getDepthStratum(0)

    # FIXME... not sure how to get around the private accesses
    # Maps dmplex cell id -> firedrake cell index
    def cell_id_dmp_to_fd(ndx):
        return top._cell_numbering.getOffset(ndx)

    # Maps dmplex vert id -> firedrake vert index
    def vert_id_dmp_to_fd(ndx):
        return top._vertex_numbering.getOffset(ndx)

    # We will fill in the values as we go
    vertex_indices = -np.ones((top.num_cells(), top.ufl_cell().num_vertices()),
                              dtype=np.int32)
    # This will map fd cell ndx -> list of fd cell indices which share a vertex
    cell_to_nodal_neighbors = {}
    # This will map dmplex facet id -> list of adjacent
    #                                  (fd cell ndx, firedrake local fac num)
    facet_to_cells = {}
    # This will map dmplex vert id -> list of fd cell
    #                                 indices which touch this vertex,
    # Primarily used to construct cell_to_nodal_neighbors
    vert_to_cells = {}

    # Loop through each cell (cell closure is all the dmplex ids for any
    # verts, faces, etc. associated with the cell)
    for fd_cell_ndx, closure_dmp_ids in enumerate(top.cell_closure):
        # Store the vertex indices
        dmp_verts = closure_dmp_ids[np.logical_and(v_start <= closure_dmp_ids,
                                                   closure_dmp_ids < v_end)]
        fd_verts = np.array([vert_id_dmp_to_fd(dmp_vert)
                             for dmp_vert in dmp_verts])
        vertex_indices[fd_cell_ndx][:] = fd_verts[:]

        # Record this cell as touching the facet and remember its local
        # facet number (the order it appears)
        dmp_fac_ids = closure_dmp_ids[np.logical_and(f_start <= closure_dmp_ids,
                                                     closure_dmp_ids < f_end)]
        for loc_fac_nr, dmp_fac_id in enumerate(dmp_fac_ids):
            # make sure there is a list to append to and append
            facet_to_cells.setdefault(dmp_fac_id, [])
            facet_to_cells[dmp_fac_id].append((fd_cell_ndx, loc_fac_nr))

        # Record this vertex as touching the cell, and mark this cell
        # as nodally adjacent (in cell_to_nodal_neighbors) to any
        # cells already documented as touching this cell
        cell_to_nodal_neighbors[fd_cell_ndx] = []
        for dmp_vert_id in dmp_verts:
            vert_to_cells.setdefault(dmp_vert_id, [])
            for other_cell_ndx in vert_to_cells[dmp_vert_id]:
                cell_to_nodal_neighbors[fd_cell_ndx].append(other_cell_ndx)
                cell_to_nodal_neighbors[other_cell_ndx].append(fd_cell_ndx)
            vert_to_cells[dmp_vert_id].append(fd_cell_ndx)

    # Next go ahead and compute nodal adjacency by creating
    # neighbors and neighbor_starts as specified by :class:`NodalAdjacency`
    neighbors = []
    neighbors_starts = np.zeros(top.num_cells() + 1, dtype=np.int32)
    for iel in range(len(cell_to_nodal_neighbors)):
        neighbors += cell_to_nodal_neighbors[iel]
        neighbors_starts[iel+1] = len(neighbors)

    neighbors = np.array(neighbors, dtype=np.int32)

    nodal_adjacency = NodalAdjacency(neighbors_starts=neighbors_starts,
                                     neighbors=neighbors)

    return (vertex_indices, nodal_adjacency)


def _get_firedrake_boundary_tags(fdrake_mesh):
    """
    Return a tuple of bdy tags as requested in
    the construction of a :mod:`meshmode` :class:`Mesh`

    The tags used are :class:`meshmode.mesh.BTAG_ALL`,
    :class:`meshmode.mesh.BTAG_REALLY_ALL`, and
    any markers in the mesh topology's exterior facets
    (see :attr:`firedrake.mesh.MeshTopology.exterior_facets.unique_markers`)

    :arg fdrake_mesh: A :mod:`firedrake` :class:`MeshTopology` or
        :class:`MeshGeometry`

    :return: A tuple of boundary tags
    """
    bdy_tags = [BTAG_ALL, BTAG_REALLY_ALL]

    unique_markers = fdrake_mesh.topology.exterior_facets.unique_markers
    if unique_markers is not None:
        bdy_tags += list(unique_markers)

    return tuple(bdy_tags)


def _get_firedrake_facial_adjacency_groups(fdrake_mesh_topology):
    """
    Return facial_adjacency_groups corresponding to
    the given firedrake mesh topology. Note that as we do not
    have geometric information, elements may need to be
    flipped later.

    :arg fdrake_mesh_topology: A :mod:`firedrake` instance of class
        :class:`MeshTopology` or :class:`MeshGeometry`.
    :return: A list of maps to :class:`FacialAdjacencyGroup`s as required
        by a :mod:`meshmode` :class:`Mesh`
    """
    top = fdrake_mesh_topology.topology
    # We only need one group
    # for interconnectivity and one for boundary connectivity.
    # The tricky part is moving from firedrake local facet numbering
    # (ordered lexicographically by the vertex excluded from the face)
    # and meshmode's facet ordering: obtained from a simplex element
    # group
    mm_simp_group = SimplexElementGroup(1, None, None,
                                        dim=top.cell_dimension())
    mm_face_vertex_indices = mm_simp_group.face_vertex_indices()
    # map firedrake local face number to meshmode local face number
    fd_loc_fac_nr_to_mm = {}
    # Figure out which vertex is excluded to get the corresponding
    # firedrake local index
    for mm_loc_fac_nr, face in enumerate(mm_face_vertex_indices):
        for fd_loc_fac_nr in range(top.ufl_cell().num_vertices()):
            if fd_loc_fac_nr not in face:
                fd_loc_fac_nr_to_mm[fd_loc_fac_nr] = mm_loc_fac_nr
                break

    # First do the interconnectivity group

    # Get the firedrake cells associated to each interior facet
    int_facet_cell = top.interior_facets.facet_cell
    # Get the firedrake local facet numbers and map them to the
    # meshmode local facet numbers
    int_fac_loc_nr = top.interior_facets.local_facet_dat.data
    int_fac_loc_nr = \
        np.array([[fd_loc_fac_nr_to_mm[fac_nr] for fac_nr in fac_nrs]
                  for fac_nrs in int_fac_loc_nr])
    # elements neighbors element_faces neighbor_faces are as required
    # for a :class:`FacialAdjacencyGroup`.

    int_elements = int_facet_cell.flatten()
    int_neighbors = np.concatenate((int_facet_cell[:, 1], int_facet_cell[:, 0]))
    int_element_faces = int_fac_loc_nr.flatten().astype(Mesh.face_id_dtype)
    int_neighbor_faces = np.concatenate((int_fac_loc_nr[:, 1],
                                        int_fac_loc_nr[:, 0]))
    int_neighbor_faces = int_neighbor_faces.astype(Mesh.face_id_dtype)

    interconnectivity_grp = FacialAdjacencyGroup(igroup=0, ineighbor_group=0,
                                                 elements=int_elements,
                                                 neighbors=int_neighbors,
                                                 element_faces=int_element_faces,
                                                 neighbor_faces=int_neighbor_faces)

    # Now look at exterior facets

    # We can get the elements directly from exterior facets
    ext_elements = top.exterior_facets.facet_cell.flatten()

    ext_element_faces = top.exterior_facets.local_facet_dat.data
    ext_element_faces = ext_element_faces.astype(Mesh.face_id_dtype)
    ext_neighbor_faces = np.zeros(ext_element_faces.shape, dtype=np.int32)
    ext_neighbor_faces = ext_neighbor_faces.astype(Mesh.face_id_dtype)

    # Now we need to tag the boundary
    bdy_tags = _get_firedrake_boundary_tags(top)
    boundary_tag_to_index = {bdy_tag: i for i, bdy_tag in enumerate(bdy_tags)}

    def boundary_tag_bit(boundary_tag):
        try:
            return 1 << boundary_tag_to_index[boundary_tag]
        except KeyError:
            raise 0

    ext_neighbors = np.zeros(ext_elements.shape, dtype=np.int32)
    for ifac, marker in enumerate(top.exterior_facets.markers):
        ext_neighbors[ifac] = -(boundary_tag_bit(BTAG_ALL)
                                | boundary_tag_bit(BTAG_REALLY_ALL)
                                | boundary_tag_bit(marker))

    exterior_grp = FacialAdjacencyGroup(igroup=0, ineighbor=None,
                                        elements=ext_elements,
                                        element_faces=ext_element_faces,
                                        neighbors=ext_neighbors,
                                        neighbor_faces=ext_neighbor_faces)

    return [{0: interconnectivity_grp, None: exterior_grp}]

# }}}


# {{{ Orientation computation

def _get_firedrake_orientations(fdrake_mesh, unflipped_group, vertices,
                                normals=None, no_normals_warn=True):
    """
    Return the orientations of the mesh elements:

    :arg fdrake_mesh: A :mod:`firedrake` instance of :class:`MeshGeometry`
    :arg unflipped_group: A :class:`SimplexElementGroup` instance with
        (potentially) some negatively oriented elements.
    :arg vertices: The vertex coordinates as a numpy array of shape
        *(ambient_dim, nvertices)*
    :arg normals: As described in the kwargs of :func:`import_firedrake_mesh`
    :arg no_normals_warn: As described in the kwargs of
        :func:`import_firedrake_mesh`

    :return: A numpy array, the *i*th element is > 0 if the *ith* element
        is positively oriented, < 0 if negatively oriented.
        Mesh must have co-dimension 0 or 1.
    """
    # compute orientations
    tdim = fdrake_mesh.topological_dimension()
    gdim = fdrake_mesh.geometric_dimension()

    orient = None
    if gdim == tdim:
        # We use :mod:`meshmode` to check our orientations
        from meshmode.mesh.processing import \
            find_volume_mesh_element_group_orientation

        orient = find_volume_mesh_element_group_orientation(vertices,
                                                            unflipped_group)

    if tdim == 1 and gdim == 2:
        # In this case we have a 1-surface embedded in 2-space
        orient = np.ones(fdrake_mesh.num_cells())
        if normals:
            for i, (normal, vertices) in enumerate(zip(np.array(normals),
                                                       vertices)):
                if np.cross(normal, vertices) < 0:
                    orient[i] = -1.0
        elif no_normals_warn:
            warn("Assuming all elements are positively-oriented.")

    elif tdim == 2 and gdim == 3:
        # In this case we have a 2-surface embedded in 3-space
        orient = fdrake_mesh.cell_orientations().dat.data
        r"""
            Convert (0 \implies negative, 1 \implies positive) to
            (-1 \implies negative, 1 \implies positive)
        """
        orient *= 2
        orient -= np.ones(orient.shape, dtype=orient.dtype)
    #Make sure the mesh fell into one of the cases
    """
    NOTE : This should be guaranteed by previous checks,
           but is here anyway in case of future development.
    """
    assert orient is not None, "something went wrong, contact the developer"
    return orient

# }}}


# {{{ Mesh importing from firedrake

def import_firedrake_mesh(fdrake_mesh, normals=None, no_normals_warn=None):
    """
    Create a :mod:`meshmode` :class:`Mesh` from a :mod:`firedrake`
    :class:`MeshGeometry` with the same cells/elements, vertices, nodes,
    mesh order, and facial adjacency.

    The vertex and node coordinates will be the same, as well
    as the cell/element ordering. However, :mod:`firedrake`
    does not require elements to be positively oriented,
    so any negative elements are flipped
    as in :func:`meshmode.processing.flip_simplex_element_group`.

    The flipped cells/elements are identified by the returned
    *firedrake_orient* array

    :arg fdrake_mesh: A :mod:`firedrake` :class:`MeshGeometry`.
        This mesh **must** be in a space of ambient dimension
        1, 2, or 3 and have co-dimension of 0 or 1.
        It must use a simplex as a reference element.

        In the case of a 2-dimensional mesh embedded in 3-space,
        the method ``fdrake_mesh.init_cell_orientations`` must
        have been called.

        In the case of a 1-dimensional mesh embedded in 2-space,
        see parameters *normals* and *no_normals_warn*.

        Finally, the ``coordinates`` attribute must have a function
        space whose *finat_element* associates a degree
        of freedom with each vertex. In particular,
        this means that the vertices of the mesh must have well-defined
        coordinates.
        For those unfamiliar with :mod:`firedrake`, you can
        verify this by looking at

        .. code-block:: python

            import six
            coords_fspace = fdrake_mesh.coordinates.function_space()
            vertex_entity_dofs = coords_fspace.finat_element.entity_dofs()[0]
            for entity, dof_list in six.iteritems(vertex_entity_dofs):
                assert len(dof_list) > 0
    :arg normals: **Only** used if *fdrake_mesh* is a 1-surface
        embedded in 2-space. In this case,
            - If *None* then
              all elements are assumed to be positively oriented.
            - Else, should be a list/array whose *i*th entry
              is the normal for the *i*th element (*i*th
              in *mesh.coordinate.function_space()*'s
              :attr:`cell_node_list`)
    :arg no_normals_warn: If *True* (the default), raises a warning
        if *fdrake_mesh* is a 1-surface embedded in 2-space
        and *normals* is *None*.

    :return: A tuple *(meshmode mesh, firedrake_orient)*.
         ``firedrake_orient < 0`` is *True* for any negatively
         oriented firedrake cell (which was flipped by meshmode)
         and False for any positively oriented firedrake cell
         (whcih was not flipped by meshmode).
    """
    # Type validation
    from firedrake.mesh import MeshGeometry
    if not isinstance(fdrake_mesh, MeshGeometry):
        raise TypeError(":arg:`fdrake_mesh_topology` must be a "
                        ":mod:`firedrake` :class:`MeshGeometry`, "
                        "not %s." % type(fdrake_mesh))
    assert fdrake_mesh.ufl_cell().is_simplex(), "Mesh must use simplex cells"
    gdim = fdrake_mesh.geometric_dimension()
    tdim = fdrake_mesh.topological_dimension()
    assert gdim in [1, 2, 3], "Mesh must be in space of ambient dim 1, 2, or 3"
    assert gdim - tdim in [0, 1], "Mesh co-dimension must be 0 or 1"
    fdrake_mesh.init()

    # Get all the nodal information we can from the topology
    bdy_tags = _get_firedrake_boundary_tags(fdrake_mesh)
    vertex_indices, nodal_adjacency = _get_firedrake_nodal_info(fdrake_mesh)

    # Grab the mesh reference element and cell dimension
    coord_finat_elt = fdrake_mesh.coordinates.function_space().finat_element
    cell_dim = fdrake_mesh.cell_dimension()

    # Get finat unit nodes and map them onto the meshmode reference simplex
    from meshmode.interop.firedrake.reference_cell import (
        get_affine_reference_simplex_mapping, get_finat_element_unit_nodes)
    finat_unit_nodes = get_finat_element_unit_nodes(coord_finat_elt)
    fd_ref_to_mm = get_affine_reference_simplex_mapping(cell_dim, True)
    finat_unit_nodes = fd_ref_to_mm(finat_unit_nodes)

    # Now grab the nodes
    coords = fdrake_mesh.coordinates
    cell_node_list = coords.function_space().cell_node_list
    nodes = np.real(coords.dat.data[cell_node_list])
    # Add extra dim in 1D so that have [nelements][nunit_nodes][dim]
    if len(nodes.shape) == 2:
        nodes = np.reshape(nodes, nodes.shape + (1,))
    # Now we want the nodes to actually have shape [dim][nelements][nunit_nodes]
    nodes = np.transpose(nodes, (2, 0, 1))

    # make a group (possibly with some elements that need to be flipped)
    unflipped_group = SimplexElementGroup(coord_finat_elt.degree,
                                          vertex_indices,
                                          nodes,
                                          dim=cell_dim,
                                          unit_nodes=finat_unit_nodes)

    # Next get the vertices (we'll need these for the orientations)

    coord_finat = fdrake_mesh.coordinates.function_space().finat_element
    # unit_vertex_indices are the element-local indices of the nodes
    # which coincide with the vertices, i.e. for element *i*,
    # vertex 0's coordinates would be nodes[i][unit_vertex_indices[0]].
    # This assumes each vertex has some node which coincides with it...
    # which is normally fine to assume for firedrake meshes.
    unit_vertex_indices = []
    # iterate through the dofs associated to each vertex on the
    # reference element
    for _, dofs in sorted(six.iteritems(coord_finat.entity_dofs()[0])):
        assert len(dofs) == 1, \
            "The function space of the mesh coordinates must have" \
            " exactly one degree of freedom associated with " \
            " each vertex in order to determine vertex coordinates"
        dof, = dofs
        unit_vertex_indices.append(dof)

    # Now get the vertex coordinates
    vertices = {}
    for icell, cell_vertex_indices in enumerate(vertex_indices):
        for local_vert_id, global_vert_id in enumerate(cell_vertex_indices):
            if global_vert_id in vertices:
                continue
            local_node_nr = unit_vertex_indices[local_vert_id]
            vertices[global_vert_id] = nodes[:, icell, local_node_nr]
    # Stuff the vertices in a *(dim, nvertices)*-shaped numpy array
    vertices = np.array([vertices[i] for i in range(len(vertices))]).T

    # Use the vertices to compute the orientations and flip the group
    orient = _get_firedrake_orientations(fdrake_mesh, unflipped_group, vertices,
                                         normals=normals,
                                         no_normals_warn=no_normals_warn)
    from meshmode.mesh.processing import flip_simplex_element_group
    group = flip_simplex_element_group(vertices, unflipped_group, orient < 0)

    # Now, any flipped element had its 0 vertex and 1 vertex exchanged.
    # This changes the local facet nr, so we need to create and then
    # fix our facial adjacency groups. To do that, we need to figure
    # out which local facet numbers switched.
    mm_simp_group = SimplexElementGroup(1, None, None,
                                        dim=fdrake_mesh.cell_dimension())
    face_vertex_indices = mm_simp_group.face_vertex_indices()
    # face indices of the faces not containing vertex 0 and not
    # containing vertex 1, respectively
    no_zero_face_ndx, no_one_face_ndx = None, None
    for iface, face in enumerate(face_vertex_indices):
        if 0 not in face:
            no_zero_face_ndx = iface
        elif 1 not in face:
            no_one_face_ndx = iface

    unflipped_facial_adjacency_groups = \
        _get_firedrake_facial_adjacency_groups(fdrake_mesh)

    def flip_local_face_indices(faces, elements):
        faces = np.copy(faces)
        to_no_one = np.logical_and(orient[elements] < 0,
                                   faces == no_zero_face_ndx)
        to_no_zero = np.logical_and(orient[elements] < 0,
                                    faces == no_one_face_ndx)
        faces[to_no_one], faces[to_no_zero] = no_one_face_ndx, no_zero_face_ndx
        return faces

    facial_adjacency_groups = []
    for igroup, fagrps in enumerate(unflipped_facial_adjacency_groups):
        facial_adjacency_groups.append({})
        for ineighbor_group, fagrp in six.iteritems(fagrps):
            new_element_faces = flip_local_face_indices(fagrp.element_faces,
                                                        fagrp.elements)
            if ineighbor_group is None:
                new_neighbor_faces = fagrp.neighbor_faces
            else:
                new_neighbor_faces = \
                    flip_local_face_indices(fagrp.neighbor_faces,
                                            fagrp.neighbors)
            new_fagrp = FacialAdjacencyGroup(igroup=igroup,
                                             ineighbor_group=ineighbor_group,
                                             elements=fagrp.elements,
                                             element_faces=new_element_faces,
                                             neighbors=fagrp.neighbors,
                                             neighbor_faces=new_neighbor_faces)
            facial_adjacency_groups[igroup][ineighbor_group] = new_fagrp

    return (Mesh(vertices, [group],
                 boundary_tags=bdy_tags,
                 nodal_adjacency=nodal_adjacency,
                 facial_adjacency_groups=facial_adjacency_groups),
            orient)

# }}}


# {{{ Mesh exporting to firedrake

# TODO : Keep boundary tagging
def export_mesh_to_firedrake(mesh, group_nr=None, comm=None):
    """
    Create a firedrake mesh corresponding to one :class:`Mesh`'s
    :class:`SimplexElementGroup`.

    :param mesh: A :class:`meshmode.mesh.Mesh` to convert with
        at least one :class:`SimplexElementGroup`
    :param group_nr: The group number to be converted into a firedrake
        mesh. The corresponding group must be of type
        :class:`SimplexElementGroup`. If *None* and
        *mesh* has only one group, that group is used. Otherwise,
        a *ValueError* is raised.
    :param comm: The communicator to build the dmplex mesh on
    """
    if not isinstance(mesh, Mesh):
        raise TypeError(":arg:`mesh` must of type :class:`meshmode.mesh.Mesh`,"
                        " not %s." % type(mesh))
    if group_nr is None:
        if len(mesh.groups) != 1:
            raise ValueError(":arg:`group_nr` is *None* but :arg:`mesh` has "
                             "more than one group.")
        else:
            group_nr = 0
    assert group_nr >= 0 and group_nr < len(mesh.groups)
    assert isinstance(mesh.groups[group_nr], SimplexElementGroup)
    assert mesh.vertices is not None

    # Get a dmplex object and then a mesh topology
    group = mesh.groups[group_nr]
    mm2fd_indices = np.unique(group.vertex_indices.flatten())
    coords = mesh.vertices[:, mm2fd_indices].T
    cells = mm2fd_indices[group.vertex_indices]

    if comm is None:
        from pyop2.mpi import COMM_WORLD
        comm = COMM_WORLD
    # FIXME : this is a private member...
    import firedrake.mesh as fd_mesh
    plex = fd_mesh._from_cell_list(group.dim, cells, coords, comm)

    # TODO : Allow reordering? This makes the connection slower
    #        but (in principle) firedrake is reordering nodes
    #        to speed up cache access for their computations.
    #        Users might want to be able to choose whether
    #        or not to reorder based on if caching/conversion
    #        is bottle-necking
    topology = fd_mesh.Mesh(plex, reorder=False)

    # Now make a coordinates function
    from firedrake import VectorFunctionSpace, Function
    coords_fspace = VectorFunctionSpace(topology, 'CG', group.order,
                                        dim=mesh.ambient_dim)
    coords = Function(coords_fspace)

    # get firedrake unit nodes and map onto meshmode reference element
    fd_ref_cell_to_mm = get_affine_reference_simplex_mapping(group.dim, True)
    fd_unit_nodes = get_finat_element_unit_nodes(coords_fspace.finat_element)
    fd_unit_nodes = fd_ref_cell_to_mm(fd_unit_nodes)

    from meshmode.discretization.poly_element import (
        InterpolatoryQuadratureSimplexElementGroup)
    el_group = InterpolatoryQuadratureSimplexElementGroup(group, group.order,
                                                          group.node_nr_base)
    resampling_mat = resampling_matrix(el_group.basis(),
                                       new_nodes=fd_unit_nodes,
                                       old_nodes=group.unit_nodes)
    # nodes is shaped *(ambient dim, nelements, nunit nodes)
    coords.dat.data[coords_fspace.cell_node_list, :] = \
        np.matmul(group.nodes, resampling_mat.T).transpose((1, 2, 0))

    return fd_mesh.Mesh(coords, reorder=False)

# }}}

# vim: foldmethod=marker
