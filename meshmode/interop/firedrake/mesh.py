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

from warnings import warn  # noqa
import logging
import numpy as np

from modepy import resampling_matrix, simplex_best_available_basis

from meshmode.mesh import (BTAG_ALL, BTAG_REALLY_ALL, BTAG_INDUCED_BOUNDARY,
    FacialAdjacencyGroup, Mesh, NodalAdjacency, SimplexElementGroup)
from meshmode.interop.firedrake.reference_cell import (
    get_affine_reference_simplex_mapping, get_finat_element_unit_nodes)

from pytools import ProcessLogger

__doc__ = """
.. autofunction:: import_firedrake_mesh
.. autofunction:: export_mesh_to_firedrake
"""


logger = logging.getLogger(__name__)


# {{{ functions to extract information from Mesh Topology


def _get_firedrake_nodal_info(fdrake_mesh_topology, cells_to_use=None):
    """
    Get nodal adjacency and vertex indices corresponding
    to a firedrake mesh topology. Note that as we do not use
    geometric information, there is no guarantee that elements
    have a positive orientation.

    The elements (in firedrake lingo, the cells)
    are guaranteed to have the same numbering in :mod:`meshmode`
    as :mod:`firedrdake`

    :arg fdrake_mesh_topology: A :mod:`firedrake` instance of class
        `firedrake.mesh.MeshTopology` or `firedrake.mesh.MeshGeometry`.

    :arg cells_to_use: Ignored if *None*. Otherwise, assumed to be
        a numpy array holding the firedrake cell indices to use.
        Any cells not on this array are ignored.
        Cells used are assumed to appear exactly once in the array.
        The cell's element index in the nodal adjacency will be its
        index in *cells_to_use*.

    :return: Returns *vertex_indices* as a numpy array of shape
        *(nelements, ref_element.nvertices)* (as described by
        the ``vertex_indices`` attribute of a :class:`MeshElementGroup`)
        and a :class:`NodalAdjacency` constructed from
        *fdrake_mesh_topology*
        as a tuple *(vertex_indices, nodal_adjacency)*.

        Even if *cells_to_use* is not *None*, the vertex indices
        are still the global firedrake vertex indices.
    """
    top = fdrake_mesh_topology.topology

    # If you don't understand dmplex, look at the PETSc reference
    # here: https://cse.buffalo.edu/~knepley/classes/caam519/CSBook.pdf
    # used to get topology info
    top_dm = top.topology_dm

    # Get range of dmplex ids for cells, facets, and vertices
    f_start, f_end = top_dm.getHeightStratum(1)
    v_start, v_end = top_dm.getDepthStratum(0)

    # FIXME : not sure how to get around the private accesses
    # Maps dmplex vert id -> firedrake vert index
    vert_id_dmp_to_fd = top._vertex_numbering.getOffset

    # We will fill in the values of vertex indices as we go
    if cells_to_use is None:
        num_cells = top.num_cells()
    else:
        num_cells = np.size(cells_to_use)
    from pyop2.datatypes import IntType
    vertex_indices = -np.ones((num_cells, top.ufl_cell().num_vertices()),
                              dtype=IntType)
    # This will map fd cell ndx (or its new index as dictated by
    #                            *cells_to_use* if *cells_to_use*
    #                            is not *None*)
    # -> list of fd cell indices which share a vertex
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
    cell_closure = top.cell_closure
    if cells_to_use is not None:
        cell_closure = cell_closure[cells_to_use, :]
    for fd_cell_ndx, closure_dmp_ids in enumerate(cell_closure):
        # Store the vertex indices
        dmp_verts = closure_dmp_ids[np.logical_and(v_start <= closure_dmp_ids,
                                                   closure_dmp_ids < v_end)]
        vertex_indices[fd_cell_ndx][:] = np.array([
            vert_id_dmp_to_fd(dmp_vert) for dmp_vert in dmp_verts])

        # Record this cell as touching the facet and remember its local
        # facet number (the index at which it appears)
        dmp_fac_ids = closure_dmp_ids[np.logical_and(f_start <= closure_dmp_ids,
                                                     closure_dmp_ids < f_end)]
        for loc_fac_nr, dmp_fac_id in enumerate(dmp_fac_ids):
            facet_to_cells.setdefault(dmp_fac_id, []).append((fd_cell_ndx,
                                                              loc_fac_nr))

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

    # make sure that no -1s remain in vertex_indices (i.e. none are left unset)
    assert np.all(vertex_indices >= 0)

    # Next go ahead and compute nodal adjacency by creating
    # neighbors and neighbor_starts as specified by :class:`NodalAdjacency`
    neighbors = []
    if cells_to_use is None:
        num_cells = top.num_cells()
    else:
        num_cells = np.size(cells_to_use)
    neighbors_starts = np.zeros(num_cells + 1, dtype=IntType)
    for iel in range(len(cell_to_nodal_neighbors)):
        neighbors += cell_to_nodal_neighbors[iel]
        neighbors_starts[iel+1] = len(neighbors)

    neighbors = np.array(neighbors, dtype=IntType)

    nodal_adjacency = NodalAdjacency(neighbors_starts=neighbors_starts,
                                     neighbors=neighbors)

    return vertex_indices, nodal_adjacency


def _get_firedrake_boundary_tags(fdrake_mesh, tag_induced_boundary=False):
    """
    Return a tuple of bdy tags as requested in
    the construction of a :mod:`meshmode` :class:`Mesh`

    The tags used are :class:`meshmode.mesh.BTAG_ALL`,
    :class:`meshmode.mesh.BTAG_REALLY_ALL`, and
    any markers in the mesh topology's exterior facets
    (see :attr:`firedrake.mesh.MeshTopology.exterior_facets.unique_markers`)

    :arg fdrake_mesh: A `firedrake.mesh.MeshTopology` or
        `firedrake.mesh.MeshGeometry`
    :arg tag_induced_boundary: If *True*, tag induced boundary with
        :class:`~meshmode.mesh.BTAG_INDUCED_BOUNDARY`

    :return: A tuple of boundary tags
    """
    bdy_tags = [BTAG_ALL, BTAG_REALLY_ALL]
    if tag_induced_boundary:
        bdy_tags.append(BTAG_INDUCED_BOUNDARY)

    unique_markers = fdrake_mesh.topology.exterior_facets.unique_markers
    if unique_markers is not None:
        bdy_tags.extend(unique_markers)

    return bdy_tags


def _get_firedrake_facial_adjacency_groups(fdrake_mesh_topology,
                                           cells_to_use=None):
    """
    Return facial_adjacency_groups corresponding to
    the given firedrake mesh topology. Note that as we do not
    have geometric information, elements may need to be
    flipped later.

    :arg fdrake_mesh_topology: A :mod:`firedrake` instance of class
        `firedrake.mesh.MeshTopology` or `firedrake.mesh.MeshGeometry`.
    :arg cells_to_use: If *None*, then this argument is ignored.
        Otherwise, assumed to be a numpy array of unique firedrake
        cell ids indicating which cells of the mesh to include,
        as well as inducing a new cell index for those cells.
        Also, in this case boundary faces are tagged
        with :class:`meshmode.mesh.BTAG_INDUCED_BOUNDARY`
        if they are not a boundary face in *fdrake_mesh_topology* but become a
        boundary because the opposite cell is not in *cells_to_use*.  Boundary
        faces in *fdrake_mesh_topology* are marked with :class:`BTAG_ALL`. Both
        are marked with :class:`BTAG_REALLY_ALL`.

    :return: A list of maps to :class:`FacialAdjacencyGroup`s as required
        by a :mod:`meshmode` :class:`Mesh`.
    """
    top = fdrake_mesh_topology.topology
    # We only need one group
    # for interconnectivity and one for boundary connectivity.
    # The tricky part is moving from firedrake local facet numbering
    # (ordered lexicographically by the vertex excluded from the face,
    #  search for "local facet number" in the following paper for
    #  a reference on this...
    # https://spiral.imperial.ac.uk/bitstream/10044/1/28819/2/mlange-firedrake-dmplex-accepted.pdf  # noqa : E501
    # )
    # and meshmode's facet ordering: obtained from a simplex element
    # group
    mm_simp_group = SimplexElementGroup(1, None, None,
                                        dim=top.cell_dimension())
    mm_face_vertex_indices = mm_simp_group.face_vertex_indices()
    # map firedrake local face number to meshmode local face number
    fd_loc_fac_nr_to_mm = {}
    # Figure out which vertex is excluded to get the corresponding
    # firedrake local index
    all_local_facet_nrs = set(range(top.ufl_cell().num_vertices()))
    for mm_local_facet_nr, face in enumerate(mm_face_vertex_indices):
        fd_local_facet_nr = all_local_facet_nrs - set(face)
        assert len(fd_local_facet_nr) == 1
        (fd_local_facet_nr,) = fd_local_facet_nr  # extract id from set({id})
        fd_loc_fac_nr_to_mm[fd_local_facet_nr] = mm_local_facet_nr

    # build a look-up table from firedrake markers to the appropriate values
    # in the neighbors array for the external and internal facial adjacency
    # groups
    bdy_tags = _get_firedrake_boundary_tags(
        top, tag_induced_boundary=cells_to_use is not None)
    boundary_tag_to_index = {bdy_tag: i for i, bdy_tag in enumerate(bdy_tags)}
    marker_to_neighbor_value = {}
    from meshmode.mesh import _boundary_tag_bit
    # for convenience,
    # None maps to the boundary tag for a boundary facet with no marker
    marker_to_neighbor_value[None] = \
        -(_boundary_tag_bit(bdy_tags, boundary_tag_to_index, BTAG_REALLY_ALL)
          | _boundary_tag_bit(bdy_tags, boundary_tag_to_index, BTAG_ALL))
    # firedrake exterior facets with no marker are assigned the
    # a dummy marker
    from firedrake.mesh import unmarked as fd_unmarked
    marker_to_neighbor_value[fd_unmarked] = marker_to_neighbor_value[None]
    # Now figure out the appropriate tags for each firedrake markers
    for marker in top.exterior_facets.unique_markers:
        marker_to_neighbor_value[marker] = \
            -(_boundary_tag_bit(bdy_tags, boundary_tag_to_index, marker)
              | -marker_to_neighbor_value[None])

    # {{{ build the FacialAdjacencyGroup for internal connectivity

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
    # If only using some of the cells
    from pyop2.datatypes import IntType
    if cells_to_use is not None:
        to_keep = np.isin(int_elements, cells_to_use)
        cells_to_use_inv = dict(zip(cells_to_use,
                                    np.arange(np.size(cells_to_use),
                                              dtype=IntType)))

        # Keep the cells that we are using and change old cell index
        # to new cell index
        int_elements = np.vectorize(cells_to_use_inv.__getitem__)(
            int_elements[to_keep])
        int_element_faces = int_element_faces[to_keep]
        int_neighbors = int_neighbors[to_keep]
        int_neighbor_faces = int_neighbor_faces[to_keep]
        # For neighbor cells, change to new cell index or record
        # as a new boundary (if the neighbor cell is not being used)
        newly_created_exterior_facs = []
        for ndx, icell in enumerate(int_neighbors):
            try:
                int_neighbors[ndx] = cells_to_use_inv[icell]
            except KeyError:
                newly_created_exterior_facs.append(ndx)
        # Make boolean array: 1 if a newly created exterior facet, 0 if
        #                     remains an interior facet
        newly_created_exterior_facs = np.isin(np.arange(np.size(int_elements)),
                                              newly_created_exterior_facs)
        new_ext_elements = int_elements[newly_created_exterior_facs]
        new_ext_element_faces = int_element_faces[newly_created_exterior_facs]
        new_ext_neighbor_tag = -(_boundary_tag_bit(bdy_tags,
                                                  boundary_tag_to_index,
                                                  BTAG_REALLY_ALL)
                                | _boundary_tag_bit(bdy_tags,
                                                    boundary_tag_to_index,
                                                    BTAG_INDUCED_BOUNDARY))
        new_ext_neighbors = np.full(new_ext_elements.shape,
                                    new_ext_neighbor_tag,
                                    dtype=IntType)
        new_ext_neighbor_faces = np.full(new_ext_elements.shape,
                                         0,
                                         dtype=Mesh.face_id_dtype)
        # Remove any (previously) interior facets that have become exterior
        # facets
        remaining_int_facs = np.logical_not(newly_created_exterior_facs)
        int_elements = int_elements[remaining_int_facs]
        int_element_faces = int_element_faces[remaining_int_facs]
        int_neighbors = int_neighbors[remaining_int_facs]
        int_neighbor_faces = int_neighbor_faces[remaining_int_facs]

    interconnectivity_grp = FacialAdjacencyGroup(igroup=0, ineighbor_group=0,
                                                 elements=int_elements,
                                                 neighbors=int_neighbors,
                                                 element_faces=int_element_faces,
                                                 neighbor_faces=int_neighbor_faces)

    # }}}

    # {{{ build the FacialAdjacencyGroup for boundary faces

    # We can get the elements directly from exterior facets
    ext_elements = top.exterior_facets.facet_cell.flatten()

    ext_element_faces = np.array([fd_loc_fac_nr_to_mm[fac_nr] for fac_nr in
                                  top.exterior_facets.local_facet_dat.data],
                                 dtype=Mesh.face_id_dtype)
    ext_neighbor_faces = np.zeros(ext_element_faces.shape,
                                  dtype=Mesh.face_id_dtype)
    # If only using some of the cells, throw away unused cells and
    # move to new cell index
    exterior_facet_markers = top.exterior_facets.markers
    if cells_to_use is not None:
        to_keep = np.isin(ext_elements, cells_to_use)
        ext_elements = np.vectorize(cells_to_use_inv.__getitem__)(
            ext_elements[to_keep])
        ext_element_faces = ext_element_faces[to_keep]
        ext_neighbor_faces = ext_neighbor_faces[to_keep]
        if exterior_facet_markers is not None:
            exterior_facet_markers = exterior_facet_markers[to_keep]

    # tag the boundary, making sure to record custom tags
    # (firedrake "markers") if present
    if top.exterior_facets.markers is not None:
        ext_neighbors = np.zeros(ext_elements.shape, dtype=IntType)
        for ifac, marker in enumerate(exterior_facet_markers):
            ext_neighbors[ifac] = marker_to_neighbor_value[marker]
    else:
        ext_neighbors = np.full(ext_elements.shape,
                                marker_to_neighbor_value[None],
                                dtype=IntType)

    # If not using all the cells, some interior facets may have become
    # exterior facets:
    if cells_to_use is not None:
        # Record any newly created exterior facets
        ext_elements = np.concatenate((ext_elements, new_ext_elements))
        ext_element_faces = np.concatenate((ext_element_faces,
                                            new_ext_element_faces))
        ext_neighbor_faces = np.concatenate((ext_neighbor_faces,
                                             new_ext_neighbor_faces))
        ext_neighbors = np.concatenate((ext_neighbors, new_ext_neighbors))

    exterior_grp = FacialAdjacencyGroup(igroup=0, ineighbor=None,
                                        elements=ext_elements,
                                        element_faces=ext_element_faces,
                                        neighbors=ext_neighbors,
                                        neighbor_faces=ext_neighbor_faces)

    # }}}

    return [{0: interconnectivity_grp, None: exterior_grp}]

# }}}


# {{{ Orientation computation

def _get_firedrake_orientations(fdrake_mesh, unflipped_group, vertices,
                                cells_to_use,
                                normals=None, no_normals_warn=True):
    r"""
    Return the orientations of the mesh elements:

    :arg fdrake_mesh: As described in :func:`import_firedrake_mesh`
    :arg unflipped_group: A :class:`SimplexElementGroup` instance with
        (potentially) some negatively oriented elements.
    :arg vertices: The vertex coordinates as a numpy array of shape
        *(ambient_dim, nvertices)* (the vertices of *unflipped_group*)
    :arg normals: As described in :func:`import_firedrake_mesh`
    :arg no_normals_warn: As described in :func:`import_firedrake_mesh`
    :arg cells_to_use: If *None*, then ignored. Otherwise, a numpy array
        of unique firedrake cell indices indicating which cells to use.

    :return: A numpy array, the *i*\ th element is > 0 if the *i*\ th element
        is positively oriented, < 0 if negatively oriented.
        Mesh must have co-dimension 0 or 1. If *cells_to_use* is not
        *None*, then the *i*\ th entry corresponds to the
        *cells_to_use[i]*\ th element.
    """
    # compute orientations
    tdim = fdrake_mesh.topological_dimension()
    gdim = fdrake_mesh.geometric_dimension()

    orient = None
    if gdim == tdim:
        # If the co-dimension is 0, :mod:`meshmode` has a convenient
        # function to compute cell orientations
        from meshmode.mesh.processing import \
            find_volume_mesh_element_group_orientation

        orient = find_volume_mesh_element_group_orientation(vertices,
                                                            unflipped_group)

    elif tdim == 1 and gdim == 2:
        # In this case we have a 1-surface embedded in 2-space.
        # Firedrake does not provide any convenient way of
        # letting the user set cell orientations in this case, so we
        # have to ask the user for cell normals directly.
        if cells_to_use is None:
            num_cells = fdrake_mesh.num_cells()
        else:
            num_cells = np.size(cells_to_use)
        orient = np.ones(num_cells)
        if normals:
            for i, (normal, vert_indices) in enumerate(
                    zip(np.array(normals), unflipped_group.vertex_indices)):
                edge = vertices[:, vert_indices[1]] - vertices[:, vert_indices[0]]
                if np.cross(normal, edge) < 0:
                    orient[i] = -1.0
        elif no_normals_warn:
            warn("Assuming all elements are positively-oriented.")

    elif tdim == 2 and gdim == 3:
        # In this case we have a 2-surface embedded in 3-space.
        # In this case, we assume the user has called
        # :func:`firedrake.mesh.MeshGeometry.init_cell_orientations`, see
        # https://www.firedrakeproject.org/variational-problems.html#ensuring-consistent-cell-orientations  # noqa : E501
        # for a tutorial on how these are usually initialized.
        #
        # Unfortunately, *init_cell_orientations* is currently only implemented
        # in 3D, so we can't use this in the 1/2 case.
        orient = fdrake_mesh.cell_orientations().dat.data
        if cells_to_use is not None:
            orient = orient[cells_to_use]
        r"""
            Convert (0 \implies negative, 1 \implies positive) to
            (-1 \implies negative, 1 \implies positive)
        """
        orient *= 2
        orient -= 1
    # Make sure the mesh fell into one of the cases
    # Nb : This should be guaranteed by previous checks,
    #      but is here anyway in case of future development.
    assert orient is not None
    return orient

# }}}


# {{{ Mesh importing from firedrake

def import_firedrake_mesh(fdrake_mesh, cells_to_use=None,
                          normals=None, no_normals_warn=None):
    """
    Create a :class:`meshmode.mesh.Mesh`
    from a `firedrake.mesh.MeshGeometry`
    with the same cells/elements, vertices, nodes,
    mesh order, and facial adjacency.

    The vertex and node coordinates will be the same, as well
    as the cell/element ordering. However, :mod:`firedrake`
    does not require elements to be positively oriented,
    so any negative elements are flipped
    as in :func:`meshmode.mesh.processing.flip_simplex_element_group`.

    The flipped cells/elements are identified by the returned
    *firedrake_orient* array

    :arg fdrake_mesh: `firedrake.mesh.MeshGeometry`.
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

            coords_fspace = fdrake_mesh.coordinates.function_space()
            vertex_entity_dofs = coords_fspace.finat_element.entity_dofs()[0]
            for entity, dof_list in vertex_entity_dofs.items():
                assert len(dof_list) > 0

    :arg cells_to_use: *cells_to_use* is primarily intended for use
        internally by :func:`~meshmode.interop.firedrake.connection.\
build_connection_from_firedrake`.
        *cells_to_use* must be either

        1. *None*, in which case this argument is ignored, or
        2. a numpy array of unique firedrake cell indexes.

        In case (2.),
        only cells whose index appears in *cells_to_use* are included
        in the resultant mesh, and their index in *cells_to_use*
        becomes the element index in the resultant mesh element group.
        Any faces or vertices which do not touch a cell in
        *cells_to_use* are also ignored.
        Note that in this latter case, some faces that are not
        boundaries in *fdrake_mesh* may become boundaries in the
        returned mesh. These "induced" boundaries are marked with
        :class:`~meshmode.mesh.BTAG_INDUCED_BOUNDARY`
        instead of :class:`~meshmode.mesh.BTAG_ALL`.

    :arg normals: **Only** used if *fdrake_mesh* is a 1-surface
        embedded in 2-space. In this case,

            - If *None* then
              all elements are assumed to be positively oriented.
            - Else, should be a list/array whose *i*\\ th entry
              is the normal for the *i*\\ th element (*i*\\ th
              in *mesh.coordinate.function_space()*'s
              *cell_node_list*)

    :arg no_normals_warn: If *True* (the default), raises a warning
        if *fdrake_mesh* is a 1-surface embedded in 2-space
        and *normals* is *None*.

    :return: A tuple *(meshmode mesh, firedrake_orient)*.
         ``firedrake_orient < 0`` is *True* for any negatively
         oriented firedrake cell (which was flipped by meshmode)
         and False for any positively oriented firedrake cell
         (which was not flipped by meshmode).
    """
    # Type validation
    from firedrake.mesh import MeshGeometry
    if not isinstance(fdrake_mesh, MeshGeometry):
        raise TypeError("'fdrake_mesh_topology' must be an instance of "
                        "firedrake.mesh.MeshGeometry, "
                        "not '%s'." % type(fdrake_mesh))
    if cells_to_use is not None:
        if not isinstance(cells_to_use, np.ndarray):
            raise TypeError("'cells_to_use' must be a np.ndarray or "
                            "*None*")
        assert len(cells_to_use.shape) == 1
        assert np.size(np.unique(cells_to_use)) == np.size(cells_to_use), \
            ":arg:`cells_to_use` must have unique entries"
        assert np.all(np.logical_and(cells_to_use >= 0,
                                     cells_to_use < fdrake_mesh.num_cells()))
    assert fdrake_mesh.ufl_cell().is_simplex(), "Mesh must use simplex cells"
    gdim = fdrake_mesh.geometric_dimension()
    tdim = fdrake_mesh.topological_dimension()
    assert gdim in [1, 2, 3], "Mesh must be in space of ambient dim 1, 2, or 3"
    assert gdim - tdim in [0, 1], "Mesh co-dimension must be 0 or 1"
    # firedrake meshes are not guaranteed be fully instantiated until
    # the .init() method is called. In particular, the coordinates function
    # may not be accessible if we do not call init(). If the mesh has
    # already been initialized, nothing will change. For more details
    # on why we need a second initialization, see
    # this pull request:
    # https://github.com/firedrakeproject/firedrake/pull/627
    # which details how Firedrake implements a mesh's coordinates
    # as a function on that very same mesh
    fdrake_mesh.init()

    # Get all the nodal information we can from the topology
    bdy_tags = _get_firedrake_boundary_tags(
        fdrake_mesh, tag_induced_boundary=cells_to_use is not None)

    with ProcessLogger(logger, "Retrieving vertex indices and computing "
                       "NodalAdjacency from firedrake mesh"):
        vertex_indices, nodal_adjacency = \
            _get_firedrake_nodal_info(fdrake_mesh, cells_to_use=cells_to_use)

        # If only using some cells, vertices may need new indices as many
        # will be removed
        if cells_to_use is not None:
            vert_ndx_new2old = np.unique(vertex_indices.flatten())
            vert_ndx_old2new = dict(zip(vert_ndx_new2old,
                                        np.arange(np.size(vert_ndx_new2old),
                                                  dtype=vertex_indices.dtype)))
            vertex_indices = \
                np.vectorize(vert_ndx_old2new.__getitem__)(vertex_indices)

    with ProcessLogger(logger, "Building (possibly) unflipped "
                       "SimplexElementGroup from firedrake unit nodes/nodes"):

        # Grab the mesh reference element and cell dimension
        coord_finat_elt = fdrake_mesh.coordinates.function_space().finat_element
        cell_dim = fdrake_mesh.cell_dimension()

        # Get finat unit nodes and map them onto the meshmode reference simplex
        finat_unit_nodes = get_finat_element_unit_nodes(coord_finat_elt)
        fd_ref_to_mm = get_affine_reference_simplex_mapping(cell_dim, True)
        finat_unit_nodes = fd_ref_to_mm(finat_unit_nodes)

        # Now grab the nodes
        coords = fdrake_mesh.coordinates
        cell_node_list = coords.function_space().cell_node_list
        if cells_to_use is not None:
            cell_node_list = cell_node_list[cells_to_use]
        nodes = np.real(coords.dat.data[cell_node_list])
        # Add extra dim in 1D for shape (nelements, nunit_nodes, dim)
        if tdim == 1:
            nodes = np.reshape(nodes, nodes.shape + (1,))
        # Transpose nodes to have shape (dim, nelements, nunit_nodes)
        nodes = np.transpose(nodes, (2, 0, 1))

        # make a group (possibly with some elements that need to be flipped)
        unflipped_group = SimplexElementGroup(coord_finat_elt.degree,
                                              vertex_indices,
                                              nodes,
                                              dim=cell_dim,
                                              unit_nodes=finat_unit_nodes)

    # Next get the vertices (we'll need these for the orientations)
    with ProcessLogger(logger, "Obtaining vertex coordinates"):
        coord_finat = fdrake_mesh.coordinates.function_space().finat_element
        # unit_vertex_indices are the element-local indices of the nodes
        # which coincide with the vertices, i.e. for element *i*,
        # vertex 0's coordinates would be nodes[i][unit_vertex_indices[0]].
        # This assumes each vertex has some node which coincides with it...
        # which is normally fine to assume for firedrake meshes.
        unit_vertex_indices = []
        # iterate through the dofs associated to each vertex on the
        # reference element
        for _, dofs in sorted(coord_finat.entity_dofs()[0].items()):
            assert len(dofs) == 1, \
                "The function space of the mesh coordinates must have" \
                " exactly one degree of freedom associated with " \
                " each vertex in order to determine vertex coordinates"
            dof, = dofs
            unit_vertex_indices.append(dof)

        # Now get the vertex coordinates as *(dim, nvertices)*-shaped array
        if cells_to_use is not None:
            nvertices = np.size(vert_ndx_new2old)
        else:
            nvertices = fdrake_mesh.num_vertices()
        vertices = np.ndarray((gdim, nvertices), dtype=nodes.dtype)
        recorded_verts = set()
        for icell, cell_vertex_indices in enumerate(vertex_indices):
            for local_vert_id, global_vert_id in enumerate(cell_vertex_indices):
                if global_vert_id not in recorded_verts:
                    recorded_verts.add(global_vert_id)
                    local_node_nr = unit_vertex_indices[local_vert_id]
                    vertices[:, global_vert_id] = nodes[:, icell, local_node_nr]

    # Use the vertices to compute the orientations and flip the group
    with ProcessLogger(logger, "Computing cell orientations"):
        orient = _get_firedrake_orientations(fdrake_mesh,
                                             unflipped_group,
                                             vertices,
                                             cells_to_use=cells_to_use,
                                             normals=normals,
                                             no_normals_warn=no_normals_warn)

    with ProcessLogger(logger, "Flipping group"):
        from meshmode.mesh.processing import flip_simplex_element_group
        group = flip_simplex_element_group(vertices, unflipped_group, orient < 0)

    # Now, any flipped element had its 0 vertex and 1 vertex exchanged.
    # This changes the local facet nr, so we need to create and then
    # fix our facial adjacency groups. To do that, we need to figure
    # out which local facet numbers switched.
    face_vertex_indices = group.face_vertex_indices()
    # face indices of the faces not containing vertex 0 and not
    # containing vertex 1, respectively
    no_zero_face_ndx, no_one_face_ndx = None, None
    for iface, face in enumerate(face_vertex_indices):
        if 0 not in face:
            no_zero_face_ndx = iface
        elif 1 not in face:
            no_one_face_ndx = iface

    with ProcessLogger(logger, "Building (possibly) unflipped "
                       "FacialAdjacencyGroups"):
        unflipped_facial_adjacency_groups = \
            _get_firedrake_facial_adjacency_groups(fdrake_mesh,
                                                   cells_to_use=cells_to_use)

    # applied below to take elements and element_faces
    # (or neighbors and neighbor_faces) and flip in any faces that need to
    # be flipped.
    def flip_local_face_indices(faces, elements):
        faces = np.copy(faces)
        neg_elements = np.full(elements.shape, False)
        # To handle neighbor case, we only need to flip at elements
        # who have a neighbor, i.e. where neighbors is not a negative
        # bitmask of bdy tags
        neg_elements[elements >= 0] = (orient[elements[elements >= 0]] < 0)
        no_zero = np.logical_and(neg_elements, faces == no_zero_face_ndx)
        no_one = np.logical_and(neg_elements, faces == no_one_face_ndx)
        faces[no_zero], faces[no_one] = no_one_face_ndx, no_zero_face_ndx
        return faces

    # Create new facial adjacency groups that have been flipped
    with ProcessLogger(logger, "Flipping FacialAdjacencyGroups"):
        facial_adjacency_groups = []
        for igroup, fagrps in enumerate(unflipped_facial_adjacency_groups):
            facial_adjacency_groups.append({})
            for ineighbor_group, fagrp in fagrps.items():
                new_element_faces = flip_local_face_indices(fagrp.element_faces,
                                                            fagrp.elements)
                new_neighbor_faces = flip_local_face_indices(fagrp.neighbor_faces,
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

def export_mesh_to_firedrake(mesh, group_nr=None, comm=None):
    r"""
    Create a firedrake mesh corresponding to one
    :class:`~meshmode.mesh.Mesh`'s
    :class:`~meshmode.mesh.SimplexElementGroup`.

    :param mesh: A :class:`~meshmode.mesh.Mesh` to convert with
        at least one :class:`~meshmode.mesh.SimplexElementGroup`.
        'mesh.is_conforming' must evaluate to *True*.
        'mesh' must have vertices supplied, i.e.
        'mesh.vertices' must not be *None*.
    :param group_nr: The group number to be converted into a firedrake
        mesh. The corresponding group must be of type
        :class:`~meshmode.mesh.SimplexElementGroup`. If *None* and
        *mesh* has only one group, that group is used. Otherwise,
        a *ValueError* is raised.
    :param comm: The communicator to build the dmplex mesh on

    :return: A tuple *(fdrake_mesh, fdrake_cell_ordering, perm2cell)*
        where

        * *fdrake_mesh* is a :mod:`firedrake`
          `firedrake.mesh.MeshGeometry` corresponding to
          *mesh*
        * *fdrake_cell_ordering* is a numpy array whose *i*\ th
          element in *mesh* (i.e. the *i*\ th element in
          *mesh.groups[group_nr].vertex_indices*) corresponds to the
          *fdrake_cell_ordering[i]*\ th :mod:`firedrake` cell
        * *perm2cell* is a dictionary, mapping tuples to
          1-D numpy arrays of meshmode element indices.
          Each meshmode element index
          appears in exactly one of these arrays. The corresponding
          tuple describes how firedrake reordered the local vertex
          indices on that cell. In particular, if *c*
          is in the list *perm2cell[p]* for a tuple *p*, then
          the *p[i]*\ th local vertex of the *fdrake_cell_ordering[c]*\ th
          firedrake cell corresponds to the *i*\ th local vertex
          of the *c*\ th meshmode element.

    .. warning::
        Currently, no custom boundary tags are exported along with the mesh.
        :mod:`firedrake` seems to only allow one marker on each facet, whereas
        :mod:`meshmode` allows many.
    """
    if not isinstance(mesh, Mesh):
        raise TypeError("'mesh' must of type meshmode.mesh.Mesh,"
                        " not '%s'." % type(mesh))
    if group_nr is None:
        if len(mesh.groups) != 1:
            raise ValueError("'group_nr' is *None* but 'mesh' has "
                             "more than one group.")
        group_nr = 0
    if not isinstance(group_nr, int):
        raise TypeError("Expecting 'group_nr' to be of type int, not "
                        f"'{type(group_nr)}'")
    if group_nr < 0 or group_nr >= len(mesh.groups):
        raise ValueError("'group_nr' is an invalid group index:"
                         f" '{group_nr}' fails to satisfy "
                         f"0 <= {group_nr} < {len(mesh.groups)}")
    if not isinstance(mesh.groups[group_nr], SimplexElementGroup):
        raise TypeError("Expecting 'mesh.groups[group_nr]' to be of type "
                        "meshmode.mesh.SimplexElementGroup, not "
                        f"'{type(mesh.groups[group_nr])}'")
    if mesh.vertices is None:
        raise ValueError("'mesh' has no vertices "
                         "('mesh.vertices' is *None*)")
    if not mesh.is_conforming:
        raise ValueError(f"'mesh.is_conforming' is {mesh.is_conforming} "
                         "instead of *True*. Converting non-conforming "
                         " meshes to Firedrake is not supported")

    # Get the vertices and vertex indices of the requested group
    with ProcessLogger(logger, "Obtaining vertices from selected group"):
        group = mesh.groups[group_nr]
        fd2mm_indices = np.unique(group.vertex_indices.flatten())
        coords = mesh.vertices[:, fd2mm_indices].T
        mm2fd_indices = dict(zip(fd2mm_indices, np.arange(np.size(fd2mm_indices))))
        cells = np.vectorize(mm2fd_indices.__getitem__)(group.vertex_indices)

    # Get a dmplex object and then a mesh topology
    with ProcessLogger(logger, "Building dmplex object and MeshTopology"):
        if comm is None:
            from pyop2.mpi import COMM_WORLD
            comm = COMM_WORLD
        # FIXME : not sure how to get around the private accesses
        import firedrake.mesh as fd_mesh
        plex = fd_mesh._from_cell_list(group.dim, cells, coords, comm)
        # Nb : One might be tempted to pass reorder=False and thereby save some
        #      hassle in exchange for forcing firedrake to have slightly
        #      less efficient caching. Unfortunately, that only prevents
        #      the cells from being reordered, and does not prevent the
        #      vertices from being (locally) reordered on each cell...
        #      the tl;dr is we don't actually save any hassle
        top = fd_mesh.Mesh(plex, dim=mesh.ambient_dim)  # mesh topology
        top.init()

    # Get new element ordering:
    with ProcessLogger(logger, "Determining permutations applied"
                       " to local vertex numbers"):
        c_start, c_end = top.topology_dm.getHeightStratum(0)
        cell_index_mm2fd = np.vectorize(top._cell_numbering.getOffset)(
            np.arange(c_start, c_end))
        v_start, v_end = top.topology_dm.getDepthStratum(0)

        # Firedrake goes crazy reordering local vertex numbers,
        # we've got to work to figure out what changes they made.
        #
        # *perm2cells* will map permutations of local vertex numbers to
        #              the list of all the meshmode cells
        #              which firedrake reordered according to that permutation
        #
        #              Permutations on *n* vertices are stored as a tuple
        #              containing all of the integers *0*, *1*, *2*, ..., *n-1*
        #              exactly once. A permutation *p*
        #              represents relabeling the *i*\ th local vertex
        #              of a meshmode element as the *p[i]*\ th local vertex
        #              in the corresponding firedrake cell.
        #
        #              *perm2cells[p]* is a list of all the meshmode element indices
        #              for which *p* represents the reordering applied by firedrake
        perm2cells = {}
        for mm_cell_id, dmp_ids in enumerate(top.cell_closure[cell_index_mm2fd]):
            # look at order of vertices in firedrake cell
            vert_dmp_ids = \
                dmp_ids[np.logical_and(v_start <= dmp_ids, dmp_ids < v_end)]
            fdrake_order = vert_dmp_ids - v_start
            # get original order
            mm_order = mesh.groups[group_nr].vertex_indices[mm_cell_id]
            # want permutation p so that mm_order[p] = fdrake_order
            # To do so, look at permutations acting by composition.
            #
            # mm_order \circ argsort(mm_order) =
            #     fdrake_order \circ argsort(fdrake_order)
            # so
            # mm_order \circ argsort(mm_order) \circ inv(argsort(fdrake_order))
            #  = fdrake_order
            #
            # argsort acts as an inverse, so the desired permutation is:
            perm = tuple(np.argsort(mm_order)[np.argsort(np.argsort(fdrake_order))])
            perm2cells.setdefault(perm, [])
            perm2cells[perm].append(mm_cell_id)

        # Make perm2cells map to numpy arrays instead of lists
        perm2cells = {perm: np.array(cells)
                      for perm, cells in perm2cells.items()}

    # Now make a coordinates function
    with ProcessLogger(logger, "Building firedrake function "
                       "space for mesh coordinates"):
        from firedrake import VectorFunctionSpace, Function
        coords_fspace = VectorFunctionSpace(top, "CG", group.order,
                                            dim=mesh.ambient_dim)
        coords = Function(coords_fspace)

    # get firedrake unit nodes and map onto meshmode reference element
    fd_ref_cell_to_mm = get_affine_reference_simplex_mapping(group.dim, True)
    fd_unit_nodes = get_finat_element_unit_nodes(coords_fspace.finat_element)
    fd_unit_nodes = fd_ref_cell_to_mm(fd_unit_nodes)

    basis = simplex_best_available_basis(group.dim, group.order)
    resampling_mat = resampling_matrix(basis,
                                       new_nodes=fd_unit_nodes,
                                       old_nodes=group.unit_nodes)
    # Store the meshmode data resampled to firedrake unit nodes
    # (but still in meshmode order)
    resampled_group_nodes = np.matmul(group.nodes, resampling_mat.T)

    # Now put the nodes in the right local order
    # nodes is shaped *(ambient dim, nelements, nunit nodes)*
    with ProcessLogger(logger, "Storing meshmode mesh coordinates"
                       " in firedrake nodal order"):
        from meshmode.mesh.processing import get_simplex_element_flip_matrix
        for perm, cells in perm2cells.items():
            flip_mat = get_simplex_element_flip_matrix(group.order,
                                                       fd_unit_nodes,
                                                       perm)
            flip_mat = np.rint(flip_mat).astype(np.int32)
            resampled_group_nodes[:, cells, :] = \
                np.matmul(resampled_group_nodes[:, cells, :], flip_mat.T)

    # store resampled data in right cell ordering
    with ProcessLogger(logger, "resampling mesh coordinates to "
                       "firedrake unit nodes"):
        reordered_cell_node_list = coords_fspace.cell_node_list[cell_index_mm2fd]
        coords.dat.data[reordered_cell_node_list, :] = \
            resampled_group_nodes.transpose((1, 2, 0))

    return fd_mesh.Mesh(coords), cell_index_mm2fd, perm2cells

# }}}

# vim: foldmethod=marker
