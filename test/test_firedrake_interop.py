from __future__ import annotations


__copyright__ = "Copyright (C) 2020 Benjamin Sepanski"

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

import logging

import numpy as np
import pytest

from arraycontext import ArrayContextFactory, pytest_generate_tests_for_array_contexts

from meshmode import _acf  # noqa: F401
from meshmode.array_context import PytestPyOpenCLArrayContextFactory
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import (
    InterpolatoryQuadratureSimplexGroupFactory,
)
from meshmode.dof_array import DOFArray
from meshmode.interop.firedrake import (
    build_connection_from_firedrake,
    build_connection_to_firedrake,
    import_firedrake_mesh,
)
from meshmode.mesh import BTAG_ALL, BTAG_INDUCED_BOUNDARY, Mesh, check_bc_coverage


logger = logging.getLogger(__name__)
pytest_generate_tests = pytest_generate_tests_for_array_contexts(
        [PytestPyOpenCLArrayContextFactory])

CLOSE_ATOL = 1e-12


@pytest.fixture(params=["annulus.msh",
                        "blob2d-order1-h4e-2.msh",
                        "blob2d-order1-h6e-2.msh",
                        "blob2d-order1-h8e-2.msh",
                        "blob2d-order4-h4e-2.msh",
                        "blob2d-order4-h6e-2.msh",
                        "blob2d-order4-h8e-2.msh",
                        ])
def mm_mesh(request):
    return request.param


@pytest.fixture(params=["FiredrakeUnitIntervalMesh",
                        "FiredrakeUnitSquareMesh",
                        "FiredrakeUnitSquareMesh-order2",
                        "FiredrakeUnitCubeMesh",
                        "annulus.msh",
                        "blob2d-order1-h4e-2.msh",
                        "blob2d-order1-h6e-2.msh",
                        "blob2d-order1-h8e-2.msh",
                        ])
def fdrake_mesh(request):
    return request.param


@pytest.fixture(params=[1, 4], ids=["P^1", "P^4"])
def fspace_degree(request):
    return request.param


def make_mm_mesh(name: str) -> Mesh:
    from meshmode.mesh.io import read_gmsh
    from meshmode.mesh.processing import remove_unused_vertices
    return remove_unused_vertices(read_gmsh(name))


def make_firedrake_mesh(name: str):
    from firedrake import (
        Function,
        Mesh,
        SpatialCoordinate,
        UnitCubeMesh,
        UnitIntervalMesh,
        UnitSquareMesh,
        VectorFunctionSpace,
    )

    if name == "FiredrakeUnitIntervalMesh":
        return UnitIntervalMesh(100)
    elif name == "FiredrakeUnitSquareMesh":
        return UnitSquareMesh(10, 10)
    elif name == "FiredrakeUnitSquareMesh-order2":
        m = UnitSquareMesh(10, 10)
        fspace = VectorFunctionSpace(m, "CG", 2)
        coords = Function(fspace).interpolate(SpatialCoordinate(m))
        return Mesh(coords)
    elif name == "FiredrakeUnitCubeMesh":
        return UnitCubeMesh(5, 5, 5)
    elif name not in ("annulus.msh", "blob2d-order1-h4e-2.msh",
                      "blob2d-order1-h6e-2.msh", "blob2d-order1-h8e-2.msh"):
        raise ValueError(f"Unexpected value for mesh name: {name}")

    # Firedrake can't read in higher order meshes from gmsh,
    # so we can only use the order1 blobs
    fd_mesh = Mesh(name)
    fd_mesh.init()

    return fd_mesh


# {{{ Basic conversion checks for the function space

def check_consistency(fdrake_fspace, discr, group_nr=0):
    """
    While nodes may change, vertex conversion should be *identical* up to
    reordering, ensure this is the case for DG spaces. Also ensure the
    meshes have the same basic properties and the function space/discretization
    agree across firedrake vs meshmode
    """
    # Get the unit vertex indices (in each cell)
    fdrake_mesh = fdrake_fspace.mesh()
    cfspace = fdrake_mesh.coordinates.function_space()
    entity_dofs = cfspace.finat_element.entity_dofs()[0]
    fdrake_unit_vert_indices = []
    for _, local_node_nrs in sorted(entity_dofs.items()):
        assert len(local_node_nrs) == 1
        fdrake_unit_vert_indices.append(local_node_nrs[0])

    # get the firedrake vertices, in no particular order
    fdrake_vert_indices = cfspace.cell_node_list[:, fdrake_unit_vert_indices]
    fdrake_vert_indices = np.unique(fdrake_vert_indices)
    fdrake_verts = fdrake_mesh.coordinates.dat.data[fdrake_vert_indices, ...]
    if fdrake_mesh.geometric_dimension() == 1:
        fdrake_verts = fdrake_verts[:, np.newaxis]

    meshmode_verts = discr.mesh.vertices

    # Ensure the meshmode mesh has one group and make sure both
    # meshes agree on some basic properties
    assert len(discr.mesh.groups) == 1
    fdrake_mesh_fspace = fdrake_mesh.coordinates.function_space()
    fdrake_mesh_order = fdrake_mesh_fspace.finat_element.degree
    assert discr.mesh.groups[group_nr].dim == fdrake_mesh.topological_dimension()
    assert discr.mesh.groups[group_nr].order == fdrake_mesh_order
    assert discr.mesh.groups[group_nr].nelements == fdrake_mesh.num_cells()
    assert discr.mesh.nvertices == fdrake_mesh.num_vertices()

    # Ensure that the vertex sets are identical up to reordering
    # Nb: I got help on this from stack overflow:
    # https://stackoverflow.com/questions/38277143/sort-2d-numpy-array-lexicographically  # noqa: E501
    lex_sorted_mm_verts = meshmode_verts[:, np.lexsort(meshmode_verts)]
    lex_sorted_fdrake_verts = fdrake_verts[np.lexsort(fdrake_verts.T)]
    np.testing.assert_allclose(lex_sorted_mm_verts, lex_sorted_fdrake_verts.T,
                               atol=1e-15)

    # Ensure the discretization and the firedrake function space agree on
    # some basic properties
    finat_elt = fdrake_fspace.finat_element
    assert len(discr.groups) == 1
    assert discr.groups[group_nr].order == finat_elt.degree
    assert discr.groups[group_nr].nunit_dofs == finat_elt.space_dimension()
    assert discr.ndofs == fdrake_fspace.node_count


def test_from_fd_consistency(
            actx_factory: ArrayContextFactory,
            fdrake_mesh,
            fspace_degree):
    """
    Check basic consistency with a FiredrakeConnection built from firedrake
    """
    pytest.importorskip("firedrake")
    actx = actx_factory()

    from firedrake import FunctionSpace

    # make discretization from firedrake
    fdrake_mesh = make_firedrake_mesh(fdrake_mesh)
    fdrake_fspace = FunctionSpace(fdrake_mesh, "DG", fspace_degree)

    fdrake_connection = build_connection_from_firedrake(actx, fdrake_fspace)
    discr = fdrake_connection.discr
    # Check consistency
    check_consistency(fdrake_fspace, discr)


def test_to_fd_consistency(actx_factory: ArrayContextFactory, mm_mesh, fspace_degree):
    pytest.importorskip("firedrake")
    actx = actx_factory()

    mm_mesh = make_mm_mesh(mm_mesh)
    fspace_degree += mm_mesh.groups[0].order

    factory = InterpolatoryQuadratureSimplexGroupFactory(fspace_degree)
    discr = Discretization(actx, mm_mesh, factory)
    fdrake_connection = build_connection_to_firedrake(discr)
    fdrake_fspace = fdrake_connection.firedrake_fspace()
    # Check consistency
    check_consistency(fdrake_fspace, discr)

# }}}


# {{{ Now check the FiredrakeConnection consistency when restricted to bdy

def test_from_boundary_consistency(actx_factory: ArrayContextFactory,
                                   fdrake_mesh,
                                   fspace_degree):
    """
    Make basic checks that FiredrakeConnection restricted to cells
    near the boundary is not doing
    something obviously wrong,
    i.e. that the firedrake boundary tags partition the converted meshmode mesh,
    that the firedrake boundary tags correspond to the same physical
    regions in the converted meshmode mesh as in the original firedrake mesh,
    and that each boundary tag is associated to the same number of facets
    in the converted meshmode mesh as in the original firedrake mesh.
    """
    pytest.importorskip("firedrake")
    actx = actx_factory()

    from firedrake import FunctionSpace

    fdrake_mesh = make_firedrake_mesh(fdrake_mesh)
    fdrake_fspace = FunctionSpace(fdrake_mesh, "DG", fspace_degree)

    frombdy_conn = \
        build_connection_from_firedrake(actx,
                                        fdrake_fspace,
                                        restrict_to_boundary="on_boundary")

    # Ensure the meshmode mesh has one group and make sure both
    # meshes agree on some basic properties
    discr = frombdy_conn.discr
    assert len(discr.mesh.groups) == 1
    fdrake_mesh_fspace = fdrake_mesh.coordinates.function_space()
    fdrake_mesh_order = fdrake_mesh_fspace.finat_element.degree
    assert discr.mesh.groups[0].dim == fdrake_mesh.topological_dimension()
    assert discr.mesh.groups[0].order == fdrake_mesh_order

    # Get the unit vertex indices (in each cell)
    fdrake_mesh = fdrake_fspace.mesh()
    cfspace = fdrake_mesh.coordinates.function_space()
    entity_dofs = cfspace.finat_element.entity_dofs()[0]
    fdrake_unit_vert_indices = []
    for _, local_node_nrs in sorted(entity_dofs.items()):
        assert len(local_node_nrs) == 1
        fdrake_unit_vert_indices.append(local_node_nrs[0])
    fdrake_unit_vert_indices = np.array(fdrake_unit_vert_indices)

    # only look at cells "near" bdy (with >= 1 vertex on)
    from meshmode.interop.firedrake.connection import _get_cells_to_use
    cells_near_bdy = _get_cells_to_use(fdrake_mesh, "on_boundary")
    # get the firedrake vertices of cells near the boundary,
    # in no particular order
    fdrake_vert_indices = \
        cfspace.cell_node_list[cells_near_bdy,
                               fdrake_unit_vert_indices[:, np.newaxis]]
    fdrake_vert_indices = np.unique(fdrake_vert_indices)
    fdrake_verts = fdrake_mesh.coordinates.dat.data[fdrake_vert_indices, ...]
    if fdrake_mesh.geometric_dimension() == 1:
        fdrake_verts = fdrake_verts[:, np.newaxis]
    # Get meshmode vertices (shaped like (dim, nverts))
    meshmode_verts = discr.mesh.vertices

    # Ensure that the vertices of firedrake elements on
    # the boundary are identical to the resultant meshes' vertices up to
    # reordering
    # Nb: I got help on this from stack overflow:
    # https://stackoverflow.com/questions/38277143/sort-2d-numpy-array-lexicographically  # noqa: E501
    lex_sorted_mm_verts = meshmode_verts[:, np.lexsort(meshmode_verts)]
    lex_sorted_fdrake_verts = fdrake_verts[np.lexsort(fdrake_verts.T)]
    np.testing.assert_allclose(lex_sorted_mm_verts, lex_sorted_fdrake_verts.T,
                               atol=CLOSE_ATOL)

    # Ensure the discretization and the firedrake function space reference element
    # agree on some basic properties
    finat_elt = fdrake_fspace.finat_element
    assert len(discr.groups) == 1
    assert discr.groups[0].order == finat_elt.degree
    assert discr.groups[0].nunit_dofs == finat_elt.space_dimension()

# }}}


# {{{ Boundary tags checking

@pytest.mark.parametrize(
    ("mesh_name", "bdy_ids", "coord_indices", "coord_values"), [
        ("square",
         [1, 2, 3, 4],
         [0, 0, 1, 1],
         [0.0, 1.0, 0.0, 1.0]),
        ("cube",
         [1, 2, 3, 4, 5, 6],
         [0, 0, 1, 1, 2, 2],
         [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
    ])
@pytest.mark.parametrize("only_convert_bdy", (True, False))
def test_bdy_tags(mesh_name, bdy_ids, coord_indices, coord_values,
                  only_convert_bdy):
    """
    Make sure the given boundary ids cover the converted mesh.
    Make sure that the given coordinate have the given value for the
    corresponding boundary tag (see :mod:`firedrake.utility_meshes`'s
    documentation to see how the boundary tags for its utility meshes are
    defined)
    """
    pytest.importorskip("firedrake")

    from firedrake import UnitCubeMesh, UnitSquareMesh

    if mesh_name == "square":
        square_or_cube_mesh = UnitSquareMesh(10, 10)
    elif mesh_name == "cube":
        square_or_cube_mesh = UnitCubeMesh(5, 5, 5)
    else:
        raise ValueError(f"Unknown mesh name: {mesh_name!r}")

    cells_to_use = None
    if only_convert_bdy:
        from meshmode.interop.firedrake.connection import _get_cells_to_use
        cells_to_use = _get_cells_to_use(square_or_cube_mesh, "on_boundary")
    mm_mesh, _orient = import_firedrake_mesh(square_or_cube_mesh,
                                             cells_to_use=cells_to_use)
    # Check disjoint coverage of bdy ids and BTAG_ALL
    check_bc_coverage(mm_mesh, [BTAG_ALL])
    check_bc_coverage(mm_mesh, bdy_ids)

    from meshmode.mesh import BoundaryAdjacencyGroup
    ext_grps = [
        grp for grp in mm_mesh.facial_adjacency_groups[0]
        if isinstance(grp, BoundaryAdjacencyGroup)]

    # count number of times the boundary tag appears in the meshmode mesh,
    # should be the same as in the firedrake mesh
    bdy_id_to_mm_count = {}
    for ext_grp in ext_grps:
        # if mm_mesh has boundaries flagged as not boundaries we need to skip them
        if ext_grp.boundary_tag == BTAG_INDUCED_BOUNDARY:
            continue
        bdy_id_to_mm_count.setdefault(ext_grp.boundary_tag, 0)
        bdy_id_to_mm_count[ext_grp.boundary_tag] += len(ext_grp.elements)

    # Verify that the number of meshes tagged with a boundary tag
    # is the same in meshmode and firedrake for each tag in *bdy_ids*
    from meshmode.interop.firedrake.mesh import _get_facet_markers
    fdrake_bdy_ids, fdrake_counts = \
        np.unique(_get_facet_markers(
            square_or_cube_mesh.topology.topology_dm,
            square_or_cube_mesh.exterior_facets.facets), return_counts=True)
    assert set(fdrake_bdy_ids) == set(bdy_ids)
    for bdy_id, fdrake_count in zip(fdrake_bdy_ids, fdrake_counts, strict=True):
        assert fdrake_count == bdy_id_to_mm_count[bdy_id]

    # Now make sure we have identified the correct faces
    face_vertex_indices = mm_mesh.groups[0].face_vertex_indices()
    for bdy_id_index, bdy_id in enumerate(bdy_ids):
        matching_ext_grps = [
            grp for grp in ext_grps
            if grp.boundary_tag == bdy_id]
        assert len(matching_ext_grps) == 1
        ext_grp = matching_ext_grps[0]
        for iel, ifac in zip(ext_grp.elements, ext_grp.element_faces, strict=True):
            el_vert_indices = mm_mesh.groups[0].vertex_indices[iel]
            # numpy nb: have to have comma to use advanced indexing
            face_vert_indices = el_vert_indices[face_vertex_indices[ifac], ]
            # shape: *(ambient dim, num vertices on face)*
            face_verts = mm_mesh.vertices[:, face_vert_indices]
            # Figure out which coordinate should have a fixed value, and what
            # that value is. Also, count how many times each boundary tag appears
            coord_index = coord_indices[bdy_id_index]
            val = coord_values[bdy_id_index]
            assert np.max(np.abs(face_verts[coord_index, :] - val)) < CLOSE_ATOL

# }}}


# TODO : Add test for FiredrakeConnection built from meshmode
#        where group_nr != 0

# {{{  Double check functions are being transported correctly


@pytest.mark.parametrize("fdrake_mesh_name,fdrake_mesh_pars,dim",
    [("UnitInterval", [10, 20, 30], 1),
     ("UnitSquare", [10, 20, 30], 2),
     ("UnitCube", [10, 20, 30], 3),
     ("blob2d-order1", ["8e-2", "6e-2", "4e-2"], 2),
     pytest.param("blob2d-order4", ["8e-2", "6e-2", "4e-2"], 2,
                  marks=pytest.mark.xfail),
     ("warp", [10, 20, 30], 2),
     ("warp", [10, 20, 30], 3),
     ])
@pytest.mark.parametrize("only_convert_bdy", [False, True])
def test_from_fd_transfer(actx_factory: ArrayContextFactory, fspace_degree,
                          fdrake_mesh_name, fdrake_mesh_pars, dim,
                          only_convert_bdy):
    """
    Make sure creating a function which projects onto
    one dimension then transports it is the same
    (up to resampling error) as projecting to one
    dimension on the transported mesh
    """
    pytest.importorskip("firedrake")
    actx = actx_factory()

    # build estimate-of-convergence recorder
    from pytools.convergence import EOCRecorder

    # (fd -> mm ? True : False, dimension projecting onto)
    eoc_recorders = {(True, d): EOCRecorder() for d in range(dim)}
    if not only_convert_bdy:
        for d in range(dim):
            eoc_recorders[False, d] = EOCRecorder()

    def get_fdrake_mesh_and_h_from_par(mesh_par):
        from firedrake import Mesh, UnitCubeMesh, UnitIntervalMesh, UnitSquareMesh

        if fdrake_mesh_name == "UnitInterval":
            assert dim == 1
            n = mesh_par
            fdrake_mesh = UnitIntervalMesh(n)
            h = 1/n
        elif fdrake_mesh_name == "UnitSquare":
            assert dim == 2
            n = mesh_par
            fdrake_mesh = UnitSquareMesh(n, n)
            h = 1/n
        elif fdrake_mesh_name == "UnitCube":
            assert dim == 3
            n = mesh_par
            fdrake_mesh = UnitCubeMesh(n, n, n)
            h = 1/n
        elif fdrake_mesh_name in ("blob2d-order1", "blob2d-order4"):
            assert dim == 2
            if fdrake_mesh_name == "blob2d-order1":
                fdrake_mesh = Mesh(f"{fdrake_mesh_name}-h{mesh_par}.msh",
                                   dim=dim)
            else:
                from meshmode.interop.firedrake import export_mesh_to_firedrake
                from meshmode.mesh.io import read_gmsh
                mm_mesh = read_gmsh(f"{fdrake_mesh_name}-h{mesh_par}.msh",
                                    force_ambient_dim=dim)
                fdrake_mesh, _, _ = export_mesh_to_firedrake(mm_mesh)
            h = float(mesh_par)
        elif fdrake_mesh_name == "warp":
            from meshmode.interop.firedrake import export_mesh_to_firedrake
            from meshmode.mesh.generation import generate_warped_rect_mesh
            mm_mesh = generate_warped_rect_mesh(dim, order=4,
                nelements_side=mesh_par)
            fdrake_mesh, _, _ = export_mesh_to_firedrake(mm_mesh)
            h = 1/mesh_par
        else:
            raise ValueError("fdrake_mesh_name not recognized")

        return (fdrake_mesh, h)

    from firedrake import Function, FunctionSpace, SpatialCoordinate, sin

    # Record error for each refinement of each mesh
    for mesh_par in fdrake_mesh_pars:
        fdrake_mesh, h = get_fdrake_mesh_and_h_from_par(mesh_par)
        # make function space and build connection
        fdrake_fspace = FunctionSpace(fdrake_mesh, "DG", fspace_degree)
        if only_convert_bdy:
            fdrake_connection = \
                build_connection_from_firedrake(actx,
                                                fdrake_fspace,
                                                restrict_to_boundary="on_boundary")
        else:
            fdrake_connection = build_connection_from_firedrake(actx, fdrake_fspace)
        # get this for making functions in firedrake
        spatial_coord = SpatialCoordinate(fdrake_mesh)

        # get nodes in handier format for making meshmode functions
        discr = fdrake_connection.discr
        # nodes is np array (ambient_dim,) of DOFArray (ngroups,)
        # of arrays (nelements, nunit_dofs), we want a single np array
        # of shape (ambient_dim, nelements, nunit_dofs)
        nodes = discr.nodes()
        group_nodes = np.array([actx.to_numpy(dof_arr[0]) for dof_arr in nodes])

        # Now, for each coordinate d, test transferring the function
        # x -> sin(dth component of x)
        for d in range(dim):
            fdrake_f = Function(fdrake_fspace).interpolate(sin(spatial_coord[d]))
            # transport fdrake function and put in numpy
            fd2mm_f = fdrake_connection.from_firedrake(fdrake_f, actx=actx)
            fd2mm_f = actx.to_numpy(fd2mm_f[0])
            meshmode_f = np.sin(group_nodes[d, :, :])

            # record fd -> mm error
            err = np.max(np.abs(fd2mm_f - meshmode_f))
            eoc_recorders[True, d].add_data_point(h, err)

            if not only_convert_bdy:
                # now transport mm -> fd
                meshmode_f_dofarr = discr.zeros(actx)
                meshmode_f_dofarr[0][:] = meshmode_f
                mm2fd_f = fdrake_connection.from_meshmode(meshmode_f_dofarr)
                # record mm -> fd error
                err = np.max(np.abs(fdrake_f.dat.data - mm2fd_f.dat.data))
                eoc_recorders[False, d].add_data_point(h, err)

    # assert that order is correct or error is "low enough"
    for ((fd2mm, d), eoc_rec) in eoc_recorders.items():
        print(f"\nfiredrake -> meshmode: {fd2mm}\nvector *x* -> *sin(x[{d}])*\n",
              eoc_rec)
        assert (
            eoc_rec.order_estimate() >= fspace_degree
            or eoc_rec.max_error() < 2e-14)


@pytest.mark.parametrize("mesh_name,mesh_pars,dim",
    [("blob2d-order1", ["8e-2", "6e-2", "4e-2"], 2),
     ("blob2d-order4", ["8e-2", "6e-2", "4e-2"], 2),
     ("warp", [10, 20, 30], 2),
     ("warp", [10, 20, 30], 3),
     ])
def test_to_fd_transfer(
            actx_factory: ArrayContextFactory,
            fspace_degree,
            mesh_name,
            mesh_pars,
            dim):
    """
    Make sure creating a function which projects onto
    one dimension then transports it is the same
    (up to resampling error) as projecting to one
    dimension on the transported mesh
    """
    pytest.importorskip("firedrake")
    actx = actx_factory()

    # build estimate-of-convergence recorder
    from pytools.convergence import EOCRecorder

    # dimension projecting onto -> EOCRecorder
    eoc_recorders = {d: EOCRecorder() for d in range(dim)}

    from firedrake import Function, SpatialCoordinate

    # Get each of the refinements of the meshmeshes and record
    # conversions errors
    for mesh_par in mesh_pars:
        if mesh_name in ("blob2d-order1", "blob2d-order4"):
            assert dim == 2
            from meshmode.mesh.io import read_gmsh
            mm_mesh = read_gmsh(f"{mesh_name}-h{mesh_par}.msh",
                                force_ambient_dim=dim)
            h = float(mesh_par)
        elif mesh_name == "warp":
            from meshmode.mesh.generation import generate_warped_rect_mesh
            mm_mesh = generate_warped_rect_mesh(dim, order=4,
                nelements_side=mesh_par)
            h = 1/mesh_par
        else:
            raise ValueError("mesh_name not recognized")

        # Make discr and connect it to firedrake
        factory = InterpolatoryQuadratureSimplexGroupFactory(fspace_degree)
        discr = Discretization(actx, mm_mesh, factory)

        fdrake_connection = build_connection_to_firedrake(discr)
        fdrake_fspace = fdrake_connection.firedrake_fspace()
        spatial_coord = SpatialCoordinate(fdrake_fspace.mesh())

        # get the group's nodes in a numpy array
        nodes = discr.nodes()
        group_nodes = np.array([actx.to_numpy(dof_arr[0]) for dof_arr in nodes])

        for d in range(dim):
            meshmode_f = discr.zeros(actx)
            meshmode_f[0][:] = group_nodes[d, :, :]

            # connect to firedrake and evaluate expr in firedrake
            fdrake_f = Function(fdrake_fspace).interpolate(spatial_coord[d])

            # transport to firedrake and record error
            mm2fd_f = fdrake_connection.from_meshmode(meshmode_f)

            err = np.max(np.abs(fdrake_f.dat.data - mm2fd_f.dat.data))
            eoc_recorders[d].add_data_point(h, err)

    # assert that order is correct or error is "low enough"
    for d, eoc_rec in eoc_recorders.items():
        print(f"\nvector *x* -> *x[{d}]*\n", eoc_rec)
        assert (
            eoc_rec.order_estimate() >= fspace_degree
            or eoc_rec.max_error() < 2e-14)

# }}}


# {{{ Idempotency tests fd->mm->fd and (fd->)mm->fd->mm for connection

@pytest.mark.parametrize("fspace_type", ("scalar", "vector", "tensor"))
@pytest.mark.parametrize("only_convert_bdy", (False, True))
def test_from_fd_idempotency(actx_factory: ArrayContextFactory,
                             fdrake_mesh, fspace_degree,
                             fspace_type, only_convert_bdy):
    """
    Make sure fd->mm->fd and (fd->)->mm->fd->mm are identity
    """
    pytest.importorskip("firedrake")
    actx = actx_factory()

    from firedrake import (
        Function,
        FunctionSpace,
        SpatialCoordinate,
        TensorFunctionSpace,
        VectorFunctionSpace,
        as_tensor,
    )

    # Make a function space and a function with unique values at each node
    fdrake_mesh = make_firedrake_mesh(fdrake_mesh)
    if fspace_type == "scalar":
        fdrake_fspace = FunctionSpace(fdrake_mesh, "DG", fspace_degree)
        # Just use the node nr
        fdrake_unique = Function(fdrake_fspace)
        fdrake_unique.dat.data[:] = np.arange(fdrake_unique.dat.data.shape[0])
    elif fspace_type == "vector":
        fdrake_fspace = VectorFunctionSpace(fdrake_mesh, "DG", fspace_degree)
        # use the coordinates
        xx = SpatialCoordinate(fdrake_fspace.mesh())
        fdrake_unique = Function(fdrake_fspace).interpolate(xx)
    elif fspace_type == "tensor":
        fdrake_fspace = TensorFunctionSpace(fdrake_mesh, "DG", fspace_degree)
        # use the coordinates, duplicated into the right tensor shape
        xx = SpatialCoordinate(fdrake_fspace.mesh())
        dim = fdrake_fspace.mesh().geometric_dimension()
        unique_expr = as_tensor([xx for _ in range(dim)])
        fdrake_unique = Function(fdrake_fspace).interpolate(unique_expr)

    # If only converting boundary, first go ahead and do one round of
    # fd->mm->fd. This will zero out any degrees of freedom absent in
    # the meshmode mesh (because they are not associated to cells
    #                    with >= 1 node on the boundary)
    #
    # Otherwise, just continue as normal
    if only_convert_bdy:
        fdrake_connection = \
            build_connection_from_firedrake(actx,
                                            fdrake_fspace,
                                            restrict_to_boundary="on_boundary")
        temp = fdrake_connection.from_firedrake(fdrake_unique, actx=actx)
        fdrake_unique = fdrake_connection.from_meshmode(temp)
    else:
        fdrake_connection = build_connection_from_firedrake(actx, fdrake_fspace)

    # Test for idempotency fd->mm->fd
    mm_field = fdrake_connection.from_firedrake(fdrake_unique, actx=actx)
    fdrake_unique_copy = Function(fdrake_fspace)
    fdrake_connection.from_meshmode(mm_field, out=fdrake_unique_copy)

    np.testing.assert_allclose(fdrake_unique_copy.dat.data,
                               fdrake_unique.dat.data,
                               atol=CLOSE_ATOL)

    # Test for idempotency (fd->)mm->fd->mm
    mm_field_copy = fdrake_connection.from_firedrake(fdrake_unique_copy,
                                                     actx=actx)
    if fspace_type == "scalar":
        np.testing.assert_allclose(actx.to_numpy(mm_field_copy[0]),
                                   actx.to_numpy(mm_field[0]),
                                   atol=CLOSE_ATOL)
    else:
        for dof_arr_cp, dof_arr in zip(mm_field_copy.flatten(),
                                       mm_field.flatten(), strict=True):
            np.testing.assert_allclose(actx.to_numpy(dof_arr_cp[0]),
                                       actx.to_numpy(dof_arr[0]),
                                       atol=CLOSE_ATOL)


def test_to_fd_idempotency(actx_factory: ArrayContextFactory, mm_mesh, fspace_degree):
    """
    Make sure mm->fd->mm and (mm->)->fd->mm->fd are identity
    """
    pytest.importorskip("firedrake")
    actx = actx_factory()

    # make sure degree is higher order than mesh
    mm_mesh = make_mm_mesh(mm_mesh)
    fspace_degree += mm_mesh.groups[0].order

    # Make a function space and a function with unique values at each node
    factory = InterpolatoryQuadratureSimplexGroupFactory(fspace_degree)
    discr = Discretization(actx, mm_mesh, factory)
    fdrake_connection = build_connection_to_firedrake(discr)
    fdrake_mesh = fdrake_connection.firedrake_fspace().mesh()
    dtype = fdrake_mesh.coordinates.dat.data.dtype

    mm_unique = discr.zeros(actx, dtype=dtype)
    unique_vals = np.arange(np.size(mm_unique[0]), dtype=dtype)
    mm_unique[0].set(unique_vals.reshape(mm_unique[0].shape))
    mm_unique_copy = DOFArray(actx, (mm_unique[0].copy(),))

    # Test for idempotency mm->fd->mm
    fdrake_unique = fdrake_connection.from_meshmode(mm_unique)
    fdrake_connection.from_firedrake(fdrake_unique, out=mm_unique_copy)

    np.testing.assert_allclose(actx.to_numpy(mm_unique_copy[0]),
                               actx.to_numpy(mm_unique[0]),
                               atol=CLOSE_ATOL)

    # Test for idempotency (mm->)fd->mm->fd
    fdrake_unique_copy = fdrake_connection.from_meshmode(mm_unique_copy)
    np.testing.assert_allclose(fdrake_unique_copy.dat.data,
                               fdrake_unique.dat.data,
                               atol=CLOSE_ATOL)

# }}}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: foldmethod=marker
