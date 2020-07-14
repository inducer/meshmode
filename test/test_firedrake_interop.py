# TODO:
#      * Make sure from_meshmode and from_firedrake receive
#        DOFArrays
#      * Make sure output of from_firedrake is treated as
#        DOFArray
#      * Run tests and debug
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

import numpy as np
import pyopencl as cl
import six

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)

from meshmode.array_tools import PyOpenCLArrayContext

from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import (
    InterpolatoryQuadratureSimplexGroupFactory)

from meshmode.mesh import BTAG_ALL, BTAG_REALLY_ALL, check_bc_coverage

from meshmode.interop.firedrake import (
    FromFiredrakeConnection, FromBdyFiredrakeConnection, ToFiredrakeConnection,
    import_firedrake_mesh)
from meshmode.interop.firedrake.connection import _compute_cells_near_bdy

import pytest

import logging
logger = logging.getLogger(__name__)

# skip testing this module if cannot import firedrake
firedrake = pytest.importorskip("firedrake")

from firedrake import (
    UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh,
    FunctionSpace, VectorFunctionSpace, TensorFunctionSpace,
    Function, SpatialCoordinate, Constant, as_tensor)


CLOSE_ATOL = 10**-12


@pytest.fixture(params=["annulus.msh",
                        "blob2d-order1-h4e-2.msh",
                        "blob2d-order1-h6e-2.msh",
                        "blob2d-order1-h8e-2.msh",
                        "blob2d-order4-h4e-2.msh",
                        "blob2d-order4-h6e-2.msh",
                        "blob2d-order4-h8e-2.msh",
                        ])
def mm_mesh(request):
    from meshmode.mesh.io import read_gmsh
    return read_gmsh(request.param)


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
    mesh_name = request.param
    if mesh_name == "FiredrakeUnitIntervalMesh":
        return UnitIntervalMesh(100)
    if mesh_name == "FiredrakeUnitSquareMesh":
        return UnitSquareMesh(10, 10)
    if mesh_name == "FiredrakeUnitSquareMesh-order2":
        m = UnitSquareMesh(10, 10)
        fspace = VectorFunctionSpace(m, 'CG', 2)
        coords = Function(fspace).interpolate(SpatialCoordinate(m))
        from firedrake.mesh import Mesh
        return Mesh(coords)
    if mesh_name == "FiredrakeUnitCubeMesh":
        return UnitCubeMesh(5, 5, 5)

    # Firedrake can't read in higher order meshes from gmsh,
    # so we can only use the order1 blobs
    from firedrake import Mesh
    fd_mesh = Mesh(mesh_name)
    fd_mesh.init()
    return fd_mesh


@pytest.fixture(params=["CG", "DG"])
def fdrake_family(request):
    return request.param


@pytest.fixture(params=[1, 3], ids=["P^1", "P^3"])
def fspace_degree(request):
    return request.param


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
    for _, local_node_nrs in sorted(six.iteritems(entity_dofs)):
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
    assert discr.groups[group_nr].nunit_nodes == finat_elt.space_dimension()
    assert discr.nnodes == fdrake_fspace.node_count


def test_fd2mm_consistency(ctx_factory, fdrake_mesh, fspace_degree):
    """
    Check basic consistency with a FromFiredrakeConnection
    """
    # make discretization from firedrake
    fdrake_fspace = FunctionSpace(fdrake_mesh, 'DG', fspace_degree)

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    fdrake_connection = FromFiredrakeConnection(actx, fdrake_fspace)
    discr = fdrake_connection.discr
    # Check consistency
    check_consistency(fdrake_fspace, discr)


def test_mm2fd_consistency(ctx_factory, mm_mesh, fspace_degree):
    fspace_degree += mm_mesh.groups[0].order

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    factory = InterpolatoryQuadratureSimplexGroupFactory(fspace_degree)
    discr = Discretization(actx, mm_mesh, factory)
    fdrake_connection = ToFiredrakeConnection(discr)
    fdrake_fspace = fdrake_connection.firedrake_fspace()
    # Check consistency
    check_consistency(fdrake_fspace, discr)

# }}}


# {{{ Now check the FromBdyFiredrakeConnection consistency

def test_from_bdy_consistency(ctx_factory,
                              fdrake_mesh,
                              fdrake_family,
                              fspace_degree):
    """
    Make basic checks that FiredrakeFromBdyConnection is not doing something
    obviouisly wrong, i.e. that it has proper tagging, that it has
    the right number of cells, etc.
    """
    fdrake_fspace = FunctionSpace(fdrake_mesh, fdrake_family, fspace_degree)

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    frombdy_conn = FromBdyFiredrakeConnection(actx,
                                              fdrake_fspace,
                                              "on_boundary")

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
    for _, local_node_nrs in sorted(six.iteritems(entity_dofs)):
        assert len(local_node_nrs) == 1
        fdrake_unit_vert_indices.append(local_node_nrs[0])
    fdrake_unit_vert_indices = np.array(fdrake_unit_vert_indices)

    # only look at cells "near" bdy (with >= 1 vertex on)
    cells_near_bdy = _compute_cells_near_bdy(fdrake_mesh, 'on_boundary')
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
    assert discr.groups[0].nunit_nodes == finat_elt.space_dimension()

# }}}


# {{{ Boundary tags checking

bdy_tests = [(UnitSquareMesh(10, 10),
             [1, 2, 3, 4],
             [0, 0, 1, 1],
             [0.0, 1.0, 0.0, 1.0]),
             (UnitCubeMesh(5, 5, 5),
              [1, 2, 3, 4, 5, 6],
              [0, 0, 1, 1, 2, 2],
              [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
             ]


@pytest.mark.parametrize("square_or_cube_mesh,bdy_ids,coord_indices,coord_values",
                         bdy_tests)
@pytest.mark.parametrize("only_convert_bdy", (True, False))
def test_bdy_tags(square_or_cube_mesh, bdy_ids, coord_indices, coord_values,
                  only_convert_bdy):
    """
    Make sure the given boundary ids cover the converted mesh.
    Make sure that the given coordinate have the given value for the
    corresponding boundary tag (see :mod:`firedrake`'s documentation
    to see how the boundary tags for its utility meshes are defined)
    """
    cells_to_use = None
    if only_convert_bdy:
        cells_to_use = _compute_cells_near_bdy(square_or_cube_mesh, 'on_boundary')
    mm_mesh, orient = import_firedrake_mesh(square_or_cube_mesh,
                                            cells_to_use=cells_to_use)
    # Ensure meshmode required boundary tags are there
    assert set([BTAG_ALL, BTAG_REALLY_ALL]) <= set(mm_mesh.boundary_tags)
    # Check disjoint coverage of bdy ids and BTAG_ALL
    check_bc_coverage(mm_mesh, [BTAG_ALL])
    check_bc_coverage(mm_mesh, bdy_ids)

    # count number of times the boundary tag appears in the meshmode mesh,
    # should be the same as in the firedrake mesh
    bdy_id_to_mm_count = {}
    # Now make sure we have identified the correct faces
    face_vertex_indices = mm_mesh.groups[0].face_vertex_indices()
    ext_grp = mm_mesh.facial_adjacency_groups[0][None]
    for iel, ifac, bdy_flags in zip(
            ext_grp.elements, ext_grp.element_faces, ext_grp.neighbors):
        el_vert_indices = mm_mesh.groups[0].vertex_indices[iel]
        # numpy nb: have to have comma to use advanced indexing
        face_vert_indices = el_vert_indices[face_vertex_indices[ifac], ]
        # shape: *(ambient dim, num vertices on face)*
        face_verts = mm_mesh.vertices[:, face_vert_indices]
        # Figure out which coordinate should have a fixed value, and what
        # that value is. Also, count how many times each boundary tag appears
        coord_index, val = None, None
        for bdy_id_index, bdy_id in enumerate(bdy_ids):
            if mm_mesh.boundary_tag_bit(bdy_id) & -bdy_flags:
                bdy_id_to_mm_count.setdefault(bdy_id, 0)
                bdy_id_to_mm_count[bdy_id] += 1
                coord_index = coord_indices[bdy_id_index]
                val = coord_values[bdy_id_index]
                break
        assert np.max(np.abs(face_verts[coord_index, :] - val)) < CLOSE_ATOL

    # Verify that the number of meshes tagged with a boundary tag
    # is the same in meshmode and firedrake for each tag in *bdy_ids*
    fdrake_bdy_ids, fdrake_counts = \
        np.unique(square_or_cube_mesh.exterior_facets.markers, return_counts=True)
    assert set(fdrake_bdy_ids) == set(bdy_ids)
    for bdy_id, fdrake_count in zip(fdrake_bdy_ids, fdrake_counts):
        assert fdrake_count == bdy_id_to_mm_count[bdy_id]

# }}}


# TODO : Add test for ToFiredrakeConnection where group_nr != 0
# {{{  Double check functions are being transported correctly

def alternating_sum_fd(spatial_coord):
    """
    Return an expression x1 - x2 + x3 -+...
    """
    return sum(
        [(-1)**i * spatial_coord
         for i, spatial_coord in enumerate(spatial_coord)]
    )


def alternating_sum_mm(nodes):
    """
    Take the *(dim, nnodes)* array nodes and return an array
    holding the alternating sum of the coordinates of each node
    """
    alternator = np.ones(nodes.shape[0])
    alternator[1::2] *= -1
    return np.matmul(alternator, nodes)


# In 1D/2D/3D check constant 1,
# projection to x1, x1/x1+x2/x1+x2+x3, and x1/x1-x2/x1-x2+x3.
# This should show that projection to any coordinate in 1D/2D/3D
# transfers correctly.
test_functions = [
    (lambda spatial_coord: Constant(1.0), lambda nodes: np.ones(nodes.shape[1])),
    (lambda spatial_coord: spatial_coord[0], lambda nodes: nodes[0, :]),
    (sum, lambda nodes: np.sum(nodes, axis=0)),
    (alternating_sum_fd, alternating_sum_mm)
]


@pytest.mark.parametrize("fdrake_f_expr,meshmode_f_eval", test_functions)
@pytest.mark.parametrize("only_convert_bdy", (False, True))
def test_from_fd_transfer(ctx_factory,
                          fdrake_mesh, fdrake_family, fspace_degree,
                          fdrake_f_expr, meshmode_f_eval,
                          only_convert_bdy):
    """
    Make sure creating a function then transporting it is the same
    (up to resampling error) as creating a function on the transported
    mesh
    """
    # make function space and function
    fdrake_fspace = FunctionSpace(fdrake_mesh, fdrake_family, fspace_degree)
    spatial_coord = SpatialCoordinate(fdrake_mesh)

    fdrake_f = Function(fdrake_fspace).interpolate(fdrake_f_expr(spatial_coord))

    # build connection
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    if only_convert_bdy:
        fdrake_connection = FromBdyFiredrakeConnection(actx, fdrake_fspace,
                                                       'on_boundary')
    else:
        fdrake_connection = FromFiredrakeConnection(actx, fdrake_fspace)

    # transport fdrake function
    fd2mm_f = fdrake_connection.from_firedrake(fdrake_f)

    # build same function in meshmode
    discr = fdrake_connection.discr
    nodes = discr.nodes().get(queue=queue)
    meshmode_f = meshmode_f_eval(nodes)

    # fd -> mm should be same as creating in meshmode
    np.testing.assert_allclose(fd2mm_f, meshmode_f, atol=CLOSE_ATOL)

    if not only_convert_bdy:
        # now transport mm -> fd
        mm2fd_f = \
            fdrake_connection.from_meshmode(meshmode_f,
                                            assert_fdrake_discontinuous=False,
                                            continuity_tolerance=1e-8)
        # mm -> fd should be same as creating in firedrake
        np.testing.assert_allclose(fdrake_f.dat.data, mm2fd_f.dat.data,
                                   atol=CLOSE_ATOL)


@pytest.mark.parametrize("fdrake_f_expr,meshmode_f_eval", test_functions)
def test_to_fd_transfer(ctx_factory, mm_mesh, fspace_degree,
                        fdrake_f_expr, meshmode_f_eval):
    """
    Make sure creating a function then transporting it is the same
    (up to resampling error) as creating a function on the transported
    mesh
    """
    fspace_degree += mm_mesh.groups[0].order
    # Make discr and evaluate function in meshmode
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    factory = InterpolatoryQuadratureSimplexGroupFactory(fspace_degree)
    discr = Discretization(actx, mm_mesh, factory)

    nodes = discr.nodes().get(queue=queue)
    meshmode_f = meshmode_f_eval(nodes)

    # connect to firedrake and evaluate expr in firedrake
    fdrake_connection = ToFiredrakeConnection(discr)
    fdrake_fspace = fdrake_connection.firedrake_fspace()
    spatial_coord = SpatialCoordinate(fdrake_fspace.mesh())
    fdrake_f = Function(fdrake_fspace).interpolate(fdrake_f_expr(spatial_coord))

    # transport to firedrake and make sure this is the same
    mm2fd_f = fdrake_connection.from_meshmode(meshmode_f)

    np.testing.assert_allclose(mm2fd_f.dat.data, fdrake_f.dat.data,
                               atol=CLOSE_ATOL)

# }}}


# {{{ Idempotency tests fd->mm->fd and (fd->)mm->fd->mm for connection

@pytest.mark.parametrize("fspace_type", ("scalar", "vector", "tensor"))
@pytest.mark.parametrize("only_convert_bdy", (False, True))
def test_from_fd_idempotency(ctx_factory,
                             fdrake_mesh, fdrake_family, fspace_degree,
                             fspace_type, only_convert_bdy):
    """
    Make sure fd->mm->fd and (fd->)->mm->fd->mm are identity
    """
    # Make a function space and a function with unique values at each node
    if fspace_type == "scalar":
        fdrake_fspace = FunctionSpace(fdrake_mesh, fdrake_family, fspace_degree)
        # Just use the node nr
        fdrake_unique = Function(fdrake_fspace)
        fdrake_unique.dat.data[:] = np.arange(fdrake_unique.dat.data.shape[0])
    elif fspace_type == "vector":
        fdrake_fspace = VectorFunctionSpace(fdrake_mesh, fdrake_family,
                                             fspace_degree)
        # use the coordinates
        xx = SpatialCoordinate(fdrake_fspace.mesh())
        fdrake_unique = Function(fdrake_fspace).interpolate(xx)
    elif fspace_type == "tensor":
        fdrake_fspace = TensorFunctionSpace(fdrake_mesh,
                                            fdrake_family,
                                            fspace_degree)
        # use the coordinates, duplicated into the right tensor shape
        xx = SpatialCoordinate(fdrake_fspace.mesh())
        dim = fdrake_fspace.mesh().geometric_dimension()
        unique_expr = as_tensor([xx for _ in range(dim)])
        fdrake_unique = Function(fdrake_fspace).interpolate(unique_expr)

    # Make connection
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    # If only converting boundary, first go ahead and do one round of
    # fd->mm->fd. This will zero out any degrees of freedom absent in
    # the meshmode mesh (because they are not associated to cells
    #                    with >= 1 node on the boundary)
    #
    # Otherwise, just continue as normal
    if only_convert_bdy:
        fdrake_connection = FromBdyFiredrakeConnection(actx, fdrake_fspace,
                                                       'on_boundary')
        temp = fdrake_connection.from_firedrake(fdrake_unique)
        fdrake_unique = \
            fdrake_connection.from_meshmode(temp,
                                            assert_fdrake_discontinuous=False,
                                            continuity_tolerance=1e-8)
    else:
        fdrake_connection = FromFiredrakeConnection(actx, fdrake_fspace)

    # Test for idempotency fd->mm->fd
    mm_field = fdrake_connection.from_firedrake(fdrake_unique)
    fdrake_unique_copy = Function(fdrake_fspace)
    fdrake_connection.from_meshmode(mm_field, fdrake_unique_copy,
                                    assert_fdrake_discontinuous=False,
                                    continuity_tolerance=1e-8)

    np.testing.assert_allclose(fdrake_unique_copy.dat.data,
                               fdrake_unique.dat.data,
                               atol=CLOSE_ATOL)

    # Test for idempotency (fd->)mm->fd->mm
    mm_field_copy = fdrake_connection.from_firedrake(fdrake_unique_copy)
    np.testing.assert_allclose(mm_field_copy, mm_field, atol=CLOSE_ATOL)


def test_to_fd_idempotency(ctx_factory, mm_mesh, fspace_degree):
    """
    Make sure mm->fd->mm and (mm->)->fd->mm->fd are identity
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    # make sure degree is higher order than mesh
    fspace_degree += mm_mesh.groups[0].order

    # Make a function space and a function with unique values at each node
    factory = InterpolatoryQuadratureSimplexGroupFactory(fspace_degree)
    discr = Discretization(actx, mm_mesh, factory)
    fdrake_connection = ToFiredrakeConnection(discr)
    mm_unique = np.arange(discr.nnodes, dtype=np.float64)
    mm_unique_copy = np.copy(mm_unique)

    # Test for idempotency mm->fd->mm
    fdrake_unique = fdrake_connection.from_meshmode(mm_unique)
    fdrake_connection.from_firedrake(fdrake_unique, mm_unique_copy)

    np.testing.assert_allclose(mm_unique_copy, mm_unique, atol=CLOSE_ATOL)

    # Test for idempotency (mm->)fd->mm->fd
    fdrake_unique_copy = fdrake_connection.from_meshmode(mm_unique_copy)
    np.testing.assert_allclose(fdrake_unique_copy.dat.data,
                               fdrake_unique.dat.data,
                               atol=CLOSE_ATOL)

# }}}

# vim: foldmethod=marker
