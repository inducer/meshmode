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

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)

from meshmode.interop.firedrake import FromFiredrakeConnection

import pytest

import logging
logger = logging.getLogger(__name__)

# skip testing this module if cannot import firedrake
firedrake = pytest.importorskip("firedrake")

from firedrake import (
    UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh,
    FunctionSpace, VectorFunctionSpace, Function,
    SpatialCoordinate, Constant)


CLOSE_ATOL = 10**-12


@pytest.fixture(params=["FiredrakeUnitIntervalMesh",
                        "FiredrakeUnitSquareMesh",
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


@pytest.fixture(params=[1, 2, 3], ids=["P^1", "P^2", "P^3"])
def fdrake_degree(request):
    return request.param


# {{{ Basic conversion checks for the function space

def test_discretization_consistency(ctx_factory, fdrake_mesh, fdrake_degree):
    """
    While nodes may change, vertex conversion should be *identical* up to
    reordering, ensure this is the case for DG spaces. Also ensure the
    meshes have the same basic properties and the function space/discretization
    agree across firedrake vs meshmode
    """
    # get fdrake_verts (shaped like (nverts, dim))
    fdrake_verts = fdrake_mesh.coordinates.dat.data
    if fdrake_mesh.geometric_dimension() == 1:
        fdrake_verts = fdrake_verts[:, np.newaxis]

    # Get meshmode vertices (shaped like (dim, nverts))
    fdrake_fspace = FunctionSpace(fdrake_mesh, 'DG', fdrake_degree)
    cl_ctx = ctx_factory()
    fdrake_connection = FromFiredrakeConnection(cl_ctx, fdrake_fspace)
    to_discr = fdrake_connection.to_discr
    meshmode_verts = to_discr.mesh.vertices

    # Ensure the meshmode mesh has one group and make sure both
    # meshes agree on some basic properties
    assert len(to_discr.mesh.groups) == 1
    fdrake_mesh_fspace = fdrake_mesh.coordinates.function_space()
    fdrake_mesh_order = fdrake_mesh_fspace.finat_element.degree
    assert to_discr.mesh.groups[0].order == fdrake_mesh_order
    assert to_discr.mesh.groups[0].nelements == fdrake_mesh.num_cells()
    assert to_discr.mesh.nvertices == fdrake_mesh.num_vertices()

    # Ensure that the vertex sets are identical up to reordering
    # Nb: I got help on this from stack overflow:
    # https://stackoverflow.com/questions/38277143/sort-2d-numpy-array-lexicographically  # noqa: E501
    lex_sorted_mm_verts = meshmode_verts[:, np.lexsort(meshmode_verts)]
    lex_sorted_fdrake_verts = fdrake_verts[np.lexsort(fdrake_verts.T)]
    np.testing.assert_array_equal(lex_sorted_mm_verts, lex_sorted_fdrake_verts.T)

    # Ensure the discretization and the firedrake function space agree on
    # some basic properties
    finat_elt = fdrake_fspace.finat_element
    assert len(to_discr.groups) == 1
    assert to_discr.groups[0].order == finat_elt.degree
    assert to_discr.groups[0].nunit_nodes == finat_elt.space_dimension()
    assert to_discr.nnodes == fdrake_fspace.node_count

# }}}


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
def test_function_transfer(ctx_factory,
                           fdrake_mesh, fdrake_family, fdrake_degree,
                           fdrake_f_expr, meshmode_f_eval):
    """
    Make sure creating a function then transporting it is the same
    (up to resampling error) as creating a function on the transported
    mesh
    """
    fdrake_fspace = FunctionSpace(fdrake_mesh, fdrake_family, fdrake_degree)
    spatial_coord = SpatialCoordinate(fdrake_mesh)

    fdrake_f = Function(fdrake_fspace).interpolate(fdrake_f_expr(spatial_coord))

    cl_ctx = ctx_factory()
    fdrake_connection = FromFiredrakeConnection(cl_ctx, fdrake_fspace)
    transported_f = fdrake_connection.from_firedrake(fdrake_f)

    to_discr = fdrake_connection.to_discr
    with cl.CommandQueue(cl_ctx) as queue:
        nodes = to_discr.nodes().get(queue=queue)
    meshmode_f = meshmode_f_eval(nodes)

    np.testing.assert_allclose(transported_f, meshmode_f, atol=CLOSE_ATOL)

# }}}


# {{{ Idempotency tests fd->mm->fd and (fd->)mm->fd->mm for connection

def check_idempotency(fdrake_connection, fdrake_function):
    """
    Make sure fd->mm->fd and mm->fd->mm are identity
    """
    vdim = None
    if len(fdrake_function.dat.data.shape) > 1:
        vdim = fdrake_function.dat.data.shape[1]
    fdrake_fspace = fdrake_connection.from_fspace(dim=vdim)

    # Test for idempotency fd->mm->fd
    mm_field = fdrake_connection.from_firedrake(fdrake_function)
    fdrake_function_copy = Function(fdrake_fspace)
    fdrake_connection.from_meshmode(mm_field, fdrake_function_copy,
                                    assert_fdrake_discontinuous=False,
                                    continuity_tolerance=1e-8)

    np.testing.assert_allclose(fdrake_function_copy.dat.data,
                               fdrake_function.dat.data,
                               atol=CLOSE_ATOL)

    # Test for idempotency (fd->)mm->fd->mm
    mm_field_copy = fdrake_connection.from_firedrake(fdrake_function_copy)
    np.testing.assert_allclose(mm_field_copy, mm_field, atol=CLOSE_ATOL)


def test_scalar_idempotency(ctx_factory, fdrake_mesh,
                            fdrake_family, fdrake_degree):
    """
    Make sure fd->mm->fd and mm->fd->mm are identity for scalar spaces
    """
    fdrake_fspace = FunctionSpace(fdrake_mesh, fdrake_family, fdrake_degree)

    # Make a function with unique values at each node
    fdrake_unique = Function(fdrake_fspace)
    fdrake_unique.dat.data[:] = np.arange(fdrake_unique.dat.data.shape[0])

    # test idempotency
    cl_ctx = ctx_factory()
    fdrake_connection = FromFiredrakeConnection(cl_ctx, fdrake_fspace)
    check_idempotency(fdrake_connection, fdrake_unique)


def test_vector_idempotency(ctx_factory, fdrake_mesh,
                            fdrake_family, fdrake_degree):
    """
    Make sure fd->mm->fd and mm->fd->mm are identity for vector spaces
    """
    fdrake_vfspace = VectorFunctionSpace(fdrake_mesh, fdrake_family, fdrake_degree)

    # Make a function with unique values at each node
    xx = SpatialCoordinate(fdrake_vfspace.mesh())
    fdrake_unique = Function(fdrake_vfspace).interpolate(xx)

    # test idempotency
    cl_ctx = ctx_factory()
    fdrake_connection = FromFiredrakeConnection(cl_ctx, fdrake_vfspace)
    check_idempotency(fdrake_connection, fdrake_unique)

# }}}

# vim: foldmethod=marker
