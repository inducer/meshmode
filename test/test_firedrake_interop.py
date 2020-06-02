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
    SpatialCoordinate)


CLOSE_ATOL = 10**-12


@pytest.fixture(params=[1, 2, 3], ids=["1D", "2D", "3D"])
def fdrake_mesh(request):
    dim = request.param
    if dim == 1:
        return UnitIntervalMesh(100)
    if dim == 2:
        return UnitSquareMesh(10, 10)
    if dim == 3:
        return UnitCubeMesh(5, 5, 5)
    return None


@pytest.fixture(params=[1, 2, 3], ids=["P^1", "P^2", "P^3"])
def fdrake_degree(request):
    return request.param


# {{{ Idempotency tests fd->mm->fd and (fd->)mm->fd->mm for connection

def check_idempotency(fdrake_connection, fdrake_function):
    """
    Make sure fd->mm->fd and mm->fd->mm are identity for DG spaces
    """
    fdrake_fspace = fdrake_connection.from_function_space()

    # Test for idempotency fd->mm->fd
    mm_field = fdrake_connection.from_firedrake(fdrake_function)
    fdrake_function_copy = Function(fdrake_fspace)
    fdrake_connection.from_meshmode(mm_field, fdrake_function_copy)

    print(np.sum(np.abs(fdrake_function.dat.data - fdrake_function_copy.dat.data)))
    print(type(fdrake_function.dat.data))
    np.testing.assert_allclose(fdrake_function_copy.dat.data,
                               fdrake_function.dat.data,
                               atol=CLOSE_ATOL)

    # Test for idempotency (fd->)mm->fd->mm
    mm_field_copy = fdrake_connection.from_firedrake(fdrake_function_copy)
    np.testing.assert_allclose(mm_field_copy, mm_field, atol=CLOSE_ATOL)


def test_scalar_idempotency(ctx_factory, fdrake_mesh, fdrake_degree):
    """
    Make sure fd->mm->fd and mm->fd->mm are identity for scalar DG spaces
    """
    fdrake_fspace = FunctionSpace(fdrake_mesh, 'DG', fdrake_degree)

    # Make a function with unique values at each node
    fdrake_unique = Function(fdrake_fspace)
    fdrake_unique.dat.data[:] = np.arange(fdrake_unique.dat.data.shape[0])

    # test idempotency
    cl_ctx = ctx_factory()
    fdrake_connection = FromFiredrakeConnection(cl_ctx, fdrake_fspace)
    check_idempotency(fdrake_connection, fdrake_unique)


def test_vector_idempotency(ctx_factory, fdrake_mesh, fdrake_degree):
    """
    Make sure fd->mm->fd and mm->fd->mm are identity for vector DG spaces
    """
    fdrake_vfspace = VectorFunctionSpace(fdrake_mesh, 'DG', fdrake_degree)

    # Make a function with unique values at each node
    xx = SpatialCoordinate(fdrake_vfspace.mesh())
    fdrake_unique = Function(fdrake_vfspace).interpolate(xx)

    # test idempotency
    cl_ctx = ctx_factory()
    fdrake_connection = FromFiredrakeConnection(cl_ctx, fdrake_vfspace)
    check_idempotency(fdrake_connection, fdrake_unique)

# }}}
