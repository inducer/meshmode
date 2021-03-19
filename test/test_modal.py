__copyright__ = "Copyright (C) 2021 University of Illinois Board of Trustees"

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


from functools import partial
import numpy as np

import meshmode                       # noqa: F401
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests
    )
from meshmode.dof_array import DOFArray
from meshmode.mesh import (
    SimplexElementGroup,
    TensorProductElementGroup
    )

from meshmode.discretization.poly_element import (
    # Simplex group factories
    ModalSimplexGroupFactory,
    InterpolatoryQuadratureSimplexGroupFactory,
    PolynomialWarpAndBlendGroupFactory,
    PolynomialRecursiveNodesGroupFactory,
    PolynomialEquidistantSimplexGroupFactory,
    # Tensor product group factories
    ModalTensorProductGroupFactory,
    LegendreGaussLobattoTensorProductGroupFactory,
    # Quadrature-based (non-interpolatory) group factories
    QuadratureSimplexGroupFactory
    )

from meshmode.discretization.nodal import NodalDiscretization
from meshmode.discretization.modal import ModalDiscretization
from meshmode.discretization.connection.modal import (
    ModalDiscretizationConnection,
    ModalInverseDiscretizationConnection
    )

from meshmode.dof_array import thaw

import meshmode.mesh.generation as mgen
import pytest


@pytest.mark.parametrize("nodal_group_factory", [
    InterpolatoryQuadratureSimplexGroupFactory,
    PolynomialWarpAndBlendGroupFactory,
    partial(PolynomialRecursiveNodesGroupFactory, family="lgl"),
    PolynomialEquidistantSimplexGroupFactory,
    LegendreGaussLobattoTensorProductGroupFactory,
    ])
def test_inverse_modal_connections(actx_factory, nodal_group_factory):

    if nodal_group_factory is LegendreGaussLobattoTensorProductGroupFactory:
        group_cls = TensorProductElementGroup
        modal_group_factory = ModalTensorProductGroupFactory
    else:
        group_cls = SimplexElementGroup
        modal_group_factory = ModalSimplexGroupFactory

    actx = actx_factory()
    order = 4

    def f(x):
        return 2*actx.np.sin(20*x) + 0.5*actx.np.cos(10*x)

    # Make a regular rectangle mesh
    mesh = mgen.generate_regular_rect_mesh(
        a=(0, 0)*2, b=(5, 3), n=(10, 6,), order=order, group_cls=group_cls)

    # Make discretizations
    nodal_disc = NodalDiscretization(actx, mesh, nodal_group_factory(order))
    modal_disc = ModalDiscretization(actx, mesh, modal_group_factory(order))

    # Make connections
    nodal_to_modal_conn = ModalDiscretizationConnection(
        nodal_disc, modal_disc
    )
    modal_to_nodal_conn = ModalInverseDiscretizationConnection(
        modal_disc, nodal_disc
    )

    x_nodal = thaw(actx, nodal_disc.nodes()[0])
    nodal_f = f(x_nodal)

    # Map nodal coefficients of f to modal coefficients
    modal_f = nodal_to_modal_conn(nodal_f)
    # Now map the modal coefficients back to nodal
    nodal_f_2 = modal_to_nodal_conn(modal_f)

    # This error should be small since we composed a map with
    # its inverse
    err = actx.np.linalg.norm(nodal_f - nodal_f_2)

    assert err <= 1e-13


@pytest.mark.parametrize("quad_group_factory", [
    QuadratureSimplexGroupFactory
    ])
def test_modal_coefficients_by_projection(actx_factory, quad_group_factory):
    group_cls = SimplexElementGroup
    modal_group_factory = ModalSimplexGroupFactory
    actx = actx_factory()
    order = 10
    m_order = 5

    # Make a regular rectangle mesh
    mesh = mgen.generate_regular_rect_mesh(
        a=(0, 0)*2, b=(5, 3), n=(10, 6,), order=order, group_cls=group_cls)

    # Make discretizations
    nodal_disc = NodalDiscretization(actx, mesh, quad_group_factory(order))
    modal_disc = ModalDiscretization(actx, mesh, modal_group_factory(m_order))

    # Make connections one using quadrature projection
    nodal_to_modal_conn_quad = ModalDiscretizationConnection(
        nodal_disc, modal_disc, allow_approximate_quad=True
    )

    def f(x):
        return 2*actx.np.sin(5*x)

    x_nodal = thaw(actx, nodal_disc.nodes()[0])
    nodal_f = f(x_nodal)

    # Compute modal coefficients we expect to get
    import modepy as mp

    grp, = nodal_disc.groups
    shape = mp.Simplex(grp.dim)
    space = mp.space_for_shape(shape, order=m_order)
    basis = mp.orthonormal_basis_for_space(space, shape)
    quad = grp._quadrature_rule()

    nodal_f_data = actx.to_numpy(nodal_f[0])
    vdm = mp.vandermonde(basis.functions, quad.nodes)
    w_diag = np.diag(quad.weights)

    modal_data = []
    for _, nodal_data in enumerate(nodal_f_data):
        # Compute modal data in each element: V.T * W * nodal_data
        elem_modal_f = np.dot(vdm.T, np.dot(w_diag, nodal_data))
        modal_data.append(elem_modal_f)

    modal_data = actx.from_numpy(np.asarray(modal_data))
    modal_f_expected = DOFArray(actx, data=(modal_data,))

    # Map nodal coefficients using the quadrature-based projection
    modal_f_computed = nodal_to_modal_conn_quad(nodal_f)

    err = actx.np.linalg.norm(modal_f_expected - modal_f_computed)

    assert err <= 1e-13


@pytest.mark.parametrize("quad_group_factory", [
    QuadratureSimplexGroupFactory
    ])
def test_quadrature_based_modal_connection_reverse(actx_factory, quad_group_factory):

    group_cls = SimplexElementGroup
    modal_group_factory = ModalSimplexGroupFactory
    actx = actx_factory()
    order = 10
    m_order = 5

    # Make a regular rectangle mesh
    mesh = mgen.generate_regular_rect_mesh(
        a=(0, 0)*2, b=(5, 3), n=(10, 6,), order=order, group_cls=group_cls)

    # Make discretizations
    nodal_disc = NodalDiscretization(actx, mesh, quad_group_factory(order))
    modal_disc = ModalDiscretization(actx, mesh, modal_group_factory(m_order))

    # Make connections one using quadrature projection
    nodal_to_modal_conn_quad = ModalDiscretizationConnection(
        nodal_disc, modal_disc
    )

    # And the reverse connection
    modal_to_nodal_conn = ModalInverseDiscretizationConnection(
        modal_disc, nodal_disc
    )

    def f(x):
        return 1 + 2*x + 3*x**2

    x_nodal = thaw(actx, nodal_disc.nodes()[0])
    nodal_f = f(x_nodal)

    # Map nodal coefficients using the quadrature-based projection
    modal_f_quad = nodal_to_modal_conn_quad(nodal_f)

    # Back to nodal
    nodal_f_computed = modal_to_nodal_conn(modal_f_quad)

    err = actx.np.linalg.norm(nodal_f - nodal_f_computed)

    assert err <= 1e-11


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
