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


from dataclasses import dataclass
from functools import partial
import numpy as np
import numpy.linalg as la

from pytools.obj_array import make_obj_array

import meshmode                       # noqa: F401
from meshmode.array_context import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests
    )

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

from meshmode.dof_array import thaw, flatten

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
        nodal_disc, modal_disc)
    modal_to_nodal_conn = ModalInverseDiscretizationConnection(
        modal_disc, nodal_disc)

    x_nodal = thaw(actx, nodal_disc.nodes()[0])
    nodal_f = f(x_nodal)

    # Map nodal coefficients of f to modal coefficients
    modal_f = nodal_to_modal_conn(nodal_f)
    # Now map the modal coefficients back to nodal
    nodal_f_2 = modal_to_nodal_conn(modal_f)

    # This error should be small since we composed a map with
    # its inverse
    err = actx.np.linalg.norm(nodal_f - nodal_f_2)
    assert err <= 1e-12


@pytest.mark.parametrize("quad_group_factory", [
    InterpolatoryQuadratureSimplexGroupFactory
    #QuadratureSimplexGroupFactory
    ])
def test_quadrature_based_modal_connection(actx_factory, quad_group_factory):

    group_cls = SimplexElementGroup
    modal_group_factory = ModalSimplexGroupFactory
    actx = actx_factory()
    order = 4
    m_order = 4

    # Make a regular rectangle mesh
    mesh = mgen.generate_regular_rect_mesh(
        a=(0, 0)*2, b=(5, 3), n=(10, 6,), order=order, group_cls=group_cls)

    # Make discretizations
    nodal_disc = NodalDiscretization(actx, mesh, quad_group_factory(order))
    modal_disc = ModalDiscretization(actx, mesh, modal_group_factory(m_order))

    # Make connections one using quadrature projection and one using
    # the Vandermonde inverse
    nodal_to_modal_conn_quad = ModalDiscretizationConnection(
        nodal_disc, modal_disc, allow_approximate_quad=True)
    nodal_to_modal_conn_vinv = ModalDiscretizationConnection(
        nodal_disc, modal_disc, allow_approximate_quad=False)

    def f(x):
        return 0.25*actx.np.cos(2*x)

    x_nodal = thaw(actx, nodal_disc.nodes()[0])
    nodal_f = f(x_nodal)

    # Map nodal coefficients using the quadrature-based projection
    modal_f_quad = nodal_to_modal_conn_quad(nodal_f)
    modal_f_vinv = nodal_to_modal_conn_vinv(nodal_f)


    err = actx.np.linalg.norm(modal_f_quad - modal_f_vinv)
    assert err <= 1e-8



if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
