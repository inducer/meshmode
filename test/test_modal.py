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

from arraycontext import thaw, _acf     # noqa: F401
from arraycontext import (              # noqa: F401
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests
    )

from meshmode.dof_array import DOFArray, flat_norm
from meshmode.mesh import (
    SimplexElementGroup,
    TensorProductElementGroup
    )

from meshmode.discretization.poly_element import (
    # Simplex group factories
    ModalSimplexGroupFactory,
    InterpolatoryQuadratureSimplexGroupFactory,
    PolynomialWarpAndBlend2DRestrictingGroupFactory,
    PolynomialWarpAndBlend3DRestrictingGroupFactory,
    PolynomialRecursiveNodesGroupFactory,
    PolynomialEquidistantSimplexGroupFactory,
    # Tensor product group factories
    ModalTensorProductGroupFactory,
    LegendreGaussLobattoTensorProductGroupFactory,
    # Quadrature-based (non-interpolatory) group factories
    QuadratureSimplexGroupFactory
    )

from meshmode.discretization import Discretization
from meshmode.discretization.connection.modal import (
    NodalToModalDiscretizationConnection,
    ModalToNodalDiscretizationConnection
    )

import meshmode.mesh.generation as mgen
import pytest


@pytest.mark.parametrize("nodal_group_factory", [
    InterpolatoryQuadratureSimplexGroupFactory,
    PolynomialWarpAndBlend2DRestrictingGroupFactory,
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
        a=(0, 0), b=(5, 3), npoints_per_axis=(10, 6), order=order,
        group_cls=group_cls)

    # Make discretizations
    nodal_disc = Discretization(actx, mesh, nodal_group_factory(order))
    modal_disc = Discretization(actx, mesh, modal_group_factory(order))

    # Make connections
    nodal_to_modal_conn = NodalToModalDiscretizationConnection(
        nodal_disc, modal_disc
    )
    modal_to_nodal_conn = ModalToNodalDiscretizationConnection(
        modal_disc, nodal_disc
    )

    x_nodal = thaw(nodal_disc.nodes()[0], actx)
    nodal_f = f(x_nodal)

    # Map nodal coefficients of f to modal coefficients
    modal_f = nodal_to_modal_conn(nodal_f)
    # Now map the modal coefficients back to nodal
    nodal_f_2 = modal_to_nodal_conn(modal_f)

    # This error should be small since we composed a map with
    # its inverse
    err = flat_norm(nodal_f - nodal_f_2)

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
        a=(0, 0), b=(5, 3), npoints_per_axis=(10, 6), order=order,
        group_cls=group_cls)

    # Make discretizations
    nodal_disc = Discretization(actx, mesh, quad_group_factory(order))
    modal_disc = Discretization(actx, mesh, modal_group_factory(m_order))

    # Make connections one using quadrature projection
    nodal_to_modal_conn_quad = NodalToModalDiscretizationConnection(
        nodal_disc, modal_disc, allow_approximate_quad=True
    )

    def f(x):
        return 2*actx.np.sin(5*x)

    x_nodal = thaw(nodal_disc.nodes()[0], actx)
    nodal_f = f(x_nodal)

    # Compute modal coefficients we expect to get
    import modepy as mp

    grp, = nodal_disc.groups
    shape = mp.Simplex(grp.dim)
    space = mp.space_for_shape(shape, order=m_order)
    basis = mp.orthonormal_basis_for_space(space, shape)
    quad = grp.quadrature_rule()

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

    err = flat_norm(modal_f_expected - modal_f_computed)

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
        a=(0, 0), b=(5, 3), npoints_per_axis=(10, 6), order=order,
        group_cls=group_cls)

    # Make discretizations
    nodal_disc = Discretization(actx, mesh, quad_group_factory(order))
    modal_disc = Discretization(actx, mesh, modal_group_factory(m_order))

    # Make connections one using quadrature projection
    nodal_to_modal_conn_quad = NodalToModalDiscretizationConnection(
        nodal_disc, modal_disc
    )

    # And the reverse connection
    modal_to_nodal_conn = ModalToNodalDiscretizationConnection(
        modal_disc, nodal_disc
    )

    def f(x):
        return 1 + 2*x + 3*x**2

    x_nodal = thaw(nodal_disc.nodes()[0], actx)
    nodal_f = f(x_nodal)

    # Map nodal coefficients using the quadrature-based projection
    modal_f_quad = nodal_to_modal_conn_quad(nodal_f)

    # Back to nodal
    nodal_f_computed = modal_to_nodal_conn(modal_f_quad)

    err = flat_norm(nodal_f - nodal_f_computed)

    assert err <= 1e-11


@pytest.mark.parametrize("nodal_group_factory", [
    InterpolatoryQuadratureSimplexGroupFactory,
    "warp_and_blend",
    LegendreGaussLobattoTensorProductGroupFactory,
    ])
@pytest.mark.parametrize(("dim", "mesh_pars"), [
    (2, [10, 20, 30]),
    (3, [10, 20, 30]),
    ])
def test_modal_truncation(actx_factory, nodal_group_factory,
                          dim, mesh_pars):

    if nodal_group_factory == "warp_and_blend":
        nodal_group_factory = {
                2: PolynomialWarpAndBlend2DRestrictingGroupFactory,
                3: PolynomialWarpAndBlend3DRestrictingGroupFactory,
                }[dim]

    if nodal_group_factory is LegendreGaussLobattoTensorProductGroupFactory:
        group_cls = TensorProductElementGroup
        modal_group_factory = ModalTensorProductGroupFactory
    else:
        group_cls = SimplexElementGroup
        modal_group_factory = ModalSimplexGroupFactory

    actx = actx_factory()
    order = 5
    truncated_order = 3

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    def f(x):
        return actx.np.sin(2*x)

    for mesh_par in mesh_pars:

        # Make the mesh
        mesh = mgen.generate_warped_rect_mesh(dim, order=order,
                                              nelements_side=mesh_par,
                                              group_cls=group_cls)
        h = 1/mesh_par

        # Make discretizations
        nodal_disc = Discretization(actx, mesh, nodal_group_factory(order))
        modal_disc = Discretization(actx, mesh, modal_group_factory(order))

        # Make connections (nodal -> modal)
        nodal_to_modal_conn = NodalToModalDiscretizationConnection(
            nodal_disc, modal_disc
        )

        # And the reverse connection (modal -> nodal)
        modal_to_nodal_conn = ModalToNodalDiscretizationConnection(
            modal_disc, nodal_disc
        )

        x_nodal = thaw(nodal_disc.nodes()[0], actx)
        nodal_f = f(x_nodal)

        # Map to modal
        modal_f = nodal_to_modal_conn(nodal_f)

        # Now we compute the basis function indices corresonding
        # to modes > truncated_order
        mgrp, = modal_disc.groups
        mgrp_mode_ids = mgrp.basis_obj().mode_ids
        truncation_matrix = np.identity(len(mgrp_mode_ids))
        for mode_idx, mode_id in enumerate(mgrp_mode_ids):
            if sum(mode_id) > truncated_order:
                truncation_matrix[mode_idx, mode_idx] = 0

        # Zero out the modal coefficients corresponding to
        # the targeted modes.
        modal_f_data = actx.to_numpy(modal_f[0])
        num_elem, _ = modal_f_data.shape
        for el_idx in range(num_elem):
            modal_f_data[el_idx] = np.dot(truncation_matrix,
                                          modal_f_data[el_idx])

        modal_f_data = actx.from_numpy(modal_f_data)
        modal_f_truncated = DOFArray(actx, data=(modal_f_data,))

        # Now map truncated modal coefficients back to nodal
        nodal_f_truncated = modal_to_nodal_conn(modal_f_truncated)

        err = flat_norm(nodal_f - nodal_f_truncated)
        eoc_rec.add_data_point(h, actx.to_numpy(err))
        threshold_lower = 0.8*truncated_order
        threshold_upper = 1.2*truncated_order

    assert threshold_upper >= eoc_rec.order_estimate() >= threshold_lower


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
