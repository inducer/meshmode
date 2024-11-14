__copyright__ = "Copyright (C) 2021 Alexandru Fikl"

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
from typing import Optional, Type

import numpy as np

import pyopencl as cl
from pytools import keyed_memoize_in
from pytools.obj_array import make_obj_array

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.transform_metadata import FirstAxisIsElementsTag


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def plot_solution(actx, vis, filename, discr, t, x):
    names_and_fields = []

    try:
        from pytential import bind, sym
        kappa = bind(discr, sym.mean_curvature(discr.ambient_dim))(actx)
        names_and_fields.append(("kappa", kappa))
    except ImportError:
        pass

    vis.write_vtk_file(filename, names_and_fields, overwrite=True)


def reconstruct_discr_from_nodes(actx, discr, x):
    @keyed_memoize_in(actx,
            (reconstruct_discr_from_nodes, "to_mesh_interp_matrix"),
            lambda grp: grp.discretization_key())
    def to_mesh_interp_matrix(grp) -> np.ndarray:
        import modepy as mp
        mat = mp.resampling_matrix(
                grp.basis_obj().functions,
                grp.mesh_el_group.unit_nodes,
                grp.unit_nodes)

        return actx.freeze(actx.from_numpy(mat))

    def resample_nodes_to_mesh(grp, igrp, iaxis):
        discr_nodes = x[iaxis][igrp]

        grp_unit_nodes = grp.unit_nodes.reshape(-1)
        meg_unit_nodes = grp.mesh_el_group.unit_nodes.reshape(-1)

        tol = 10 * np.finfo(grp_unit_nodes.dtype).eps
        if (grp_unit_nodes.shape == meg_unit_nodes.shape
                and np.linalg.norm(grp_unit_nodes - meg_unit_nodes) < tol):
            return discr_nodes

        return actx.einsum("ij,ej->ei",
                           to_mesh_interp_matrix(grp),
                           discr_nodes,
                           tagged=(FirstAxisIsElementsTag(),))

    from dataclasses import replace

    megs = []
    for igrp, grp in enumerate(discr.groups):
        nodes = np.stack([
            actx.to_numpy(resample_nodes_to_mesh(grp, igrp, iaxis))
            for iaxis in range(discr.ambient_dim)
            ])

        meg = replace(grp.mesh_el_group, vertex_indices=None, nodes=nodes)
        megs.append(meg)

    mesh = discr.mesh.copy(groups=megs, vertices=None)
    return discr.copy(actx, mesh=mesh)


def advance(actx, dt, t, x, fn):
    # NOTE: everybody's favorite three stage SSP RK3 method
    k1 = x + dt * fn(t, x)
    k2 = 3.0 / 4.0 * x + 1.0 / 4.0 * (k1 + dt * fn(t + dt, k1))
    return 1.0 / 3.0 * x + 2.0 / 3.0 * (k2 + dt * fn(t + 0.5 * dt, k2))


def run(actx, *,
        ambient_dim: int = 3,
        resolution: Optional[int] = None,
        target_order: int = 4,
        tmax: float = 1.0,
        timestep: float = 1.0e-2,
        group_factory_name: str = "warp_and_blend",
        visualize: bool = True):
    if ambient_dim not in (2, 3):
        raise ValueError(f"unsupported dimension: {ambient_dim}")

    mesh_order = target_order
    radius = 1.0

    # {{{ geometry

    # {{{ element groups

    import modepy as mp

    import meshmode.discretization.poly_element as poly

    # NOTE: picking the same unit nodes for the mesh and the discr saves
    # a bit of work when reconstructing after a time step

    if group_factory_name == "warp_and_blend":
        group_factory_cls: Type[poly.HomogeneousOrderBasedGroupFactory] = (
            poly.PolynomialWarpAndBlend2DRestrictingGroupFactory)

        unit_nodes = mp.warp_and_blend_nodes(ambient_dim - 1, mesh_order)
    elif group_factory_name == "quadrature":
        group_factory_cls = poly.InterpolatoryQuadratureSimplexGroupFactory

        if ambient_dim == 2:
            unit_nodes = mp.LegendreGaussQuadrature(
                    mesh_order, force_dim_axis=True).nodes
        else:
            unit_nodes = mp.VioreanuRokhlinSimplexQuadrature(mesh_order, 2).nodes
    else:
        raise ValueError(f"unknown group factory: '{group_factory_name}'")

    # }}}

    # {{{ discretization

    import meshmode.mesh.generation as gen
    if ambient_dim == 2:
        nelements = 8192 if resolution is None else resolution
        mesh = gen.make_curve_mesh(
                lambda t: radius * gen.ellipse(1.0, t),
                np.linspace(0.0, 1.0, nelements + 1),
                order=mesh_order,
                unit_nodes=unit_nodes)
    else:
        nrounds = 4 if resolution is None else resolution
        mesh = gen.generate_sphere(radius,
                uniform_refinement_rounds=nrounds,
                order=mesh_order,
                unit_nodes=unit_nodes)

    from meshmode.discretization import Discretization
    discr0 = Discretization(actx, mesh, group_factory_cls(target_order))

    logger.info("ndofs:     %d", discr0.ndofs)
    logger.info("nelements: %d", discr0.mesh.nelements)

    # }}}

    if visualize:
        from meshmode.discretization.visualization import make_visualizer
        vis = make_visualizer(actx, discr0,
                vis_order=target_order,
                # NOTE: setting this to True will add some unnecessary
                # resampling in Discretization.nodes for the vis_discr underneath
                force_equidistant=False)

    # }}}

    # {{{ ode

    def velocity_field(nodes, alpha=1.0):
        return make_obj_array([
            alpha * nodes[0], -alpha * nodes[1], 0.0 * nodes[0]
            ][:ambient_dim])

    def source(t, x):
        discr = reconstruct_discr_from_nodes(actx, discr0, x)
        u = velocity_field(actx.thaw(discr.nodes()))

        # {{{

        # NOTE: these are just here because this was at some point used to
        # profile some more operators (turned out well!)

        from meshmode.discretization import num_reference_derivative
        x = actx.thaw(discr.nodes()[0])
        gradx = sum(
                num_reference_derivative(discr, (i,), x)
                for i in range(discr.dim))
        intx = sum(
            actx.np.sum(xi * wi)
            for xi, wi in zip(x, discr.quad_weights(), strict=True))

        assert gradx is not None
        assert intx is not None

        # }}}

        return u

    # }}}

    # {{{ evolve

    maxiter = int(tmax // timestep) + 1
    dt = tmax / maxiter + 1.0e-15

    x = actx.thaw(discr0.nodes())
    t = 0.0

    if visualize:
        filename = f"moving-geometry-{0:09d}.vtu"
        plot_solution(actx, vis, filename, discr0, t, x)

    for n in range(1, maxiter + 1):
        x = advance(actx, dt, t, x, source)
        t += dt

        if visualize:
            discr = reconstruct_discr_from_nodes(actx, discr0, x)
            vis = make_visualizer(actx, discr, vis_order=target_order)
            # vis = vis.copy_with_same_connectivity(actx, discr)

            filename = f"moving-geometry-{n:09d}.vtu"
            plot_solution(actx, vis, filename, discr, t, x)

        logger.info("[%05d/%05d] t = %.5e/%.5e dt = %.5e",
                n, maxiter, t, tmax, dt)

    # }}}


if __name__ == "__main__":
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue, force_device_scalars=True)

    from pytools import ProcessTimer
    for _ in range(1):
        with ProcessTimer() as p:
            run(actx,
                    ambient_dim=3,
                    group_factory_name="warp_and_blend",
                    tmax=1.0,
                    timestep=1.0e-2,
                    visualize=False)

        logger.info("elapsed: %.3fs wall %.2fx cpu",
                p.wall_elapsed, p.process_elapsed / p.wall_elapsed)
