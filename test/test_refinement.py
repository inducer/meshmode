from __future__ import annotations


__copyright__ = "Copyright (C) 2014-6 Shivam Gupta, Andreas Kloeckner"

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
import pathlib
from functools import partial

import numpy as np
import pytest

from arraycontext import ArrayContextFactory, pytest_generate_tests_for_array_contexts

import meshmode.mesh.generation as mgen
from meshmode import _acf  # noqa: F401
from meshmode.array_context import PytestPyOpenCLArrayContextFactory
from meshmode.discretization.poly_element import (
    GaussLegendreTensorProductGroupFactory,
    InterpolatoryQuadratureSimplexGroupFactory,
    LegendreGaussLobattoTensorProductGroupFactory,
    PolynomialEquidistantSimplexGroupFactory,
    PolynomialWarpAndBlend2DRestrictingGroupFactory,
    PolynomialWarpAndBlend3DRestrictingGroupFactory,
)
from meshmode.dof_array import flat_norm
from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
from meshmode.mesh.refinement import RefinerWithoutAdjacency


logger = logging.getLogger(__name__)
pytest_generate_tests = pytest_generate_tests_for_array_contexts(
        [PytestPyOpenCLArrayContextFactory])

thisdir = pathlib.Path(__file__).parent


def get_blob_mesh(mesh_par, order=4):
    # from meshmode.mesh.io import generate_gmsh, FileSource
    # return generate_gmsh(
    #         FileSource("blob-2d.step"), 2, order=order,
    #         force_ambient_dim=2,
    #         other_options=[
    #             "-string", "Mesh.CharacteristicLengthMax = %s;" % mesh_par]
    #         )

    from meshmode.mesh.io import read_gmsh
    return read_gmsh(
            str(thisdir / f"blob2d-order{order}-h{mesh_par}.msh"),
            force_ambient_dim=2)


def random_refine_flags(fract, mesh):
    all_els = list(range(mesh.nelements))

    flags = np.zeros(mesh.nelements)
    from random import seed, shuffle
    seed(17)
    shuffle(all_els)
    for i in range(int(mesh.nelements * fract)):
        flags[all_els[i]] = 1

    return flags


def even_refine_flags(spacing, mesh):
    flags = np.zeros(mesh.nelements)
    flags[::spacing] = 1
    return flags


def empty_refine_flags(mesh):
    return np.zeros(mesh.nelements)


def uniform_refine_flags(mesh):
    return np.ones(mesh.nelements)


@pytest.mark.parametrize("group_factory", [
    InterpolatoryQuadratureSimplexGroupFactory,
    "warp_and_blend",
    PolynomialEquidistantSimplexGroupFactory,

    LegendreGaussLobattoTensorProductGroupFactory,
    GaussLegendreTensorProductGroupFactory,
    ])
@pytest.mark.parametrize(("mesh_name", "dim", "mesh_pars"), [
    ("circle", 1, [20, 30, 40]),
    ("blob", 2, ["8e-2", "6e-2", "4e-2"]),
    ("warp", 2, [4, 5, 6]),
    ("warp", 3, [4, 5, 6]),
])
@pytest.mark.parametrize("mesh_order", [1, 4, 5])
@pytest.mark.parametrize("refine_flags", [
    # FIXME: slow
    #uniform_refine_flags,
    #partial(random_refine_flags, 0.4)
    partial(even_refine_flags, 2)
])
# test_refinement_connection(cl._csc, PolynomialWarpAndBlend2DRestrictingGroupFactory, 'warp', 2, [4, 5, 6], 5, partial(even_refine_flags, 2))  # noqa: E501
def test_refinement_connection(
        actx_factory: ArrayContextFactory, group_factory,
        mesh_name, dim, mesh_pars, mesh_order, refine_flags, visualize=False):

    if group_factory == "warp_and_blend":
        group_factory = {
                1: PolynomialWarpAndBlend2DRestrictingGroupFactory,
                2: PolynomialWarpAndBlend2DRestrictingGroupFactory,
                3: PolynomialWarpAndBlend3DRestrictingGroupFactory,
                }[dim]

    group_cls = group_factory.mesh_group_class
    if issubclass(group_cls, TensorProductElementGroup):
        if mesh_name in ["circle", "blob"]:
            pytest.skip("mesh does not have tensor product support")

    from random import seed
    seed(13)

    actx = actx_factory()

    # discretization order
    order = 5

    from pytools.convergence import EOCRecorder

    from meshmode.discretization import Discretization
    from meshmode.discretization.connection import (
        check_connection,
        make_refinement_connection,
    )
    eoc_rec = EOCRecorder()

    for mesh_par in mesh_pars:
        # {{{ get mesh

        if mesh_name == "circle":
            assert dim == 1
            h = 1 / mesh_par
            mesh = mgen.make_curve_mesh(
                mgen.circle,
                np.linspace(0, 1, mesh_par + 1),
                order=mesh_order)
        elif mesh_name == "blob":
            if mesh_order == 5:
                pytest.xfail("https://gitlab.tiker.net/inducer/meshmode/issues/2")
            assert dim == 2
            mesh = get_blob_mesh(mesh_par, mesh_order)
            h = float(mesh_par)
        elif mesh_name == "warp":
            # FIXME: Leftover from n -> nelements_per_axis/npoints_per_axis change;
            # should be nelements_per_axis, but if changed EOC order dips below
            # threshold and test fails. Likely just need to tweak mesh sizes.
            mesh = mgen.generate_warped_rect_mesh(dim, order=mesh_order,
                    npoints_side=mesh_par, group_cls=group_cls)
            h = 1/mesh_par
        else:
            raise ValueError("mesh_name not recognized")

        # }}}

        from meshmode.mesh.processing import find_bounding_box
        mesh_bbox_low, mesh_bbox_high = find_bounding_box(mesh)
        mesh_ext = mesh_bbox_high-mesh_bbox_low

        def f(x):
            result = 1
            if mesh_name == "blob":
                factor = 15
            else:
                factor = 9

            for iaxis in range(len(x)):
                result = result * actx.np.sin(
                        factor * (x[iaxis]/mesh_ext[iaxis]))  # noqa: B023

            return result

        discr = Discretization(actx, mesh, group_factory(order))

        refiner = RefinerWithoutAdjacency(mesh)
        flags = refine_flags(mesh)
        refiner.refine(flags)

        connection = make_refinement_connection(
            actx, refiner, discr, group_factory(order))
        check_connection(actx, connection)

        fine_discr = connection.to_discr

        x = actx.thaw(discr.nodes())
        x_fine = actx.thaw(fine_discr.nodes())
        f_coarse = f(x)
        f_interp = connection(f_coarse)
        f_true = f(x_fine)

        if visualize == "dots":
            import matplotlib.cm as cm
            import matplotlib.pyplot as plt

            x = x.get(actx.queue)
            err = np.array(np.log10(
                1e-16 + np.abs((f_interp - f_true).get(actx.queue))), dtype=float)

            # pylint: disable=no-member
            cmap = cm.ScalarMappable(cmap=cm.jet)
            cmap.set_array(err)
            plt.scatter(x[0], x[1], c=cmap.to_rgba(err), s=20, cmap=cmap)
            plt.colorbar(cmap)
            plt.show()

        elif visualize == "vtk":
            from meshmode.discretization.visualization import make_visualizer
            fine_vis = make_visualizer(actx, fine_discr, mesh_order)

            fine_vis.write_vtk_file(
                    f"refine-fine-{mesh_name}-{dim}d-{mesh_par}.vtu", [
                        ("f_interp", f_interp),
                        ("f_true", f_true),
                        ])

        err = flat_norm(f_interp - f_true, np.inf)
        eoc_rec.add_data_point(h, actx.to_numpy(err))

    order_slack = 0.5
    if mesh_name == "blob" and order > 1:
        order_slack = 1

    print(eoc_rec)
    assert (
            eoc_rec.order_estimate() >= order-order_slack
            or eoc_rec.max_error() < 1e-14)


@pytest.mark.parametrize(("group_cls", "with_adjacency"), [
    (SimplexElementGroup, True),
    (SimplexElementGroup, False),
    (TensorProductElementGroup, False)
    ])
def test_uniform_refinement(group_cls, with_adjacency):
    make_mesh = partial(mgen.generate_box_mesh, (
            np.linspace(0.0, 1.0, 2),
            np.linspace(0.0, 1.0, 3),
            np.linspace(0.0, 1.0, 2)),
            order=4, group_cls=group_cls)
    mesh = make_mesh()

    from meshmode.mesh.refinement import refine_uniformly
    mesh = refine_uniformly(mesh, 1, with_adjacency=with_adjacency)


@pytest.mark.parametrize("refinement_rounds", [0, 1, 2])
def test_conformity_of_uniform_mesh(refinement_rounds):
    mesh = mgen.generate_sphere(r=1.0, order=4,
            uniform_refinement_rounds=refinement_rounds)

    assert mesh.is_conforming

    from meshmode.mesh import BTAG_ALL, is_boundary_tag_empty
    assert is_boundary_tag_empty(mesh, BTAG_ALL)


@pytest.mark.parametrize("mesh_name", ["torus", "icosphere", "cylinder"])
def test_refine_surfaces(actx_factory: ArrayContextFactory, mesh_name, visualize=False):
    if mesh_name == "torus":
        mesh = mgen.generate_torus(10, 1, 40, 4, order=4)
    elif mesh_name == "icosphere":
        mesh = mgen.generate_sphere(1, order=4)
    elif mesh_name == "cylinder":
        mesh = mgen.generate_surface_of_revolution(
            lambda x, y: np.ones(x.shape),
            np.linspace(1, 2),
            np.linspace(0, 2*np.pi, 6, endpoint=False), order=4)
    else:
        raise ValueError(f"invalid mesh name '{mesh_name}'")

    if visualize:
        actx = actx_factory()
        from meshmode.mesh.visualization import vtk_visualize_mesh
        vtk_visualize_mesh(actx, mesh, "surface.vtu")

    # check for absence of node-vertex consistency error
    from meshmode.mesh.refinement import refine_uniformly
    refined_mesh = refine_uniformly(mesh, 1)

    if visualize:
        actx = actx_factory()
        from meshmode.mesh.visualization import vtk_visualize_mesh
        vtk_visualize_mesh(actx, refined_mesh, "surface-refined.vtu")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
