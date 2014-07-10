from __future__ import division

__copyright__ = "Copyright (C) 2014 Andreas Kloeckner"

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
import numpy.linalg as la
import pyopencl as cl
import pyopencl.array  # noqa
import pyopencl.clmath  # noqa

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)

import pytest

import logging
logger = logging.getLogger(__name__)


def test_boundary_interpolation(ctx_getter):
    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    from meshmode.mesh.io import generate_gmsh, FileSource
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
    from meshmode.discretization.connection import make_boundary_restriction

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    order = 4

    for h in [1e-1, 3e-2, 1e-2]:
        print "BEGIN GEN"
        mesh = generate_gmsh(
                FileSource("blob-2d.step"), 2, order=order,
                force_ambient_dim=2,
                other_options=[
                    "-string", "Mesh.CharacteristicLengthMax = %s;" % h]
                )
        print "END GEN"

        vol_discr = Discretization(cl_ctx, mesh,
                InterpolatoryQuadratureSimplexGroupFactory(order))
        print "h=%s -> %d elements" % (
                h, sum(mgrp.nelements for mgrp in mesh.groups))

        x = vol_discr.nodes()[0].with_queue(queue)
        f = 0.1*cl.clmath.sin(30*x)

        bdry_mesh, bdry_discr, bdry_connection = make_boundary_restriction(
                queue, vol_discr, InterpolatoryQuadratureSimplexGroupFactory(order))

        bdry_x = bdry_discr.nodes()[0].with_queue(queue)
        bdry_f = 0.1*cl.clmath.sin(30*bdry_x)
        bdry_f_2 = bdry_connection(queue, f)

        err = la.norm((bdry_f-bdry_f_2).get(), np.inf)
        eoc_rec.add_data_point(h, err)

    print eoc_rec
    assert eoc_rec.order_estimate() >= order-0.5


def test_element_orientation():
    from meshmode.mesh.io import generate_gmsh, FileSource

    mesh_order = 3

    mesh = generate_gmsh(
            FileSource("blob-2d.step"), 2, order=mesh_order,
            force_ambient_dim=2,
            other_options=["-string", "Mesh.CharacteristicLengthMax = 0.02;"]
            )

    from meshmode.mesh.processing import (perform_flips,
            find_volume_mesh_element_orientations)
    mesh_orient = find_volume_mesh_element_orientations(mesh)

    assert (mesh_orient > 0).all()

    from random import randrange
    flippy = np.zeros(mesh.nelements, np.int8)
    for i in range(int(0.3*mesh.nelements)):
        flippy[randrange(0, mesh.nelements)] = 1

    mesh = perform_flips(mesh, flippy, skip_tests=True)

    mesh_orient = find_volume_mesh_element_orientations(mesh)

    assert ((mesh_orient < 0) == (flippy > 0)).all()


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("order", [1, 3])
def test_sanity_single_element(ctx_getter, dim, order, visualize=False):
    pytest.importorskip("pytential")

    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    from modepy.tools import UNIT_VERTICES
    vertices = UNIT_VERTICES[dim].T.copy()

    center = np.empty(dim, np.float64)
    center.fill(-0.5)

    import modepy as mp
    from meshmode.mesh import SimplexElementGroup, Mesh
    mg = SimplexElementGroup(
            order=order,
            vertex_indices=np.arange(dim+1, dtype=np.int32).reshape(1, -1),
            nodes=mp.warp_and_blend_nodes(dim, order).reshape(dim, 1, -1),
            dim=dim)

    mesh = Mesh(vertices, [mg])

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            PolynomialWarpAndBlendGroupFactory
    vol_discr = Discretization(cl_ctx, mesh,
            PolynomialWarpAndBlendGroupFactory(order+3))

    # {{{ volume calculation check

    vol_x = vol_discr.nodes().with_queue(queue)

    vol_one = vol_x[0].copy()
    vol_one.fill(1)
    from pytential import norm, integral  # noqa

    from pytools import factorial
    true_vol = 1/factorial(dim) * 2**dim

    comp_vol = integral(vol_discr, queue, vol_one)
    rel_vol_err = abs(true_vol - comp_vol) / true_vol

    assert rel_vol_err < 1e-12

    # }}}

    # {{{ boundary discretization

    from meshmode.discretization.connection import make_boundary_restriction
    bdry_mesh, bdry_discr, bdry_connection = make_boundary_restriction(
            queue, vol_discr, PolynomialWarpAndBlendGroupFactory(order + 3))

    # }}}

    # {{{ visualizers

    from meshmode.discretization.visualization import make_visualizer
    #vol_vis = make_visualizer(queue, vol_discr, 4)
    bdry_vis = make_visualizer(queue, bdry_discr, 4)

    # }}}

    from pytential import bind, sym
    bdry_normals = bind(bdry_discr, sym.normal())(queue).as_vector(dtype=object)

    if visualize:
        bdry_vis.write_vtk_file("boundary.vtu", [
            ("bdry_normals", bdry_normals)
            ])

    from pytential import bind, sym
    normal_outward_check = bind(bdry_discr,
            sym.normal()
            |
            (sym.Nodes() + 0.5*sym.ones_vec(dim)),
            )(queue).as_scalar() > 0

    assert normal_outward_check.get().all(), normal_outward_check.get()


# python test_meshmode.py 'test_sanity_balls(cl._csc, "disk-radius-1.step", 2, 2, visualize=True)'  # noqa
@pytest.mark.parametrize(("src_file", "dim"), [
    ("disk-radius-1.step", 2),
    ("ball-radius-1.step", 3),
    ])
@pytest.mark.parametrize("mesh_order", [1, 2])
def test_sanity_balls(ctx_getter, src_file, dim, mesh_order,
        visualize=False):
    pytest.importorskip("pytential")

    logging.basicConfig(level=logging.INFO)

    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    from pytools.convergence import EOCRecorder
    vol_eoc_rec = EOCRecorder()
    surf_eoc_rec = EOCRecorder()

    # overkill
    quad_order = mesh_order

    from pytential import bind, sym

    for h in [0.2, 0.15, 0.1]:
        from meshmode.mesh.io import generate_gmsh, FileSource
        mesh = generate_gmsh(
                FileSource(src_file), dim, order=mesh_order,
                other_options=["-string", "Mesh.CharacteristicLengthMax = %g;" % h],
                force_ambient_dim=dim)

        logger.info("%d elements" % mesh.nelements)

        # {{{ discretizations and connections

        from meshmode.discretization import Discretization
        from meshmode.discretization.poly_element import \
                InterpolatoryQuadratureSimplexGroupFactory
        vol_discr = Discretization(ctx, mesh,
                InterpolatoryQuadratureSimplexGroupFactory(quad_order))

        from meshmode.discretization.connection import make_boundary_restriction
        bdry_mesh, bdry_discr, bdry_connection = make_boundary_restriction(
                queue, vol_discr, InterpolatoryQuadratureSimplexGroupFactory(quad_order))

        # }}}

        # {{{ visualizers

        from meshmode.discretization.visualization import make_visualizer
        vol_vis = make_visualizer(queue, vol_discr, 20)
        bdry_vis = make_visualizer(queue, bdry_discr, 20)

        # }}}

        from math import gamma
        true_surf = 2*np.pi**(dim/2)/gamma(dim/2)
        true_vol = true_surf/dim

        vol_x = vol_discr.nodes().with_queue(queue)

        vol_one = vol_x[0].copy()
        vol_one.fill(1)
        from pytential import norm, integral  # noqa

        comp_vol = integral(vol_discr, queue, vol_one)
        rel_vol_err = abs(true_vol - comp_vol) / true_vol
        vol_eoc_rec.add_data_point(h, rel_vol_err)
        print "VOL", true_vol, comp_vol

        bdry_x = bdry_discr.nodes().with_queue(queue)

        bdry_one_exact = bdry_x[0].copy()
        bdry_one_exact.fill(1)

        bdry_one = bdry_connection(queue, vol_one).with_queue(queue)
        intp_err = norm(bdry_discr, queue, bdry_one-bdry_one_exact)
        assert intp_err < 1e-14

        comp_surf = integral(bdry_discr, queue, bdry_one)
        rel_surf_err = abs(true_surf - comp_surf) / true_surf
        surf_eoc_rec.add_data_point(h, rel_surf_err)
        print "SURF", true_surf, comp_surf

        if visualize:
            vol_vis.write_vtk_file("volume-h=%g.vtu" % h, [
                ("f", vol_one),
                ("area_el", bind(vol_discr, sym.area_element())(queue)),
                ])
            bdry_vis.write_vtk_file("boundary-h=%g.vtu" % h, [("f", bdry_one)])

        # {{{ check normals point outward

        normal_outward_check = bind(bdry_discr,
                sym.normal() | sym.Nodes(),
                )(queue).as_scalar() > 0

        assert normal_outward_check.get().all(), normal_outward_check.get()

        # }}}

    print "---------------------------------"
    print "VOLUME"
    print "---------------------------------"
    print vol_eoc_rec
    assert vol_eoc_rec.order_estimate() >= mesh_order

    print "---------------------------------"
    print "SURFACE"
    print "---------------------------------"
    print surf_eoc_rec
    assert surf_eoc_rec.order_estimate() >= mesh_order


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
