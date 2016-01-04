from __future__ import division, absolute_import, print_function

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

from six.moves import range
import numpy as np
import numpy.linalg as la
import pyopencl as cl
import pyopencl.array  # noqa
import pyopencl.clmath  # noqa

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)

from meshmode.discretization.poly_element import (
        InterpolatoryQuadratureSimplexGroupFactory,
        PolynomialWarpAndBlendGroupFactory
        )
from meshmode.mesh import BTAG_ALL
from meshmode.discretization.connection import \
        FRESTR_ALL_FACES, FRESTR_INTERIOR_FACES

import pytest

import logging
logger = logging.getLogger(__name__)


# {{{ circle mesh

def test_circle_mesh(do_plot=False):
    from meshmode.mesh.io import generate_gmsh, FileSource
    print("BEGIN GEN")
    mesh = generate_gmsh(
            FileSource("circle.step"), 2, order=2,
            force_ambient_dim=2,
            other_options=[
                "-string", "Mesh.CharacteristicLengthMax = 0.05;"]
            )
    print("END GEN")
    print(mesh.nelements)

    from meshmode.mesh.processing import affine_map
    mesh = affine_map(mesh, A=3*np.eye(2))

    if do_plot:
        from meshmode.mesh.visualization import draw_2d_mesh
        draw_2d_mesh(mesh, fill=None, draw_nodal_adjacency=True,
                set_bounding_box=True)
        import matplotlib.pyplot as pt
        pt.show()

# }}}


# {{{ convergence of boundary interpolation

@pytest.mark.parametrize("group_factory", [
    InterpolatoryQuadratureSimplexGroupFactory,
    PolynomialWarpAndBlendGroupFactory
    ])
@pytest.mark.parametrize("boundary_tag", [
    BTAG_ALL,
    FRESTR_ALL_FACES,
    FRESTR_INTERIOR_FACES,
    ])
@pytest.mark.parametrize(("mesh_name", "dim", "mesh_pars"), [
    ("blob", 2, [1e-1, 8e-2, 5e-2]),
    ("warp", 2, [10, 20, 30]),
    ("warp", 3, [10, 20, 30]),
    ])
@pytest.mark.parametrize("per_face_groups", [False, True])
def test_boundary_interpolation(ctx_getter, group_factory, boundary_tag,
        mesh_name, dim, mesh_pars, per_face_groups):
    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    from meshmode.discretization import Discretization
    from meshmode.discretization.connection import make_face_restriction

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    order = 4

    def f(x):
        return 0.1*cl.clmath.sin(30*x)

    for mesh_par in mesh_pars:
        # {{{ get mesh

        if mesh_name == "blob":
            assert dim == 2

            h = mesh_par

            from meshmode.mesh.io import generate_gmsh, FileSource
            print("BEGIN GEN")
            mesh = generate_gmsh(
                    FileSource("blob-2d.step"), 2, order=order,
                    force_ambient_dim=2,
                    other_options=[
                        "-string", "Mesh.CharacteristicLengthMax = %s;" % h]
                    )
            print("END GEN")
        elif mesh_name == "warp":
            from meshmode.mesh.generation import generate_warped_rect_mesh
            mesh = generate_warped_rect_mesh(dim, order=4, n=mesh_par)

            h = 1/mesh_par
        else:
            raise ValueError("mesh_name not recognized")

        # }}}

        vol_discr = Discretization(cl_ctx, mesh,
                group_factory(order))
        print("h=%s -> %d elements" % (
                h, sum(mgrp.nelements for mgrp in mesh.groups)))

        x = vol_discr.nodes()[0].with_queue(queue)
        vol_f = f(x)

        bdry_connection = make_face_restriction(
                vol_discr, group_factory(order),
                boundary_tag, per_face_groups=per_face_groups)
        bdry_discr = bdry_connection.to_discr

        bdry_x = bdry_discr.nodes()[0].with_queue(queue)
        bdry_f = f(bdry_x)
        bdry_f_2 = bdry_connection(queue, vol_f)

        if mesh_name == "blob" and dim == 2:
            mat = bdry_connection.full_resample_matrix(queue).get(queue)
            bdry_f_2_by_mat = mat.dot(vol_f.get())

            mat_error = la.norm(bdry_f_2.get(queue=queue) - bdry_f_2_by_mat)
            assert mat_error < 1e-14, mat_error

        err = la.norm((bdry_f-bdry_f_2).get(), np.inf)
        eoc_rec.add_data_point(h, err)

    print(eoc_rec)
    assert (
            eoc_rec.order_estimate() >= order-0.5
            or eoc_rec.max_error() < 1e-14)


@pytest.mark.parametrize("group_factory", [
    InterpolatoryQuadratureSimplexGroupFactory,
    PolynomialWarpAndBlendGroupFactory
    ])
@pytest.mark.parametrize(("mesh_name", "dim", "mesh_pars"), [
    ("blob", 2, [1e-1, 8e-2, 5e-2]),
    ("warp", 2, [3, 5, 7]),
    ("warp", 3, [3, 5]),
    ])
def test_opposite_face_interpolation(ctx_getter, group_factory,
        mesh_name, dim, mesh_pars):
    logging.basicConfig(level=logging.INFO)

    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    from meshmode.discretization import Discretization
    from meshmode.discretization.connection import (
            make_face_restriction, make_opposite_face_connection)

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    order = 5

    def f(x):
        return 0.1*cl.clmath.sin(30*x)

    for mesh_par in mesh_pars:
        # {{{ get mesh

        if mesh_name == "blob":
            assert dim == 2

            h = mesh_par

            from meshmode.mesh.io import generate_gmsh, FileSource
            print("BEGIN GEN")
            mesh = generate_gmsh(
                    FileSource("blob-2d.step"), 2, order=order,
                    force_ambient_dim=2,
                    other_options=[
                        "-string", "Mesh.CharacteristicLengthMax = %s;" % h]
                    )
            print("END GEN")
        elif mesh_name == "warp":
            from meshmode.mesh.generation import generate_warped_rect_mesh
            mesh = generate_warped_rect_mesh(dim, order=4, n=mesh_par)

            h = 1/mesh_par
        else:
            raise ValueError("mesh_name not recognized")

        # }}}

        vol_discr = Discretization(cl_ctx, mesh,
                group_factory(order))
        print("h=%s -> %d elements" % (
                h, sum(mgrp.nelements for mgrp in mesh.groups)))

        bdry_connection = make_face_restriction(
                vol_discr, group_factory(order),
                FRESTR_INTERIOR_FACES)
        bdry_discr = bdry_connection.to_discr

        opp_face = make_opposite_face_connection(bdry_connection)

        bdry_x = bdry_discr.nodes()[0].with_queue(queue)
        bdry_f = f(bdry_x)

        bdry_f_2 = opp_face(queue, bdry_f)

        err = la.norm((bdry_f-bdry_f_2).get(), np.inf)
        eoc_rec.add_data_point(h, err)

    print(eoc_rec)
    assert (
            eoc_rec.order_estimate() >= order-0.5
            or eoc_rec.max_error() < 1e-13)

# }}}


# {{{ element orientation

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

# }}}


# {{{ merge and map

def test_merge_and_map(ctx_getter, visualize=False):
    from meshmode.mesh.io import generate_gmsh, FileSource

    mesh_order = 3

    mesh = generate_gmsh(
            FileSource("blob-2d.step"), 2, order=mesh_order,
            force_ambient_dim=2,
            other_options=["-string", "Mesh.CharacteristicLengthMax = 0.02;"]
            )

    from meshmode.mesh.processing import merge_disjoint_meshes, affine_map
    mesh2 = affine_map(mesh, A=np.eye(2), b=np.array([5, 0]))

    mesh3 = merge_disjoint_meshes((mesh2, mesh))

    if visualize:
        from meshmode.discretization import Discretization
        from meshmode.discretization.poly_element import \
                PolynomialWarpAndBlendGroupFactory
        cl_ctx = ctx_getter()
        queue = cl.CommandQueue(cl_ctx)

        discr = Discretization(cl_ctx, mesh3,
                PolynomialWarpAndBlendGroupFactory(3))

        from meshmode.discretization.visualization import make_visualizer
        vis = make_visualizer(queue, discr, 1)
        vis.write_vtk_file("merged.vtu", [])

# }}}


# {{{ sanity checks: single element

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
    from meshmode.mesh import SimplexElementGroup, Mesh, BTAG_ALL
    mg = SimplexElementGroup(
            order=order,
            vertex_indices=np.arange(dim+1, dtype=np.int32).reshape(1, -1),
            nodes=mp.warp_and_blend_nodes(dim, order).reshape(dim, 1, -1),
            dim=dim)

    mesh = Mesh(vertices, [mg], nodal_adjacency=None, facial_adjacency_groups=None)

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

    from meshmode.discretization.connection import make_face_restriction
    bdry_connection = make_face_restriction(
            vol_discr, PolynomialWarpAndBlendGroupFactory(order + 3),
            BTAG_ALL)
    bdry_discr = bdry_connection.to_discr

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

# }}}


# {{{ sanity checks: ball meshes

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

    for h in [0.2, 0.14, 0.1]:
        from meshmode.mesh.io import generate_gmsh, FileSource
        mesh = generate_gmsh(
                FileSource(src_file), dim, order=mesh_order,
                other_options=["-string", "Mesh.CharacteristicLengthMax = %g;" % h],
                force_ambient_dim=dim)

        logger.info("%d elements" % mesh.nelements)

        # {{{ discretizations and connections

        from meshmode.discretization import Discretization
        vol_discr = Discretization(ctx, mesh,
                InterpolatoryQuadratureSimplexGroupFactory(quad_order))

        from meshmode.discretization.connection import make_face_restriction
        bdry_connection = make_face_restriction(
                vol_discr,
                InterpolatoryQuadratureSimplexGroupFactory(quad_order),
                BTAG_ALL)
        bdry_discr = bdry_connection.to_discr

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
        print("VOL", true_vol, comp_vol)

        bdry_x = bdry_discr.nodes().with_queue(queue)

        bdry_one_exact = bdry_x[0].copy()
        bdry_one_exact.fill(1)

        bdry_one = bdry_connection(queue, vol_one).with_queue(queue)
        intp_err = norm(bdry_discr, queue, bdry_one-bdry_one_exact)
        assert intp_err < 1e-14

        comp_surf = integral(bdry_discr, queue, bdry_one)
        rel_surf_err = abs(true_surf - comp_surf) / true_surf
        surf_eoc_rec.add_data_point(h, rel_surf_err)
        print("SURF", true_surf, comp_surf)

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

    print("---------------------------------")
    print("VOLUME")
    print("---------------------------------")
    print(vol_eoc_rec)
    assert vol_eoc_rec.order_estimate() >= mesh_order

    print("---------------------------------")
    print("SURFACE")
    print("---------------------------------")
    print(surf_eoc_rec)
    assert surf_eoc_rec.order_estimate() >= mesh_order

# }}}


# {{{ rect/box mesh generation

def test_rect_mesh(do_plot=False):
    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh()

    if do_plot:
        from meshmode.mesh.visualization import draw_2d_mesh
        draw_2d_mesh(mesh, fill=None, draw_nodal_adjacency=True)
        import matplotlib.pyplot as pt
        pt.show()


def test_box_mesh(ctx_getter, visualize=False):
    from meshmode.mesh.generation import generate_box_mesh
    mesh = generate_box_mesh(3*(np.linspace(0, 1, 5),))

    if visualize:
        from meshmode.discretization import Discretization
        from meshmode.discretization.poly_element import \
                PolynomialWarpAndBlendGroupFactory
        cl_ctx = ctx_getter()
        queue = cl.CommandQueue(cl_ctx)

        discr = Discretization(cl_ctx, mesh,
                PolynomialWarpAndBlendGroupFactory(1))

        from meshmode.discretization.visualization import make_visualizer
        vis = make_visualizer(queue, discr, 1)
        vis.write_vtk_file("box.vtu", [])

# }}}


# {{{ as_python stringification

def test_as_python():
    from meshmode.mesh.generation import generate_box_mesh
    mesh = generate_box_mesh(3*(np.linspace(0, 1, 5),))

    # These implicitly compute these adjacency structures.
    mesh.nodal_adjacency
    mesh.facial_adjacency_groups

    from meshmode.mesh import as_python
    code = as_python(mesh)

    print(code)
    exec_dict = {}
    exec(compile(code, "gen_code.py", "exec"), exec_dict)

    mesh_2 = exec_dict["make_mesh"]()

    assert mesh == mesh_2

# }}}


# {{{ lookup tree for element finding

def test_lookup_tree(do_plot=False):
    from meshmode.mesh.generation import make_curve_mesh, cloverleaf
    mesh = make_curve_mesh(cloverleaf, np.linspace(0, 1, 1000), order=3)

    from meshmode.mesh.tools import make_element_lookup_tree
    tree = make_element_lookup_tree(mesh)

    from meshmode.mesh.processing import find_bounding_box
    bbox_min, bbox_max = find_bounding_box(mesh)

    extent = bbox_max-bbox_min

    for i in range(20):
        pt = bbox_min + np.random.rand(2) * extent
        print(pt)
        for igrp, iel in tree.generate_matches(pt):
            print(igrp, iel)

    if do_plot:
        with open("tree.dat", "w") as outf:
            tree.visualize(outf)

# }}}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
