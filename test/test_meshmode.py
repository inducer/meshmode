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

from functools import partial
from six.moves import range
import numpy as np
import numpy.linalg as la

from pytools.obj_array import make_obj_array

from meshmode.array_context import (  # noqa
        pytest_generate_tests_for_pyopencl_array_context
        as pytest_generate_tests)

from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
from meshmode.discretization.poly_element import (
        InterpolatoryQuadratureSimplexGroupFactory,
        PolynomialWarpAndBlendGroupFactory,
        PolynomialRecursiveNodesGroupFactory,
        PolynomialEquidistantSimplexGroupFactory,
        )
from meshmode.mesh import Mesh, BTAG_ALL
from meshmode.dof_array import thaw, flat_norm, flatten, unflatten
from meshmode.discretization.connection import \
        FACE_RESTR_ALL, FACE_RESTR_INTERIOR
import meshmode.mesh.generation as mgen

import pytest

import logging
logger = logging.getLogger(__name__)


# {{{ circle mesh

def test_circle_mesh(visualize=False):
    from meshmode.mesh.io import generate_gmsh, FileSource
    logger.info("BEGIN GEN")
    mesh = generate_gmsh(
            FileSource("circle.step"), 2, order=2,
            force_ambient_dim=2,
            other_options=[
                "-string", "Mesh.CharacteristicLengthMax = 0.05;"],
            target_unit="MM",
            )
    logger.info("END GEN")
    logger.info("nelements: %d", mesh.nelements)

    from meshmode.mesh.processing import affine_map
    mesh = affine_map(mesh, A=3*np.eye(2))

    if visualize:
        from meshmode.mesh.visualization import draw_2d_mesh
        draw_2d_mesh(mesh,
                fill=None,
                draw_vertex_numbers=False,
                draw_nodal_adjacency=True,
                set_bounding_box=True)
        import matplotlib.pyplot as pt
        pt.axis("equal")
        pt.savefig("circle_mesh", dpi=300)

# }}}


# {{{ test visualizer

@pytest.mark.parametrize("dim", [1, 2, 3])
def test_parallel_vtk_file(actx_factory, dim):
    r"""
    Simple test just generates a sample parallel PVTU file
    and checks it against the expected result.  The expected
    result is just a file in the tests directory.
    """
    logging.basicConfig(level=logging.INFO)

    actx = actx_factory()

    nelements = 64
    target_order = 4

    if dim == 1:
        mesh = mgen.make_curve_mesh(
                mgen.NArmedStarfish(5, 0.25),
                np.linspace(0.0, 1.0, nelements + 1),
                target_order)
    elif dim == 2:
        mesh = mgen.generate_torus(5.0, 1.0, order=target_order)
    elif dim == 3:
        mesh = mgen.generate_warped_rect_mesh(dim, target_order, 5)
    else:
        raise ValueError("unknown dimensionality")

    from meshmode.discretization import Discretization
    discr = Discretization(actx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

    from meshmode.discretization.visualization import make_visualizer
    vis = make_visualizer(actx, discr, target_order)

    class FakeComm:
        def Get_rank(self):  # noqa: N802
            return 0

        def Get_size(self):  # noqa: N802
            return 2

    file_name_pattern = f"visualizer_vtk_linear_{dim}_{{rank}}.vtu"
    pvtu_filename = file_name_pattern.format(rank=0).replace("vtu", "pvtu")

    vis.write_parallel_vtk_file(
            FakeComm(),
            file_name_pattern,
            [
                ("scalar", discr.zeros(actx)),
                ("vector", make_obj_array([discr.zeros(actx) for i in range(dim)]))
                ],
            overwrite=True)

    import os
    assert(os.path.exists(pvtu_filename))

    import filecmp
    assert(filecmp.cmp("ref-"+pvtu_filename, pvtu_filename))


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_visualizers(actx_factory, dim):
    logging.basicConfig(level=logging.INFO)

    actx = actx_factory()

    nelements = 64
    target_order = 4

    if dim == 1:
        mesh = mgen.make_curve_mesh(
                mgen.NArmedStarfish(5, 0.25),
                np.linspace(0.0, 1.0, nelements + 1),
                target_order)
    elif dim == 2:
        mesh = mgen.generate_torus(5.0, 1.0, order=target_order)
    elif dim == 3:
        mesh = mgen.generate_warped_rect_mesh(dim, target_order, 5)
    else:
        raise ValueError("unknown dimensionality")

    from meshmode.discretization import Discretization
    discr = Discretization(actx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

    from meshmode.discretization.visualization import make_visualizer
    vis = make_visualizer(actx, discr, target_order)

    vis.write_vtk_file(f"visualizer_vtk_linear_{dim}.vtu",
            [], overwrite=True)

    with pytest.raises(RuntimeError):
        vis.write_vtk_file(f"visualizer_vtk_lagrange_{dim}.vtu",
                [], overwrite=True, use_high_order=True)

    if mesh.dim <= 2:
        field = thaw(actx, discr.nodes()[0])

    if mesh.dim == 2:
        try:
            vis.show_scalar_in_matplotlib_3d(field, do_show=False)
        except ImportError:
            logger.info("matplotlib not available")

    if mesh.dim <= 2:
        try:
            vis.show_scalar_in_mayavi(field, do_show=False)
        except ImportError:
            logger.info("mayavi not avaiable")

    vis = make_visualizer(actx, discr, target_order,
            force_equidistant=True)
    vis.write_vtk_file(f"visualizer_vtk_lagrange_{dim}.vtu",
            [], overwrite=True, use_high_order=True)

# }}}


# {{{ test boundary tags

def test_boundary_tags():
    from meshmode.mesh.io import read_gmsh
    # ensure tags are read in
    mesh = read_gmsh('annulus.msh')
    if not {'outer_bdy', 'inner_bdy'} <= set(mesh.boundary_tags):
        print("Mesh boundary tags:", mesh.boundary_tags)
        raise ValueError('Tags not saved by mesh')

    # correct answers
    num_on_outer_bdy = 26
    num_on_inner_bdy = 13

    # check how many elements are marked on each boundary
    num_marked_outer_bdy = 0
    num_marked_inner_bdy = 0
    outer_btag_bit = mesh.boundary_tag_bit('outer_bdy')
    inner_btag_bit = mesh.boundary_tag_bit('inner_bdy')
    for igrp in range(len(mesh.groups)):
        bdry_fagrp = mesh.facial_adjacency_groups[igrp].get(None, None)

        if bdry_fagrp is None:
            continue

        for i, nbrs in enumerate(bdry_fagrp.neighbors):
            if (-nbrs) & outer_btag_bit:
                num_marked_outer_bdy += 1
            if (-nbrs) & inner_btag_bit:
                num_marked_inner_bdy += 1

    # raise errors if wrong number of elements marked
    if num_marked_inner_bdy != num_on_inner_bdy:
        raise ValueError("%i marked on inner boundary, should be %i" %
                         (num_marked_inner_bdy, num_on_inner_bdy))

    if num_marked_outer_bdy != num_on_outer_bdy:
        raise ValueError("%i marked on outer boundary, should be %i" %
                         (num_marked_outer_bdy, num_on_outer_bdy))

    # ensure boundary is covered
    from meshmode.mesh import check_bc_coverage
    check_bc_coverage(mesh, ['inner_bdy', 'outer_bdy'])

# }}}


# {{{ test custom boundary tags on box mesh

@pytest.mark.parametrize(("dim", "nelem"), [
    (1, 20),
    (2, 20),
    (3, 10),
    ])
@pytest.mark.parametrize("group_factory", [
    SimplexElementGroup,

    # FIXME: Not implemented: TPE.face_vertex_indices
    # TensorProductElementGroup
    ])
def test_box_boundary_tags(dim, nelem, group_factory):
    from meshmode.mesh.generation import generate_regular_rect_mesh
    from meshmode.mesh import is_boundary_tag_empty
    from meshmode.mesh import check_bc_coverage
    if dim == 1:
        a = (0,)
        b = (1,)
        n = (nelem,)
        btag_to_face = {"btag_test_1": ["+x"],
                        "btag_test_2": ["-x"]}
    elif dim == 2:
        a = (0, -1)
        b = (1, 1)
        n = (nelem, nelem)
        btag_to_face = {"btag_test_1": ["+x", "-y"],
                        "btag_test_2": ["+y", "-x"]}
    elif dim == 3:
        a = (0, -1, -1)
        b = (1, 1, 1)
        n = (nelem, nelem, nelem)
        btag_to_face = {"btag_test_1": ["+x", "-y", "-z"],
                        "btag_test_2": ["+y", "-x", "+z"]}
    mesh = generate_regular_rect_mesh(a=a, b=b,
                                      n=n, order=3,
                                      boundary_tag_to_face=btag_to_face,
                                      group_factory=group_factory)
    # correct answer
    if dim == 1:
        num_on_bdy = 1
    else:
        num_on_bdy = dim*(dim-1)*(nelem-1)**(dim-1)

    assert not is_boundary_tag_empty(mesh, "btag_test_1")
    assert not is_boundary_tag_empty(mesh, "btag_test_2")
    check_bc_coverage(mesh, ['btag_test_1', 'btag_test_2'])

    # check how many elements are marked on each boundary
    num_marked_bdy_1 = 0
    num_marked_bdy_2 = 0
    btag_1_bit = mesh.boundary_tag_bit("btag_test_1")
    btag_2_bit = mesh.boundary_tag_bit("btag_test_2")
    for igrp in range(len(mesh.groups)):
        bdry_fagrp = mesh.facial_adjacency_groups[igrp].get(None, None)

        if bdry_fagrp is None:
            continue

        for i, nbrs in enumerate(bdry_fagrp.neighbors):
            if (-nbrs) & btag_1_bit:
                num_marked_bdy_1 += 1
            if (-nbrs) & btag_2_bit:
                num_marked_bdy_2 += 1

    # raise errors if wrong number of elements marked
    if num_marked_bdy_1 != num_on_bdy:
        raise ValueError("%i marked on custom boundary 1, should be %i" %
                         (num_marked_bdy_1, num_on_bdy))
    if num_marked_bdy_2 != num_on_bdy:
        raise ValueError("%i marked on custom boundary 2, should be %i" %
                         (num_marked_bdy_2, num_on_bdy))


# }}}


# {{{ convergence of boundary interpolation

@pytest.mark.parametrize("group_factory", [
    InterpolatoryQuadratureSimplexGroupFactory,
    PolynomialWarpAndBlendGroupFactory,
    partial(PolynomialRecursiveNodesGroupFactory, family="lgl"),
    #partial(PolynomialRecursiveNodesGroupFactory, family="gc"),
    ])
@pytest.mark.parametrize("boundary_tag", [
    BTAG_ALL,
    FACE_RESTR_ALL,
    FACE_RESTR_INTERIOR,
    ])
@pytest.mark.parametrize(("mesh_name", "dim", "mesh_pars"), [
    ("blob", 2, ["8e-2", "6e-2", "4e-2"]),
    ("warp", 2, [10, 20, 30]),
    ("warp", 3, [10, 20, 30]),
    ])
@pytest.mark.parametrize("per_face_groups", [False, True])
def test_boundary_interpolation(actx_factory, group_factory, boundary_tag,
        mesh_name, dim, mesh_pars, per_face_groups):
    actx = actx_factory()

    from meshmode.discretization import Discretization
    from meshmode.discretization.connection import (
            make_face_restriction, check_connection)

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    order = 4

    def f(x):
        return 0.1*actx.np.sin(30*x)

    for mesh_par in mesh_pars:
        # {{{ get mesh

        if mesh_name == "blob":
            assert dim == 2

            h = float(mesh_par)

            #from meshmode.mesh.io import generate_gmsh, FileSource
            # print("BEGIN GEN")
            # mesh = generate_gmsh(
            #         FileSource("blob-2d.step"), 2, order=order,
            #         force_ambient_dim=2,
            #         other_options=[
            #             "-string", "Mesh.CharacteristicLengthMax = %s;" % h]
            #         )
            # print("END GEN")
            from meshmode.mesh.io import read_gmsh
            mesh = read_gmsh(
                    "blob2d-order%d-h%s.msh" % (order, mesh_par),
                    force_ambient_dim=2)
        elif mesh_name == "warp":
            from meshmode.mesh.generation import generate_warped_rect_mesh
            mesh = generate_warped_rect_mesh(dim, order=4, n=mesh_par)

            h = 1/mesh_par
        else:
            raise ValueError("mesh_name not recognized")

        # }}}

        vol_discr = Discretization(actx, mesh, group_factory(order))
        print("h=%s -> %d elements" % (
                h, sum(mgrp.nelements for mgrp in mesh.groups)))

        x = thaw(actx, vol_discr.nodes()[0])
        vol_f = f(x)

        bdry_connection = make_face_restriction(
                actx, vol_discr, group_factory(order),
                boundary_tag, per_face_groups=per_face_groups)
        check_connection(actx, bdry_connection)
        bdry_discr = bdry_connection.to_discr

        bdry_x = thaw(actx, bdry_discr.nodes()[0])
        bdry_f = f(bdry_x)
        bdry_f_2 = bdry_connection(vol_f)

        if mesh_name == "blob" and dim == 2 and mesh.nelements < 500:
            mat = actx.to_numpy(bdry_connection.full_resample_matrix(actx))
            bdry_f_2_by_mat = mat.dot(actx.to_numpy(flatten(vol_f)))

            mat_error = la.norm(actx.to_numpy(flatten(bdry_f_2)) - bdry_f_2_by_mat)
            assert mat_error < 1e-14, mat_error

        err = flat_norm(bdry_f-bdry_f_2, np.inf)
        eoc_rec.add_data_point(h, err)

    order_slack = 0.75 if mesh_name == "blob" else 0.5
    print(eoc_rec)
    assert (
            eoc_rec.order_estimate() >= order-order_slack
            or eoc_rec.max_error() < 3e-14)

# }}}


# {{{ boundary-to-all-faces connecttion

@pytest.mark.parametrize(("mesh_name", "dim", "mesh_pars"), [
    ("blob", 2, [1e-1, 8e-2, 5e-2]),
    ("warp", 2, [10, 20, 30]),
    ("warp", 3, [10, 20, 30]),
    ])
@pytest.mark.parametrize("per_face_groups", [False, True])
def test_all_faces_interpolation(actx_factory, mesh_name, dim, mesh_pars,
        per_face_groups):
    actx = actx_factory()

    from meshmode.discretization import Discretization
    from meshmode.discretization.connection import (
            make_face_restriction, make_face_to_all_faces_embedding,
            check_connection)

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    order = 4

    def f(x):
        return 0.1*actx.np.sin(30*x)

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
                        "-string", "Mesh.CharacteristicLengthMax = %s;" % h],
                    target_unit="MM",
                    )
            print("END GEN")
        elif mesh_name == "warp":
            from meshmode.mesh.generation import generate_warped_rect_mesh
            mesh = generate_warped_rect_mesh(dim, order=4, n=mesh_par)

            h = 1/mesh_par
        else:
            raise ValueError("mesh_name not recognized")

        # }}}

        vol_discr = Discretization(actx, mesh,
                PolynomialWarpAndBlendGroupFactory(order))
        print("h=%s -> %d elements" % (
                h, sum(mgrp.nelements for mgrp in mesh.groups)))

        all_face_bdry_connection = make_face_restriction(
                actx, vol_discr, PolynomialWarpAndBlendGroupFactory(order),
                FACE_RESTR_ALL, per_face_groups=per_face_groups)
        all_face_bdry_discr = all_face_bdry_connection.to_discr

        for ito_grp, ceg in enumerate(all_face_bdry_connection.groups):
            for ibatch, batch in enumerate(ceg.batches):
                assert np.array_equal(
                        actx.to_numpy(actx.thaw(batch.from_element_indices)),
                        np.arange(vol_discr.mesh.nelements))

                if per_face_groups:
                    assert ito_grp == batch.to_element_face
                else:
                    assert ibatch == batch.to_element_face

        all_face_x = thaw(actx, all_face_bdry_discr.nodes()[0])
        all_face_f = f(all_face_x)

        all_face_f_2 = all_face_bdry_discr.zeros(actx)

        for boundary_tag in [
                BTAG_ALL,
                FACE_RESTR_INTERIOR,
                ]:
            bdry_connection = make_face_restriction(
                    actx, vol_discr, PolynomialWarpAndBlendGroupFactory(order),
                    boundary_tag, per_face_groups=per_face_groups)
            bdry_discr = bdry_connection.to_discr

            bdry_x = thaw(actx, bdry_discr.nodes()[0])
            bdry_f = f(bdry_x)

            all_face_embedding = make_face_to_all_faces_embedding(
                    actx, bdry_connection, all_face_bdry_discr)

            check_connection(actx, all_face_embedding)

            all_face_f_2 = all_face_f_2 + all_face_embedding(bdry_f)

        err = flat_norm(all_face_f-all_face_f_2, np.inf)
        eoc_rec.add_data_point(h, err)

    print(eoc_rec)
    assert (
            eoc_rec.order_estimate() >= order-0.5
            or eoc_rec.max_error() < 1e-14)

# }}}


# {{{ convergence of opposite-face interpolation

@pytest.mark.parametrize("group_factory", [
    InterpolatoryQuadratureSimplexGroupFactory,
    PolynomialWarpAndBlendGroupFactory
    ])
@pytest.mark.parametrize(("mesh_name", "dim", "mesh_pars"), [
    ("segment", 1, [8, 16, 32]),
    ("blob", 2, [1e-1, 8e-2, 5e-2]),
    ("warp", 2, [3, 5, 7]),
    ("warp", 3, [3, 5]),
    ])
def test_opposite_face_interpolation(actx_factory, group_factory,
        mesh_name, dim, mesh_pars):
    logging.basicConfig(level=logging.INFO)
    actx = actx_factory()

    from meshmode.discretization import Discretization
    from meshmode.discretization.connection import (
            make_face_restriction, make_opposite_face_connection,
            check_connection)

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    order = 5

    def f(x):
        return 0.1*actx.np.sin(30*x)

    for mesh_par in mesh_pars:
        # {{{ get mesh

        if mesh_name == "segment":
            assert dim == 1

            from meshmode.mesh.generation import generate_box_mesh
            mesh = generate_box_mesh(
                    [np.linspace(-0.5, 0.5, mesh_par)],
                    order=order)
            h = 1.0 / mesh_par
        elif mesh_name == "blob":
            assert dim == 2

            h = mesh_par

            from meshmode.mesh.io import generate_gmsh, FileSource
            print("BEGIN GEN")
            mesh = generate_gmsh(
                    FileSource("blob-2d.step"), 2, order=order,
                    force_ambient_dim=2,
                    other_options=[
                        "-string", "Mesh.CharacteristicLengthMax = %s;" % h],
                    target_unit="MM",
                    )
            print("END GEN")
        elif mesh_name == "warp":
            from meshmode.mesh.generation import generate_warped_rect_mesh
            mesh = generate_warped_rect_mesh(dim, order=4, n=mesh_par)

            h = 1/mesh_par
        else:
            raise ValueError("mesh_name not recognized")

        # }}}

        vol_discr = Discretization(actx, mesh,
                group_factory(order))
        print("h=%s -> %d elements" % (
                h, sum(mgrp.nelements for mgrp in mesh.groups)))

        bdry_connection = make_face_restriction(
                actx, vol_discr, group_factory(order),
                FACE_RESTR_INTERIOR)
        bdry_discr = bdry_connection.to_discr

        opp_face = make_opposite_face_connection(actx, bdry_connection)
        check_connection(actx, opp_face)

        bdry_x = thaw(actx, bdry_discr.nodes()[0])
        bdry_f = f(bdry_x)
        bdry_f_2 = opp_face(bdry_f)

        err = flat_norm(bdry_f-bdry_f_2, np.inf)
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
            other_options=["-string", "Mesh.CharacteristicLengthMax = 0.02;"],
            target_unit="MM",
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


# {{{ element orientation: canned 3D meshes

# python test_meshmode.py 'test_sanity_balls(cl._csc, "disk-radius-1.step", 2, 2, visualize=True)'  # noqa
@pytest.mark.parametrize(("what", "mesh_gen_func"), [
    ("ball", lambda: mgen.generate_icosahedron(1, 1)),
    ("torus", lambda: mgen.generate_torus(5, 1)),
    ])
def test_orientation_3d(actx_factory, what, mesh_gen_func, visualize=False):
    pytest.importorskip("pytential")

    logging.basicConfig(level=logging.INFO)
    actx = actx_factory()

    mesh = mesh_gen_func()

    logger.info("%d elements" % mesh.nelements)

    from meshmode.discretization import Discretization
    discr = Discretization(actx, mesh,
            PolynomialWarpAndBlendGroupFactory(3))

    from pytential import bind, sym

    # {{{ check normals point outward

    if what == "torus":
        nodes = sym.nodes(mesh.ambient_dim).as_vector()
        angle = sym.atan2(nodes[1], nodes[0])
        center_nodes = sym.make_obj_array([
                5*sym.cos(angle),
                5*sym.sin(angle),
                0*angle])
        normal_outward_expr = (
                sym.normal(mesh.ambient_dim) | (nodes-center_nodes))

    else:
        normal_outward_expr = (
                sym.normal(mesh.ambient_dim) | sym.nodes(mesh.ambient_dim))

    normal_outward_check = actx.to_numpy(
            flatten(bind(discr, normal_outward_expr)(actx).as_scalar())) > 0

    assert normal_outward_check.all(), normal_outward_check

    # }}}

    normals = bind(discr, sym.normal(mesh.ambient_dim).xproject(1))(actx)

    if visualize:
        from meshmode.discretization.visualization import make_visualizer
        vis = make_visualizer(actx, discr, 3)

        vis.write_vtk_file("orientation_3d_%s_normals.vtu" % what, [
            ("normals", normals),
            ])

# }}}


# {{{ merge and map

def test_merge_and_map(actx_factory, visualize=False):
    from meshmode.mesh.io import generate_gmsh, FileSource
    from meshmode.mesh.generation import generate_box_mesh
    from meshmode.discretization.poly_element import (
            PolynomialWarpAndBlendGroupFactory,
            LegendreGaussLobattoTensorProductGroupFactory)

    mesh_order = 3

    if 1:
        mesh = generate_gmsh(
                FileSource("blob-2d.step"), 2, order=mesh_order,
                force_ambient_dim=2,
                other_options=["-string", "Mesh.CharacteristicLengthMax = 0.02;"],
                target_unit="MM",
                )

        discr_grp_factory = PolynomialWarpAndBlendGroupFactory(3)
    else:
        mesh = generate_box_mesh(
                (
                    np.linspace(0, 1, 4),
                    np.linspace(0, 1, 4),
                    np.linspace(0, 1, 4),
                    ),
                10, group_factory=TensorProductElementGroup)

        discr_grp_factory = LegendreGaussLobattoTensorProductGroupFactory(3)

    from meshmode.mesh.processing import merge_disjoint_meshes, affine_map
    mesh2 = affine_map(mesh,
            A=np.eye(mesh.ambient_dim),
            b=np.array([2, 0, 0])[:mesh.ambient_dim])

    mesh3 = merge_disjoint_meshes((mesh2, mesh))
    mesh3.facial_adjacency_groups

    mesh3.copy()

    if visualize:
        from meshmode.discretization import Discretization
        actx = actx_factory()
        discr = Discretization(actx, mesh3, discr_grp_factory)

        from meshmode.discretization.visualization import make_visualizer
        vis = make_visualizer(actx, discr, 3, element_shrink_factor=0.8)
        vis.write_vtk_file("merge_and_map.vtu", [])

# }}}


# {{{ sanity checks: single element

@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("order", [1, 3])
def test_sanity_single_element(actx_factory, dim, order, visualize=False):
    pytest.importorskip("pytential")
    actx = actx_factory()

    from modepy.tools import unit_vertices
    vertices = unit_vertices(dim).T.copy()

    center = np.empty(dim, np.float64)
    center.fill(-0.5)

    import modepy as mp
    from meshmode.mesh import SimplexElementGroup, Mesh, BTAG_ALL
    mg = SimplexElementGroup(
            order=order,
            vertex_indices=np.arange(dim+1, dtype=np.int32).reshape(1, -1),
            nodes=mp.warp_and_blend_nodes(dim, order).reshape(dim, 1, -1),
            dim=dim)

    mesh = Mesh(vertices, [mg], is_conforming=True)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            PolynomialWarpAndBlendGroupFactory
    vol_discr = Discretization(actx, mesh,
            PolynomialWarpAndBlendGroupFactory(order+3))

    # {{{ volume calculation check

    vol_x = thaw(actx, vol_discr.nodes())

    vol_one = vol_x[0] * 0 + 1
    from pytential import norm, integral  # noqa

    from pytools import factorial
    true_vol = 1/factorial(dim) * 2**dim

    comp_vol = integral(vol_discr, vol_one)
    rel_vol_err = abs(true_vol - comp_vol) / true_vol

    assert rel_vol_err < 1e-12

    # }}}

    # {{{ boundary discretization

    from meshmode.discretization.connection import make_face_restriction
    bdry_connection = make_face_restriction(
            actx, vol_discr, PolynomialWarpAndBlendGroupFactory(order + 3),
            BTAG_ALL)
    bdry_discr = bdry_connection.to_discr

    # }}}

    from pytential import bind, sym
    bdry_normals = bind(bdry_discr, sym.normal(dim))(actx).as_vector(dtype=object)

    if visualize:
        from meshmode.discretization.visualization import make_visualizer
        bdry_vis = make_visualizer(actx, bdry_discr, 4)

        bdry_vis.write_vtk_file("sanity_single_element_boundary.vtu", [
            ("normals", bdry_normals)
            ])

    normal_outward_check = bind(bdry_discr,
            sym.normal(dim)
            | (sym.nodes(dim) + 0.5*sym.ones_vec(dim)),
            )(actx).as_scalar()

    normal_outward_check = actx.to_numpy(flatten(normal_outward_check) > 0)
    assert normal_outward_check.all(), normal_outward_check

# }}}


# {{{ sanity check: volume interpolation on scipy/qhull delaunay meshes in nD

@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("order", [3])
def test_sanity_qhull_nd(actx_factory, dim, order):
    pytest.importorskip("scipy")

    logging.basicConfig(level=logging.INFO)
    actx = actx_factory()

    from scipy.spatial import Delaunay
    verts = np.random.rand(1000, dim)
    dtri = Delaunay(verts)

    from meshmode.mesh.io import from_vertices_and_simplices
    mesh = from_vertices_and_simplices(dtri.points.T, dtri.simplices,
            fix_orientation=True)

    from meshmode.discretization import Discretization
    low_discr = Discretization(actx, mesh,
            PolynomialEquidistantSimplexGroupFactory(order))
    high_discr = Discretization(actx, mesh,
            PolynomialEquidistantSimplexGroupFactory(order+1))

    from meshmode.discretization.connection import make_same_mesh_connection
    cnx = make_same_mesh_connection(actx, high_discr, low_discr)

    def f(x):
        return 0.1*actx.np.sin(x)

    x_low = thaw(actx, low_discr.nodes()[0])
    f_low = f(x_low)

    x_high = thaw(actx, high_discr.nodes()[0])
    f_high_ref = f(x_high)

    f_high_num = cnx(f_low)

    err = (
            flat_norm(f_high_ref-f_high_num, np.inf)
            / flat_norm(f_high_ref, np.inf))

    print(err)
    assert err < 1e-2

# }}}


# {{{ sanity checks: ball meshes

# python test_meshmode.py 'test_sanity_balls(cl._csc, "disk-radius-1.step", 2, 2, visualize=True)'  # noqa
@pytest.mark.parametrize(("src_file", "dim"), [
    ("disk-radius-1.step", 2),
    ("ball-radius-1.step", 3),
    ])
@pytest.mark.parametrize("mesh_order", [1, 2])
def test_sanity_balls(actx_factory, src_file, dim, mesh_order, visualize=False):
    pytest.importorskip("pytential")

    logging.basicConfig(level=logging.INFO)
    actx = actx_factory()

    from pytools.convergence import EOCRecorder
    vol_eoc_rec = EOCRecorder()
    surf_eoc_rec = EOCRecorder()

    # overkill
    quad_order = mesh_order

    from pytential import bind, sym

    for h in [0.2, 0.1, 0.05]:
        from meshmode.mesh.io import generate_gmsh, FileSource
        mesh = generate_gmsh(
                FileSource(src_file), dim, order=mesh_order,
                other_options=["-string", "Mesh.CharacteristicLengthMax = %g;" % h],
                force_ambient_dim=dim,
                target_unit="MM")

        logger.info("%d elements" % mesh.nelements)

        # {{{ discretizations and connections

        from meshmode.discretization import Discretization
        vol_discr = Discretization(actx, mesh,
                InterpolatoryQuadratureSimplexGroupFactory(quad_order))

        from meshmode.discretization.connection import make_face_restriction
        bdry_connection = make_face_restriction(
                actx,
                vol_discr,
                InterpolatoryQuadratureSimplexGroupFactory(quad_order),
                BTAG_ALL)
        bdry_discr = bdry_connection.to_discr

        # }}}

        from math import gamma
        true_surf = 2*np.pi**(dim/2)/gamma(dim/2)
        true_vol = true_surf/dim

        vol_x = thaw(actx, vol_discr.nodes())

        vol_one = vol_x[0]*0 + 1
        from pytential import norm, integral  # noqa

        comp_vol = integral(vol_discr, vol_one)
        rel_vol_err = abs(true_vol - comp_vol) / true_vol
        vol_eoc_rec.add_data_point(h, rel_vol_err)
        print("VOL", true_vol, comp_vol)

        bdry_x = thaw(actx, bdry_discr.nodes())

        bdry_one_exact = bdry_x[0] * 0 + 1

        bdry_one = bdry_connection(vol_one)
        intp_err = norm(bdry_discr, bdry_one-bdry_one_exact)
        assert intp_err < 1e-14

        comp_surf = integral(bdry_discr, bdry_one)
        rel_surf_err = abs(true_surf - comp_surf) / true_surf
        surf_eoc_rec.add_data_point(h, rel_surf_err)
        print("SURF", true_surf, comp_surf)

        if visualize:
            from meshmode.discretization.visualization import make_visualizer
            vol_vis = make_visualizer(actx, vol_discr, 7)
            bdry_vis = make_visualizer(actx, bdry_discr, 7)

            name = src_file.split("-")[0]
            vol_vis.write_vtk_file("sanity_balls_volume_%s_%g.vtu" % (name, h), [
                ("f", vol_one),
                ("area_el", bind(
                    vol_discr,
                    sym.area_element(mesh.ambient_dim, mesh.ambient_dim))
                    (actx)),
                ])

            bdry_vis.write_vtk_file("sanity_balls_boundary_%s_%g.vtu" % (name, h), [
                ("f", bdry_one)
                ])

        # {{{ check normals point outward

        normal_outward_check = bind(bdry_discr,
                sym.normal(mesh.ambient_dim) | sym.nodes(mesh.ambient_dim),
                )(actx).as_scalar()

        normal_outward_check = actx.to_numpy(flatten(normal_outward_check) > 0)
        assert normal_outward_check.all(), normal_outward_check

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

def test_rect_mesh(visualize=False):
    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh()

    if visualize:
        from meshmode.mesh.visualization import draw_2d_mesh
        draw_2d_mesh(mesh, fill=None, draw_nodal_adjacency=True)
        import matplotlib.pyplot as pt
        pt.show()


def test_box_mesh(actx_factory, visualize=False):
    from meshmode.mesh.generation import generate_box_mesh
    mesh = generate_box_mesh(3*(np.linspace(0, 1, 5),))

    if visualize:
        from meshmode.discretization import Discretization
        from meshmode.discretization.poly_element import \
                PolynomialWarpAndBlendGroupFactory

        actx = actx_factory()
        discr = Discretization(actx, mesh,
                PolynomialWarpAndBlendGroupFactory(7))

        from meshmode.discretization.visualization import make_visualizer
        vis = make_visualizer(actx, discr, 7)
        vis.write_vtk_file("box_mesh.vtu", [])

# }}}


def test_mesh_copy():
    from meshmode.mesh.generation import generate_box_mesh
    mesh = generate_box_mesh(3*(np.linspace(0, 1, 5),))
    mesh.copy()


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


# {{{ test lookup tree for element finding

def test_lookup_tree(visualize=False):
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

    if visualize:
        with open("tree.dat", "w") as outf:
            tree.visualize(outf)

# }}}


# {{{ test_nd_quad_submesh

@pytest.mark.parametrize("dims", [2, 3, 4])
def test_nd_quad_submesh(dims):
    from meshmode.mesh.tools import nd_quad_submesh
    from pytools import generate_nonnegative_integer_tuples_below as gnitb

    node_tuples = list(gnitb(3, dims))

    for i, nt in enumerate(node_tuples):
        print(i, nt)

    assert len(node_tuples) == 3**dims

    elements = nd_quad_submesh(node_tuples)

    for e in elements:
        print(e)

    assert len(elements) == 2**dims

# }}}


# {{{ test_quad_mesh_2d

def test_quad_mesh_2d():
    from meshmode.mesh.io import generate_gmsh, ScriptWithFilesSource
    print("BEGIN GEN")
    mesh = generate_gmsh(
            ScriptWithFilesSource(
                """
                Merge "blob-2d.step";
                Mesh.CharacteristicLengthMax = 0.05;
                Recombine Surface "*" = 0.0001;
                Mesh 2;
                Save "output.msh";
                """,
                ["blob-2d.step"]),
            force_ambient_dim=2,
            target_unit="MM",
            )
    print("END GEN")
    print(mesh.nelements)

# }}}


# {{{ test_quad_mesh_3d

# This currently (gmsh 2.13.2) crashes gmsh. A massaged version of this using
# 'cube.step' succeeded in generating 'hybrid-cube.msh' and 'cubed-cube.msh'.
def no_test_quad_mesh_3d():
    from meshmode.mesh.io import generate_gmsh, ScriptWithFilesSource
    print("BEGIN GEN")
    mesh = generate_gmsh(
            ScriptWithFilesSource(
                """
                Merge "ball-radius-1.step";
                // Mesh.CharacteristicLengthMax = 0.1;

                Mesh.RecombineAll=1;
                Mesh.Recombine3DAll=1;
                Mesh.Algorithm = 8;
                Mesh.Algorithm3D = 9;
                // Mesh.Smoothing = 0;

                // Mesh.ElementOrder = 3;

                Mesh 3;
                Save "output.msh";
                """,
                ["ball-radius-1.step"]),
            )
    print("END GEN")
    print(mesh.nelements)

# }}}


# {{{ test_quad_single_element

def test_quad_single_element():
    from meshmode.mesh.generation import make_group_from_vertices

    vertices = np.array([
                [0.91, 1.10],
                [2.64, 1.27],
                [0.97, 2.56],
                [3.00, 3.41],
                ]).T
    mg = make_group_from_vertices(
            vertices,
            np.array([[0, 1, 2, 3]], dtype=np.int32),
            30, group_factory=TensorProductElementGroup)

    Mesh(vertices, [mg], nodal_adjacency=None, facial_adjacency_groups=None)
    if 0:
        import matplotlib.pyplot as plt
        plt.plot(
                mg.nodes[0].reshape(-1),
                mg.nodes[1].reshape(-1), "o")
        plt.show()

# }}}


# {{{ test_quad_multi_element

def test_quad_multi_element():
    from meshmode.mesh.generation import generate_box_mesh
    mesh = generate_box_mesh(
            (
                np.linspace(3, 8, 4),
                np.linspace(3, 8, 4),
                np.linspace(3, 8, 4),
                ),
            10, group_factory=TensorProductElementGroup)

    if 0:
        import matplotlib.pyplot as plt
        mg = mesh.groups[0]
        plt.plot(
                mg.nodes[0].reshape(-1),
                mg.nodes[1].reshape(-1), "o")
        plt.show()

# }}}


# {{{ test_vtk_overwrite

def test_vtk_overwrite(actx_factory):
    pytest.importorskip("pyvisfile")

    def _try_write_vtk(writer, obj):
        import os
        from meshmode import FileExistsError

        filename = "vtk_overwrite_temp.vtu"
        if os.path.exists(filename):
            os.remove(filename)

        writer(filename, [])
        with pytest.raises(FileExistsError):
            writer(filename, [])

        writer(filename, [], overwrite=True)
        if os.path.exists(filename):
            os.remove(filename)

    actx = actx_factory()
    target_order = 7

    from meshmode.mesh.generation import generate_torus
    mesh = generate_torus(10.0, 2.0, order=target_order)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
    discr = Discretization(
            actx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

    from meshmode.discretization.visualization import make_visualizer
    from meshmode.discretization.visualization import \
            write_nodal_adjacency_vtk_file
    from meshmode.mesh.visualization import write_vertex_vtk_file

    vis = make_visualizer(actx, discr, 1)
    _try_write_vtk(vis.write_vtk_file, discr)

    _try_write_vtk(lambda x, y, **kwargs:
            write_vertex_vtk_file(discr.mesh, x, **kwargs), discr.mesh)
    _try_write_vtk(lambda x, y, **kwargs:
            write_nodal_adjacency_vtk_file(x, discr.mesh, **kwargs), discr.mesh)

# }}}


# {{{ test_mesh_to_tikz

def test_mesh_to_tikz():
    from meshmode.mesh.io import generate_gmsh, FileSource

    h = 0.3
    order = 1

    mesh = generate_gmsh(
            FileSource("../test/blob-2d.step"), 2, order=order,
            force_ambient_dim=2,
            other_options=[
                "-string", "Mesh.CharacteristicLengthMax = %s;" % h],
            target_unit="MM",
            )

    from meshmode.mesh.visualization import mesh_to_tikz
    mesh_to_tikz(mesh)

# }}}


def test_affine_map():
    from meshmode.mesh.tools import AffineMap
    for d in range(1, 5):
        for i in range(100):
            a = np.random.randn(d, d)+10*np.eye(d)
            b = np.random.randn(d)

            m = AffineMap(a, b)

            assert la.norm(m.inverted().matrix - la.inv(a)) < 1e-10*la.norm(a)

            x = np.random.randn(d)

            m_inv = m.inverted()

            assert la.norm(x-m_inv(m(x))) < 1e-10


def test_mesh_without_vertices(actx_factory):
    actx = actx_factory()

    # create a mesh
    from meshmode.mesh.generation import generate_icosphere
    mesh = generate_icosphere(r=1.0, order=4)

    # create one without the vertices
    from meshmode.mesh import Mesh
    grp, = mesh.groups
    groups = [grp.copy(nodes=grp.nodes, vertex_indices=None) for grp in mesh.groups]
    mesh = Mesh(None, groups, is_conforming=False)

    # try refining it
    from meshmode.mesh.refinement import refine_uniformly
    mesh = refine_uniformly(mesh, 1)

    # make sure the world doesn't end
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory as GroupFactory
    discr = Discretization(actx, mesh, GroupFactory(4))
    thaw(actx, discr.nodes())

    from meshmode.discretization.visualization import make_visualizer
    make_visualizer(actx, discr, 4)


@pytest.mark.parametrize("curve_name", ["ellipse", "arc"])
def test_open_curved_mesh(curve_name):
    def arc_curve(t, start=0, end=np.pi):
        return np.vstack([
            np.cos((end - start) * t + start),
            np.sin((end - start) * t + start)
            ])

    if curve_name == "ellipse":
        from functools import partial
        from meshmode.mesh.generation import ellipse
        curve_f = partial(ellipse, 2.0)
        closed = True
    elif curve_name == "arc":
        curve_f = arc_curve
        closed = False
    else:
        raise ValueError("unknown curve")

    from meshmode.mesh.generation import make_curve_mesh
    nelements = 32
    order = 4
    make_curve_mesh(curve_f,
            np.linspace(0.0, 1.0, nelements + 1),
            order=order,
            closed=closed)


def _generate_cross_warped_rect_mesh(dim, order, n):
    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(0,)*dim, b=(1,)*dim,
            n=(n,)*dim, order=order)

    def m(x):
        results = np.empty_like(x)
        results[0] = 1 + 1.5 * (x[0] + 0.25) * (x[1] + 0.3)
        results[1] = x[1]
        return results

    from meshmode.mesh.processing import map_mesh
    return map_mesh(mesh, m)


@pytest.mark.parametrize("mesh_name", [
    "box2d", "box3d",
    "warped_box2d", "warped_box3d", "cross_warped_box",
    "circle", "ellipse",
    "sphere", "torus"
    ])
def test_is_affine_group_check(mesh_name):
    from meshmode.mesh.generation import (
            generate_regular_rect_mesh, generate_warped_rect_mesh,
            make_curve_mesh, ellipse,
            generate_icosphere, generate_torus)

    order = 4
    nelements = 16

    if mesh_name.startswith("box"):
        dim = int(mesh_name[-2])
        is_affine = True
        mesh = generate_regular_rect_mesh(
                a=(-0.5,)*dim, b=(0.5,)*dim,
                n=(nelements,)*dim, order=order)
    elif mesh_name.startswith("warped_box"):
        dim = int(mesh_name[-2])
        is_affine = False
        mesh = generate_warped_rect_mesh(dim, order, nelements)
    elif mesh_name == "cross_warped_box":
        dim = 2
        is_affine = False
        mesh = _generate_cross_warped_rect_mesh(dim, order, nelements)
    elif mesh_name == "circle":
        is_affine = False
        mesh = make_curve_mesh(
                lambda t: ellipse(1.0, t),
                np.linspace(0.0, 1.0, nelements + 1), order=order)
    elif mesh_name == "ellipse":
        is_affine = False
        mesh = make_curve_mesh(
                lambda t: ellipse(2.0, t),
                np.linspace(0.0, 1.0, nelements + 1), order=order)
    elif mesh_name == "sphere":
        is_affine = False
        mesh = generate_icosphere(r=1.0, order=order)
    elif mesh_name == "torus":
        is_affine = False
        mesh = generate_torus(10.0, 2.0, order=order)
    else:
        raise ValueError("unknown mesh name: {}".format(mesh_name))

    assert all(grp.is_affine for grp in mesh.groups) == is_affine


@pytest.mark.parametrize("ambient_dim", [1, 2, 3])
def test_mesh_multiple_groups(actx_factory, ambient_dim, visualize=False):
    actx = actx_factory()

    order = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(-0.5,)*ambient_dim, b=(0.5,)*ambient_dim,
            n=(8,)*ambient_dim, order=order)
    assert len(mesh.groups) == 1

    from meshmode.mesh.processing import split_mesh_groups
    element_flags = np.any(
            mesh.vertices[0, mesh.groups[0].vertex_indices] < 0.0,
            axis=1).astype(np.int)
    mesh = split_mesh_groups(mesh, element_flags)

    assert len(mesh.groups) == 2
    assert mesh.facial_adjacency_groups
    assert mesh.nodal_adjacency

    if visualize and ambient_dim == 2:
        from meshmode.mesh.visualization import draw_2d_mesh
        draw_2d_mesh(mesh,
                draw_vertex_numbers=False,
                draw_element_numbers=True,
                draw_face_numbers=False,
                set_bounding_box=True)

        import matplotlib.pyplot as plt
        plt.savefig("test_mesh_multiple_groups_2d_elements.png", dpi=300)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            PolynomialWarpAndBlendGroupFactory as GroupFactory
    discr = Discretization(actx, mesh, GroupFactory(order))

    if visualize:
        group_id = discr.empty(actx, dtype=np.int)
        for igrp, vec in enumerate(group_id):
            vec.fill(igrp)

        from meshmode.discretization.visualization import make_visualizer
        vis = make_visualizer(actx, discr, vis_order=order)
        vis.write_vtk_file("mesh_multiple_groups.vtu", [
            ("group_id", group_id)
            ], overwrite=True)

    # check face restrictions
    from meshmode.discretization.connection import (
            make_face_restriction,
            make_face_to_all_faces_embedding,
            make_opposite_face_connection,
            check_connection)
    for boundary_tag in [BTAG_ALL, FACE_RESTR_INTERIOR, FACE_RESTR_ALL]:
        conn = make_face_restriction(actx, discr, GroupFactory(order),
                boundary_tag=boundary_tag,
                per_face_groups=False)
        check_connection(actx, conn)

        bdry_f = conn.to_discr.zeros(actx) + 1

        if boundary_tag == FACE_RESTR_INTERIOR:
            opposite = make_opposite_face_connection(actx, conn)
            check_connection(actx, opposite)

            op_bdry_f = opposite(bdry_f)
            error = flat_norm(bdry_f - op_bdry_f, np.inf)
            assert error < 1.0e-11, error

        if boundary_tag == FACE_RESTR_ALL:
            embedding = make_face_to_all_faces_embedding(actx, conn, conn.to_discr)
            check_connection(actx, embedding)

            em_bdry_f = embedding(bdry_f)
            error = flat_norm(bdry_f - em_bdry_f)
            assert error < 1.0e-11, error


def test_array_context_np_workalike(actx_factory):
    actx = actx_factory()

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(-0.5,)*2, b=(0.5,)*2, n=(8,)*2, order=3)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            PolynomialWarpAndBlendGroupFactory as GroupFactory
    discr = Discretization(actx, mesh, GroupFactory(3))

    for sym_name, n_args in [
            ("sin", 1),
            ("exp", 1),
            ("arctan2", 2),
            ("minimum", 2),
            ("maximum", 2),
            ("where", 3),
            ]:
        args = [np.random.randn(discr.ndofs) for i in range(n_args)]
        ref_result = getattr(np, sym_name)(*args)

        actx_args = [unflatten(actx, discr, actx.from_numpy(arg)) for arg in args]

        actx_result = actx.to_numpy(
                flatten(getattr(actx.np, sym_name)(*actx_args)))

        assert np.allclose(actx_result, ref_result)


def test_dof_array_comparison(actx_factory):
    actx = actx_factory()

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(-0.5,)*2, b=(0.5,)*2, n=(8,)*2, order=3)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            PolynomialWarpAndBlendGroupFactory as GroupFactory
    discr = Discretization(actx, mesh, GroupFactory(3))

    import operator
    for op in [
            operator.lt,
            operator.le,
            operator.gt,
            operator.ge,
            ]:
        np_arg = np.random.randn(discr.ndofs)
        arg = unflatten(actx, discr, actx.from_numpy(np_arg))
        zeros = discr.zeros(actx)

        comp = op(arg, zeros)
        np_comp = actx.to_numpy(flatten(comp))
        assert np.array_equal(np_comp, op(np_arg, 0))

        comp = op(arg, 0)
        np_comp = actx.to_numpy(flatten(comp))
        assert np.array_equal(np_comp, op(np_arg, 0))

        comp = op(0, arg)
        np_comp = actx.to_numpy(flatten(comp))
        assert np.array_equal(np_comp, op(0, np_arg))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
