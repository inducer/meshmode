__copyright__ = "Copyright (C) 2020 Andreas Kloeckner"

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
import numpy.linalg as la
import pytest
import meshmode         # noqa: F401
from meshmode.array_context import (  # noqa
        pytest_generate_tests_for_pyopencl_array_context
        as pytest_generate_tests)

from meshmode.mesh import Mesh, SimplexElementGroup, TensorProductElementGroup
from meshmode.discretization.poly_element import (
        PolynomialWarpAndBlendGroupFactory,
        LegendreGaussLobattoTensorProductGroupFactory,
        )
import meshmode.mesh.generation as mgen
from meshmode import _acf  # noqa: F401


import logging
logger = logging.getLogger(__name__)


# {{{ test_nonequal_rect_mesh_generation

@pytest.mark.parametrize(("dim", "mesh_type"), [
    (1, None),
    (2, None),
    (2, "X"),
    (3, None),
    ])
def test_nonequal_rect_mesh_generation(actx_factory, dim, mesh_type,
        visualize=False):
    """Test that ``generate_regular_rect_mesh`` works with non-equal arguments
    across axes.
    """
    actx = actx_factory()

    mesh = mgen.generate_regular_rect_mesh(
            a=(0,)*dim, b=(5, 3, 4)[:dim], npoints_per_axis=(10, 6, 7)[:dim],
            order=3, mesh_type=mesh_type)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            PolynomialWarpAndBlendGroupFactory as GroupFactory
    discr = Discretization(actx, mesh, GroupFactory(3))

    if visualize:
        from meshmode.discretization.visualization import make_visualizer
        vis = make_visualizer(actx, discr, 3)

        vis.write_vtk_file("nonuniform.vtu", [], overwrite=True)

# }}}


# {{{ rect/box mesh generation

def test_rect_mesh(visualize=False):
    mesh = mgen.generate_regular_rect_mesh(nelements_per_axis=(4, 4))

    if visualize:
        from meshmode.mesh.visualization import draw_2d_mesh
        draw_2d_mesh(mesh, fill=None, draw_nodal_adjacency=True)
        import matplotlib.pyplot as pt
        pt.show()


def test_box_mesh(actx_factory, visualize=False):
    mesh = mgen.generate_box_mesh(3*(np.linspace(0, 1, 5),))

    if visualize:
        from meshmode.discretization import Discretization

        actx = actx_factory()
        discr = Discretization(actx, mesh,
                PolynomialWarpAndBlendGroupFactory(7))

        from meshmode.discretization.visualization import make_visualizer
        vis = make_visualizer(actx, discr, 7)
        vis.write_vtk_file("box_mesh.vtu", [])

# }}}


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


def test_mesh_copy():
    mesh = mgen.generate_box_mesh(3*(np.linspace(0, 1, 5),))
    mesh.copy()


# {{{ as_python stringification

def test_mesh_as_python():
    mesh = mgen.generate_box_mesh(3*(np.linspace(0, 1, 5),))

    # These implicitly compute these adjacency structures.
    assert mesh.nodal_adjacency
    assert mesh.facial_adjacency_groups

    from meshmode.mesh import as_python
    code = as_python(mesh)

    print(code)
    exec_dict = {}
    exec(compile(code, "gen_code.py", "exec"), exec_dict)

    mesh_2 = exec_dict["make_mesh"]()

    assert mesh == mesh_2

# }}}


# {{{ test_affine_map

def test_affine_map():
    from meshmode.mesh.tools import AffineMap
    for d in range(1, 5):
        for _ in range(100):
            a = np.random.randn(d, d)+10*np.eye(d)
            b = np.random.randn(d)

            m = AffineMap(a, b)
            assert la.norm(m.inverted().matrix - la.inv(a)) < 1e-10*la.norm(a)

            x = np.random.randn(d)
            m_inv = m.inverted()
            assert la.norm(x-m_inv(m(x))) < 1e-10


@pytest.mark.parametrize("ambient_dim", [2, 3])
def test_mesh_rotation(ambient_dim, visualize=False):
    order = 3

    if ambient_dim == 2:
        nelements = 32
        mesh = mgen.make_curve_mesh(
                partial(mgen.ellipse, 2.0),
                np.linspace(0.0, 1.0, nelements + 1),
                order=order)
    elif ambient_dim == 3:
        mesh = mgen.generate_torus(4.0, 2.0, order=order)
    else:
        raise ValueError("unsupported dimension")

    from meshmode.mesh.processing import _get_rotation_matrix_from_angle_and_axis
    mat = _get_rotation_matrix_from_angle_and_axis(
            np.pi/3.0, np.array([1.0, 2.0, 1.4]))

    # check that the matrix is in the rotation group
    assert abs(abs(la.det(mat)) - 1) < 10e-14
    assert la.norm(mat @ mat.T - np.eye(3)) < 1.0e-14

    from meshmode.mesh.processing import rotate_mesh_around_axis
    rotated_mesh = rotate_mesh_around_axis(mesh,
            theta=np.pi/2.0,
            axis=np.array([1, 0, 0]))

    if visualize:
        from meshmode.mesh.visualization import write_vertex_vtk_file
        write_vertex_vtk_file(mesh, "mesh_rotation_original.vtu")
        write_vertex_vtk_file(rotated_mesh, "mesh_rotation_rotated.vtu")

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


# {{{ test_quad_single_element

def test_quad_single_element(visualize=False):
    vertices = np.array([
                [0.91, 1.10],
                [2.64, 1.27],
                [0.97, 2.56],
                [3.00, 3.41],
                ]).T
    mg = mgen.make_group_from_vertices(
            vertices,
            np.array([[0, 1, 2, 3]], dtype=np.int32),
            30, group_cls=TensorProductElementGroup)

    Mesh(vertices, [mg], nodal_adjacency=None, facial_adjacency_groups=None)
    if visualize:
        import matplotlib.pyplot as plt
        plt.plot(
                mg.nodes[0].reshape(-1),
                mg.nodes[1].reshape(-1), "o")
        plt.show()

# }}}


# {{{ merge and map

@pytest.mark.parametrize("group_cls", [
    SimplexElementGroup,
    TensorProductElementGroup
    ])
def test_merge_and_map(actx_factory, group_cls, visualize=False):
    from meshmode.mesh.io import generate_gmsh, FileSource

    order = 3
    mesh_order = 3

    if group_cls is SimplexElementGroup:
        mesh = generate_gmsh(
                FileSource("blob-2d.step"), 2, order=mesh_order,
                force_ambient_dim=2,
                other_options=["-string", "Mesh.CharacteristicLengthMax = 0.02;"],
                target_unit="MM",
                )

        discr_grp_factory = PolynomialWarpAndBlendGroupFactory(order)
    else:
        ambient_dim = 3
        mesh = mgen.generate_regular_rect_mesh(
                a=(0,)*ambient_dim, b=(1,)*ambient_dim,
                nelements_per_axis=(4,)*ambient_dim, order=mesh_order,
                group_cls=group_cls)

        discr_grp_factory = LegendreGaussLobattoTensorProductGroupFactory(order)

    from meshmode.mesh.processing import merge_disjoint_meshes, affine_map
    mesh2 = affine_map(mesh,
            A=np.eye(mesh.ambient_dim),
            b=np.array([2, 0, 0])[:mesh.ambient_dim])

    mesh3 = merge_disjoint_meshes((mesh2, mesh))
    assert mesh3.facial_adjacency_groups

    mesh4 = mesh3.copy()

    if visualize:
        from meshmode.discretization import Discretization
        actx = actx_factory()
        discr = Discretization(actx, mesh4, discr_grp_factory)

        from meshmode.discretization.visualization import make_visualizer
        vis = make_visualizer(actx, discr, 3, element_shrink_factor=0.8)
        vis.write_vtk_file("merge_and_map.vtu", [])

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
    for _ in range(int(0.3*mesh.nelements)):
        flippy[randrange(0, mesh.nelements)] = 1

    mesh = perform_flips(mesh, flippy, skip_tests=True)

    mesh_orient = find_volume_mesh_element_orientations(mesh)

    assert ((mesh_orient < 0) == (flippy > 0)).all()

# }}}


# {{{ test_open_curved_mesh

@pytest.mark.parametrize("curve_name", ["ellipse", "arc"])
def test_open_curved_mesh(curve_name):
    def arc_curve(t, start=0, end=np.pi):
        return np.vstack([
            np.cos((end - start) * t + start),
            np.sin((end - start) * t + start)
            ])

    if curve_name == "ellipse":
        curve_f = partial(mgen.ellipse, 2.0)
        closed = True
    elif curve_name == "arc":
        curve_f = arc_curve
        closed = False
    else:
        raise ValueError("unknown curve")

    nelements = 32
    order = 4
    mgen.make_curve_mesh(curve_f,
            np.linspace(0.0, 1.0, nelements + 1),
            order=order,
            closed=closed)

# }}}


# {{{ test_is_affine_group_check
def _generate_cross_warped_rect_mesh(dim, order, nelements_side):
    mesh = mgen.generate_regular_rect_mesh(
            a=(0,)*dim, b=(1,)*dim,
            nelements_per_axis=(nelements_side,)*dim, order=order)

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
    order = 4
    nelements = 16

    if mesh_name.startswith("box"):
        dim = int(mesh_name[-2])
        is_affine = True
        mesh = mgen.generate_regular_rect_mesh(
                a=(-0.5,)*dim, b=(0.5,)*dim,
                nelements_per_axis=(nelements,)*dim, order=order)
    elif mesh_name.startswith("warped_box"):
        dim = int(mesh_name[-2])
        is_affine = False
        mesh = mgen.generate_warped_rect_mesh(dim, order, nelements_side=nelements)
    elif mesh_name == "cross_warped_box":
        dim = 2
        is_affine = False
        mesh = _generate_cross_warped_rect_mesh(dim, order, nelements)
    elif mesh_name == "circle":
        is_affine = False
        mesh = mgen.make_curve_mesh(
                lambda t: mgen.ellipse(1.0, t),
                np.linspace(0.0, 1.0, nelements + 1), order=order)
    elif mesh_name == "ellipse":
        is_affine = False
        mesh = mgen.make_curve_mesh(
                lambda t: mgen.ellipse(2.0, t),
                np.linspace(0.0, 1.0, nelements + 1), order=order)
    elif mesh_name == "sphere":
        is_affine = False
        mesh = mgen.generate_icosphere(r=1.0, order=order)
    elif mesh_name == "torus":
        is_affine = False
        mesh = mgen.generate_torus(10.0, 2.0, order=order)
    else:
        raise ValueError(f"unknown mesh name: {mesh_name}")

    assert all(grp.is_affine for grp in mesh.groups) == is_affine

# }}}


# {{{ test_quad_multi_element

def test_quad_multi_element(visualize=False):
    mesh = mgen.generate_box_mesh(
            (
                np.linspace(3, 8, 4),
                np.linspace(3, 8, 4),
                np.linspace(3, 8, 4),
                ),
            10, group_cls=TensorProductElementGroup)

    if visualize:
        import matplotlib.pyplot as plt
        mg = mesh.groups[0]
        plt.plot(
                mg.nodes[0].reshape(-1),
                mg.nodes[1].reshape(-1), "o")
        plt.show()

# }}}


# {{{ test lookup tree for element finding

def test_lookup_tree(visualize=False):
    mesh = mgen.make_curve_mesh(mgen.cloverleaf, np.linspace(0, 1, 1000), order=3)

    from meshmode.mesh.tools import make_element_lookup_tree
    tree = make_element_lookup_tree(mesh)

    from meshmode.mesh.processing import find_bounding_box
    bbox_min, bbox_max = find_bounding_box(mesh)

    extent = bbox_max-bbox_min

    for _ in range(20):
        pt = bbox_min + np.random.rand(2) * extent
        print(pt)
        for igrp, iel in tree.generate_matches(pt):
            print(igrp, iel)

    if visualize:
        with open("tree.dat", "w") as outf:
            tree.visualize(outf)

# }}}


# {{{ test boundary tags

def test_boundary_tags():
    from meshmode.mesh.io import read_gmsh
    # ensure tags are read in
    mesh = read_gmsh("annulus.msh")
    if not {"outer_bdy", "inner_bdy"} <= set(mesh.boundary_tags):
        print("Mesh boundary tags:", mesh.boundary_tags)
        raise ValueError("Tags not saved by mesh")

    # correct answers
    num_on_outer_bdy = 26
    num_on_inner_bdy = 13

    # check how many elements are marked on each boundary
    num_marked_outer_bdy = 0
    num_marked_inner_bdy = 0
    outer_btag_bit = mesh.boundary_tag_bit("outer_bdy")
    inner_btag_bit = mesh.boundary_tag_bit("inner_bdy")
    for igrp in range(len(mesh.groups)):
        bdry_fagrp = mesh.facial_adjacency_groups[igrp].get(None, None)

        if bdry_fagrp is None:
            continue

        for nbrs in bdry_fagrp.neighbors:
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
    check_bc_coverage(mesh, ["inner_bdy", "outer_bdy"])

# }}}


# {{{ test custom boundary tags on box mesh

@pytest.mark.parametrize(("dim", "nelem", "mesh_type"), [
    (1, 20, None),
    (2, 20, None),
    (2, 20, "X"),
    (3, 10, None),
    ])
@pytest.mark.parametrize("group_cls", [
    SimplexElementGroup,
    TensorProductElementGroup
    ])
def test_box_boundary_tags(dim, nelem, mesh_type, group_cls, visualize=False):
    if group_cls is TensorProductElementGroup and mesh_type is not None:
        pytest.skip("mesh type not supported on tensor product elements")

    from meshmode.mesh import is_boundary_tag_empty
    from meshmode.mesh import check_bc_coverage

    if dim == 1:
        a = (0,)
        b = (1,)
        nelements_per_axis = (nelem,)
        btag_to_face = {"btag_test_1": ["+x"],
                        "btag_test_2": ["-x"]}
    elif dim == 2:
        a = (0, -1)
        b = (1, 1)
        nelements_per_axis = (nelem,)*2
        btag_to_face = {"btag_test_1": ["+x", "-y"],
                        "btag_test_2": ["+y", "-x"]}
    elif dim == 3:
        a = (0, -1, -1)
        b = (1, 1, 1)
        nelements_per_axis = (nelem,)*3
        btag_to_face = {"btag_test_1": ["+x", "-y", "-z"],
                        "btag_test_2": ["+y", "-x", "+z"]}
    mesh = mgen.generate_regular_rect_mesh(a=a, b=b,
                                      nelements_per_axis=nelements_per_axis, order=3,
                                      boundary_tag_to_face=btag_to_face,
                                      group_cls=group_cls,
                                      mesh_type=mesh_type)

    if visualize and dim == 2:
        from meshmode.mesh.visualization import draw_2d_mesh
        draw_2d_mesh(mesh, draw_element_numbers=False, draw_vertex_numbers=False)
        import matplotlib.pyplot as plt
        plt.show()

    # correct answer
    if dim == 1:
        num_on_bdy = 1
    elif group_cls is TensorProductElementGroup:
        num_on_bdy = dim * nelem**(dim-1)
    elif group_cls is SimplexElementGroup:
        num_on_bdy = dim * (dim-1) * nelem**(dim-1)
    else:
        raise AssertionError()

    assert not is_boundary_tag_empty(mesh, "btag_test_1")
    assert not is_boundary_tag_empty(mesh, "btag_test_2")
    check_bc_coverage(mesh, ["btag_test_1", "btag_test_2"])

    # check how many elements are marked on each boundary
    num_marked_bdy_1 = 0
    num_marked_bdy_2 = 0
    btag_1_bit = mesh.boundary_tag_bit("btag_test_1")
    btag_2_bit = mesh.boundary_tag_bit("btag_test_2")
    for igrp in range(len(mesh.groups)):
        bdry_fagrp = mesh.facial_adjacency_groups[igrp].get(None, None)

        if bdry_fagrp is None:
            continue

        for nbrs in bdry_fagrp.neighbors:
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


# {{{ test_quad_mesh_2d

@pytest.mark.parametrize(("ambient_dim", "filename"),
        [(2, "blob-2d.step"), (3, "ball-radius-1.step")])
def test_quad_mesh_2d(ambient_dim, filename, visualize=False):
    from meshmode.mesh.io import generate_gmsh, ScriptWithFilesSource
    logger.info("BEGIN GEN")

    mesh = generate_gmsh(
            ScriptWithFilesSource(
                f"""
                Merge "{filename}";
                Mesh.CharacteristicLengthMax = 0.05;
                Recombine Surface "*" = 0.0001;
                Mesh 2;
                Save "output.msh";
                """,
                [filename]),
            order=1,
            force_ambient_dim=ambient_dim,
            target_unit="MM",
            )

    logger.info("END GEN")
    logger.info("nelements: %d", mesh.nelements)

    groups = []
    for grp in mesh.groups:
        if not isinstance(grp, TensorProductElementGroup):
            # NOTE: gmsh isn't guaranteed to recombine all elements, so we
            # could still have some simplices sitting around, so skip them
            groups.append(grp.copy())
            continue

        g = mgen.make_group_from_vertices(mesh.vertices,
                grp.vertex_indices, grp.order,
                group_cls=TensorProductElementGroup)
        assert g.nodes.shape == (mesh.ambient_dim, grp.nelements, grp.nunit_nodes)

        groups.append(g)

    mesh_from_vertices = Mesh(mesh.vertices, groups=groups, is_conforming=True)

    if visualize:
        from meshmode.mesh.visualization import write_vertex_vtk_file
        write_vertex_vtk_file(mesh, "quad_mesh_2d_orig.vtu")
        write_vertex_vtk_file(mesh_from_vertices, "quad_mesh_2d_groups.vtu")

# }}}


# {{{ test_quad_mesh_3d

@pytest.mark.parametrize("mesh_name", [
    # this currently (2020-11-05 with gmsh 4.6.0) does not recombine anything
    # or flat out crashes gmsh
    # "ball",

    "cube",
    ])
def test_quad_mesh_3d(mesh_name, order=3, visualize=False):
    if mesh_name == "ball":
        from meshmode.mesh.io import ScriptWithFilesSource
        script = ScriptWithFilesSource(
            """
            Merge "ball-radius-1.step";
            // Mesh.CharacteristicLengthMax = 0.1;

            Mesh.RecombineAll = 1;
            Mesh.Recombine3DAll = 1;
            Mesh.Recombine3DLevel = 2;

            Mesh.Algorithm = 8;
            Mesh.Algorithm3D = 8;

            Mesh 3;
            Save "output.msh";
            """,
            ["ball-radius-1.step"])

        with open("ball-quad.geo", "w") as f:
            f.write(script.source)

    elif mesh_name == "cube":
        from meshmode.mesh.io import ScriptSource
        script = ScriptSource(
            """
            SetFactory("OpenCASCADE");
            Box(1) = {0, 0, 0, 1, 1, 1};

            Transfinite Line "*" = 8;
            Transfinite Surface "*";
            Transfinite Volume "*";

            Mesh.RecombineAll = 1;
            Mesh.Recombine3DAll = 1;
            Mesh.Recombine3DLevel = 2;
            """, "geo")
    else:
        raise ValueError(f"unknown mesh name: '{mesh_name}'")

    np.set_printoptions(linewidth=200)
    from meshmode.mesh.io import generate_gmsh
    logger.info("BEGIN GEN")
    mesh = generate_gmsh(script, 3, order=order, target_unit="MM")
    logger.info("END GEN")

    if visualize:
        from meshmode.mesh.visualization import write_vertex_vtk_file
        write_vertex_vtk_file(mesh, f"quad_mesh_3d_{mesh_name}.vtu", overwrite=True)

# }}}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
