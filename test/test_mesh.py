__copyright__ = """
Copyright (C) 2020 Andreas Kloeckner
Copyright (C) 2021 University of Illinois Board of Trustees
"""

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
from dataclasses import replace
from functools import partial

import numpy as np
import numpy.linalg as la
import pytest

from arraycontext import pytest_generate_tests_for_array_contexts

import meshmode.mesh.generation as mgen
import meshmode.mesh.io as mio
from meshmode import _acf  # noqa: F401
from meshmode.array_context import PytestPyOpenCLArrayContextFactory
from meshmode.discretization.poly_element import (
    LegendreGaussLobattoTensorProductGroupFactory, default_simplex_group_factory)
from meshmode.mesh import (
    BoundaryAdjacencyGroup, InteriorAdjacencyGroup, Mesh, SimplexElementGroup,
    TensorProductElementGroup)
from meshmode.mesh.tools import AffineMap


logger = logging.getLogger(__name__)
pytest_generate_tests = pytest_generate_tests_for_array_contexts(
        [PytestPyOpenCLArrayContextFactory])

thisdir = pathlib.Path(__file__).parent


def _get_rotation(amount, axis, center=None):
    """
    Return a matrix (if *center* is ``None``) or
    :class:`~meshmode.mesh.tools.AffineMap` (if *center* is not ``None``)
    corresponding to a rotation by *amount* (in radians) through a vector *axis*
    centered at *center*. *center* defaults to the origin if not specified.
    """
    from meshmode.mesh.processing import _get_rotation_matrix_from_angle_and_axis
    matrix = _get_rotation_matrix_from_angle_and_axis(amount, axis)
    if center is None:
        return matrix
    else:
        # x0 + matrix @ (x - x0) = matrix @ x + (I - matrix) @ x0
        offset = (np.eye(3) - matrix) @ center
        return AffineMap(matrix, offset)


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
    discr = Discretization(actx, mesh, default_simplex_group_factory(dim, 3))

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
                default_simplex_group_factory(mesh.dim, 7))

        from meshmode.discretization.visualization import make_visualizer
        vis = make_visualizer(actx, discr, 7)
        vis.write_vtk_file("box_mesh.vtu", [])

    # Simplex, 1D
    mesh = mgen.generate_box_mesh(
        (np.linspace(0, 1, 3),), group_cls=SimplexElementGroup)

    assert mesh.ambient_dim == 1
    assert mesh.dim == 1
    assert mesh.nvertices == 3
    assert len(mesh.groups) == 1
    assert mesh.nelements == 2

    # Tensor product, 1D
    mesh = mgen.generate_box_mesh(
        (np.linspace(0, 1, 3),), group_cls=TensorProductElementGroup)

    assert mesh.ambient_dim == 1
    assert mesh.dim == 1
    assert mesh.nvertices == 3
    assert len(mesh.groups) == 1
    assert mesh.nelements == 2

    # Simplex, 2D, non-X
    mesh = mgen.generate_box_mesh(
        2*(np.linspace(0, 1, 3),), group_cls=SimplexElementGroup)

    assert mesh.ambient_dim == 2
    assert mesh.dim == 2
    assert mesh.nvertices == 9
    assert len(mesh.groups) == 1
    assert mesh.nelements == 8

    # Simplex, 2D, X
    mesh = mgen.generate_box_mesh(
        2*(np.linspace(0, 1, 3),), group_cls=SimplexElementGroup, mesh_type="X")

    assert mesh.ambient_dim == 2
    assert mesh.dim == 2
    assert mesh.nvertices == 13
    assert len(mesh.groups) == 1
    assert mesh.nelements == 16

    # Tensor product, 2D
    mesh = mgen.generate_box_mesh(
        2*(np.linspace(0, 1, 3),), group_cls=TensorProductElementGroup)

    assert mesh.ambient_dim == 2
    assert mesh.dim == 2
    assert mesh.nvertices == 9
    assert len(mesh.groups) == 1
    assert mesh.nelements == 4

    # Simplex, 3D
    mesh = mgen.generate_box_mesh(
        3*(np.linspace(0, 1, 3),), group_cls=SimplexElementGroup)

    assert mesh.ambient_dim == 3
    assert mesh.dim == 3
    assert mesh.nvertices == 27
    assert len(mesh.groups) == 1
    assert mesh.nelements == 48

    # Tensor product, 3D
    mesh = mgen.generate_box_mesh(
        3*(np.linspace(0, 1, 3),), group_cls=TensorProductElementGroup)

    assert mesh.ambient_dim == 3
    assert mesh.dim == 3
    assert mesh.nvertices == 27
    assert len(mesh.groups) == 1
    assert mesh.nelements == 8

    # Simplex, empty mesh
    mesh = mgen.generate_box_mesh(
        3*(np.empty((0,)),), group_cls=SimplexElementGroup)

    assert mesh.ambient_dim == 3
    assert mesh.dim == 3
    assert mesh.nvertices == 0
    assert len(mesh.groups) == 1
    assert mesh.nelements == 0

    # Tensor product, empty mesh
    mesh = mgen.generate_box_mesh(
        3*(np.empty((0,)),), group_cls=TensorProductElementGroup)

    assert mesh.ambient_dim == 3
    assert mesh.dim == 3
    assert mesh.nvertices == 0
    assert len(mesh.groups) == 1
    assert mesh.nelements == 0

# }}}


# {{{ circle mesh

def test_circle_mesh(visualize=False):
    from meshmode.mesh.io import FileSource, generate_gmsh
    logger.info("BEGIN GEN")
    mesh = generate_gmsh(
            FileSource(str(thisdir / "circle.step")), 2, order=2,
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
    for d in range(1, 5):
        for _ in range(100):
            a = np.random.randn(d, d)+10*np.eye(d)
            b = np.random.randn(d)

            m = AffineMap(a, b)
            assert la.norm(m.inverted().matrix - la.inv(a)) < 1e-10*la.norm(a)

            x = np.random.randn(d)
            m_inv = m.inverted()
            assert la.norm(x-m_inv(m(x))) < 1e-10


def test_partial_affine_map(dim=2):
    orig_mesh = mgen.generate_regular_rect_mesh(
            a=(0,)*dim, b=(5, 3, 4)[:dim], npoints_per_axis=(10, 6, 7)[:dim],
            order=1)

    from meshmode.mesh.processing import affine_map
    mesh = affine_map(orig_mesh, b=np.pi)
    mesh = affine_map(orig_mesh, b=np.pi)
    assert la.norm(orig_mesh.vertices - mesh.vertices + np.pi) < 1.0e-14

    mesh = affine_map(orig_mesh, b=np.array([np.pi] * dim))
    assert la.norm(orig_mesh.vertices - mesh.vertices + np.pi) < 1.0e-14

    mesh = affine_map(orig_mesh, A=np.pi)
    mesh = affine_map(orig_mesh, A=np.pi)
    assert la.norm(orig_mesh.vertices - mesh.vertices / np.pi) < 1.0e-14

    mesh = affine_map(orig_mesh, A=np.pi * np.eye(dim))
    assert la.norm(orig_mesh.vertices - mesh.vertices / np.pi) < 1.0e-14


def test_affine_map_with_facial_adjacency_maps(visualize=False):
    orig_mesh = mgen.generate_annular_cylinder_slice_mesh(
        4, (1, 2, 0), 0.5, 1, periodic=True)

    if visualize:
        from meshmode.mesh.visualization import write_vertex_vtk_file
        write_vertex_vtk_file(orig_mesh, "affine_map_facial_adj_original.vtu")

    from meshmode.mesh.processing import affine_map

    tol = 1e-12

    def almost_equal(map1, map2):
        def component_almost_equal(array1, array2):
            if isinstance(array1, np.ndarray) and isinstance(array2, np.ndarray):
                return la.norm(array1 - array2) < tol
            else:
                return array1 == array2

        return (
            component_almost_equal(map1.matrix, map2.matrix)
            and component_almost_equal(map1.offset, map2.offset))

    # Matrix only
    mesh = affine_map(orig_mesh, A=_get_rotation(np.pi/2, axis=np.array([0, 0, 1])))

    if visualize:
        write_vertex_vtk_file(mesh, "affine_map_facial_adj_matrix.vtu")

    int_grps = [
        fagrp for fagrp in mesh.facial_adjacency_groups[0]
        if isinstance(fagrp, InteriorAdjacencyGroup)]
    assert len(int_grps) == 3

    lower_grp = int_grps[1]
    upper_grp = int_grps[2]

    assert almost_equal(
        lower_grp.aff_map,
        _get_rotation(
            np.pi/2, axis=np.array([0, 0, 1]), center=np.array([-2, 1, 0])))
    assert almost_equal(
        upper_grp.aff_map,
        _get_rotation(
            -np.pi/2, axis=np.array([0, 0, 1]), center=np.array([-2, 1, 0])))

    # Offset only
    mesh = affine_map(orig_mesh, b=np.array([0, -2, 0]))

    if visualize:
        write_vertex_vtk_file(mesh, "affine_map_facial_adj_offset.vtu")

    int_grps = [
        fagrp for fagrp in mesh.facial_adjacency_groups[0]
        if isinstance(fagrp, InteriorAdjacencyGroup)]
    assert len(int_grps) == 3

    lower_grp = int_grps[1]
    upper_grp = int_grps[2]

    assert almost_equal(
        lower_grp.aff_map,
        _get_rotation(
            np.pi/2, axis=np.array([0, 0, 1]), center=np.array([1, 0, 0])))
    assert almost_equal(
        upper_grp.aff_map,
        _get_rotation(
            -np.pi/2, axis=np.array([0, 0, 1]), center=np.array([1, 0, 0])))

    # Matrix and offset
    aff_map = _get_rotation(
        np.pi/2, axis=np.array([0, 0, 1]), center=np.array([1, 1, 0]))
    mesh = affine_map(orig_mesh, A=aff_map.matrix, b=aff_map.offset)

    if visualize:
        write_vertex_vtk_file(mesh, "affine_map_facial_adj_matrix_and_offset.vtu")

    int_grps = [
        fagrp for fagrp in mesh.facial_adjacency_groups[0]
        if isinstance(fagrp, InteriorAdjacencyGroup)]
    assert len(int_grps) == 3

    lower_grp = int_grps[1]
    upper_grp = int_grps[2]

    assert almost_equal(
        lower_grp.aff_map,
        _get_rotation(
            np.pi/2, axis=np.array([0, 0, 1]), center=np.array([0, 1, 0])))
    assert almost_equal(
        upper_grp.aff_map,
        _get_rotation(
            -np.pi/2, axis=np.array([0, 0, 1]), center=np.array([0, 1, 0])))


@pytest.mark.parametrize("ambient_dim", [2, 3])
def test_mesh_rotation(ambient_dim, visualize=False):
    order = 3

    if ambient_dim == 2:
        nelements = 256
        mesh = mgen.make_curve_mesh(
                # partial(mgen.ellipse, 2.0),
                partial(mgen.clamp_piecewise, 1.0, 2 / 3, np.pi / 6),
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
    from meshmode.mesh.io import FileSource, generate_gmsh

    h = 0.3
    order = 1

    mesh = generate_gmsh(
            FileSource(str(thisdir / "blob-2d.step")), 2, order=order,
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
    from meshmode.mesh.io import FileSource, generate_gmsh

    order = 3
    mesh_order = 3

    if group_cls is SimplexElementGroup:
        mesh = generate_gmsh(
                FileSource(str(thisdir / "blob-2d.step")), 2, order=mesh_order,
                force_ambient_dim=2,
                other_options=["-string", "Mesh.CharacteristicLengthMax = 0.02;"],
                target_unit="MM",
                )

        discr_grp_factory = default_simplex_group_factory(base_dim=2, order=order)
    else:
        ambient_dim = 3
        mesh = mgen.generate_regular_rect_mesh(
                a=(0,)*ambient_dim, b=(1,)*ambient_dim,
                nelements_per_axis=(4,)*ambient_dim, order=mesh_order,
                group_cls=group_cls)

        discr_grp_factory = LegendreGaussLobattoTensorProductGroupFactory(order)

    from meshmode.mesh.processing import affine_map, merge_disjoint_meshes
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

def test_element_orientation_via_flipping():
    from meshmode.mesh.io import FileSource, generate_gmsh

    mesh_order = 3

    mesh = generate_gmsh(
            FileSource(str(thisdir / "blob-2d.step")), 2, order=mesh_order,
            force_ambient_dim=2,
            other_options=["-string", "Mesh.CharacteristicLengthMax = 0.02;"],
            target_unit="MM",
            )

    from meshmode.mesh.processing import (
        find_volume_mesh_element_orientations, perform_flips)
    mesh_orient = find_volume_mesh_element_orientations(mesh)

    assert (mesh_orient > 0).all()

    from random import randrange
    flippy = np.zeros(mesh.nelements, np.int8)
    for _ in range(int(0.3*mesh.nelements)):
        flippy[randrange(0, mesh.nelements)] = 1

    mesh = perform_flips(mesh, flippy, skip_tests=True)

    mesh_orient = find_volume_mesh_element_orientations(mesh)

    assert ((mesh_orient < 0) == (flippy > 0)).all()


@pytest.mark.parametrize("order", [1, 2, 3])
def test_element_orientation_via_single_elements(order):
    from meshmode.mesh.processing import find_volume_mesh_element_group_orientation

    def check(vertices, element_indices, tol=1e-14):
        grp = mgen.make_group_from_vertices(vertices, element_indices, order)
        orient = find_volume_mesh_element_group_orientation(vertices, grp)
        return (
                np.where(orient > tol)[0],
                np.where(orient < tol)[0],
                np.where(np.abs(orient) <= tol)[0])

    # References:
    # https://github.com/inducer/meshmode/pull/314
    # https://github.com/lukeolson/mesh_orientation/blob/460bb2b634e2abb6aa32c3d02e2c732969bf08bf/check.py
    # https://math.stackexchange.com/questions/4209203/signed-volume-for-tetrahedra-n-simplices

    # 3D (pos)
    vertices = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 0],
                  [0, 0, 1]])
    elements = np.array([[0, 1, 2, 3]])
    el_ind_pos, el_ind_neg, el_ind_zero = check(vertices.T, elements)
    assert len(el_ind_pos) == 1
    assert len(el_ind_neg) == 0
    assert len(el_ind_zero) == 0

    # (neg)
    elements = np.array([[1, 0, 2, 3]])
    el_ind_pos, el_ind_neg, el_ind_zero = check(vertices.T, elements)
    assert len(el_ind_pos) == 0
    assert len(el_ind_neg) == 1
    assert len(el_ind_zero) == 0

    # 2D
    # CCW (positive)
    vertices = np.array([[1, 0],
                  [0, 1],
                  [0, 0]])
    elements = np.array([[0, 1, 2]])
    el_ind_pos, el_ind_neg, el_ind_zero = check(vertices.T, elements)
    assert len(el_ind_pos) == 1
    assert len(el_ind_neg) == 0
    assert len(el_ind_zero) == 0

    # CW (negative)
    elements = np.array([[0, 2, 1]])
    el_ind_pos, el_ind_neg, el_ind_zero = check(vertices.T, elements)
    assert len(el_ind_pos) == 0
    assert len(el_ind_neg) == 1
    assert len(el_ind_zero) == 0

    mesh = mio.read_gmsh(str(thisdir / "testmesh.msh"), force_ambient_dim=2,
                         mesh_construction_kwargs={"skip_tests": True})
    mgrp, = mesh.groups
    el_ind_pos, el_ind_neg, el_ind_zero = check(mesh.vertices, mgrp.vertex_indices)
    assert len(el_ind_pos) == 1
    assert len(el_ind_neg) == 1
    assert len(el_ind_zero) == 0

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
        mesh = mgen.generate_sphere(r=1.0, order=order)
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
    mesh = read_gmsh(str(thisdir / "annulus.msh"))

    # correct answers
    num_on_outer_bdy = 26
    num_on_inner_bdy = 13

    # check how many elements are marked on each boundary
    num_marked_outer_bdy = 0
    num_marked_inner_bdy = 0
    for igrp in range(len(mesh.groups)):
        bdry_fagrps = [
            fagrp for fagrp in mesh.facial_adjacency_groups[igrp]
            if isinstance(fagrp, BoundaryAdjacencyGroup)]
        for bdry_fagrp in bdry_fagrps:
            if bdry_fagrp.boundary_tag == "outer_bdy":
                num_marked_outer_bdy += len(bdry_fagrp.elements)
            if bdry_fagrp.boundary_tag == "inner_bdy":
                num_marked_inner_bdy += len(bdry_fagrp.elements)

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


# {{{ test volume tags

def test_volume_tags():
    from meshmode.mesh.io import read_gmsh
    mesh, tag_to_elements_map = read_gmsh(
        str(thisdir / "testmesh_multivol.msh"), return_tag_to_elements_map=True)

    assert len(tag_to_elements_map) == 2

    assert "Vol1" in tag_to_elements_map
    assert "Vol2" in tag_to_elements_map

    assert isinstance(tag_to_elements_map["Vol1"], np.ndarray)

    assert np.all(tag_to_elements_map["Vol1"] == np.array([0]))
    assert np.all(tag_to_elements_map["Vol2"] == np.array([1]))

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

    from meshmode.mesh import (
        check_bc_coverage, is_boundary_tag_empty, mesh_has_boundary)

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

    assert mesh_has_boundary(mesh, "btag_test_1")
    assert mesh_has_boundary(mesh, "btag_test_2")
    # Make sure mesh_has_boundary is working
    assert not mesh_has_boundary(mesh, "btag_test_3")

    assert not is_boundary_tag_empty(mesh, "btag_test_1")
    assert not is_boundary_tag_empty(mesh, "btag_test_2")
    check_bc_coverage(mesh, ["btag_test_1", "btag_test_2"])

    # check how many elements are marked on each boundary
    num_marked_bdy_1 = 0
    num_marked_bdy_2 = 0
    for igrp in range(len(mesh.groups)):
        bdry_fagrps = [
            fagrp for fagrp in mesh.facial_adjacency_groups[igrp]
            if isinstance(fagrp, BoundaryAdjacencyGroup)]
        for bdry_fagrp in bdry_fagrps:
            if bdry_fagrp.boundary_tag == "btag_test_1":
                num_marked_bdy_1 += len(bdry_fagrp.elements)
            if bdry_fagrp.boundary_tag == "btag_test_2":
                num_marked_bdy_2 += len(bdry_fagrp.elements)

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
    from meshmode.mesh.io import ScriptWithFilesSource, generate_gmsh
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
                [str(thisdir / filename)]),
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


# {{{ test_cube_icosahedron

@pytest.mark.parametrize("order", [2, 3])
def test_cube_icosphere(actx_factory, order, visualize=False):
    mesh = mgen.generate_sphere(
            r=1.0, order=order,
            group_cls=TensorProductElementGroup,
            uniform_refinement_rounds=2,
            )

    if not visualize:
        return

    from meshmode.mesh.visualization import vtk_visualize_mesh
    actx = actx_factory()
    vtk_visualize_mesh(actx, mesh,
            f"quad_icosphere_order_{order:03d}.vtu",
            vtk_high_order=False, overwrite=True)

# }}}


# {{{ test_tensor_torus

@pytest.mark.parametrize("order", [3, 4])
def test_tensor_torus(actx_factory, order, visualize=False):
    mesh = mgen.generate_torus(
            r_major=10.0, r_minor=5,
            n_major=24, n_minor=12,
            order=order,
            group_cls=TensorProductElementGroup,
            )

    if not visualize:
        return

    from meshmode.mesh.visualization import vtk_visualize_mesh
    actx = actx_factory()
    vtk_visualize_mesh(actx, mesh,
            f"quad_torus_order_{order:03d}.vtu",
            vtk_high_order=False, overwrite=True)

# }}}


# {{{ test_node_vertex_consistency_check

def test_node_vertex_consistency_check(actx_factory):
    actx = actx_factory()

    from meshmode import InconsistentVerticesError

    dtype = np.float64
    tol = 1e3 * np.finfo(dtype).eps

    # Mesh bounds invariance

    def gen_rect_mesh_with_perturbed_vertices(
            a, b, nelements_per_axis, perturb_amount):
        mesh_unperturbed = mgen.generate_regular_rect_mesh(
            a=a, b=b, nelements_per_axis=nelements_per_axis)
        return mesh_unperturbed.copy(  # noqa: F841
            vertices=(
                mesh_unperturbed.vertices
                + perturb_amount*np.ones(mesh_unperturbed.vertices.shape)),
            node_vertex_consistency_tolerance=tol)

    # Find critical perturbation amount "w" such that vertices shifted by w works
    # but 10*w doesn't
    h = 1
    nelems = 8
    size = h*nelems
    crit_perturb = None
    for p in range(32):
        w = h*10**(p-16)
        try:
            gen_rect_mesh_with_perturbed_vertices(
                a=(-size/2,), b=(size/2,), nelements_per_axis=(nelems,),
                perturb_amount=w)
        except InconsistentVerticesError:
            if p > 0:
                crit_perturb = w/10
            break
    if crit_perturb is None:
        raise RuntimeError("failed to find critical vertex perturbation amount")

    # Scale invariance
    nelems = 8
    for p in range(32):
        h = 10**(p-16)
        size = h*nelems
        perturb = crit_perturb*h
        gen_rect_mesh_with_perturbed_vertices(
            a=(-size/2,)*3, b=(size/2,)*3, nelements_per_axis=(nelems,)*3,
            perturb_amount=perturb)
        if 10*perturb > tol:
            with pytest.raises(InconsistentVerticesError):
                gen_rect_mesh_with_perturbed_vertices(
                    a=(-size/2,)*3, b=(size/2,)*3, nelements_per_axis=(nelems,)*3,
                    perturb_amount=10*perturb)

    # Translation invariance
    h = 1
    nelems = 8
    size = h*nelems
    for p in range(10):
        shift = 10**p-1
        perturb = crit_perturb*(size/2 + shift)/(size/2)
        gen_rect_mesh_with_perturbed_vertices(
            a=(shift-size/2,)*3, b=(shift+size/2,)*3,
            nelements_per_axis=(nelems,)*3,
            perturb_amount=perturb)
        with pytest.raises(InconsistentVerticesError):
            gen_rect_mesh_with_perturbed_vertices(
                a=(shift-size/2,)*3, b=(shift+size/2,)*3,
                nelements_per_axis=(nelems,)*3,
                perturb_amount=10*perturb)

    # Aspect ratio invariance
    h = 1
    nelems = 8
    size = h*nelems
    for p in range(10):
        h_x = 10**p * h
        size_x = h_x * nelems
        perturb = crit_perturb*h_x
        gen_rect_mesh_with_perturbed_vertices(
            a=(-size_x/2,) + (-size/2,)*2,
            b=(size_x/2,) + (size/2,)*2,
            nelements_per_axis=(nelems,)*3,
            perturb_amount=perturb)
        with pytest.raises(InconsistentVerticesError):
            gen_rect_mesh_with_perturbed_vertices(
                a=(-size_x/2,) + (-size/2,)*2,
                b=(size_x/2,) + (size/2,)*2,
                nelements_per_axis=(nelems,)*3,
                perturb_amount=10*perturb)

    # Mesh size relative to element size invariance
    h = 1
    for p in range(5):
        nelems = 2**(5-p)
        size = h*nelems
        perturb = crit_perturb
        gen_rect_mesh_with_perturbed_vertices(
            a=(-size/2,)*3, b=(size/2,)*3, nelements_per_axis=(nelems,)*3,
            perturb_amount=perturb)
        with pytest.raises(InconsistentVerticesError):
            gen_rect_mesh_with_perturbed_vertices(
                a=(-size/2,)*3, b=(size/2,)*3, nelements_per_axis=(nelems,)*3,
                perturb_amount=10*perturb)

    # Zero-D elements
    h = 1
    nelems = 7
    size = h*nelems
    vol_mesh = mgen.generate_regular_rect_mesh(
        a=(-size/2,), b=(size/2,),
        nelements_per_axis=(nelems,))
    from meshmode.discretization import Discretization
    group_factory = default_simplex_group_factory(1, 1)
    vol_discr = Discretization(actx, vol_mesh, group_factory)
    from meshmode.discretization.connection import (
        FACE_RESTR_ALL, make_face_restriction)
    make_face_restriction(
        actx, vol_discr, group_factory, FACE_RESTR_ALL, per_face_groups=False)

    # Zero-D elements at the origin
    h = 1
    nelems = 8
    size = h*nelems
    vol_mesh = mgen.generate_regular_rect_mesh(
        a=(-size/2,), b=(size/2,),
        nelements_per_axis=(nelems,))
    group_factory = default_simplex_group_factory(1, 1)
    vol_discr = Discretization(actx, vol_mesh, group_factory)
    make_face_restriction(
        actx, vol_discr, group_factory, FACE_RESTR_ALL, per_face_groups=False)

    # Element vertex indices rotated
    with pytest.raises(InconsistentVerticesError):
        vol_mesh_unrotated = mgen.generate_regular_rect_mesh(
            a=(-1,)*2, b=(1,)*2,
            nelements_per_axis=(8,)*2)
        vol_mesh = vol_mesh_unrotated.copy(  # noqa: F841
            groups=[
                replace(
                    grp,
                    vertex_indices=np.roll(grp.vertex_indices, 1, axis=1))
                for grp in vol_mesh_unrotated.groups])

# }}}


# {{{ mesh boundary gluing

@pytest.mark.parametrize("use_tree", [False, True])
def test_glued_mesh(use_tree):
    n = 4
    center = (1, 2, 3)

    orig_mesh = mgen.generate_annular_cylinder_slice_mesh(n, center, 0.5, 1)

    map_lower_to_upper = _get_rotation(np.pi/2, np.array([0, 0, 1]), center)
    map_upper_to_lower = _get_rotation(-np.pi/2, np.array([0, 0, 1]), center)

    from meshmode.mesh.processing import BoundaryPairMapping, glue_mesh_boundaries
    mesh = glue_mesh_boundaries(
        orig_mesh, bdry_pair_mappings_and_tols=[
            (BoundaryPairMapping("-theta", "+theta", map_lower_to_upper), 1e-12)
        ], use_tree=use_tree)

    int_grps = [
        fagrp for fagrp in mesh.facial_adjacency_groups[0]
        if isinstance(fagrp, InteriorAdjacencyGroup)]
    assert len(int_grps) == 3

    bdry_grps = [
        fagrp for fagrp in mesh.facial_adjacency_groups[0]
        if isinstance(fagrp, BoundaryAdjacencyGroup)]
    assert len(bdry_grps) == (
        4  # +/-r and +/-z
        + 3  # BTAG_NONE, BTAG_ALL, BTAG_REALLY_ALL
        )

    lower_grp = int_grps[1]
    upper_grp = int_grps[2]

    from pytools import single_valued

    n_lower_faces = single_valued((
        len(lower_grp.elements),
        len(lower_grp.element_faces),
        len(lower_grp.neighbors),
        len(lower_grp.neighbor_faces)))
    assert n_lower_faces == 2*n**2
    assert lower_grp.aff_map == map_lower_to_upper

    n_upper_faces = single_valued((
        len(upper_grp.elements),
        len(upper_grp.element_faces),
        len(upper_grp.neighbors),
        len(upper_grp.neighbor_faces)))
    assert n_upper_faces == 2*n**2
    assert upper_grp.aff_map == map_upper_to_lower

    lower_face_indices = np.full(
        (orig_mesh.groups[0].nfaces, orig_mesh.groups[0].nelements), -1)
    upper_face_indices = np.full(
        (orig_mesh.groups[0].nfaces, orig_mesh.groups[0].nelements), -1)

    lower_face_indices[lower_grp.element_faces, lower_grp.elements] = (
        np.indices((n_lower_faces,)))
    upper_face_indices[upper_grp.element_faces, upper_grp.elements] = (
        np.indices((n_upper_faces,)))

    indices = upper_face_indices[lower_grp.neighbor_faces, lower_grp.neighbors]
    assert np.all(indices >= 0)
    assert np.all(upper_grp.neighbors[indices] == lower_grp.elements)
    assert np.all(upper_grp.neighbor_faces[indices] == lower_grp.element_faces)

    indices = lower_face_indices[upper_grp.neighbor_faces, upper_grp.neighbors]
    assert np.all(indices >= 0)
    assert np.all(lower_grp.neighbors[indices] == upper_grp.elements)
    assert np.all(lower_grp.neighbor_faces[indices] == upper_grp.element_faces)


def test_glued_mesh_matrix_only():
    n = 4
    orig_mesh = mgen.generate_annular_cylinder_slice_mesh(n, (0, 0, 0), 0.5, 1)

    matrix_lower_to_upper = _get_rotation(np.pi/2, np.array([0, 0, 1]))
    matrix_upper_to_lower = _get_rotation(-np.pi/2, np.array([0, 0, 1]))

    map_lower_to_upper = AffineMap(matrix=matrix_lower_to_upper)
    map_upper_to_lower = AffineMap(matrix=matrix_upper_to_lower)

    from meshmode.mesh.processing import BoundaryPairMapping, glue_mesh_boundaries
    mesh = glue_mesh_boundaries(
        orig_mesh, bdry_pair_mappings_and_tols=[
            (BoundaryPairMapping("-theta", "+theta", map_lower_to_upper), 1e-12)
        ])

    int_grps = [
        fagrp for fagrp in mesh.facial_adjacency_groups[0]
        if isinstance(fagrp, InteriorAdjacencyGroup)]

    lower_grp = int_grps[1]
    upper_grp = int_grps[2]

    assert lower_grp.aff_map == map_lower_to_upper
    assert upper_grp.aff_map == map_upper_to_lower


def test_glued_mesh_offset_only():
    n = 4
    orig_mesh = mgen.generate_annular_cylinder_slice_mesh(n, (0, 0, 0), 0.5, 1)

    offset_lower_to_upper = np.array([0, 0, 1])
    offset_upper_to_lower = np.array([0, 0, -1])

    map_lower_to_upper = AffineMap(offset=offset_lower_to_upper)
    map_upper_to_lower = AffineMap(offset=offset_upper_to_lower)

    from meshmode.mesh.processing import BoundaryPairMapping, glue_mesh_boundaries
    mesh = glue_mesh_boundaries(
        orig_mesh, bdry_pair_mappings_and_tols=[
            (BoundaryPairMapping("-z", "+z", map_lower_to_upper), 1e-12)
        ])

    int_grps = [
        fagrp for fagrp in mesh.facial_adjacency_groups[0]
        if isinstance(fagrp, InteriorAdjacencyGroup)]

    lower_grp = int_grps[1]
    upper_grp = int_grps[2]

    assert lower_grp.aff_map == map_lower_to_upper
    assert upper_grp.aff_map == map_upper_to_lower

# }}}


# {{{ test_mesh_grid

@pytest.mark.parametrize("mesh_name", ["starfish3", "dumbbell", "torus"])
@pytest.mark.parametrize("has_offset", [True, False])
def test_mesh_grid(actx_factory, mesh_name, has_offset, visualize=False):

    target_order = 4
    if mesh_name == "starfish3":
        nelements = 128
        mesh = mgen.make_curve_mesh(
            mgen.starfish3,
            np.linspace(0.0, 1.0, nelements + 1),
            order=target_order,
            )
        offset = (np.array([5, 0]), np.array([2.5, 3.0]))
    elif mesh_name == "dumbbell":
        nelements = 512
        mesh = mgen.make_curve_mesh(
            partial(mgen.wobbly_dumbbell, 0.01, 0.99, 1, 100),
            np.linspace(0.0, 1.0, nelements + 1),
            order=target_order,
            )
        offset = (np.array([3, 0]), np.array([0, 1]))
    elif mesh_name == "torus":
        mesh = mgen.generate_torus(2.0, 1.0, order=target_order)
        offset = (np.array([6.5, 0, 0]), np.array([0, 6.5, 0]), np.array([0, 0, 3]))
    else:
        raise ValueError(f"unknown mesh name: '{mesh_name}'")

    from meshmode.mesh.processing import make_mesh_grid
    shape = (6, 3, 2)[:mesh.ambient_dim]
    mgrid = make_mesh_grid(
        mesh,
        shape=shape,
        offset=offset if has_offset else None,
        skip_tests=False
        )

    from pytools import product
    m = len(mgrid.groups)
    assert m == product(shape)

    def separated(x, y):
        return np.all(
            np.linalg.norm(x[:, :, None] - y[:, None, :], axis=0) > 0.1
            )

    assert all(
        separated(mgrid.groups[i].nodes, mgrid.groups[j].nodes)
        for i, j in zip(range(m), range(m)) if i != j)

    if not visualize:
        return

    from meshmode.mesh.visualization import vtk_visualize_mesh
    actx = actx_factory()
    vtk_visualize_mesh(actx, mgrid,
            f"mesh_grid_{mesh_name}_{has_offset}.vtu".lower(),
            vtk_high_order=False, overwrite=True)

# }}}


def test_urchin():
    mgen.generate_urchin(3, 2, 4, 1e-4)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
