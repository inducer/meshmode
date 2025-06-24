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

import logging
import pathlib
from functools import partial

import numpy as np
import numpy.linalg as la
import pytest

from arraycontext import (
    ArrayContextFactory,
    flatten,
    pytest_generate_tests_for_array_contexts,
)

import meshmode.mesh.generation as mgen
from meshmode import _acf  # noqa: F401
from meshmode.array_context import (
    PytestPyOpenCLArrayContextFactory,
    PytestPytatoPyOpenCLArrayContextFactory,
)
from meshmode.discretization.connection import FACE_RESTR_ALL, FACE_RESTR_INTERIOR
from meshmode.discretization.poly_element import (
    InterpolatoryQuadratureSimplexGroupFactory,
    LegendreGaussLobattoTensorProductGroupFactory,
    PolynomialEquidistantSimplexGroupFactory,
    PolynomialRecursiveNodesGroupFactory,
    PolynomialWarpAndBlend2DRestrictingGroupFactory,
    PolynomialWarpAndBlend3DRestrictingGroupFactory,
    default_simplex_group_factory,
)
from meshmode.dof_array import flat_norm
from meshmode.mesh import (
    BTAG_ALL,
    Mesh,
    MeshElementGroup,
    SimplexElementGroup,
    TensorProductElementGroup,
    make_mesh,
)


logger = logging.getLogger(__name__)
pytest_generate_tests = pytest_generate_tests_for_array_contexts(
        [PytestPytatoPyOpenCLArrayContextFactory,
         PytestPyOpenCLArrayContextFactory,
         ])

thisdir = pathlib.Path(__file__).parent


def normalize_group_factory(dim, grp_factory):
    if grp_factory == "warp_and_blend":
        return {
            0: PolynomialWarpAndBlend2DRestrictingGroupFactory,
            1: PolynomialWarpAndBlend2DRestrictingGroupFactory,
            2: PolynomialWarpAndBlend2DRestrictingGroupFactory,
            3: PolynomialWarpAndBlend3DRestrictingGroupFactory,
            }[dim]
    else:
        assert not isinstance(grp_factory, str)
        return grp_factory


# {{{ convergence of boundary interpolation

@pytest.mark.parametrize("group_factory", [
    InterpolatoryQuadratureSimplexGroupFactory,
    "warp_and_blend",
    partial(PolynomialRecursiveNodesGroupFactory, family="lgl"),

    # Redundant, no information gain.
    # partial(PolynomialRecursiveNodesGroupFactory, family="gc"),

    LegendreGaussLobattoTensorProductGroupFactory,
    ])
@pytest.mark.parametrize("boundary_tag", [
    BTAG_ALL,
    FACE_RESTR_ALL,
    FACE_RESTR_INTERIOR,
    ])
@pytest.mark.parametrize(("mesh_name", "dim", "mesh_pars"), [
    ("blob", 2, ["8e-2", "6e-2", "4e-2"]),

    # If "warp" works, "rect" likely does, too.
    # ("rect", 3, [10, 20, 30]),

    ("warp", 2, [10, 20, 30]),
    ("warp", 3, [10, 20, 30]),
    ])
@pytest.mark.parametrize("per_face_groups", [True, False])
def test_boundary_interpolation(
            actx_factory: ArrayContextFactory,
            group_factory,
            boundary_tag,
            mesh_name: str,
            dim,
            mesh_pars,
            per_face_groups):
    if (group_factory is LegendreGaussLobattoTensorProductGroupFactory
            and mesh_name == "blob"):
        pytest.skip("tensor products not implemented on blobs")

    actx = actx_factory()

    group_factory = normalize_group_factory(dim, group_factory)
    if group_factory is LegendreGaussLobattoTensorProductGroupFactory:
        group_cls = TensorProductElementGroup
    else:
        group_cls = SimplexElementGroup

    from pytools.convergence import EOCRecorder

    from meshmode.discretization import Discretization
    from meshmode.discretization.connection import (
        check_connection,
        make_face_restriction,
    )
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
                    str(thisdir / f"blob2d-order{order}-h{mesh_par}.msh"),
                    force_ambient_dim=2)
        elif mesh_name == "warp":
            mesh = mgen.generate_warped_rect_mesh(dim, order=order,
                    nelements_side=mesh_par, group_cls=group_cls)

            h = 1/mesh_par

        elif mesh_name == "rect":
            mesh = mgen.generate_regular_rect_mesh(a=(0,)*dim, b=(1,)*dim,
                    order=order, nelements_per_axis=(mesh_par,)*dim,
                    group_cls=group_cls)

            h = 1/mesh_par
        else:
            raise ValueError("mesh_name not recognized")

        # }}}

        vol_discr = Discretization(actx, mesh, group_factory(order))
        print("h=%s -> %d elements" % (
                h, sum(mgrp.nelements for mgrp in mesh.groups)))

        x = actx.thaw(vol_discr.nodes()[0])
        vol_f = f(x)

        bdry_connection = make_face_restriction(
                actx, vol_discr, group_factory(order),
                boundary_tag, per_face_groups=per_face_groups)
        check_connection(actx, bdry_connection)
        bdry_discr = bdry_connection.to_discr

        bdry_x = actx.thaw(bdry_discr.nodes()[0])
        bdry_f = f(bdry_x)
        bdry_f_2 = bdry_connection(vol_f)

        if mesh_name == "blob" and dim == 2 and mesh.nelements < 500:
            from meshmode.discretization.connection.direct import (
                make_direct_full_resample_matrix,
            )
            mat = actx.to_numpy(
                    make_direct_full_resample_matrix(actx, bdry_connection))
            bdry_f_2_by_mat = mat.dot(actx.to_numpy(flatten(vol_f, actx)))

            mat_error = la.norm(
                    actx.to_numpy(flatten(bdry_f_2, actx)) - bdry_f_2_by_mat)
            assert mat_error < 1e-14, mat_error

        err = flat_norm(bdry_f-bdry_f_2, np.inf)
        eoc_rec.add_data_point(h, actx.to_numpy(err))

    order_slack = 0.75 if mesh_name == "blob" else 0.5
    print(eoc_rec)
    assert (
            eoc_rec.order_estimate() >= order-order_slack
            or eoc_rec.max_error() < 3.6e-13)

# }}}


# {{{ boundary-to-all-faces connection

@pytest.mark.parametrize("group_factory", [
    InterpolatoryQuadratureSimplexGroupFactory,
    "warp_and_blend",
    partial(PolynomialRecursiveNodesGroupFactory, family="lgl"),
    LegendreGaussLobattoTensorProductGroupFactory,
    ])
@pytest.mark.parametrize(("mesh_name", "dim", "mesh_pars"), [
    ("blob", 2, [1e-1, 8e-2, 5e-2]),
    ("warp", 2, [10, 20, 30]),
    ("warp", 3, [10, 20, 30]),
    ])
@pytest.mark.parametrize("per_face_groups", [False, True])
def test_all_faces_interpolation(actx_factory: ArrayContextFactory, group_factory,
        mesh_name, dim, mesh_pars, per_face_groups):
    if (group_factory is LegendreGaussLobattoTensorProductGroupFactory
            and mesh_name == "blob"):
        pytest.skip("tensor products not implemented on blobs")

    actx = actx_factory()

    group_factory = normalize_group_factory(dim, group_factory)

    if group_factory is LegendreGaussLobattoTensorProductGroupFactory:
        group_cls = TensorProductElementGroup
    else:
        group_cls = SimplexElementGroup

    from pytools.convergence import EOCRecorder

    from meshmode.discretization import Discretization
    from meshmode.discretization.connection import (
        check_connection,
        make_face_restriction,
        make_face_to_all_faces_embedding,
    )
    eoc_rec = EOCRecorder()

    order = 4

    def f(x):
        return 0.1*actx.np.sin(30*x)

    for mesh_par in mesh_pars:
        # {{{ get mesh

        if mesh_name == "blob":
            assert dim == 2

            h = mesh_par

            from meshmode.mesh.io import FileSource, generate_gmsh
            print("BEGIN GEN")
            mesh = generate_gmsh(
                    FileSource(str(thisdir / "blob-2d.step")), 2, order=order,
                    force_ambient_dim=2,
                    other_options=[
                        "-string", f"Mesh.CharacteristicLengthMax = {h};"],
                    target_unit="MM",
                    )
            print("END GEN")
        elif mesh_name == "warp":
            mesh = mgen.generate_warped_rect_mesh(dim, order=4,
                    nelements_side=mesh_par, group_cls=group_cls)

            h = 1/mesh_par
        else:
            raise ValueError("mesh_name not recognized")

        # }}}

        vol_discr = Discretization(actx, mesh, group_factory(order))
        print("h=%s -> %d elements" % (
                h, sum(mgrp.nelements for mgrp in mesh.groups)))

        all_face_bdry_connection = make_face_restriction(
                actx, vol_discr, group_factory(order),
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

        all_face_x = actx.thaw(all_face_bdry_discr.nodes()[0])
        all_face_f = f(all_face_x)

        all_face_f_2 = all_face_bdry_discr.zeros(actx)

        for boundary_tag in [
                BTAG_ALL,
                FACE_RESTR_INTERIOR,
                ]:
            bdry_connection = make_face_restriction(
                    actx, vol_discr, group_factory(order),
                    boundary_tag, per_face_groups=per_face_groups)
            bdry_discr = bdry_connection.to_discr

            bdry_x = actx.thaw(bdry_discr.nodes()[0])
            bdry_f = f(bdry_x)

            all_face_embedding = make_face_to_all_faces_embedding(
                    actx, bdry_connection, all_face_bdry_discr)

            check_connection(actx, all_face_embedding)

            all_face_f_2 = all_face_f_2 + all_face_embedding(bdry_f)

        err = flat_norm(all_face_f-all_face_f_2, np.inf)
        eoc_rec.add_data_point(h, actx.to_numpy(err))

    print(eoc_rec)
    assert (
            eoc_rec.order_estimate() >= order-0.5
            or eoc_rec.max_error() < 1e-14)

# }}}


# {{{ convergence of opposite-face interpolation

@pytest.mark.parametrize("group_factory", [
    InterpolatoryQuadratureSimplexGroupFactory,
    "warp_and_blend",
    LegendreGaussLobattoTensorProductGroupFactory,
    ])
@pytest.mark.parametrize(("mesh_name", "dim", "mesh_pars"), [
    ("segment", 1, [8, 16, 32]),
    ("blob", 2, [1e-1, 8e-2, 5e-2]),
    ("warp", 2, [3, 5, 7]),
    ("warp", 3, [5, 7]),
    ("periodic", 2, [3, 5, 7]),
    ("periodic", 3, [5, 7])
    ])
def test_opposite_face_interpolation(actx_factory: ArrayContextFactory, group_factory,
        mesh_name, dim, mesh_pars):
    if (group_factory is LegendreGaussLobattoTensorProductGroupFactory
            and mesh_name in ["segment", "blob", "periodic"]):
        pytest.skip(f"tensor products not implemented on {mesh_name}")

    logging.basicConfig(level=logging.INFO)
    actx = actx_factory()

    group_factory = normalize_group_factory(dim, group_factory)

    if group_factory is LegendreGaussLobattoTensorProductGroupFactory:
        group_cls = TensorProductElementGroup
    else:
        group_cls = SimplexElementGroup

    from pytools.convergence import EOCRecorder

    from meshmode.discretization import Discretization
    from meshmode.discretization.connection import (
        check_connection,
        make_face_restriction,
        make_opposite_face_connection,
    )
    eoc_rec = EOCRecorder()

    order = 5

    def f(x):
        return 0.1*actx.np.sin(30*x)

    for mesh_par in mesh_pars:
        # {{{ get mesh

        if mesh_name == "segment":
            assert dim == 1

            mesh = mgen.generate_box_mesh(
                    [np.linspace(-0.5, 0.5, mesh_par)],
                    order=order,
                    group_cls=group_cls)
            h = 1.0 / mesh_par
        elif mesh_name == "blob":
            assert dim == 2

            h = mesh_par

            from meshmode.mesh.io import FileSource, generate_gmsh
            print("BEGIN GEN")
            mesh = generate_gmsh(
                    FileSource(str(thisdir / "blob-2d.step")), 2, order=order,
                    force_ambient_dim=2,
                    other_options=[
                        "-string", f"Mesh.CharacteristicLengthMax = {h};"],
                    target_unit="MM",
                    )
            print("END GEN")
        elif mesh_name == "warp":
            mesh = mgen.generate_warped_rect_mesh(dim, order=order,
                    nelements_side=mesh_par, group_cls=group_cls)

            h = 1/mesh_par
        elif mesh_name == "periodic":
            assert dim == 2 or dim == 3

            if dim == 2:
                mesh = mgen.generate_regular_rect_mesh(
                    a=(-np.pi/2,)*dim,
                    b=((3*np.pi)/2,)*dim,
                    nelements_per_axis=(mesh_par,)*dim,
                    periodic=(True, False))

                h = 1/mesh_par
            else:
                mesh = mgen.generate_annular_cylinder_slice_mesh(
                    mesh_par, (1, 2, 3), 0.5, 1, periodic=True)

                h = 1/mesh_par
        else:
            raise ValueError("mesh_name not recognized")

        # }}}

        vol_discr = Discretization(actx, mesh, group_factory(order))
        print("h=%s -> %d elements" % (
                h, sum(mgrp.nelements for mgrp in mesh.groups)))

        bdry_connection = make_face_restriction(
                actx, vol_discr, group_factory(order),
                FACE_RESTR_INTERIOR)
        bdry_discr = bdry_connection.to_discr

        opp_face = make_opposite_face_connection(actx, bdry_connection)
        check_connection(actx, opp_face)

        bdry_x = actx.thaw(bdry_discr.nodes()[0])
        bdry_f = f(bdry_x)
        bdry_f_2 = opp_face(bdry_f)

        # Ensure test coverage for alternate modes in DirectConnection
        for force_loopy, force_no_merged_batches in [
                (False, True),
                (True, False),
                (True, True),
                ]:
            bdry_f_2_alt = opp_face(bdry_f,
                    _force_use_loopy=force_loopy,
                    _force_no_merged_batches=force_no_merged_batches)
            assert actx.to_numpy(flat_norm(bdry_f_2 - bdry_f_2_alt, np.inf)) < 1e-14

        err = flat_norm(bdry_f-bdry_f_2, np.inf)
        eoc_rec.add_data_point(h, actx.to_numpy(err))

    print(eoc_rec)
    assert (
            eoc_rec.order_estimate() >= order-0.5
            or eoc_rec.max_error() < 1.7e-13)

# }}}


# {{{ element orientation: canned 3D meshes

# python test_meshmode.py "test_sanity_balls(cl._csc, "disk-radius-1.step", 2, 2, visualize=True)"  # noqa: E501
@pytest.mark.parametrize(("what", "mesh_gen_func"), [
    ("ball", lambda: mgen.generate_icosahedron(1, 1)),
    ("torus", lambda: mgen.generate_torus(5, 1)),
    ])
def test_orientation_3d(
            actx_factory: ArrayContextFactory,
            what,
            mesh_gen_func,
            visualize=False):
    from arraycontext import PyOpenCLArrayContext
    pytest.importorskip("pytential")

    logging.basicConfig(level=logging.INFO)
    actx = actx_factory()

    if not isinstance(actx, PyOpenCLArrayContext):
        pytest.skip(f"{actx}: not supported by pytential")

    mesh = mesh_gen_func()

    logger.info("%d elements", mesh.nelements)

    from meshmode.discretization import Discretization
    discr = Discretization(actx, mesh,
            default_simplex_group_factory(base_dim=3, order=3))

    from pytential import bind, sym

    # {{{ check normals point outward

    if what == "torus":
        nodes = sym.nodes(mesh.ambient_dim).as_vector()
        angle = sym.arctan2(nodes[1], nodes[0])
        center_nodes = sym.make_obj_array([
                5*sym.cos(angle),
                5*sym.sin(angle),
                0*angle])
        normal_outward_expr = (
                sym.normal(mesh.ambient_dim) | (nodes-center_nodes))

    else:
        normal_outward_expr = (
                sym.normal(mesh.ambient_dim) | sym.nodes(mesh.ambient_dim))

    normal_outward_check = actx.to_numpy(flatten(
            bind(discr, normal_outward_expr)(actx).as_scalar(),
            actx)) > 0

    assert normal_outward_check.all(), normal_outward_check

    # }}}

    normals = bind(discr, sym.normal(mesh.ambient_dim).xproject(1))(actx)

    if visualize:
        from meshmode.discretization.visualization import make_visualizer
        vis = make_visualizer(actx, discr, 3)

        vis.write_vtk_file(f"orientation_3d_{what}_normals.vtu", [
            ("normals", normals),
            ])

# }}}


# {{{ sanity checks: single element

@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("mesh_order", [1, 3])
@pytest.mark.parametrize("group_cls", [
    SimplexElementGroup,
    TensorProductElementGroup,
    ])
def test_sanity_single_element(
            actx_factory: ArrayContextFactory,
            dim,
            mesh_order,
            group_cls,
            visualize=False):
    from arraycontext import PyOpenCLArrayContext
    pytest.importorskip("pytential")
    actx = actx_factory()

    if not isinstance(actx, PyOpenCLArrayContext):
        pytest.skip(f"{actx}: not supported by pytential")

    if group_cls is SimplexElementGroup:
        group_factory = default_simplex_group_factory(dim, order=mesh_order + 3)
    elif group_cls is TensorProductElementGroup:
        group_factory = LegendreGaussLobattoTensorProductGroupFactory(mesh_order + 3)
    else:
        raise TypeError

    import modepy as mp
    shape = group_cls._modepy_shape_cls(dim)
    space = mp.space_for_shape(shape, mesh_order)

    vertices = mp.unit_vertices_for_shape(shape)
    nodes = mp.edge_clustered_nodes_for_space(space, shape).reshape(dim, 1, -1)
    vertex_indices = np.arange(shape.nvertices, dtype=np.int32).reshape(1, -1)

    mg = group_cls.make_group(mesh_order, vertex_indices, nodes, dim=dim)
    mesh = make_mesh(vertices, [mg], is_conforming=True)

    from meshmode.discretization import Discretization
    vol_discr = Discretization(actx, mesh, group_factory)

    # {{{ volume calculation check

    if isinstance(mg, SimplexElementGroup):
        from math import factorial
        true_vol = 1/factorial(dim) * 2**dim
    elif isinstance(mg, TensorProductElementGroup):
        true_vol = 2**dim
    else:
        raise TypeError

    nodes = actx.thaw(vol_discr.nodes())
    vol_one = 1 + 0 * nodes[0]

    from pytential import integral
    comp_vol = integral(vol_discr, vol_one)
    rel_vol_err = abs(true_vol - comp_vol) / true_vol

    assert rel_vol_err < 1e-12

    # }}}

    # {{{ boundary discretization

    from meshmode.discretization.connection import make_face_restriction
    bdry_connection = make_face_restriction(
            actx, vol_discr, group_factory,
            BTAG_ALL)
    bdry_discr = bdry_connection.to_discr

    # }}}

    from pytential import bind, sym
    bdry_normals = bind(bdry_discr, sym.normal(dim).as_vector())(actx)

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

    normal_outward_check = actx.to_numpy(flatten(normal_outward_check > 0, actx))
    assert normal_outward_check.all(), normal_outward_check

# }}}


# {{{ sanity checks: no elements

@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("mesh_order", [1, 3])
@pytest.mark.parametrize("group_cls", [
    SimplexElementGroup,
    TensorProductElementGroup,
    ])
def test_sanity_no_elements(
            actx_factory: ArrayContextFactory,
            dim,
            mesh_order,
            group_cls,
            visualize=False):
    from arraycontext import PyOpenCLArrayContext
    pytest.importorskip("pytential")
    actx = actx_factory()

    if not isinstance(actx, PyOpenCLArrayContext):
        pytest.skip(f"{actx}: not supported by pytential")

    if group_cls is SimplexElementGroup:
        group_factory = default_simplex_group_factory(dim, order=mesh_order + 3)
    elif group_cls is TensorProductElementGroup:
        group_factory = LegendreGaussLobattoTensorProductGroupFactory(mesh_order + 3)
    else:
        raise TypeError

    import modepy as mp
    shape = group_cls._modepy_shape_cls(dim)
    space = mp.space_for_shape(shape, mesh_order)
    nunit_nodes = mp.edge_clustered_nodes_for_space(space, shape).shape[1]

    vertices = np.empty((dim, 0))
    nodes = np.empty((dim, 0, nunit_nodes))
    vertex_indices = np.empty((0, shape.nvertices), dtype=np.int32)

    mg = group_cls.make_group(mesh_order, vertex_indices, nodes, dim=dim)
    mesh = make_mesh(vertices, [mg], is_conforming=True)

    from meshmode.discretization import Discretization
    vol_discr = Discretization(actx, mesh, group_factory)

    # {{{ volume calculation check

    nodes = actx.thaw(vol_discr.nodes())
    vol_one = 1 + 0 * nodes[0]

    from pytential import integral
    assert integral(vol_discr, vol_one) == 0.

    # }}}

    # {{{ boundary discretization

    from meshmode.discretization.connection import make_face_restriction
    bdry_connection = make_face_restriction(
            actx, vol_discr, group_factory,
            BTAG_ALL)
    bdry_discr = bdry_connection.to_discr

    # }}}

    from pytential import bind, sym
    normal_outward_check = bind(bdry_discr,
            sym.normal(dim)
            | (sym.nodes(dim) + 0.5*sym.ones_vec(dim)),
            )(actx).as_scalar()

    normal_outward_check = actx.to_numpy(flatten(normal_outward_check > 0, actx))
    assert normal_outward_check.all(), normal_outward_check

# }}}


# {{{ sanity check: volume interpolation on scipy/qhull delaunay meshes in nD

@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("order", [3])
def test_sanity_qhull_nd(actx_factory: ArrayContextFactory, dim, order):
    pytest.importorskip("scipy")

    logging.basicConfig(level=logging.INFO)
    actx = actx_factory()
    rng = np.random.default_rng(seed=42)

    from scipy.spatial import Delaunay  # pylint: disable=no-name-in-module
    verts = rng.random(size=(1000, dim))
    dtri = Delaunay(verts)

    # pylint: disable=no-member
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

    x_low = actx.thaw(low_discr.nodes()[0])
    f_low = f(x_low)

    x_high = actx.thaw(high_discr.nodes()[0])
    f_high_ref = f(x_high)

    f_high_num = cnx(f_low)

    err = (
            flat_norm(f_high_ref-f_high_num, np.inf)
            / flat_norm(f_high_ref, np.inf))

    print(err)
    assert actx.to_numpy(err) < 1e-2

# }}}


# {{{ sanity checks: ball meshes

# python test_meshmode.py "test_sanity_balls(cl._csc, "disk-radius-1.step", 2, 2, visualize=True)"  # noqa: E501
@pytest.mark.parametrize(("src_file", "dim"), [
    ("disk-radius-1.step", 2),
    ("ball-radius-1.step", 3),
    ])
@pytest.mark.parametrize("mesh_order", [1, 2])
def test_sanity_balls(
            actx_factory: ArrayContextFactory,
            src_file,
            dim,
            mesh_order,
            visualize=False):
    from arraycontext import PyOpenCLArrayContext
    pytest.importorskip("pytential")

    logging.basicConfig(level=logging.INFO)
    actx = actx_factory()

    if not isinstance(actx, PyOpenCLArrayContext):
        pytest.skip(f"{actx}: not supported by pytential")

    from pytools.convergence import EOCRecorder
    vol_eoc_rec = EOCRecorder()
    surf_eoc_rec = EOCRecorder()

    # overkill
    quad_order = mesh_order

    from pytential import bind, sym

    for h in [0.2, 0.1, 0.05]:
        from meshmode.mesh.io import FileSource, generate_gmsh
        mesh = generate_gmsh(
                FileSource(src_file), dim, order=mesh_order,
                other_options=["-string", f"Mesh.CharacteristicLengthMax = {h};"],
                force_ambient_dim=dim,
                target_unit="MM")

        logger.info("%d elements", mesh.nelements)

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

        vol_x = actx.thaw(vol_discr.nodes())

        vol_one = vol_x[0]*0 + 1
        from pytential import integral, norm

        comp_vol = actx.to_numpy(integral(vol_discr, vol_one))
        rel_vol_err = abs(true_vol - comp_vol) / true_vol
        vol_eoc_rec.add_data_point(h, rel_vol_err)
        print("VOL", true_vol, comp_vol)

        bdry_x = actx.thaw(bdry_discr.nodes())

        bdry_one_exact = bdry_x[0] * 0 + 1

        bdry_one = bdry_connection(vol_one)
        intp_err = norm(bdry_discr, bdry_one-bdry_one_exact)
        assert intp_err < 1e-14

        comp_surf = actx.to_numpy(integral(bdry_discr, bdry_one))
        rel_surf_err = abs(true_surf - comp_surf) / true_surf
        surf_eoc_rec.add_data_point(h, rel_surf_err)
        print("SURF", true_surf, comp_surf)

        if visualize:
            from meshmode.discretization.visualization import make_visualizer
            vol_vis = make_visualizer(actx, vol_discr, 7)
            bdry_vis = make_visualizer(actx, bdry_discr, 7)

            name = src_file.split("-")[0]
            vol_vis.write_vtk_file(f"sanity_balls_volume_{name}_{h:g}.vtu", [
                ("f", vol_one),
                ("area_el", bind(
                    vol_discr,
                    sym.area_element(mesh.ambient_dim, mesh.ambient_dim))
                    (actx)),
                ])

            bdry_vis.write_vtk_file(f"sanity_balls_boundary_{name}_{h:g}.vtu", [
                ("f", bdry_one)
                ])

        # {{{ check normals point outward

        normal_outward_check = bind(bdry_discr,
                sym.normal(mesh.ambient_dim) | sym.nodes(mesh.ambient_dim),
                )(actx).as_scalar()

        normal_outward_check = actx.to_numpy(flatten(normal_outward_check > 0, actx))
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


# {{{ mesh without vertices

def test_mesh_without_vertices(actx_factory: ArrayContextFactory):
    actx = actx_factory()

    # create a mesh
    mesh = mgen.generate_sphere(r=1.0, order=4)

    # create one without the vertices
    from dataclasses import replace
    groups = [
        replace(grp, nodes=grp.nodes, vertex_indices=None)
        for grp in mesh.groups]
    mesh = make_mesh(None, groups, is_conforming=None)

    # try refining it
    from meshmode.mesh.refinement import refine_uniformly
    mesh = refine_uniformly(mesh, 1)

    # make sure the world doesn't end
    from meshmode.discretization import Discretization
    discr = Discretization(actx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(4))
    actx.thaw(discr.nodes())

    from meshmode.discretization.visualization import make_visualizer
    make_visualizer(actx, discr, 4)

# }}}


# {{{ test_mesh_multiple_groups

@pytest.mark.parametrize("ambient_dim", [1, 2, 3])
def test_mesh_multiple_groups(
            actx_factory: ArrayContextFactory,
            ambient_dim,
            visualize=False):
    actx = actx_factory()

    order = 4

    mesh = mgen.generate_regular_rect_mesh(
            a=(-0.5,)*ambient_dim, b=(0.5,)*ambient_dim,
            nelements_per_axis=(8,)*ambient_dim, order=order)
    assert len(mesh.groups) == 1

    from meshmode.mesh.processing import split_mesh_groups
    element_flags = np.any(
            mesh.vertices[0, mesh.groups[0].vertex_indices] < 0.0,
            axis=1).astype(np.int64)
    mesh = split_mesh_groups(mesh, element_flags)
    assert isinstance(mesh, Mesh)

    assert len(mesh.groups) == 2            # pylint: disable=no-member
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

    def grp_factory(mesh_el_group: MeshElementGroup):
        index = None

        for i, meg in enumerate(mesh.groups):  # pylint: disable=no-member
            if meg is mesh_el_group:
                index = i

        if mesh_el_group.dim == mesh.ambient_dim:
            assert index is not None

        return default_simplex_group_factory(
                base_dim=ambient_dim, order=order + 2 if index == 0 else order
                )(mesh_el_group)

    discr = Discretization(actx, mesh, grp_factory)

    if visualize:
        group_id = discr.empty(actx, dtype=np.int32)
        for igrp, vec in enumerate(group_id):
            vec.fill(igrp)

        from meshmode.discretization.visualization import make_visualizer
        vis = make_visualizer(actx, discr, vis_order=order)
        vis.write_vtk_file("mesh_multiple_groups.vtu", [
            ("group_id", group_id)
            ], overwrite=True)

    # check face restrictions
    from meshmode.discretization.connection import (
        check_connection,
        make_face_restriction,
        make_face_to_all_faces_embedding,
        make_opposite_face_connection,
    )
    for boundary_tag in [BTAG_ALL, FACE_RESTR_INTERIOR, FACE_RESTR_ALL]:
        conn = make_face_restriction(actx, discr,
                group_factory=grp_factory,
                boundary_tag=boundary_tag,
                per_face_groups=False)
        check_connection(actx, conn)

        bdry_f = conn.to_discr.zeros(actx) + 1

        if boundary_tag == FACE_RESTR_INTERIOR:
            opposite = make_opposite_face_connection(actx, conn)
            check_connection(actx, opposite)

            op_bdry_f = opposite(bdry_f)

            # Ensure test coverage for alternate modes in DirectConnection
            for force_loopy, force_no_merged_batches in [
                    (False, True),
                    (True, False),
                    (True, True),
                    ]:
                op_bdry_f_2 = opposite(bdry_f,
                        _force_use_loopy=force_loopy,
                        _force_no_merged_batches=force_no_merged_batches)
                error = flat_norm(op_bdry_f - op_bdry_f_2, np.inf)
                assert actx.to_numpy(error) < 1e-15

            error = flat_norm(bdry_f - op_bdry_f, np.inf)
            assert actx.to_numpy(error) < 1.0e-11, error

        if boundary_tag == FACE_RESTR_ALL:
            embedding = make_face_to_all_faces_embedding(actx, conn, conn.to_discr)
            check_connection(actx, embedding)

            em_bdry_f = embedding(bdry_f)
            error = flat_norm(bdry_f - em_bdry_f)
            assert actx.to_numpy(error) < 1.0e-11, error

    # check some derivatives (nb: flatten is a generator)
    import pytools
    ref_axes = pytools.flatten([[i] for i in range(ambient_dim)])

    from meshmode.discretization import num_reference_derivative
    x = actx.thaw(discr.nodes())
    num_reference_derivative(discr, ref_axes, x[0])

# }}}


@pytest.mark.parametrize("ambient_dim", [2, 3])
def test_mesh_with_interior_unit_nodes(actx_factory: ArrayContextFactory, ambient_dim):
    actx = actx_factory()

    # NOTE: smaller orders or coarser meshes make the cases fail the
    # node_vertex_consistency test; the default warp_and_blend_nodes have
    # nodes at the vertices, so they pass for much smaller tolerances

    order = 8
    nelements = 32
    n_minor = 2 * nelements
    uniform_refinement_rounds = 4

    import modepy as mp
    if ambient_dim == 2:
        unit_nodes = mp.LegendreGaussQuadrature(
                order, force_dim_axis=True).nodes

        mesh = mgen.make_curve_mesh(
                partial(mgen.ellipse, 2.0),
                np.linspace(0.0, 1.0, nelements + 1), order=order,
                unit_nodes=unit_nodes)
    elif ambient_dim == 3:
        unit_nodes = mp.VioreanuRokhlinSimplexQuadrature(order, 2).nodes

        mesh = mgen.generate_torus(4.0, 2.0,
                n_major=2*n_minor, n_minor=n_minor,
                order=order, unit_nodes=unit_nodes)

        mesh = mgen.generate_sphere(1.0,
                uniform_refinement_rounds=uniform_refinement_rounds,
                order=order, unit_nodes=unit_nodes)
    else:
        raise ValueError(f"unsupported dimension: '{ambient_dim}'")

    assert mesh.facial_adjacency_groups
    assert mesh.nodal_adjacency

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import QuadratureSimplexGroupFactory
    discr = Discretization(actx, mesh,
            QuadratureSimplexGroupFactory(order))

    from meshmode.discretization.connection import make_face_restriction
    conn = make_face_restriction(actx, discr,
            group_factory=QuadratureSimplexGroupFactory(order),
            boundary_tag=FACE_RESTR_ALL)
    assert conn


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
