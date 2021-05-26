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
from dataclasses import dataclass
import logging
logger = logging.getLogger(__name__)

import numpy as np
# import numpy.linalg as la
import pytest

from pytools.obj_array import make_obj_array

from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
from meshmode.discretization.poly_element import (
        PolynomialWarpAndBlendGroupFactory,
        InterpolatoryQuadratureSimplexGroupFactory,
        LegendreGaussLobattoTensorProductGroupFactory,
        )
import meshmode.mesh.generation as mgen

from arraycontext import thaw, _acf         # noqa: F401
from arraycontext import (                  # noqa: F401
        pytest_generate_tests_for_pyopencl_array_context
        as pytest_generate_tests)


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
        mesh = mgen.generate_warped_rect_mesh(dim, target_order, nelements_side=4)
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
    assert os.path.exists(pvtu_filename)

    import filecmp
    assert filecmp.cmp(f"ref-{pvtu_filename}", pvtu_filename)


@dataclass
class VisualizerData:
    g: np.ndarray


@pytest.mark.parametrize(("dim", "group_cls"), [
    (1, SimplexElementGroup),
    (2, SimplexElementGroup),
    (3, SimplexElementGroup),
    (2, TensorProductElementGroup),
    (3, TensorProductElementGroup),
    ])
def test_visualizers(actx_factory, dim, group_cls):
    actx = actx_factory()

    nelements = 64
    target_order = 4

    is_simplex = issubclass(group_cls, SimplexElementGroup)
    if dim == 1:
        mesh = mgen.make_curve_mesh(
                mgen.NArmedStarfish(5, 0.25),
                np.linspace(0.0, 1.0, nelements + 1),
                target_order)
    elif dim == 2:
        if is_simplex:
            mesh = mgen.generate_torus(5.0, 1.0, order=target_order)
        else:
            mesh = mgen.generate_regular_rect_mesh(
                    a=(0,)*dim, b=(1,)*dim, nelements_per_axis=(4,)*dim,
                    group_cls=group_cls,
                    order=target_order)
    elif dim == 3:
        if is_simplex:
            mesh = mgen.generate_warped_rect_mesh(dim, target_order,
                    nelements_side=4)
        else:
            mesh = mgen.generate_regular_rect_mesh(
                    a=(0,)*dim, b=(1,)*dim, nelements_per_axis=(4,)*dim,
                    group_cls=group_cls,
                    order=target_order)
    else:
        raise ValueError("unknown dimensionality")

    if is_simplex:
        group_factory = InterpolatoryQuadratureSimplexGroupFactory
    else:
        group_factory = LegendreGaussLobattoTensorProductGroupFactory

    from meshmode.discretization import Discretization
    discr = Discretization(actx, mesh, group_factory(target_order))

    nodes = thaw(discr.nodes(), actx)
    f = actx.np.sqrt(sum(nodes**2)) + 1j*nodes[0]
    g = VisualizerData(g=f)
    names_and_fields = [("f", f), ("g", g)]
    names_and_fields = [("f", f)]

    from meshmode.discretization.visualization import make_visualizer
    vis = make_visualizer(actx, discr, target_order)

    eltype = "simplex" if is_simplex else "box"
    basename = f"visualizer_vtk_{eltype}_{dim}d"
    vis.write_vtk_file(f"{basename}_linear.vtu", names_and_fields, overwrite=True)

    with pytest.raises(RuntimeError):
        vis.write_vtk_file(f"{basename}_lagrange.vtu",
                names_and_fields, overwrite=True, use_high_order=True)

    try:
        basename = f"visualizer_xdmf_{eltype}_{dim}d"
        vis.write_xdmf_file(f"{basename}.xmf", names_and_fields, overwrite=True)
    except ImportError:
        logger.info("h5py not available")

    if mesh.dim == 2 and is_simplex:
        try:
            vis.show_scalar_in_matplotlib_3d(f, do_show=False)
        except ImportError:
            logger.info("matplotlib not available")

    if mesh.dim <= 2 and is_simplex:
        try:
            vis.show_scalar_in_mayavi(f, do_show=False)
        except ImportError:
            logger.info("mayavi not available")

    vis = make_visualizer(actx, discr, target_order,
            force_equidistant=True)

    basename = f"visualizer_vtk_{eltype}_{dim}d"
    vis.write_vtk_file(f"{basename}_lagrange.vtu",
            names_and_fields, overwrite=True, use_high_order=True)


@pytest.mark.parametrize("ambient_dim", [2, 3])
def test_copy_visualizer(actx_factory, ambient_dim, visualize=True):
    actx = actx_factory()
    target_order = 4

    if ambient_dim == 2:
        nelements = 128
        mesh = mgen.make_curve_mesh(
                partial(mgen.ellipse, 1.0),
                np.linspace(0.0, 1.0, nelements + 1),
                target_order)
    elif ambient_dim == 3:
        mesh = mgen.generate_icosphere(1.0, target_order,
                uniform_refinement_rounds=2)
    else:
        raise ValueError(f"unsupported dimension: {ambient_dim}")

    from meshmode.mesh.processing import affine_map
    translated_mesh = affine_map(mesh,
            b=np.array([2.5, 0.0, 0.0][:ambient_dim])
            )

    from meshmode.discretization import Discretization
    discr = Discretization(actx, mesh,
            PolynomialWarpAndBlendGroupFactory(target_order))
    translated_discr = Discretization(actx, translated_mesh,
            PolynomialWarpAndBlendGroupFactory(target_order))

    from meshmode.discretization.visualization import make_visualizer
    vis = make_visualizer(actx, discr, target_order, force_equidistant=True)
    assert vis._vtk_connectivity
    assert vis._vtk_lagrange_connectivity

    translated_vis = vis.copy_with_same_connectivity(actx, translated_discr)
    assert translated_vis._cached_vtk_connectivity is not None
    assert translated_vis._cached_vtk_lagrange_connectivity is not None

    assert translated_vis._vtk_connectivity \
            is vis._vtk_connectivity
    assert translated_vis._vtk_lagrange_connectivity \
            is vis._vtk_lagrange_connectivity

    if not visualize:
        return

    vis.write_vtk_file(
            f"visualizer_copy_{ambient_dim}d_orig.vtu", [], overwrite=True)
    translated_vis.write_vtk_file(
            f"visualizer_copy_{ambient_dim}d_translated.vtu", [], overwrite=True)

# }}}


# {{{ test_vtk_overwrite

def test_vtk_overwrite(actx_factory):
    pytest.importorskip("pyvisfile")

    def _try_write_vtk(writer, obj):
        import os

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

    mesh = mgen.generate_torus(10.0, 2.0, order=target_order)

    from meshmode.discretization import Discretization
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


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
