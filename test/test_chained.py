__copyright__ = "Copyright (C) 2018 Alexandru Fikl"

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

import pytest
from meshmode.array_context import (  # noqa
        pytest_generate_tests_for_pyopencl_array_context
        as pytest_generate_tests)

from meshmode.dof_array import thaw, flatten

import logging
logger = logging.getLogger(__name__)


def create_discretization(actx, ndim,
                          nelements=42,
                          mesh_name=None,
                          order=4):
    # construct mesh
    if ndim == 2:
        from functools import partial
        from meshmode.mesh.generation import make_curve_mesh, ellipse, starfish

        if mesh_name is None:
            mesh_name = "ellipse"

        t = np.linspace(0.0, 1.0, nelements + 1)
        if mesh_name == "ellipse":
            mesh = make_curve_mesh(partial(ellipse, 2), t, order=order)
        elif mesh_name == "starfish":
            mesh = make_curve_mesh(starfish, t, order=order)
        else:
            raise ValueError(f"unknown mesh name: {mesh_name}")
    elif ndim == 3:
        from meshmode.mesh.generation import generate_torus
        from meshmode.mesh.generation import generate_warped_rect_mesh

        if mesh_name is None:
            mesh_name = "torus"

        if mesh_name == "torus":
            mesh = generate_torus(10.0, 5.0, order=order,
                    n_minor=nelements, n_major=nelements)
        elif mesh_name == "warp":
            mesh = generate_warped_rect_mesh(ndim, order=order, n=nelements)
        else:
            raise ValueError(f"unknown mesh name: {mesh_name}")
    else:
        raise ValueError(f"unsupported dimension: {ndim}")

    # create discretization
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
    discr = Discretization(actx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(order))

    return discr


def create_refined_connection(actx, discr, threshold=0.3):
    from meshmode.mesh.refinement import RefinerWithoutAdjacency
    from meshmode.discretization.connection import make_refinement_connection
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory

    flags = np.random.rand(discr.mesh.nelements) < threshold
    refiner = RefinerWithoutAdjacency(discr.mesh)
    refiner.refine(flags)

    discr_order = discr.groups[0].order
    connection = make_refinement_connection(actx, refiner, discr,
            InterpolatoryQuadratureSimplexGroupFactory(discr_order))

    return connection


def create_face_connection(actx, discr):
    from meshmode.discretization.connection import FACE_RESTR_ALL
    from meshmode.discretization.connection import make_face_restriction
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory

    discr_order = discr.groups[0].order
    connection = make_face_restriction(actx, discr,
            InterpolatoryQuadratureSimplexGroupFactory(discr_order),
            FACE_RESTR_ALL,
            per_face_groups=True)

    return connection


@pytest.mark.skip(reason="implementation detail")
@pytest.mark.parametrize("ndim", [2, 3])
def test_chained_batch_table(actx_factory, ndim, visualize=False):
    from meshmode.discretization.connection.chained import \
        _build_element_lookup_table

    actx = actx_factory()

    discr = create_discretization(actx, ndim)
    connections = []
    conn = create_refined_connection(actx, discr)
    connections.append(conn)
    conn = create_refined_connection(actx, conn.to_discr)
    connections.append(conn)

    from meshmode.discretization.connection import ChainedDiscretizationConnection
    chained = ChainedDiscretizationConnection(connections)

    conn = chained.connections[0]
    el_table = _build_element_lookup_table(actx, conn)
    for igrp, grp in enumerate(conn.groups):
        for ibatch, batch in enumerate(grp.batches):
            ifrom = batch.from_element_indices.get(actx.queue)
            jfrom = el_table[igrp][batch.to_element_indices.get(actx.queue)]

            assert np.all(ifrom == jfrom)
        assert np.min(el_table[igrp]) >= 0


@pytest.mark.skip(reason="implementation detail")
@pytest.mark.parametrize("ndim", [2, 3])
def test_chained_new_group_table(actx_factory, ndim, visualize=False):
    from meshmode.discretization.connection.chained import \
        _build_new_group_table

    actx = actx_factory()

    discr = create_discretization(actx, ndim,
                                  nelements=8,
                                  order=2)
    connections = []
    conn = create_refined_connection(actx, discr)
    connections.append(conn)
    conn = create_refined_connection(actx, conn.to_discr)
    connections.append(conn)

    grp_to_grp, grp_info = _build_new_group_table(connections[0],
                                                  connections[1])
    if visualize:
        import matplotlib.pyplot as pt

        pt.figure(figsize=(10, 8))
        for k, v in grp_to_grp.items():
            print(k)
            print(v)

            igrp, ibatch, jgrp, jbatch = k
            mgroup, mbatch = v
            from_group_index = connections[0].groups[igrp] \
                    .batches[ibatch].from_group_index

            from_unit_nodes = connections[0].from_discr \
                    .groups[from_group_index].unit_nodes
            to_unit_nodes = grp_info[mgroup][mbatch].result_unit_nodes

            if ndim == 2:
                pt.plot(from_unit_nodes, "o")
                pt.plot(to_unit_nodes, "^")
            else:
                pt.plot(from_unit_nodes[0], from_unit_nodes[1], "o")
                pt.plot(to_unit_nodes[0], to_unit_nodes[1], "^")

            pt.savefig("test_grp_to_grp_{}d_{:05d}_{:05d}.png"
                        .format(ndim, mgroup, mbatch), dpi=300)
            pt.clf()


@pytest.mark.parametrize("ndim", [2, 3])
def test_chained_connection(actx_factory, ndim, visualize=False):
    actx = actx_factory()

    discr = create_discretization(actx, ndim, nelements=10)
    connections = []
    conn = create_refined_connection(actx, discr, threshold=np.inf)
    connections.append(conn)
    conn = create_refined_connection(actx, conn.to_discr, threshold=np.inf)
    connections.append(conn)

    from meshmode.discretization.connection import \
            ChainedDiscretizationConnection
    chained = ChainedDiscretizationConnection(connections)

    def f(x):
        from functools import reduce
        return 0.1 * reduce(lambda x, y: x * actx.np.sin(5 * y), x)

    x = thaw(actx, connections[0].from_discr.nodes())
    fx = f(x)
    f1 = chained(fx)
    f2 = connections[1](connections[0](fx))

    assert actx.np.linalg.norm(f1-f2, np.inf) / actx.np.linalg.norm(f2) < 1e-11


@pytest.mark.parametrize("ndim", [2, 3])
def test_chained_full_resample_matrix(actx_factory, ndim, visualize=False):
    from meshmode.discretization.connection.chained import \
        make_full_resample_matrix

    actx = actx_factory()

    discr = create_discretization(actx, ndim, order=2, nelements=12)
    connections = []
    conn = create_refined_connection(actx, discr)
    connections.append(conn)
    conn = create_refined_connection(actx, conn.to_discr)
    connections.append(conn)

    from meshmode.discretization.connection import \
            ChainedDiscretizationConnection
    chained = ChainedDiscretizationConnection(connections)

    def f(x):
        from functools import reduce
        return 0.1 * reduce(lambda x, y: x * actx.np.sin(5 * y), x)

    resample_mat = actx.to_numpy(make_full_resample_matrix(actx, chained))

    x = thaw(actx, connections[0].from_discr.nodes())
    fx = f(x)
    f1 = resample_mat @ actx.to_numpy(flatten(fx))
    f2 = actx.to_numpy(flatten(chained(fx)))
    f3 = actx.to_numpy(flatten(connections[1](connections[0](fx))))

    assert np.allclose(f1, f2)
    assert np.allclose(f2, f3)


@pytest.mark.parametrize(("ndim", "chain_type"), [
    (2, 1), (2, 2), (3, 1), (3, 3)])
def test_chained_to_direct(actx_factory, ndim, chain_type,
                           nelements=128, visualize=False):
    import time
    from meshmode.discretization.connection.chained import \
        flatten_chained_connection

    actx = actx_factory()

    discr = create_discretization(actx, ndim, nelements=nelements)
    connections = []
    if chain_type == 1:
        conn = create_refined_connection(actx, discr)
        connections.append(conn)
        conn = create_refined_connection(actx, conn.to_discr)
        connections.append(conn)
    elif chain_type == 2:
        conn = create_refined_connection(actx, discr)
        connections.append(conn)
        conn = create_refined_connection(actx, conn.to_discr)
        connections.append(conn)
        conn = create_refined_connection(actx, conn.to_discr)
        connections.append(conn)
    elif chain_type == 3 and ndim == 3:
        conn = create_refined_connection(actx, discr, threshold=np.inf)
        connections.append(conn)
        conn = create_face_connection(actx, conn.to_discr)
        connections.append(conn)
    else:
        raise ValueError("unknown test case")

    from meshmode.discretization.connection import \
            ChainedDiscretizationConnection
    chained = ChainedDiscretizationConnection(connections)

    t_start = time.time()
    direct = flatten_chained_connection(actx, chained)
    t_end = time.time()
    if visualize:
        print("[TIME] Flatten: {:.5e}".format(t_end - t_start))

    if chain_type < 3:
        to_element_indices = np.full(direct.to_discr.mesh.nelements, 0,
                                     dtype=np.int)
        for grp in direct.groups:
            for batch in grp.batches:
                for i in batch.to_element_indices.get(actx.queue):
                    to_element_indices[i] += 1
        assert np.min(to_element_indices) > 0

    def f(x):
        from functools import reduce
        return 0.1 * reduce(lambda x, y: x * actx.np.sin(5 * y), x)

    x = thaw(actx, connections[0].from_discr.nodes())
    fx = f(x)

    t_start = time.time()
    f1 = actx.to_numpy(flatten(direct(fx)))
    t_end = time.time()
    if visualize:
        print("[TIME] Direct: {:.5e}".format(t_end - t_start))

    t_start = time.time()
    f2 = actx.to_numpy(flatten(chained(fx)))
    t_end = time.time()
    if visualize:
        print("[TIME] Chained: {:.5e}".format(t_end - t_start))

    if visualize and ndim == 2:
        import matplotlib.pyplot as pt

        pt.figure(figsize=(10, 8), dpi=300)
        pt.plot(f1, label="Direct")
        pt.plot(f2, label="Chained")
        pt.ylim([np.min(f2) - 0.1, np.max(f2) + 0.1])
        pt.legend()
        pt.savefig("test_chained_to_direct.png")
        pt.clf()

    assert np.allclose(f1, f2)


@pytest.mark.parametrize(("ndim", "mesh_name"), [
    (2, "starfish"),
    (3, "torus")])
def test_reversed_chained_connection(actx_factory, ndim, mesh_name):
    actx = actx_factory()

    def run(nelements, order):
        discr = create_discretization(actx, ndim,
            nelements=nelements,
            order=order,
            mesh_name=mesh_name)

        threshold = 1.0
        connections = []
        conn = create_refined_connection(actx,
                discr, threshold=threshold)
        connections.append(conn)
        if ndim == 2:
            # NOTE: additional refinement makes the 3D meshes explode in size
            conn = create_refined_connection(actx,
                    conn.to_discr, threshold=threshold)
            connections.append(conn)
            conn = create_refined_connection(actx,
                    conn.to_discr, threshold=threshold)
            connections.append(conn)

        from meshmode.discretization.connection import \
                ChainedDiscretizationConnection
        chained = ChainedDiscretizationConnection(connections)
        from meshmode.discretization.connection import \
                L2ProjectionInverseDiscretizationConnection
        reverse = L2ProjectionInverseDiscretizationConnection(chained)

        # create test vector
        from_nodes = thaw(actx, chained.from_discr.nodes())
        to_nodes = thaw(actx, chained.to_discr.nodes())

        from_x = 0
        to_x = 0
        for d in range(ndim):
            from_x += actx.np.cos(from_nodes[d]) ** (d + 1)
            to_x += actx.np.cos(to_nodes[d]) ** (d + 1)

        from_interp = reverse(to_x)

        return (1.0 / nelements,
                actx.np.linalg.norm(from_interp - from_x, np.inf)
                / actx.np.linalg.norm(from_x, np.inf))

    from pytools.convergence import EOCRecorder
    eoc = EOCRecorder()

    order = 4
    mesh_sizes = [16, 32, 48, 64, 96, 128]

    for n in mesh_sizes:
        h, error = run(n, order)
        eoc.add_data_point(h, error)

    print(eoc)

    assert eoc.order_estimate() > (order + 1 - 0.5)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
