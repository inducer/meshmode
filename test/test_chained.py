from __future__ import division, absolute_import, print_function

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

import pyopencl as cl
import pyopencl.array  # noqa
import pyopencl.clmath  # noqa

import pytest
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)

import logging
logger = logging.getLogger(__name__)


def create_discretization(queue, ndim,
                          nelements=42,
                          mesh_order=5,
                          discr_order=5):
    ctx = queue.context

    # construct base mesh
    if ndim == 2:
        from functools import partial
        from meshmode.mesh.generation import make_curve_mesh, ellipse
        mesh = make_curve_mesh(partial(ellipse, 2),
                               np.linspace(0.0, 1.0, nelements + 1),
                               order=mesh_order)
    elif ndim == 3:
        from meshmode.mesh.generation import generate_torus
        mesh = generate_torus(10.0, 5.0, order=mesh_order)
    else:
        raise ValueError("Unsupported dimension: {}".format(ndim))

    # create base discretization
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
    discr = Discretization(ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(discr_order))

    return discr


def create_refined_connection(queue, discr, threshold=0.3):
    from meshmode.mesh.refinement import RefinerWithoutAdjacency
    from meshmode.discretization.connection import make_refinement_connection
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory

    flags = np.random.rand(discr.mesh.nelements) < threshold
    refiner = RefinerWithoutAdjacency(discr.mesh)
    refiner.refine(flags)

    discr_order = discr.groups[0].order
    connection = make_refinement_connection(refiner, discr,
            InterpolatoryQuadratureSimplexGroupFactory(discr_order))

    return connection


def create_face_connection(queue, discr):
    from meshmode.discretization.connection import FACE_RESTR_ALL
    from meshmode.discretization.connection import make_face_restriction
    from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory

    discr_order = discr.groups[0].order
    connection = make_face_restriction(discr,
            InterpolatoryQuadratureSimplexGroupFactory(discr_order),
            FACE_RESTR_ALL,
            per_face_groups=True)

    return connection


@pytest.mark.parametrize("ndim", [2, 3])
def test_chained_batch_table(ctx_factory, ndim, visualize=False):
    from meshmode.discretization.connection.chained import \
        _build_element_lookup_table

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    discr = create_discretization(queue, ndim)
    connections = []
    conn = create_refined_connection(queue, discr)
    connections.append(conn)
    conn = create_refined_connection(queue, conn.to_discr)
    connections.append(conn)

    from meshmode.discretization.connection import ChainedDiscretizationConnection
    chained = ChainedDiscretizationConnection(connections)

    conn = chained.connections[1]
    el_to_batch = _build_element_lookup_table(queue, conn)
    for igrp, grp in enumerate(conn.groups):
        for ibatch, batch in enumerate(grp.batches):
            for i, k in enumerate(batch.from_element_indices.get(queue)):
                assert (i, igrp, ibatch) in el_to_batch[k]

    for k, batches in enumerate(el_to_batch):
        for i, igrp, ibatch in batches:
            batch = conn.groups[igrp].batches[ibatch]
            assert k == batch.from_element_indices[i]


@pytest.mark.parametrize("ndim", [2, 3])
def test_chained_new_group_table(ctx_factory, ndim, visualize=False):
    from meshmode.discretization.connection.chained import \
        _build_new_group_table

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    discr = create_discretization(queue, ndim,
                                  nelements=8,
                                  mesh_order=2,
                                  discr_order=2)
    connections = []
    conn = create_refined_connection(queue, discr)
    connections.append(conn)
    conn = create_refined_connection(queue, conn.to_discr)
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
                pt.plot(from_unit_nodes, 'o')
                pt.plot(to_unit_nodes, '^')
            else:
                pt.plot(from_unit_nodes[0], from_unit_nodes[1], 'o')
                pt.plot(to_unit_nodes[0], to_unit_nodes[1], '^')

            pt.savefig('test_grp_to_grp_{}d_{:05d}_{:05d}.png'
                        .format(ndim, mgroup, mbatch), dpi=300)
            pt.clf()


@pytest.mark.parametrize("ndim", [2, 3])
def test_chained_full_resample_matrix(ctx_factory, ndim, visualize=False):
    from meshmode.discretization.connection.chained import \
        make_full_resample_matrix

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    discr = create_discretization(queue, ndim,
                                  mesh_order=2,
                                  discr_order=2)
    connections = []
    conn = create_refined_connection(queue, discr)
    connections.append(conn)
    conn = create_refined_connection(queue, conn.to_discr)
    connections.append(conn)

    from meshmode.discretization.connection import ChainedDiscretizationConnection
    chained = ChainedDiscretizationConnection(connections)

    def f(x):
        from six.moves import reduce
        return 0.1 * reduce(lambda x, y: x * cl.clmath.sin(5 * y), x)

    resample_mat = make_full_resample_matrix(queue, chained).get(queue)

    x = connections[0].from_discr.nodes().with_queue(queue)
    fx = f(x)
    f1 = np.dot(resample_mat, fx.get(queue))
    f2 = chained(queue, fx).get(queue)
    f3 = connections[1](queue, connections[0](queue, fx)).get(queue)

    assert np.allclose(f1, f2)
    assert np.allclose(f2, f3)


@pytest.mark.parametrize("ndim", [2, 3])
def test_chained_to_direct(ctx_factory, ndim, chain_type=1, visualize=False):
    import time
    from meshmode.discretization.connection.chained import \
        flatten_chained_connection

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    discr = create_discretization(queue, ndim)
    connections = []
    if chain_type == 1:
        conn = create_refined_connection(queue, discr)
        connections.append(conn)
        conn = create_refined_connection(queue, conn.to_discr)
        connections.append(conn)
    elif chain_type == 2:
        conn = create_refined_connection(queue, discr)
        connections.append(conn)
        conn = create_refined_connection(queue, conn.to_discr)
        connections.append(conn)
        conn = create_refined_connection(queue, conn.to_discr)
        connections.append(conn)
    elif chain_type == 3 and ndim == 3:
        conn = create_refined_connection(queue, discr, threshold=np.inf)
        connections.append(conn)
        conn = create_face_connection(queue, conn.to_discr)
        connections.append(conn)
    else:
        raise ValueError('unknown test case')

    from meshmode.discretization.connection import ChainedDiscretizationConnection
    chained = ChainedDiscretizationConnection(connections)
    direct = flatten_chained_connection(queue, chained)

    if chain_type < 3:
        to_element_indices = np.full(direct.to_discr.mesh.nelements, 0,
                                     dtype=np.int)
        for grp in direct.groups:
            for batch in grp.batches:
                for i in batch.to_element_indices.get(queue):
                    to_element_indices[i] += 1
        assert np.min(to_element_indices) > 0

    def f(x):
        from six.moves import reduce
        return 0.1 * reduce(lambda x, y: x * cl.clmath.sin(5 * y), x)

    x = connections[0].from_discr.nodes().with_queue(queue)
    fx = f(x)

    t_start = time.time()
    f1 = direct(queue, fx).get(queue)
    t_end = time.time()
    if visualize:
        print('[TIME] Direct: {:.5e}'.format(t_end - t_start))

    t_start = time.time()
    f2 = chained(queue, fx).get(queue)
    t_end = time.time()
    if visualize:
        print('[TIME] Chained: {:.5e}'.format(t_end - t_start))

    t_start = time.time()
    f3 = connections[1](queue, connections[0](queue, fx)).get(queue)
    t_end = time.time()
    if visualize:
        print('[TIME] Chained: {:.5e}'.format(t_end - t_start))

    if visualize and ndim == 2:
        import matplotlib.pyplot as pt

        pt.figure(figsize=(10, 8))
        pt.plot(f1, label='Direct')
        pt.plot(f2, label='Chained')
        pt.ylim([np.min(f2) - 0.1, np.max(f2) + 0.1])
        pt.legend()
        pt.savefig('test_chained_to_direct.png', dpi=300)
        pt.clf()

    assert np.allclose(f1, f2)
    assert np.allclose(f2, f3)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
