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
import os

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as pt

import pyopencl as cl
import pyopencl.array  # noqa
import pyopencl.clmath  # noqa

import pytest
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)

import logging
logger = logging.getLogger(__name__)

def create_chained_connection(queue, ndim,
                              nelements=42,
                              mesh_order=2,
                              discr_order=2,
                              visualize=False):
    ctx = queue.context
    connections = [None, None]

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

    # connection 1: refine
    from meshmode.mesh.refinement import RefinerWithoutAdjacency
    from meshmode.discretization.connection import make_refinement_connection
    refiner = RefinerWithoutAdjacency(mesh)

    threshold = 0.2 if ndim == 3 else 0.5
    flags = np.random.rand(refiner.get_current_mesh().nelements) < threshold
    refiner.refine(flags)
    connections[0] = make_refinement_connection(refiner, discr,
            InterpolatoryQuadratureSimplexGroupFactory(discr_order))

    if visualize and ndim == 2:
        from_nodes = connections[0].from_discr.nodes().get(queue)
        to_nodes = connections[0].to_discr.nodes().get(queue)

        pt.plot(to_nodes[0], to_nodes[1], 'k')
        pt.plot(from_nodes[0], from_nodes[1], 'o', mfc='none')
        pt.plot(to_nodes[0], to_nodes[1], '.')
        filename = os.path.join(os.path.dirname(__file__),
            "test_chained_0_{}d.png".format(ndim))
        pt.tight_layout()
        pt.savefig(filename, dpi=300)
        pt.clf()

    # connection 2: refine / restrict to face
    flags = np.random.rand(refiner.get_current_mesh().nelements) < threshold
    refiner.refine(flags)
    connections[1] = make_refinement_connection(refiner,
            connections[0].to_discr,
            InterpolatoryQuadratureSimplexGroupFactory(discr_order))

    if visualize and ndim == 2:
        from_nodes = connections[1].from_discr.nodes().get(queue)
        to_nodes = connections[1].to_discr.nodes().get(queue)

        pt.plot(to_nodes[0], to_nodes[1], 'k')
        pt.plot(from_nodes[0], from_nodes[1], 'o', mfc='none')
        pt.plot(to_nodes[0], to_nodes[1], '.')
        filename = os.path.join(os.path.dirname(__file__),
            "test_chained_1_{}d.png".format(ndim))
        pt.tight_layout()
        pt.savefig(filename, dpi=300)
        pt.clf()

    from meshmode.discretization.connection import ChainedDiscretizationConnection
    chained = ChainedDiscretizationConnection(connections)

    return chained


@pytest.mark.parametrize("ndim", [2, 3])
def test_chained_full_resample_matrix(ctx_factory, ndim, visualize=False):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    chained = create_chained_connection(queue, ndim, visualize=visualize)
    connections = chained.connections

    x = connections[0].from_discr.nodes().with_queue(queue)
    def f(x):
        from six.moves import reduce
        return 0.1 * reduce(lambda x, y: x * cl.clmath.sin(5 * y), x)

    fx = f(x)
    resample_mat = chained.full_resample_matrix(queue).get(queue)
    f1 = np.dot(resample_mat, fx.get(queue))
    f2 = chained(queue, fx).get(queue)
    f3 = connections[1](queue, connections[0](queue, fx)).get(queue)

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
