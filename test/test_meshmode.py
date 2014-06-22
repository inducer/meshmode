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


def test_boundary_interpolation(ctx_getter):
    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    from meshmode.mesh.io import generate_gmsh, FileSource
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            QuadratureSimplexGroupFactory
    from pytools.convergence import EOCRecorder
    from meshmode.discretization.connection import make_boundary_restriction

    eoc_rec = EOCRecorder()

    order = 4

    for h in [1e-1, 3e-2, 1e-2]:
        print "BEGIN GEN"
        mesh = generate_gmsh(
                FileSource("blob-2d.step"), 2, order=order,
                force_ambient_dimension=2,
                other_options=[
                    "-string", "Mesh.CharacteristicLengthMax = %s;" % h]
                )
        print "END GEN"

        vol_discr = Discretization(cl_ctx, mesh,
                QuadratureSimplexGroupFactory(order))
        print "h=%s -> %d elements" % (
                h, sum(mgrp.nelements for mgrp in mesh.groups))

        x = vol_discr.nodes()[0].with_queue(queue)
        f = 0.1*cl.clmath.sin(30*x)

        bdry_mesh, bdry_discr, bdry_connection = make_boundary_restriction(
                queue, vol_discr, QuadratureSimplexGroupFactory(order))

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
            force_ambient_dimension=2,
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

    mesh = perform_flips(mesh, flippy)

    mesh_orient = find_volume_mesh_element_orientations(mesh)

    assert ((mesh_orient < 0) == (flippy > 0)).all()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
