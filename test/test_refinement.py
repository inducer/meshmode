from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2014-6 Shivam Gupta, Andreas Kloeckner"

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

import pytest

import numpy as np
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)
from meshmode.mesh.generation import (  # noqa
        generate_icosahedron, generate_box_mesh)
from meshmode.mesh.refinement.utils import check_nodal_adj_against_geometry
from meshmode.mesh.refinement import Refiner

import logging
logger = logging.getLogger(__name__)

from functools import partial


def gen_blob_mesh():
    from meshmode.mesh.io import generate_gmsh, FileSource
    return generate_gmsh(
            FileSource("blob-2d.step"), 2, order=1,
            force_ambient_dim=2,
            other_options=[
                "-string", "Mesh.CharacteristicLengthMax = %s;" % 0.2]
            )


def random_refine_flags(fract, mesh):
    all_els = list(range(mesh.nelements))

    flags = np.zeros(mesh.nelements)
    from random import shuffle
    shuffle(all_els)
    for i in range(int(mesh.nelements * fract)):
        flags[all_els[i]] = 1

    return flags


def uniform_refine_flags(mesh):
    return np.ones(mesh.nelements)


@pytest.mark.parametrize(("case_name", "mesh_gen", "flag_gen", "num_generations"), [
    # Fails?
    # ("icosahedron",
    #     partial(generate_icosahedron, 1, order=1),
    #     partial(random_refine_flags, 0.4),
    #     3),

    ("rect2d_rand",
        partial(generate_box_mesh, (
            np.linspace(0, 1, 3),
            np.linspace(0, 1, 3),
            ), order=1),
        partial(random_refine_flags, 0.4),
        4),

    ("rect2d_unif",
        partial(generate_box_mesh, (
            np.linspace(0, 1, 2),
            np.linspace(0, 1, 2),
            ), order=1),
        uniform_refine_flags,
        3),

    ("blob2d_rand",
        gen_blob_mesh,
        partial(random_refine_flags, 0.4),
        4),

    ("rect3d_rand",
        partial(generate_box_mesh, (
            np.linspace(0, 1, 2),
            np.linspace(0, 1, 3),
            np.linspace(0, 1, 2),
            ), order=1),
        partial(random_refine_flags, 0.4),
        3),

    ("rect3d_unif",
        partial(generate_box_mesh, (
            np.linspace(0, 1, 2),
            np.linspace(0, 1, 2)), order=1),
        uniform_refine_flags,
        3),
    ])
def test_refinement(case_name, mesh_gen, flag_gen, num_generations):
    from random import seed
    seed(13)

    mesh = mesh_gen()

    r = Refiner(mesh)

    for igen in range(num_generations):
        flags = flag_gen(mesh)
        mesh = r.refine(flags)

        check_nodal_adj_against_geometry(mesh)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
