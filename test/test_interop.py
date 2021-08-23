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
import pytest

from arraycontext import thaw

from meshmode.array_context import PytestPyOpenCLArrayContextFactory
from arraycontext import pytest_generate_tests_for_array_contexts
pytest_generate_tests = pytest_generate_tests_for_array_contexts(
        [PytestPyOpenCLArrayContextFactory])
from meshmode.dof_array import flat_norm

import logging
logger = logging.getLogger(__name__)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.octave
def test_nodal_dg_interop(actx_factory, dim):
    pytest.importorskip("oct2py")
    actx = actx_factory()

    from meshmode.interop.nodal_dg import download_nodal_dg_if_not_present
    download_nodal_dg_if_not_present()
    order = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(-0.5,)*dim, b=(0.5,)*dim, nelements_per_axis=(8,)*dim, order=order)

    from meshmode.interop.nodal_dg import NodalDGContext
    with NodalDGContext("./nodal-dg/Codes1.1") as ndgctx:
        ndgctx.set_mesh(mesh, order=order)

        discr = ndgctx.get_discr(actx)

        for ax in range(dim):
            x_ax = ndgctx.pull_dof_array(actx, ndgctx.AXES[ax])
            err = flat_norm(x_ax-thaw(discr.nodes()[ax], actx), np.inf)
            assert err < 1e-15

        n0 = thaw(discr.nodes()[0], actx)

        ndgctx.push_dof_array("n0", n0)
        n0_2 = ndgctx.pull_dof_array(actx, "n0")

        assert flat_norm(n0 - n0_2, np.inf) < 1e-15


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
