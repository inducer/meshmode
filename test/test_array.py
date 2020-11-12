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

import numpy as np

import meshmode         # noqa: F401
from meshmode.array_context import (  # noqa
        pytest_generate_tests_for_pyopencl_array_context
        as pytest_generate_tests)

from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import (
        PolynomialWarpAndBlendGroupFactory,
        )
from meshmode.dof_array import flatten, unflatten

import logging
logger = logging.getLogger(__name__)


def test_array_context_np_workalike(actx_factory):
    actx = actx_factory()

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(-0.5,)*2, b=(0.5,)*2, n=(8,)*2, order=3)

    discr = Discretization(actx, mesh, PolynomialWarpAndBlendGroupFactory(3))

    for sym_name, n_args in [
            ("sin", 1),
            ("exp", 1),
            ("arctan2", 2),
            ("minimum", 2),
            ("maximum", 2),
            ("where", 3),
            ("conj", 1),
            ]:
        args = [np.random.randn(discr.ndofs) for i in range(n_args)]
        ref_result = getattr(np, sym_name)(*args)

        actx_args = [unflatten(actx, discr, actx.from_numpy(arg)) for arg in args]

        actx_result = actx.to_numpy(
                flatten(getattr(actx.np, sym_name)(*actx_args)))

        assert np.allclose(actx_result, ref_result)


def test_dof_array_comparison(actx_factory):
    actx = actx_factory()

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(-0.5,)*2, b=(0.5,)*2, n=(8,)*2, order=3)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            PolynomialWarpAndBlendGroupFactory as GroupFactory
    discr = Discretization(actx, mesh, GroupFactory(3))

    import operator
    for op in [
            operator.lt,
            operator.le,
            operator.gt,
            operator.ge,
            ]:
        np_arg = np.random.randn(discr.ndofs)
        arg = unflatten(actx, discr, actx.from_numpy(np_arg))
        zeros = discr.zeros(actx)

        comp = op(arg, zeros)
        np_comp = actx.to_numpy(flatten(comp))
        assert np.array_equal(np_comp, op(np_arg, 0))

        comp = op(arg, 0)
        np_comp = actx.to_numpy(flatten(comp))
        assert np.array_equal(np_comp, op(np_arg, 0))

        comp = op(0, arg)
        np_comp = actx.to_numpy(flatten(comp))
        assert np.array_equal(np_comp, op(0, np_arg))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
