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
from meshmode.discretization.poly_element import PolynomialWarpAndBlendGroupFactory
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


def test_dof_array_arithmetic_same_as_numpy(actx_factory):
    actx = actx_factory()

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(-0.5,)*2, b=(0.5,)*2, n=(3,)*2, order=1)

    discr = Discretization(actx, mesh, PolynomialWarpAndBlendGroupFactory(3))

    import operator
    from pytools import generate_nonnegative_integer_tuples_below as gnitb
    from random import uniform, randrange
    for op_func, n_args, use_integers in [
            (operator.add, 2, False),
            (operator.sub, 2, False),
            (operator.mul, 2, False),
            (operator.truediv, 2, False),
            (operator.pow, 2, False),
            # FIXME pyopencl.Array doesn't do mod.
            #(operator.mod, 2, True),
            #(operator.mod, 2, False),
            # FIXME: Two outputs
            #(divmod, 2, False),

            (operator.and_, 2, True),
            (operator.xor, 2, True),
            (operator.or_, 2, True),

            (operator.le, 2, False),
            (operator.ge, 2, False),
            (operator.lt, 2, False),
            (operator.gt, 2, False),
            (operator.eq, 2, True),
            (operator.ne, 2, True),

            (operator.pos, 1, False),
            (operator.neg, 1, False),
            (operator.abs, 1, False),

            (lambda ary: ary.real, 1, False),
            (lambda ary: ary.imag, 1, False),
            ]:
        for is_array_flags in gnitb(2, n_args):
            if sum(is_array_flags) == 0:
                # all scalars, no need to test
                continue
            args = [
                    (0.5+np.random.rand(discr.ndofs)
                        if not use_integers else
                        np.random.randint(3, 200, discr.ndofs))

                    if is_array_flag else
                    (uniform(0.5, 2)
                        if not use_integers
                        else randrange(3, 200))
                    for is_array_flag in is_array_flags]
            ref_result = op_func(*args)

            actx_args = [
                    unflatten(actx, discr, actx.from_numpy(arg))
                    if isinstance(arg, np.ndarray) else arg
                    for arg in args]

            actx_result = actx.to_numpy(flatten(op_func(*actx_args)))

            assert np.allclose(actx_result, ref_result)


def test_dof_array_comparison(actx_factory):
    actx = actx_factory()

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(-0.5,)*2, b=(0.5,)*2, n=(8,)*2, order=3)

    discr = Discretization(actx, mesh, PolynomialWarpAndBlendGroupFactory(3))

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
