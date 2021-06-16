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

from arraycontext import _acf  # noqa: F401
from functools import partial
import numpy as np  # noqa: F401
import numpy.linalg as la  # noqa: F401

import meshmode         # noqa: F401
from arraycontext import (  # noqa
        pytest_generate_tests_for_pyopencl_array_context
        as pytest_generate_tests)

from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
from meshmode.discretization.poly_element import (
        PolynomialWarpAndBlend2DRestrictingGroupFactory,
        PolynomialWarpAndBlend3DRestrictingGroupFactory,
        PolynomialEquidistantSimplexGroupFactory,
        LegendreGaussLobattoTensorProductGroupFactory,
        PolynomialRecursiveNodesGroupFactory,
        )
from meshmode.discretization import Discretization
from meshmode.discretization.connection import FACE_RESTR_ALL
import meshmode.mesh.generation as mgen

import pytest

import logging
logger = logging.getLogger(__name__)


def connection_is_permutation(actx, conn):
    for i_tgrp, cgrp in enumerate(conn.groups):
        for i_batch, batch in enumerate(cgrp.batches):
            if not len(batch.from_element_indices):
                continue

            point_pick_indices = conn._resample_point_pick_indices(
                    actx, i_tgrp, i_batch)

            if point_pick_indices is None:
                return False

    return True


@pytest.mark.parametrize("group_factory", [
        "warp_and_blend",
        PolynomialEquidistantSimplexGroupFactory,
        LegendreGaussLobattoTensorProductGroupFactory,
        partial(PolynomialRecursiveNodesGroupFactory, family="lgl"),
        ])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_bdry_restriction_is_permutation(actx_factory, group_factory, dim, order):
    """Check that restriction to the boundary and opposite-face swap
    for the element groups, orders and dimensions above is actually just
    indirect access.
    """
    actx = actx_factory()

    if group_factory == "warp_and_blend":
        group_factory = {
                2: PolynomialWarpAndBlend2DRestrictingGroupFactory,
                3: PolynomialWarpAndBlend3DRestrictingGroupFactory,
                }[dim]

    if group_factory is LegendreGaussLobattoTensorProductGroupFactory:
        group_cls = TensorProductElementGroup
    else:
        group_cls = SimplexElementGroup

    mesh = mgen.generate_warped_rect_mesh(dim, order=order, nelements_side=5,
            group_cls=group_cls)

    vol_discr = Discretization(actx, mesh, group_factory(order))
    from meshmode.discretization.connection import (
            make_face_restriction, make_opposite_face_connection)
    bdry_connection = make_face_restriction(
            actx, vol_discr, group_factory(order),
            FACE_RESTR_ALL)

    assert connection_is_permutation(actx, bdry_connection)

    is_lgl = group_factory is LegendreGaussLobattoTensorProductGroupFactory

    # FIXME: This should pass unconditionally
    should_pass = (
            (dim == 3 and order < 2)
            or (dim == 2 and not is_lgl)
            or (dim == 2 and is_lgl and order < 4)
            )

    if should_pass:
        opp_face = make_opposite_face_connection(actx, bdry_connection)
        assert connection_is_permutation(actx, opp_face)
    else:
        pytest.xfail("https://github.com/inducer/meshmode/pull/105")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
