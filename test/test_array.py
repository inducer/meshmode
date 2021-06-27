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

from dataclasses import dataclass
import pytest
import numpy as np

from meshmode import _acf  # noqa: F401
from meshmode.array_context import PytestPyOpenCLArrayContextFactory
from arraycontext import pytest_generate_tests_for_array_contexts
pytest_generate_tests = pytest_generate_tests_for_array_contexts(
        [PytestPyOpenCLArrayContextFactory])

from arraycontext import (
        thaw,
        dataclass_array_container,
        with_container_arithmetic)

from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import default_simplex_group_factory
from meshmode.dof_array import DOFArray, flat_norm, array_context_for_pickling

from pytools.obj_array import make_obj_array

import logging
logger = logging.getLogger(__name__)


@with_container_arithmetic(bcast_obj_array=False, rel_comparison=True)
@dataclass_array_container
@dataclass(frozen=True)
class MyContainer:
    name: str
    mass: DOFArray
    momentum: np.ndarray
    enthalpy: DOFArray

    @property
    def array_context(self):
        return self.mass.array_context


def test_flatten_unflatten(actx_factory):
    actx = actx_factory()

    ambient_dim = 2
    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(-0.5,)*ambient_dim,
            b=(+0.5,)*ambient_dim,
            n=(3,)*ambient_dim, order=1)
    discr = Discretization(actx, mesh, default_simplex_group_factory(ambient_dim, 3))
    a = np.random.randn(discr.ndofs)

    from meshmode.dof_array import flatten, unflatten
    a_round_trip = actx.to_numpy(flatten(unflatten(actx, discr, actx.from_numpy(a))))
    assert np.array_equal(a, a_round_trip)

    from meshmode.dof_array import flatten_to_numpy, unflatten_from_numpy
    a_round_trip = flatten_to_numpy(actx, unflatten_from_numpy(actx, discr, a))
    assert np.array_equal(a, a_round_trip)

    x = thaw(discr.nodes(), actx)
    avg_mass = DOFArray(actx, tuple([
        (np.pi + actx.zeros((grp.nelements, 1), a.dtype)) for grp in discr.groups
        ]))

    c = MyContainer(name="flatten",
            mass=avg_mass,
            momentum=make_obj_array([x, x, x]),
            enthalpy=x)

    from meshmode.dof_array import unflatten_like
    c_round_trip = unflatten_like(actx, flatten(c), c)
    assert flat_norm(c - c_round_trip) < 1.0e-8


def _get_test_containers(actx, ambient_dim=2):
    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(-0.5,)*ambient_dim,
            b=(+0.5,)*ambient_dim,
            n=(3,)*ambient_dim, order=1)
    discr = Discretization(actx, mesh, default_simplex_group_factory(ambient_dim, 3))
    x = thaw(discr.nodes()[0], actx)

    # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
    dataclass_of_dofs = MyContainer(
            name="container",
            mass=x,
            momentum=make_obj_array([x, x]),
            enthalpy=x)

    ary_dof = x
    ary_of_dofs = make_obj_array([x, x, x])
    mat_of_dofs = np.empty((2, 2), dtype=object)
    for i in np.ndindex(mat_of_dofs.shape):
        mat_of_dofs[i] = x

    return ary_dof, ary_of_dofs, mat_of_dofs, dataclass_of_dofs


@pytest.mark.parametrize("ord", [2, np.inf])
def test_container_norm(actx_factory, ord):
    actx = actx_factory()

    ary_dof, ary_of_dofs, mat_of_dofs, dc_of_dofs = _get_test_containers(actx)

    from pytools.obj_array import make_obj_array
    c = MyContainer(name="hey", mass=1, momentum=make_obj_array([2, 3]), enthalpy=5)
    n1 = actx.np.linalg.norm(make_obj_array([c, c]), ord)
    n2 = np.linalg.norm([1, 2, 3, 5]*2, ord)

    assert abs(n1 - n2) < 1e-12
    assert abs(flat_norm(ary_dof, ord) - actx.np.linalg.norm(ary_dof, ord)) < 1e-12


def test_dof_array_pickling(actx_factory):
    actx = actx_factory()
    ary_dof, ary_of_dofs, mat_of_dofs, dc_of_dofs = _get_test_containers(actx)

    from pickle import loads, dumps
    with array_context_for_pickling(actx):
        pkl = dumps((mat_of_dofs, dc_of_dofs))

    with array_context_for_pickling(actx):
        mat2_of_dofs, dc2_of_dofs = loads(pkl)

    assert flat_norm(mat_of_dofs - mat2_of_dofs, np.inf) == 0
    assert flat_norm(dc_of_dofs - dc2_of_dofs, np.inf) == 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
