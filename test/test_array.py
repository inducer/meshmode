from __future__ import annotations


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

import logging
from dataclasses import dataclass

import numpy as np
import pytest

from arraycontext import (
    ArrayContextFactory,
    dataclass_array_container,
    flatten,
    pytest_generate_tests_for_array_contexts,
    with_container_arithmetic,
)
from pytools.obj_array import make_obj_array
from pytools.tag import Tag

from meshmode import _acf  # noqa: F401
from meshmode.array_context import (
    PytestPyOpenCLArrayContextFactory,
    PytestPytatoPyOpenCLArrayContextFactory,
)
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import default_simplex_group_factory
from meshmode.dof_array import DOFArray, array_context_for_pickling, flat_norm


logger = logging.getLogger(__name__)
pytest_generate_tests = pytest_generate_tests_for_array_contexts(
        [PytestPytatoPyOpenCLArrayContextFactory,
         PytestPyOpenCLArrayContextFactory,
         ])


@with_container_arithmetic(bcast_obj_array=False,
                           rel_comparison=True,
                           _bcast_actx_array_type=False,
                           _cls_has_array_context_attr=True)
@dataclass_array_container
@dataclass(frozen=True)
class MyContainer:
    name: str
    mass: DOFArray
    momentum: np.ndarray
    enthalpy: DOFArray

    __array_ufunc__ = None

    @property
    def array_context(self):
        return self.mass.array_context


# {{{ test_container_norm

def _get_test_containers(actx, ambient_dim=2):
    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(-0.5,)*ambient_dim,
            b=(+0.5,)*ambient_dim,
            nelements_per_axis=(3,)*ambient_dim, order=1)
    discr = Discretization(actx, mesh, default_simplex_group_factory(ambient_dim, 3))
    x = actx.thaw(discr.nodes()[0])

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
def test_container_norm(actx_factory: ArrayContextFactory, ord):
    actx = actx_factory()
    c_test = _get_test_containers(actx)

    # {{{ flat_norm

    from pytools.obj_array import make_obj_array
    c = MyContainer(name="hey", mass=1, momentum=make_obj_array([2, 3]), enthalpy=5)
    c_obj_ary = make_obj_array([c, c])

    n1 = flat_norm(c_obj_ary, ord)
    n2 = np.linalg.norm([1, 2, 3, 5]*2, ord)
    assert abs(n1 - n2) < 1e-12

    # check nested container with only Numbers (and no actx)
    assert abs(flat_norm(c_obj_ary, ord=ord) - n2) < 1.0e-12
    assert abs(
            flat_norm(np.array([1, 1], dtype=object), ord=ord)
            - np.linalg.norm([1, 1], ord=ord)) < 1.0e-12

    # check nested
    n1 = actx.to_numpy(flat_norm(c_test[1], ord=ord))
    n2 = np.linalg.norm([
        np.linalg.norm(actx.to_numpy(flatten(ary, actx)), ord=ord) for ary in c_test[1]
        ], ord=ord)
    assert abs(n1 - n2) < 1e-12

    # }}}

# }}}


# {{{ test_dof_array_pickling

def test_dof_array_pickling(actx_factory: ArrayContextFactory):
    actx = actx_factory()
    _, _, mat_of_dofs, dc_of_dofs = _get_test_containers(actx)

    from pickle import dumps, loads
    with array_context_for_pickling(actx):
        pkl = dumps((mat_of_dofs, dc_of_dofs))

    with array_context_for_pickling(actx):
        mat2_of_dofs, dc2_of_dofs = loads(pkl)

    assert actx.to_numpy(flat_norm(mat_of_dofs - mat2_of_dofs, np.inf)) == 0
    assert actx.to_numpy(flat_norm(dc_of_dofs - dc2_of_dofs, np.inf)) == 0


class FooTag(Tag):
    pass


class FooAxisTag(Tag):
    pass


class FooAxisTag2(Tag):
    pass


def test_dof_array_pickling_tags(actx_factory: ArrayContextFactory):
    actx = actx_factory()

    from pickle import dumps, loads

    state = DOFArray(actx, (
        actx.np.zeros((10, 10), dtype=np.float64),
        actx.np.zeros((10, 10), dtype=np.float64),
        ))

    state = actx.thaw(actx.freeze(actx.tag(FooTag(), state)))
    state = actx.thaw(actx.freeze(actx.tag_axis(0, FooAxisTag(), state)))
    state = actx.thaw(actx.freeze(actx.tag_axis(1, FooAxisTag2(), state)))

    with array_context_for_pickling(actx):
        pkl = dumps((state, ))

    with array_context_for_pickling(actx):
        loaded_state, = loads(pkl)

    for i in range(len(state._data)):
        si = state._data[i]
        li = loaded_state._data[i]
        assert si.tags == li.tags

        for iax in range(len(si.axes)):
            assert si.axes[iax].tags == li.axes[iax].tags

# }}}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
