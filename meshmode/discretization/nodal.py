__copyright__ = """
Copyright (C) 2013-2021 Andreas Kloeckner
Copyright (C) 2021 University of Illinois Board of Trustees
"""

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

from pytools import memoize_in, memoize_method
from pytools.obj_array import make_obj_array
from meshmode.array_context import ArrayContext, make_loopy_program

# underscored because it shouldn't be imported from here.
from meshmode.dof_array import DOFArray as _DOFArray

from meshmode.discretization import DiscretizationBase
from meshmode.discretization.poly_element import (
        ModalSimplexGroupFactory,
        ModalTensorProductGroupFactory)


class NodalDiscretization(DiscretizationBase):
    """An unstructured composite nodal discretization.

    .. attribute:: ndofs

    .. automethod:: nodes
    .. automethod:: num_reference_derivative
    .. automethod:: quad_weights
    """

    def __init__(self, actx: ArrayContext, mesh, group_factory,
            real_dtype=np.float64):
        """
        :param actx: A :class:`ArrayContext` used to perform computation needed
            during initial set-up of the mesh.
        :param mesh: A :class:`meshmode.mesh.Mesh` over which the discretization is
            built.
        :param group_factory: An :class:`ElementGroupFactory`. Note that the
            element groups must be subclasses of :class:`NodalElementGroupBase`.
        :param real_dtype: The :mod:`numpy` data type used for representing real
            data, either :class:`numpy.float32` or :class:`numpy.float64`.
        """

        # Some sanity checking so we don't do something horrendous later on
        if isinstance(group_factory, (ModalSimplexGroupFactory,
                                      ModalTensorProductGroupFactory)):
            raise ValueError("Cannot use a modal element group factory with "
                             "a NodalDiscretization.")

        super().__init__(actx, mesh, group_factory, real_dtype)

    @property
    def ndofs(self):
        return sum(grp.ndofs for grp in self.groups)

    @memoize_method
    def nodes(self):
        r"""
        :returns: object array of shape ``(ambient_dim,)`` containing
            :class:`~meshmode.dof_array.DOFArray`\ s of node coordinates.
        """

        actx = self._setup_actx

        @memoize_in(actx, (NodalDiscretization, "nodes_prg"))
        def prg():
            return make_loopy_program(
                """{[iel,idof,j]:
                    0<=iel<nelements and
                    0<=idof<ndiscr_nodes and
                    0<=j<nmesh_nodes}""",
                """
                    result[iel, idof] = \
                        sum(j, resampling_mat[idof, j] * nodes[iel, j])
                    """,
                name="nodes")

        return make_obj_array([
            _DOFArray(None, tuple(
                actx.freeze(
                    actx.call_loopy(
                        prg(),
                        resampling_mat=actx.from_numpy(
                            grp.from_mesh_interp_matrix()),
                        nodes=actx.from_numpy(grp.mesh_el_group.nodes[iaxis])
                        )["result"])
                for grp in self.groups))
            for iaxis in range(self.ambient_dim)])

    def num_reference_derivative(self, ref_axes, vec):
        actx = vec.array_context
        ref_axes = list(ref_axes)

        @memoize_in(actx, (NodalDiscretization, "reference_derivative_prg"))
        def prg():
            return make_loopy_program(
                """{[iel,idof,j]:
                    0<=iel<nelements and
                    0<=idof,j<nunit_dofs}""",
                "result[iel,idof] = sum(j, diff_mat[idof, j] * vec[iel, j])",
                name="diff")

        def get_mat(grp):
            mat = None
            for ref_axis in ref_axes:
                next_mat = grp.diff_matrices()[ref_axis]
                if mat is None:
                    mat = next_mat
                else:
                    mat = np.dot(next_mat, mat)

            return mat

        return _DOFArray(actx, tuple(
                actx.call_loopy(
                    prg(), diff_mat=actx.from_numpy(get_mat(grp)), vec=vec[grp.index]
                    )["result"]
                for grp in self.groups))

    @memoize_method
    def quad_weights(self):
        """:returns: A :class:`~meshmode.dof_array.DOFArray` with quadrature weights.
        """
        actx = self._setup_actx

        @memoize_in(actx, (NodalDiscretization, "quad_weights_prg"))
        def prg():
            return make_loopy_program(
                "{[iel,idof]: 0<=iel<nelements and 0<=idof<nunit_dofs}",
                "result[iel,idof] = weights[idof]",
                name="quad_weights")

        return _DOFArray(None, tuple(
                actx.freeze(
                    actx.call_loopy(
                        prg(),
                        weights=actx.from_numpy(grp.weights),
                        nelements=grp.nelements,
                        )["result"])
                for grp in self.groups))


# Create a temporary alias for NodalDiscretization until completely
# phased out in libraries using meshmode.
def Discretization(*args, **kwargs):  # noqa: N802
    from warnings import warn
    warn("Discretization will be depreciated in favor of NodalDiscretization")
    return NodalDiscretization(*args, **kwargs)
