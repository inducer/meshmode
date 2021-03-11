__copyright__ = """
Copyright (C) 2013-2021 Andreas Kloeckner
Copyright (C) 2021 Thomas Gibson
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


from pytools import memoize_in, memoize_method
from pytools.obj_array import make_obj_array
from meshmode.array_context import make_loopy_program

# underscored because it shouldn't be imported from here.
from meshmode.dof_array import DOFArray as _DOFArray

from meshmode.discretization import DiscretizationBase


class ModalDiscretization(DiscretizationBase):
    """An unstructured composite modal discretization."""

    @memoize_method
    def modes(self):
        r"""
        :returns: object array of shape ``(ambient_dim,)`` containing
            :class:`~meshmode.dof_array.DOFArray`\ s of modal coefficients.
        """

        actx = self._setup_actx

        @memoize_in(actx, (ModalDiscretization, "modes_prg"))
        def prg():
            return make_loopy_program(
                """{[iel,idof,j]:
                    0<=iel<nelements and
                    0<=idof<ndiscr_nodes and
                    0<=j<nmesh_nodes}""",
                """
                    result[iel, idof] = \
                        sum(j, resampling_mat[idof, j] * modes[iel, j])
                    """,
                name="modes")

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
