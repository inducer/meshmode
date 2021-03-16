__copyright__ = """
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
import numpy.linalg as la
import modepy as mp
import loopy as lp

from meshmode.array_context import make_loopy_program
from meshmode.dof_array import DOFArray
from meshmode.discretization.connection.direct import DiscretizationConnection
from meshmode.discretization.modal import ModalDiscretization
from meshmode.discretization.nodal import NodalDiscretization

from pytools import memoize_in
from pytools.obj_array import obj_array_vectorized_n_args


class ModalDiscretizationConnection(DiscretizationConnection):

    def __init__(self, from_discr, to_discr):

        if not isinstance(from_discr, NodalDiscretization):
            raise ValueError("from_discr must be a NodalDiscretization "
                    "object to use this connection.")

        if not isinstance(to_discr, ModalDiscretization):
            raise ValueError("to_discr must be a ModalDiscretization "
                    "object to use this connection.")

        super().__init__(
                from_discr=from_discr,
                to_discr=to_discr,
                is_surjective=True)

    def _quadrature_projection(self, actx, ary, result, grp):

        # Handle the case with non-interpolatory element groups
        # I.e. use the quadrature rule and orthonormal basis to directly
        # compute the modal coefficients:
        # c_i = inner_product(v, phi_i), where phi_i is the ith orthonormal
        # basis vector
        @memoize_in(actx, (ModalDiscretizationConnection,
                           "quadrature_proj_eval_knl"))
        def quad_proj_keval():
            return make_loopy_program([
                "{[iel]: 0 <= iel < nelements}",
                "{[idof_quad]: 0 <= idof_quad < n_from_nodes}"
                ],
                """
                for iel
                    <>tmp = sum(idof_quad,
                            ary[iel, idof_quad]
                            * basis[idof_quad]
                            * weights[idof_quad])

                    result[iel, ibasis] = result[iel, ibasis] + tmp
                end
                """,
                [
                    lp.GlobalArg("ary", None,
                        shape=("n_from_elements", "n_from_nodes")),
                    lp.GlobalArg("result", None,
                        shape=("n_to_elements", "n_to_nodes")),
                    lp.GlobalArg("basis", None,
                        shape="n_from_nodes"),
                    lp.GlobalArg("weights", None,
                        shape="n_from_nodes"),
                    lp.ValueArg("n_from_elements", np.int32),
                    lp.ValueArg("n_to_elements", np.int32),
                    lp.ValueArg("n_to_nodes", np.int32),
                    lp.ValueArg("ibasis", np.int32),
                    "..."
                    ],
                name="quadrature_proj_eval_knl")

        for ibasis, basis_fn in enumerate(grp.basis()):
            basis = actx.from_numpy(basis_fn(grp.unit_nodes))

            actx.call_loopy(quad_proj_keval(),
                            ibasis=ibasis,
                            ary=ary[grp.index],
                            basis=basis,
                            weights=grp.weights,
                            result=result[grp.index])

        return result

    def _invert_vandermonde(self, actx, ary, result, grp):

        # Simple mat-mul kernel to apply the inverse of the
        # Vandermonde matrix
        @memoize_in(actx, (ModalDiscretizationConnection,
                           "vandermond_inv_eval_knl"))
        def vinv_keval():
            return make_loopy_program([
                "{[iel]: 0 <= iel < nelements}",
                "{[idof]: 0 <= idof < n_from_dofs}",
                "{[jdof]: 0 <= jdof < n_from_dofs}"
                ],
                """
                    result[iel, idof] = result[iel, idof] + \
                        sum(jdof, vdm_inv[idof, jdof] * nodal_coeffs[iel, jdof])
                """,
                [
                    lp.GlobalArg("nodal_coeffs", None,
                        shape=("nelements", "n_from_dofs")),
                    "..."
                    ],
                name="vandermond_inv_eval_knl")

        vdm = mp.vandermonde(grp.basis(), grp.unit_nodes)
        vdm_inv = actx.from_numpy(la.inv(vdm))
        actx.call_loopy(vinv_keval(),
                        result=result[grp.index],
                        vdm_inv=vdm_inv,
                        nodal_coeffs=ary[grp.index])

        return result

    @obj_array_vectorized_n_args
    def __call__(self, ary):

        if not isinstance(ary, DOFArray):
            raise TypeError("Non-array passed to discretization connection")

        if ary.shape != (len(self.from_discr.groups),):
            raise ValueError("Invalid shape of incoming nodal data")

        actx = ary.array_context

        result = self.to_discr.zeros(actx, dtype=ary.entry_dtype)

        for _, grp in enumerate(self.from_discr.groups):
            # TODO: Put in if-statements to determine which routine to call
            result = self._invert_vandermonde(actx, ary, result, grp)

        return result


class ModalInverseDiscretizationConnection(DiscretizationConnection):

    def __init__(self, from_discr, to_discr):

        if not isinstance(from_discr, ModalDiscretization):
            raise ValueError("from_discr must be a ModalDiscretization "
                    "object to use this connection.")

        if not isinstance(to_discr, NodalDiscretization):
            raise ValueError("to_discr must be a NodalDiscretization "
                    "object to use this connection.")

        super().__init__(
                from_discr=from_discr,
                to_discr=to_discr,
                is_surjective=True)

    @obj_array_vectorized_n_args
    def __call__(self, ary):

        if not isinstance(ary, DOFArray):
            raise TypeError("Non-array passed to discretization connection")

        if ary.shape != (len(self.from_discr.groups),):
            raise ValueError("Invalid shape of incoming modal data")

        actx = ary.array_context

        # Evaluates the action of the Vandermonde matrix on the
        # vector of modal coefficeints to obtain nodal values
        @memoize_in(actx, (ModalInverseDiscretizationConnection,
                           "modinv_evaluation_knl"))
        def keval():
            return make_loopy_program([
                "{[iel]: 0 <= iel < nelements}",
                "{[idof]: 0 <= idof < n_to_dofs}",
                "{[ibasis]: 0 <= ibasis < n_to_dofs}"
                ],
                """
                    result[iel, idof] = result[iel, idof] + \
                        sum(ibasis, vdm[idof, ibasis] * coefficients[iel, ibasis])
                """,
                [
                    lp.GlobalArg("coefficients", None,
                        shape=("nelements", "n_to_dofs")),
                    "..."
                    ],
                name="modinv_evaluation_knl")

        result = self.to_discr.zeros(actx, dtype=ary.entry_dtype)

        for igrp, grp in enumerate(self.to_discr.groups):
            vdm = actx.from_numpy(mp.vandermonde(grp.basis(), grp.unit_nodes))
            actx.call_loopy(
                    keval(),
                    result=result[grp.index],
                    vdm=vdm,
                    coefficients=ary[grp.index])

        return result
