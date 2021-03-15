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

    @obj_array_vectorized_n_args
    def __call__(self, ary):

        if not isinstance(ary, DOFArray):
            raise TypeError("Non-array passed to discretization connection")

        if ary.shape != (len(self.from_discr.groups),):
            raise ValueError("Invalid shape of incoming nodal data")

        actx = ary.array_context

        # TODO: Handle the case with non-interpolatory element groups
        # I.e. use the quadrature rule and orthonormal basis to directly
        # compute the modal coefficients:
        # c_i = inner_product(v, phi_i), where phi_i is the ith orthonormal
        # basis vector

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

        result = self.to_discr.zeros(actx, dtype=ary.entry_dtype)

        for igrp, grp in enumerate(self.to_discr.groups):
            vdm = actx.from_numpy(mp.vandermonde(grp.basis(), grp.unit_nodes))
            actx.call_loopy(
                    vinv_keval(),
                    result=result[grp.index],
                    vdm_inv=la.inv(vdm),
                    nodal_coeffs=ary[grp.index])

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
