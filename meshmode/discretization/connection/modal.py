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
from meshmode.discretization.poly_element import (
    QuadratureSimplexElementGroup, InterpolatoryQuadratureSimplexElementGroup)
from meshmode.discretization.connection.direct import DiscretizationConnection
from meshmode.discretization.modal import ModalDiscretization
from meshmode.discretization.nodal import NodalDiscretization

from pytools import memoize_in
from pytools.obj_array import obj_array_vectorized_n_args


class ModalDiscretizationConnection(DiscretizationConnection):

    def __init__(self, from_discr, to_discr, allow_approximate_quad=False):

        if not isinstance(from_discr, NodalDiscretization):
            raise TypeError("from_discr must be a NodalDiscretization "
                            "object to use this connection.")

        if not isinstance(to_discr, ModalDiscretization):
            raise TypeError("to_discr must be a ModalDiscretization "
                            "object to use this connection.")

        if to_discr.mesh != from_discr.mesh:
            raise ValueError("Both `from_discr` and `to_discr` must be on "
                             "the same mesh.")

        super().__init__(
                from_discr=from_discr,
                to_discr=to_discr,
                is_surjective=True)

        self._allow_approximate_quad = allow_approximate_quad

    def _project_via_quadrature(self, actx, ary, result, grp, modal_basis_fns):

        # Handle the case with non-interpolatory element groups or
        # quadrature-based element groups for overintegration.
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
                "{[ibasis]: 0 <= ibasis < n_to_dofs}"
                ],
                """
                    result[iel, ibasis] = sum(idof_quad,
                                              ary[iel, idof_quad
                                              * basis[idof_quad]
                                              * weights[idof_quad])
                """,
                name="quadrature_proj_eval_knl")

        for ibasis, basis_fn in enumerate(modal_basis_fns):

            basis = actx.from_numpy(basis_fn(grp.unit_nodes))
            weights = actx.from_numpy(grp.weights)

            actx.call_loopy(quad_proj_keval(),
                            ibasis=ibasis,
                            ary=ary[grp.index],
                            basis=basis,
                            weights=weights,
                            result=result[grp.index])

        return result

    def _compute_coeffs_via_inv_vandermonde(self, actx, ary, result, grp):

        # Simple mat-mul kernel to apply the inverse of the
        # Vandermonde matrix
        @memoize_in(actx, (ModalDiscretizationConnection,
                           "apply_inv_vandermonde_knl"))
        def vinv_keval():
            return make_loopy_program([
                "{[iel]: 0 <= iel < nelements}",
                "{[idof]: 0 <= idof < n_from_dofs}",
                "{[jdof]: 0 <= jdof < n_from_dofs}"
                ],
                """
                    result[iel, idof] = sum(jdof,
                                            vdm_inv[idof, jdof]
                                            * nodal_coeffs[iel, jdof])
                """,
                name="apply_inv_vandermonde_knl")

        # Extract Vandermonde and compute its inverse
        # TODO: need to figure out how to cache this inverse
        # Idea: Use element group as a key and create a lookup
        # dictionary that is memoized in the array context
        vdm = mp.vandermonde(grp.basis(), grp.unit_nodes)
        vdm_inv = la.inv(vdm)
        vdm_inv = actx.from_numpy(vdm_inv)

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

        for igrp, grp in enumerate(self.from_discr.groups):

            if isinstance(grp, QuadratureSimplexElementGroup):

                mgrp = self.to_discr.groups[igrp]

                if not self._allow_approximate_quad:
                    assert grp._quadrature_rule().exact_to >= 2*mgrp.order, \
                    "If quadrature rule is not exact, set `approximate_quad=True`."

                # Use the modal (orthonormal basis) from the `to_discr`
                # to evaluate the Vandermonde matrix for non-interpolatory
                # element groups
                basis = mgrp.orthonormal_basis()
                result = self._project_via_quadrature(
                    actx, ary, result, grp, basis)
            else:
                result = self._compute_coeffs_via_inv_vandermonde(
                    actx, ary, result, grp)

        return result


class ModalInverseDiscretizationConnection(DiscretizationConnection):

    def __init__(self, from_discr, to_discr):

        if not isinstance(from_discr, ModalDiscretization):
            raise TypeError("from_discr must be a ModalDiscretization "
                            "object to use this connection.")

        if not isinstance(to_discr, NodalDiscretization):
            raise TypeError("to_discr must be a NodalDiscretization "
                            "object to use this connection.")

        if to_discr.mesh != from_discr.mesh:
            raise ValueError("Both `from_discr` and `to_discr` must be on "
                             "the same mesh.")

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
                "{[ibasis]: 0 <= ibasis < n_from_dofs}"
                ],
                """
                    result[iel, idof] = sum(ibasis,
                                            vdm[idof, ibasis]
                                            * coefficients[iel, ibasis])
                """,
                name="modinv_evaluation_knl")

        result_data = ()
        for igrp, grp in enumerate(self.to_discr.groups):

            basis_fns = self.from_discr.groups[igrp].orthonormal_basis()
            vdm = mp.vandermonde(basis_fns, grp.unit_nodes)
            vdm = actx.from_numpy(vdm)

            output = actx.call_loopy(keval(),
                                     vdm=vdm,
                                     coefficients=ary[grp.index])
            result_data += (output['result'],)

        return DOFArray(actx, data=result_data)
