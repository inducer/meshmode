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

from meshmode.array_context import make_loopy_program
from meshmode.dof_array import DOFArray
from meshmode.discretization import InterpolatoryElementGroupBase
from meshmode.discretization.poly_element import QuadratureSimplexElementGroup
from meshmode.discretization.connection.direct import DiscretizationConnection
from meshmode.discretization.modal import ModalDiscretization
from meshmode.discretization.nodal import NodalDiscretization

from pytools import memoize_in
from pytools.obj_array import obj_array_vectorized_n_args


class NodalToModalDiscretizationConnection(DiscretizationConnection):
    """A concrete subclass of :class:`DiscretizationConnection`, which
    maps nodal data to its modal representation. This connection can
    be used with both unisolvent and non-unisolvent element groups.

    .. note::

        This connection requires that both nodal and modal discretizations
        are defined on the *same* mesh.

    .. attribute:: from_discr

    .. attribute:: to_discr

    .. attribute:: groups

        a list of :class:`DiscretizationConnectionElementGroup`
        instances, with a one-to-one correspondence to the groups in
        :attr:`to_discr`.

    .. automethod:: __call__

    """

    def __init__(self, from_discr, to_discr, allow_approximate_quad=False):
        """
        :arg from_discr: an instance of
            :class:`meshmode.discretization.nodal.NodalDiscretization`
        :arg to_discr: an instance of
            :class:`meshmode.discretization.modal.ModalDiscretization`
        :arg allow_approximate_quad: an optional :class:`bool` flag indicating
            whether to proceed with numerically approximating (via quadrature)
            modal coefficients, despite using an insufficient (not high enough
            approximation order for exact integration) quadrature method.
            The default value is *False*.
        """

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

    def _project_via_quadrature(self, actx, ary, grp, mgrp):

        if not mgrp.is_orthonormal_basis:
            raise ValueError("An orthonormal basis is required to "
                             "perform a quadrature-based projection.")

        # Handle the case with non-interpolatory element groups or
        # quadrature-based element groups
        @memoize_in(actx, (NodalToModalDiscretizationConnection,
                           "apply_quadrature_proj_knl"))
        def quad_proj_keval():
            return make_loopy_program([
                "{[iel]: 0 <= iel < nelements}",
                "{[idof]: 0 <= idof < n_to_dofs}",
                "{[ibasis]: 0 <= ibasis < n_from_dofs}"
                ],
                """
                    result[iel, idof] = sum(ibasis,
                                            vtw[idof, ibasis]
                                            * nodal_coeffs[iel, ibasis])
                """,
                name="apply_quadrature_proj_knl")

        vdm = mp.vandermonde(mgrp.basis(), grp.unit_nodes)
        w_diag = np.diag(grp.weights)
        vtw = np.dot(vdm.T, w_diag)
        vtw = actx.from_numpy(vtw)

        output = actx.call_loopy(quad_proj_keval(),
                                 nodal_coeffs=ary[grp.index],
                                 vtw=vtw)

        return output

    def _compute_coeffs_via_inv_vandermonde(self, actx, ary, grp):

        # Simple mat-mul kernel to apply the inverse of the
        # Vandermonde matrix
        @memoize_in(actx, (NodalToModalDiscretizationConnection,
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

        @memoize_in(actx, (NodalToModalDiscretizationConnection, grp))
        def get_vandermonde_inverse():
            # Extract Vandermonde and compute its inverse
            vdm = mp.vandermonde(grp.basis(), grp.unit_nodes)
            vdm_inv = la.inv(vdm)
            vdm_inv = actx.from_numpy(vdm_inv)
            return vdm_inv

        vdm_inv = get_vandermonde_inverse()
        output = actx.call_loopy(vinv_keval(),
                                 vdm_inv=vdm_inv,
                                 nodal_coeffs=ary[grp.index])

        return output

    @obj_array_vectorized_n_args
    def __call__(self, ary):
        r"""Computes modal coefficients data from a functions
        nodal coefficients. For interpolatory (unisolvent) element
        groups, this is performed via:

        .. math::

            y = V^{-1} [\text{nodal basis coefficients}]

        where :math:`V_{i,j} = \phi_j(x_i)` is the generalized
        Vandermonde matrix, :math:`\phi_j` is a nodal basis,
        and :math:`x_i` are nodal points on the reference
        element defining the nodal discretization.

        For non-interpolatory element groups (for example,
        :class:`meshmode.discretization.poly_element.QuadratureSimplexElementGroup`),
        modal coefficients are computed using the underlying quadrature rule
        :math:`(w_q, x_q)`, and an orthonormal basis :math:`\psi_i`
        spanning the modal discretization space. The modal coefficients
        are then obtained via:

        .. math::

            y = V^T W [\text{nodal basis coefficients}]

        where :math:`V_{i, j} = \psi_j(x_i)` is the Vandermonde matrix
        constructed from the orthonormal basis evaluated at the quadrature
        nodes :math:`x_i`, and :math:`W = \text{Diag}(w_q)` is a diagonal
        matrix containing the quadrature weights :math:`w_q`.

        :arg ary: a :class:`meshmode.dof_array.DOFArray` containing
            nodal coefficient data.
        """

        if not isinstance(ary, DOFArray):
            raise TypeError("Non-array passed to discretization connection")

        if ary.shape != (len(self.from_discr.groups),):
            raise ValueError("Invalid shape of incoming nodal data")

        actx = ary.array_context
        result_data = []

        for igrp, grp in enumerate(self.from_discr.groups):

            mgrp = self.to_discr.groups[igrp]

            # For element groups without an interpolatory nodal basis,
            # we use an orthonormal basis and a quadrature rule
            # to compute the modal coefficients.
            if isinstance(grp, QuadratureSimplexElementGroup):

                if (
                    grp._quadrature_rule().exact_to < 2*mgrp.order
                    and not self._allow_approximate_quad
                ):
                    raise ValueError("Quadrature rule is not exact, please "
                                     "set `allow_approximate_quad=True`")

                output = self._project_via_quadrature(
                    actx, ary, grp, mgrp)

            # Handle all other interpolatory element groups by
            # inverting the Vandermonde matrix to compute the
            # modal coefficients
            elif isinstance(grp, InterpolatoryElementGroupBase):
                output = self._compute_coeffs_via_inv_vandermonde(
                    actx, ary, grp)
            else:
                raise NotImplementedError(
                    "Don't know how to project from group types "
                    "%s to %s" % (grp.__class__.__name__,
                                  mgrp.__class__.__name__)
                    )

            result_data.append(output["result"])

        return DOFArray(actx, data=tuple(result_data))


class ModalToNodalDiscretizationConnection(DiscretizationConnection):
    """A concrete subclass of :class:`DiscretizationConnection`, which
    maps modal data back to its nodal representation.

    .. note::

        This connection requires that both nodal and modal discretizations
        are defined on the *same* mesh.

    .. attribute:: from_discr

    .. attribute:: to_discr

    .. attribute:: groups

        a list of :class:`DiscretizationConnectionElementGroup`
        instances, with a one-to-one correspondence to the groups in
        :attr:`to_discr`.

    .. automethod:: __call__

    """

    def __init__(self, from_discr, to_discr):
        """
        :arg from_discr: an instance of
            :class:`meshmode.discretization.modal.ModalDiscretization`
        :arg to_discr: an instance of
            :class:`meshmode.discretization.nodal.NodalDiscretization`
        """

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
        r"""Computes nodal coefficients from modal data via

        .. math::

            y = V [\text{modal basis coefficients}]

        where :math:`V_{i,j} = \phi_j(x_i)` is the generalized
        Vandermonde matrix, :math:`\phi_j` is an orthonormal (modal)
        basis, and :math:`x_i` are nodal points on the reference
        element defining the nodal discretization.

        :arg ary: a :class:`meshmode.dof_array.DOFArray` containing
            modal coefficient data.
        """

        if not isinstance(ary, DOFArray):
            raise TypeError("Non-array passed to discretization connection")

        if ary.shape != (len(self.from_discr.groups),):
            raise ValueError("Invalid shape of incoming modal data")

        actx = ary.array_context

        # Evaluates the action of the Vandermonde matrix on the
        # vector of modal coefficeints to obtain nodal values
        @memoize_in(actx, (ModalToNodalDiscretizationConnection,
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

        result_data = []
        for igrp, grp in enumerate(self.to_discr.groups):

            basis_fns = self.from_discr.groups[igrp].orthonormal_basis()
            vdm = mp.vandermonde(basis_fns, grp.unit_nodes)
            vdm = actx.from_numpy(vdm)

            output = actx.call_loopy(keval(),
                                     vdm=vdm,
                                     coefficients=ary[grp.index])
            result_data.append(output["result"])

        return DOFArray(actx, data=tuple(result_data))
