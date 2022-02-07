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

from arraycontext import (NotAnArrayContainerError,
    serialize_container, deserialize_container,)
from meshmode.transform_metadata import (FirstAxisIsElementsTag,
    IsDOFArray, IsOpArray, EinsumArgsTags)
from meshmode.discretization import InterpolatoryElementGroupBase
from meshmode.discretization.poly_element import QuadratureSimplexElementGroup
from meshmode.discretization.connection.direct import DiscretizationConnection
from immutables import Map

from pytools import keyed_memoize_in


class NodalToModalDiscretizationConnection(DiscretizationConnection):
    r"""A concrete subclass of :class:`DiscretizationConnection`, which
    maps nodal data to its modal representation. For interpolatory
    (unisolvent) element groups, the mapping from nodal to modal
    representations is performed via:

    .. math::

        y = V^{-1} [\text{nodal basis coefficients}]

    where :math:`V_{i,j} = \phi_j(x_i)` is the generalized
    Vandermonde matrix, :math:`\phi_j` is a nodal basis,
    and :math:`x_i` are nodal points on the reference
    element defining the nodal discretization.

    For non-interpolatory element groups (for example,
    :class:`~meshmode.discretization.poly_element.QuadratureSimplexElementGroup`),
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

    .. note::

        This connection requires that both nodal and modal discretizations
        are defined on the *same* mesh.

    .. attribute:: from_discr

        An instance of :class:`meshmode.discretization.Discretization` containing
        :class:`~meshmode.discretization.NodalElementGroupBase` element groups

    .. attribute:: to_discr

        An instance of :class:`meshmode.discretization.Discretization` containing
        :class:`~meshmode.discretization.ModalElementGroupBase` element groups

    .. attribute:: groups

        A list of :class:`DiscretizationConnectionElementGroup`
        instances, with a one-to-one correspondence to the groups in
        :attr:`to_discr`.

    .. automethod:: __init__
    .. automethod:: __call__

    """

    def __init__(self, from_discr, to_discr, allow_approximate_quad=False):
        """
        :arg from_discr: a :class:`meshmode.discretization.Discretization`
            containing :class:`~meshmode.discretization.NodalElementGroupBase`
            element groups.
        :arg to_discr: a :class:`meshmode.discretization.Discretization`
            containing :class:`~meshmode.discretization.ModalElementGroupBase`
            element groups.
        :arg allow_approximate_quad: an optional :class:`bool` flag indicating
            whether to proceed with numerically approximating (via quadrature)
            modal coefficients, even when the underlying quadrature method
            is not exact. The default value is *False*.
        """

        if not from_discr.is_nodal:
            raise ValueError("`from_discr` must be defined on nodal "
                             "element groups to use this connection.")

        if not to_discr.is_modal:
            raise ValueError("`to_discr` must be defined on modal "
                             "element groups to use this connection.")

        if to_discr.mesh is not from_discr.mesh:
            raise ValueError("Both `from_discr` and `to_discr` must be on "
                             "the same mesh.")

        super().__init__(
                from_discr=from_discr,
                to_discr=to_discr,
                is_surjective=True)

        self._allow_approximate_quad = allow_approximate_quad

    def _project_via_quadrature(self, actx, ary, grp, mgrp):
        # Handle the case with non-interpolatory element groups or
        # quadrature-based element groups

        @keyed_memoize_in(actx, (NodalToModalDiscretizationConnection,
                                 "quadrature_matrix"),
                                lambda grp, mgrp: (
                                    grp.discretization_key(),
                                    mgrp.discretization_key(),
                                    ))
        def quadrature_matrix(grp, mgrp):
            vdm = mp.vandermonde(mgrp.basis_obj().functions,
                                 grp.unit_nodes)
            w_diag = np.diag(grp.quadrature_rule().weights)
            vtw = np.dot(vdm.T, w_diag)
            return actx.from_numpy(vtw)

        return actx.einsum("ib,eb->ei",
                           quadrature_matrix(grp, mgrp),
                           ary[grp.index],
                           tagged=(FirstAxisIsElementsTag(),))

    def _compute_coeffs_via_inv_vandermonde(self, actx, ary, grp):

        # Simple mat-mul kernel to apply the inverse of the
        # Vandermonde matrix
        @keyed_memoize_in(actx, (NodalToModalDiscretizationConnection,
                                 "vandermonde_inverse"),
                          lambda grp: grp.discretization_key())
        def vandermonde_inverse(grp):
            vdm = mp.vandermonde(grp.basis_obj().functions,
                                 grp.unit_nodes)
            vdm_inv = la.inv(vdm)
            return actx.from_numpy(vdm_inv)

        kd_tag = EinsumArgsTags(Map({"arg0": [IsOpArray()],
                    "arg1": [IsDOFArray()], "out": [IsDOFArray()]}))

        return actx.einsum("ij,ej->ei",
                           vandermonde_inverse(grp),
                           ary[grp.index],
                           tagged=(FirstAxisIsElementsTag(), kd_tag,))

    def __call__(self, ary):
        """Computes modal coefficients data from a functions
        nodal coefficients.

        :arg ary: a :class:`~meshmode.dof_array.DOFArray`, or an
            :class:`arraycontext.ArrayContainer` of them, containing nodal
            coefficient data.
        """
        # {{{ recurse into array containers

        from meshmode.dof_array import DOFArray
        if not isinstance(ary, DOFArray):
            try:
                iterable = serialize_container(ary)
            except NotAnArrayContainerError:
                pass
            else:
                return deserialize_container(ary, [
                    (key, self(subary)) for key, subary in iterable
                    ])

        # }}}

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

                if (grp.quadrature_rule().exact_to < 2*mgrp.order
                        and not self._allow_approximate_quad):
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

            result_data.append(output)

        return DOFArray(actx, data=tuple(result_data))


class ModalToNodalDiscretizationConnection(DiscretizationConnection):
    r"""A concrete subclass of :class:`DiscretizationConnection`, which
    maps modal data back to its nodal representation. This is computed
    via:

    .. math::

        y = V [\text{modal basis coefficients}]

    where :math:`V_{i,j} = \phi_j(x_i)` is the generalized
    Vandermonde matrix, :math:`\phi_j` is an orthonormal (modal)
    basis, and :math:`x_i` are nodal points on the reference
    element defining the nodal discretization.

    .. note::

        This connection requires that both nodal and modal discretizations
        are defined on the *same* mesh.

    .. attribute:: from_discr

        An instance of :class:`meshmode.discretization.Discretization` containing
        :class:`~meshmode.discretization.ModalElementGroupBase` element groups

    .. attribute:: to_discr

        An instance of :class:`meshmode.discretization.Discretization` containing
        :class:`~meshmode.discretization.NodalElementGroupBase` element groups

    .. attribute:: groups

        A list of :class:`DiscretizationConnectionElementGroup`
        instances, with a one-to-one correspondence to the groups in
        :attr:`to_discr`.

    .. automethod:: __init__
    .. automethod:: __call__

    """

    def __init__(self, from_discr, to_discr):
        """
        :arg from_discr: a :class:`meshmode.discretization.Discretization`
            containing :class:`~meshmode.discretization.ModalElementGroupBase`
            element groups.
        :arg to_discr: a :class:`meshmode.discretization.Discretization`
            containing :class:`~meshmode.discretization.NodalElementGroupBase`
            element groups.
        """

        if not from_discr.is_modal:
            raise ValueError("`from_discr` must be defined on modal "
                             "element groups to use this connection.")

        if not to_discr.is_nodal:
            raise ValueError("`to_discr` must be defined on nodal "
                             "element groups to use this connection.")

        if to_discr.mesh is not from_discr.mesh:
            raise ValueError("Both `from_discr` and `to_discr` must be on "
                             "the same mesh.")

        super().__init__(
                from_discr=from_discr,
                to_discr=to_discr,
                is_surjective=True)

    def __call__(self, ary):
        """Computes nodal coefficients from modal data.

        :arg ary: a :class:`~meshmode.dof_array.DOFArray`, or an
            :class:`arraycontext.ArrayContainer` of them, containing modal
            coefficient data.
        """
        # {{{ recurse into array containers

        from meshmode.dof_array import DOFArray
        if not isinstance(ary, DOFArray):
            try:
                iterable = serialize_container(ary)
            except NotAnArrayContainerError:
                pass
            else:
                return deserialize_container(ary, [
                    (key, self(subary)) for key, subary in iterable
                    ])

        # }}}

        if not isinstance(ary, DOFArray):
            raise TypeError("Non-array passed to discretization connection")

        if ary.shape != (len(self.from_discr.groups),):
            raise ValueError("Invalid shape of incoming modal data")

        actx = ary.array_context

        @keyed_memoize_in(actx, (ModalToNodalDiscretizationConnection, "matrix"),
                           lambda to_grp, from_grp: (
                               to_grp.discretization_key(),
                               from_grp.discretization_key(),
                               ))
        def matrix(to_grp, from_grp):
            vdm = mp.vandermonde(from_grp.basis_obj().functions,
                                 to_grp.unit_nodes)
            return actx.from_numpy(vdm)

        result_data = tuple(
            actx.einsum("ib,eb->ei",
                        matrix(grp, self.from_discr.groups[igrp]),
                        ary[grp.index],
                        tagged=(FirstAxisIsElementsTag(),))
            for igrp, grp in enumerate(self.to_discr.groups)
        )
        return DOFArray(actx, data=result_data)
