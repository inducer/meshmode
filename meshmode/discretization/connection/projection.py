__copyright__ = """Copyright (C) 2018 Alexandru Fikl"""

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

from pytools import keyed_memoize_method, memoize_in
from pytools.obj_array import obj_array_vectorized_n_args

import loopy as lp

from meshmode.array_context import make_loopy_program
from meshmode.dof_array import DOFArray
from meshmode.discretization.connection.direct import (
        DiscretizationConnection,
        DirectDiscretizationConnection)
from meshmode.discretization.connection.chained import \
        ChainedDiscretizationConnection


class L2ProjectionInverseDiscretizationConnection(DiscretizationConnection):
    """Creates an inverse :class:`DiscretizationConnection` from an existing
    connection to allow transporting from the original connection's
    *to_discr* to *from_discr*.

    .. attribute:: from_discr
    .. attribute:: to_discr
    .. attribute:: is_surjective

    .. attribute:: conn
    .. automethod:: __call__

    """

    def __new__(cls, connections, is_surjective=False):
        if isinstance(connections, DirectDiscretizationConnection):
            return DiscretizationConnection.__new__(cls)
        elif isinstance(connections, ChainedDiscretizationConnection):
            if len(connections.connections) == 0:
                return connections

            return cls(connections.connections, is_surjective=is_surjective)
        else:
            conns = []
            for cnx in reversed(connections):
                conns.append(cls(cnx, is_surjective=is_surjective))

            return ChainedDiscretizationConnection(conns)

    def __init__(self, conn, is_surjective=False):
        if conn.from_discr.dim != conn.to_discr.dim:
            raise RuntimeError("cannot transport from face to element")

        if not all(g.is_orthogonal_basis() for g in conn.to_discr.groups):
            raise RuntimeError("`to_discr` must have an orthogonal basis")

        self.conn = conn
        super().__init__(
                from_discr=self.conn.to_discr,
                to_discr=self.conn.from_discr,
                is_surjective=is_surjective)

    @keyed_memoize_method(key=lambda actx: ())
    def _batch_weights(self, actx):
        """Computes scaled quadrature weights for each interpolation batch in
        :attr:`conn`. The quadrature weights can be used to integrate over
        child elements in the domain of the parent element, by a change of
        variables.

        :return: a dictionary with keys ``(group_id, batch_id)``.
        """

        from pymbolic.geometric_algebra import MultiVector
        from functools import reduce
        from operator import xor

        def det(v):
            nnodes = v[0].shape[0]
            det_v = np.empty(nnodes)

            for i in range(nnodes):
                outer_product = reduce(xor, [MultiVector(x[i, :].T) for x in v])
                det_v[i] = abs((outer_product.I | outer_product).as_scalar())

            return det_v

        weights = {}
        jac = np.empty(self.to_discr.dim, dtype=np.object)

        for igrp, grp in enumerate(self.to_discr.groups):
            for ibatch, batch in enumerate(self.conn.groups[igrp].batches):
                for iaxis in range(grp.dim):
                    mat = grp.diff_matrices()[iaxis]
                    jac[iaxis] = mat.dot(batch.result_unit_nodes.T)

                weights[igrp, ibatch] = actx.freeze(actx.from_numpy(
                    det(jac) * grp.weights))

        return weights

    @obj_array_vectorized_n_args
    def __call__(self, ary):
        if not isinstance(ary, DOFArray):
            raise TypeError("non-array passed to discretization connection")

        if ary.shape != (len(self.from_discr.groups),):
            raise ValueError("invalid shape of incoming resampling data")

        actx = ary.array_context

        @memoize_in(actx, (L2ProjectionInverseDiscretizationConnection,
            "conn_projection_knl"))
        def kproj():
            return make_loopy_program([
                "{[iel]: 0 <= iel < nelements}",
                "{[idof_quad]: 0 <= idof_quad < n_from_nodes}"
                ],
                """
                for iel
                    <> element_dot = sum(idof_quad,
                                ary[from_element_indices[iel], idof_quad]
                                * basis[idof_quad] * weights[idof_quad])

                    result[to_element_indices[iel], ibasis] = \
                            result[to_element_indices[iel], ibasis] + element_dot
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
                name="conn_projection_knl")

        @memoize_in(actx, (L2ProjectionInverseDiscretizationConnection,
            "conn_evaluation_knl"))
        def keval():
            return make_loopy_program([
                "{[iel]: 0 <= iel < nelements}",
                "{[idof]: 0 <= idof < n_to_nodes}",
                "{[ibasis]: 0 <= ibasis < n_to_nodes}"
                ],
                """
                    result[iel, idof] = result[iel, idof] + \
                        sum(ibasis, vdm[idof, ibasis] * coefficients[iel, ibasis])
                """,
                [
                    lp.GlobalArg("coefficients", None,
                        shape=("nelements", "n_to_nodes")),
                    "..."
                    ],
                name="conn_evaluate_knl")

        # compute weights on each refinement of the reference element
        weights = self._batch_weights(actx)

        # perform dot product (on reference element) to get basis coefficients
        c = self.to_discr.zeros(actx, dtype=ary.entry_dtype)

        for igrp, (tgrp, cgrp) in enumerate(
                zip(self.to_discr.groups, self.conn.groups)):
            for ibatch, batch in enumerate(cgrp.batches):
                sgrp = self.from_discr.groups[batch.from_group_index]

                for ibasis, basis_fn in enumerate(sgrp.basis()):
                    basis = actx.from_numpy(
                            basis_fn(batch.result_unit_nodes).flatten())

                    # NOTE: batch.*_element_indices are reversed here because
                    # they are from the original forward connection, but
                    # we are going in reverse here. a bit confusing, but
                    # saves on recreating the connection groups and batches.
                    actx.call_loopy(kproj(),
                            ibasis=ibasis,
                            ary=ary[sgrp.index],
                            basis=basis,
                            weights=weights[igrp, ibatch],
                            result=c[igrp],
                            from_element_indices=batch.to_element_indices,
                            to_element_indices=batch.from_element_indices)

        # evaluate at unit_nodes to get the vector on to_discr
        result = self.to_discr.zeros(actx, dtype=ary.entry_dtype)
        for igrp, grp in enumerate(self.to_discr.groups):
            from modepy import vandermonde
            vdm = actx.from_numpy(vandermonde(grp.basis(), grp.unit_nodes))
            actx.call_loopy(
                    keval(),
                    result=result[grp.index],
                    vdm=vdm,
                    coefficients=c[grp.index])

        return result


# vim: foldmethod=marker
