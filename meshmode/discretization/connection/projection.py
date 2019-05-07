from __future__ import division, print_function, absolute_import

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
import pyopencl as cl
import pyopencl.array  # noqa

from loopy.version import MOST_RECENT_LANGUAGE_VERSION
from pytools import memoize_method, memoize_in

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
        super(L2ProjectionInverseDiscretizationConnection, self).__init__(
                from_discr=self.conn.to_discr,
                to_discr=self.conn.from_discr,
                is_surjective=is_surjective)

    @memoize_method
    def _batch_weights(self):
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

                weights[igrp, ibatch] = det(jac) * grp.weights

        return weights

    def __call__(self, queue, vec):
        @memoize_in(self, "conn_projection_knl")
        def kproj():
            import loopy as lp
            knl = lp.make_kernel([
                "{[k]: 0 <= k < nelements}",
                "{[j]: 0 <= j < n_from_nodes}"
                ],
                """
                for k
                    <> element_dot = \
                            sum(j, vec[from_element_indices[k], j] * \
                                   basis[j] * weights[j])

                    result[to_element_indices[k], ibasis] = \
                            result[to_element_indices[k], ibasis] + element_dot
                end
                """,
                [
                    lp.GlobalArg("vec", None,
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
                    '...'
                    ],
                name="conn_projection_knl",
                lang_version=MOST_RECENT_LANGUAGE_VERSION)

            return knl

        @memoize_in(self, "conn_evaluation_knl")
        def keval():
            import loopy as lp
            knl = lp.make_kernel([
                "{[k]: 0 <= k < nelements}",
                "{[j]: 0 <= j < n_to_nodes}"
                ],
                """
                    result[k, j] = result[k, j] + \
                            coefficients[k, ibasis] * basis[j]
                """,
                [
                    lp.GlobalArg("coefficients", None,
                        shape=("nelements", "n_to_nodes")),
                    lp.ValueArg("ibasis", np.int32),
                    '...'
                    ],
                name="conn_evaluate_knl",
                default_offset=lp.auto,
                lang_version=MOST_RECENT_LANGUAGE_VERSION)

            return knl

        if not isinstance(vec, cl.array.Array):
            raise TypeError("non-array passed to discretization connection")

        if vec.shape != (self.from_discr.nnodes,):
            raise ValueError("invalid shape of incoming resampling data")

        # compute weights on each refinement of the reference element
        weights = self._batch_weights()

        # perform dot product (on reference element) to get basis coefficients
        c = self.to_discr.zeros(queue, dtype=vec.dtype)
        for igrp, (tgrp, cgrp) in enumerate(
                zip(self.to_discr.groups, self.conn.groups)):
            for ibatch, batch in enumerate(cgrp.batches):
                sgrp = self.from_discr.groups[batch.from_group_index]

                for ibasis, basis_fn in enumerate(sgrp.basis()):
                    basis = basis_fn(batch.result_unit_nodes).flatten()

                    # NOTE: batch.*_element_indices are reversed here because
                    # they are from the original forward connection, but
                    # we are going in reverse here. a bit confusing, but
                    # saves on recreating the connection groups and batches.
                    kproj()(queue,
                            ibasis=ibasis,
                            vec=sgrp.view(vec),
                            basis=basis,
                            weights=weights[igrp, ibatch],
                            result=tgrp.view(c),
                            from_element_indices=batch.to_element_indices,
                            to_element_indices=batch.from_element_indices)

        # evaluate at unit_nodes to get the vector on to_discr
        result = self.to_discr.zeros(queue, dtype=vec.dtype)
        for igrp, grp in enumerate(self.to_discr.groups):
            for ibasis, basis_fn in enumerate(grp.basis()):
                basis = basis_fn(grp.unit_nodes).flatten()

                keval()(queue,
                        ibasis=ibasis,
                        result=grp.view(result),
                        basis=basis,
                        coefficients=grp.view(c))

        return result


# vim: foldmethod=marker
