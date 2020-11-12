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

from pytools import Record
from pytools.obj_array import obj_array_vectorized_n_args

import modepy as mp
from meshmode.discretization.connection.direct import \
        DiscretizationConnection


# {{{ chained discretization connection

class ChainedDiscretizationConnection(DiscretizationConnection):
    """Aggregates multiple :class:`DiscretizationConnection` instances
    into a single one.

    .. attribute:: connections
    """

    def __init__(self, connections, from_discr=None):
        if connections:
            if from_discr is not None:
                assert from_discr is connections[0].from_discr
            else:
                from_discr = connections[0].from_discr
            is_surjective = all(cnx.is_surjective for cnx in connections)
            to_discr = connections[-1].to_discr
        else:
            if from_discr is None:
                raise ValueError("connections may not be empty if from_discr "
                        "is not specified")

            to_discr = from_discr

            # It's an identity
            is_surjective = True

        super().__init__(
                from_discr, to_discr, is_surjective=is_surjective)

        self.connections = connections

    @obj_array_vectorized_n_args
    def __call__(self, ary):
        for cnx in self.connections:
            ary = cnx(ary)

        return ary

# }}}


# {{{ flatten chained connection

class _ConnectionBatchData(Record):
    pass


def _iterbatches(groups):
    for igrp, grp in enumerate(groups):
        for ibatch, batch in enumerate(grp.batches):
            yield (igrp, ibatch), (grp, batch)


def _build_element_lookup_table(actx, conn):
    el_table = [np.full(g.nelements, -1, dtype=np.int)
                for g in conn.to_discr.groups]

    for (igrp, _), (_, batch) in _iterbatches(conn.groups):
        el_table[igrp][actx.to_numpy(batch.to_element_indices)] = \
                actx.to_numpy(batch.from_element_indices)

    return el_table


def _build_new_group_table(from_conn, to_conn):
    def find_batch(nodes, gtb):
        for igrp, batches in enumerate(gtb):
            for ibatch, batch in enumerate(batches):
                if np.allclose(nodes, batch.result_unit_nodes):
                    return (igrp, ibatch)
        return (-1, -1)

    nfrom_groups = len(from_conn.groups)
    nto_groups = len(to_conn.groups)

    # construct a table from (old groups) -> (new groups)
    # NOTE: we try to reduce the number of new groups and batches by matching
    # the `result_unit_nodes` and only adding a new batch if necessary
    grp_to_grp = {}
    batch_info = [[] for i in range(nfrom_groups * nto_groups)]
    for (igrp, ibatch), (fgrp, fbatch) in _iterbatches(from_conn.groups):
        for (jgrp, jbatch), (tgrp, tbatch) in _iterbatches(to_conn.groups):
            # compute result_unit_nodes
            ffgrp = from_conn.from_discr.groups[fbatch.from_group_index]
            from_matrix = mp.resampling_matrix(
                    ffgrp.basis(),
                    fbatch.result_unit_nodes,
                    ffgrp.unit_nodes)
            result_unit_nodes = from_matrix.dot(ffgrp.unit_nodes.T)

            tfgrp = to_conn.from_discr.groups[tbatch.from_group_index]
            to_matrix = mp.resampling_matrix(
                    tfgrp.basis(),
                    tbatch.result_unit_nodes,
                    tfgrp.unit_nodes)
            result_unit_nodes = to_matrix.dot(result_unit_nodes).T

            # find new (group, batch)
            (igrp_new, ibatch_new) = find_batch(result_unit_nodes, batch_info)
            if igrp_new < 0:
                igrp_new = nto_groups * igrp + jgrp
                ibatch_new = len(batch_info[igrp_new])

                batch_info[igrp_new].append(_ConnectionBatchData(
                    from_group_index=fbatch.from_group_index,
                    result_unit_nodes=result_unit_nodes,
                    to_element_face=tbatch.to_element_face))

            grp_to_grp[igrp, ibatch, jgrp, jbatch] = (igrp_new, ibatch_new)

    return grp_to_grp, batch_info


def _build_batches(actx, from_bins, to_bins, batch):
    from meshmode.discretization.connection.direct import \
            InterpolationBatch

    def to_device(x):
        return actx.freeze(actx.from_numpy(np.asarray(x)))

    for ibatch, (from_bin, to_bin) in enumerate(zip(from_bins, to_bins)):
        yield InterpolationBatch(
                from_group_index=batch[ibatch].from_group_index,
                from_element_indices=to_device(from_bin),
                to_element_indices=to_device(to_bin),
                result_unit_nodes=batch[ibatch].result_unit_nodes,
                to_element_face=batch[ibatch].to_element_face)


def flatten_chained_connection(actx, connection):
    """Collapse a connection into a direct connection.

    If the given connection is already a
    :class:`~meshmode.discretization.connection.DirectDiscretizationConnection`
    nothing is done. However, if the connection is a
    :class:`~meshmode.discretization.connection.ChainedDiscretizationConnection`,
    a new direct connection is constructed that transports from
    :attr:`~meshmode.discretization.connection.DiscretizationConnection.from_discr`
    to
    :attr:`~meshmode.discretization.connection.DiscretizationConnection.to_discr`.

    The new direct connection will have a number of groups and batches that
    is, at worse, the product of all the connections in the chain. For
    example, if we consider a connection between a discretization and a
    two-level refinement, both levels will have :math:`n` groups and
    :math:`m + 1` batches per group, where :math:`m` is the number of
    subdivisions of an element (exact number depends on implementation
    details in
    :func:`~meshmode.discretization.connection.make_refinement_connection`).
    However, a direct connection from level :math:`0` to level :math:`2`
    will have at worst :math:`n^2` groups and each group will have
    :math:`(m + 1)^2` batches.

    .. warning::

        If a large number of connections is chained, the number of groups and
        batches can become very large.

    :arg actx: An instance of :class:`meshmode.array_context.ArrayContext`.
    :arg connection: An instance of
        :class:`~meshmode.discretization.connection.DiscretizationConnection`.
    :return: An instance of
        :class:`~meshmode.discretization.connection.DirectDiscretizationConnection`.
    """
    from meshmode.discretization.connection import (
            DirectDiscretizationConnection,
            DiscretizationConnectionElementGroup,
            make_same_mesh_connection)

    if not hasattr(connection, "connections"):
        return connection

    if not connection.connections:
        return make_same_mesh_connection(actx, connection.to_discr,
                                         connection.from_discr)

    # recursively build direct connections
    connections = connection.connections
    direct_connections = []
    for conn in connections:
        direct_connections.append(flatten_chained_connection(actx, conn))

    # merge all the direct connections
    from_conn = direct_connections[0]
    for to_conn in direct_connections[1:]:
        el_table = _build_element_lookup_table(actx, from_conn)
        grp_to_grp, batch_info = _build_new_group_table(from_conn, to_conn)

        # distribute the indices to new groups and batches
        from_bins = [[np.empty(0, dtype=np.int) for _ in g] for g in batch_info]
        to_bins = [[np.empty(0, dtype=np.int) for _ in g] for g in batch_info]

        for (igrp, ibatch), (_, from_batch) in _iterbatches(from_conn.groups):
            from_to_element_indices = actx.to_numpy(from_batch.to_element_indices)

            for (jgrp, jbatch), (_, to_batch) in _iterbatches(to_conn.groups):
                igrp_new, ibatch_new = grp_to_grp[igrp, ibatch, jgrp, jbatch]

                jfrom = actx.to_numpy(to_batch.from_element_indices)
                jto = actx.to_numpy(to_batch.to_element_indices)

                mask = np.isin(jfrom, from_to_element_indices)
                from_bins[igrp_new][ibatch_new] = \
                    np.hstack([from_bins[igrp_new][ibatch_new],
                               el_table[igrp][jfrom[mask]]])
                to_bins[igrp_new][ibatch_new] = \
                    np.hstack([to_bins[igrp_new][ibatch_new],
                               jto[mask]])

        # build new groups
        groups = []
        for igrp, (from_bin, to_bin) in enumerate(zip(from_bins, to_bins)):
            groups.append(DiscretizationConnectionElementGroup(
                list(_build_batches(actx, from_bin, to_bin,
                                    batch_info[igrp]))))

        from_conn = DirectDiscretizationConnection(
            from_discr=from_conn.from_discr,
            to_discr=to_conn.to_discr,
            groups=groups,
            is_surjective=connection.is_surjective)

    return from_conn

# }}}


# {{{ build chained resample matrix

def make_full_resample_matrix(actx, connection):
    """Build a dense matrix representing the discretization connection.

    This is based on
    :func:`~meshmode.discretization.connection.direct.make_direct_full_resample_matrix`.
    If a chained connection is given, the matrix is constructed recursively
    for each connection and multiplied left to right.

    .. warning::

        This method will be very slow, both in terms of speed and memory
        usage, and should only be used for testing or if absolutely necessary.

    :arg actx: a :class:`meshmode.array_context.ArrayContext`.
    :arg connection: a
        :class:`~meshmode.discretization.connection.DiscretizationConnection`.
    :return: a :class:`pyopencl.array.Array` of shape
        `(connection.from_discr.ndofs, connection.to_discr.ndofs)`.
    """
    from meshmode.discretization.connection.direct import \
            DirectDiscretizationConnection, make_direct_full_resample_matrix

    if isinstance(connection, DirectDiscretizationConnection):
        return make_direct_full_resample_matrix(actx, connection)

    if not isinstance(connection, ChainedDiscretizationConnection):
        raise TypeError("only 'ChainedDiscretizationConnection's are supported")

    if not connection.connections:
        result = np.eye(connection.to_discr.ndofs)
        return actx.from_numpy(result)

    acc = actx.to_numpy(
            make_full_resample_matrix(actx, connection.connections[0]))
    for conn in connection.connections[1:]:
        resampler = actx.to_numpy(make_full_resample_matrix(actx, conn))
        acc = resampler @ acc

    return actx.from_numpy(acc)

# }}}


# vim: foldmethod=marker
