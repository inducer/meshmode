from __future__ import division, print_function, absolute_import

__copyright__ = """
Copyright (C) 2014 Andreas Kloeckner
Copyright (C) 2018 Alexandru Fikl
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

from six.moves import range, zip

import numpy as np
import pyopencl as cl
import pyopencl.array  # noqa

from loopy.version import MOST_RECENT_LANGUAGE_VERSION
from pytools import Record
from pytools import memoize_method, memoize_in

import modepy as mp

from meshmode.discretization.connection.same_mesh import \
        make_same_mesh_connection
from meshmode.discretization.connection.face import (
        FACE_RESTR_INTERIOR, FACE_RESTR_ALL,
        make_face_restriction, make_face_to_all_faces_embedding)
from meshmode.discretization.connection.opposite_face import \
        make_opposite_face_connection
from meshmode.discretization.connection.refinement import \
        make_refinement_connection


import logging
logger = logging.getLogger(__name__)


__all__ = [
        "DiscretizationConnection",
        "make_same_mesh_connection",
        "FACE_RESTR_INTERIOR", "FACE_RESTR_ALL",
        "make_face_restriction",
        "make_face_to_all_faces_embedding",
        "make_opposite_face_connection",
        "make_refinement_connection"
        ]

__doc__ = """
.. autoclass:: DiscretizationConnection
.. autoclass:: DirectDiscretizationConnection

.. autofunction:: make_same_mesh_connection

.. autofunction:: FACE_RESTR_INTERIOR
.. autofunction:: FACE_RESTR_ALL

.. autofunction:: make_face_restriction
.. autofunction:: make_face_to_all_faces_embedding

.. autofunction:: make_opposite_face_connection

.. autofunction:: make_refinement_connection

Implementation details
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: InterpolationBatch

.. autoclass:: DiscretizationConnectionElementGroup
"""


class _ConnectionGroupData(Record):
    pass


# {{{ interpolation batch

class InterpolationBatch(object):
    """One interpolation batch captures how a batch of elements *within* an
    element group should be an interpolated. Note that while it's possible that
    an interpolation batch takes care of interpolating an entire element group
    from source to target, that's not *necessarily* the case. Consider the case
    of extracting boundary values of a discretization. For, say, a triangle, at
    least three different interpolation batches are needed to cover boundary
    edges that fall onto each of the three edges of the unit triangle.

    .. attribute:: from_group_index

        An integer indicating from which element group in the *from* discretization
        the data should be interpolated.

    .. attribute:: from_element_indices

        ``element_id_t [nelements]``. (a :class:`pyopencl.array.Array`)
        This contains the (group-local) element index (relative to
        :attr:`from_group_index` from which this "*to*" element's data will be
        interpolated.

    .. attribute:: to_element_indices

        ``element_id_t [nelements]``. (a :class:`pyopencl.array.Array`)
        This contains the (group-local) element index to which this "*to*"
        element's data will be interpolated.

    .. attribute:: result_unit_nodes

        A :class:`numpy.ndarray` of shape
        ``(from_group.dim,to_group.nunit_nodes)``
        storing the coordinates of the nodes (in unit coordinates
        of the *from* reference element) from which the node
        locations of this element should be interpolated.

    .. autoattribute:: nelements

    .. attribute:: to_element_face

        *int* or *None*. (a :class:`pyopencl.array.Array` if existent) If this
        interpolation batch targets interpolation *to* a face, then this number
        captures the face number (on all elements referenced by
        :attr:`from_element_indices` to which this batch interpolates. (Since
        there is a fixed set of "from" unit nodes per batch, one batch will
        always go to a single face index.)
    """

    def __init__(self, from_group_index, from_element_indices,
            to_element_indices, result_unit_nodes, to_element_face):
        self.from_group_index = from_group_index
        self.from_element_indices = from_element_indices
        self.to_element_indices = to_element_indices
        self.result_unit_nodes = result_unit_nodes
        self.to_element_face = to_element_face

    @property
    def nelements(self):
        return len(self.from_element_indices)

# }}}


# {{{ connection element group

class DiscretizationConnectionElementGroup(object):
    """
    .. attribute:: batches

        A list of :class:`InterpolationBatch` instances.
    """
    def __init__(self, batches):
        self.batches = batches

# }}}


# {{{ connection classes

class DiscretizationConnection(object):
    """Abstract interface for transporting a DOF vector from one
    :class:`meshmode.discretization.Discretization` to another.
    Possible applications include:

    *   upsampling/downsampling on the same mesh
    *   restricition to the boundary
    *   interpolation to a refined/coarsened mesh
    *   interpolation onto opposing faces

    .. attribute:: from_discr

    .. attribute:: to_discr

    .. attribute:: is_surjective

        A :class:`bool` indicating whether every output degree
        of freedom is set by the connection.

    .. automethod:: __call__
    """
    def __init__(self, from_discr, to_discr, is_surjective):
        if from_discr.cl_context != to_discr.cl_context:
            raise ValueError("from_discr and to_discr must live in the "
                    "same OpenCL context")

        self.cl_context = from_discr.cl_context

        if from_discr.mesh.vertex_id_dtype != to_discr.mesh.vertex_id_dtype:
            raise ValueError("from_discr and to_discr must agree on the "
                    "vertex_id_dtype")

        if from_discr.mesh.element_id_dtype != to_discr.mesh.element_id_dtype:
            raise ValueError("from_discr and to_discr must agree on the "
                    "element_id_dtype")

        self.from_discr = from_discr
        self.to_discr = to_discr

        self.is_surjective = is_surjective

    def full_resample_matrix(self):
        raise NotImplementedError()

    def direct_connection(self):
        raise NotImplementedError()

    def __call__(self, queue, vec):
        raise NotImplementedError()


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

        super(ChainedDiscretizationConnection, self).__init__(
                from_discr, to_discr, is_surjective=is_surjective)

        self.connections = connections

    def full_resample_matrix(self, queue):
        if not self.connections:
            result = cl.array.to_device(queue, np.eye(self.to_discr.nnodes))
            return result

        acc = self.connections[0].full_resample_matrix(queue).get(queue)
        for cnx in self.connections[1:]:
            resampler = cnx.full_resample_matrix(queue).get(queue)
            acc = resampler.dot(acc)

        return cl.array.to_device(queue, acc)

    def _element_to_batch(self, queue, conn):
        nelements = np.sum([g.nelements for g in conn.from_discr.groups])
        el_map = [[] for _ in range(nelements)]

        for i, igrp in enumerate(conn.groups):
            for j, ibatch in enumerate(igrp.batches):
                for p, k in enumerate(ibatch.from_element_indices.get(queue)):
                    el_map[k].append((i, j, p))

        return el_map

    def _merge_groups(self, from_discr, from_groups, to_conn):
        def iterbatches(groups):
            for igrp, group in enumerate(groups):
                for ibatch, batch in enumerate(group.batches):
                    yield (igrp, ibatch), (group, batch)

        def find_batch(nodes, gtb):
            for igrp, batches in enumerate(gtb):
                for ibatch, batch in enumerate(batches):
                    if np.allclose(nodes, batch.result_unit_nodes):
                        return (igrp, ibatch)
            return (-1, -1)

        nfrom_groups = len(from_groups)
        nto_groups = len(to_conn.groups)

        grp_to_grp = {}
        grp_to_batch = [[] for i in range(nfrom_groups * nto_groups)]
        for (igrp, ibatch), (fgrp, fbatch) in iterbatches(from_groups):
            for (jgrp, jbatch), (tgrp, tbatch) in iterbatches(to_conn.groups):
                # compute result_unit_nodes
                ffgrp = from_discr.groups[fbatch.from_group_index]
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
                (mgroup, mbatch) = find_batch(result_unit_nodes, grp_to_batch)
                if mgroup < 0:
                    mgroup = nto_groups * igrp + jgrp

                    grp_to_batch[mgroup].append(_ConnectionGroupData(
                        from_group_index=fbatch.from_group_index,
                        from_batch_index=(igrp, ibatch),
                        to_batch_index=(jgrp, jbatch),
                        result_unit_nodes=result_unit_nodes,
                        to_element_face=tbatch.to_element_face))
                    mbatch = len(grp_to_batch[mgroup]) - 1

                grp_to_grp[(igrp, ibatch, jgrp, jbatch)] = (mgroup, mbatch)

        return grp_to_grp, grp_to_batch

    def _construct_batches(self, queue, from_bins, to_bins, grp_to_batch):
        def to_device(x):
            return cl.array.to_device(queue, np.asarray(x))

        for ibatch, (from_bin, to_bin) in enumerate(zip(from_bins, to_bins)):
            info = grp_to_batch[ibatch]

            yield InterpolationBatch(
                    from_group_index=info.from_group_index,
                    from_element_indices=to_device(from_bin),
                    to_element_indices=to_device(to_bin),
                    result_unit_nodes=info.result_unit_nodes,
                    to_element_face=info.to_element_face)

    def direct_connection(self, queue):
        def iterbatches(groups):
            for igrp, grp in enumerate(groups):
                for ibatch, batch in enumerate(grp.batches):
                    yield (igrp, ibatch), (grp, batch)

        if not self.connections:
            return make_same_mesh_connection(self.to_discr, self.from_discr)

        connections = []
        for conn in self.connections:
            connections.append(conn.direct_connection())

        groups = connections[0].groups
        for to_conn in connections[1:]:
            # create an inverse mapping for the elements in to_conn.from_discr
            el_to_batch = self._element_to_batch(queue, to_conn)
            # create new group structure to go from groups -> to_conn.to_discr
            grp_to_grp, grp_to_batch = self._merge_groups(self.from_discr,
                                                          groups,
                                                          to_conn)

            # distribute the indices to new groups and batches
            from_bins = [[[] for _ in g] for g in grp_to_batch]
            to_bins = [[[] for _ in g] for g in grp_to_batch]

            for (igrp, ibatch), (_, batch) in iterbatches(groups):
                for ifrom, imid in zip(batch.from_element_indices.get(queue),
                                      batch.to_element_indices.get(queue)):
                    for jgrp, jbatch, iel in el_to_batch[imid]:
                        mgrp, mbatch = grp_to_grp[(igrp, ibatch, jgrp, jbatch)]
                        ito = to_conn.groups[jgrp].batches[jbatch] \
                                    .to_element_indices[iel].get(queue)

                        from_bins[mgrp][mbatch].append(ifrom)
                        to_bins[mgrp][mbatch].append(ito)

            # construct new groups
            groups = []
            for igrp, (from_bin, to_bin) in enumerate(zip(from_bins, to_bins)):
                groups.append(DiscretizationConnectionElementGroup(
                    list(self._construct_batches(queue, from_bin, to_bin,
                                                 grp_to_batch[igrp]))))

        return DirectDiscretizationConnection(
            from_discr=self.from_discr,
            to_discr=self.to_discr,
            groups=groups,
            is_surjective=self.is_surjective)

    def __call__(self, queue, vec):
        for cnx in self.connections:
            vec = cnx(queue, vec)

        return vec


class DirectDiscretizationConnection(DiscretizationConnection):
    """A concrete :class:`DiscretizationConnection` supported by interpolation
    data.

    .. attribute:: from_discr

    .. attribute:: to_discr

    .. attribute:: groups

        a list of :class:`DiscretizationConnectionElementGroup`
        instances, with a one-to-one correspondence to the groups in
        :attr:`to_discr`.

    .. attribute:: is_surjective

        A :class:`bool` indicating whether every output degree
        of freedom is set by the connection.

    .. automethod:: __call__

    .. automethod:: full_resample_matrix

    """

    def __init__(self, from_discr, to_discr, groups, is_surjective):
        super(DirectDiscretizationConnection, self).__init__(
                from_discr, to_discr, is_surjective)

        self.groups = groups

    @memoize_method
    def _resample_matrix(self, to_group_index, ibatch_index):
        ibatch = self.groups[to_group_index].batches[ibatch_index]
        from_grp = self.from_discr.groups[ibatch.from_group_index]

        nfrom_unit_nodes = from_grp.unit_nodes.shape[1]
        if np.array_equal(from_grp.unit_nodes, ibatch.result_unit_nodes):
            # Nodes are exactly identical? We can 'interpolate' even when there
            # isn't a basis.

            result = np.eye(nfrom_unit_nodes)

        else:
            if len(from_grp.basis()) != nfrom_unit_nodes:
                from meshmode.discretization import NoninterpolatoryElementGroupError
                raise NoninterpolatoryElementGroupError(
                        "%s does not support interpolation because it is not "
                        "unisolvent (its unit node count does not match its "
                        "number of basis functions). Using connections requires "
                        "the ability to interpolate." % type(from_grp).__name__)

            result = mp.resampling_matrix(
                    from_grp.basis(),
                    ibatch.result_unit_nodes, from_grp.unit_nodes)

        with cl.CommandQueue(self.cl_context) as queue:
            return cl.array.to_device(queue, result).with_queue(None)

    @memoize_method
    def _resample_point_pick_indices(self, to_group_index, ibatch_index,
            tol_multiplier=None):
        """If :meth:`_resample_matrix` *R* is a row subset of a permutation matrix *P*,
        return the index subset I so that, loosely, ``x[I] == R @ x``.

        Will return *None* if no such index array exists, or a
        :class:`pyopencl.array.Array` containing the index subset.
        """

        with cl.CommandQueue(self.cl_context) as queue:
            mat = self._resample_matrix(to_group_index, ibatch_index).get(
                    queue=queue)

        nrows, ncols = mat.shape
        result = np.zeros(nrows, dtype=self.to_discr.mesh.element_id_dtype)

        if tol_multiplier is None:
            tol_multiplier = 50

        tol = np.finfo(mat.dtype).eps * tol_multiplier

        for irow in range(nrows):
            one_indices, = np.where(np.abs(mat[irow] - 1) < tol)
            zero_indices, = np.where(np.abs(mat[irow]) < tol)

            if len(one_indices) != 1:
                return None
            if len(zero_indices) != ncols - 1:
                return None

            one_index, = one_indices
            result[irow] = one_index

        with cl.CommandQueue(self.cl_context) as queue:
            return cl.array.to_device(queue, result).with_queue(None)

    def full_resample_matrix(self, queue):
        """Build a dense matrix representing this discretization connection.

        .. warning::

            On average, this will be exceedingly expensive (:math:`O(N^2)` in
            the number *N* of discretization points) in terms of memory usage
            and thus not what you'd typically want.
        """

        @memoize_in(self, "oversample_mat_knl")
        def knl():
            import loopy as lp
            knl = lp.make_kernel(
                """{[k,i,j]:
                    0<=k<nelements and
                    0<=i<n_to_nodes and
                    0<=j<n_from_nodes}""",
                "result[itgt_base + to_element_indices[k]*n_to_nodes + i, \
                        isrc_base + from_element_indices[k]*n_from_nodes + j] \
                    = resample_mat[i, j]",
                [
                    lp.GlobalArg("result", None,
                        shape="nnodes_tgt, nnodes_src",
                        offset=lp.auto),
                    lp.ValueArg("itgt_base,isrc_base", np.int32),
                    lp.ValueArg("nnodes_tgt,nnodes_src", np.int32),
                    "...",
                    ],
                name="oversample_mat",
                lang_version=MOST_RECENT_LANGUAGE_VERSION)

            knl = lp.split_iname(knl, "i", 16, inner_tag="l.0")
            return lp.tag_inames(knl, dict(k="g.0"))

        result = cl.array.zeros(
                queue,
                (self.to_discr.nnodes, self.from_discr.nnodes),
                dtype=self.to_discr.real_dtype)

        for i_tgrp, (tgrp, cgrp) in enumerate(
                zip(self.to_discr.groups, self.groups)):
            for i_batch, batch in enumerate(cgrp.batches):
                if not len(batch.from_element_indices):
                    continue

                sgrp = self.from_discr.groups[batch.from_group_index]

                knl()(queue,
                      resample_mat=self._resample_matrix(i_tgrp, i_batch),
                      result=result,
                      itgt_base=tgrp.node_nr_base,
                      isrc_base=sgrp.node_nr_base,
                      from_element_indices=batch.from_element_indices,
                      to_element_indices=batch.to_element_indices)

        return result

    def direct_connection(self):
        return self

    def __call__(self, queue, vec):
        @memoize_in(self, "resample_by_mat_knl")
        def mat_knl():
            import loopy as lp
            knl = lp.make_kernel(
                """{[k,i,j]:
                    0<=k<nelements and
                    0<=i<n_to_nodes and
                    0<=j<n_from_nodes}""",
                "result[to_element_indices[k], i] \
                    = sum(j, resample_mat[i, j] \
                    * vec[from_element_indices[k], j])",
                [
                    lp.GlobalArg("result", None,
                        shape="nelements_result, n_to_nodes",
                        offset=lp.auto),
                    lp.GlobalArg("vec", None,
                        shape="nelements_vec, n_from_nodes",
                        offset=lp.auto),
                    lp.ValueArg("nelements_result", np.int32),
                    lp.ValueArg("nelements_vec", np.int32),
                    "...",
                    ],
                name="resample_by_mat",
                lang_version=MOST_RECENT_LANGUAGE_VERSION)

            knl = lp.split_iname(knl, "i", 16, inner_tag="l.0")
            return lp.tag_inames(knl, dict(k="g.0"))

        @memoize_in(self, "resample_by_picking_knl")
        def pick_knl():
            import loopy as lp
            knl = lp.make_kernel(
                """{[k,i,j]:
                    0<=k<nelements and
                    0<=i<n_to_nodes}""",
                "result[to_element_indices[k], i] \
                    = vec[from_element_indices[k], pick_list[i]]",
                [
                    lp.GlobalArg("result", None,
                        shape="nelements_result, n_to_nodes",
                        offset=lp.auto),
                    lp.GlobalArg("vec", None,
                        shape="nelements_vec, n_from_nodes",
                        offset=lp.auto),
                    lp.ValueArg("nelements_result", np.int32),
                    lp.ValueArg("nelements_vec", np.int32),
                    lp.ValueArg("n_from_nodes", np.int32),
                    "...",
                    ],
                name="resample_by_picking",
                lang_version=MOST_RECENT_LANGUAGE_VERSION)

            knl = lp.split_iname(knl, "i", 16, inner_tag="l.0")
            return lp.tag_inames(knl, dict(k="g.0"))

        if not isinstance(vec, cl.array.Array):
            raise TypeError("non-array passed to discretization connection")

        if self.is_surjective:
            result = self.to_discr.empty(dtype=vec.dtype)
        else:
            result = self.to_discr.zeros(queue, dtype=vec.dtype)

        if vec.shape != (self.from_discr.nnodes,):
            raise ValueError("invalid shape of incoming resampling data")

        for i_tgrp, (tgrp, cgrp) in enumerate(
                zip(self.to_discr.groups, self.groups)):
            for i_batch, batch in enumerate(cgrp.batches):
                sgrp = self.from_discr.groups[batch.from_group_index]

                if not len(batch.from_element_indices):
                    continue

                point_pick_indices = self._resample_point_pick_indices(
                        i_tgrp, i_batch)

                if point_pick_indices is None:
                    mat_knl()(queue,
                            resample_mat=self._resample_matrix(i_tgrp, i_batch),
                            result=tgrp.view(result), vec=sgrp.view(vec),
                            from_element_indices=batch.from_element_indices,
                            to_element_indices=batch.to_element_indices)

                else:
                    pick_knl()(queue,
                            pick_list=point_pick_indices,
                            result=tgrp.view(result), vec=sgrp.view(vec),
                            from_element_indices=batch.from_element_indices,
                            to_element_indices=batch.to_element_indices)

        return result

# }}}


# {{{ check connection

def check_connection(connection):
    from_discr = connection.from_discr
    to_discr = connection.to_discr

    assert len(connection.groups) == len(to_discr.groups)

    with cl.CommandQueue(to_discr.cl_context) as queue:
        for cgrp, tgrp in zip(connection.groups, to_discr.groups):
            for batch in cgrp.batches:
                fgrp = from_discr.groups[batch.from_group_index]

                from_element_indices = batch.from_element_indices.get(queue)
                to_element_indices = batch.to_element_indices.get(queue)

                assert (0 <= from_element_indices).all()
                assert (0 <= to_element_indices).all()
                assert (from_element_indices < fgrp.nelements).all()
                assert (to_element_indices < tgrp.nelements).all()
                if batch.to_element_face is not None:
                    assert 0 <= batch.to_element_face < fgrp.mesh_el_group.nfaces

# }}}


# vim: foldmethod=marker
