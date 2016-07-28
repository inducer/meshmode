# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

__copyright__ = "Copyright (C) 2014 Andreas Kloeckner"

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
from pytools import memoize_method, memoize_in

from meshmode.discretization.connection.same_mesh import \
        make_same_mesh_connection
from meshmode.discretization.connection.face import (
        FRESTR_INTERIOR_FACES, FRESTR_ALL_FACES,
        make_face_restriction, make_face_to_all_faces_embedding)
from meshmode.discretization.connection.opposite_face import \
        make_opposite_face_connection

import logging
logger = logging.getLogger(__name__)


__all__ = [
        "DiscretizationConnection",
        "make_same_mesh_connection",
        "FRESTR_INTERIOR_FACES", "FRESTR_ALL_FACES",
        "make_face_restriction",
        "make_face_to_all_faces_embedding",
        "make_opposite_face_connection"
        ]

__doc__ = """
.. autoclass:: DiscretizationConnection

.. autofunction:: make_same_mesh_connection

.. autofunction:: FRESTR_INTERIOR_FACES
.. autofunction:: FRESTR_ALL_FACES
.. autofunction:: make_face_restriction
.. autofunction:: make_face_to_all_faces_embedding

.. autofunction:: make_opposite_face_connection

Implementation details
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: InterpolationBatch

.. autoclass:: DiscretizationConnectionElementGroup
"""


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


# {{{ connection class

class DiscretizationConnection(object):
    """Data supporting an interpolation-like operation that takes in data on
    one discretization and returns it on another. Implemented applications
    include:

    *   upsampling/downsampling on the same mesh
    *   restricition to the boundary
    *   interpolation to a refined/coarsened mesh
    *   interpolation onto opposing faces

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

        self.cl_context = from_discr.cl_context

        self.from_discr = from_discr
        self.to_discr = to_discr
        self.groups = groups

        self.is_surjective = is_surjective

    @memoize_method
    def _resample_matrix(self, to_group_index, ibatch_index):
        import modepy as mp
        ibatch = self.groups[to_group_index].batches[ibatch_index]
        from_grp = self.from_discr.groups[ibatch.from_group_index]

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
                name="oversample_mat")

            knl = lp.split_iname(knl, "i", 16, inner_tag="l.0")
            return lp.tag_inames(knl, dict(k="g.0"))

        result = cl.array.zeros(
                queue,
                (self.to_discr.nnodes, self.from_discr.nnodes),
                dtype=self.to_discr.real_dtype)

        for i_tgrp, (tgrp, cgrp) in enumerate(
                zip(self.to_discr.groups, self.groups)):
            for i_batch, batch in enumerate(cgrp.batches):
                if len(batch.from_element_indices):
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
                name="resample_by_mat")

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
                name="resample_by_picking")

            knl = lp.split_iname(knl, "i", 16, inner_tag="l.0")
            return lp.tag_inames(knl, dict(k="g.0"))

        if not isinstance(vec, cl.array.Array):
            return vec

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


# {{{ refinement connection


def _map_unit_nodes_to_children(unit_nodes, tesselation):
    """
    Given a collection of unit nodes, return the coordinates of the
    unit nodes mapped onto each of the children of the reference
    element.

    The tesselation should follow the format of
    :func:`meshmode.mesh.tesselate.tesselatetri()` or
    :func:`meshmode.mesh.tesselate.tesselatetet()`.

    `unit_nodes` should be relative to the unit simplex coordinates in
    :module:`modepy`.

    :arg unit_nodes: shaped `(dim, nunit_nodes)`
    :arg tesselation: With attributes `ref_vertices`, `children`
    """
    ref_vertices = np.array(tesselation.ref_vertices, dtype=np.float)

    for child_element in tesselation.children:
        center = np.vstack(ref_vertices[child_element[0]])
        # Scale by 1/2 since sides in the tesselation have length 2.
        aff_mat = (ref_vertices.T[:, child_element[1:]] - center) / 2
        # (-1, -1, ...) in unit_nodes = (0, 0, ...) in ref_vertices.
        # Hence the translation by +/- 1.
        yield aff_mat.dot(unit_nodes + 1) + center - 1


def _build_interpolation_batches_for_group(
        queue, group_idx, coarse_discr_group, fine_discr_group, record):
    """
    To map between discretizations, we sort each of the fine mesh
    elements into an interpolation batch.  Which batch they go
    into is determined by where the refined unit nodes live
    relative to the coarse reference element.

    For instance, consider the following refinement:

     ______      ______
    |\     |    |\    e|
    | \    |    |d\    |
    |  \   |    |__\   |
    |   \  | => |\c|\  |
    |    \ |    |a\|b\ |
    |     \|    |  |  \|
     ‾‾‾‾‾‾      ‾‾‾‾‾‾

    Here, the discretization unit nodes for elements a,b,c,d,e
    will each have different positions relative to the reference
    element, so each element gets its own batch. On the other
    hand, for

     ______      ______
    |\     |    |\ f|\e|
    | \    |    |d\ |g\|
    |  \   |    |__\|__|
    |   \  | => |\c|\  |
    |    \ |    |a\|b\h|
    |     \|    |  |  \|
     ‾‾‾‾‾‾      ‾‾‾‾‾‾

    the pairs {a,e}, {b,f}, {c,g}, {d,h} can share interpolation
    batches because their unit nodes are mapped from the same part
    of the reference element.
    """
    num_children = len(record.tesselation.children) \
                   if record.tesselation else 0
    from_bins = [[] for i in range(1 + num_children)]
    to_bins = [[] for i in range(1 + num_children)]
    for elt_idx, refinement_result in enumerate(record.element_mapping):
        if len(refinement_result) == 1:
            # Not refined -> interpolates to self
            from_bins[0].append(elt_idx)
            to_bins[0].append(refinement_result[0])
        else:
            assert len(refinement_result) == num_children
            # Refined -> interpolates to children
            for from_bin, to_bin, child_idx in zip(
                    from_bins[1:], to_bins[1:], refinement_result):
                from_bin.append(elt_idx)
                to_bin.append(child_idx)

    fine_unit_nodes = fine_discr_group.unit_nodes
    mapped_unit_nodes = _map_unit_nodes_to_children(
        fine_unit_nodes, record.tesselation)

    from itertools import chain
    for from_bin, to_bin, unit_nodes in zip(
            from_bins, to_bins,
            chain([fine_unit_nodes], mapped_unit_nodes)):
        if not from_bin:
            continue
        yield InterpolationBatch(
            from_group_index=group_idx,
            from_element_indices=cl.array.to_device(queue, np.asarray(from_bin)),
            to_element_indices=cl.array.to_device(queue, np.asarray(to_bin)),
            result_unit_nodes=unit_nodes,
            to_element_face=None)


def make_refinement_connection(refiner, coarse_discr, group_factory):
    """
    :arg refiner: An instance of :class:`meshmode.mesh.refinement.Refiner`

    :arg coarse_discr: An instance of
        :class:`meshmode.mesh.discretization.Discretization`

    :arg group_factory: An instance of
        :class:`meshmode.mesh.discretization.ElementGroupFactory`. Used for
        discretizing the fine mesh.

    :return: A :class:`DiscretizationConnection` mapping `coarse_discr` to a
        discretization on the fine mesh
    """
    coarse_mesh = refiner.get_previous_mesh()
    fine_mesh = refiner.last_mesh
    assert coarse_discr.mesh is coarse_mesh

    from meshmode.discretization import Discretization
    fine_discr = Discretization(
        coarse_discr.cl_context,
        fine_mesh,
        group_factory,
        real_dtype=coarse_discr.real_dtype)

    logger.info("building refinement connection: start")

    groups = []
    with cl.CommandQueue(fine_discr.cl_context) as queue:
        for group_idx, (coarse_discr_group, fine_discr_group, record) in \
                enumerate(zip(coarse_discr.groups, fine_discr.groups,
                              refiner.group_refinement_records)):
            groups.append(
                DiscretizationConnectionElementGroup(
                    list(_build_interpolation_batches_for_group(
                            queue, group_idx, coarse_discr_group,
                            fine_discr_group, record))))

    logger.info("building refinement connection: done")

    return DiscretizationConnection(
        from_discr=coarse_discr,
        to_discr=fine_discr,
        groups=groups,
        is_surjective=True)

# }}}

# vim: foldmethod=marker
