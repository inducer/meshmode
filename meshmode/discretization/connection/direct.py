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


import numpy as np

import loopy as lp
from pytools import memoize_in, keyed_memoize_method
from pytools.obj_array import obj_array_vectorized_n_args
from meshmode.array_context import ArrayContext, make_loopy_program


# {{{ interpolation batch

class InterpolationBatch:
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

        *int* or *None*. If this
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

class DiscretizationConnectionElementGroup:
    """
    .. attribute:: batches

        A list of :class:`InterpolationBatch` instances.
    """
    def __init__(self, batches):
        self.batches = batches

# }}}


# {{{ connection classes

class DiscretizationConnection:
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
        if from_discr.mesh.vertex_id_dtype != to_discr.mesh.vertex_id_dtype:
            raise ValueError("from_discr and to_discr must agree on the "
                    "vertex_id_dtype")

        if from_discr.mesh.element_id_dtype != to_discr.mesh.element_id_dtype:
            raise ValueError("from_discr and to_discr must agree on the "
                    "element_id_dtype")

        self.from_discr = from_discr
        self.to_discr = to_discr

        self.is_surjective = is_surjective

    def __call__(self, ary):
        raise NotImplementedError()


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

    """

    def __init__(self, from_discr, to_discr, groups, is_surjective):
        super().__init__(
                from_discr, to_discr, is_surjective)

        self.groups = groups

    @keyed_memoize_method(key=lambda actx, to_group_index, ibatch_index:
            (to_group_index, ibatch_index))
    def _resample_matrix(self, actx: ArrayContext, to_group_index, ibatch_index):
        import modepy as mp
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

        return actx.freeze(actx.from_numpy(result))

    @keyed_memoize_method(lambda actx, to_group_index, ibatch_index,
            tol_multiplier=None: (to_group_index, ibatch_index, tol_multiplier))
    def _resample_point_pick_indices(self, actx: ArrayContext,
            to_group_index, ibatch_index,
            tol_multiplier=None):
        """If :meth:`_resample_matrix` *R* is a row subset of a permutation matrix *P*,
        return the index subset I so that, loosely, ``x[I] == R @ x``.

        Will return *None* if no such index array exists, or a
        :class:`pyopencl.array.Array` containing the index subset.
        """

        mat = actx.to_numpy(actx.thaw(
                self._resample_matrix(actx, to_group_index, ibatch_index)))

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

        return actx.freeze(actx.from_numpy(result))

    def full_resample_matrix(self, actx):
        from warnings import warn
        warn("This method is deprecated. Use 'make_direct_full_resample_matrix' "
                "instead.", DeprecationWarning, stacklevel=2)

        return make_direct_full_resample_matrix(actx, self)

    @obj_array_vectorized_n_args
    def __call__(self, ary):
        from meshmode.dof_array import DOFArray
        if not isinstance(ary, DOFArray):
            raise TypeError("non-array passed to discretization connection")

        if ary.shape != (len(self.from_discr.groups),):
            raise ValueError("invalid shape of incoming resampling data")

        actx = ary.array_context

        @memoize_in(actx, (DirectDiscretizationConnection, "resample_by_mat_knl"))
        def mat_knl():
            knl = make_loopy_program(
                """{[iel, idof, j]:
                    0<=iel<nelements and
                    0<=idof<n_to_nodes and
                    0<=j<n_from_nodes}""",
                "result[to_element_indices[iel], idof] \
                    = sum(j, resample_mat[idof, j] \
                    * ary[from_element_indices[iel], j])",
                [
                    lp.GlobalArg("result", None,
                        shape="nelements_result, n_to_nodes",
                        offset=lp.auto),
                    lp.GlobalArg("ary", None,
                        shape="nelements_vec, n_from_nodes",
                        offset=lp.auto),
                    lp.ValueArg("nelements_result", np.int32),
                    lp.ValueArg("nelements_vec", np.int32),
                    "...",
                    ],
                name="resample_by_mat")

            return knl

        @memoize_in(actx,
                (DirectDiscretizationConnection, "resample_by_picking_knl"))
        def pick_knl():
            knl = make_loopy_program(
                """{[iel, idof]:
                    0<=iel<nelements and
                    0<=idof<n_to_nodes}""",
                "result[to_element_indices[iel], idof] \
                    = ary[from_element_indices[iel], pick_list[idof]]",
                [
                    lp.GlobalArg("result", None,
                        shape="nelements_result, n_to_nodes",
                        offset=lp.auto),
                    lp.GlobalArg("ary", None,
                        shape="nelements_vec, n_from_nodes",
                        offset=lp.auto),
                    lp.ValueArg("nelements_result", np.int32),
                    lp.ValueArg("nelements_vec", np.int32),
                    lp.ValueArg("n_from_nodes", np.int32),
                    "...",
                    ],
                name="resample_by_picking")

            return knl

        if self.is_surjective:
            result = self.to_discr.empty(actx, dtype=ary.entry_dtype)
        else:
            result = self.to_discr.zeros(actx, dtype=ary.entry_dtype)

        for i_tgrp, (tgrp, cgrp) in enumerate(
                zip(self.to_discr.groups, self.groups)):
            for i_batch, batch in enumerate(cgrp.batches):
                if not len(batch.from_element_indices):
                    continue

                point_pick_indices = self._resample_point_pick_indices(
                        actx, i_tgrp, i_batch)

                if point_pick_indices is None:
                    actx.call_loopy(mat_knl(),
                            resample_mat=self._resample_matrix(
                                actx, i_tgrp, i_batch),
                            result=result[i_tgrp],
                            ary=ary[batch.from_group_index],
                            from_element_indices=batch.from_element_indices,
                            to_element_indices=batch.to_element_indices)

                else:
                    actx.call_loopy(pick_knl(),
                            pick_list=point_pick_indices,
                            result=result[i_tgrp],
                            ary=ary[batch.from_group_index],
                            from_element_indices=batch.from_element_indices,
                            to_element_indices=batch.to_element_indices)

        return result

# }}}


# {{{ dense resampling matrix

def make_direct_full_resample_matrix(actx, conn):
    """Build a dense matrix representing this discretization connection.

    .. warning::

        On average, this will be exceedingly expensive (:math:`O(N^2)` in
        the number *N* of discretization points) in terms of memory usage
        and thus not what you'd typically want, other than maybe for
        testing.

    .. note::

        This function assumes a flattened DOF array, as produced by
        :class:`~meshmode.dof_array.flatten`.

    :arg actx: an :class:`~meshmode.array_context.ArrayContext`.
    :arg conn: a :class:`DirectDiscretizationConnection`.
    """

    if not isinstance(conn, DirectDiscretizationConnection):
        raise TypeError("can only construct a full resampling matrix "
                "for a DirectDiscretizationConnection.")

    @memoize_in(actx, (make_direct_full_resample_matrix, "oversample_mat_knl"))
    def knl():
        return make_loopy_program(
            """{[iel, idof, j]:
                0<=iel<nelements and
                0<=idof<n_to_nodes and
                0<=j<n_from_nodes}""",
            "result[itgt_base + to_element_indices[iel]*n_to_nodes + idof, \
                    isrc_base + from_element_indices[iel]*n_from_nodes + j] \
                = resample_mat[idof, j]",
            [
                lp.GlobalArg("result", None,
                    shape="nnodes_tgt, nnodes_src",
                    offset=lp.auto),
                lp.ValueArg("itgt_base,isrc_base", np.int32),
                lp.ValueArg("nnodes_tgt,nnodes_src", np.int32),
                "...",
                ],
            name="oversample_mat")

    to_discr_ndofs = sum(grp.nelements*grp.nunit_dofs
            for grp in conn.to_discr.groups)
    from_discr_ndofs = sum(grp.nelements*grp.nunit_dofs
            for grp in conn.from_discr.groups)

    result = actx.zeros(
            (to_discr_ndofs, from_discr_ndofs),
            dtype=conn.to_discr.real_dtype)

    from_group_sizes = [
            grp.nelements*grp.nunit_dofs
            for grp in conn.from_discr.groups]
    from_group_starts = np.cumsum([0] + from_group_sizes)

    tgt_node_nr_base = 0
    for i_tgrp, (tgrp, cgrp) in enumerate(
            zip(conn.to_discr.groups, conn.groups)):
        for i_batch, batch in enumerate(cgrp.batches):
            if not len(batch.from_element_indices):
                continue

            actx.call_loopy(knl(),
                    resample_mat=conn._resample_matrix(actx, i_tgrp, i_batch),
                    result=result,
                    itgt_base=tgt_node_nr_base,
                    isrc_base=from_group_starts[batch.from_group_index],
                    from_element_indices=batch.from_element_indices,
                    to_element_indices=batch.to_element_indices)

        tgt_node_nr_base += tgrp.nelements*tgrp.nunit_dofs

    return result

# }}}

# vim: foldmethod=marker
