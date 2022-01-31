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
import numpy.linalg as la

import loopy as lp
from meshmode.transform_metadata import (
        ConcurrentElementInameTag, ConcurrentDOFInameTag,
        IsOpArray, IsDOFArray, ParameterValue)
from pytools import memoize_in, keyed_memoize_method
from arraycontext import (
        ArrayContext, NotAnArrayContainerError,
        serialize_container, deserialize_container, make_loopy_program)


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

        .. note::

            This attribute is not required. It exists only to carry along
            metadata from
            :func:`~meshmode.discretization.connection.make_face_restriction`
            to routines
            that build upon its output, such as
            :func:`~meshmode.discretization.connection.make_opposite_face_connection`.
            If you are not building
            or consuming face restrictions, it is safe to leave this
            unset and/or ignore it. This attribute probably belongs in a subclass,
            but that refactoring hasn't happened yet. (Sorry!)
    """

    def __init__(self, from_group_index, from_element_indices,
            to_element_indices, result_unit_nodes, to_element_face):
        self.from_group_index = from_group_index
        self.from_element_indices = from_element_indices
        self.to_element_indices = to_element_indices
        self.result_unit_nodes = result_unit_nodes
        self.to_element_face = to_element_face
        self._global_from_element_indices_cache = None

    @property
    def nelements(self):
        return len(self.from_element_indices)

    def _global_from_element_indices(self, actx, to_group):
        """Returns a version of :attr:`from_element_indices` that is usable
        without :attr:`to_element_indices`.  Elements for which no 'from'-side
        data exists (the result will be set to zero) are marked with a
        "from-element index" of -1.

        :arg: actx: A :class:`arraycontext.ArrayContext` with which to compute
            the index set if not already available.
        :arg to_group: The :class:`~meshmode.discretization.ElementGroup`
            that holds the result of this interpolation batch.
        """
        if self._global_from_element_indices_cache is not None:
            return self._global_from_element_indices_cache

        # FIXME: This is a workaround for a loopy kernel that was producing
        # incorrect results on some machines (details:
        # https://github.com/inducer/meshmode/pull/255).
        from_element_indices = actx.to_numpy(self.from_element_indices)
        to_element_indices = actx.to_numpy(self.to_element_indices)
        numpy_result = np.full(to_group.nelements, -1)
        numpy_result[to_element_indices] = from_element_indices
        result = actx.freeze(actx.from_numpy(numpy_result))

        self._global_from_element_indices_cache = result
        return result

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
    *   computing modal data from nodal coefficients
    *   computing nodal coefficients from modal data

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
        """Apply the connection. If applicable, may return a view of the data
        instead of a copy, i.e. changes to *ary* may or may not appear
        in the result returned by this method, and vice versa.
        """
        raise NotImplementedError()


class IdentityDiscretizationConnection(DiscretizationConnection):
    """A no-op connection from a :class:`~meshmode.discretization.Discretization`
    to the same discretization that returns the same data unmodified.
    """
    def __init__(self, discr):
        super().__init__(discr, discr, True)

    def __call__(self, ary):
        return ary


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
            from_grp_basis_fcts = from_grp.basis_obj().functions
            if len(from_grp_basis_fcts) != nfrom_unit_nodes:
                from meshmode.discretization import NoninterpolatoryElementGroupError
                raise NoninterpolatoryElementGroupError(
                        "%s does not support interpolation because it is not "
                        "unisolvent (its unit node count does not match its "
                        "number of basis functions). Using connections requires "
                        "the ability to interpolate." % type(from_grp).__name__)

            result = mp.resampling_matrix(
                    from_grp_basis_fcts,
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

        ibatch = self.groups[to_group_index].batches[ibatch_index]
        from_grp = self.from_discr.groups[ibatch.from_group_index]

        if tol_multiplier is None:
            tol_multiplier = 250

        tol = np.finfo(ibatch.result_unit_nodes.dtype).eps * tol_multiplier

        dim, ntgt_nodes = ibatch.result_unit_nodes.shape
        if dim == 0:
            assert ntgt_nodes == 1
            return actx.freeze(actx.from_numpy(np.array([0], dtype=np.int32)))

        dist_vecs = (ibatch.result_unit_nodes.reshape(dim, -1, 1)
                - from_grp.unit_nodes.reshape(dim, 1, -1))
        dists = la.norm(dist_vecs, axis=0, ord=2)

        result = np.zeros(ntgt_nodes, dtype=self.to_discr.mesh.element_id_dtype)

        for irow in range(ntgt_nodes):
            close_indices, = np.where(dists[irow] < tol)

            if len(close_indices) != 1:
                return None

            close_index, = close_indices
            result[irow] = close_index

        return actx.freeze(actx.from_numpy(result))

    def full_resample_matrix(self, actx):
        from warnings import warn
        warn("This method is deprecated. Use 'make_direct_full_resample_matrix' "
                "instead.", DeprecationWarning, stacklevel=2)

        return make_direct_full_resample_matrix(actx, self)

    def __call__(self, ary, _force_no_inplace_updates=False):
        """
        :arg ary: a :class:`~meshmode.dof_array.DOFArray`, or an
            :class:`arraycontext.ArrayContainer` of them, containing nodal
            coefficient data on :attr:`from_discr`.

        :arg _force_no_inplace_updates: private argument only used to ensure
            test coverge of all code paths.
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
                    (key, self(subary, _force_no_inplace_updates))
                    for key, subary in iterable
                    ])

        # }}}

        if __debug__:
            from meshmode.dof_array import check_dofarray_against_discr
            check_dofarray_against_discr(self.from_discr, ary)

        if (ary.array_context.permits_inplace_modification
                and not _force_no_inplace_updates):
            return self._apply_with_inplace_updates(ary)
        else:
            return self._apply_without_inplace_updates(ary)

    # {{{ _apply_without_inplace_updates

    def _apply_without_inplace_updates(self, ary):
        from meshmode.dof_array import DOFArray
        actx = ary.array_context

        @memoize_in(actx,
                (DirectDiscretizationConnection, "resample_by_mat_knl"))
        def batch_mat_knl():
            t_unit = make_loopy_program(
                [
                    "{[iel]: 0 <= iel < nelements}",
                    "{[idof]: 0 <= idof < n_to_nodes}",
                    "{[jdof]: 0 <= jdof < n_from_nodes}"
                ],
                """
                # https://github.com/inducer/loopy/issues/427
                if from_element_indices[iel] != -1
                    <> rowres = sum(jdof, resample_mat[idof, jdof]
                            * ary[from_element_indices[iel], jdof])
                end
                result[iel, idof] =  rowres if from_element_indices[iel] != -1 else 0
                """,
                [
                    lp.GlobalArg("ary", None,
                        shape="nelements_vec, n_from_nodes",
                        offset=lp.auto),
                    lp.ValueArg("nelements_vec", np.int32),
                    "...",
                ],
                name="resample_by_mat",
            )
            return lp.tag_inames(t_unit, {
                "iel": ConcurrentElementInameTag(),
                "idof": ConcurrentDOFInameTag()})

        @memoize_in(actx,
                (DirectDiscretizationConnection, "resample_by_picking_knl"))
        def batch_pick_knl():
            t_unit = make_loopy_program(
                [
                    "{[iel]: 0 <= iel < nelements}",
                    "{[idof]: 0 <= idof < n_to_nodes}"
                ],
                """
                    result[iel, idof] = (
                        ary[from_element_indices[iel], pick_list[idof]]
                        if from_element_indices[iel] != -1 else 0)
                """,
                [
                    lp.GlobalArg("ary", None,
                        shape="nelements_vec, n_from_nodes",
                        offset=lp.auto),
                    lp.ValueArg("nelements_vec", np.int32),
                    lp.ValueArg("n_from_nodes", np.int32),
                    "...",
                ],
                name="resample_by_picking",
            )
            return lp.tag_inames(t_unit, {
                "iel": ConcurrentElementInameTag(),
                "idof": ConcurrentDOFInameTag()})

        group_data = []
        for i_tgrp, cgrp in enumerate(self.groups):
            # Loop over each batch in a group and evaluate the
            # batch-contribution
            batched_data = []
            for i_batch, batch in enumerate(cgrp.batches):
                if not len(batch.from_element_indices):
                    continue

                point_pick_indices = self._resample_point_pick_indices(
                        actx, i_tgrp, i_batch)

                if point_pick_indices is None:
                    from_element = actx.thaw(batch._global_from_element_indices(
                        actx,
                        self.to_discr.groups[i_tgrp]))
                    grp_ary = ary[batch.from_group_index]
                    mat = self._resample_matrix(actx, i_tgrp, i_batch)
                    if actx.permits_advanced_indexing:
                        batch_result = actx.np.where(
                                            actx.np.not_equal(from_element
                                                              .reshape(-1, 1),
                                                              -1),
                                            actx.einsum("ij,ej->ei",
                                                        mat,
                                                        grp_ary[from_element]),
                                            0)
                    else:
                        batch_result = actx.call_loopy(
                            batch_mat_knl(),
                            resample_mat=mat,
                            ary=grp_ary,
                            from_element_indices=from_element,
                            n_to_nodes=self.to_discr.groups[i_tgrp].nunit_dofs
                        )["result"]

                else:
                    from_vec = ary[batch.from_group_index]
                    from_element_indices = actx.thaw(
                        batch._global_from_element_indices(
                            actx, self.to_discr.groups[i_tgrp])
                        )
                    pick_list = actx.thaw(point_pick_indices)

                    if actx.permits_advanced_indexing:
                        batch_result = actx.np.where(
                            actx.np.not_equal(from_element_indices.reshape((-1, 1)),
                                              -1),
                            from_vec[from_element_indices.reshape((-1, 1)),
                                     pick_list],
                            0)
                    else:
                        batch_result = actx.call_loopy(
                            batch_pick_knl(),
                            pick_list=pick_list,
                            ary=from_vec,
                            from_element_indices=from_element_indices,
                            n_to_nodes=self.to_discr.groups[i_tgrp].nunit_dofs
                        )["result"]

                batched_data.append(batch_result)

            # After computing each batched result, take the sum
            # to get the entire contribution over the group
            if batched_data:
                group_data.append(sum(batched_data))
            else:
                # If no batched data at all, return zeros for this
                # particular group array
                group_data.append(
                    actx.zeros(
                        shape=(self.to_discr.groups[i_tgrp].nelements,
                               self.to_discr.groups[i_tgrp].nunit_dofs),
                        dtype=ary.entry_dtype
                    )
                )

        return DOFArray(actx, data=tuple(group_data))

    # }}}

    # {{{ _apply_with_inplace_updates

    def _apply_with_inplace_updates(self, ary):
        actx = ary.array_context

        @memoize_in(actx, (DirectDiscretizationConnection,
            "resample_by_mat_knl_inplace"))
        def mat_knl(nelements_vec, nelements_result, n_to_nodes, n_from_nodes,
                result_dtype, rmat_dtype, ary_dtype):
            t_unit = make_loopy_program(
                """{[iel, idof, j]:
                    0<=iel<nelements and
                    0<=idof<n_to_nodes and
                    0<=j<n_from_nodes}""",
                "result[to_element_indices[iel], idof] \
                    = sum(j, resample_mat[idof, j] \
                    * ary[from_element_indices[iel], j])",
                [
                    lp.GlobalArg("result", result_dtype,
                        shape="nelements_result, n_to_nodes",
                        offset=lp.auto, tags=[IsDOFArray()]),
                    lp.GlobalArg("resample_mat", rmat_dtype,
                        shape="n_to_nodes, n_from_nodes",
                        offset=lp.auto, tags=[IsOpArray()]),
                    lp.GlobalArg("ary", ary_dtype,
                        shape="nelements_vec, n_from_nodes",
                        offset=lp.auto, tags=[IsDOFArray()]),
                    lp.ValueArg("n_to_nodes", np.int32,
                        tags=[ParameterValue(n_to_nodes)]),
                    lp.ValueArg("n_from_nodes", np.int32,
                        tags=[ParameterValue(n_from_nodes)]),
                    lp.ValueArg("nelements_result", np.int32,
                        tags=[ParameterValue(nelements_result)]),
                    lp.ValueArg("nelements_vec", np.int32,
                        tags=[ParameterValue(nelements_vec)]),
                    "...",
                    ],
                name="resample_by_mat_inplace")

            return lp.tag_inames(t_unit, {
                "iel": ConcurrentElementInameTag(),
                "idof": ConcurrentDOFInameTag()})

        @memoize_in(actx,
                (DirectDiscretizationConnection, "resample_by_picking_knl_inplace"))
        def pick_knl(n_to_nodes):
            t_unit = make_loopy_program(
                """{[iel, idof]:
                    0<=iel<nelements and
                    0<=idof<n_to_nodes}""",
                "result[to_element_indices[iel], idof] \
                    = ary[from_element_indices[iel], pick_list[idof]]",
                [
                    lp.GlobalArg("result", None,
                        shape="nelements_result, n_to_nodes",
                        offset=lp.auto, tags=[IsDOFArray()]),
                    lp.GlobalArg("ary", None,
                        shape="nelements_vec, n_from_nodes",
                        offset=lp.auto, tags=[IsDOFArray()]),
                    lp.ValueArg("nelements_result", np.int32),
                    lp.ValueArg("nelements_vec", np.int32),
                    lp.ValueArg("n_from_nodes", np.int32),
                    lp.ValueArg("n_to_nodes", np.int32,
                        tags=[ParameterValue(n_to_nodes)]),
                    "...",
                    ],
                name="resample_by_picking_inplace")

            return lp.tag_inames(t_unit, {
                "iel": ConcurrentElementInameTag(),
                "idof": ConcurrentDOFInameTag()})

        if self.is_surjective:
            result = self.to_discr.empty(actx, dtype=ary.entry_dtype)
        else:
            result = self.to_discr.zeros(actx, dtype=ary.entry_dtype)

        for i_tgrp, cgrp in enumerate(self.groups):
            for i_batch, batch in enumerate(cgrp.batches):
                if not len(batch.from_element_indices):
                    continue

                point_pick_indices = self._resample_point_pick_indices(
                        actx, i_tgrp, i_batch)

                if point_pick_indices is None:
                    resample_mat = self._resample_matrix(actx, i_tgrp, i_batch)
                    n_to_nodes, n_from_nodes = resample_mat.shape
                    nelements_result, _ = result[i_tgrp].shape
                    nelements_vec, _ = ary[batch.from_group_index].shape
                    result_dtype = result[i_tgrp].dtype
                    rmat_dtype = resample_mat.dtype
                    ary_dtype = ary[batch.from_group_index].dtype

                    actx.call_loopy(mat_knl(nelements_vec, nelements_result,
                                n_to_nodes, n_from_nodes,
                                result_dtype, rmat_dtype, ary_dtype),
                            resample_mat=resample_mat,
                            result=result[i_tgrp],
                            ary=ary[batch.from_group_index],
                            from_element_indices=batch.from_element_indices,
                            to_element_indices=batch.to_element_indices)

                else:
                    nelements, n_to_nodes = result[i_tgrp].shape
                    actx.call_loopy(pick_knl(n_to_nodes),
                            pick_list=point_pick_indices,
                            result=result[i_tgrp],
                            ary=ary[batch.from_group_index],
                            from_element_indices=batch.from_element_indices,
                            to_element_indices=batch.to_element_indices)

        return result

    # }}}

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

    :arg actx: an :class:`~arraycontext.ArrayContext`.
    :arg conn: a :class:`DirectDiscretizationConnection`.
    """

    if not isinstance(conn, DirectDiscretizationConnection):
        raise TypeError("can only construct a full resampling matrix "
                "for a DirectDiscretizationConnection.")

    @memoize_in(actx, (make_direct_full_resample_matrix, "oversample_mat_knl"))
    def knl():
        return make_loopy_program(
            [
                "{[idof_init]: 0 <= idof_init < nnodes_tgt}",
                "{[jdof_init]: 0 <= jdof_init < nnodes_src}",
                "{[iel]: 0 <= iel < nelements}",
                "{[idof]: 0 <= idof < n_to_nodes}",
                "{[jdof]: 0 <= jdof < n_from_nodes}"
            ],
            """
                result[idof_init, jdof_init] = 0 {id=init}
                ... gbarrier {id=barrier, dep=init}
                result[itgt_base + to_element_indices[iel]*n_to_nodes + idof,      \
                       isrc_base + from_element_indices[iel]*n_from_nodes + jdof]  \
                           = resample_mat[idof, jdof] {dep=barrier}
            """,
            [
                lp.GlobalArg("result", None,
                    shape="nnodes_tgt, nnodes_src",
                    offset=lp.auto, tags=[IsDOFArray()]),
                lp.ValueArg("itgt_base, isrc_base", np.int32),
                lp.ValueArg("nnodes_tgt, nnodes_src", np.int32),
                "...",
            ],
            name="oversample_mat"
        )

    to_discr_ndofs = sum(grp.nelements*grp.nunit_dofs
            for grp in conn.to_discr.groups)
    from_discr_ndofs = sum(grp.nelements*grp.nunit_dofs
            for grp in conn.from_discr.groups)

    from_group_sizes = [
            grp.nelements*grp.nunit_dofs
            for grp in conn.from_discr.groups]
    from_group_starts = np.cumsum([0] + from_group_sizes)

    tgt_node_nr_base = 0
    mats = []
    for i_tgrp, (tgrp, cgrp) in enumerate(
            zip(conn.to_discr.groups, conn.groups)):
        for i_batch, batch in enumerate(cgrp.batches):
            if not len(batch.from_element_indices):
                continue

            mats.append(
                actx.call_loopy(
                    knl(),
                    resample_mat=conn._resample_matrix(actx, i_tgrp, i_batch),
                    itgt_base=tgt_node_nr_base,
                    isrc_base=from_group_starts[batch.from_group_index],
                    from_element_indices=batch.from_element_indices,
                    to_element_indices=batch.to_element_indices,
                    nnodes_tgt=to_discr_ndofs,
                    nnodes_src=from_discr_ndofs,
                )["result"]
            )

        tgt_node_nr_base += tgrp.nelements*tgrp.nunit_dofs

    return sum(mats)

# }}}

# vim: foldmethod=marker
