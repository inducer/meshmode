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


import logging
from typing import Any

import numpy as np
import numpy.linalg as la

import loopy as lp
from meshmode.transform_metadata import (
        ConcurrentElementInameTag, ConcurrentDOFInameTag)
from pytools import memoize_in, keyed_memoize_method
from arraycontext import (
        ArrayContext, make_loopy_program,
        is_array_container, map_array_container)

from dataclasses import dataclass

logger = logging.getLogger(__name__)


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


# {{{ _FromGroupPickData

@dataclass
class _FromGroupPickData:
    """Represents information needed to pick DOFs from one source element
    group to a target element group. Note that the connection between these
    groups must be such that the information transfer can occur by indirect
    access, no interpolation can occur. Each target element's DOFs can be read
    from the source element via a different "pick list", however, chosen from
    :attr:`dof_pick_lists` via :attr:`dof_pick_list_index`. The information
    typically summarizes multiple :class:`InterpolationBatch`es.

    .. attribute:: from_group_index

        The element group index in the
        :attr:`DirectDiscretizationConnection.from_discr` from which information
        is retrieved.

    .. attribute:: dof_pick_lists

        A frozen array of shape ``(npick_lists, ntgt_dofs)`` of a type controlled
        by the array context.

    .. attribute:: dof_pick_list_index
        A frozen array of shape ``(nelements_tgt)`` of a type controlled
        by the array context, indicating which pick list each element should use.

    .. attribute:: from_element_indices

        An frozen array of shape ``(nelements_tgt)`` of a type controlled
        by the array context, indicating from which source element each target
        element should gather its data.

    .. attribute:: is_surjective
    """

    from_group_index: int
    dof_pick_lists: Any
    dof_pick_list_index: Any
    from_element_indices: Any
    is_surjective: bool

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


# {{{ connection base class

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

# }}}


# {{{ identity connection

class IdentityDiscretizationConnection(DiscretizationConnection):
    """A no-op connection from a :class:`~meshmode.discretization.Discretization`
    to the same discretization that returns the same data unmodified.
    """
    def __init__(self, discr):
        super().__init__(discr, discr, True)

    def __call__(self, ary):
        return ary

# }}}


# {{{ direct connection

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
        self._global_point_pick_info_cache = None

    # {{{ _resample_matrix

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

    # }}}

    # {{{ _resample_point_pick_indices

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

    # }}}

    def full_resample_matrix(self, actx):
        from warnings import warn
        warn("This method is deprecated. Use 'make_direct_full_resample_matrix' "
                "instead.", DeprecationWarning, stacklevel=2)

        return make_direct_full_resample_matrix(actx, self)

    # {{{ __call__

    def __call__(self, ary, _force_no_inplace_updates=False):
        # _force_no_inplace_updates: Only used to ensure test coverage
        # of both code paths.

        from meshmode.dof_array import DOFArray
        if is_array_container(ary) and not isinstance(ary, DOFArray):
            return map_array_container(self, ary)

        if not isinstance(ary, DOFArray):
            raise TypeError("non-array passed to discretization connection")

        if ary.shape != (len(self.from_discr.groups),):
            raise ValueError("invalid shape of incoming resampling data")

        if (ary.array_context.permits_inplace_modification
                and not _force_no_inplace_updates):
            return self._apply_with_inplace_updates(ary)
        else:
            return self._apply_without_inplace_updates(ary)

    # }}}

    # {{{ _global_point_pick_info_cache

    def _per_target_group_pick_info(self, actx, i_tgrp):
        """Returns a list of :class:`_FromGroupPickData`, one per source group
        from which data ist to be transferred, or *None*, if conditions for
        this representation are not met.
        """
        cgrp = self.groups[i_tgrp]
        tgrp = self.to_discr.groups[i_tgrp]

        batch_dof_pick_lists = [
                self._resample_point_pick_indices(actx, i_tgrp, i_batch)
                for i_batch in range(len(cgrp.batches))]

        all_batches_pickable = all(
                bpi is not None for bpi in batch_dof_pick_lists)
        if not all_batches_pickable:
            return None

        batch_dof_pick_lists = [
                actx.to_numpy(pick_list) for pick_list in batch_dof_pick_lists]

        batch_source_groups = sorted({
            batch.from_group_index for batch in cgrp.batches})

        # no source data ta
        if not batch_source_groups:
            return None

        result = []
        for source_group_index in batch_source_groups:
            batch_indices_for_this_source_group = [
                    i for i, batch in enumerate(cgrp.batches)
                    if batch.from_group_index == source_group_index]

            # {{{ find and weed out duplicate dof pick lists

            dof_pick_lists = list({tuple(batch_dof_pick_lists[bi])
                    for bi in batch_indices_for_this_source_group})
            dof_pick_list_to_index = {
                    p_ind: i for i, p_ind in enumerate(dof_pick_lists)}
            # shape: (number of pick lists, nunit_dofs_tgt)
            dof_pick_lists = np.array(dof_pick_lists)

            # }}}

            from_el_indices = np.empty(
                    tgrp.nelements, dtype=self.from_discr.mesh.element_id_dtype)
            from_el_indices.fill(-1)
            dof_pick_list_index = np.zeros(tgrp.nelements, dtype=np.int8)
            assert len(dof_pick_lists)-1 <= np.iinfo(dof_pick_list_index.dtype).max

            for source_batch_index in batch_indices_for_this_source_group:
                source_batch = cgrp.batches[source_batch_index]

                to_el_ind = actx.to_numpy(actx.thaw(source_batch.to_element_indices))
                if (from_el_indices[to_el_ind] != -1).any():
                    from warnings import warn
                    warn("per-batch target elements not disjoint during "
                            "attempted merge")
                    return None

                from_el_indices[to_el_ind] = \
                        actx.to_numpy(actx.thaw(source_batch.from_element_indices))
                dof_pick_list_index[to_el_ind] = \
                        dof_pick_list_to_index[
                                tuple(batch_dof_pick_lists[source_batch_index])]

            result.append(
                    _FromGroupPickData(
                        from_group_index=source_group_index,
                        from_element_indices=actx.freeze(actx.from_numpy(
                            from_el_indices)),
                        dof_pick_lists=actx.freeze(actx.from_numpy(
                            dof_pick_lists)),
                        dof_pick_list_index=actx.freeze(actx.from_numpy(
                            dof_pick_list_index)),
                        is_surjective=(from_el_indices != -1).all()
                        ))

        return result

    def _global_point_pick_info(self, actx):
        """Return a list (of length matching the number of target groups)
        containing *None* or a list of :class:`_FromGroupPickData` instances.
        """

        if self._global_point_pick_info_cache is not None:
            return self._global_point_pick_info_cache

        self._global_point_pick_info_cache = [
                self._per_target_group_pick_info(actx, i_tgrp)
                for i_tgrp in range(len(self.groups))]
        return self._global_point_pick_info_cache

    # }}}

    # {{{ _apply_without_inplace_updates

    def _apply_without_inplace_updates(self, ary):
        from meshmode.dof_array import DOFArray
        actx = ary.array_context

        # {{{ kernels

        @memoize_in(actx,
                (DirectDiscretizationConnection, "resample_by_mat_knl"))
        def batch_mat_knl():
            t_unit = make_loopy_program(
                [
                    "{[iel]: 0 <= iel < nelements}",
                    "{[idof]: 0 <= idof < nunit_dofs_tgt}",
                    "{[jdof]: 0 <= jdof < nunit_dofs_src}"
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
                        shape="nelements_vec, nunit_dofs_src",
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
                (DirectDiscretizationConnection, "resample_by_picking_batch_knl"))
        def batch_pick_knl():
            t_unit = make_loopy_program(
                [
                    "{[iel]: 0 <= iel < nelements}",
                    "{[idof]: 0 <= idof < nunit_dofs_tgt}"
                ],
                """
                    result[iel, idof] = (
                        ary[from_element_indices[iel], pick_list[idof]]
                        if from_element_indices[iel] != -1 else 0)
                """,
                [
                    lp.GlobalArg("ary", None,
                        shape="nelements_vec, nunit_dofs_src",
                        offset=lp.auto),
                    lp.ValueArg("nelements_vec", np.int32),
                    lp.ValueArg("nunit_dofs_src", np.int32),
                    "...",
                ],
                name="resample_by_picking_batch",
            )
            return lp.tag_inames(t_unit, {
                "iel": ConcurrentElementInameTag(),
                "idof": ConcurrentDOFInameTag()})

        @memoize_in(actx,
                (DirectDiscretizationConnection, "resample_by_picking_group_knl"))
        def group_pick_knl():
            t_unit = make_loopy_program(
                [
                    "{[iel]: 0 <= iel < nelements}",
                    "{[idof]: 0 <= idof < nunit_dofs_tgt}"
                ],
                """
                    result[iel, idof] = (
                        ary[
                                from_element_indices[iel],
                                dof_pick_lists[dof_pick_list_index[iel], idof]
                            ]
                        if from_element_indices[iel] != -1 else 0)
                """,
                [
                    lp.GlobalArg("ary", None,
                        shape="nelements_src, nunit_dofs_src",
                        offset=lp.auto),
                    lp.GlobalArg("dof_pick_lists", None,
                        shape="nelements_tgt, nunit_dofs_tgt",
                        offset=lp.auto),
                    lp.ValueArg("nelements_tgt", np.int32),
                    lp.ValueArg("nelements_src", np.int32),
                    lp.ValueArg("nunit_dofs_src", np.int32),
                    "...",
                ],
                name="resample_by_picking_group",
            )
            return lp.tag_inames(t_unit, {
                "iel": ConcurrentElementInameTag(),
                "idof": ConcurrentDOFInameTag()})

        # }}}

        point_pick_info = self._global_point_pick_info(actx)

        group_arrays = []
        for i_tgrp, (cgrp, group_point_pick_info) in enumerate(
                zip(self.groups, point_pick_info)):

            if group_point_pick_info is not None:
                group_array_2_contributions = [
                    actx.call_loopy(
                        group_pick_knl(),
                        dof_pick_lists=fgpd.dof_pick_lists,
                        dof_pick_list_index=fgpd.dof_pick_list_index,
                        ary=ary[fgpd.from_group_index],
                        from_element_indices=fgpd.from_element_indices,
                        nunit_dofs_tgt=self.to_discr.groups[i_tgrp].nunit_dofs
                    )["result"]
                    for fgpd in group_point_pick_info]

                assert group_array_2_contributions
                group_array = sum(group_array_2_contributions)

            elif cgrp.batches:
                # Loop over each batch in a group and evaluate the
                # batch-contribution
                batched_data = []
                for i_batch, batch in enumerate(cgrp.batches):
                    if not len(batch.from_element_indices):
                        continue

                    point_pick_indices = self._resample_point_pick_indices(
                            actx, i_tgrp, i_batch)

                    if point_pick_indices is None:
                        batch_result = actx.call_loopy(
                            batch_mat_knl(),
                            resample_mat=self._resample_matrix(
                                actx, i_tgrp, i_batch
                            ),
                            ary=ary[batch.from_group_index],
                            from_element_indices=batch._global_from_element_indices(
                                actx, self.to_discr.groups[i_tgrp]),
                            nunit_dofs_tgt=self.to_discr.groups[i_tgrp].nunit_dofs
                        )["result"]

                    else:
                        batch_result = actx.call_loopy(
                            batch_pick_knl(),
                            pick_list=point_pick_indices,
                            ary=ary[batch.from_group_index],
                            from_element_indices=batch._global_from_element_indices(
                                actx, self.to_discr.groups[i_tgrp]),
                            nunit_dofs_tgt=self.to_discr.groups[i_tgrp].nunit_dofs
                        )["result"]

                    batched_data.append(batch_result)
                # After computing each batched result, take the sum
                # to get the entire contribution over the group
                group_array = sum(batched_data)

            else:
                # If no batched data at all, return zeros for this
                # particular group array
                group_array = actx.zeros(
                        shape=(self.to_discr.groups[i_tgrp].nelements,
                               self.to_discr.groups[i_tgrp].nunit_dofs),
                        dtype=ary.entry_dtype)

            group_arrays.append(group_array)

        return DOFArray(actx, data=tuple(group_arrays))

    # }}}

    # {{{ _apply_with_inplace_updates

    def _apply_with_inplace_updates(self, ary):
        actx = ary.array_context

        @memoize_in(actx, (DirectDiscretizationConnection,
            "resample_by_mat_knl_inplace"))
        def mat_knl():
            t_unit = make_loopy_program(
                """{[iel, idof, j]:
                    0<=iel<nelements and
                    0<=idof<nunit_dofs_tgt and
                    0<=j<nunit_dofs_src}""",
                "result[to_element_indices[iel], idof] \
                    = sum(j, resample_mat[idof, j] \
                    * ary[from_element_indices[iel], j])",
                [
                    lp.GlobalArg("result", None,
                        shape="nelements_result, nunit_dofs_tgt",
                        offset=lp.auto),
                    lp.GlobalArg("ary", None,
                        shape="nelements_vec, nunit_dofs_src",
                        offset=lp.auto),
                    lp.ValueArg("nelements_result", np.int32),
                    lp.ValueArg("nelements_vec", np.int32),
                    "...",
                    ],
                name="resample_by_mat_inplace")

            return lp.tag_inames(t_unit, {
                "iel": ConcurrentElementInameTag(),
                "idof": ConcurrentDOFInameTag()})

        @memoize_in(actx,
                (DirectDiscretizationConnection, "resample_by_picking_knl_inplace"))
        def pick_knl():
            t_unit = make_loopy_program(
                """{[iel, idof]:
                    0<=iel<nelements and
                    0<=idof<nunit_dofs_tgt}""",
                "result[to_element_indices[iel], idof] \
                    = ary[from_element_indices[iel], pick_list[idof]]",
                [
                    lp.GlobalArg("result", None,
                        shape="nelements_result, nunit_dofs_tgt",
                        offset=lp.auto),
                    lp.GlobalArg("ary", None,
                        shape="nelements_vec, nunit_dofs_src",
                        offset=lp.auto),
                    lp.ValueArg("nelements_result", np.int32),
                    lp.ValueArg("nelements_vec", np.int32),
                    lp.ValueArg("nunit_dofs_src", np.int32),
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
                "{[idof_init]: 0 <= idof_init < ndofs_tgt}",
                "{[jdof_init]: 0 <= jdof_init < ndofs_src}",
                "{[iel]: 0 <= iel < nelements}",
                "{[idof]: 0 <= idof < nunit_dofs_tgt}",
                "{[jdof]: 0 <= jdof < nunit_dofs_src}"
            ],
            """
                result[idof_init, jdof_init] = 0 {id=init}
                ... gbarrier {id=barrier, dep=init}
                result[
                    itgt_base + to_element_indices[iel]*nunit_dofs_tgt + idof,
                    isrc_base + from_element_indices[iel]*nunit_dofs_src + jdof] \
                           = resample_mat[idof, jdof] {dep=barrier}
            """,
            [
                lp.GlobalArg("result", None,
                    shape="ndofs_tgt, ndofs_src",
                    offset=lp.auto),
                lp.ValueArg("itgt_base, isrc_base", np.int32),
                lp.ValueArg("ndofs_tgt, ndofs_src", np.int32),
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
                    ndofs_tgt=to_discr_ndofs,
                    ndofs_src=from_discr_ndofs,
                )["result"]
            )

        tgt_node_nr_base += tgrp.nelements*tgrp.nunit_dofs

    return sum(mats)

# }}}

# vim: foldmethod=marker
