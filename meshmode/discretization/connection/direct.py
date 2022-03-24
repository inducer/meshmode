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
from abc import ABC, abstractmethod

from typing import Sequence, Optional, List, Tuple

import loopy as lp
from meshmode.transform_metadata import (
        ConcurrentElementInameTag, ConcurrentDOFInameTag)
from pytools import memoize_in, keyed_memoize_method
from arraycontext import (
        ArrayContext, NotAnArrayContainerError,
        serialize_container, deserialize_container, make_loopy_program,
        )
from arraycontext.container import ArrayT, ArrayOrContainerT

from meshmode.discretization import Discretization, ElementGroupBase
from meshmode.dof_array import DOFArray

from dataclasses import dataclass


# {{{ interpolation batch

@dataclass
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

        An array of dtype/shape ``element_id_t [nelements]``.
        This contains the (group-local) element index (relative to
        :attr:`from_group_index` from which this "*to*" element's data will be
        interpolated.

    .. attribute:: to_element_indices

        An array of dtype/shape ``element_id_t [nelements]``.
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
            to routines that build upon its output, such as
            :func:`~meshmode.discretization.connection.make_opposite_face_connection`.
            If you are not building or consuming face restrictions, it is safe
            to leave this unset and/or ignore it. This attribute probably
            belongs in a subclass, but that refactoring hasn't happened yet.
            (Sorry!)
    """
    from_group_index: int
    from_element_indices: ArrayT
    to_element_indices: ArrayT
    result_unit_nodes: np.ndarray
    to_element_face: Optional[int]

    def __post_init__(self):
        self._global_from_element_indices_cache: \
                Optional[Tuple[ArrayT, ArrayT]] = None

    @property
    def nelements(self) -> int:
        return len(self.from_element_indices)

    def _global_from_element_indices(
            self, actx: ArrayContext, to_group: ElementGroupBase
            ) -> Tuple[ArrayT, ArrayT]:
        """Returns a version of :attr:`from_element_indices` that is usable
        without :attr:`to_element_indices`, consisting of a tuple.
        The first entry of the tuple is an array of flags indicating
        whether 'from'-side data exists (0 if not), nonzero otherwise,
        and the second is an array that works like :attr:`from_element_indices`.
        In entries where no 'from'-side data exists, the entry in this
        second array will be zero.

        .. note::

            In a prior version of this code, presence and source index were
            contained in a single array, with an invalid value (-1) signifying
            that no data was available on the 'from' side. This turned out
            to be a bad idea: :mod:`pytato` might decide to evaluate ("materialize")
            an indirect access like ``source_data[from_element_indices]``, which
            would lead to out-of-bounds memory accesses.
        """
        if self._global_from_element_indices_cache is not None:
            return self._global_from_element_indices_cache

        # FIXME: This is a workaround for a loopy kernel that was producing
        # incorrect results on some machines (details:
        # https://github.com/inducer/meshmode/pull/255).
        from_element_indices = actx.to_numpy(self.from_element_indices)
        to_element_indices = actx.to_numpy(self.to_element_indices)

        np_full_from_element_indices = np.full(to_group.nelements, -1)
        np_full_from_element_indices[to_element_indices] = from_element_indices
        np_from_el_present = (np_full_from_element_indices != -1)
        np_full_from_element_indices[~np_from_el_present] = 0

        from_el_present = actx.freeze(actx.from_numpy(
            np_from_el_present.astype(np.int8)))
        full_from_element_indices = actx.freeze(
                actx.from_numpy(np_full_from_element_indices))

        self._global_from_element_indices_cache = (
                from_el_present, full_from_element_indices)
        return self._global_from_element_indices_cache

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

    .. attribute:: from_el_present

        Non-zero if source data for this entry is present. Otherwise zero,
        in which case the expected output of the connection for DOFs
        associated with the element is zero.

    .. attribute:: from_element_indices

        An frozen array of shape ``(nelements_tgt)`` of a type controlled
        by the array context, indicating from which source element each target
        element should gather its data.

        .. note::

            In a prior version of this code, presence and source index were
            contained in a single array, with an invalid value (-1) signifying
            that no data was available on the 'from' side. This turned out
            to be a bad idea: :mod:`pytato` might decide to evaluate ("materialize")
            an indirect access like ``source_data[from_element_indices]``, which
            would lead to out-of-bounds memory accesses.

    .. attribute:: is_surjective
    """

    from_group_index: int
    dof_pick_lists: ArrayT
    dof_pick_list_index: ArrayT
    from_el_present: ArrayT
    from_element_indices: ArrayT
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

class DiscretizationConnection(ABC):
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
    def __init__(self, from_discr: Discretization, to_discr: Discretization,
            is_surjective: bool) -> None:
        if from_discr.mesh.vertex_id_dtype != to_discr.mesh.vertex_id_dtype:
            raise ValueError("from_discr and to_discr must agree on the "
                    "vertex_id_dtype")

        if from_discr.mesh.element_id_dtype != to_discr.mesh.element_id_dtype:
            raise ValueError("from_discr and to_discr must agree on the "
                    "element_id_dtype")

        self.from_discr = from_discr
        self.to_discr = to_discr

        self.is_surjective = is_surjective

    @abstractmethod
    def __call__(self, ary: ArrayOrContainerT) -> ArrayOrContainerT:
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
    def __init__(self, discr: Discretization) -> None:
        super().__init__(discr, discr, True)

    def __call__(self, ary: ArrayOrContainerT) -> ArrayOrContainerT:
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

    def __init__(self,
            from_discr: Discretization, to_discr: Discretization,
            groups: Sequence[DiscretizationConnectionElementGroup],
            is_surjective: bool) -> None:
        super().__init__(from_discr, to_discr, is_surjective)

        self.groups = groups
        self._global_point_pick_info_cache = None

    # {{{ _resample_matrix

    @keyed_memoize_method(key=lambda actx, to_group_index, ibatch_index:
            (to_group_index, ibatch_index))
    def _resample_matrix(self, actx: ArrayContext, to_group_index: int,
            ibatch_index: int):
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

    def full_resample_matrix(self, actx: ArrayContext):
        from warnings import warn
        warn("This method is deprecated. Use 'make_direct_full_resample_matrix' "
                "instead.", DeprecationWarning, stacklevel=2)

        return make_direct_full_resample_matrix(actx, self)

    # {{{ _global_point_pick_info_cache

    def _per_target_group_pick_info(
            self, actx: ArrayContext, i_tgrp: int
            ) -> Optional[Sequence[_FromGroupPickData]]:
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

        # no source data
        if not batch_source_groups:
            return None

        result: List[_FromGroupPickData] = []
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

            from_el_present = (from_el_indices != -1)
            from_el_indices[~from_el_present] = 0

            result.append(
                    _FromGroupPickData(
                        from_group_index=source_group_index,
                        dof_pick_lists=actx.freeze(actx.from_numpy(
                            dof_pick_lists)),
                        dof_pick_list_index=actx.freeze(actx.from_numpy(
                            dof_pick_list_index)),
                        from_el_present=actx.freeze(
                            actx.from_numpy(from_el_present.astype(np.int8))),
                        from_element_indices=actx.freeze(actx.from_numpy(
                            from_el_indices)),
                        is_surjective=from_el_present.all()
                        ))

        return result

    def _global_point_pick_info(
            self, actx: ArrayContext
            ) -> Sequence[Optional[Sequence[_FromGroupPickData]]]:
        if self._global_point_pick_info_cache is not None:
            return self._global_point_pick_info_cache

        self._global_point_pick_info_cache = [
                self._per_target_group_pick_info(actx, i_tgrp)
                for i_tgrp in range(len(self.groups))]
        return self._global_point_pick_info_cache

    # }}}

    # {{{ __call__

    def __call__(
            self, ary: ArrayOrContainerT, *,
            _force_use_loopy: bool = False,
            _force_no_merged_batches: bool = False,
            ) -> ArrayOrContainerT:
        """
        :arg ary: a :class:`~meshmode.dof_array.DOFArray`, or an
            :class:`arraycontext.ArrayContainer` of them, containing nodal
            coefficient data on :attr:`from_discr`.

        """
        # _force_use_loopy, _force_no_merged_batches:
        # private arguments only used to ensure test coverge of all code paths.

        # {{{ recurse into array containers

        if not isinstance(ary, DOFArray):
            try:
                iterable = serialize_container(ary)
            except NotAnArrayContainerError:
                pass
            else:
                return deserialize_container(ary, [
                    (key, self(subary,
                        _force_use_loopy=_force_use_loopy,
                        _force_no_merged_batches=_force_no_merged_batches))
                    for key, subary in iterable
                    ])

        # }}}

        if __debug__:
            from meshmode.dof_array import check_dofarray_against_discr
            check_dofarray_against_discr(self.from_discr, ary)

        assert isinstance(ary, DOFArray)

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
                result[iel, idof] = (
                    sum(jdof, resample_mat[idof, jdof]
                            * ary[from_element_indices[iel], jdof])
                    if from_el_present[iel] else 0)
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
                (DirectDiscretizationConnection, "resample_by_picking_knl"))
        def batch_pick_knl():
            t_unit = make_loopy_program(
                [
                    "{[iel]: 0 <= iel < nelements}",
                    "{[idof]: 0 <= idof < nunit_dofs_tgt}"
                ],
                """
                    result[iel, idof] = (
                        ary[from_element_indices[iel], pick_list[idof]]
                        if from_el_present[iel] else 0)
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
        def group_pick_knl(is_surjective: bool):

            if is_surjective:
                if_present = ""
            else:
                if_present = "if from_el_present[iel] else 0"

            t_unit = make_loopy_program(
                [
                    "{[iel]: 0 <= iel < nelements}",
                    "{[idof]: 0 <= idof < nunit_dofs_tgt}"
                ],
                f"""
                    result[iel, idof] = (
                        ary[
                                from_element_indices[iel],
                                dof_pick_lists[dof_pick_list_index[iel], idof]
                            ]
                        { if_present })
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

        group_arrays = []
        for i_tgrp, (cgrp, group_pick_info) in enumerate(
                zip(self.groups, self._global_point_pick_info(actx))):

            group_array_contributions = []

            if _force_no_merged_batches:
                group_pick_info = None

            if group_pick_info is not None:
                group_array_contributions = []

                if actx.permits_advanced_indexing and not _force_use_loopy:
                    for fgpd in group_pick_info:
                        from_element_indices = actx.thaw(fgpd.from_element_indices)

                        grp_ary_contrib = ary[fgpd.from_group_index][
                                    from_element_indices.reshape((-1, 1)),
                                    actx.thaw(fgpd.dof_pick_lists)[
                                        actx.thaw(fgpd.dof_pick_list_index)]
                                    ]

                        if not fgpd.is_surjective:
                            from_el_present = actx.thaw(fgpd.from_el_present)
                            grp_ary_contrib = actx.np.where(
                                from_el_present.reshape((-1, 1)),
                                grp_ary_contrib,
                                0)

                        group_array_contributions.append(grp_ary_contrib)
                else:
                    for fgpd in group_pick_info:
                        group_knl_kwargs = {}
                        if not fgpd.is_surjective:
                            group_knl_kwargs["from_el_present"] = \
                                    fgpd.from_el_present

                        group_array_contributions.append(
                            actx.call_loopy(
                                group_pick_knl(fgpd.is_surjective),
                                dof_pick_lists=fgpd.dof_pick_lists,
                                dof_pick_list_index=fgpd.dof_pick_list_index,
                                ary=ary[fgpd.from_group_index],
                                from_element_indices=fgpd.from_element_indices,
                                nunit_dofs_tgt=(
                                    self.to_discr.groups[i_tgrp].nunit_dofs),
                                **group_knl_kwargs)["result"])

                assert group_array_contributions
                group_array = sum(group_array_contributions)
            elif cgrp.batches:
                for i_batch, batch in enumerate(cgrp.batches):
                    if not len(batch.from_element_indices):
                        continue

                    point_pick_indices = self._resample_point_pick_indices(
                            actx, i_tgrp, i_batch)

                    from_el_present, from_element_indices = \
                            batch._global_from_element_indices(
                                    actx, self.to_discr.groups[i_tgrp])
                    from_el_present = actx.thaw(from_el_present)
                    from_element_indices = actx.thaw(from_element_indices)

                    if point_pick_indices is None:
                        grp_ary = ary[batch.from_group_index]
                        mat = self._resample_matrix(actx, i_tgrp, i_batch)
                        if actx.permits_advanced_indexing and not _force_use_loopy:
                            batch_result = actx.np.where(
                                    from_el_present.reshape(-1, 1),
                                    actx.einsum("ij,ej->ei",
                                        mat, grp_ary[from_element_indices]),
                                    0)
                        else:
                            batch_result = actx.call_loopy(
                                batch_mat_knl(),
                                resample_mat=mat,
                                ary=grp_ary,
                                from_el_present=from_el_present,
                                from_element_indices=from_element_indices,
                                nunit_dofs_tgt=(
                                    self.to_discr.groups[i_tgrp].nunit_dofs)
                            )["result"]

                    else:
                        from_vec = ary[batch.from_group_index]
                        pick_list = actx.thaw(point_pick_indices)

                        if actx.permits_advanced_indexing and not _force_use_loopy:
                            batch_result = actx.np.where(
                                from_el_present.reshape(-1, 1),
                                from_vec[from_element_indices.reshape(
                                    (-1, 1)), pick_list],
                                0)
                        else:
                            batch_result = actx.call_loopy(
                                batch_pick_knl(),
                                pick_list=pick_list,
                                ary=from_vec,
                                from_el_present=from_el_present,
                                from_element_indices=from_element_indices,
                                nunit_dofs_tgt=(
                                    self.to_discr.groups[i_tgrp].nunit_dofs)
                            )["result"]

                    group_array_contributions.append(batch_result)

            if group_array_contributions:
                group_array = sum(group_array_contributions)
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
        :class:`~arraycontext.flatten`.

    :arg actx: an :class:`~arraycontext.ArrayContext`.
    :arg conn: a :class:`DirectDiscretizationConnection`.
    """

    if not isinstance(conn, DirectDiscretizationConnection):
        raise TypeError("can only construct a full resampling matrix "
                "for a DirectDiscretizationConnection.")

    @memoize_in(actx, (make_direct_full_resample_matrix, "oversample_mat_knl"))
    def knl():
        t_unit = make_loopy_program(
            [
                "{[idof_init]: 0 <= idof_init < nnodes_tgt}",
                "{[jdof_init]: 0 <= jdof_init < nnodes_src}",
                "{[iel]: 0 <= iel < nelements}",
                "{[idof]: 0 <= idof < nunit_dofs_tgt}",
                "{[jdof]: 0 <= jdof < nunit_dofs_src}"
            ],
            """
                result[idof_init, jdof_init] = 0 {id=init}
                ... gbarrier {id=barrier, dep=init}
                result[itgt_base + to_element_indices[iel]*nunit_dofs_tgt + idof,
                       isrc_base + from_element_indices[iel]*nunit_dofs_src + jdof] \
                           = resample_mat[idof, jdof] {dep=barrier}
            """,
            [
                lp.GlobalArg("result", None,
                    shape="nnodes_tgt, nnodes_src",
                    offset=lp.auto),
                lp.ValueArg("itgt_base, isrc_base", np.int32),
                lp.ValueArg("nnodes_tgt, nnodes_src", np.int32),
                ...,
            ],
            name="oversample_mat"
        )

        return lp.tag_inames(t_unit, {
                "iel": ConcurrentElementInameTag(),
                "idof": ConcurrentDOFInameTag(),
                # FIXME: jdof is also concurrent, but the tranform in
                # `meshmode.array_context` does not handle two of them right now
                # "jdof": ConcurrentDOFInameTag(),
                })

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
