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
from meshmode.discretization import Discretization, ElementGroupBase
from meshmode.array_context import (ArrayContext, make_loopy_program,
        PyOpenCLArrayContext, PytatoArrayContext)


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
        return self.from_element_indices.shape[0]

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

    def __call__(self, ary):
        from meshmode.dof_array import DOFArray
        from meshmode.discretization import get_nelements_symbol_for_grp
        if not isinstance(ary, DOFArray):
            raise TypeError("non-array passed to discretization connection")

        actx = ary.array_context

        @memoize_in(actx,
                (DirectDiscretizationConnection, "resample_by_mat_knl",
                    self.is_surjective,))
        def mat_knl():
            if self.is_surjective:
                domains = [
                        """
                        {[iel, idof, j]:
                        0<=iel<nelements and
                        0<=idof<n_to_nodes and
                        0<=j<n_from_nodes}
                        """,
                        ]

                instructions = """
                result[to_element_indices[iel], idof] \
                        = sum(j, resample_mat[idof, j] \
                        * ary[from_element_indices[iel], j])
                        """
            else:
                domains = [
                        """
                        {[iel_init, idof_init]:
                        0<=iel_init<nelements_result and
                        0<=idof_init<n_to_nodes}
                        """,
                        """
                        {[iel, idof, j]:
                        0<=iel<nelements and
                        0<=idof<n_to_nodes and
                        0<=j<n_from_nodes}
                        """,
                        ]

                instructions = """
                result[iel_init, idof_init] = 0 {id=init}
                ... gbarrier {id=barrier, dep=init}
                result[to_element_indices[iel], idof] \
                        = sum(j, resample_mat[idof, j] \
                        * ary[from_element_indices[iel], j]) {dep=barrier}
                        """
            knl = make_loopy_program(
                    domains,
                    instructions,
                    [
                        lp.GlobalArg("result", None,
                            shape="nelements_result, n_to_nodes"),
                        lp.GlobalArg("ary", None,
                            shape="nelements_vec, n_from_nodes"),
                        lp.ValueArg("nelements_result", np.int),
                        lp.ValueArg("nelements_vec", np.int),
                        lp.ValueArg("n_from_nodes", np.int),
                        "...",
                    ],
                    name="resample_by_mat")

            return knl

        @memoize_in(actx,
                (DirectDiscretizationConnection, "resample_by_picking_knl",
                    self.is_surjective))
        def pick_knl():
            if self.is_surjective:
                domains = [
                        """{[iel, idof]:
                        0<=iel<nelements and
                        0<=idof<n_to_nodes}"""]
                instructions = """
                result[to_element_indices[iel], idof] \
                    = ary[from_element_indices[iel], pick_list[idof]]
                """
            else:
                domains = [
                        """
                        {[iel_init, idof_init]:
                        0<=iel_init<nelements_result and
                        0<=idof_init<n_to_nodes}
                        """,
                        """
                        {[iel, idof]:
                        0<=iel<nelements and
                        0<=idof<n_to_nodes}
                        """]
                instructions = """
                result[iel_init, idof_init] = 0 {id=init}
                ... gbarrier {id=barrier, dep=init}
                result[to_element_indices[iel], idof] \
                    = ary[from_element_indices[iel], pick_list[idof]] {dep=barrier}
                """
            knl = make_loopy_program(
                domains,
                instructions,
                [
                    lp.GlobalArg("result", None,
                        shape="nelements_result, n_to_nodes"),
                    lp.GlobalArg("ary", None,
                        shape="nelements_vec, n_from_nodes"),
                    lp.ValueArg("nelements_result", np.int),
                    lp.ValueArg("nelements_vec", np.int),
                    lp.ValueArg("n_from_nodes", np.int),
                    "...",
                    ],
                name="resample_by_picking")

            return knl

        if ary.shape != (len(self.from_discr.groups),):
            raise ValueError("invalid shape of incoming resampling data")

        group_idx_to_result = []

        for i_tgrp, (tgrp, cgrp) in enumerate(
                zip(self.to_discr.groups, self.groups)):

            kernels = []   # get kernels for each batch; to be fused eventually
            kwargs = {}  # kwargs to the fused kernel
            for i_batch, batch in enumerate(cgrp.batches):
                if batch.from_element_indices.size == 0:
                    continue

                point_pick_indices = self._resample_point_pick_indices(
                        actx, i_tgrp, i_batch)

                if point_pick_indices is None:
                    knl = mat_knl()
                    knl = lp.rename_argument(knl, "resample_mat",
                        f"resample_mat_{i_batch}")
                    kwargs[f"resample_mat_{i_batch}"] = (
                            self._resample_matrix(actx, i_tgrp, i_batch))
                    knlname = "resample_by_mat"
                else:
                    knl = pick_knl()
                    knl = lp.rename_argument(knl, "pick_list",
                        f"pick_list_{i_batch}")
                    kwargs[f"pick_list_{i_batch}"] = point_pick_indices
                    knlname = "resample_by_picking"

                knl = lp.fix_parameters(knl, n_from_nodes=ary[
                        batch.from_group_index].shape[-1],
                        n_to_nodes=tgrp.nunit_dofs)

                # {{{ enforce different namespaces for the kernels

                # necessary to avoid using same induction variable for
                # loops across batches

                for iname in sorted(knl[knlname].all_inames()):
                    knl = lp.rename_iname(knl, iname, f"{iname}_{i_batch}")

                knl = lp.rename_argument(knl, "ary", f"ary_{i_batch}")
                knl = lp.rename_argument(knl, "from_element_indices",
                    f"from_element_indices_{i_batch}")
                knl = lp.rename_argument(knl, "to_element_indices",
                    f"to_element_indices_{i_batch}")
                knl = lp.rename_argument(knl, "nelements",
                    f"nelements_{i_batch}")
                knl = lp.rename_argument(knl, "nelements_vec",
                    f"nelements_vec_{i_batch}")

                # }}}

                kwargs[f"ary_{i_batch}"] = ary[batch.from_group_index]
                kwargs[f"from_element_indices_{i_batch}"] = (
                    batch.from_element_indices)
                kwargs[f"to_element_indices_{i_batch}"] = (
                    batch.to_element_indices)
                kwargs[f"nelements_{i_batch}"] = get_nelements_symbol_for_batch(
                        self.to_discr, self.from_discr, tgrp, i_batch, batch, actx)

                kwargs[f"nelements_vec_{i_batch}"] = get_nelements_symbol_for_grp(
                        self.from_discr,
                        self.from_discr.groups[batch.from_group_index], actx)

                kernels.append(knl)

            fused_knl = lp.fuse_kernels(kernels)
            # order of operations doesn't matter
            fused_knl = lp.add_nosync(fused_knl, "global", "writes:result",
                                      "writes:result", bidirectional=True,
                                      force=True)

            result_dict = actx.call_loopy(fused_knl,
                    nelements_result=get_nelements_symbol_for_grp(
                        self.to_discr, tgrp, actx),
                    **kwargs)

            group_idx_to_result.append(result_dict["result"])

        from meshmode.dof_array import DOFArray
        return DOFArray.from_list(actx, group_idx_to_result)

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
    raise NotImplementedError("Stateful access, must rewrite this section.")

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
                    shape="nnodes_tgt, nnodes_src"),
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


def get_nelements_symbol_for_batch(to_discr: Discretization, from_discr:
        Discretization, to_group: ElementGroupBase, ibatch: int,
        batch: InterpolationBatch, actx: ArrayContext):
    if isinstance(actx, PyOpenCLArrayContext):
        return batch.nelements
    elif isinstance(actx, PytatoArrayContext):
        symbol_name = (
                f"nelements_{to_discr.id}_{from_discr.id}_{to_group.index}_{ibatch}")
        if symbol_name not in actx.ns:
            # FIXME: probably needs to be provided via a method in ArrayContext
            import pytato as pt
            pt.make_data_wrapper(actx.ns, np.array(batch.nelements), symbol_name)

        assert actx.ns[symbol_name].data == np.array(batch.nelements)

        return symbol_name
    else:
        raise NotImplementedError()

# vim: foldmethod=marker
