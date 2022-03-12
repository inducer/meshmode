import pyopencl.array as cl_array
import kanren
import pytato as pt
import unification
import logging

from functools import partial, reduce
from arraycontext.impl.pytato.fake_numpy import PytatoFakeNumpyNamespace
from arraycontext import rec_map_reduce_array_container
from meshmode.transform_metadata import DiscretizationEntityAxisTag
from pytato.loopy import LoopyCall
from pytato.array import EinsumElementwiseAxis, EinsumReductionAxis
from pytato.transform import ArrayOrNames
from arraycontext import ArrayContainer
from arraycontext.container.traversal import rec_map_array_container
from typing import Set, Mapping, Tuple, Union
logger = logging.getLogger(__name__)


MAX_UNIFY_RETRIES = 50  # used by unify_discretization_entity_tags


def _can_be_eagerly_computed(ary) -> bool:
    from pytato.transform import InputGatherer
    from pytato.array import Placeholder
    return all(not isinstance(inp, Placeholder)
               for inp in InputGatherer()(ary))


class EagerReduceComputingPytatoFakeNumpyNamespace(PytatoFakeNumpyNamespace):
    """
    A Numpy-namespace that computes the reductions eagerly whenever possible.
    """
    def sum(self, a, axis=None, dtype=None):
        if (rec_map_reduce_array_container(all,
                                           _can_be_eagerly_computed, a)
                and axis is None):

            def _pt_sum(ary):
                return cl_array.sum(self._array_context.freeze(ary),
                                 dtype=dtype,
                                 queue=self._array_context.queue)

            return self._array_context.thaw(rec_map_reduce_array_container(sum,
                                                                           _pt_sum,
                                                                           a))
        else:
            return super().sum(a, axis=axis, dtype=dtype)

    def min(self, a, axis=None):
        if (rec_map_reduce_array_container(all,
                                           _can_be_eagerly_computed, a)
                and axis is None):
            queue = self._array_context.queue
            frozen_result = rec_map_reduce_array_container(
                partial(reduce, partial(cl_array.minimum, queue=queue)),
                lambda ary: cl_array.min(self._array_context.freeze(ary),
                                         queue=queue),
                a)
            return self._array_context.thaw(frozen_result)
        else:
            return super().min(a, axis=axis)

    def max(self, a, axis=None):
        if (rec_map_reduce_array_container(all,
                                           _can_be_eagerly_computed, a)
                and axis is None):
            queue = self._array_context.queue
            frozen_result = rec_map_reduce_array_container(
                partial(reduce, partial(cl_array.maximum, queue=queue)),
                lambda ary: cl_array.max(self._array_context.freeze(ary),
                                         queue=queue),
                a)
            return self._array_context.thaw(frozen_result)
        else:
            return super().max(a, axis=axis)


# {{{ solve for discretization metadata for arrays' axes

class DiscretizationEntityConstraintCollector(pt.transform.Mapper):
    """
    .. warning::

        Instances of this mapper type store state that are only for visiting a
        single DAG. Using a single instance for collecting the constraints on
        multiple DAGs is undefined behavior.
    """
    def __init__(self):
        super().__init__()
        self._visited_ids: Set[int] = set()

        # axis_to_var: mapping from (array, iaxis) to the kanren variable to be
        # used for unification.
        self.axis_to_tag_var: Mapping[Tuple[pt.Array, int],
                                      unification.variable.Var] = {}
        self.variables_to_solve: Set[unification.variable.Var] = set()
        self.constraints = []

    # type-ignore reason: CachedWalkMapper.rec's type does not match
    # WalkMapper.rec's type
    def rec(self, expr: ArrayOrNames) -> None:  # type: ignore
        if id(expr) in self._visited_ids:
            return

        # type-ignore reason: super().rec expects either 'Array' or
        # 'AbstractResultWithNamedArrays', passed 'ArrayOrNames'
        super().rec(expr)  # type: ignore
        self._visited_ids.add(id(expr))

    def get_kanren_var_for_axis_tag(self,
                                    expr: pt.Array,
                                    iaxis: int
                                    ) -> unification.variable.Var:
        key = (expr, iaxis)

        if key not in self.axis_to_tag_var:
            self.axis_to_tag_var[key] = kanren.var()

        return self.axis_to_tag_var[key]

    def _record_all_axes_to_be_solved_if_impl_stored(self, expr):
        if expr.tags_of_type(pt.tags.ImplStored):
            for iaxis in range(expr.ndim):
                self.variables_to_solve.add(self.get_kanren_var_for_axis_tag(expr,
                                                                             iaxis))

    def _record_all_axes_to_be_solved(self, expr):
        for iaxis in range(expr.ndim):
            self.variables_to_solve.add(self.get_kanren_var_for_axis_tag(expr,
                                                                         iaxis))

    def record_constraint(self, lhs, rhs):
        self.constraints.append((lhs, rhs))

    def record_eq_constraints_from_tags(self, expr: pt.Array) -> None:
        for iaxis, axis in enumerate(expr.axes):
            if axis.tags_of_type(DiscretizationEntityAxisTag):
                discr_tag, = axis.tags_of_type(DiscretizationEntityAxisTag)
                axis_var = self.get_kanren_var_for_axis_tag(expr, iaxis)
                self.record_constraint(axis_var, discr_tag)

    def _map_input_base(self, expr: pt.InputArgumentBase
                        ) -> None:
        self.record_eq_constraints_from_tags(expr)
        self._record_all_axes_to_be_solved_if_impl_stored(expr)

        for dim in expr.shape:
            if isinstance(dim, pt.Array):
                self.rec(dim)

    map_placeholder = _map_input_base
    map_data_wrapper = _map_input_base
    map_size_param = _map_input_base

    def map_index_lambda(self, expr: pt.IndexLambda) -> None:
        from pytato.utils import are_shape_components_equal
        from pytato.raising import index_lambda_to_high_level_op
        from pytato.raising import (BinaryOp, FullOp, WhereOp,
                                    BroadcastOp, C99CallOp, ReduceOp)

        # {{{ record constraints for expr and its subexprs.

        self.record_eq_constraints_from_tags(expr)
        self._record_all_axes_to_be_solved_if_impl_stored(expr)

        for dim in expr.shape:
            if isinstance(dim, pt.Array):
                self.rec(dim)

        for bnd in expr.bindings.values():
            self.rec(bnd)

        # }}}

        hlo = index_lambda_to_high_level_op(expr)

        if isinstance(hlo, BinaryOp):
            subexprs = (hlo.x1, hlo.x2)
        elif isinstance(hlo, WhereOp):
            subexprs = (hlo.condition, hlo.then, hlo.else_)
        elif isinstance(hlo, FullOp):
            # A full-op does not impose any constraints
            subexprs = ()
        elif isinstance(hlo, BroadcastOp):
            subexprs = (hlo.x,)
        elif isinstance(hlo, C99CallOp):
            subexprs = hlo.args
        elif isinstance(hlo, ReduceOp):
            # {{{ ReduceOp doesn't quite involve broadcasting

            i_out_axis = 0
            for i_in_axis in range(hlo.x.ndim):
                if i_in_axis not in hlo.axes:
                    in_tag_var = self.get_kanren_var_for_axis_tag(hlo.x,
                                                                  i_in_axis)
                    out_tag_var = self.get_kanren_var_for_axis_tag(expr,
                                                                   i_out_axis)
                    self.record_constraint(in_tag_var, out_tag_var)
                    i_out_axis += 1

            assert i_out_axis == expr.ndim

            # }}}

            for axis in hlo.axes:
                self.variables_to_solve.add(self.get_kanren_var_for_axis_tag(hlo.x,
                                                                             axis))
            return

        else:
            raise NotImplementedError(type(hlo))

        for subexpr in subexprs:
            if isinstance(subexpr, pt.Array):
                for i_in_axis, i_out_axis in zip(
                        range(subexpr.ndim),
                        range(expr.ndim-subexpr.ndim, expr.ndim)):
                    in_dim = subexpr.shape[i_in_axis]
                    out_dim = expr.shape[i_out_axis]
                    if are_shape_components_equal(in_dim, out_dim):
                        in_tag_var = self.get_kanren_var_for_axis_tag(subexpr,
                                                                      i_in_axis)
                        out_tag_var = self.get_kanren_var_for_axis_tag(expr,
                                                                       i_out_axis)

                        self.record_constraint(in_tag_var, out_tag_var)
                    else:
                        # broadcasted axes, cannot belong to the same
                        # discretization entity.
                        assert are_shape_components_equal(in_dim, 1)

    def map_stack(self, expr: pt.Stack) -> None:
        self.record_eq_constraints_from_tags(expr)
        self._record_all_axes_to_be_solved_if_impl_stored(expr)
        # TODO; I think the axis corresponding to 'axis' need not be solved.
        for ary in expr.arrays:
            self.rec(ary)

        for iaxis in range(expr.ndim):
            for ary in expr.arrays:
                if iaxis < expr.axis:
                    in_tag_var = self.get_kanren_var_for_axis_tag(ary,
                                                                  iaxis)
                    out_tag_var = self.get_kanren_var_for_axis_tag(expr,
                                                                   iaxis)

                    self.record_constraint(in_tag_var, out_tag_var)
                elif iaxis == expr.axis:
                    pass
                elif iaxis > expr.axis:
                    in_tag_var = self.get_kanren_var_for_axis_tag(ary,
                                                                  iaxis-1)
                    out_tag_var = self.get_kanren_var_for_axis_tag(expr,
                                                                   iaxis)

                    self.record_constraint(in_tag_var, out_tag_var)
                else:
                    raise AssertionError

    def map_concatenate(self, expr: pt.Concatenate) -> None:
        self.record_eq_constraints_from_tags(expr)
        self._record_all_axes_to_be_solved_if_impl_stored(expr)
        # TODO; I think the axis corresponding to 'axis' need not be solved.
        for ary in expr.arrays:
            self.rec(ary)

        for ary in expr.arrays:
            assert ary.ndim == expr.ndim
            for iaxis in range(expr.ndim):
                if iaxis != expr.axis:
                    # non-concatenated axes share the dimensions.
                    in_tag_var = self.get_kanren_var_for_axis_tag(ary,
                                                                  iaxis)
                    out_tag_var = self.get_kanren_var_for_axis_tag(expr,
                                                                   iaxis)
                    self.record_constraint(in_tag_var, out_tag_var)

    def map_axis_permutation(self, expr: pt.AxisPermutation
                             ) -> None:
        self.record_eq_constraints_from_tags(expr)
        self._record_all_axes_to_be_solved_if_impl_stored(expr)
        self.rec(expr.array)

        assert expr.ndim == expr.array.ndim

        for out_axis in range(expr.ndim):
            in_axis = expr.axis_permutation[out_axis]
            out_tag = self.get_kanren_var_for_axis_tag(expr, out_axis)
            in_tag = self.get_kanren_var_for_axis_tag(expr, in_axis)
            self.record_constraint(out_tag, in_tag)

    def map_basic_index(self, expr: pt.IndexBase) -> None:
        from pytato.array import NormalizedSlice
        from pytato.utils import are_shape_components_equal

        self.record_eq_constraints_from_tags(expr)
        self._record_all_axes_to_be_solved_if_impl_stored(expr)
        self.rec(expr.array)

        i_out_axis = 0

        assert len(expr.indices) == expr.array.ndim

        for i_in_axis, idx in enumerate(expr.indices):
            if isinstance(idx, int):
                pass
            else:
                assert isinstance(idx, NormalizedSlice)
                if (idx.step == 1
                        and are_shape_components_equal(idx.start, 0)
                        and are_shape_components_equal(idx.stop,
                                                       expr.array.shape[i_in_axis])):

                    i_in_axis_tag = self.get_kanren_var_for_axis_tag(expr.array,
                                                                     i_in_axis)
                    i_out_axis_tag = self.get_kanren_var_for_axis_tag(expr,
                                                                      i_out_axis)
                    self.record_constraint(i_in_axis_tag, i_out_axis_tag)

                i_out_axis += 1

    def map_contiguous_advanced_index(self,
                                      expr: pt.AdvancedIndexInContiguousAxes
                                      ) -> None:
        from pytato.array import NormalizedSlice
        from pytato.utils import (partition, get_shape_after_broadcasting,
                                  are_shapes_equal, are_shape_components_equal)

        self.record_eq_constraints_from_tags(expr)
        self._record_all_axes_to_be_solved_if_impl_stored(expr)
        self.rec(expr.array)
        for idx in expr.indices:
            if isinstance(idx, pt.Array):
                self.rec(idx)

        i_adv_indices, i_basic_indices = partition(
            lambda idx: isinstance(expr.indices[idx], NormalizedSlice),
            range(len(expr.indices)))
        npre_advanced_basic_indices = len([i_idx
                                      for i_idx in i_basic_indices
                                      if i_idx < i_adv_indices[0]])
        npost_advanced_basic_indices = len([i_idx
                                       for i_idx in i_basic_indices
                                       if i_idx > i_adv_indices[-1]])

        indirection_arrays = [expr.indices[i_idx] for i_idx in i_adv_indices]
        assert are_shapes_equal(
            get_shape_after_broadcasting(indirection_arrays),
            expr.shape[
                npre_advanced_basic_indices:expr.ndim-npost_advanced_basic_indices])

        for subexpr in indirection_arrays:
            if isinstance(subexpr, pt.Array):
                for i_in_axis, i_out_axis in zip(
                        range(subexpr.ndim),
                        range(expr.ndim-subexpr.ndim+npre_advanced_basic_indices,
                              expr.ndim-npost_advanced_basic_indices)):
                    in_dim = subexpr.shape[i_in_axis]
                    out_dim = expr.shape[i_out_axis]
                    if are_shape_components_equal(in_dim, out_dim):
                        in_tag_var = self.get_kanren_var_for_axis_tag(subexpr,
                                                                      i_in_axis)
                        out_tag_var = self.get_kanren_var_for_axis_tag(expr,
                                                                       i_out_axis)

                        self.record_constraint(in_tag_var, out_tag_var)
                    else:
                        # broadcasted axes, cannot belong to the same
                        # discretization entity.
                        assert are_shape_components_equal(in_dim, 1)

    def map_non_contiguous_advanced_index(self,
                                          expr: pt.AdvancedIndexInNoncontiguousAxes
                                          ) -> None:
        self.record_eq_constraints_from_tags(expr)
        self._record_all_axes_to_be_solved_if_impl_stored(expr)
        self.rec(expr.array)
        for idx in expr.indices:
            if isinstance(idx, pt.Array):
                self.rec(idx)

    def map_reshape(self, expr: pt.Reshape) -> None:
        self.record_eq_constraints_from_tags(expr)
        self._record_all_axes_to_be_solved_if_impl_stored(expr)
        self.rec(expr.array)
        # we can add constraints to reshape that only include new axes in its
        # reshape.
        # Other reshapes do not 'conserve' the types in our type-system.
        # Well *what if*. Let's just say this type inference fails for
        # non-trivial 'reshapes'. So, what are the 'trivial' reshapes?
        # trivial reshapes:
        # (x1, x2, ... xn) -> ((1,)*, x1, (1,)*, x2, (1,)*, x3, (1,)*, ..., xn, 1*)
        # given all(x1!=1, x2!=1, x3!=1, .. xn!= 1)
        if ((1 not in (expr.array.shape))  # leads to ambiguous newaxis
                and (set(expr.shape) <= (set(expr.array.shape) | {1}))):
            i_in_axis = 0
            for i_out_axis, dim in enumerate(expr.shape):
                if dim != 1:
                    assert dim == expr.array.shape[i_in_axis]
                    i_in_axis_tag = self.get_kanren_var_for_axis_tag(expr.array,
                                                                     i_in_axis)
                    i_out_axis_tag = self.get_kanren_var_for_axis_tag(expr,
                                                                      i_out_axis)
                    self.record_constraint(i_in_axis_tag, i_out_axis_tag)
                    i_in_axis += 1
        else:
            # print(f"Skipping: {expr.array.shape} -> {expr.shape}")
            # Wacky reshape => bail.
            return

    def map_einsum(self, expr: pt.Einsum) -> None:

        self.record_eq_constraints_from_tags(expr)
        self._record_all_axes_to_be_solved_if_impl_stored(expr)

        for arg in expr.args:
            self.rec(arg)

        descr_to_tag = {}
        for iaxis in range(expr.ndim):
            descr_to_tag[EinsumElementwiseAxis(iaxis)] = (
                self.get_kanren_var_for_axis_tag(expr, iaxis))

        for access_descrs, arg in zip(expr.access_descriptors,
                                      expr.args):
            # if an einsum is stored => every argument's axes must
            # also be inferred, even those that are getting reduced.
            for iarg_axis, descr in enumerate(access_descrs):
                in_tag_var = self.get_kanren_var_for_axis_tag(arg,
                                                              iarg_axis)

                if descr in descr_to_tag:
                    self.record_constraint(descr_to_tag[descr], in_tag_var)
                else:
                    descr_to_tag[descr] = in_tag_var

                if isinstance(descr, EinsumReductionAxis):
                    self.variables_to_solve.add(in_tag_var)

    def map_dict_of_named_arrays(self, expr: pt.DictOfNamedArrays
                                 ) -> None:
        for _, subexpr in sorted(expr._data.items()):
            self.rec(subexpr)
            self._record_all_axes_to_be_solved(subexpr)

    def map_loopy_call(self, expr: LoopyCall) -> None:
        for _, subexpr in sorted(expr.bindings.items()):
            if isinstance(subexpr, pt.Array):
                if not isinstance(subexpr, pt.InputArgumentBase):
                    self._record_all_axes_to_be_solved(subexpr)
                self.rec(subexpr)

        # there's really no good way to propagate the metadata in this case.
        # One *could* raise the loopy kernel instruction expressions to
        # high level ops, but that's really involved and probably not worth it.

    def map_named_array(self, expr: pt.NamedArray) -> None:
        self.record_eq_constraints_from_tags(expr)
        self.rec(expr._container)

    def map_distributed_send_ref_holder(self,
                                        expr: pt.DistributedSendRefHolder
                                        ) -> None:
        self.record_eq_constraints_from_tags(expr)
        self.rec(expr.passthrough_data)
        for idim in range(expr.ndim):
            assert (expr.passthrough_data.shape[idim]
                    == expr.shape[idim])
            self.record_constraint(
                self.get_kanren_var_for_axis_tag(expr.passthrough_data,
                                                 idim),
                self.get_kanren_var_for_axis_tag(expr, idim)
            )

    def map_distributed_recv(self,
                             expr: pt.DistributedRecv) -> None:
        self.record_eq_constraints_from_tags(expr)


def unify_discretization_entity_tags(expr: Union[ArrayContainer, ArrayOrNames]
                                     ) -> ArrayOrNames:
    if not isinstance(expr, (pt.Array, pt.DictOfNamedArrays)):
        return rec_map_array_container(unify_discretization_entity_tags,
                                       expr)

    from collections import defaultdict
    discr_unification_helper = DiscretizationEntityConstraintCollector()
    discr_unification_helper(expr)
    tag_var_to_axis = {}
    variables_to_solve = []

    for (axis, var) in discr_unification_helper.axis_to_tag_var.items():
        tag_var_to_axis[var] = axis
        if var in discr_unification_helper.variables_to_solve:
            variables_to_solve.append(var)

    lhs = [cnstrnt[0] for cnstrnt in discr_unification_helper.constraints]
    rhs = [cnstrnt[1] for cnstrnt in discr_unification_helper.constraints]
    assert len(lhs) == len(rhs)
    solutions = {}

    for i_retry in range(MAX_UNIFY_RETRIES):
        old_solutions = solutions.copy()
        solutions = unification.unify(lhs, rhs,
                                      {l_expr: r_expr
                                       for l_expr, r_expr in solutions.items()
                                       if isinstance(r_expr,
                                                     DiscretizationEntityAxisTag)})
        if solutions == old_solutions:
            logger.info(f"Unification converged after {i_retry} iterations.")
            break
    else:
        logger.warn(f"Could not converge after {MAX_UNIFY_RETRIES} iterations.")

    # Ideally it might be better to enable this, but that would be too
    # restrictive as not all computation graphs result in DOFArray ouptuts
    # if not (frozenset(variables_to_solve) <= frozenset(solutions)):
    #     raise RuntimeError("Unification failed.")

    # ary_to_axes_tags: mapping from array to a mapping from iaxis to the
    # solved tag.
    ary_to_axes_tags = defaultdict(dict)
    for var in solutions:
        ary, axis = tag_var_to_axis[var]
        if isinstance(solutions[var], DiscretizationEntityAxisTag):
            ary_to_axes_tags[ary][axis] = solutions[var]
        if var in variables_to_solve and (
                not isinstance(solutions[var], DiscretizationEntityAxisTag)):
            raise RuntimeError(f"Could not solve for {var}.")

    def attach_tags(expr: ArrayOrNames) -> ArrayOrNames:
        if not isinstance(expr, pt.Array):
            return expr

        for iaxis, solved_tag in ary_to_axes_tags[expr].items():
            if expr.axes[iaxis].tags_of_type(DiscretizationEntityAxisTag):
                discr_tag, = (expr
                              .axes[iaxis]
                              .tags_of_type(DiscretizationEntityAxisTag))
                assert discr_tag == solved_tag
            else:
                if not isinstance(solved_tag, DiscretizationEntityAxisTag):
                    actual_tag = discr_unification_helper.axis_to_tag_var[(expr,
                                                                           iaxis)]
                    assert actual_tag in discr_unification_helper.variables_to_solve
                    assert actual_tag in variables_to_solve
                    raise ValueError(f"In {expr!r}, axis={iaxis}'s type cannot be "
                                     "inferred.")
                expr = expr.with_tagged_axis(iaxis, solved_tag)

        if isinstance(expr, pt.Einsum):
            redn_descr_to_entity_type = {}
            for access_descrs, arg in zip(expr.access_descriptors,
                                          expr.args):
                for iaxis, access_descr in enumerate(access_descrs):
                    if isinstance(access_descr, EinsumReductionAxis):
                        redn_descr_to_entity_type[access_descr] = (
                            ary_to_axes_tags[arg][iaxis])

            if (frozenset(redn_descr_to_entity_type)
                    != frozenset(expr.redn_descr_to_redn_dim)):
                raise ValueError

            for redn_descr, solved_tag in redn_descr_to_entity_type.items():
                if not isinstance(solved_tag, DiscretizationEntityAxisTag):
                    raise ValueError(f"In {expr!r}, redn_descr={redn_descr}'s"
                                     " type cannot be inferred.")
                expr = expr.with_tagged_redn_dim(redn_descr, solved_tag)

        if isinstance(expr, pt.IndexLambda):
            from pytato.raising import (index_lambda_to_high_level_op,
                                        ReduceOp)

            hlo = index_lambda_to_high_level_op(expr)
            if isinstance(hlo, ReduceOp):
                for iaxis in hlo.axes:
                    solved_tag = ary_to_axes_tags[hlo.x][iaxis]
                    if not isinstance(solved_tag, DiscretizationEntityAxisTag):
                        raise ValueError(f"In {expr!r}, redn_descr={iaxis}'s"
                                        " type cannot be inferred.")

                    expr = expr.with_tagged_redn_dim(iaxis, solved_tag)

        return expr

    return pt.transform.map_and_copy(expr, attach_tags)

# }}}


class UnInferredStoredArrayCatcher(pt.transform.CachedWalkMapper):
    """
    Raises a :class:`ValueError` if a stored array has axes without a
    :class:`DiscretizationEntityAxisTag` tagged to it.
    """
    def post_visit(self, expr: ArrayOrNames) -> None:
        if (isinstance(expr, pt.Array)
                and expr.tags_of_type(pt.tags.ImplStored)):
            if any(len(axis.tags_of_type(DiscretizationEntityAxisTag)) != 1
                   for axis in expr.axes):
                raise ValueError(f"{expr!r} doesn't have all its axes inferred.")

        if (isinstance(expr, pt.IndexLambda)
                and any(len(redn_dim.tags_of_type(DiscretizationEntityAxisTag)) != 1
                        for redn_dim in expr.reduction_dims.values())):
            raise ValueError(f"{expr!r} doesn't have all its redn axes inferred.")

        if (isinstance(expr, pt.Einsum)
                and any(len(redn_dim.tags_of_type(DiscretizationEntityAxisTag)) != 1
                        for redn_dim in expr.redn_descr_to_redn_dim.values())):
            raise ValueError(f"{expr!r} doesn't have all its redn axes inferred.")

        if isinstance(expr, pt.DictOfNamedArrays):
            if any(any(len(axis.tags_of_type(DiscretizationEntityAxisTag)) != 1
                       for axis in subexpr.axes)
                   for subexpr in expr._data.values()):
                raise ValueError(f"{expr!r} doesn't have all its axes inferred.")

        from pytato.loopy import LoopyCall

        if isinstance(expr, LoopyCall):
            if any(any(len(axis.tags_of_type(DiscretizationEntityAxisTag)) != 1
                       for axis in subexpr.axes)
                   for subexpr in expr.bindings.values()
                   if (isinstance(subexpr, pt.Array)
                       and not isinstance(subexpr, pt.InputArgumentBase)
                       and subexpr.ndim != 0)):
                raise ValueError(f"{expr!r} doesn't have all its axes inferred.")


def are_all_stored_arrays_inferred(expr: ArrayOrNames):
    UnInferredStoredArrayCatcher()(expr)

# vim: fdm=marker
