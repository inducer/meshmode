__copyright__ = "Copyright (C) 2023 Kaushik Kulkarni"

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

import pytato as pt
from typing import Callable, Dict, FrozenSet, List, Mapping, Optional, Tuple
from pytato.array import (InputArgumentBase, IndexLambda,
                          Stack, Concatenate, AdvancedIndexInContiguousAxes,
                          AdvancedIndexInNoncontiguousAxes, BasicIndex,
                          Einsum, Roll, Array, Reshape, DictOfNamedArrays,
                          DataWrapper, Placeholder, IndexBase)
from pytato.transform import (ArrayOrNames, CombineMapper, Mapper,
                              MappedT)
from immutables import Map


# {{{ fuse_dof_pick_lists

def _is_materialized(expr: Array) -> bool:
    """
    Returns true if an array is materialized. An array is considered to be
    materialized if it is either a :class:`pytato.array.InputArgumentBase` or
    is tagged with :class:`pytato.tags.ImplStored`.
    """
    from pytato.tags import ImplStored
    return (isinstance(expr, InputArgumentBase)
            or bool(expr.tags_of_type(ImplStored)))


def _can_index_lambda_propagate_indirections_without_changing_axes(
        expr: IndexLambda) -> bool:

    from pytato.utils import are_shapes_equal
    from pytato.raising import (index_lambda_to_high_level_op,
                                BinaryOp)
    hlo = index_lambda_to_high_level_op(expr)
    return (isinstance(hlo, BinaryOp)
                and ((not isinstance(hlo.x1, pt.Array))
                     or (not isinstance(hlo.x2, pt.Array))
                     or are_shapes_equal(hlo.x1.shape, hlo.x2.shape)))


def _is_advanced_indexing_from_resample_by_picking(
    expr: AdvancedIndexInContiguousAxes
) -> bool:

    from pytato.utils import are_shapes_equal

    if expr.ndim != 2 or expr.array.ndim != 2:
        # only worry about dofs-to-dofs like resamplings.
        return False

    idx1, idx2 = expr.indices

    if (not isinstance(idx1, Array)) or (not isinstance(idx2, Array)):
        # only worry about resamplings of the form
        # `u[from_el_indices, dof_pick_list]`.
        return False

    if (idx1.ndim != 2) or (idx2.ndim != 2):
        return False

    if not are_shapes_equal(idx1.shape, (idx2.shape[0], 1)):
        return False

    return True


# **Note: implementation of _CanPickIndirectionsBePropagated is restrictive on
# purpose.**
# Although this could be generalized to get a tighter condition on when indirections
# can legally be propagated, (for now) we are only interested at patterns commonly
# seen in meshmode-based expression graphs.
class _CanPickIndirectionsBePropagated(Mapper):
    """
    Mapper to test whether the dof pick lists and element pick lists can be
    propagated towards the operands.
    """
    def __init__(self) -> None:
        self._cache: Dict[Tuple[ArrayOrNames, int, int], bool] = {}
        super().__init__()

    # type-ignore-reason: incompatible function signature with Mapper.rec
    def rec(self, expr: ArrayOrNames,  # type: ignore[override]
            iel_axis: int, idof_axis: int) -> bool:
        if isinstance(expr, Array):
            assert 0 <= iel_axis < expr.ndim
            assert 0 <= idof_axis < expr.ndim
            # the condition below ensures that we are only dealing with indirections
            # appearing at contiguous locations.
            assert abs(iel_axis-idof_axis) == 1

        if isinstance(expr, Array) and _is_materialized(expr):
            return True

        key = (expr, iel_axis, idof_axis)
        try:
            return self._cache[key]
        except KeyError:
            result = super().rec(expr, iel_axis, idof_axis)
            self._cache[key] = result
            return result

    def _map_input_base(self, expr: InputArgumentBase,
                        iel_axis: int, idof_axis: int) -> bool:
        return True

    map_placeholder = _map_input_base
    map_data_wrapper = _map_input_base

    def map_index_lambda(self,
                         expr: IndexLambda,
                         iel_axis: int,
                         idof_axis: int) -> bool:
        if _can_index_lambda_propagate_indirections_without_changing_axes(expr):
            return all([self.rec(bnd, iel_axis, idof_axis)
                        for bnd in expr.bindings.values()])
        else:
            return False

    def map_stack(self, expr: Stack, iel_axis: int, idof_axis: int) -> bool:
        if expr.axis in {iel_axis, idof_axis}:
            return False
        else:
            if iel_axis < expr.axis:
                assert idof_axis < expr.axis
                return all(self.rec(ary, iel_axis, idof_axis) for ary in expr.arrays)
            else:
                assert idof_axis > expr.axis
                return all(self.rec(ary, iel_axis-1, idof_axis-1)
                           for ary in expr.arrays)

    def map_concatenate(self,
                        expr: Concatenate,
                        iel_axis: int,
                        idof_axis: int) -> bool:
        if expr.axis in {iel_axis, idof_axis}:
            return False
        else:
            return all(self.rec(ary, iel_axis, idof_axis) for ary in expr.arrays)

    def map_einsum(self, expr: Einsum, iel_axis: int, idof_axis: int) -> bool:
        from pytato.array import EinsumElementwiseAxis

        for arg, acc_descrs in zip(expr.args, expr.access_descriptors):
            try:
                arg_iel_axis = acc_descrs.index(EinsumElementwiseAxis(iel_axis))
                arg_idof_axis = acc_descrs.index(EinsumElementwiseAxis(idof_axis))
            except ValueError:
                return False
            else:
                if abs(arg_iel_axis - arg_idof_axis) != 1:
                    return False

                if not self.rec(arg, arg_iel_axis, arg_idof_axis):
                    return False

        return True

    def map_roll(self, expr: Roll, iel_axis: int, idof_axis: int) -> bool:
        return False

    def map_non_contiguous_advanced_index(self,
                                          expr: AdvancedIndexInNoncontiguousAxes,
                                          iel_axis: int,
                                          idof_axis: int) -> bool:
        # TODO: In meshmode based codes non-contiguous advanced indices are rare
        # i.e. not a first order concern to optimize across these nodes.
        return False

    def map_contiguous_advanced_index(self,
                                      expr: AdvancedIndexInContiguousAxes,
                                      iel_axis: int,
                                      idof_axis: int) -> bool:

        return (_is_advanced_indexing_from_resample_by_picking(expr)
                and iel_axis == 0 and idof_axis == 1
                and self.rec(expr.array, iel_axis, idof_axis))

    def map_basic_index(self, expr: BasicIndex,
                        iel_axis: int, idof_axis: int) -> bool:
        # TODO: In meshmode based codes slices are rare i.e. not a first order
        # concern to optimize across these nodes.
        return False

    def map_reshape(self, expr: Reshape,
                    iel_axis: int, idof_axis: int) -> bool:
        # TODO: In meshmode based codes reshapes in flux computations on sub-domains
        # are rare i.e. not a first order concern to optimize across these nodes.
        return False


def _fuse_from_element_indices(from_element_indices: Tuple[Array, ...]):
    assert all(from_el_idx.ndim == 2 for from_el_idx in from_element_indices)
    assert all(from_el_idx.shape[1] == 1 for from_el_idx in from_element_indices)

    result = from_element_indices[-1]
    for from_el_idx in from_element_indices[-2::-1]:
        result = result[from_el_idx, 0]

    return result


def _fuse_dof_pick_lists(dof_pick_lists: Tuple[Array, ...], from_element_indices:
                         Tuple[Array, ...]):
    assert all(from_el_idx.ndim == 2 for from_el_idx in from_element_indices)
    assert all(dof_pick_list.ndim == 2 for dof_pick_list in dof_pick_lists)
    assert all(from_el_idx.shape[1] == 1 for from_el_idx in from_element_indices)

    result = dof_pick_lists[-1]
    for from_el_idx, dof_pick_list in zip(from_element_indices[-2::-1],
                                          dof_pick_lists[-2::-1]):
        result = result[from_el_idx, dof_pick_list]

    return result


def _pick_list_fusers_map_materialized_node(rec_expr: Array,
                                            iel_axis: Optional[int],
                                            idof_axis: Optional[int],
                                            from_element_indices: Tuple[Array, ...],
                                            dof_pick_lists: Tuple[Array, ...]
                                            ) -> Array:

    if iel_axis is not None:
        assert idof_axis is not None
        assert len(from_element_indices) != 0
        assert len(from_element_indices) == len(dof_pick_lists)

        fused_from_element_indices = _fuse_from_element_indices(from_element_indices)
        fused_dof_pick_lists = _fuse_dof_pick_lists(dof_pick_lists,
                                                    from_element_indices)
        if iel_axis < idof_axis:
            assert idof_axis == (iel_axis+1)
            indices = (slice(None),)*iel_axis + (fused_from_element_indices,
                                                 fused_dof_pick_lists)
        else:
            assert iel_axis == (idof_axis+1)
            indices = (slice(None),)*iel_axis + (fused_dof_pick_lists,
                                                 fused_from_element_indices)

        return rec_expr[indices]
    else:
        assert idof_axis is None
        return rec_expr


class PickListFusers(Mapper):
    def __init__(self) -> None:
        self.can_pick_indirections_be_propagated = _CanPickIndirectionsBePropagated()
        self._cache: Dict[Tuple[Array, Optional[int], Optional[int],
                                Tuple[Array, ...], Tuple[Array, ...]], Array] = {}
        super().__init__()

    # type-ignore-reason: incompatible signature with Mapper.rec
    def rec(self,  # type: ignore[override]
            expr: Array,
            iel_axis: Optional[int],
            idof_axis: Optional[int],
            from_element_indices: Tuple[Array, ...],
            dof_pick_lists: Tuple[Array, ...],
            ) -> Array:
        if not isinstance(expr, Array):
            raise ValueError("Mapping AbstractResultWithNamedArrays"
                             " is illegal for PickListFusers. Pass arrays"
                             " instead.")

        if iel_axis is not None:
            assert idof_axis is not None
            assert 0 <= iel_axis < expr.ndim
            assert 0 <= idof_axis < expr.ndim
            # the condition below ensures that we are only dealing with indirections
            # appearing at contiguous locations.
            assert abs(iel_axis-idof_axis) == 1
        else:
            assert idof_axis is None
            assert len(from_element_indices) == 0

        assert len(dof_pick_lists) == len(from_element_indices)

        key = (expr, iel_axis, idof_axis, from_element_indices, dof_pick_lists)
        try:
            return self._cache[key]
        except KeyError:
            result = super().rec(expr, iel_axis, idof_axis,
                                 from_element_indices, dof_pick_lists)
            self._cache[key] = result
            return result

    # type-ignore-reason: incompatible signature with Mapper.__call__
    def __call__(self,  # type: ignore[override]
                 expr: Array,
                 iel_axis: Optional[int],
                 idof_axis: Optional[int],
                 from_element_indices: Tuple[Array, ...],
                 dof_pick_lists: Tuple[Array, ...],
                 ) -> Array:
        return self.rec(expr, iel_axis, idof_axis,
                        from_element_indices, dof_pick_lists)

    def _map_input_base(self,
                        expr: InputArgumentBase,
                        iel_axis: int,
                        idof_axis: int,
                        from_element_indices: Tuple[Array, ...],
                        dof_pick_lists: Tuple[Array, ...]) -> Array:
        return _pick_list_fusers_map_materialized_node(
            expr, iel_axis, idof_axis, from_element_indices, dof_pick_lists)

    map_placeholder = _map_input_base
    map_data_wrapper = _map_input_base

    def map_index_lambda(self,
                         expr: IndexLambda,
                         iel_axis: Optional[int],
                         idof_axis: Optional[int],
                         from_element_indices: Tuple[Array, ...],
                         dof_pick_lists: Tuple[Array, ...]) -> Array:
        if _is_materialized(expr):
            # Stop propagating indirections and return indirections collected till
            # this point.
            rec_expr = IndexLambda(
                expr.expr,
                expr.shape,
                expr.dtype,
                Map({name: self.rec(bnd, None, None, (), ())
                     for name, bnd in expr.bindings.items()}),
                var_to_reduction_descr=expr.var_to_reduction_descr,
                tags=expr.tags,
                axes=expr.axes
            )
            return _pick_list_fusers_map_materialized_node(
                rec_expr, iel_axis, idof_axis, from_element_indices, dof_pick_lists)

        if iel_axis is not None:
            assert idof_axis is not None
            assert _can_index_lambda_propagate_indirections_without_changing_axes(
                expr)
            from pytato.utils import are_shapes_equal
            new_el_dim, new_dofs_dim = dof_pick_lists[0].shape
            assert are_shapes_equal(from_element_indices[0].shape, (new_el_dim, 1))

            new_shape = tuple(
                new_el_dim if idim == iel_axis else (
                    new_dofs_dim if idim == idof_axis else dim)
                for idim, dim in enumerate(expr.shape))

            return IndexLambda(
                expr.expr,
                new_shape,
                expr.dtype,
                Map({name: self.rec(bnd, iel_axis, idof_axis,
                                    from_element_indices,
                                    dof_pick_lists)
                     for name, bnd in expr.bindings.items()}),
                var_to_reduction_descr=expr.var_to_reduction_descr,
                tags=expr.tags,
                axes=expr.axes
            )
        else:
            return IndexLambda(
                expr.expr,
                expr.shape,
                expr.dtype,
                Map({name: self.rec(bnd, None, None, (), ())
                     for name, bnd in expr.bindings.items()}),
                var_to_reduction_descr=expr.var_to_reduction_descr,
                tags=expr.tags,
                axes=expr.axes
            )

    def map_contiguous_advanced_index(self,
                                      expr: AdvancedIndexInContiguousAxes,
                                      iel_axis: Optional[int],
                                      idof_axis: Optional[int],
                                      from_element_indices: Tuple[Array, ...],
                                      dof_pick_lists: Tuple[Array, ...],
                                      ) -> Array:
        if _is_materialized(expr):
            # Stop propagating indirections and return indirections collected till
            # this point.
            rec_expr = AdvancedIndexInContiguousAxes(
                self.rec(expr.array, None, None, (), ()),
                expr.indices,
                tags=expr.tags,
                axes=expr.axes)
            return _pick_list_fusers_map_materialized_node(
                rec_expr, iel_axis, idof_axis, from_element_indices, dof_pick_lists)

        if self.can_pick_indirections_be_propagated(expr,
                                                    iel_axis or 0,
                                                    idof_axis or 1):
            idx1, idx2 = expr.indices
            assert isinstance(idx1, Array) and isinstance(idx2, Array)
            return self.rec(expr.array, 0, 1,
                            from_element_indices + (idx1,),
                            dof_pick_lists + (idx2,))
        else:
            assert iel_axis is None and idof_axis is None
            return AdvancedIndexInContiguousAxes(
                self.rec(expr.array, iel_axis, idof_axis,
                         from_element_indices, dof_pick_lists),
                expr.indices,
                tags=expr.tags,
                axes=expr.axes
            )

    def map_einsum(self,
                   expr: Einsum,
                   iel_axis: int,
                   idof_axis: int,
                   from_element_indices: Tuple[Array, ...],
                   dof_pick_lists: Tuple[Array, ...]) -> Array:
        from pytato.array import EinsumElementwiseAxis

        if _is_materialized(expr):
            # Stop propagating indirections and return indirections collected till
            # this point.
            rec_expr = Einsum(expr.access_descriptors,
                              args=tuple(self.rec(arg, None, None, (), ())
                                         for arg in expr.args),
                              redn_axis_to_redn_descr=expr.redn_axis_to_redn_descr,
                              index_to_access_descr=expr.index_to_access_descr,
                              tags=expr.tags,
                              axes=expr.axes)
            return _pick_list_fusers_map_materialized_node(
                rec_expr, iel_axis, idof_axis, from_element_indices, dof_pick_lists)

        if iel_axis is not None:
            assert idof_axis is not None
            new_args: List[Array] = []
            for arg, acc_descrs in zip(expr.args, expr.access_descriptors):
                arg_iel_axis = acc_descrs.index(EinsumElementwiseAxis(iel_axis))
                arg_idof_axis = acc_descrs.index(EinsumElementwiseAxis(idof_axis))
                new_args.append(
                    self.rec(arg, arg_iel_axis, arg_idof_axis,
                             from_element_indices, dof_pick_lists)
                )
            return Einsum(expr.access_descriptors,
                          args=tuple(new_args),
                          redn_axis_to_redn_descr=expr.redn_axis_to_redn_descr,
                          index_to_access_descr=expr.index_to_access_descr,
                          tags=expr.tags,
                          axes=expr.axes)
        else:
            assert idof_axis is None
            return Einsum(expr.access_descriptors,
                          args=tuple(self.rec(arg, None, None, (), ())
                                     for arg in expr.args),
                          redn_axis_to_redn_descr=expr.redn_axis_to_redn_descr,
                          index_to_access_descr=expr.index_to_access_descr,
                          tags=expr.tags,
                          axes=expr.axes)

    def map_stack(self, expr: Stack, iel_axis: int, idof_axis: int,
                  from_element_indices: Tuple[Array, ...],
                  dof_pick_lists: Tuple[Array, ...],
                  ) -> Array:

        if _is_materialized(expr):
            # Stop propagating indirections and return indirections collected till
            # this point.
            rec_expr = Stack(tuple(self.rec(ary, None, None, (), ())
                                   for ary in expr.arrays),
                             expr.axis,
                             tags=expr.tags,
                             axes=expr.axes)
            return _pick_list_fusers_map_materialized_node(
                rec_expr, iel_axis, idof_axis, from_element_indices, dof_pick_lists)

        if iel_axis is not None:
            assert idof_axis is not None
            if iel_axis < expr.axis:
                assert idof_axis < expr.axis
                return Stack(tuple(self.rec(ary, iel_axis, idof_axis,
                                            from_element_indices, dof_pick_lists)
                                   for ary in expr.arrays),
                             expr.axis,
                             tags=expr.tags,
                             axes=expr.axes)
            else:
                assert idof_axis > expr.axis
                return Stack(tuple(self.rec(ary, iel_axis-1, idof_axis-1,
                                            from_element_indices, dof_pick_lists)
                                   for ary in expr.arrays),
                             expr.axis,
                             tags=expr.tags,
                             axes=expr.axes)
                return self.rec(expr.array, iel_axis-1, idof_axis-1,
                                from_element_indices, dof_pick_lists)
        else:
            assert idof_axis is None
            return Stack(tuple(self.rec(ary, iel_axis, idof_axis,
                                        from_element_indices, dof_pick_lists)
                               for ary in expr.arrays),
                         expr.axis,
                         tags=expr.tags,
                         axes=expr.axes)

    def map_concatenate(self,
                        expr: Concatenate,
                        iel_axis: Optional[int],
                        idof_axis: Optional[int],
                        from_element_indices: Tuple[Array, ...],
                        dof_pick_lists: Tuple[Array, ...],
                        ) -> Array:
        if _is_materialized(expr):
            # Stop propagating indirections and return indirections collected till
            # this point.
            rec_expr = Concatenate(tuple(self.rec(ary, None, None, (), ())
                                         for ary in expr.arrays),
                                   expr.axis,
                                   tags=expr.tags,
                                   axes=expr.axes)
            return _pick_list_fusers_map_materialized_node(
                rec_expr, iel_axis, idof_axis, from_element_indices, dof_pick_lists)

        return Concatenate(tuple(self.rec(ary, iel_axis, idof_axis,
                                          from_element_indices, dof_pick_lists)
                                 for ary in expr.arrays),
                           expr.axis,
                           tags=expr.tags,
                           axes=expr.axes)

    def map_roll(self,
                 expr: Roll,
                 iel_axis: Optional[int],
                 idof_axis: Optional[int],
                 from_element_indices: Tuple[Array, ...],
                 dof_pick_lists: Tuple[Array, ...]) -> Array:

        rec_expr = Roll(self.rec(expr.array, None, None, (), ()),
                        expr.shift,
                        expr.axis,
                        tags=expr.tags,
                        axes=expr.axes)
        if _is_materialized(expr):
            return _pick_list_fusers_map_materialized_node(
                rec_expr, iel_axis, idof_axis, from_element_indices, dof_pick_lists)

        assert iel_axis is None and idof_axis is None
        return rec_expr

    def map_non_contiguous_advanced_index(self,
                                          expr: AdvancedIndexInNoncontiguousAxes,
                                          iel_axis: Optional[int],
                                          idof_axis: Optional[int],
                                          from_element_indices: Tuple[Array, ...],
                                          dof_pick_lists: Tuple[Array, ...]
                                          ) -> Array:
        rec_expr = AdvancedIndexInNoncontiguousAxes(
            self.rec(expr.array, None, None, (), ()),
            expr.indices,
            tags=expr.tags,
            axes=expr.axes
        )

        if _is_materialized(expr):
            return _pick_list_fusers_map_materialized_node(
                rec_expr, iel_axis, idof_axis, from_element_indices, dof_pick_lists)

        assert iel_axis is None and idof_axis is None
        return rec_expr

    def map_basic_index(self, expr: BasicIndex,
                        iel_axis: Optional[int],
                        idof_axis: Optional[int],
                        from_element_indices: Tuple[Array, ...],
                        dof_pick_lists: Tuple[Array, ...]) -> Array:

        rec_expr = BasicIndex(
            self.rec(expr.array, None, None, (), ()),
            expr.indices,
            tags=expr.tags,
            axes=expr.axes
        )

        if _is_materialized(expr):
            return _pick_list_fusers_map_materialized_node(
                rec_expr, iel_axis, idof_axis, from_element_indices, dof_pick_lists)

        assert iel_axis is None and idof_axis is None
        return rec_expr

    def map_reshape(self,
                    expr: Reshape,
                    iel_axis: Optional[int],
                    idof_axis: Optional[int],
                    from_element_indices: Tuple[Array, ...],
                    dof_pick_lists: Tuple[Array, ...]) -> Array:
        rec_expr = Reshape(
            self.rec(expr.array, None, None, (), ()),
            expr.newshape,
            expr.order,
            tags=expr.tags,
            axes=expr.axes
        )

        if _is_materialized(expr):
            return _pick_list_fusers_map_materialized_node(
                rec_expr, iel_axis, idof_axis, from_element_indices, dof_pick_lists)

        assert iel_axis is None and idof_axis is None
        return rec_expr


def fuse_dof_pick_lists(expr: DictOfNamedArrays) -> DictOfNamedArrays:
    mapper = PickListFusers()

    return DictOfNamedArrays(
        {name: mapper(subexpr, None, None, (), ())
         for name, subexpr in sorted(expr._data.items(), key=lambda x: x[0])}
    )

# }}}


# {{{ fold indirection constants

class _ConstantIndirectionArrayCollector(CombineMapper[FrozenSet[Array]]):
    def __init__(self) -> None:
        from pytato.transform import InputGatherer
        super().__init__()
        self.get_inputs = InputGatherer()

    def combine(self, *args: FrozenSet[Array]) -> FrozenSet[Array]:
        from functools import reduce
        return reduce(frozenset.union, args, frozenset())

    def _map_input_base(self, expr: InputArgumentBase) -> FrozenSet[Array]:
        return frozenset()

    map_placeholder = _map_input_base
    map_data_wrapper = _map_input_base
    map_size_param = _map_input_base

    def _map_index_base(self, expr: IndexBase) -> FrozenSet[Array]:
        rec_results: List[FrozenSet[Array]] = []

        rec_results.append(self.rec(expr.array))

        for idx in expr.indices:
            if isinstance(idx, Array):
                if any(isinstance(inp, Placeholder)
                       for inp in self.get_inputs(idx)):
                    rec_results.append(self.rec(idx))
                else:
                    rec_results.append(frozenset([idx]))

        return self.combine(*rec_results)


def fold_constant_indirections(
        expr: MappedT,
        evaluator: Callable[[DictOfNamedArrays], Mapping[str, DataWrapper]]
) -> MappedT:
    """
    Returns a copy of *expr* with constant indirection expressions frozen.

    :arg evaluator: A callable that takes in a
        :class:`~pytato.array.DictOfNamedArrays` and returns a mapping from the
        name of every named array to it's corresponding evaluated array as an
        instance of :class:`~pytato.array.DataWrapper`.
    """
    from pytools import UniqueNameGenerator
    from pytato.array import make_dict_of_named_arrays
    import collections.abc as abc
    from pytato.transform import map_and_copy

    vng = UniqueNameGenerator()
    arys_to_evaluate = _ConstantIndirectionArrayCollector()(expr)
    dict_of_named_arrays = make_dict_of_named_arrays(
        {vng("_pt_folded_cnst"): ary for ary in arys_to_evaluate}
    )
    del arys_to_evaluate
    evaluated_arys = evaluator(dict_of_named_arrays)

    if not isinstance(evaluated_arys, abc.Mapping):
        raise TypeError("evaluator did not return a mapping")

    if set(evaluated_arys.keys()) != set(dict_of_named_arrays.keys()):
        raise ValueError("evaluator must return a mapping with "
                         f"the keys: '{set(dict_of_named_arrays.keys())}'.")

    for key, ary in evaluated_arys.items():
        if not isinstance(ary, DataWrapper):
            raise TypeError(f"evaluated array for '{key}' not a DataWrapper")

    before_to_after_subst = {
        dict_of_named_arrays._data[name]: evaluated_ary
        for name, evaluated_ary in evaluated_arys.items()
    }

    def _replace_with_folded_constants(subexpr: ArrayOrNames) -> ArrayOrNames:
        if isinstance(subexpr, Array):
            return before_to_after_subst.get(subexpr, subexpr)
        else:
            return subexpr

    return map_and_copy(expr, _replace_with_folded_constants)

# }}}


# vim: fdm=marker
