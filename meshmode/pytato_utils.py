import pyopencl.array as cl_array
import pytato as pt
import logging

from functools import partial, reduce
from arraycontext.impl.pytato.fake_numpy import PytatoFakeNumpyNamespace
from arraycontext import rec_map_reduce_array_container
from meshmode.transform_metadata import DiscretizationEntityAxisTag
from pytato.transform import ArrayOrNames
from pytato.transform.metadata import (
    AxesTagsEquationCollector as BaseAxesTagsEquationCollector)
from arraycontext import ArrayContainer
from arraycontext.container.traversal import rec_map_array_container
from typing import Union
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

class AxesTagsEquationCollector(BaseAxesTagsEquationCollector):
    def map_reshape(self, expr: pt.Reshape) -> None:
        super().map_reshape(expr)

        if (expr.size > 0
                and (1 not in (expr.array.shape))  # leads to ambiguous newaxis
                and (set(expr.shape) <= (set(expr.array.shape) | {1}))):
            i_in_axis = 0
            for i_out_axis, dim in enumerate(expr.shape):
                if dim != 1:
                    assert dim == expr.array.shape[i_in_axis]
                    self.record_equation(
                                    self.get_var_for_axis(expr.array,
                                                          i_in_axis),
                                    self.get_var_for_axis(expr,
                                                          i_out_axis)
                    )
                    i_in_axis += 1
        else:
            # print(f"Skipping: {expr.array.shape} -> {expr.shape}")
            # Wacky reshape => bail.
            pass


def unify_discretization_entity_tags(expr: Union[ArrayContainer, ArrayOrNames]
                                     ) -> ArrayOrNames:
    if not isinstance(expr, (pt.Array, pt.DictOfNamedArrays)):
        return rec_map_array_container(unify_discretization_entity_tags,
                                       expr)

    return pt.unify_axes_tags(expr,
                              tag_t=DiscretizationEntityAxisTag,
                              equations_collector_t=AxesTagsEquationCollector)

# }}}


# vim: fdm=marker
