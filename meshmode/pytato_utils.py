from functools import partial, reduce
from arraycontext.impl.pytato.fake_numpy import PytatoFakeNumpyNamespace
from arraycontext import rec_map_reduce_array_container
import pyopencl.array as cl_array


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

# vim: fdm=marker
