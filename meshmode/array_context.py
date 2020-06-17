from __future__ import division

__copyright__ = "Copyright (C) 2020 Andreas Kloeckner"

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
from loopy.version import MOST_RECENT_LANGUAGE_VERSION
from pytools import memoize_method

__doc__ = """
.. autofunction:: make_loopy_program
.. autoclass:: ArrayContext
.. autoclass:: PyOpenCLArrayContext
"""


def make_loopy_program(domains, statements, kernel_data=["..."], name=None):
    """Return a :class:`loopy.Program` suitable for use with
    :meth:`ArrayContext.call_loopy`.
    """
    return lp.make_kernel(
            domains,
            statements,
            kernel_data=kernel_data,
            options=lp.Options(
                no_numpy=True,
                return_dict=True),
            name=name,
            lang_version=MOST_RECENT_LANGUAGE_VERSION)


# {{{ ArrayContext

class ArrayContext:
    """An interface that allows a :class:`Discretization` to create and interact with
    arrays of degrees of freedom without fully specifying their types.

    .. automethod:: empty
    .. automethod:: zeros
    .. automethod:: empty_like
    .. automethod:: zeros_like
    .. automethod:: from_numpy
    .. automethod:: to_numpy
    .. automethod:: call_loopy
    .. automethod:: special_func
    .. automethod:: freeze
    .. automethod:: thaw

    .. versionadded:: 2020.2
    """

    def empty(self, shape, dtype):
        raise NotImplementedError

    def zeros(self, shape, dtype):
        raise NotImplementedError

    def empty_like(self, ary):
        return self.empty(shape=ary.shape, dtype=ary.dtype)

    def zeros_like(self, ary):
        return self.zeros(shape=ary.shape, dtype=ary.dtype)

    def from_numpy(self, array: np.ndarray):
        """
        :returns: the :class:`numpy.ndarray` *array* converted to the
            array context's array type.
        """
        raise NotImplementedError

    def to_numpy(self, array):
        """
        :returns: *array*, an array recognized by the context, converted
            to a :class:`numpy.ndarray`.
        """
        raise NotImplementedError

    def call_loopy(self, program, **kwargs):
        """Execute the :mod:`loopy` program *program* on the arguments
        *kwargs*.

        *program* is a :class:`loopy.LoopKernel` or :class:`loopy.Program`.
        It is expected to not yet be transformed for execution speed.
        It must have :class:`loopy.Options.return_dict` set.

        :return: a :class:`dict` of outputs from the program, each an
            array understood by the context.
        """
        raise NotImplementedError

    @memoize_method
    def _get_special_func_loopy_program(self, name, nargs):
        from pymbolic import var
        iel = var("iel")
        idof = var("idof")
        return make_loopy_program(
                "{[iel, idof]: 0<=iel<nelements and 0<=idof<ndofs}",
                [
                    lp.Assignment(
                        var("out")[iel, idof],
                        var(name)(*[
                            var("inp%d" % i)[iel, idof] for i in range(nargs)]))
                    ],
                name="actx_special_%s" % name)

    @memoize_method
    def special_func(self, name):
        """Returns a callable for the special function *name*, where *name* is a
        (potentially dotted) function name resolvable by :mod:`loopy`.

        The returned callable will vectorize over object arrays, including
        :class:`meshmode.dof_array.DOFArray`.
        """
        def f(*args):
            # FIXME: Maybe involve loopy type inference?
            result = self.empty(args[0].shape, args[0].dtype)
            prg = self._get_special_func_loopy_program(name, len(args))
            self.call_loopy(prg, out=result,
                    **{"inp%d" % i: arg for i, arg in enumerate(args)})
            return result

        from pytools.obj_array import obj_array_vectorized_n_args
        return obj_array_vectorized_n_args(f)

    def freeze(self, array):
        """Return a version of the context-defined array *array* that is
        'frozen', i.e. suitable for long-term storage and reuse. Frozen arrays
        do not support arithmetic. For example, in the context of
        OpenCL, this might entail stripping the array of an associated queue,
        whereas in a lazily-evaluated context, it might mean that the array is
        evaluated and stored.

        Freezing makes the array independent of this :class:`ArrayContext`;
        it is permitted to :meth:`thaw` it in a different one, as long as that
        context understands the array format.
        """
        raise NotImplementedError

    def thaw(self, array):
        """Take a 'frozen' array
        """
        raise NotImplementedError

# }}}


# {{{ PyOpenCLArrayContext

class PyOpenCLArrayContext(ArrayContext):
    """
    A :class:`ArrayContext` that uses :class:`pyopencl.array.Array` instances
    for DOF arrays.

    .. attribute:: context

        A :class:`pyopencl.Context`.

    .. attribute:: queue

        A :class:`pyopencl.CommandQueue`.

    .. attribute:: allocator
    """

    def __init__(self, queue, allocator=None):
        self.context = queue.context
        self.queue = queue
        self.allocator = allocator

    # {{{ ArrayContext interface

    def empty(self, shape, dtype):
        import pyopencl.array as cla
        return cla.empty(self.queue, shape=shape, dtype=dtype,
                allocator=self.allocator)

    def zeros(self, shape, dtype):
        import pyopencl.array as cla
        return cla.zeros(self.queue, shape=shape, dtype=dtype,
                allocator=self.allocator)

    def from_numpy(self, np_array: np.ndarray):
        import pyopencl.array as cla
        return cla.to_device(self.queue, np_array, allocator=self.allocator)

    def to_numpy(self, array):
        return array.get(queue=self.queue)

    def call_loopy(self, program, **kwargs):
        program = self.transform_loopy_program(program)
        assert program.options.return_dict
        assert program.options.no_numpy

        evt, result = program(self.queue, **kwargs, allocator=self.allocator)
        return result

    def freeze(self, array):
        array.finish()
        return array.with_queue(None)

    def thaw(self, array):
        return array.with_queue(self.queue)

    # }}}

    @memoize_method
    def transform_loopy_program(self, program):
        # FIXME: This assumes that the iname 'iel' exists.
        # FIXME: This could be much smarter.
        import loopy as lp
        if "idof" in program.all_inames():
            program = lp.split_iname(program, "idof", 16, inner_tag="l.0")
        return lp.tag_inames(program, dict(iel="g.0"))

# }}}


# vim: foldmethod=marker
