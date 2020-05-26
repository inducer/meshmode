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
.. autoclass:: ArrayContext
.. autoclass:: PyOpenCLArrayContext
"""


# {{{ ArrayContext

class ArrayContext:
    """An interface that allows a :class:`Discretization` to create and interact with
    arrays of degrees of freedom without fully specifying their types.

    .. automethod:: empty
    .. automethod:: zeros
    .. automethod:: from_numpy_constant
    .. automethod:: from_numpy_data
    .. automethod:: to_numpy
    .. automethod:: call_loopy
    .. automethod:: finalize
    """

    def empty(self, shape, dtype):
        raise NotImplementedError

    def zeros(self, shape, dtype):
        raise NotImplementedError

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

    def call_loopy(self, program, **args):
        """Execute the :mod:`loopy` program *program* on the arguments
        *args*.

        *program* is a :class:`loopy.LoopKernel` or :class:`loopy.Program`.
        It is expected to not yet be transformed for execution speed.
        It must have :class:`loopy.Options.return_dict` set.

        :return: a :class:`dict` of outputs from the program, each an
        array understood by the context.
        """
        raise NotImplementedError

    def finalize(self, array):
        """Return a version of the context-defined array *array* that
        is 'finalized', i.e. suitable for long-term storage and reuse.
        For example, in the context of OpenCL, this might entail
        stripping the array of an associated queue, whereas in a
        lazily-evaluated context, it might mean that the array is
        evaluated and stored.
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

        A :class:`pyopencl.CommandQueue` or *None*.

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

    def call_loopy(self, program, **args):
        program = self.transform_loopy_program(program)
        assert program.options.return_dict
        assert program.options.no_numpy

        evt, result = program(self.queue, **args)
        return result

    def finalize(self, array):
        array.finish()
        return array.with_queue(None)

    # }}}

    @memoize_method
    def transform_loopy_program(self, program):
        # FIXME: This assumes that inames 'idof' and 'iel' exist.
        # FIXME: This could be much smarter.
        import loopy as lp
        program = lp.split_iname(program, "idof", 16, inner_tag="l.0")
        return lp.tag_inames(program, dict(iel="g.0"))

# }}}


def make_loopy_program(domains, statements, name=None):
    return lp.make_kernel(
            domains,
            statements,
            options=lp.Options(
                no_numpy=True,
                return_dict=True),
            name=name,
            lang_version=MOST_RECENT_LANGUAGE_VERSION)


# vim: foldmethod=marker
