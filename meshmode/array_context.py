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
from pytools.obj_array import obj_array_vectorized_n_args

__doc__ = """
.. autofunction:: make_loopy_program
.. autoclass:: ArrayContext
.. autoclass:: PyOpenCLArrayContext
"""


def make_loopy_program(domains, statements, kernel_data=["..."],
        name="mm_actx_kernel"):
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
            default_offset=lp.auto,
            name=name,
            lang_version=MOST_RECENT_LANGUAGE_VERSION)


# {{{ ArrayContext

class _BaseFakeNumpyNamespace:
    def __init__(self, array_context):
        self._array_context = array_context

    def __getattr__(self, name):
        def f(*args):
            actx = self._array_context
            # FIXME: Maybe involve loopy type inference?
            result = actx.empty(args[0].shape, args[0].dtype)
            prg = actx._get_scalar_func_loopy_program(
                    name, nargs=len(args), naxes=len(args[0].shape))
            actx.call_loopy(prg, out=result,
                    **{"inp%d" % i: arg for i, arg in enumerate(args)})
            return result

        return obj_array_vectorized_n_args(f)


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
    .. attribute:: np

         Provides access to a namespace that serves as a work-alike to
         :mod:`numpy`.  The actual level of functionality provided is up to the
         individual array context implementation, however the functions and
         objects available under this namespace must not behave differently
         from :mod:`numpy`.

         As a baseline, special functions available through :mod:`loopy`
         (e.g. ``sin``, ``exp``) are accessible through this interface.

         Callables accessible through this namespace vectorize over object
         arrays, including :class:`meshmode.dof_array.DOFArray`.

    .. automethod:: freeze
    .. automethod:: thaw

    .. versionadded:: 2020.2
    """

    def __init__(self):
        self.np = self._get_fake_numpy_namespace()

    def _get_fake_numpy_namespace(self):
        return _BaseFakeNumpyNamespace(self)

    def empty(self, shape, dtype):
        raise NotImplementedError

    def zeros(self, shape, dtype):
        raise NotImplementedError

    def empty_like(self, ary):
        return self.empty(shape=ary.shape, dtype=ary.dtype)

    def zeros_like(self, ary):
        return self.zeros(shape=ary.shape, dtype=ary.dtype)

    def from_numpy(self, array: np.ndarray):
        r"""
        :returns: the :class:`numpy.ndarray` *array* converted to the
            array context's array type. The returned array will be
            :meth:`thaw`\ ed.
        """
        raise NotImplementedError

    def to_numpy(self, array):
        r"""
        :returns: *array*, an array recognized by the context, converted
            to a :class:`numpy.ndarray`. *array* must be
            :meth:`thaw`\ ed.
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
    def _get_scalar_func_loopy_program(self, name, nargs, naxes):
        if name == "arctan2":
            name = "atan2"
        elif name == "atan2":
            from warnings import warn
            warn("'atan2' in ArrayContext.np is deprecated. Use 'arctan2', "
                    "as in numpy2. This will be disallowed in 2021.",
                    DeprecationWarning, stacklevel=3)

        from pymbolic import var

        var_names = ["i%d" % i for i in range(naxes)]
        size_names = ["n%d" % i for i in range(naxes)]
        subscript = tuple(var(vname) for vname in var_names)
        from islpy import make_zero_and_vars
        v = make_zero_and_vars(var_names, params=size_names)
        domain = v[0].domain()
        for vname, sname in zip(var_names, size_names):
            domain = domain & v[0].le_set(v[vname]) & v[vname].lt_set(v[sname])

        domain_bset, = domain.get_basic_sets()

        return make_loopy_program(
                [domain_bset],
                [
                    lp.Assignment(
                        var("out")[subscript],
                        var(name)(*[
                            var("inp%d" % i)[subscript] for i in range(nargs)]))
                    ],
                name="actx_special_%s" % name)

    def freeze(self, array):
        """Return a version of the context-defined array *array* that is
        'frozen', i.e. suitable for long-term storage and reuse. Frozen arrays
        do not support arithmetic. For example, in the context of
        :class:`~pyopencl.array.Array`, this might mean stripping the array
        of an associated command queue, whereas in a lazily-evaluated context,
        it might mean that the array is evaluated and stored.

        Freezing makes the array independent of this :class:`ArrayContext`;
        it is permitted to :meth:`thaw` it in a different one, as long as that
        context understands the array format.
        """
        raise NotImplementedError

    def thaw(self, array):
        """Take a 'frozen' array and return a new array representing the data in
        *array* that is able to perform arithmetic and other operations, using
        the execution resources of this context. In the context of
        :class:`~pyopencl.array.Array`, this might mean that the array is
        equipped with a command queue, whereas in a lazily-evaluated context,
        it might mean that the returned array is a symbol bound to
        the data in *array*.

        The returned array may not be used with other contexts while thawed.
        """
        raise NotImplementedError

# }}}


# {{{ PyOpenCLArrayContext

class _PyOpenCLFakeNumpyNamespace(_BaseFakeNumpyNamespace):
    def __getattr__(self, name):
        if name in ["minimum", "maximum"]:
            import pyopencl.array as cl_array
            return obj_array_vectorized_n_args(getattr(cl_array, name))

        return super().__getattr__(name)

    @obj_array_vectorized_n_args
    def where(self, criterion, then, else_):
        import pyopencl.array as cl_array
        return cl_array.if_positive(criterion != 0, then, else_)


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
        super().__init__()
        self.context = queue.context
        self.queue = queue
        self.allocator = allocator

    def _get_fake_numpy_namespace(self):
        return _PyOpenCLFakeNumpyNamespace(self)

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

        # accommodate loopy with and without kernel callables
        try:
            options = program.options
        except AttributeError:
            options = program.root_kernel.options
        assert options.return_dict
        assert options.no_numpy
        if not (options.return_dict and options.no_numpy):
            raise ValueError("Loopy program passed to call_loopy must "
                    "have return_dict and no_numpy options set. "
                    "Did you use meshmode.array_context.make_loopy_program "
                    "to create this program?")

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
        # FIXME: This could be much smarter.
        import loopy as lp
        # accommodate loopy with and without kernel callables
        try:
            all_inames = program.all_inames()
        except AttributeError:
            all_inames = program.root_kernel.all_inames()

        inner_iname = None
        if "iel" not in all_inames and "i0" in all_inames:
            outer_iname = "i0"

            if "i1" in all_inames:
                inner_iname = "i1"
        else:
            outer_iname = "iel"

            if "idof" in all_inames:
                inner_iname = "idof"

        if inner_iname is not None:
            program = lp.split_iname(program, inner_iname, 16, inner_tag="l.0")
        return lp.tag_inames(program, {outer_iname: "g.0"})

# }}}


# vim: foldmethod=marker
