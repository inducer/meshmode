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
import operator
from functools import partialmethod
from numbers import Number
import time

__doc__ = """
.. autofunction:: make_loopy_program
.. autoclass:: ArrayContext
.. autoclass:: PyOpenCLArrayContext
"""


def make_loopy_program(domains, statements, kernel_data=["..."],
        name="mm_actx_kernel"):
    """Return a :class:`loopy.LoopKernel` suitable for use with
    :meth:`ArrayContext.call_loopy`.
    """
    return lp.make_function(
            domains,
            statements,
            kernel_data=kernel_data,
            options=lp.Options(
                no_numpy=True,
                return_dict=True),
            # FIXME: Not sure why is this here.
            # default_offset=lp.auto,
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

    @obj_array_vectorized_n_args
    def conjugate(self, x):
        # NOTE: conjugate distribute over object arrays, but it looks for a
        # `conjugate` ufunc, while some implementations only have the shorter
        # `conj` (e.g. cl.array.Array), so this should work for everybody
        return x.conj()

    conj = conjugate


class ArrayContext:
    """An interface that allows a
    :class:`~meshmode.discretization.Discretization` to create and interact
    with arrays of degrees of freedom without fully specifying their types.

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

        *program* is a :class:`loopy.LoopKernel` or :class:`loopy.LoopKernel`.
        It is expected to not yet be transformed for execution speed.
        It must have :attr:`loopy.Options.return_dict` set.

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
    @obj_array_vectorized_n_args
    def maximum(self, x, y):
        import pyopencl.array as cl_array
        return cl_array.maximum(x, y)

    @obj_array_vectorized_n_args
    def minimum(self, x, y):
        import pyopencl.array as cl_array
        return cl_array.minimum(x, y)

    @obj_array_vectorized_n_args
    def where(self, criterion, then, else_):
        import pyopencl.array as cl_array
        return cl_array.if_positive(criterion != 0, then, else_)

    @obj_array_vectorized_n_args
    def reshape(self, a, newshape):
        import pyopencl.array as cl_array
        return cl_array.reshape(a, newshape)

    @obj_array_vectorized_n_args
    def concatenate(self, arrays, axis=0):
        import pyopencl.array as cl_array
        return cl_array.concatenate(arrays, axis,
                self._array_context.queue, self._array_context.allocator)


class PyOpenCLArrayContext(ArrayContext):
    """
    A :class:`ArrayContext` that uses :class:`pyopencl.array.Array` instances
    for DOF arrays.

    .. attribute:: context

        A :class:`pyopencl.Context`.

    .. attribute:: queue

        A :class:`pyopencl.CommandQueue`.

    .. attribute:: allocator

        A PyOpenCL memory allocator. Can also be `None` (default) or `False` to
        use the default allocator. Please note that running with the default
        allocator allocates and deallocates OpenCL buffers directly. If lots
        of arrays are created (e.g. as results of computation), the associated cost
        may become significant. Using e.g. :class:`pyopencl.tools.MemoryPool`
        as the allocator can help avoid this cost.
    """

    def __init__(self, queue, allocator=None, wait_event_queue_length=None):
        r"""
        :arg wait_event_queue_length: The length of a queue of
            :class:`~pyopencl.Event` objects that are maintained by the
            array context, on a per-kernel-name basis. The events returned
            from kernel execution are appended to the queue, and Once the
            length of the queue exceeds *wait_event_queue_length*, the
            first event in the queue :meth:`pyopencl.Event.wait`\ ed on.

            *wait_event_queue_length* may be set to *False* to disable this feature.

            The use of *wait_event_queue_length* helps avoid enqueuing
            large amounts of work (and, potentially, allocating large amounts
            of memory) far ahead of the actual OpenCL execution front,
            by limiting the number of each type (name, really) of kernel
            that may reside unexecuted in the queue at one time.

        .. note::

            For now, *wait_event_queue_length* should be regarded as an
            experimental feature that may change or disappear at any minute.
        """
        super().__init__()
        self.context = queue.context
        self.queue = queue
        self.allocator = allocator if allocator else None

        if wait_event_queue_length is None:
            wait_event_queue_length = 10

        self._wait_event_queue_length = wait_event_queue_length
        self._kernel_name_to_wait_event_queue = {}

        import pyopencl as cl
        if allocator is None and queue.device.type & cl.device_type.GPU:
            from warnings import warn
            warn("PyOpenCLArrayContext created without an allocator on a GPU. "
                 "This can lead to high numbers of memory allocations. "
                 "Please consider using a pyopencl.tools.MemoryPool. "
                 "Run with allocator=False to disable this warning.")

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
        if not (options.return_dict and options.no_numpy):
            raise ValueError("Loopy program passed to call_loopy must "
                    "have return_dict and no_numpy options set. "
                    "Did you use meshmode.array_context.make_loopy_program "
                    "to create this program?")

        evt, result = program(self.queue, **kwargs, allocator=self.allocator)

        if self._wait_event_queue_length is not False:
            wait_event_queue = self._kernel_name_to_wait_event_queue.setdefault(
                    program.name, [])

            wait_event_queue.append(evt)
            if len(wait_event_queue) > self._wait_event_queue_length:
                wait_event_queue.pop(0).wait()

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
        elif "iel" in all_inames:
            outer_iname = "iel"

            if "idof" in all_inames:
                inner_iname = "idof"
        else:
            # cannot "fit" the optimization strategy for the provided kernel
            # => bail
            return program

        if inner_iname is not None:
            program = lp.split_iname(program, inner_iname, 16, inner_tag="l.0")
        return lp.tag_inames(program, {outer_iname: "g.0"})

# }}}


# {{{ pytest integration

def pytest_generate_tests_for_pyopencl_array_context(metafunc):
    import pyopencl as cl
    from pyopencl.tools import _ContextFactory

    class ArrayContextFactory(_ContextFactory):
        def __call__(self):
            ctx = super().__call__()
            return PyOpenCLArrayContext(
                    cl.CommandQueue(ctx),
                    # CI machines are often quite limited in their memory.
                    # Avoid enqueueing ahead by too much. See
                    # https://github.com/inducer/grudge/pull/23
                    # for the saga that led to this. Bring popcorn.
                    wait_event_queue_length=2)

        def __str__(self):
            return ("<array context factory for <pyopencl.Device '%s' on '%s'>" %
                    (self.device.name.strip(),
                     self.device.platform.name.strip()))

    import pyopencl.tools as cl_tools
    arg_names = cl_tools.get_pyopencl_fixture_arg_names(
            metafunc, extra_arg_names=["actx_factory"])

    if not arg_names:
        return

    arg_values, ids = cl_tools.get_pyopencl_fixture_arg_values()
    if "actx_factory" in arg_names:
        if "ctx_factory" in arg_names or "ctx_getter" in arg_names:
            raise RuntimeError("Cannot use both an 'actx_factory' and a "
                    "'ctx_factory' / 'ctx_getter' as arguments.")

        for arg_dict in arg_values:
            arg_dict["actx_factory"] = ArrayContextFactory(arg_dict["device"])

    arg_values = [
            tuple(arg_dict[name] for name in arg_names)
            for arg_dict in arg_values
            ]

    metafunc.parametrize(arg_names, arg_values, ids=ids)

# }}}


# {{{ PytatoArrayContext

class _PytatoFakeNumpyNamespace(_BaseFakeNumpyNamespace):

    @property
    def ns(self):
        return self._array_context.ns

    @obj_array_vectorized_n_args
    def exp(self, x):
        import pytato as pt
        return pt.exp(x)

    @obj_array_vectorized_n_args
    def reshape(self, a, newshape):
        import pytato as pt
        return pt.reshape(a, newshape)

    @obj_array_vectorized_n_args
    def concatenate(self, arrays, axis=0):
        import pytato as pt
        return pt.concatenate(arrays, axis)


class PytatoArrayContext(ArrayContext):
    """
    A :class:`ArrayContext` that uses :mod:`pytato` data types to represent
    the DOF arrays targeting OpenCL for offloading operations.

    .. attribute:: context

        A :class:`pyopencl.Context`.

    .. attribute:: queue

        A :class:`pyopencl.CommandQueue`.
    """

    def __init__(self, ctx, queue=None):
        import pytato as pt
        import pyopencl as cl
        super().__init__()
        self.ns = pt.Namespace()
        self.opencl_ctx = ctx
        if queue is None:
            queue = cl.CommandQueue(ctx)

        self.queue = queue
        self.np = self._get_fake_numpy_namespace()

    def _get_fake_numpy_namespace(self):
        return _PytatoFakeNumpyNamespace(self)

    # {{{ ArrayContext interface

    def empty(self, shape, dtype):
        raise NotImplementedError("Debugging phase: should not call empty")
        import pytato as pt
        return pt.make_placeholder(self.ns, name=self.ns.name_gen("u"),
                shape=shape, dtype=dtype)

    def symbolic_array_var(self, shape, dtype, name=None):
        import pytato as pt
        return pt.make_placeholder(self.ns, shape=shape, dtype=dtype, name=name)

    def zeros(self, shape, dtype):
        import pytato as pt
        return pt.zeros(self.ns, shape, dtype)

    def from_numpy(self, np_array: np.ndarray):
        import pytato as pt
        import pyopencl.array as cla
        cl_array = cla.to_device(self.queue, np_array)
        return pt.make_data_wrapper(self.ns, cl_array)

    def to_numpy(self, array):
        cl_array = self.freeze(array)
        return cl_array.get(queue=self.queue)

    def call_loopy(self, program, **kwargs):
        entrypoint, = set(program.callables_table)
        from pytato.loopy import call_loopy
        return call_loopy(self.ns, program, kwargs, entrypoint)

    def freeze(self, array):
        import pytato as pt

        if isinstance(array, pt.Placeholder):
            raise ValueError("freezing placeholder would return garbage valued"
                    " arrays")
        if not isinstance(array, pt.Array):
            raise TypeError("PytatoArrayContext.freeze invoked with non-pt arrays")

        prg = pt.generate_loopy(array, target=pt.PyOpenCLTarget(self.queue))
        evt, (cl_array,) = prg()
        evt.wait()

        return cl_array.with_queue(None)

    def thaw(self, array):
        import pytato as pt
        import pyopencl.array as cla

        if not isinstance(array, cla.Array):
            raise TypeError("PytatoArrayContext.thaw expects CL arrays, got "
                    f"{type(array)}")

        return pt.make_data_wrapper(self.ns, array.with_queue(self.queue))

    # }}}

# }}}


# {{{ DebugArrayContext

class _DebugFakeNumpyNamespace(_BaseFakeNumpyNamespace):
    @property
    def actx_np(self):
        return self._array_context.actx.np

    @property
    def ref_actx_np(self):
        return self._array_context.ref_actx.np

    @obj_array_vectorized_n_args
    def exp(self, x):
        ary = self.actx_np.exp(x.array)
        ref_ary = self.ref_actx_np.exp(x.ref_array)

        result_dbg_ary = DebugArray(x.actx, ary,
                                    x.ref_actx, ref_ary)

        # DEBUG #
        result_dbg_ary.debug()

        return result_dbg_ary

    @obj_array_vectorized_n_args
    def reshape(self, a, newshape):
        ary = self.actx_np.reshape(a.array, newshape)
        ref_ary = self.ref_actx_np.reshape(a.ref_array, newshape)

        result_dbg_ary = DebugArray(a.actx, ary,
                                    a.ref_actx, ref_ary)

        # DEBUG #
        result_dbg_ary.debug()

        return result_dbg_ary

    @obj_array_vectorized_n_args
    def concatenate(self, arrays, axis=0):
        ary = self.actx_np.concatenate([array.array for array in arrays], axis)
        ref_ary = self.ref_actx_np.concatenate([array.ref_array
                                                for array in arrays], axis)

        result_dbg_ary = DebugArray(arrays[0].actx, ary,
                                    arrays[0].ref_actx, ref_ary)

        # DEBUG #
        result_dbg_ary.debug()

        return result_dbg_ary


class DebugArray:
    def __init__(self, actx, array, ref_actx, ref_array):
        self.array = array
        self.ref_array = ref_array
        self.actx = actx
        self.ref_actx = ref_actx

        assert isinstance(actx, ArrayContext)
        assert isinstance(ref_actx, ArrayContext)

    def __getitem__(self, key):
        ary = self.array[key]
        ref_ary = self.ref_array[key]

        result_dbg_ary = DebugArray(self.actx, ary,
                                    self.ref_actx, ref_ary)

        # DEBUG #
        result_dbg_ary.debug()

        return result_dbg_ary

    def debug(self):
        np_array = self.actx.to_numpy(self.array)
        np_ref_array = self.ref_actx.to_numpy(self.ref_array)
        np.testing.assert_almost_equal(np_array, np_ref_array)
        print(f"[{time.asctime()}]: Verified")

    def _binary_op(self, op, other, reverse=False):
        if isinstance(other, DebugArray):
            ary = op(self.array, other.array)
            ref_ary = op(self.ref_array, other.ref_array)
        else:
            assert isinstance(other, Number)
            ary = op(self.array, other)
            ref_ary = op(self.ref_array, other)

        result_dbg_ary = DebugArray(self.actx, ary,
                                    self.ref_actx, ref_ary)

        # DEBUG #
        result_dbg_ary.debug()

        return result_dbg_ary

    def _unary_op(self, op):
        ary = op(self.array)
        ref_ary = op(self.ref_array)

        result_dbg_ary = DebugArray(self.actx, ary,
                                    self.ref_actx, ref_ary)

        # DEBUG #
        result_dbg_ary.debug()

        return result_dbg_ary

    @property
    def shape(self):
        assert self.array.shape == self.ref_array.shape
        return self.array.shape

    @property
    def dtype(self):
        assert self.array.dtype == self.ref_array.dtype
        return self.array.dtype

    @property
    def size(self):
        assert self.array.size == self.ref_array.size
        return self.array.size

    __mul__ = partialmethod(_binary_op, operator.mul)
    __rmul__ = partialmethod(_binary_op, operator.mul, reverse=True)

    __add__ = partialmethod(_binary_op, operator.add)
    __radd__ = partialmethod(_binary_op, operator.add, reverse=True)

    __sub__ = partialmethod(_binary_op, operator.sub)
    __rsub__ = partialmethod(_binary_op, operator.sub, reverse=True)

    __truediv__ = partialmethod(_binary_op, operator.truediv)
    __rtruediv__ = partialmethod(_binary_op, operator.truediv, reverse=True)

    __pow__ = partialmethod(_binary_op, operator.pow)
    __rpow__ = partialmethod(_binary_op, operator.pow, reverse=True)

    __neg__ = partialmethod(_unary_op, operator.neg)


class DebugArrayContext(ArrayContext):
    """
    An array context to debug between 2 array contexts.

    .. attribute:: actx

        A :class:`meshmode.ArrayContext`.

    .. attribute:: ref_actx

        Reference :class:`meshmode.ArrayContext`.
    """

    def __init__(self, actx, ref_actx):
        super().__init__()
        self.actx = actx
        self.ref_actx = ref_actx

    def _get_fake_numpy_namespace(self):
        return _DebugFakeNumpyNamespace(self)

    # {{{ ArrayContext interface

    def empty(self, shape, dtype):
        return DebugArray(self.actx, self.actx.empty(shape, dtype),
                          self.ref_actx, self.ref_actx.empty(shape, dtype))

    def zeros(self, shape, dtype):
        return DebugArray(self.actx, self.actx.zeros(shape, dtype),
                          self.ref_actx, self.ref_actx.zeros(shape, dtype))

    def from_numpy(self, np_array: np.ndarray):
        return DebugArray(self.actx, self.actx.from_numpy(np_array),
                          self.ref_actx, self.ref_actx.from_numpy(np_array))

    def to_numpy(self, array):
        assert isinstance(array, DebugArray)

        # DEBUG #
        array.debug()

        return self.actx.to_numpy(array.array)

    def call_loopy(self, program, **kwargs):
        unmangled_kwargs = {key: val.array if isinstance(val, DebugArray) else val
                            for key, val in kwargs.items()}
        unmangled_ref_kwargs = {key: val.ref_array if isinstance(val, DebugArray)
                                     else val
                                for key, val in kwargs.items()}
        result = self.actx.call_loopy(program, **unmangled_kwargs)
        ref_result = self.ref_actx.call_loopy(program, **unmangled_ref_kwargs)

        assert result.keys() == ref_result.keys()

        dbg_array_result_dict = {}

        for key in result.keys():
            val = result[key]
            ref_val = ref_result[key]

            result_as_debug_array = DebugArray(self.actx, val,
                                               self.ref_actx, ref_val)

            # DEBUG #
            result_as_debug_array.debug()

            dbg_array_result_dict[key] = result_as_debug_array

        return dbg_array_result_dict

    def freeze(self, array):
        result = DebugArray(self.actx, self.actx.freeze(array.array),
                            self.ref_actx, self.ref_actx.freeze(array.ref_array))

        # DEBUG #
        result.debug()

        return result

    def thaw(self, array):
        result = DebugArray(self.actx, self.actx.thaw(array.array),
                            self.ref_actx, self.ref_actx.thaw(array.ref_array))

        # DEBUG #
        result.debug()

        return result

    # }}}

# }}}

# vim: foldmethod=marker
