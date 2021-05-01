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

from typing import Union, Sequence
from functools import partial
import operator
import numpy as np
import loopy as lp
from loopy.version import MOST_RECENT_LANGUAGE_VERSION
from pytools import memoize_method
from pytools.tag import Tag
from abc import ABC, abstractmethod


__doc__ = """
.. autofunction:: make_loopy_program
.. autoclass:: CommonSubexpressionTag
.. autoclass:: FirstAxisIsElementsTag
.. autoclass:: ArrayContext
.. autoclass:: PyOpenCLArrayContext
.. autofunction:: pytest_generate_tests_for_pyopencl_array_context
"""


_DEFAULT_LOOPY_OPTIONS = lp.Options(
        no_numpy=True,
        return_dict=True)


def make_loopy_program(domains, statements, kernel_data=None,
        name="mm_actx_kernel"):
    """Return a :class:`loopy.LoopKernel` suitable for use with
    :meth:`ArrayContext.call_loopy`.
    """
    if kernel_data is None:
        kernel_data = ["..."]

    return lp.make_kernel(
            domains,
            statements,
            kernel_data=kernel_data,
            options=_DEFAULT_LOOPY_OPTIONS,
            default_offset=lp.auto,
            name=name,
            lang_version=MOST_RECENT_LANGUAGE_VERSION)


def _loopy_get_default_entrypoint(t_unit):
    try:
        # main and "kernel callables" branch
        return t_unit.default_entrypoint
    except AttributeError:
        try:
            return t_unit.root_kernel
        except AttributeError:
            raise TypeError("unable to find default entry point for loopy "
                    "translation unit")


# {{{ ArrayContext

class _BaseFakeNumpyNamespace:
    def __init__(self, array_context):
        self._array_context = array_context
        self.linalg = self._get_fake_numpy_linalg_namespace()

    def _get_fake_numpy_linalg_namespace(self):
        return _BaseFakeNumpyLinalgNamespace(self.array_context)

    _numpy_math_functions = frozenset({
        # https://numpy.org/doc/stable/reference/routines.math.html

        # FIXME: Heads up: not all of these are supported yet.
        # But I felt it was important to only dispatch actually existing
        # numpy functions to loopy.

        # Trigonometric functions
        "sin", "cos", "tan", "arcsin", "arccos", "arctan", "hypot", "arctan2",
        "degrees", "radians", "unwrap", "deg2rad", "rad2deg",

        # Hyperbolic functions
        "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh",

        # Rounding
        "around", "round_", "rint", "fix", "floor", "ceil", "trunc",

        # Sums, products, differences

        # FIXME: Many of These are reductions or scans.
        # "prod", "sum", "nanprod", "nansum", "cumprod", "cumsum", "nancumprod",
        # "nancumsum", "diff", "ediff1d", "gradient", "cross", "trapz",

        # Exponents and logarithms
        "exp", "expm1", "exp2", "log", "log10", "log2", "log1p", "logaddexp",
        "logaddexp2",

        # Other special functions
        "i0", "sinc",

        # Floating point routines
        "signbit", "copysign", "frexp", "ldexp", "nextafter", "spacing",
        # Rational routines
        "lcm", "gcd",

        # Arithmetic operations
        "add", "reciprocal", "positive", "negative", "multiply", "divide", "power",
        "subtract", "true_divide", "floor_divide", "float_power", "fmod", "mod",
        "modf", "remainder", "divmod",

        # Handling complex numbers
        "angle", "real", "imag",
        # Implemented below:
        # "conj", "conjugate",

        # Miscellaneous
        "convolve", "clip", "sqrt", "cbrt", "square", "absolute", "abs", "fabs",
        "sign", "heaviside", "maximum", "fmax", "nan_to_num",

        # FIXME:
        # "interp",

        })

    _numpy_to_c_arc_functions = {
            "arcsin": "asin",
            "arccos": "acos",
            "arctan": "atan",
            "arctan2": "atan2",

            "arcsinh": "asinh",
            "arccosh": "acosh",
            "arctanh": "atanh",
            }

    _c_to_numpy_arc_functions = {c_name: numpy_name
            for numpy_name, c_name in _numpy_to_c_arc_functions.items()}

    def __getattr__(self, name):
        def loopy_implemented_elwise_func(*args):
            actx = self._array_context
            # FIXME: Maybe involve loopy type inference?
            result = actx.empty(args[0].shape, args[0].dtype)
            prg = actx._get_scalar_func_loopy_program(
                    c_name, nargs=len(args), naxes=len(args[0].shape))
            actx.call_loopy(prg, out=result,
                    **{"inp%d" % i: arg for i, arg in enumerate(args)})
            return result

        if name in self._c_to_numpy_arc_functions:
            from warnings import warn
            warn(f"'{name}' in ArrayContext.np is deprecated. "
                    "Use '{c_to_numpy_arc_functions[name]}' as in numpy. "
                    "The old name will stop working in 2021.",
                    DeprecationWarning, stacklevel=3)

        # normalize to C names anyway
        c_name = self._numpy_to_c_arc_functions.get(name, name)

        # limit which functions we try to hand off to loopy
        if name in self._numpy_math_functions:
            from meshmode.dof_array import obj_or_dof_array_vectorized_n_args
            return obj_or_dof_array_vectorized_n_args(loopy_implemented_elwise_func)
        else:
            raise AttributeError(name)

    def _new_like(self, ary, alloc_like):
        # FIXME: DOFArray should not be here (circular dependencies)
        from meshmode.dof_array import DOFArray
        from numbers import Number

        if isinstance(ary, DOFArray):
            return DOFArray(self._array_context, tuple([
                alloc_like(subary) for subary in ary
                ]))
        elif isinstance(ary, np.ndarray) and ary.dtype.char == "O":
            raise NotImplementedError("operation not implemented for object arrays")
        elif isinstance(ary, Number):
            # NOTE: `np.zeros_like(x)` returns `array(x, shape=())`, which
            # is best implemented by concrete array contexts, if at all
            raise NotImplementedError("operation not implemented for scalars")
        else:
            return alloc_like(ary)

    def empty_like(self, ary):
        return self._new_like(ary, self._array_context.empty_like)

    def zeros_like(self, ary):
        return self._new_like(ary, self._array_context.zeros_like)

    def conjugate(self, x):
        # NOTE: conjugate distributes over object arrays, but it looks for a
        # `conjugate` ufunc, while some implementations only have the shorter
        # `conj` (e.g. cl.array.Array), so this should work for everybody.
        from meshmode.dof_array import obj_or_dof_array_vectorize
        return obj_or_dof_array_vectorize(lambda obj: obj.conj(), x)

    conj = conjugate


class _BaseFakeNumpyLinalgNamespace:
    def __init__(self, array_context):
        self._array_context = array_context


# {{{ program metadata

class CommonSubexpressionTag(Tag):
    """A tag that is applicable to arrays indicating that this same array
    may be evaluated multiple times, and that the implementation should
    eliminate those redundant evaluations if possible.

    .. versionadded:: 2021.2
    """


class FirstAxisIsElementsTag(Tag):
    """A tag that is applicable to array outputs indicating that the
    first index corresponds to element indices. This suggests that
    the implementation should set element indices as the outermost
    loop extent.

    .. versionadded:: 2021.2
    """

# }}}


class ArrayContext(ABC):
    """An interface that allows a
    :class:`~meshmode.discretization.Discretization` to create and interact
    with arrays of degrees of freedom without fully specifying their types.

    .. versionadded:: 2020.2

    .. automethod:: empty
    .. automethod:: zeros
    .. automethod:: empty_like
    .. automethod:: zeros_like
    .. automethod:: from_numpy
    .. automethod:: to_numpy
    .. automethod:: call_loopy
    .. automethod:: einsum
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
    .. automethod:: tag
    .. automethod:: tag_axis
    """

    def __init__(self):
        self.np = self._get_fake_numpy_namespace()

    def _get_fake_numpy_namespace(self):
        return _BaseFakeNumpyNamespace(self)

    @abstractmethod
    def empty(self, shape, dtype):
        pass

    @abstractmethod
    def zeros(self, shape, dtype):
        pass

    def empty_like(self, ary):
        return self.empty(shape=ary.shape, dtype=ary.dtype)

    def zeros_like(self, ary):
        return self.zeros(shape=ary.shape, dtype=ary.dtype)

    @abstractmethod
    def from_numpy(self, array: np.ndarray):
        r"""
        :returns: the :class:`numpy.ndarray` *array* converted to the
            array context's array type. The returned array will be
            :meth:`thaw`\ ed.
        """
        pass

    @abstractmethod
    def to_numpy(self, array):
        r"""
        :returns: *array*, an array recognized by the context, converted
            to a :class:`numpy.ndarray`. *array* must be
            :meth:`thaw`\ ed.
        """
        pass

    def call_loopy(self, program, **kwargs):
        """Execute the :mod:`loopy` program *program* on the arguments
        *kwargs*.

        *program* is a :class:`loopy.LoopKernel` or :class:`loopy.LoopKernel`.
        It is expected to not yet be transformed for execution speed.
        It must have :attr:`loopy.Options.return_dict` set.

        :return: a :class:`dict` of outputs from the program, each an
            array understood by the context.
        """

    @memoize_method
    def _get_scalar_func_loopy_program(self, c_name, nargs, naxes):
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
                        var(c_name)(*[
                            var("inp%d" % i)[subscript] for i in range(nargs)]))
                    ],
                name="actx_special_%s" % c_name)

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def tag(self, tags: Union[Sequence[Tag], Tag], array):
        """If the array type used by the array context is capable of capturing
        metadata, return a version of *array* with the *tags* applied. *array*
        itself is not modified.

        .. versionadded:: 2021.2
        """

    @abstractmethod
    def tag_axis(self, iaxis, tags: Union[Sequence[Tag], Tag], array):
        """If the array type used by the array context is capable of capturing
        metadata, return a version of *array* in which axis number *iaxis* has
        the *tags* applied. *array* itself is not modified.

        .. versionadded:: 2021.2
        """

    @memoize_method
    def _get_einsum_prg(self, spec, arg_names, tagged):
        return lp.make_einsum(
            spec,
            arg_names,
            options=_DEFAULT_LOOPY_OPTIONS,
            tags=tagged,
        )

    # This lives here rather than in .np because the interface does not
    # agree with numpy's all that well. Why can't it, you ask?
    # Well, optimizing generic einsum for OpenCL/GPU execution
    # is actually difficult, even in eager mode, and so without added
    # metadata describing what's happening, transform_loopy_program
    # has a very difficult (hopeless?) job to do.
    #
    # Unfortunately, the existing metadata support (cf. .tag()) cannot
    # help with eager mode execution [1], because, by definition, when the
    # result is passed to .tag(), it is already computed.
    # That's why einsum's interface here needs to be cluttered with
    # metadata, and that's why it can't live under .np.
    # [1] https://github.com/inducer/meshmode/issues/177
    def einsum(self, spec, *args, arg_names=None, tagged=()):
        """Computes the result of Einstein summation following the
        convention in :func:`numpy.einsum`.

        :arg spec: a string denoting the subscripts for
            summation as a comma-separated list of subscript labels.
            This follows the usual :func:`numpy.einsum` convention.
            Note that the explicit indicator `->` for the precise output
            form is required.
        :arg args: a sequence of array-like operands, whose order matches
            the subscript labels provided by *spec*.
        :arg arg_names: an optional iterable of string types denoting
            the names of the *args*. If *None*, default names will be
            generated.
        :arg tagged: an optional sequence of :class:`pytools.tag.Tag`
            objects specifying the tags to be applied to the operation.

        :return: the output of the einsum :mod:`loopy` program
        """
        if arg_names is None:
            arg_names = tuple("arg%d" % i for i in range(len(args)))

        prg = self._get_einsum_prg(spec, arg_names, tagged)
        return self.call_loopy(
            prg, **{arg_names[i]: arg for i, arg in enumerate(args)}
        )["out"]

# }}}


# {{{ PyOpenCLArrayContext

class _PyOpenCLFakeNumpyNamespace(_BaseFakeNumpyNamespace):
    def _get_fake_numpy_linalg_namespace(self):
        return _PyOpenCLFakeNumpyLinalgNamespace(self._array_context)

    def _bop(self, op, x, y):
        from meshmode.dof_array import obj_or_dof_array_vectorize_n_args
        return obj_or_dof_array_vectorize_n_args(op, x, y)

    def equal(self, x, y): return self._bop(operator.eq, x, y)  # noqa: E704
    def not_equal(self, x, y): return self._bop(operator.ne, x, y)  # noqa: E704
    def greater(self, x, y): return self._bop(operator.gt, x, y)  # noqa: E704
    def greater_equal(self, x, y): return self._bop(operator.ge, x, y)  # noqa: E704
    def less(self, x, y): return self._bop(operator.lt, x, y)  # noqa: E704
    def less_equal(self, x, y): return self._bop(operator.le, x, y)  # noqa: E704

    def ones_like(self, ary):
        def _ones_like(subary):
            ones = self._array_context.empty_like(subary)
            ones.fill(1)
            return ones

        return self._new_like(ary, _ones_like)

    def maximum(self, x, y):
        import pyopencl.array as cl_array
        from meshmode.dof_array import obj_or_dof_array_vectorize_n_args
        return obj_or_dof_array_vectorize_n_args(
                partial(cl_array.maximum, queue=self._array_context.queue),
                x, y)

    def minimum(self, x, y):
        import pyopencl.array as cl_array
        from meshmode.dof_array import obj_or_dof_array_vectorize_n_args
        return obj_or_dof_array_vectorize_n_args(
                partial(cl_array.minimum, queue=self._array_context.queue),
                x, y)

    def where(self, criterion, then, else_):
        import pyopencl.array as cl_array
        from meshmode.dof_array import obj_or_dof_array_vectorize_n_args

        def where_inner(inner_crit, inner_then, inner_else):
            if isinstance(inner_crit, bool):
                return inner_then if inner_crit else inner_else
            return cl_array.if_positive(inner_crit != 0, inner_then, inner_else,
                    queue=self._array_context.queue)

        return obj_or_dof_array_vectorize_n_args(where_inner, criterion, then, else_)

    def sum(self, a, dtype=None):
        import pyopencl.array as cl_array
        return cl_array.sum(
                a, dtype=dtype, queue=self._array_context.queue).get()[()]

    def min(self, a):
        import pyopencl.array as cl_array
        return cl_array.min(a, queue=self._array_context.queue).get()[()]

    def max(self, a):
        import pyopencl.array as cl_array
        return cl_array.max(a, queue=self._array_context.queue).get()[()]

    def stack(self, arrays, axis=0):
        import pyopencl.array as cla
        from meshmode.dof_array import obj_or_dof_array_vectorize_n_args
        return obj_or_dof_array_vectorize_n_args(
                lambda *args: cla.stack(arrays=args, axis=axis,  # pylint: disable=no-member  # noqa: E501
                    queue=self._array_context.queue),
                *arrays)


def _flatten_grp_array(grp_ary):
    if grp_ary.size == 0:
        # Work around https://github.com/inducer/pyopencl/pull/402
        return grp_ary._new_with_changes(
                data=None, offset=0, shape=(0,), strides=(grp_ary.dtype.itemsize,))
    if grp_ary.flags.f_contiguous:
        return grp_ary.reshape(-1, order="F")
    elif grp_ary.flags.c_contiguous:
        return grp_ary.reshape(-1, order="C")
    else:
        raise ValueError("cannot flatten group array of DOFArray for norm, "
                f"with strides {grp_ary.strides} of {grp_ary.dtype}")


class _PyOpenCLFakeNumpyLinalgNamespace(_BaseFakeNumpyLinalgNamespace):
    def norm(self, array, ord=None):
        if len(array.shape) != 1:
            raise NotImplementedError("only vector norms are implemented")

        if ord is None:
            ord = 2

        # FIXME: Handling DOFArrays here is not beautiful, but it sure does avoid
        # downstream headaches.
        from meshmode.dof_array import DOFArray
        if isinstance(array, DOFArray):
            import numpy.linalg as la
            return la.norm(np.array([
                self.norm(_flatten_grp_array(grp_ary), ord)
                for grp_ary in array]), ord)

        if array.size == 0:
            return 0

        from numbers import Number
        if ord == np.inf:
            return self._array_context.np.max(abs(array))
        elif isinstance(ord, Number) and ord > 0:
            return self._array_context.np.sum(abs(array)**ord)**(1/ord)
        else:
            raise NotImplementedError(f"unsupported value of 'ord': {ord}")


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

    def from_numpy(self, array: np.ndarray):
        import pyopencl.array as cla
        return cla.to_device(self.queue, array, allocator=self.allocator)

    def to_numpy(self, array):
        return array.get(queue=self.queue)

    def call_loopy(self, t_unit, **kwargs):
        t_unit = self.transform_loopy_program(t_unit)
        default_entrypoint = _loopy_get_default_entrypoint(t_unit)
        prg_name = default_entrypoint.name

        evt, result = t_unit(self.queue, **kwargs, allocator=self.allocator)

        if self._wait_event_queue_length is not False:
            wait_event_queue = self._kernel_name_to_wait_event_queue.setdefault(
                    prg_name, [])

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
    def transform_loopy_program(self, t_unit):
        # accommodate loopy with and without kernel callables

        default_entrypoint = _loopy_get_default_entrypoint(t_unit)
        options = default_entrypoint.options
        if not (options.return_dict and options.no_numpy):
            raise ValueError("Loopy kernel passed to call_loopy must "
                    "have return_dict and no_numpy options set. "
                    "Did you use meshmode.array_context.make_loopy_program "
                    "to create this kernel?")

        all_inames = default_entrypoint.all_inames()
        # FIXME: This could be much smarter.
        inner_iname = None
        if (len(default_entrypoint.instructions) == 1
                and isinstance(default_entrypoint.instructions[0], lp.Assignment)
                and any(isinstance(tag, FirstAxisIsElementsTag)
                    # FIXME: Firedrake branch lacks kernel tags
                    for tag in getattr(default_entrypoint, "tags", ()))):
            stmt, = default_entrypoint.instructions

            out_inames = [v.name for v in stmt.assignee.index_tuple]
            assert out_inames
            outer_iname = out_inames[0]
            if len(out_inames) >= 2:
                inner_iname = out_inames[1]

        elif "iel" in all_inames:
            outer_iname = "iel"

            if "idof" in all_inames:
                inner_iname = "idof"
        elif "i0" in all_inames:
            outer_iname = "i0"

            if "i1" in all_inames:
                inner_iname = "i1"
        else:
            raise RuntimeError(
                "Unable to reason what outer_iname and inner_iname "
                f"needs to be; all_inames is given as: {all_inames}"
            )

        if inner_iname is not None:
            t_unit = lp.split_iname(t_unit, inner_iname, 16, inner_tag="l.0")
        return lp.tag_inames(t_unit, {outer_iname: "g.0"})

    def tag(self, tags: Union[Sequence[Tag], Tag], array):
        # Sorry, not capable.
        return array

    def tag_axis(self, iaxis, tags: Union[Sequence[Tag], Tag], array):
        # Sorry, not capable.
        return array

# }}}


# {{{ pytest integration

def pytest_generate_tests_for_pyopencl_array_context(metafunc):
    """Parametrize tests for pytest to use a :mod:`pyopencl` array context.

    Performs device enumeration analogously to
    :func:`pyopencl.tools.pytest_generate_tests_for_pyopencl`.

    Using the line:

    .. code-block:: python

       from meshmode.array_context import pytest_generate_tests_for_pyopencl \
            as pytest_generate_tests

    in your pytest test scripts allows you to use the arguments ctx_factory,
    device, or platform in your test functions, and they will automatically be
    run for each OpenCL device/platform in the system, as appropriate.

    It also allows you to specify the ``PYOPENCL_TEST`` environment variable
    for device selection.
    """

    import pyopencl as cl
    from pyopencl.tools import _ContextFactory

    class ArrayContextFactory(_ContextFactory):
        def __call__(self):
            ctx = super().__call__()
            return PyOpenCLArrayContext(cl.CommandQueue(ctx))

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


# vim: foldmethod=marker
