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

import operator
from dataclasses import fields
from functools import partial, singledispatch, update_wrapper
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

import loopy as lp
from loopy.version import MOST_RECENT_LANGUAGE_VERSION

from pytools import memoize_method, single_valued
from pytools.tag import Tag


__doc__ = """
.. autofunction:: make_loopy_program
.. autoclass:: CommonSubexpressionTag
.. autoclass:: FirstAxisIsElementsTag

.. autoclass:: ArrayContainer
.. autofunction:: is_array_container
.. autofunction:: serialize_container
.. autofunction:: deserialize_container
.. autofunction:: get_container_context
.. autofunction:: get_container_context_recursively

.. autoclass:: ArrayContainerWithArithmetic
.. autoclass:: DataclassArrayContainer
.. autoclass:: DataclassArrayContainerWithArithmetic
.. autofunction:: array_container_vectorize
.. autofunction:: array_container_vectorize_n_args
.. autofunction:: map_array_container
.. autofunction:: multimap_array_container
.. autofunction:: freeze
.. autofunction:: thaw

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


# {{{ ArrayContainer

class ArrayContainerMeta(type):
    def __instancecheck__(cls, instance):
        if issubclass(type(instance), cls):
            return True

        if cls is ArrayContainer:
            return is_array_container(instance)

        return False


class ArrayContainer(metaclass=ArrayContainerMeta):
    r"""A generic container for the array type supported by the
    :class:`ArrayContext`.

    The functionality for the container is implement through
    :func:`functools.singledispatch`. The following methods are required

    * :func:`is_array_container` allows registering foreign types as containers.
      For example, object :class:`~numpy.ndarray`\ s are considered containers
      and ``isinstance(ary, ArrayContainer)`` will be *True*.
    * Serialization functionality is implemented in :func:`serialize_container`
      and :func:`deserialize_container`. This allows accessing the components
      of the container.
    * :func:`get_container_context` retrieves the :class:`ArrayContext` from
      a container, if it has one.

    The container is meant to work in a similar way to JAX's
    `PyTrees <https://jax.readthedocs.io/en/latest/pytrees.html>`__.
    """


@singledispatch
def is_array_container(ary: object):
    return False


@singledispatch
def serialize_container(ary: ArrayContainer) -> Iterable[Tuple[Any, Any]]:
    r"""Serialize the array container into an iterable over its components.

    The order of the components and their identifiers are entirely under
    the control of the container class.

    :returns: an :class:`Iterable` of 2-tuples where the first
        entry is an identifier for the component and the second entry
        is an array-like component of the :class:`ArrayContainer`.
        Components can themselves be :class:`ArrayContainer`\ s, allowing
        for arbitrarily nested structures. The identifiers need to be hashable
        but are otherwise treated as opaque.
    """
    raise NotImplementedError(type(ary).__name__)


@singledispatch
def deserialize_container_class(cls: type,
        actx: Optional["ArrayContext"],
        iterable: Iterable[Tuple[Any, Any]]):
    raise NotImplementedError(cls.__name__)


def deserialize_container(cls: type,
        actx: Optional["ArrayContext"],
        iterable: Iterable[Tuple[Any, Any]]):
    """Deserialize an iterable into an array container.

    :param actx: can be *None* for frozen arrays or if the array does not use
        an :class:`ArrayContext`.
    :param iterable: an iterable that mirrors the output of
        :meth:`serialize_container`.
    """
    return deserialize_container_class.dispatch(cls)(cls, actx, iterable)


@singledispatch
def get_container_context(ary: ArrayContainer):
    """Retrieves the :class:`ArrayContext` from the container, if any.

    This function is not recursive, so it will only search at the root level
    of the container. For the recursive version, see
    :func:`get_container_context_recursively`
    """
    return None


# {{{ object arrays

@is_array_container.register(np.ndarray)
def _(ary: np.ndarray):
    return ary.dtype.char == "O"


@serialize_container.register(np.ndarray)
def _(ary: Union[list, tuple, np.ndarray]):
    if ary.dtype.char != "O":
        raise NotImplementedError(
                f"serialization for 'numpy.ndarray' of dtype '{ary.dtype}'")

    return np.ndenumerate(ary)


@deserialize_container_class.register(np.ndarray)
def _(cls: type, actx: "ArrayContext", iterable):
    assert cls is np.ndarray
    iterable = list(iterable)

    result = cls(len(iterable), dtype=object)
    shape = np.full(len(iterable[0][0]), -1)

    for i, (index, subary) in enumerate(iterable):
        shape = np.maximum(shape, index)
        result[i] = subary

    return result.reshape(tuple(shape + 1))


@serialize_container.register(list)
@serialize_container.register(tuple)
def _(ary: Union[list, tuple]):
    # NOTE: this is only here so that `multimap_array_container` can call
    # `get_container_context_recursively(args)` directly
    return enumerate(ary)


def get_container_context_recursively(ary: Any):
    """Walks the :class:`ArrayContainer` hierarchy to find an :class:`ArrayContext`
    associated with it.

    If different components that have different array contexts are found,
    an assertion error is raised.
    """
    actx = None

    # try getting the array context directly
    if isinstance(ary, ArrayContainer):
        actx = get_container_context(ary)

    if actx is not None:
        return actx

    # try getting it recursively
    contexts = [
            get_container_context_recursively(subary)
            for _, subary in serialize_container(ary)
            ]

    contexts = [c for c in contexts if c is not None]
    if contexts:
        # FIXME: is `is` a bit strict here? maybe `type(a) is type(b)`?
        # this would allow, e.g. different components to have different
        # queues in an PyOpenCLArrayContext
        actx = single_valued(contexts, equality_pred=operator.is_)

    return actx

# }}}

# }}}


# {{{ ArrayContainerWithArithmetic

class ArrayContainerWithArithmetic(ArrayContainer):
    """Array container with basic arithmetic, comparisons and logic operators.

    .. note::

        :class:`ArrayContainerWithArithmetic` instances support elementwise
        ``<``, ``>``, ``<=``, ``>=``. (:mod:`numpy` object arrays containing
        arrays do not)

    .. autoattribute:: array_context
    """

    @property
    @abstractmethod
    def array_context(self):
        """An :class:`~meshmode.array_context.ArrayContext`."""

    @classmethod
    def unary_op(cls, op, arg):
        return map_array_container(op, arg)

    @classmethod
    def binary_op(cls, op, arg1, arg2):
        from numbers import Number
        from pytools.obj_array import obj_array_vectorize

        if isinstance(arg1, np.ndarray) or isinstance(arg2, np.ndarray):
            # do a bit of broadcasting
            if isinstance(arg1, np.ndarray):
                return obj_array_vectorize(
                        lambda subary: cls.binary_op(op, subary, arg2),
                        arg1.astype(object, copy=False))
            else:
                return obj_array_vectorize(
                        lambda subary: cls.binary_op(op, arg1, subary),
                        arg2.astype(object, copy=False))
        elif isinstance(arg1, ArrayContainer) and isinstance(arg2, ArrayContainer):
            return multimap_array_container(op, arg1, arg2)
        elif isinstance(arg1, ArrayContainer) and isinstance(arg2, Number):
            return map_array_container(lambda subary: op(subary, arg2), arg1)
        elif isinstance(arg1, Number) and isinstance(arg2, ArrayContainer):
            return map_array_container(lambda subary: op(arg1, subary), arg2)
        else:
            NotImplementedError(
                f"operation '{op.__name__}' for arrays of type "
                f"'{type(arg1).__name__}' and '{type(arg2).__name__}'")

    # bit shifts unimplemented for now

    # {{{ arithmetic

    def __add__(self, arg):
        return self.binary_op(operator.add, self, arg)

    def __sub__(self, arg):
        return self.binary_op(operator.sub, self, arg)

    def __mul__(self, arg):
        return self.binary_op(operator.mul, self, arg)

    def __truediv__(self, arg):
        return self.binary_op(operator.truediv, self, arg)

    def __pow__(self, arg):
        return self.binary_op(operator.pow, self, arg)

    def __mod__(self, arg):
        return self.binary_op(operator.mod, self, arg)

    def __divmod__(self, arg):
        return self.binary_op(divmod, self, arg)

    def __radd__(self, arg):
        return self.binary_op(operator.add, arg, self)

    def __rsub__(self, arg):
        return self.binary_op(operator.sub, arg, self)

    def __rmul__(self, arg):
        return self.binary_op(operator.mul, arg, self)

    def __rtruediv__(self, arg):
        return self.binary_op(operator.truediv, arg, self)

    def __rpow__(self, arg):
        return self.binary_op(operator.pow, arg, self)

    def __rmod__(self, arg):
        return self.binary_op(operator.mod, arg, self)

    def __rdivmod__(self, arg):
        return self.binary_op(divmod, arg, self)

    def __pos__(self):
        return self

    def __neg__(self):
        return self.unary_op(operator.neg, self)

    def __abs__(self):
        return self.unary_op(operator.abs, self)

    # }}}

    # {{{ comparison

    def __eq__(self, arg):
        return self.binary_op(self.array_context.np.equal, self, arg)

    def __ne__(self, arg):
        return self.binary_op(self.array_context.np.not_equal, self, arg)

    def __lt__(self, arg):
        return self.binary_op(self.array_context.np.less, self, arg)

    def __gt__(self, arg):
        return self.binary_op(self.array_context.np.greater, self, arg)

    def __le__(self, arg):
        return self.binary_op(self.array_context.np.less_equal, self, arg)

    def __ge__(self, arg):
        return self.binary_op(self.array_context.np.greater_equal, self, arg)

    # }}}

    # {{{ logical

    def __and__(self, arg):
        return self.binary_op(operator.and_, self, arg)

    def __xor__(self, arg):
        return self.binary_op(operator.xor, self, arg)

    def __or__(self, arg):
        return self.binary_op(operator.or_, self, arg)

    def __rand__(self, arg):
        return self.binary_op(operator.and_, arg, self)

    def __rxor__(self, arg):
        return self.binary_op(operator.xor, arg, self)

    def __ror__(self, arg):
        return self.binary_op(operator.or_, arg, self)

    # }}}


@get_container_context.register(ArrayContainerWithArithmetic)
def _(ary: ArrayContainerWithArithmetic):
    return ary.array_context

# }}}


# {{{ dataclass containers

class DataclassArrayContainer(ArrayContainer):
    """An :class:`ArrayContainer` that implements serialization and
    deserialization of :func:`~dataclasses.dataclass` fields.
    """


class DataclassArrayContainerWithArithmetic(
        ArrayContainerWithArithmetic,
        DataclassArrayContainer):
    """An :class:`DataclassArrayContainer` where each field is assumed to
    support arithmetic, e.g. through :class:`ArrayContainerWithArithmetic`.
    """


@serialize_container.register(DataclassArrayContainer)
def _(ary: DataclassArrayContainer):
    return ((var.name, getattr(ary, var.name)) for var in fields(ary))


@deserialize_container_class.register(DataclassArrayContainer)
def _(cls, actx, iterable):
    return cls(**dict(iterable))

# }}}


# {{{ ArrayContainer traversal

class _ArrayContextNotProvided:
    pass


def _update_container_context(ary, actx):
    """
    :returns: a tuple ``(actx, ary_actx)``, where the first array context
        is meant to be passed down the container hierarchy and the second one
        is meant to be used to deserialize the array container.
    """
    if actx is not _ArrayContextNotProvided:
        return actx, actx

    ary_actx = get_container_context(ary)
    actx = actx if ary_actx is None else ary_actx
    return actx, ary_actx


def _map_array_container_with_context(f, ary,
        actx=_ArrayContextNotProvided,
        scalar_cls=None,
        recursive=True):
    if scalar_cls is not None and type(ary) is scalar_cls:
        return f(ary)
    elif isinstance(ary, ArrayContainer):
        actx, ary_actx = _update_container_context(ary, actx)
        if recursive:
            f = partial(_map_array_container_with_context,
                    f, actx=actx, scalar_cls=scalar_cls)

        return deserialize_container(type(ary), ary_actx, (
                (key, f(subary)) for key, subary in serialize_container(ary)
                ))
    else:
        return f(ary)


def _multimap_array_container_with_context(f, *args,
        actx=_ArrayContextNotProvided,
        scalar_cls=None,
        recursive=True):
    container_indices = [
            i for i, arg in enumerate(args)
            if isinstance(arg, ArrayContainer)
            or (scalar_cls is not None and type(arg) is not scalar_cls)
            ]

    if not container_indices:
        return f(*args)

    template_ary = args[container_indices[0]]
    if not all(isinstance(args[i], type(template_ary)) for i in container_indices):
        raise TypeError(
                "'ArrayContainer' arguments must be of the same type: "
                f"{type(template_ary).__name__}")

    def zip_containers(arys):
        keys = (key for key, _ in serialize_container(arys[0]))
        subarys = [
                [subary for _, subary in serialize_container(ary)] for ary in arys
                ]

        from pytools import is_single_valued
        if not is_single_valued([len(ary) for ary in subarys]):
            raise ValueError(
                "all ArrayContainers must have the same number of components")

        for key, value in zip(keys, zip(*subarys)):
            yield key, value

    actx, ary_actx = _update_container_context(template_ary, actx)
    if recursive:
        f = partial(_multimap_array_container_with_context,
                f, actx=actx, scalar_cls=scalar_cls)

    result = []
    new_args = list(args)

    for key, subarys in zip_containers([args[i] for i in container_indices]):
        for i in container_indices:
            new_args[i] = subarys[i]

        result.append((key, f(*new_args)))

    return deserialize_container(type(template_ary), ary_actx, tuple(result))


def array_container_vectorize(f: Callable[[Any], Any], ary):
    r"""Applies *f* to all components of an :class:`ArrayContainer`.

    Works similarly to :func:`~pytools.obj_array.obj_array_vectorize`, but
    on arbitrary containers.

    For a recursive version, see :func:`map_array_container`.

    :param ary: a (potentially nested) structure of :class:`ArrayContainer`\ s,
        or an instance of a base array type.
    """
    return _map_array_container_with_context(f, ary, recursive=False)


def array_container_vectorize_n_args(f: Callable[[Any], Any], *args):
    r"""Applies *f* to the components of multiple :class:`ArrayContainer`\ s.

    Works similarly to :func:`~pytools.obj_array.obj_array_vectorize_n_args`,
    but on arbitrar containers. The containers must all have the same type,
    so that the function has a well-defined return type as well.

    For a recursive version, see :func:`multimap_array_container`.

    :param args: a :class:`tuple` of :class:`ArrayContainer`\ s of the same
        type and with the same structure (same number of components, etc.).
    """
    return _multimap_array_container_with_context(f, *args, recursive=False)


def map_array_container(f: Callable[[Any], Any], ary):
    r"""Applies *f* recursively to an :class:`ArrayContainer`.

    For a non-recursive version see :func:`array_container_vectorize`.

    :param ary: a (potentially nested) structure of :class:`ArrayContainer`\ s,
        or an instance of a base array type.
    """
    return _map_array_container_with_context(f, ary)


def mapped_over_array_containers(f: Callable[[Any], Any]):
    """Decorator around :func:`map_array_container`."""
    wrapper = partial(map_array_container, f)
    update_wrapper(wrapper, f)
    return wrapper


def multimap_array_container(f: Callable[[Any], Any], *args):
    r"""Applies *f* recursively to multiple :class:`ArrayContainer`\ s.

    For a non-recursive version see :func:`array_container_vectorize_n_args`.

    :param args: a :class:`tuple` of :class:`ArrayContainer`\ s of the same
        type and with the same structure (same number of components, etc.).
    """
    return _multimap_array_container_with_context(f, *args)


def multimapped_over_array_containers(f: Callable[[Any], Any]):
    """Decorator around :func:`multimap_array_container`."""
    def wrapper(*args):
        return multimap_array_container(f, *args)

    update_wrapper(wrapper, f)
    return wrapper


def freeze(ary):
    r"""Freezes recursively by going through all components of
    :class:`ArrayContainer`\ s and object arrays.

    :param ary: a tree-like (nested) structure of :meth:`~ArrayContext.thaw`\ ed
        :class:`ArrayContainer`\ s.
    """
    def _freeze(subary, actx=None):
        if isinstance(subary, ArrayContainer):
            if actx is None:
                actx = get_container_context(subary)

            return _map_array_container_with_context(
                    partial(_freeze, actx=actx),
                    subary, recursive=False)
        else:
            return actx.freeze(subary)

    if not isinstance(ary, ArrayContainer):
        raise TypeError(
                f"cannot freeze arrays of type {type(ary).__name__}; "
                "try calling 'ArrayContext.freeze' directly")

    return _freeze(ary)


def thaw(actx, ary):
    r"""Thaws recursively by going through all components of
    :class:`ArrayContainer`\ s and object arrays.

    :param ary: a tree-like (nested) structure of :meth:`~ArrayContext.freeze`\ ed
        :class:`ArrayContainer`\ s.
    """
    return _map_array_container_with_context(actx.thaw, ary, actx=actx)

# }}}


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
            return multimapped_over_array_containers(loopy_implemented_elwise_func)
        else:
            raise AttributeError(name)

    def _new_like(self, ary, alloc_like):
        from numbers import Number

        if isinstance(ary, np.ndarray) and ary.dtype.char == "O":
            # NOTE: we don't want to match numpy semantics on object arrays,
            # e.g. `np.zeros_like(x)` returns `array([0, 0, ...], dtype=object)`
            # FIXME: what about object arrays nested in an ArrayContainer?
            raise NotImplementedError("operation not implemented for object arrays")
        elif isinstance(ary, ArrayContainer):
            return map_array_container(alloc_like, ary)
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
        return map_array_container(lambda obj: obj.conj(), x)

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
    r"""An interface that allows a
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
         arrays, including :class:`meshmode.array_context.ArrayContainer`\ s.

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

    def equal(self, x, y):
        return multimap_array_container(operator.eq, x, y)

    def not_equal(self, x, y):
        return multimap_array_container(operator.ne, x, y)

    def greater(self, x, y):
        return multimap_array_container(operator.gt, x, y)

    def greater_equal(self, x, y):
        return multimap_array_container(operator.ge, x, y)

    def less(self, x, y):
        return multimap_array_container(operator.lt, x, y)

    def less_equal(self, x, y):
        return multimap_array_container(operator.le, x, y)

    def ones_like(self, ary):
        def _ones_like(subary):
            ones = self._array_context.empty_like(subary)
            ones.fill(1)
            return ones

        return self._new_like(ary, _ones_like)

    def maximum(self, x, y):
        import pyopencl.array as cl_array
        return multimap_array_container(
                partial(cl_array.maximum, queue=self._array_context.queue),
                x, y)

    def minimum(self, x, y):
        import pyopencl.array as cl_array
        return multimap_array_container(
                partial(cl_array.minimum, queue=self._array_context.queue),
                x, y)

    def where(self, criterion, then, else_):
        import pyopencl.array as cl_array

        def where_inner(inner_crit, inner_then, inner_else):
            if isinstance(inner_crit, bool):
                return inner_then if inner_crit else inner_else
            return cl_array.if_positive(inner_crit != 0, inner_then, inner_else,
                    queue=self._array_context.queue)

        return multimap_array_container(where_inner, criterion, then, else_)

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
        return multimap_array_container(
                lambda *args: cla.stack(arrays=args, axis=axis,
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
        # FIXME: Extend this to ArrayContainers somehow?
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
