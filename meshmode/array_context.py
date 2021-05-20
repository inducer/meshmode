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
import enum
from dataclasses import fields
from functools import partial, singledispatch, update_wrapper
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Sequence, Tuple, Union, Optional

import numpy as np

import loopy as lp
from loopy.version import MOST_RECENT_LANGUAGE_VERSION

from pytools import memoize_method
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

.. autoclass:: with_container_arithmetic
.. autofunction:: dataclass_array_container
.. autofunction:: map_array_container
.. autofunction:: multimap_array_container
.. autofunction:: rec_map_array_container
.. autofunction:: rec_multimap_array_container
.. autofunction:: freeze
.. autofunction:: thaw_impl
.. autofunction:: thaw

.. autoclass:: ArrayContext
.. autoclass:: PyOpenCLArrayContext

.. autofunction:: pytest_generate_tests_for_pyopencl_array_context
"""

# {{{ loopy

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

# }}}


# {{{ ArrayContainer

class ArrayContainer:
    r"""A generic container for the array type supported by the
    :class:`ArrayContext`.

    The functionality required for the container to operated is supplied via
    :func:`functools.singledispatch`. Implementations of the following functions need
    to be registered for a type serving as an :class:`ArrayContainer`:

    * :func:`serialize_container` for serialization, which gives the components
      of the array.
    * :func:`deserialize_container` for deserialization, which constructs a
      container from a set of components.
    * :func:`get_container_context` retrieves the :class:`ArrayContext` from
      a container, if it has one.

    This allows enumeration of the component arrays in a container and the
    construction of modified containers from an iterable of those component arrays.
    :func:`is_array_container` will return *True* for types that have
    a container serialization function registered.

    Packages may register their own types as array containers. They must not
    register other types (e.g. :class:`list`) as array containers.
    The type :class:`numpy.ndarray` is considered an array container, but
    only arrays with dtype *object* may be used as such. (This is so
    because object arrays cannot be distinguished from non-object arrays
    via their type.)

    The container and its serialization interface has goals and uses
    approaches similar to JAX's
    `PyTrees <https://jax.readthedocs.io/en/latest/pytrees.html>`__,
    however its implementation differs a bit.

    .. note::

        This class is used in type annotation. Inheriting from it confers no
        special meaning or behavior.
    """


@singledispatch
def serialize_container(ary: ArrayContainer) -> Iterable[Tuple[Any, Any]]:
    r"""Serialize the array container into an iterable over its components.

    The order of the components and their identifiers are entirely under
    the control of the container class.

    If *ary* is mutable, the serialization function is not required to ensure
    that the serialization result reflects the array state at the time of the
    call to :func:`serialize_container`.

    :returns: an :class:`Iterable` of 2-tuples where the first
        entry is an identifier for the component and the second entry
        is an array-like component of the :class:`ArrayContainer`.
        Components can themselves be :class:`ArrayContainer`\ s, allowing
        for arbitrarily nested structures. The identifiers need to be hashable
        but are otherwise treated as opaque.
    """
    raise NotImplementedError(type(ary).__name__)


@singledispatch
def deserialize_container(template, iterable: Iterable[Tuple[Any, Any]]):
    """Deserialize an iterable into an array container.

    :param template: an instance of an existing object that
        can be used to aid in the deserialization. For a similar choice
        see :attr:`~numpy.class.__array_finalize__`.
    :param iterable: an iterable that mirrors the output of
        :meth:`serialize_container`.
    """
    raise NotImplementedError(type(template).__name__)


def is_array_container_type(cls: type) -> bool:
    """
    :returns: *True* if the type *cls* has a registered implementation of
        :func:`serialize_container`, or if it is an :class:`ArrayContainer`.
    """
    return (
            cls is ArrayContainer
            or (serialize_container.dispatch(cls)
                is not serialize_container.__wrapped__))


def is_array_container(ary: Any) -> bool:
    """Return *True* if the instance *ary* has a registered implementation of
    :func:`serialize_container`.
    """
    return (serialize_container.dispatch(ary.__class__)
            is not serialize_container.__wrapped__)


@singledispatch
def get_container_context(ary: ArrayContainer) -> Optional["ArrayContext"]:
    """Retrieves the :class:`ArrayContext` from the container, if any.

    This function is not recursive, so it will only search at the root level
    of the container. For the recursive version, see
    :func:`get_container_context_recursively`.
    """
    return getattr(ary, "array_context", None)

# }}}


# {{{ object arrays as array containers

@serialize_container.register(np.ndarray)
def _serialize_ndarray_container(ary: np.ndarray) -> Iterable[Tuple[Any, Any]]:
    if ary.dtype.char != "O":
        raise ValueError(
                f"only object arrays are supported, given dtype '{ary.dtype}'")

    return np.ndenumerate(ary)


@deserialize_container.register(np.ndarray)
def _deserialize_ndarray_container(
        template: Any, iterable: Iterable[Tuple[Any, Any]]):
    # disallow subclasses
    assert type(template) is np.ndarray
    assert template.dtype.char == "O"

    result = type(template)(template.shape, dtype=object)
    for i, subary in iterable:
        result[i] = subary

    return result

# }}}


# {{{ get_container_context_recursively

def get_container_context_recursively(ary) -> Optional["ArrayContext"]:
    """Walks the :class:`ArrayContainer` hierarchy to find an
    :class:`ArrayContext` associated with it.

    If different components that have different array contexts are found at
    any level, an assertion error is raised.
    """
    actx = None
    if not is_array_container(ary):
        return actx

    # try getting the array context directly
    actx = get_container_context(ary)
    if actx is not None:
        return actx

    for _, subary in serialize_container(ary):
        context = get_container_context_recursively(subary)
        if context is None:
            continue

        if not __debug__:
            return context
        elif actx is None:
            actx = context
        else:
            assert actx is context

    return actx

# }}}


# {{{ with_container_arithmetic

class _OpClass(enum.Enum):
    ARITHMETIC = enum.auto
    BITWISE = enum.auto
    SHIFT = enum.auto
    EQ_COMPARISON = enum.auto
    REL_COMPARISON = enum.auto


_UNARY_OP_AND_DUNDER = [
        ("pos", "+{}", _OpClass.ARITHMETIC),
        ("neg", "-{}", _OpClass.ARITHMETIC),
        ("abs", "abs({})", _OpClass.ARITHMETIC),
        ("inv", "~{}", _OpClass.BITWISE),
        ]
_BINARY_OP_AND_DUNDER = [
        ("add", "{} + {}", True, _OpClass.ARITHMETIC),
        ("sub", "{} - {}", True, _OpClass.ARITHMETIC),
        ("mul", "{} * {}", True, _OpClass.ARITHMETIC),
        ("truediv", "{} / {}", True, _OpClass.ARITHMETIC),
        ("floordiv", "{} // {}", True, _OpClass.ARITHMETIC),
        ("pow", "{} ** {}", True, _OpClass.ARITHMETIC),
        ("mod", "{} % {}", True, _OpClass.ARITHMETIC),
        ("divmod", "divmod({}, {})", True, _OpClass.ARITHMETIC),

        ("and", "{} & {}", True, _OpClass.BITWISE),
        ("or", "{} | {}", True, _OpClass.BITWISE),
        ("xor", "{} ^ {}", True, _OpClass.BITWISE),

        ("lshift", "{} << {}", False, _OpClass.SHIFT),
        ("rshift", "{} >> {}", False, _OpClass.SHIFT),

        ("eq", "{} == {}", False, _OpClass.EQ_COMPARISON),
        ("ne", "{} != {}", False, _OpClass.EQ_COMPARISON),

        ("lt", "{} < {}", False, _OpClass.REL_COMPARISON),
        ("gt", "{} > {}", False, _OpClass.REL_COMPARISON),
        ("le", "{} <= {}", False, _OpClass.REL_COMPARISON),
        ("ge", "{} >= {}", False, _OpClass.REL_COMPARISON),
        ]


def _format_unary_op_str(op_str, arg1):
    if isinstance(arg1, tuple):
        arg1_entry, arg1_container = arg1
        return (f"{op_str.format(arg1_entry)} "
                f"for {arg1_entry} in {arg1_container}")
    else:
        return op_str.format(arg1)


def _format_binary_op_str(op_str, arg1, arg2):
    if isinstance(arg1, tuple) and isinstance(arg2, tuple):
        import sys
        if sys.version_info >= (3, 10):
            strict_arg = ", strict=__debug__"
        else:
            strict_arg = ""

        arg1_entry, arg1_container = arg1
        arg2_entry, arg2_container = arg2
        return (f"{op_str.format(arg1_entry, arg2_entry)} "
                f"for {arg1_entry}, {arg2_entry} "
                f"in zip({arg1_container}, {arg2_container}{strict_arg})")

    elif isinstance(arg1, tuple):
        arg1_entry, arg1_container = arg1
        return (f"{op_str.format(arg1_entry, arg2)} "
                f"for {arg1_entry} in {arg1_container}")

    elif isinstance(arg2, tuple):
        arg2_entry, arg2_container = arg2
        return (f"{op_str.format(arg1, arg2_entry)} "
                f"for {arg2_entry} in {arg2_container}")
    else:
        return op_str.format(arg1, arg2)


def with_container_arithmetic(
        bcast_number=True, bcast_obj_array=None, bcast_numpy_array=False,
        arithmetic=True, bitwise=False, shift=False,
        eq_comparison=None, rel_comparison=None):
    """A class decorator that implements built-in operators for array containers
    by propagating the operations to the elements of the container.

    :arg bcast_number: If *True*, numbers broadcast over the container
        (with the container as the 'outer' structure).
    :arg bcast_obj_array: If *True*, :mod:`numpy` object arrays broadcast over
        the container.  (with the container as the 'inner' structure)
    :arg bcast_numpy_array: If *True*, any :class:`numpy.ndarray` will broadcast
        over the container.  (with the container as the 'inner' structure)
    :arg arithmetic: Implement the conventional arithmetic operators, including
        ``**``, :func:`divmod`, and ``//``. Also includes ``+`` and ``-`` as well as
        :func:`abs`.
    :arg bitwise: If *True*, implement bitwise and, or, not, and inversion.
    :arg shift: If *True*, implement bit shifts.
    :arg eq_comparison: If *True*, implement ``==`` and ``!=``.
    :arg rel_comparison: If *True*, implement ``<``, ``<=``, ``>``, ``>=``.
        In that case, if *eq_comparison* is unspecified, it is also set to
        *True*.

    Each operator class also includes the "reverse" operators if applicable.

    .. note::

        To generate the code implementing the operators, this function relies on
        class methods ``_deserialize_init_arrays_code`` and
        ``_serialize_init_arrays_code``. This interface should be considered
        undocumented and subject to change, however if you are curious, you may look
        at its implementation in :class:`meshmode.dof_array.DOFArray`. For a simple
        structure type, the implementation might look like this::

            @classmethod
            def _serialize_init_arrays_code(cls, instance_name):
                return {"u": f"{instance_name}.u", "v": f"{instance_name}.v"}

            @classmethod
            def _deserialize_init_arrays_code(cls, tmpl_instance_name, args):
                return f"u={args['u']}, v={args['v']}"

    :func:`dataclass_array_container` automatically generates an appropriate
    implementation of these methods, so :func:`with_container_arithmetic`
    should nest "outside" :func:dataclass_array_container`.
    """

    # {{{ handle inputs

    if bcast_obj_array is None:
        raise TypeError("bcast_obj_array must be specified")

    if rel_comparison is None:
        raise TypeError("rel_comparison must be specified")

    if rel_comparison and eq_comparison is None:
        eq_comparison = True

    if eq_comparison is None:
        raise TypeError("eq_comparison must be specified")

    if not bcast_obj_array and bcast_numpy_array:
        raise TypeError("bcast_obj_array must be set if bcast_numpy_array is")

    if bcast_numpy_array:
        def numpy_pred(name):
            return f"isinstance({name}, np.ndarray)"
    elif bcast_obj_array:
        def numpy_pred(name):
            return f"isinstance({name}, np.ndarray) and {name}.dtype.char == 'O'"
    else:
        def numpy_pred(name):
            return "False"  # optimized away

    desired_op_classes = set()
    if arithmetic:
        desired_op_classes.add(_OpClass.ARITHMETIC)
    if bitwise:
        desired_op_classes.add(_OpClass.BITWISE)
    if shift:
        desired_op_classes.add(_OpClass.SHIFT)
    if eq_comparison:
        desired_op_classes.add(_OpClass.EQ_COMPARISON)
    if rel_comparison:
        desired_op_classes.add(_OpClass.REL_COMPARISON)

    # }}}

    def wrap(cls):
        if (not hasattr(cls, "_serialize_init_arrays_code")
                or not hasattr(cls, "_deserialize_init_arrays_code")):
            raise TypeError(f"class '{cls.__name__}' must provide serialization "
                    "code to generate arithmetic operations by implementing "
                    "'_serialize_init_arrays_code' and "
                    "'_deserialize_init_arrays_code'. If this is a dataclass, "
                    "use the 'dataclass_array_container' decorator first.")

        from pytools.codegen import CodeGenerator
        gen = CodeGenerator()
        gen("""
            from numbers import Number
            import numpy as np
            from meshmode.array_context import ArrayContainer
            """)
        gen("")

        def same_key(k1, k2):
            assert k1 == k2
            return k1

        # {{{ unary operators

        for dunder_name, op_str, op_cls in _UNARY_OP_AND_DUNDER:
            if op_cls not in desired_op_classes:
                continue

            fname = f"_{cls.__name__.lower()}_{dunder_name}"
            init_args = cls._deserialize_init_arrays_code("arg1", {
                    key_arg1: _format_unary_op_str(op_str, expr_arg1)
                    for key_arg1, expr_arg1 in
                    cls._serialize_init_arrays_code("arg1").items()
                    })

            gen(f"""
                def {fname}(arg1):
                    return cls({init_args})
                cls.__{dunder_name}__ = {fname}""")
            gen("")

        # }}}

        # {{{ binary operators

        for dunder_name, op_str, reversible, op_cls in _BINARY_OP_AND_DUNDER:
            if op_cls not in desired_op_classes:
                continue

            # {{{ "forward" binary operators

            fname = f"_{cls.__name__.lower()}_{dunder_name}"
            zip_init_args = cls._deserialize_init_arrays_code("arg1", {
                    same_key(key_arg1, key_arg2):
                    _format_binary_op_str(op_str, expr_arg1, expr_arg2)
                    for (key_arg1, expr_arg1), (key_arg2, expr_arg2) in zip(
                        cls._serialize_init_arrays_code("arg1").items(),
                        cls._serialize_init_arrays_code("arg2").items())
                    })
            bcast_init_args = cls._deserialize_init_arrays_code("arg1", {
                    key_arg1: _format_binary_op_str(op_str, expr_arg1, "arg2")
                    for key_arg1, expr_arg1 in
                    cls._serialize_init_arrays_code("arg1").items()
                    })

            gen(f"""
                def {fname}(arg1, arg2):
                    if arg2.__class__ is cls:
                        return cls({zip_init_args})
                    if {bcast_number}:  # optimized away
                        if isinstance(arg2, Number):
                            return cls({bcast_init_args})
                    if {numpy_pred("arg2")}:
                        result = np.empty_like(arg2, dtype=object)
                        for i in np.ndindex(arg2.shape):
                            result[i] = {op_str.format("arg1", "arg2[i]")}
                        return result
                    return NotImplemented
                cls.__{dunder_name}__ = {fname}""")
            gen("")

            # }}}

            # {{{ "reverse" binary operators

            if reversible:
                fname = f"_{cls.__name__.lower()}_r{dunder_name}"
                bcast_init_args = cls._deserialize_init_arrays_code("arg2", {
                        key_arg2: _format_binary_op_str(
                            op_str, "arg1", expr_arg2)
                        for key_arg2, expr_arg2 in
                        cls._serialize_init_arrays_code("arg2").items()
                        })
                gen(f"""
                    def {fname}(arg2, arg1):
                        # assert other.__cls__ is not cls

                        if {bcast_number}:  # optimized away
                            if isinstance(arg1, Number):
                                return cls({bcast_init_args})
                        if {numpy_pred("arg1")}:
                            result = np.empty_like(arg1, dtype=object)
                            for i in np.ndindex(arg1.shape):
                                result[i] = {op_str.format("arg1[i]", "arg2")}
                            return result
                        return NotImplemented

                    cls.__r{dunder_name}__ = {fname}""")
                gen("")

            # }}}

        # }}}

        # This will evaluate the module, which is all we need.
        code = gen.get().rstrip()+"\n"
        result_dict = {"_MODULE_SOURCE_CODE": code, "cls": cls}
        exec(compile(code, "<generated code>", "exec"), result_dict)

        return cls

    # we're being called as @with_container_arithmetic(...), with parens
    return wrap

# }}}


# {{{ dataclass containers

def dataclass_array_container(cls):
    """A class decorator that makes the class to which it is applied a
    :class:`ArrayContainer` by registering appropriate implementations of
    :func:`serialize_container` and :func:`deserialize_container`.
    *cls* must be a :func:`~dataclasses.dataclass`.

    Attributes that are not array containers are allowed. In order to decide
    whether an attribute is an array container, the declared attribute type
    is checked via :func:`is_array_container_type`.
    """
    from dataclasses import is_dataclass
    assert is_dataclass(cls)

    array_fields = [
            f for f in fields(cls) if is_array_container_type(f.type)]
    non_array_fields = [
            f for f in fields(cls) if not is_array_container_type(f.type)]

    if not array_fields:
        raise ValueError(f"'{cls}' must have fields with array container type "
                "in order to use the 'dataclass_array_container' decorator")

    serialize_expr = ", ".join(
            f"({f.name!r}, ary.{f.name})" for f in array_fields)
    template_kwargs = ", ".join(
            f"{f.name}=template.{f.name}" for f in non_array_fields)

    lower_cls_name = cls.__name__.lower()

    serialize_init_code = ", ".join(f"{f.name!r}: f'{{instance_name}}.{f.name}'"
            for f in array_fields)
    deserialize_init_code = ", ".join([
            f"{f.name}={{args[{f.name!r}]}}" for f in array_fields
            ] + [
            f"{f.name}={{template_instance_name}}.{f.name}"
            for f in non_array_fields
            ])

    from pytools.codegen import remove_common_indentation
    serialize_code = remove_common_indentation(f"""
        from typing import Any, Iterable, Tuple
        from meshmode.array_context import serialize_container, deserialize_container

        @serialize_container.register(cls)
        def _serialize_{lower_cls_name}(ary: cls):
            return ({serialize_expr},)

        @deserialize_container.register(cls)
        def _deserialize_{lower_cls_name}(
                template: cls, iterable: Iterable[Tuple[Any, Any]]) -> cls:
            return cls(**dict(iterable), {template_kwargs})

        # support for with_container_arithmetic

        def _serialize_init_arrays_code_{lower_cls_name}(cls, instance_name):
            return {{
                {serialize_init_code}
                }}
        cls._serialize_init_arrays_code = classmethod(
            _serialize_init_arrays_code_{lower_cls_name})

        def _deserialize_init_arrays_code_{lower_cls_name}(
                cls, template_instance_name, args):
            return f"{deserialize_init_code}"

        cls._deserialize_init_arrays_code = classmethod(
            _deserialize_init_arrays_code_{lower_cls_name})
        """)

    exec_dict = {"cls": cls}
    exec(compile(serialize_code, "<generated code>", "exec"), exec_dict)
    exec_dict["_MODULE_SOURCE_CODE"] = serialize_code

    return cls

# }}}


# {{{ ArrayContainer traversal

def _map_array_container_impl(f, ary, *, leaf_cls=None, recursive=False):
    """Helper for :func:`rec_map_array_container`.

    :param leaf_cls: class on which we call *f* directly. This is mostly
        useful in the recursive setting, where it can stop the recursion on
        specific container classes. By default, the recursion is stopped when
        a non-:class:`ArrayContainer` class is encountered.
    """
    def rec(_ary):
        if type(_ary) is leaf_cls:  # type(ary) is never None
            return f(_ary)
        elif is_array_container(_ary):
            return deserialize_container(_ary, [
                    (key, frec(subary)) for key, subary in serialize_container(_ary)
                    ])
        else:
            return f(_ary)

    frec = rec if recursive else f
    return rec(ary)


def _multimap_array_container_impl(f, *args, leaf_cls=None, recursive=False):
    """Helper for :func:`rec_multimap_array_container`.

    :param leaf_cls: class on which we call *f* directly. This is mostly
        useful in the recursive setting, where it can stop the recursion on
        specific container classes. By default, the recursion is stopped when
        a non-:class:`ArrayContainer` class is encountered.
    """
    def is_same_type(arys, template_ary):
        return all(type(arys[i]) is type(template_ary) for i in container_indices)

    def rec(*_args):
        template_ary = _args[container_indices[0]]
        assert is_same_type(_args, template_ary), \
                "expected type '{type(template_ary).__name__}'"

        if (type(template_ary) is leaf_cls
                or not is_array_container(template_ary)):
            return f(*_args)

        result = []
        new_args = list(_args)

        for subarys in zip(*[
                serialize_container(_args[i]) for i in container_indices
                ]):
            key = None
            for i, subary in zip(container_indices, subarys):
                if key is None:
                    key = subary[0]
                else:
                    assert key == subary[0]

                new_args[i] = subary[1]

            result.append((key, frec(*new_args)))

        return deserialize_container(template_ary, result)

    container_indices = [
            i for i, arg in enumerate(args)
            if is_array_container(arg) and type(arg) is not leaf_cls]

    if not container_indices:
        return f(*args)

    if len(container_indices) == 1:
        # NOTE: if we just have one ArrayContainer in args, passing it through
        # _map_array_container_impl should be faster
        def wrapper(ary):
            new_args = list(args)
            new_args[container_indices[0]] = ary
            return f(*new_args)

        update_wrapper(wrapper, f)
        return _map_array_container_impl(
                wrapper, args[container_indices[0]],
                leaf_cls=leaf_cls, recursive=recursive)

    frec = rec if recursive else f
    return rec(*args)


def map_array_container(f: Callable[[Any], Any], ary):
    r"""Applies *f* to all components of an :class:`ArrayContainer`.

    Works similarly to :func:`~pytools.obj_array.obj_array_vectorize`, but
    on arbitrary containers.

    For a recursive version, see :func:`rec_map_array_container`.

    :param ary: a (potentially nested) structure of :class:`ArrayContainer`\ s,
        or an instance of a base array type.
    """
    return _map_array_container_impl(f, ary, recursive=False)


def multimap_array_container(f: Callable[[Any], Any], *args):
    r"""Applies *f* to the components of multiple :class:`ArrayContainer`\ s.

    Works similarly to :func:`~pytools.obj_array.obj_array_vectorize_n_args`,
    but on arbitrary containers. The containers must all have the same type,
    which will also be the return type.

    For a recursive version, see :func:`rec_multimap_array_container`.

    :param args: all :class:`ArrayContainer` arguments must be of the same
        type and with the same structure (same number of components, etc.).
    """
    return _multimap_array_container_impl(f, *args, recursive=False)


def rec_map_array_container(f: Callable[[Any], Any], ary):
    r"""Applies *f* recursively to an :class:`ArrayContainer`.

    For a non-recursive version see :func:`map_array_container`.

    :param ary: a (potentially nested) structure of :class:`ArrayContainer`\ s,
        or an instance of a base array type.
    """
    return _map_array_container_impl(f, ary, recursive=True)


def mapped_over_array_containers(f: Callable[[Any], Any]):
    """Decorator around :func:`rec_map_array_container`."""
    wrapper = partial(rec_map_array_container, f)
    update_wrapper(wrapper, f)
    return wrapper


def rec_multimap_array_container(f: Callable[[Any], Any], *args):
    r"""Applies *f* recursively to multiple :class:`ArrayContainer`\ s.

    For a non-recursive version see :func:`multimap_array_container`.

    :param args: all :class:`ArrayContainer` arguments must be of the same
        type and with the same structure (same number of components, etc.).
    """
    return _multimap_array_container_impl(f, *args, recursive=True)


def multimapped_over_array_containers(f: Callable[[Any], Any]):
    """Decorator around :func:`rec_multimap_array_container`."""
    # can't use functools.partial, because its result is insufficiently
    # function-y to be used as a method definition.
    def wrapper(*args):
        return rec_multimap_array_container(f, *args)

    update_wrapper(wrapper, f)
    return wrapper


@singledispatch
def freeze(ary, actx=None):
    r"""Freezes recursively by going through all components of the
    :class:`ArrayContainer` *ary*.

    :param ary: a :meth:`~ArrayContext.thaw`\ ed :class:`ArrayContainer`.

    Array container types may use :func:`functools.singledispatch` ``.register`` to
    register additional implementations.

    See :meth:`ArrayContext.thaw`.
    """
    if is_array_container(ary):
        return _map_array_container_impl(
                partial(freeze, actx=actx), ary,
                recursive=False)
    else:
        if actx is None:
            raise TypeError(
                    f"cannot freeze arrays of type {type(ary).__name__} "
                    "when actx is not supplied. Try calling actx.freeze "
                    "directly or supplying an array context")
        else:
            return actx.freeze(ary)


@singledispatch
def thaw_impl(ary, actx):
    """Serves as the registration point (using :func:`functools.singledispatch`
    ``.register`` to register additional implementations for :func:`thaw`.

    .. note::

        This is separate from :func:`thaw` because of argument order. Use of
        :func:`functools.singledispatch` requires the 'dispatching' argument
        to come first.
    """
    if is_array_container(ary):
        return deserialize_container(ary, [
            (key, thaw_impl(subary, actx))
            for key, subary in serialize_container(ary)
            ])
    else:
        return actx.thaw(ary)


def thaw(actx, ary):
    r"""Thaws recursively by going through all components of the
    :class:`ArrayContainer` *ary*.

    :param ary: a :meth:`~ArrayContext.freeze`\ ed :class:`ArrayContainer`.

    Array container types may use :func:`functools.singledispatch` ``.register``
    (with :func:`thaw_impl`) to register additional implementations.

    See :meth:`ArrayContext.thaw`.
    """
    return thaw_impl(ary, actx)

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
        elif is_array_container(ary):
            return rec_map_array_container(alloc_like, ary)
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
        return rec_map_array_container(lambda obj: obj.conj(), x)

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

    # {{{ comparisons

    # FIXME: This should be documentation, not a comment.
    # These are here mainly because some arrays may choose to interpret
    # equality comparison as a binary predicate of structural identity,
    # i.e. more like "are you two equal", and not like numpy semantics.
    # These operations provide access to numpy-style comparisons in that
    # case.

    def equal(self, x, y):
        return rec_multimap_array_container(operator.eq, x, y)

    def not_equal(self, x, y):
        return rec_multimap_array_container(operator.ne, x, y)

    def greater(self, x, y):
        return rec_multimap_array_container(operator.gt, x, y)

    def greater_equal(self, x, y):
        return rec_multimap_array_container(operator.ge, x, y)

    def less(self, x, y):
        return rec_multimap_array_container(operator.lt, x, y)

    def less_equal(self, x, y):
        return rec_multimap_array_container(operator.le, x, y)

    # }}}

    def ones_like(self, ary):
        def _ones_like(subary):
            ones = self._array_context.empty_like(subary)
            ones.fill(1)
            return ones

        return self._new_like(ary, _ones_like)

    def maximum(self, x, y):
        import pyopencl.array as cl_array
        return rec_multimap_array_container(
                partial(cl_array.maximum, queue=self._array_context.queue),
                x, y)

    def minimum(self, x, y):
        import pyopencl.array as cl_array
        return rec_multimap_array_container(
                partial(cl_array.minimum, queue=self._array_context.queue),
                x, y)

    def where(self, criterion, then, else_):
        import pyopencl.array as cl_array

        def where_inner(inner_crit, inner_then, inner_else):
            if isinstance(inner_crit, bool):
                return inner_then if inner_crit else inner_else
            return cl_array.if_positive(inner_crit != 0, inner_then, inner_else,
                    queue=self._array_context.queue)

        return rec_multimap_array_container(where_inner, criterion, then, else_)

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
        return rec_multimap_array_container(
                lambda *args: cla.stack(arrays=args, axis=axis,
                    queue=self._array_context.queue),
                *arrays)


def _flatten_array(ary):
    import pyopencl.array as cl
    if not isinstance(ary, cl.Array):
        return ary

    if ary.size == 0:
        # Work around https://github.com/inducer/pyopencl/pull/402
        return ary._new_with_changes(
                data=None, offset=0, shape=(0,), strides=(ary.dtype.itemsize,))
    if ary.flags.f_contiguous:
        return ary.reshape(-1, order="F")
    elif ary.flags.c_contiguous:
        return ary.reshape(-1, order="C")
    else:
        raise ValueError("cannot flatten group array of DOFArray for norm, "
                f"with strides {ary.strides} of {ary.dtype}")


class _PyOpenCLFakeNumpyLinalgNamespace(_BaseFakeNumpyLinalgNamespace):
    def norm(self, ary, ord=None):
        from numbers import Number
        if isinstance(ary, Number):
            return abs(ary)

        if ord is None:
            ord = 2

        if is_array_container(ary):
            import numpy.linalg as la
            return la.norm([
                self.norm(_flatten_array(subary), ord=ord)
                for _, subary in serialize_container(ary)
                ], ord=ord)

        if len(ary.shape) != 1:
            raise NotImplementedError("only vector norms are implemented")

        if ary.size == 0:
            return 0

        if ord == np.inf:
            return self._array_context.np.max(abs(ary))
        elif isinstance(ord, Number) and ord > 0:
            return self._array_context.np.sum(abs(ary)**ord)**(1/ord)
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
