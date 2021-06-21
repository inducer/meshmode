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

from functools import partial
from pytools.tag import Tag, UniqueTag

from arraycontext import (  # noqa: F401
        ArrayContext,
        CommonSubexpressionTag, FirstAxisIsElementsTag,
        ParameterValue, IsDOFArray,
        ArrayContainer,
        is_array_container, is_array_container_type,
        serialize_container, deserialize_container,
        get_container_context, get_container_context_recursively,
        with_container_arithmetic,
        dataclass_array_container,
        map_array_container, multimap_array_container,
        rec_map_array_container, rec_multimap_array_container,
        mapped_over_array_containers,
        multimapped_over_array_containers,
        thaw as _thaw, freeze,

        PyOpenCLArrayContext,

        make_loopy_program,
        # Use the version defined in this file for now. This will need to be moved to arraycontext at some
        # point.
        #pytest_generate_tests_for_pyopencl_array_context
        )

# {{{ Tags

#class IsDOFArray(Tag):
#    """A tag to mark arrays of DOFs in :mod:`loopy` kernels. Applications
#    could use this to decide how to change the memory layout of
#    these arrays.
#    """
#    pass

class IsOpArray(Tag):
    pass

#class ParameterValue(UniqueTag):
#
#    def __init__(self, value):
#        self.value = value

    # }}}



# {{{ pytest integration


def _pytest_generate_tests_for_pyopencl_array_context(array_context_type, metafunc):
    import pyopencl as cl
    from pyopencl.tools import _ContextFactory

    class ArrayContextFactory(_ContextFactory):
        def __call__(self):
            ctx = super().__call__()
            return array_context_type(cl.CommandQueue(ctx))

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

from warnings import warn
warn("meshmode.array_context is deprecated. Import this functionality from "
        "the arraycontext top-level package instead. This shim will remain working "
        "until 2022.", DeprecationWarning, stacklevel=2)

def generate_pytest_generate_tests(array_context_type):
    """Generate a function to parametrize tests for pytest to use
    a :mod:`pyopencl` array context of the specified subtype.

    The returned function performs device enumeration analogously to
    :func:`pyopencl.tools.pytest_generate_tests_for_pyopencl`.

    Using the line:

    .. code-block:: python

       from meshmode.array_context import generate_pytest_generate_tests
       pytest_generate_tests =
            generate_pytest_generate_tests(<PyOpenCLArrayContext>)


    in your pytest test scripts allows you to use the arguments ctx_factory,
    device, or platform in your test functions, and they will automatically be
    run for each OpenCL device/platform in the system, as appropriate.

    It also allows you to specify the ``PYOPENCL_TEST`` environment variable
    for device selection.
    """

    from functools import partial

    return lambda metafunc: partial(
            _pytest_generate_tests_for_pyopencl_array_context,
            array_context_type)(metafunc)


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

    generate_pytest_generate_tests(PyOpenCLArrayContext)(metafunc)


# }}}

def thaw(actx, ary):
    warn("meshmode.array_context.thaw is deprecated. Use arraycontext.thaw instead. "
            "WARNING: The argument order is reversed between these two functions. "
            "meshmode.array_context.thaw will continue to work until 2022.",
            DeprecationWarning, stacklevel=2)

    # /!\ arg order flipped
    return _thaw(ary, actx)

# vim: foldmethod=marker
