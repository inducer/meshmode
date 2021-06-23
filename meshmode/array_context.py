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

from arraycontext import (  # noqa: F401
        ArrayContext,

        CommonSubexpressionTag, FirstAxisIsElementsTag,

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

        pytest_generate_tests_for_pyopencl_array_context
        )

from warnings import warn
warn("meshmode.array_context is deprecated. Import this functionality from "
        "the arraycontext top-level package instead. This shim will remain working "
        "until 2022.", DeprecationWarning, stacklevel=2)


def thaw(actx, ary):
    warn("meshmode.array_context.thaw is deprecated. Use arraycontext.thaw instead. "
            "WARNING: The argument order is reversed between these two functions. "
            "meshmode.array_context.thaw will continue to work until 2022.",
            DeprecationWarning, stacklevel=2)

    # /!\ arg order flipped
    return _thaw(ary, actx)

# vim: foldmethod=marker
