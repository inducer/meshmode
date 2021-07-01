"""
.. autoclass:: PyOpenCLArrayContext
"""

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

import sys
from warnings import warn
from arraycontext import PyOpenCLArrayContext as PyOpenCLArrayContextBase
from arraycontext import PytatoPyOpenCLArrayContext as PytatoPyOpenCLArrayContextBase
from arraycontext.pytest import (
        _PytestPyOpenCLArrayContextFactoryWithClass,
        register_pytest_array_context_factory)


def thaw(actx, ary):
    warn("meshmode.array_context.thaw is deprecated. Use arraycontext.thaw instead. "
            "WARNING: The argument order is reversed between these two functions. "
            "meshmode.array_context.thaw will continue to work until 2022.",
            DeprecationWarning, stacklevel=2)

    from arraycontext import thaw as _thaw
    # /!\ arg order flipped
    return _thaw(ary, actx)


# {{{ kernel transform function

def _transform_loopy_inner(t_unit):
    import loopy as lp
    from meshmode.transform_metadata import FirstAxisIsElementsTag
    from arraycontext.transform_metadata import ElementwiseMapKernelTag

    from pymbolic.primitives import Subscript, Variable

    default_ep = t_unit.default_entrypoint

    # FIXME: Firedrake branch lacks kernel tags
    kernel_tags = getattr(default_ep, "tags", ())

    # {{{ FirstAxisIsElementsTag on kernel (compatibility)

    if any(isinstance(tag, FirstAxisIsElementsTag) for tag in kernel_tags):
        if (len(default_ep.instructions) != 1
                or not isinstance(
                    default_ep.instructions[0], lp.Assignment)):
            raise ValueError("FirstAxisIsElementsTag may only be applied to "
                    "a kernel if the kernel contains a single assignment.")

        stmt, = default_ep.instructions

        if not isinstance(stmt.assignee, Subscript):
            raise ValueError("single assignment in FirstAxisIsElementsTag kernel "
                    "must be a subscript")

        output_name = stmt.assignee.aggregate.name
        new_args = [
                arg.tagged(FirstAxisIsElementsTag())
                if arg.name == output_name else arg
                for arg in default_ep.args]
        default_ep = default_ep.copy(args=new_args)
        t_unit = t_unit.with_kernel(default_ep)

    # }}}

    # {{{ ElementwiseMapKernelTag on kernel

    if any(isinstance(tag, ElementwiseMapKernelTag) for tag in kernel_tags):
        el_inames = []
        dof_inames = []
        for stmt in default_ep.instructions:
            if isinstance(stmt, lp.MultiAssignmentBase):
                for assignee in stmt.assignees:
                    if isinstance(assignee, Variable):
                        # some scalar assignee kernel => no concurrency in the
                        # workload => skip
                        continue
                    if not isinstance(assignee, Subscript):
                        raise ValueError("assignees in "
                                "ElementwiseMapKernelTag-tagged kernels must be "
                                "subscripts")

                    for i, subscript in enumerate(assignee.index_tuple[:2]):
                        if (not isinstance(subscript, Variable)
                                or subscript.name not in default_ep.all_inames()):
                            raise ValueError("subscripts in "
                                    "ElementwiseMapKernelTag-tagged kernels must be "
                                    "inames")

                        if i == 0:
                            el_inames.append(subscript.name)
                        elif i == 1:
                            dof_inames.append(subscript.name)

        return _transform_with_element_and_dof_inames(t_unit, el_inames, dof_inames)

    # }}}

    # {{{ FirstAxisIsElementsTag on output variable

    first_axis_el_args = [arg.name for arg in default_ep.args
            if any(isinstance(tag, FirstAxisIsElementsTag) for tag in arg.tags)]

    if first_axis_el_args:
        el_inames = []
        dof_inames = []

        for stmt in default_ep.instructions:
            if isinstance(stmt, lp.MultiAssignmentBase):
                for assignee in stmt.assignees:
                    if not isinstance(assignee, Subscript):
                        raise ValueError("assignees in "
                                "FirstAxisIsElementsTag-tagged kernels must be "
                                "subscripts")

                    if assignee.aggregate.name not in first_axis_el_args:
                        continue

                    subscripts = assignee.index_tuple[:2]

                    for i, subscript in enumerate(subscripts):
                        if (not isinstance(subscript, Variable)
                                or subscript.name not in default_ep.all_inames()):
                            raise ValueError("subscripts in "
                                    "FirstAxisIsElementsTag-tagged kernels must be "
                                    "inames")

                        if i == 0:
                            el_inames.append(subscript.name)
                        elif i == 1:
                            dof_inames.append(subscript.name)
        return _transform_with_element_and_dof_inames(t_unit, el_inames, dof_inames)

    # }}}

    # {{{ element/dof iname tag

    from meshmode.transform_metadata import \
            ConcurrentElementInameTag, ConcurrentDOFInameTag
    el_inames = [iname.name
            for iname in default_ep.inames.values()
            if ConcurrentElementInameTag() in iname.tags]
    dof_inames = [iname.name
            for iname in default_ep.inames.values()
            if ConcurrentDOFInameTag() in iname.tags]

    if el_inames:
        return _transform_with_element_and_dof_inames(t_unit, el_inames, dof_inames)

    # }}}

    # *shrug* no idea how to transform this thing.
    return None


def _transform_with_element_and_dof_inames(t_unit, el_inames, dof_inames):
    import loopy as lp

    if set(el_inames) & set(dof_inames):
        raise ValueError("Some inames are marked as both 'element' and 'dof' "
                "inames. These must be disjoint.")

    # Sorting ensures the same order of transformations is used every
    # time; avoids accidentally generating cache misses or kernel
    # hash conflicts.

    for dof_iname in sorted(dof_inames):
        t_unit = lp.split_iname(t_unit, dof_iname, 32, inner_tag="l.0")
    for el_iname in sorted(el_inames):
        t_unit = lp.tag_inames(t_unit, {el_iname: "g.0"})
    return t_unit

# }}}


# {{{ pyopencl array context subclass

class PyOpenCLArrayContext(PyOpenCLArrayContextBase):
    """Extends :class:`arraycontext.PyOpenCLArrayContext` with knowledge about
    program transformation for finite element programs.

    See :mod:`meshmode.transform_metadata` for relevant metadata.
    """

    def transform_loopy_program(self, t_unit):
        default_ep = t_unit.default_entrypoint
        options = default_ep.options
        if not (options.return_dict and options.no_numpy):
            raise ValueError("Loopy kernel passed to call_loopy must "
                    "have return_dict and no_numpy options set. "
                    "Did you use arraycontext.make_loopy_program "
                    "to create this kernel?")

        transformed_t_unit = _transform_loopy_inner(t_unit)

        if transformed_t_unit is not None:
            return transformed_t_unit

        warn("meshmode.array_context.PyOpenCLArrayContext."
                "transform_loopy_program fell back on "
                "arraycontext.PyOpenCLArrayContext to find a transform for "
                f"'{default_ep.name}'. "
                "Please update your program to use metadata from "
                "meshmode.transform_metadata. "
                "This code path will stop working in 2022.",
                DeprecationWarning, stacklevel=3)

        return super().transform_loopy_program(t_unit)

# }}}


# {{{ pytest actx factory

class PytestPyOpenCLArrayContextFactory(
        _PytestPyOpenCLArrayContextFactoryWithClass):
    actx_class = PyOpenCLArrayContext


# deprecated
class PytestPyOpenCLArrayContextFactoryWithHostScalars(
        _PytestPyOpenCLArrayContextFactoryWithClass):
    actx_class = PyOpenCLArrayContext
    force_device_scalars = False


register_pytest_array_context_factory("meshmode.pyopencl",
        PytestPyOpenCLArrayContextFactory)
register_pytest_array_context_factory("meshmode.pyopencl-deprecated",
        PytestPyOpenCLArrayContextFactoryWithHostScalars)

# }}}


# {{{ handle move deprecation

_actx_names = (
        "ArrayContext",

        "CommonSubexpressionTag",
        "FirstAxisIsElementsTag",

        "ArrayContainer",
        "is_array_container", "is_array_container_type",
        "serialize_container", "deserialize_container",
        "get_container_context", "get_container_context_recursively",
        "with_container_arithmetic",
        "dataclass_array_container",

        "map_array_container", "multimap_array_container",
        "rec_map_array_container", "rec_multimap_array_container",
        "mapped_over_array_containers",
        "multimapped_over_array_containers",
        "freeze",

        "make_loopy_program",

        "pytest_generate_tests_for_pyopencl_array_context"
        )


if sys.version_info >= (3, 7):
    def __getattr__(name):
        if name not in _actx_names:
            raise AttributeError(name)

        import arraycontext
        result = getattr(arraycontext, name)

        warn(f"meshmode.array_context.{name} is deprecated. "
                f"Use arraycontext.{name} instead. "
                f"meshmode.array_context.{name} will continue to work until 2022.",
                DeprecationWarning, stacklevel=2)

        return result
else:
    def _import_names():
        import arraycontext
        for name in _actx_names:
            globals()[name] = getattr(arraycontext, name)

    _import_names()

# }}}


class PytatoPyOpenCLArrayContext(PytatoPyOpenCLArrayContextBase):
    def transform_loopy_program(self, t_unit):
        return t_unit
        import loopy as lp
        from loopy.translation_unit import for_each_kernel

        nwg = 48
        nwi = (16, 2)

        @for_each_kernel
        def gridify(knl):

            # {{{ Pattern matching inames

            for insn in knl.instructions:
                if isinstance(insn, lp.CallInstruction):
                    # must be a callable kernel, don't touch.
                    pass
                elif isinstance(insn, lp.Assignment):
                    bigger_loop = None
                    smaller_loop = None
                    for iname in insn.within_inames:
                        if iname.startswith("iel"):
                            assert bigger_loop is None
                            bigger_loop = iname
                        if iname.startswith("idof"):
                            assert smaller_loop is None
                            smaller_loop = iname

                    if bigger_loop or smaller_loop:
                        if (bigger_loop is None
                                and smaller_loop is None):
                            continue
                    # else:
                    #     sorted_inames = sorted(tuple(insn.within_inames),
                    #             key=knl.get_constant_iname_length)
                    #     # print(sorted_inames)
                    #     if len(sorted_inames) == 2:
                    #         smaller_loop = sorted_inames[0]
                    #         bigger_loop = sorted_inames[1]

                        # knl = lp.chunk_iname(knl, bigger_loop, nwg,
                        #         outer_tag="g.0")
                        # knl = lp.split_iname(knl, f"{bigger_loop}_inner",
                        #         nwi[0], inner_tag="l.1")
                        knl = lp.split_iname(knl, smaller_loop,
                                nwi[1], inner_tag="l.0")
                elif isinstance(insn, lp.BarrierInstruction):
                    pass
                else:
                    raise NotImplementedError

            # }}}

            return knl

        t_unit = lp.set_options(t_unit, "insert_additional_gbarriers")

        return gridify(t_unit)

# vim: foldmethod=marker
