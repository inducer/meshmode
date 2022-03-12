"""
.. autoclass:: PyOpenCLArrayContext
.. autoclass:: PytatoPyOpenCLArrayContext
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
import logging

from warnings import warn
from arraycontext import PyOpenCLArrayContext as PyOpenCLArrayContextBase
from arraycontext import PytatoPyOpenCLArrayContext as PytatoPyOpenCLArrayContextBase
from arraycontext.pytest import (
        _PytestPyOpenCLArrayContextFactoryWithClass,
        _PytestPytatoPyOpenCLArrayContextFactory,
        register_pytest_array_context_factory)
from loopy.translation_unit import for_each_kernel

logger = logging.getLogger(__name__)


def thaw(actx, ary):
    warn("meshmode.array_context.thaw is deprecated. Use arraycontext.thaw instead. "
            "WARNING: The argument order is reversed between these two functions. "
            "meshmode.array_context.thaw will continue to work until 2022.",
            DeprecationWarning, stacklevel=2)

    return actx.thaw(ary)


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


# {{{ pytato pyopencl array context subclass

class PytatoPyOpenCLArrayContext(PytatoPyOpenCLArrayContextBase):
    def transform_dag(self, dag):
        dag = super().transform_dag(dag)

        # {{{ /!\ Remove tags from NamedArrays
        # See <https://www.github.com/inducer/pytato/issues/195>

        import pytato as pt

        def untag_loopy_call_results(expr):
            if isinstance(expr, pt.NamedArray):
                return expr.copy(tags=frozenset(),
                                 axes=(pt.Axis(frozenset()),)*expr.ndim)
            else:
                return expr

        dag = pt.transform.map_and_copy(dag, untag_loopy_call_results)

        # }}}

        return dag

    def transform_loopy_program(self, t_unit):
        # FIXME: Do not parallelize for now.
        return t_unit

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


class PytestPytatoPyOpenCLArrayContextFactory(
        _PytestPytatoPyOpenCLArrayContextFactory):

    @property
    def actx_class(self):
        return PytatoPyOpenCLArrayContext


register_pytest_array_context_factory("meshmode.pyopencl",
        PytestPyOpenCLArrayContextFactory)
register_pytest_array_context_factory("meshmode.pyopencl-deprecated",
        PytestPyOpenCLArrayContextFactoryWithHostScalars)
register_pytest_array_context_factory("meshmode.pytato_cl",
        PytestPytatoPyOpenCLArrayContextFactory)

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


@for_each_kernel
def _single_grid_work_group_transform(kernel, cl_device):
    import loopy as lp
    from meshmode.transform_metadata import (ConcurrentElementInameTag,
                                             ConcurrentDOFInameTag)

    splayed_inames = set()
    ngroups = cl_device.max_compute_units * 4  # '4' to overfill the device
    l_one_size = 4
    l_zero_size = 16

    for insn in kernel.instructions:
        if insn.within_inames in splayed_inames:
            continue

        if isinstance(insn, lp.CallInstruction):
            # must be a callable kernel, don't touch.
            pass
        elif isinstance(insn, lp.Assignment):
            bigger_loop = None
            smaller_loop = None

            if len(insn.within_inames) == 0:
                continue

            if len(insn.within_inames) == 1:
                iname, = insn.within_inames

                kernel = lp.split_iname(kernel, iname,
                                        ngroups * l_zero_size * l_one_size)
                kernel = lp.split_iname(kernel, f"{iname}_inner",
                                        l_zero_size, inner_tag="l.0")
                kernel = lp.split_iname(kernel, f"{iname}_inner_outer",
                                        l_one_size, inner_tag="l.1",
                                        outer_tag="g.0")

                splayed_inames.add(insn.within_inames)
                continue

            for iname in insn.within_inames:
                if kernel.iname_tags_of_type(iname,
                                             ConcurrentElementInameTag):
                    assert bigger_loop is None
                    bigger_loop = iname
                elif kernel.iname_tags_of_type(iname,
                                               ConcurrentDOFInameTag):
                    assert smaller_loop is None
                    smaller_loop = iname
                else:
                    pass

            if bigger_loop or smaller_loop:
                assert (bigger_loop is not None
                        and smaller_loop is not None)
            else:
                sorted_inames = sorted(tuple(insn.within_inames),
                                       key=kernel.get_constant_iname_length)
                smaller_loop = sorted_inames[0]
                bigger_loop = sorted_inames[-1]

            kernel = lp.split_iname(kernel, f"{bigger_loop}",
                                    l_one_size * ngroups)
            kernel = lp.split_iname(kernel, f"{bigger_loop}_inner",
                                    l_one_size, inner_tag="l.1", outer_tag="g.0")
            kernel = lp.split_iname(kernel, smaller_loop,
                                    l_zero_size, inner_tag="l.0")
            splayed_inames.add(insn.within_inames)
        elif isinstance(insn, lp.BarrierInstruction):
            pass
        else:
            raise NotImplementedError(type(insn))

    return kernel


def _alias_global_temporaries(t_unit):
    """
    Returns a copy of *t_unit* with temporaries of that have disjoint live
    intervals using the same :attr:`loopy.TemporaryVariable.base_storage`.
    """
    from loopy.kernel.data import AddressSpace
    from loopy.kernel import KernelState
    from loopy.schedule import (RunInstruction, EnterLoop, LeaveLoop,
                                CallKernel, ReturnFromKernel, Barrier)
    from loopy.schedule.tools import get_return_from_kernel_mapping
    from pytools import UniqueNameGenerator
    from collections import defaultdict

    kernel = t_unit.default_entrypoint
    assert kernel.state == KernelState.LINEARIZED
    temp_vars = frozenset(tv.name
                          for tv in kernel.temporary_variables.values()
                          if tv.address_space == AddressSpace.GLOBAL)
    temp_to_live_interval_start = {}
    temp_to_live_interval_end = {}
    return_from_kernel_idxs = get_return_from_kernel_mapping(kernel)

    for sched_idx, sched_item in enumerate(kernel.linearization):
        if isinstance(sched_item, RunInstruction):
            for var in (kernel.id_to_insn[sched_item.insn_id].dependency_names()
                        & temp_vars):
                if var not in temp_to_live_interval_start:
                    assert var not in temp_to_live_interval_end
                    temp_to_live_interval_start[var] = sched_idx
                assert var in temp_to_live_interval_start
                temp_to_live_interval_end[var] = return_from_kernel_idxs[sched_idx]
        elif isinstance(sched_item, (EnterLoop, LeaveLoop, CallKernel,
                                     ReturnFromKernel, Barrier)):
            # no variables are accessed within these schedule items => do
            # nothing.
            pass
        else:
            raise NotImplementedError(type(sched_item))

    vng = UniqueNameGenerator()
    # a mapping from shape to the available base storages from temp variables
    # that were dead.
    shape_to_available_base_storage = defaultdict(set)

    sched_idx_to_just_live_temp_vars = [set() for _ in kernel.linearization]
    sched_idx_to_just_dead_temp_vars = [set() for _ in kernel.linearization]

    for tv, just_alive_idx in temp_to_live_interval_start.items():
        sched_idx_to_just_live_temp_vars[just_alive_idx].add(tv)

    for tv, just_dead_idx in temp_to_live_interval_end.items():
        sched_idx_to_just_dead_temp_vars[just_dead_idx].add(tv)

    new_tvs = {}

    for sched_idx, _ in enumerate(kernel.linearization):
        just_dead_temps = sched_idx_to_just_dead_temp_vars[sched_idx]
        to_be_allocated_temps = sched_idx_to_just_live_temp_vars[sched_idx]
        for tv_name in sorted(just_dead_temps):
            tv = new_tvs[tv_name]
            assert tv.base_storage is not None
            assert tv.base_storage not in shape_to_available_base_storage[tv.nbytes]
            shape_to_available_base_storage[tv.nbytes].add(tv.base_storage)

        for tv_name in sorted(to_be_allocated_temps):
            assert len(to_be_allocated_temps) <= 1
            tv = kernel.temporary_variables[tv_name]
            assert tv.name not in new_tvs
            assert tv.base_storage is None
            if shape_to_available_base_storage[tv.nbytes]:
                base_storage = sorted(shape_to_available_base_storage[tv.nbytes])[0]
                shape_to_available_base_storage[tv.nbytes].remove(base_storage)
            else:
                base_storage = vng("_msh_actx_tmp_base")

            new_tvs[tv.name] = tv.copy(base_storage=base_storage)

    for name, tv in kernel.temporary_variables.items():
        if tv.address_space != AddressSpace.GLOBAL:
            new_tvs[name] = tv
        else:
            assert name in new_tvs

    kernel = kernel.copy(temporary_variables=new_tvs)

    return t_unit.with_kernel(kernel)


def _can_be_eagerly_computed(ary) -> bool:
    from pytato.transform import InputGatherer
    from pytato.array import Placeholder
    return all(not isinstance(inp, Placeholder)
               for inp in InputGatherer()(ary))


def deduplicate_data_wrappers(dag):
    import pytato as pt
    data_wrapper_cache = {}
    data_wrappers_encountered = 0

    def cached_data_wrapper_if_present(ary):
        nonlocal data_wrappers_encountered

        if isinstance(ary, pt.DataWrapper):

            data_wrappers_encountered += 1
            cache_key = (ary.data.base_data.int_ptr, ary.data.offset,
                         ary.shape, ary.data.strides)
            try:
                result = data_wrapper_cache[cache_key]
            except KeyError:
                result = ary
                data_wrapper_cache[cache_key] = result

            return result
        else:
            return ary

    dag = pt.transform.map_and_copy(dag, cached_data_wrapper_if_present)

    if data_wrappers_encountered:
        logger.info("data wrapper de-duplication: "
                "%d encountered, %d kept, %d eliminated",
                data_wrappers_encountered,
                len(data_wrapper_cache),
                data_wrappers_encountered - len(data_wrapper_cache))

    return dag


class SingleGridWorkBalancingPytatoArrayContext(PytatoPyOpenCLArrayContextBase):
    """
    A :class:`PytatoPyOpenCLArrayContext` that parallelizes work in an OpenCL
    kernel so that the work
    """
    def transform_loopy_program(self, t_unit):
        import loopy as lp

        t_unit = _single_grid_work_group_transform(t_unit, self.queue.device)
        t_unit = lp.set_options(t_unit, "insert_gbarriers")
        t_unit = lp.linearize(lp.preprocess_kernel(t_unit))
        t_unit = _alias_global_temporaries(t_unit)

        return t_unit

    def _get_fake_numpy_namespace(self):
        from meshmode.pytato_utils import (
            EagerReduceComputingPytatoFakeNumpyNamespace)
        return EagerReduceComputingPytatoFakeNumpyNamespace(self)

    def transform_dag(self, dag):
        import pytato as pt

        # {{{ face_mass: materialize einsum args

        def materialize_face_mass_vec(expr):
            if (isinstance(expr, pt.Einsum)
                    and pt.analysis.is_einsum_similar_to_subscript(
                        expr, "ifj,fej,fej->ei")):
                mat, jac, vec = expr.args
                return pt.einsum("ifj,fej,fej->ei",
                                 mat,
                                 jac,
                                 vec.tagged(pt.tags.ImplStored()))
            else:
                return expr

        dag = pt.transform.map_and_copy(dag, materialize_face_mass_vec)

        # }}}

        # {{{ materialize all einsums

        def materialize_einsums(ary: pt.Array) -> pt.Array:
            if isinstance(ary, pt.Einsum):
                return ary.tagged(pt.tags.ImplStored())

            return ary

        dag = pt.transform.map_and_copy(dag, materialize_einsums)

        # }}}

        dag = pt.transform.materialize_with_mpms(dag)
        dag = deduplicate_data_wrappers(dag)

        # {{{ /!\ Remove tags from Loopy call results.
        # See <https://www.github.com/inducer/pytato/issues/195>

        def untag_loopy_call_results(expr):
            from pytato.loopy import LoopyCallResult
            if isinstance(expr, LoopyCallResult):
                return expr.copy(tags=frozenset(),
                                 axes=(pt.Axis(frozenset()),)*expr.ndim)
            else:
                return expr

        dag = pt.transform.map_and_copy(dag, untag_loopy_call_results)

        # }}}

        return dag

# vim: foldmethod=marker
