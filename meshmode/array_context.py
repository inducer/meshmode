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
import numpy as np

from warnings import warn
from typing import Union, FrozenSet, Tuple, Any
from arraycontext import PyOpenCLArrayContext as PyOpenCLArrayContextBase
from arraycontext import PytatoPyOpenCLArrayContext as PytatoPyOpenCLArrayContextBase
from arraycontext.pytest import (
        _PytestPyOpenCLArrayContextFactoryWithClass,
        _PytestPytatoPyOpenCLArrayContextFactory,
        register_pytest_array_context_factory)
from loopy.translation_unit import for_each_kernel

from loopy.tools import memoize_on_disk
from pytools import ProcessLogger
from pytools.tag import UniqueTag, tag_dataclass

from meshmode.transform_metadata import (DiscretizationElementAxisTag,
                                         DiscretizationDOFAxisTag,
                                         DiscretizationFaceAxisTag,
                                         DiscretizationDimAxisTag,
                                         DiscretizationAmbientDimAxisTag,
                                         DiscretizationTopologicalDimAxisTag,
                                         DiscretizationFlattenedDOFAxisTag,
                                         DiscretizationEntityAxisTag)

from pyrsistent import pmap
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
            # FIXME: Need tighter assertion condition (this doesn't work when
            # zero-size arrays are present)
            # assert name in new_tvs
            pass

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


def get_temps_not_to_contract(knl):
    from functools import reduce
    wmap = knl.writer_map()
    rmap = knl.reader_map()

    temps_not_to_contract = set()
    for tv in knl.temporary_variables:
        if len(wmap.get(tv, set())) == 1:
            writer_id, = wmap[tv]
            writer_loop_nest = knl.id_to_insn[writer_id].within_inames
            insns_in_writer_loop_nest = reduce(frozenset.union,
                                               (knl.iname_to_insns()[iname]
                                                for iname in writer_loop_nest),
                                               frozenset())
            if (
                    (not (rmap.get(tv, frozenset())
                          <= insns_in_writer_loop_nest))
                    or len(knl.id_to_insn[writer_id].reduction_inames()) != 0
                    or any((len(knl.id_to_insn[reader_id].reduction_inames()) != 0)
                           for reader_id in rmap.get(tv, frozenset()))):
                temps_not_to_contract.add(tv)
        else:
            temps_not_to_contract.add(tv)
    return temps_not_to_contract

    # Better way to query it...
    # import loopy as lp
    # from kanren.constraints import neq as kanren_neq
    #
    # tempo = lp.relations.get_tempo(knl)
    # producero = lp.relations.get_producero(knl)
    # consumero = lp.relations.get_consumero(knl)
    # withino = lp.relations.get_withino(knl)
    # reduce_insno = lp.relations.get_reduce_insno(knl)
    #
    # # temp_k: temporary variable that cannot be contracted
    # temp_k = kanren.var()
    # producer_insn_k = kanren.var()
    # producer_loops_k = kanren.var()
    # consumer_insn_k = kanren.var()
    # consumer_loops_k = kanren.var()

    # temps_not_to_contract = kanren.run(0,
    #                                    temp_k,
    #                                    tempo(temp_k),
    #                                    producero(producer_insn_k,
    #                                              temp_k),
    #                                    consumero(consumer_insn_k,
    #                                              temp_k),
    #                                    withino(producer_insn_k,
    #                                            producer_loops_k),
    #                                    withino(consumer_insn_k,
    #                                            consumer_loops_k),
    #                                    kanren.lany(
    #                                        kanren_neq(
    #                                            producer_loops_k,
    #                                            consumer_loops_k),
    #                                        reduce_insno(consumer_insn_k)),
    #                                    results_filter=frozenset)
    # return temps_not_to_contract


def _is_iel_loop_part_of_global_dof_loops(iel: str, knl) -> bool:
    insn, = knl.iname_to_insns()[iel]
    return any(iname
               for iname in knl.id_to_insn[insn].within_inames
               if knl.iname_tags_of_type(iname, DiscretizationDOFAxisTag))


def _discr_entity_sort_key(discr_tag: DiscretizationEntityAxisTag
                           ) -> Tuple[Any, ...]:
    from dataclasses import fields
    key = [type(discr_tag).__name__]

    for field in fields(discr_tag):
        key.append(getattr(discr_tag, field.name))

    return tuple(key)


def _fuse_loops_over_a_discr_entity(knl,
                                    mesh_entity,
                                    fused_loop_prefix,
                                    should_fuse_redn_loops,
                                    orig_knl):
    import loopy as lp
    import kanren
    from functools import reduce
    taggedo = lp.relations.get_taggedo_of_type(orig_knl, mesh_entity)

    redn_loops = reduce(frozenset.union,
                        (insn.reduction_inames()
                         for insn in orig_knl.instructions),
                        frozenset())

    non_redn_loops = reduce(frozenset.union,
                            (insn.within_inames
                             for insn in orig_knl.instructions),
                            frozenset())

    # tag_k: tag of type 'mesh_entity'
    tag_k = kanren.var()
    tags = kanren.run(0,
                      tag_k,
                      taggedo(kanren.var(), tag_k),
                      results_filter=frozenset)
    for itag, tag in enumerate(
            sorted(tags, key=lambda x: _discr_entity_sort_key(x))):
        # iname_k: iname tagged with 'tag'
        iname_k = kanren.var()
        inames = kanren.run(0,
                            iname_k,
                            taggedo(iname_k, tag),
                            results_filter=frozenset)
        inames = frozenset(inames)
        if should_fuse_redn_loops:
            inames = inames & redn_loops
        else:
            inames = inames & non_redn_loops

        length_to_inames = {}
        for iname in inames:
            length = knl.get_constant_iname_length(iname)
            length_to_inames.setdefault(length, set()).add(iname)

        for i, (_, inames_to_fuse) in enumerate(
                sorted(length_to_inames.items())):
            knl = lp.rename_inames_in_batch(
                knl,
                lp.get_kennedy_unweighted_fusion_candidates(
                    knl, inames_to_fuse, prefix=f"{fused_loop_prefix}_{itag}_{i}_"))
        knl = lp.tag_inames(knl, {f"{fused_loop_prefix}_{itag}_*": tag})

    return knl


@memoize_on_disk
def fuse_same_discretization_entity_loops(knl):
    # maintain an 'orig_knl' to keep the original iname and tags before
    # transforming it.
    orig_knl = knl

    knl = _fuse_loops_over_a_discr_entity(knl, DiscretizationFaceAxisTag,
                                          "iface",
                                          False,
                                          orig_knl)

    knl = _fuse_loops_over_a_discr_entity(knl, DiscretizationElementAxisTag,
                                          "iel",
                                          False,
                                          orig_knl)

    knl = _fuse_loops_over_a_discr_entity(knl, DiscretizationDOFAxisTag,
                                          "idof",
                                          False,
                                          orig_knl)
    knl = _fuse_loops_over_a_discr_entity(knl, DiscretizationDimAxisTag,
                                          "idim",
                                          False,
                                          orig_knl)

    knl = _fuse_loops_over_a_discr_entity(knl, DiscretizationFaceAxisTag,
                                          "iface",
                                          True,
                                          orig_knl)
    knl = _fuse_loops_over_a_discr_entity(knl, DiscretizationDOFAxisTag,
                                          "idof",
                                          True,
                                          orig_knl)
    knl = _fuse_loops_over_a_discr_entity(knl, DiscretizationDimAxisTag,
                                          "idim",
                                          True,
                                          orig_knl)

    return knl


@memoize_on_disk
def contract_arrays(knl, callables_table):
    import loopy as lp
    from loopy.transform.precompute import precompute_for_single_kernel

    temps_not_to_contract = get_temps_not_to_contract(knl)
    all_temps = frozenset(knl.temporary_variables)

    logger.info("Array Contraction: Contracting "
                f"{len(all_temps-frozenset(temps_not_to_contract))} temps")

    wmap = knl.writer_map()

    for temp in sorted(all_temps - frozenset(temps_not_to_contract)):
        writer_id, = wmap[temp]
        rmap = knl.reader_map()
        ensm_tag, = knl.id_to_insn[writer_id].tags_of_type(EinsumTag)

        knl = lp.assignment_to_subst(knl, temp,
                                     remove_newly_unused_inames=False)
        if temp not in rmap:
            # no one was reading 'temp' i.e. dead code got eliminated :)
            assert f"{temp}_subst" not in knl.substitutions
            continue
        knl = precompute_for_single_kernel(
            knl, callables_table, f"{temp}_subst",
            sweep_inames=(),
            temporary_address_space=lp.AddressSpace.PRIVATE,
            compute_insn_id=f"_mm_contract_{temp}",
        )

        knl = lp.map_instructions(knl,
                                  f"id:_mm_contract_{temp}",
                                  lambda x: x.tagged(ensm_tag))

    return lp.remove_unused_inames(knl)


def _get_group_size_for_dof_array_loop(nunit_dofs):
    """
    Returns the OpenCL workgroup size for a loop iterating over the global DOFs
    of a discretization with *nunit_dofs* per cell.
    """
    if nunit_dofs == {6}:
        return 16, 6
    elif nunit_dofs == {10}:
        return 16, 10
    elif nunit_dofs == {20}:
        return 16, 10
    elif nunit_dofs == {1}:
        return 32, 1
    elif nunit_dofs == {2}:
        return 32, 2
    elif nunit_dofs == {4}:
        return 16, 4
    elif nunit_dofs == {3}:
        return 32, 3
    elif nunit_dofs == {35}:
        return 9, 7
    elif nunit_dofs == {15}:
        return 8, 8
    else:
        raise NotImplementedError(nunit_dofs)


def _get_iel_to_idofs(kernel):
    iel_inames = {iname
                  for iname in kernel.all_inames()
                  if (kernel
                      .inames[iname]
                      .tags_of_type((DiscretizationElementAxisTag,
                                     DiscretizationFlattenedDOFAxisTag)))
                  }
    idof_inames = {iname
                   for iname in kernel.all_inames()
                   if (kernel
                       .inames[iname]
                       .tags_of_type(DiscretizationDOFAxisTag))
                   }
    iface_inames = {iname
                    for iname in kernel.all_inames()
                    if (kernel
                        .inames[iname]
                        .tags_of_type(DiscretizationFaceAxisTag))
                    }
    idim_inames = {iname
                   for iname in kernel.all_inames()
                   if (kernel
                       .inames[iname]
                       .tags_of_type(DiscretizationDimAxisTag))
                   }

    iel_to_idofs = {iel: set() for iel in iel_inames}

    for insn in kernel.instructions:
        if (len(insn.within_inames) == 1
                and (insn.within_inames) <= iel_inames):
            iel, = insn.within_inames
            if all(kernel.id_to_insn[el_insn].within_inames == insn.within_inames
                    for el_insn in kernel.iname_to_insns()[iel]):
                # the iel here doesn't interfere with any idof i.e. we
                # support parallelizing such loops.
                pass
            else:
                raise NotImplementedError(f"The <iel> loop {insn.within_inames}"
                                          " does not appear as a singly nested"
                                          " loop.")
        elif ((len(insn.within_inames) == 2)
              and (len(insn.within_inames & iel_inames) == 1)
              and (len(insn.within_inames & idof_inames) == 1)):
            iel, = insn.within_inames & iel_inames
            idof, = insn.within_inames & idof_inames
            iel_to_idofs[iel].add(idof)
            if all((iel in kernel.id_to_insn[dof_insn].within_inames)
                   for dof_insn in kernel.iname_to_insns()[idof]):
                pass
            else:
                raise NotImplementedError("The <iel,idof> loop "
                                          f"'{insn.within_inames}' has the idof-loop"
                                          " that's not nested within the iel-loop.")
        elif ((len(insn.within_inames) > 2)
                and (len(insn.within_inames & iel_inames) == 1)
                and (len(insn.within_inames & idof_inames) == 1)
                and (len(insn.within_inames & (idim_inames | iface_inames))
                     == (len(insn.within_inames) - 2))):
            iel, = insn.within_inames & iel_inames
            idof, = insn.within_inames & idof_inames
            iel_to_idofs[iel].add(idof)
            if all((all({iel, idof} <= kernel.id_to_insn[non_iel_insn].within_inames
                        for non_iel_insn in kernel.iname_to_insns()[non_iel_iname]))
                   for non_iel_iname in insn.within_inames - {iel}):
                iel_to_idofs[iel].add(idof)
            else:
                raise NotImplementedError("Could not fit into  <iel,idof,iface>"
                                          " loop nest pattern.")
        else:
            raise NotImplementedError(f"Cannot fit loop nest '{insn.within_inames}'"
                                      " into known set of loop-nest patterns.")

    return pmap({iel: frozenset(idofs)
                 for iel, idofs in iel_to_idofs.items()})


def _get_iel_loop_from_insn(insn, knl):
    iel, = {iname
            for iname in insn.within_inames
            if knl.inames[iname].tags_of_type((DiscretizationElementAxisTag,
                                               DiscretizationFlattenedDOFAxisTag))}
    return iel


def _get_element_loop_topo_sorted_order(knl):
    dag = {iel: set()
           for iel in knl.all_inames()
           if knl.inames[iel].tags_of_type(DiscretizationElementAxisTag)}

    for insn in knl.instructions:
        succ_iel = _get_iel_loop_from_insn(insn, knl)
        for dep_id in insn.depends_on:
            pred_iel = _get_iel_loop_from_insn(knl.id_to_insn[dep_id], knl)
            if pred_iel != succ_iel:
                dag[pred_iel].add(succ_iel)

    from pytools.graph import compute_topological_order
    return compute_topological_order(dag, key=lambda x: x)


@tag_dataclass
class EinsumTag(UniqueTag):
    orig_loop_nest: FrozenSet[str]


def _prepare_kernel_for_parallelization(kernel):
    discr_tag_to_prefix = {DiscretizationElementAxisTag: "iel",
                           DiscretizationDOFAxisTag: "idof",
                           DiscretizationDimAxisTag: "idim",
                           DiscretizationAmbientDimAxisTag: "idim",
                           DiscretizationTopologicalDimAxisTag: "idim",
                           DiscretizationFlattenedDOFAxisTag: "imsh_nodes",
                           DiscretizationFaceAxisTag: "iface"}
    import loopy as lp
    from loopy.match import ObjTagged

    # A mapping from inames that the instruction accesss to
    # the instructions ids within that iname.
    ensm_buckets = {}
    vng = kernel.get_var_name_generator()

    for insn in kernel.instructions:
        inames = insn.within_inames | insn.reduction_inames()
        ensm_buckets.setdefault(tuple(sorted(inames)), set()).add(insn.id)

    # FIXME: Dependency violation is a big concern here
    # Waiting on the loopy feature: https://github.com/inducer/loopy/issues/550

    for ieinsm, (loop_nest, insns) in enumerate(sorted(ensm_buckets.items())):
        new_insns = [insn.tagged(EinsumTag(frozenset(loop_nest)))
                     if insn.id in insns
                     else insn
                     for insn in kernel.instructions]
        kernel = kernel.copy(instructions=new_insns)

        new_inames = []
        for iname in loop_nest:
            discr_tag, = kernel.iname_tags_of_type(iname,
                                                   DiscretizationEntityAxisTag)
            new_iname = vng(f"{discr_tag_to_prefix[type(discr_tag)]}_ensm{ieinsm}")
            new_inames.append(new_iname)

        kernel = lp.duplicate_inames(
            kernel,
            loop_nest,
            within=ObjTagged(EinsumTag(frozenset(loop_nest))),
            new_inames=new_inames,
            tags=kernel.iname_to_tags)

    return kernel


def _get_elementwise_einsum(t_unit, einsum_tag):
    import loopy as lp
    import feinsum as fnsm
    from loopy.match import ObjTagged
    from pymbolic.primitives import Variable, Subscript

    kernel = t_unit.default_entrypoint

    assert isinstance(einsum_tag, EinsumTag)
    insn_match = ObjTagged(einsum_tag)

    global_vars = ({tv.name
                    for tv in kernel.temporary_variables.values()
                    if tv.address_space == lp.AddressSpace.GLOBAL}
                   | set(kernel.arg_dict.keys()))
    insns = [insn
             for insn in kernel.instructions
             if insn_match(kernel, insn)]
    idx_tuples = set()

    for insn in insns:
        assert len(insn.assignees) == 1
        if isinstance(insn.assignee, Variable):
            if insn.assignee.name in global_vars:
                raise NotImplementedError(insn)
            else:
                assert (kernel.temporary_variables[insn.assignee.name].address_space
                        == lp.AddressSpace.PRIVATE)
        elif isinstance(insn.assignee, Subscript):
            assert insn.assignee_name in global_vars
            idx_tuples.add(tuple(idx.name
                                 for idx in insn.assignee.index_tuple))
        else:
            raise NotImplementedError(insn)

    if len(idx_tuples) != 1:
        raise NotImplementedError("Multiple einsums in the same loop nest =>"
                                  " not allowed.")
    idx_tuple, = idx_tuples
    subscript = "{lhs}, {lhs}->{lhs}".format(
        lhs="".join(chr(97+i)
                    for i in range(len(idx_tuple))))
    arg_shape = tuple(np.inf
                      if kernel.iname_tags_of_type(idx, DiscretizationElementAxisTag)
                      else kernel.get_constant_iname_length(idx)
                      for idx in idx_tuple)
    return fnsm.einsum(subscript,
                       fnsm.array(arg_shape, "float64"),
                       fnsm.array(arg_shape, "float64"))


def _combine_einsum_domains(knl):
    import islpy as isl
    from functools import reduce

    new_domains = []
    einsum_tags = reduce(
        frozenset.union,
        (insn.tags_of_type(EinsumTag)
         for insn in knl.instructions),
        frozenset())

    for tag in sorted(einsum_tags,
                      key=lambda x: sorted(x.orig_loop_nest)):
        insns = [insn
                 for insn in knl.instructions
                 if tag in insn.tags]
        inames = reduce(frozenset.union,
                        ((insn.within_inames | insn.reduction_inames())
                         for insn in insns),
                        frozenset())
        domain = knl.get_inames_domain(frozenset(inames))
        new_domains.append(domain.project_out_except(sorted(inames),
                                                     [isl.dim_type.set]))

    return knl.copy(domains=new_domains)


def _rewrite_tvs_as_base_plus_offset(t_unit, device):
    import loopy as lp
    knl = t_unit.default_entrypoint
    vng = knl.get_var_name_generator()
    nbytes_to_base_storages = {}
    for tv in knl.temporary_variables.values():
        if tv.address_space == lp.AddressSpace.GLOBAL:
            nbytes_to_base_storages.setdefault(tv.nbytes,
                                               set()).add(tv.base_storage)

    nbytes_to_new_storage_name = {nbytes: vng("_mm_base_storage")
                                  for nbytes in sorted(nbytes_to_base_storages)}

    if any(nbytes > device.max_mem_alloc_size
            for nbytes in nbytes_to_new_storage_name):
        raise RuntimeError("Some of the variables "
                           "require more memory than the CL-device "
                           "allows.")

    old_storage_to_new_storage_plus_offset = {}
    new_storage_to_alloc_nbytes = {}
    for nbytes, old_storages in nbytes_to_base_storages.items():
        new_storage_name = nbytes_to_new_storage_name[nbytes]
        offset = 0
        new_storage_to_alloc_nbytes[new_storage_name] = offset
        for old_storage in sorted(old_storages):
            assert (offset + nbytes) < device.max_mem_alloc_size
            old_storage_to_new_storage_plus_offset[old_storage] = (
                (new_storage_name, offset))
            offset = offset + nbytes
            new_storage_to_alloc_nbytes[new_storage_name] = offset
            if (offset + nbytes) > device.max_mem_alloc_size:
                new_storage_name = vng("_mm_base_storage")
                offset = 0

    del nbytes_to_new_storage_name

    new_tvs = {}
    for name, tv in knl.temporary_variables.items():
        if tv.address_space == lp.AddressSpace.GLOBAL:
            new_storage_name, offset_nbytes = (
                old_storage_to_new_storage_plus_offset[tv.base_storage])
            new_storage_size = (
                new_storage_to_alloc_nbytes[new_storage_name]
                // tv.dtype.numpy_dtype.itemsize)
            tv = tv.copy(base_storage=new_storage_name,
                         offset=offset_nbytes//tv.dtype.numpy_dtype.itemsize,
                         storage_shape=(new_storage_size,) + (1,)*(len(tv.shape)-1)
                         )

        new_tvs[name] = tv

    knl = knl.copy(temporary_variables=new_tvs)
    return t_unit.with_kernel(knl)


class FusionContractorArrayContext(
        SingleGridWorkBalancingPytatoArrayContext):

    def transform_dag(self, dag):
        import pytato as pt

        # {{{ CSE

        with ProcessLogger(logger, "transform_dag.mpms_materialization"):
            dag = pt.transform.materialize_with_mpms(dag)

        def mark_materialized_nodes_as_cse(
                    ary: Union[pt.Array,
                                pt.AbstractResultWithNamedArrays]) -> pt.Array:
            if isinstance(ary, pt.AbstractResultWithNamedArrays):
                return ary

            if ary.tags_of_type(pt.tags.ImplStored):
                return ary.tagged(pt.tags.PrefixNamed("cse"))
            else:
                return ary

        with ProcessLogger(logger, "transform_dag.naming_cse"):
            dag = pt.transform.map_and_copy(dag, mark_materialized_nodes_as_cse)

        # }}}

        # {{{ indirect addressing are non-negative

        indirection_maps = set()

        class _IndirectionMapRecorder(pt.transform.CachedWalkMapper):
            def post_visit(self, expr):
                if isinstance(expr, pt.IndexBase):
                    for idx in expr.indices:
                        if isinstance(idx, pt.Array):
                            indirection_maps.add(idx)

        _IndirectionMapRecorder()(dag)

        def tag_indices_as_non_negative(ary):
            if ary in indirection_maps:
                return ary.tagged(pt.tags.AssumeNonNegative())
            else:
                return ary

        with ProcessLogger(logger, "transform_dag.tag_indices_as_non_negative"):
            dag = pt.transform.map_and_copy(dag, tag_indices_as_non_negative)

        # }}}

        with ProcessLogger(logger, "transform_dag.deduplicate_data_wrappers"):
            dag = pt.transform.deduplicate_data_wrappers(dag)

        # {{{ get rid of copies for different views of a cl-array

        def eliminate_reshapes_of_data_wrappers(ary):
            if (isinstance(ary, pt.Reshape)
                    and isinstance(ary.array, pt.DataWrapper)):
                return pt.make_data_wrapper(ary.array.data.reshape(ary.shape),
                                            tags=ary.tags,
                                            axes=ary.axes)
            else:
                return ary

        dag = pt.transform.map_and_copy(dag,
                                        eliminate_reshapes_of_data_wrappers)

        # }}}

        # {{{ face_mass: materialize einsum args

        def materialize_face_mass_input_and_output(expr):
            if (isinstance(expr, pt.Einsum)
                    and pt.analysis.is_einsum_similar_to_subscript(
                            expr,
                            "ifj,fej,fej->ei")):
                mat, jac, vec = expr.args
                return (pt.einsum("ifj,fej,fej->ei",
                                  mat,
                                  jac,
                                  vec.tagged(pt.tags.ImplStored()))
                        .tagged((pt.tags.ImplStored(),
                                 pt.tags.PrefixNamed("face_mass"))))
            else:
                return expr

        with ProcessLogger(logger,
                           "transform_dag.materialize_face_mass_ins_and_outs"):
            dag = pt.transform.map_and_copy(dag,
                                            materialize_face_mass_input_and_output)

        # }}}

        # {{{ materialize inverse mass inputs

        def materialize_inverse_mass_inputs(expr):
            if (isinstance(expr, pt.Einsum)
                    and pt.analysis.is_einsum_similar_to_subscript(
                            expr,
                            "ei,ij,ej->ei")):
                arg1, arg2, arg3 = expr.args
                if not arg3.tags_of_type(pt.tags.PrefixNamed):
                    arg3 = arg3.tagged(pt.tags.PrefixNamed("mass_inv_inp"))
                if not arg3.tags_of_type(pt.tags.ImplStored):
                    arg3 = arg3.tagged(pt.tags.ImplStored())

                return pt.Einsum(expr.access_descriptors,
                                 (arg1, arg2, arg3),
                                 expr.axes,
                                 expr.redn_axis_to_redn_descr,
                                 expr.index_to_access_descr,
                                 expr.tags)
            else:
                return expr

        dag = pt.transform.map_and_copy(dag, materialize_inverse_mass_inputs)

        # }}}

        # {{{ materialize all einsums

        def materialize_all_einsums_or_reduces(expr):
            from pytato.raising import (index_lambda_to_high_level_op,
                                        ReduceOp)

            if isinstance(expr, pt.Einsum):
                return expr.tagged(pt.tags.ImplStored())
            elif (isinstance(expr, pt.IndexLambda)
                    and isinstance(index_lambda_to_high_level_op(expr), ReduceOp)):
                return expr.tagged(pt.tags.ImplStored())
            else:
                return expr

        with ProcessLogger(logger,
                           "transform_dag.materialize_all_einsums_or_reduces"):
            dag = pt.transform.map_and_copy(dag, materialize_all_einsums_or_reduces)

        # }}}

        # {{{ infer axis types

        from meshmode.pytato_utils import unify_discretization_entity_tags

        with ProcessLogger(logger, "transform_dag.infer_axes_tags"):
            dag = unify_discretization_entity_tags(dag)

        # }}}

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

        # {{{ remove broadcasts from einsums: help feinsum

        ensm_arg_rewrite_cache = {}

        def _get_rid_of_broadcasts_from_einsum(expr):
            # Helpful for matching against the available expressions
            # in feinsum.

            from pytato.utils import (are_shape_components_equal,
                                      are_shapes_equal)
            if isinstance(expr, pt.Einsum):
                from pytato.array import EinsumElementwiseAxis
                idx_to_len = expr._access_descr_to_axis_len()
                new_access_descriptors = []
                new_args = []
                inp_gatherer = pt.transform.InputGatherer()
                access_descr_to_axes = dict(expr.redn_axis_to_redn_descr)
                for iax, axis in enumerate(expr.axes):
                    access_descr_to_axes[EinsumElementwiseAxis(iax)] = axis

                for access_descrs, arg in zip(expr.access_descriptors,
                                              expr.args):
                    new_shape = []
                    new_access_descrs = []
                    new_axes = []
                    for iaxis, (access_descr, axis_len) in enumerate(
                            zip(access_descrs,
                                arg.shape)):
                        if not are_shape_components_equal(axis_len,
                                                          idx_to_len[access_descr]):
                            assert are_shape_components_equal(axis_len, 1)
                            if any(isinstance(inp, pt.Placeholder)
                                   for inp in inp_gatherer(arg)):
                                # do not get rid of broadcasts from parameteric
                                # data.
                                new_shape.append(axis_len)
                                new_access_descrs.append(access_descr)
                                new_axes.append(arg.axes[iaxis])
                        else:
                            new_axes.append(arg.axes[iaxis])
                            new_shape.append(axis_len)
                            new_access_descrs.append(access_descr)

                    if not are_shapes_equal(new_shape, arg.shape):
                        assert len(new_axes) == len(new_shape)
                        arg_to_freeze = (arg.reshape(new_shape)
                                         .copy(axes=tuple(
                                             access_descr_to_axes[acc_descr]
                                             for acc_descr in new_access_descrs)))

                        try:
                            new_arg = ensm_arg_rewrite_cache[arg_to_freeze]
                        except KeyError:
                            new_arg = self.thaw(self.freeze(arg_to_freeze))
                            ensm_arg_rewrite_cache[arg_to_freeze] = new_arg

                        arg = new_arg

                    assert arg.ndim == len(new_access_descrs)
                    new_args.append(arg)
                    new_access_descriptors.append(tuple(new_access_descrs))

                return pt.Einsum(tuple(new_access_descriptors),
                                 tuple(new_args),
                                 tags=expr.tags,
                                 axes=expr.axes,
                                 redn_axis_to_redn_descr=(expr
                                                          .redn_axis_to_redn_descr),
                                 index_to_access_descr=expr.index_to_access_descr)
            else:
                return expr

        dag = pt.transform.map_and_copy(dag, _get_rid_of_broadcasts_from_einsum)

        # }}}

        # {{{ remove any PartID tags

        from pytato.distributed import PartIDTag

        def remove_part_id_tags(expr):
            if isinstance(expr, pt.Array) and expr.tags_of_type(PartIDTag):
                tag, = expr.tags_of_type(PartIDTag)
                return expr.without_tags(tag)
            else:
                return expr

        dag = pt.transform.map_and_copy(dag, remove_part_id_tags)

        # }}}

        # {{{ untag outputs tagged from being tagged ImplStored

        def _untag_impl_stored(expr):
            if isinstance(expr, pt.InputArgumentBase):
                return expr
            else:
                return expr.without_tags(pt.tags.ImplStored(),
                                         verify_existence=False)

        dag = pt.make_dict_of_named_arrays({
                name: _untag_impl_stored(named_ary.expr)
                for name, named_ary in dag.items()})

        # }}}

        return dag

    def transform_loopy_program(self, t_unit):
        import loopy as lp
        from functools import reduce
        from arraycontext.impl.pytato.compile import FromArrayContextCompile

        original_t_unit = t_unit

        # from loopy.transform.instruction import simplify_indices
        # t_unit = simplify_indices(t_unit)

        knl = t_unit.default_entrypoint

        logger.info(f"Transforming kernel with {len(knl.instructions)} statements.")

        # {{{ fallback: if the inames are not inferred which mesh entity they
        # iterate over.

        for iname in knl.all_inames():
            if not knl.iname_tags_of_type(iname, DiscretizationEntityAxisTag):
                warn("Falling back to a slower transformation strategy as some"
                     " loops are uninferred which mesh entity they belong to.",
                     stacklevel=2)

                return super().transform_loopy_program(original_t_unit)

        # }}}

        # {{{ hardcode offset to 0  (sorry humanity)

        knl = knl.copy(args=[arg.copy(offset=0)
                             for arg in knl.args])

        # }}}

        # {{{ loop fusion

        with ProcessLogger(logger, "Loop Fusion"):
            knl = fuse_same_discretization_entity_loops(knl)

        # }}}

        # {{{ align kernels for fused einsums

        knl = _prepare_kernel_for_parallelization(knl)
        knl = _combine_einsum_domains(knl)

        # }}}

        # {{{ array contraction

        with ProcessLogger(logger, "Array Contraction"):
            knl = contract_arrays(knl, t_unit.callables_table)

        # }}}

        # {{{ Stats Collection (Disabled)

        if 0:
            with ProcessLogger(logger, "Counting Kernel Ops"):
                from loopy.kernel.array import ArrayBase
                from pytools import product
                knl = knl.copy(
                    silenced_warnings=(knl.silenced_warnings
                                        + ["insn_count_subgroups_upper_bound",
                                            "summing_if_branches_ops"]))

                t_unit = t_unit.with_kernel(knl)

                op_map = lp.get_op_map(t_unit, subgroup_size=32)

                c64_ops = {op_type: (op_map.filter_by(dtype=[np.complex64],
                                                      name=op_type,
                                                      kernel_name=knl.name)
                                      .eval_and_sum({}))
                            for op_type in ["add", "mul", "div"]}
                c128_ops = {op_type: (op_map.filter_by(dtype=[np.complex128],
                                                       name=op_type,
                                                       kernel_name=knl.name)
                                      .eval_and_sum({}))
                            for op_type in ["add", "mul", "div"]}
                f32_ops = ((op_map.filter_by(dtype=[np.float32],
                                             kernel_name=knl.name)
                            .eval_and_sum({}))
                           + (2 * c64_ops["add"]
                              + 6 * c64_ops["mul"]
                              + (6 + 3 + 2) * c64_ops["div"]))
                f64_ops = ((op_map.filter_by(dtype=[np.float64],
                                             kernel_name="_pt_kernel")
                            .eval_and_sum({}))
                           + (2 * c128_ops["add"]
                              + 6 * c128_ops["mul"]
                              + (6 + 3 + 2) * c128_ops["div"]))

                # {{{ footprint gathering

                nfootprint_bytes = 0

                for ary in knl.args:
                    if (isinstance(ary, ArrayBase)
                            and ary.address_space == lp.AddressSpace.GLOBAL):
                        nfootprint_bytes += (product(ary.shape)
                                            * ary.dtype.itemsize)

                for ary in knl.temporary_variables.values():
                    if ary.address_space == lp.AddressSpace.GLOBAL:
                        # global temps would be written once and read once
                        nfootprint_bytes += (2 * product(ary.shape)
                                            * ary.dtype.itemsize)

                # }}}

                if f32_ops:
                    logger.info(f"Single-prec. GFlOps: {f32_ops * 1e-9}")
                if f64_ops:
                    logger.info(f"Double-prec. GFlOps: {f64_ops * 1e-9}")
                logger.info(f"Footprint GBs: {nfootprint_bytes * 1e-9}")

        # }}}

        # {{{ check whether we can parallelize the kernel

        try:
            iel_to_idofs = _get_iel_to_idofs(knl)
        except NotImplementedError as err:
            if knl.tags_of_type(FromArrayContextCompile):
                raise err
            else:
                warn("FusionContractorArrayContext.transform_loopy_program not"
                     " broad enough (yet). Falling back to a possibly slower"
                     " transformation strategy.")
                return super().transform_loopy_program(original_t_unit)

        # }}}

        # {{{ insert barriers between consecutive iel-loops

        toposorted_iels = _get_element_loop_topo_sorted_order(knl)

        for iel_pred, iel_succ in zip(toposorted_iels[:-1],
                                      toposorted_iels[1:]):
            knl = lp.add_barrier(knl,
                                 insn_before=f"iname:{iel_pred}",
                                 insn_after=f"iname:{iel_succ}")

        # }}}

        # {{{ Parallelization strategy: Use feinsum

        t_unit = t_unit.with_kernel(knl)
        del knl

        if False and t_unit.default_entrypoint.tags_of_type(FromArrayContextCompile):
            # FIXME: Enable this branch, WIP for now and hence disabled it.
            from loopy.match import ObjTagged
            import feinsum as fnsm
            from meshmode.feinsum_transformations import FEINSUM_TO_TRANSFORMS

            assert all(insn.tags_of_type(EinsumTag)
                       for insn in t_unit.default_entrypoint.instructions
                       if isinstance(insn, lp.MultiAssignmentBase)
                       )

            einsum_tags = reduce(
                frozenset.union,
                (insn.tags_of_type(EinsumTag)
                 for insn in t_unit.default_entrypoint.instructions),
                frozenset())
            for ensm_tag in sorted(einsum_tags,
                                   key=lambda x: sorted(x.orig_loop_nest)):
                if reduce(frozenset.union,
                          (insn.reduction_inames()
                           for insn in (t_unit.default_entrypoint.instructions)
                           if ensm_tag in insn.tags),
                          frozenset()):
                    fused_einsum = fnsm.match_einsum(t_unit, ObjTagged(ensm_tag))
                else:
                    # elementwise loop
                    fused_einsum = _get_elementwise_einsum(t_unit, ensm_tag)

                try:
                    fnsm_transform = FEINSUM_TO_TRANSFORMS[
                        fnsm.normalize_einsum(fused_einsum)]
                except KeyError:
                    fnsm.query(fused_einsum,
                               self.queue.context,
                               err_if_no_results=True)
                    1/0

                t_unit = fnsm_transform(t_unit,
                                        insn_match=ObjTagged(ensm_tag))
        else:
            knl = t_unit.default_entrypoint
            for iel, idofs in sorted(iel_to_idofs.items()):
                if idofs:
                    nunit_dofs = {knl.get_constant_iname_length(idof)
                                  for idof in idofs}
                    idof, = idofs

                    l_one_size, l_zero_size = _get_group_size_for_dof_array_loop(
                        nunit_dofs)

                    knl = lp.split_iname(knl, iel, l_one_size,
                                         inner_tag="l.1", outer_tag="g.0")
                    knl = lp.split_iname(knl, idof, l_zero_size,
                                         inner_tag="l.0", outer_tag="unr")
                else:
                    knl = lp.split_iname(knl, iel, 32,
                                         outer_tag="g.0", inner_tag="l.0")

            t_unit = t_unit.with_kernel(knl)

        # }}}

        t_unit = lp.linearize(lp.preprocess_kernel(t_unit))
        t_unit = _alias_global_temporaries(t_unit)
        t_unit = _rewrite_tvs_as_base_plus_offset(t_unit, self.queue.device)

        return t_unit

# vim: foldmethod=marker
