__copyright__ = "Copyright (C) 2016 Matt Wala"

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

import logging
logger = logging.getLogger(__name__)

from pytools import log_process


# {{{ Build interpolation batches for group

def _build_interpolation_batches_for_group(
        actx, group_idx, coarse_discr_group, fine_discr_group, record):
    r"""
    To map between discretizations, we sort each of the fine mesh
    elements into an interpolation batch.  Which batch they go
    into is determined by where the refined unit nodes live
    relative to the coarse reference element.

    For instance, consider the following refinement::

         ______      ______
        |\     |    |\    e|
        | \    |    |d\    |
        |  \   |    |__\   |
        |   \  | => |\c|\  |
        |    \ |    |a\|b\ |
        |     \|    |  |  \|
         ‾‾‾‾‾‾      ‾‾‾‾‾‾

    Here, the discretization unit nodes for elements a,b,c,d,e
    will each have different positions relative to the reference
    element, so each element gets its own batch. On the other
    hand, for::

         ______      ______
        |\     |    |\ f|\e|
        | \    |    |d\ |g\|
        |  \   |    |__\|__|
        |   \  | => |\c|\  |
        |    \ |    |a\|b\h|
        |     \|    |  |  \|
         ‾‾‾‾‾‾      ‾‾‾‾‾‾

    the pairs {a,e}, {b,f}, {c,g}, {d,h} can share interpolation
    batches because their unit nodes are mapped from the same part
    of the reference element.
    """
    from meshmode.discretization.connection.direct import InterpolationBatch

    num_children = len(record.el_tess_info.children) \
                   if record.el_tess_info else 0
    from_bins = [[] for i in range(1 + num_children)]
    to_bins = [[] for i in range(1 + num_children)]
    for elt_idx, refinement_result in enumerate(record.element_mapping):
        if len(refinement_result) == 1:
            # Not refined -> interpolates to self
            from_bins[0].append(elt_idx)
            to_bins[0].append(refinement_result[0])
        else:
            assert len(refinement_result) == num_children
            # Refined -> interpolates to children
            for from_bin, to_bin, child_idx in zip(
                    from_bins[1:], to_bins[1:], refinement_result):
                from_bin.append(elt_idx)
                to_bin.append(child_idx)

    fine_unit_nodes = fine_discr_group.unit_nodes
    fine_meg = fine_discr_group.mesh_el_group

    from meshmode.mesh.refinement.utils import map_unit_nodes_to_children
    mapped_unit_nodes = map_unit_nodes_to_children(
            fine_meg, fine_unit_nodes, record.el_tess_info)

    from itertools import chain
    for from_bin, to_bin, unit_nodes in zip(
            from_bins, to_bins,
            chain([fine_unit_nodes], mapped_unit_nodes)):
        if not from_bin:
            continue
        yield InterpolationBatch(
            from_group_index=group_idx,
            from_element_indices=actx.from_numpy(np.asarray(from_bin)),
            to_element_indices=actx.from_numpy(np.asarray(to_bin)),
            result_unit_nodes=unit_nodes,
            to_element_face=None)

# }}}


@log_process(logger)
def make_refinement_connection(actx, refiner, coarse_discr, group_factory):
    """Return a
    :class:`meshmode.discretization.connection.DiscretizationConnection`
    connecting `coarse_discr` to a discretization on the fine mesh.

    :arg refiner: An instance of
        :class:`meshmode.mesh.refinement.Refiner`

    :arg coarse_discr: An instance of
        :class:`meshmode.discretization.Discretization` associated
        with the mesh given to the refiner

    :arg group_factory: An instance of
        :class:`meshmode.discretization.poly_element.ElementGroupFactory`. Used
        for discretizing the fine mesh.
    """
    from meshmode.discretization.connection import (
        DiscretizationConnectionElementGroup,
        DirectDiscretizationConnection)

    coarse_mesh = refiner.get_previous_mesh()
    fine_mesh = refiner.get_current_mesh()

    if coarse_discr.mesh != coarse_mesh:
        raise ValueError(
            "coarse_discr does not live on the same mesh given to the refiner")

    from meshmode.discretization import Discretization
    fine_discr = Discretization(
        actx,
        fine_mesh,
        group_factory,
        real_dtype=coarse_discr.real_dtype)

    groups = []
    for group_idx, (coarse_discr_group, fine_discr_group, record) in \
            enumerate(zip(coarse_discr.groups, fine_discr.groups,
                          refiner.group_refinement_records)):
        groups.append(
            DiscretizationConnectionElementGroup(
                list(_build_interpolation_batches_for_group(
                        actx, group_idx, coarse_discr_group,
                        fine_discr_group, record))))

    return DirectDiscretizationConnection(
        from_discr=coarse_discr,
        to_discr=fine_discr,
        groups=groups,
        is_surjective=True)


# vim: foldmethod=marker
