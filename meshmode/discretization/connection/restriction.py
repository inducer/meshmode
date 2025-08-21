__copyright__ = "Copyright (C) 2022 University of Illinois Board of Trustees"

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


from typing import Tuple
import numpy as np

from arraycontext import ArrayContext

from meshmode.mesh import Mesh
from meshmode.discretization import Discretization, ElementGroupBase
from meshmode.discretization.connection.direct import (
        DirectDiscretizationConnection, DiscretizationConnectionElementGroup,
        InterpolationBatch)


def _group_and_index_from_elements_indices(
        discr: Discretization, element_indices: np.ndarray
        ) -> Tuple[int, ElementGroupBase]:

    from_group_index = 0
    element_nr_base = 0
    for igrp, grp in enumerate(discr.groups):
        if np.array_equal(
                np.arange(element_nr_base, element_nr_base+grp.nelements),
                element_indices):
            from_group_index = igrp

    if from_group_index is None:
        raise NotImplementedError("element_indices does not reflect an entire "
                "element group. That's not supported at this point. Help welcome!")

    return from_group_index, grp


def make_volume_restriction(actx: ArrayContext, discr: Discretization,
        element_indices: np.ndarray) -> DirectDiscretizationConnection:
    """
    :param element_indices: An array of integers indicating element numbers
        to extract. Must be sorted in ascending order.

    .. note::

        Only supports extracting entire element groups for now.
    """

    # FIXME Generalize me to support arbitrary element subsets.

    from_group_index, from_group = _group_and_index_from_elements_indices(
            discr, element_indices)

    submesh = Mesh(
            vertices=discr.mesh.vertices,
            groups=discr.mesh.groups[from_group_index:from_group_index+1],
            vertex_id_dtype=discr.mesh.vertex_id_dtype,
            element_id_dtype=discr.mesh.element_id_dtype,
            is_conforming=discr.mesh.is_conforming)

    subdiscr = Discretization(
            actx, submesh, discr._group_factory,
            real_dtype=discr.real_dtype)

    conn_groups = [
            DiscretizationConnectionElementGroup([
                InterpolationBatch(
                    from_group_index=from_group_index,
                    from_element_indices=actx.freeze(actx.from_numpy(
                        np.arange(from_group.nelements))),
                    to_element_indices=actx.freeze(actx.from_numpy(
                        np.arange(from_group.nelements))),
                    result_unit_nodes=from_group.unit_nodes,
                    to_element_face=None,
                )])
            ]
    return DirectDiscretizationConnection(
            discr, subdiscr, conn_groups,
            is_surjective=False)


def make_volume_embedding(actx: ArrayContext, discr: Discretization,
        element_indices: np.ndarray) -> DirectDiscretizationConnection:
    """
    :param element_indices: An array of integers indicating element numbers
        to extract. Must be sorted in ascending order.

    .. note::

        Only supports extracting entire element groups for now.
    """

# vim: foldmethod=marker
