from __future__ import division

__copyright__ = "Copyright (C) 2014 Andreas Kloeckner"

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
import pyopencl as cl
import pyopencl.array  # noqa


__doc__ = """
.. autoclass:: DiscretizationConnection

.. autofunction:: make_same_mesh_connection

Implementation details
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: InterpolationGroup

.. autoclass:: DiscretizationConnectionElementGroup
"""


class InterpolationGroup(object):
    """
    .. attribute:: source_element_indices

        A :class:`numpy.ndarray` of length ``nelements``, containing the
        element index from which this "*to*" element's data will be
        interpolated.

    .. attribute:: target_element_indices

        A :class:`numpy.ndarray` of length ``nelements``, containing the
        element index to which this "*to*" element's data will be
        interpolated.

    .. attribute:: source_interpolation_nodes

        A :class:`numpy.ndarray` of shape
        ``(from_group.dim,to_group.nelements,to_group.nunit_nodes)``
        storing the coordinates of the nodes (in unit coordinates
        of the *from* reference element) from which the node
        locations of this element should be interpolated.
    """
    def __init__(self, source_element_indices,
            target_element_indices, source_interpolation_nodes):
        self.source_element_indices = source_element_indices
        self.target_element_indices = target_element_indices
        self.source_interpolation_nodes = source_interpolation_nodes

    @property
    def nelements(self):
        return len(self.source_element_indices)


class DiscretizationConnectionElementGroup(object):
    """
    .. attribute:: interpolation_groups

        A list of :class:`InterpolationGroup` instances.
    """
    def __init__(self, interpolation_groups):
        self.interpolation_groups = interpolation_groups


class DiscretizationConnection(object):
    """
    .. attribute:: from_discr

    .. attribute:: to_discr

    .. attribute:: groups

        a list of :class:`MeshConnectionGroup` instances, with
        a one-to-one correspondence to the groups in
        :attr:`from_discr` and :attr:`to_discr`.
    """

    def __init__(self, from_discr, to_discr, groups):
        if from_discr.cl_context != to_discr.cl_context:
            raise ValueError("from_discr and to_discr must live in the "
                    "same OpenCL context")

        self.cl_context = from_discr.cl_context

        self.from_discr = from_discr
        self.to_discr = to_discr
        self.groups = groups

    def __call__(self, queue, field):
        pass


# {{{ constructor functions

def make_same_mesh_connection(queue, from_discr, to_discr):
    if from_discr.mesh is not to_discr.mesh:
        raise ValueError("from_discr and to_discr must be based on "
                "the same mesh")

    assert queue.context == from_discr.cl_context
    assert queue.context == to_discr.cl_context

    groups = []
    for fgrp, tgrp in zip(from_discr.groups, to_discr.groups):
        all_elements = cl.array.arange(queue,
                fgrp.nelements,
                dtype=np.intp)
        igroup = InterpolationGroup(
                source_element_indices=all_elements,
                target_element_indices=all_elements,
                source_interpolation_nodes=tgrp.unit_nodes)

        groups.append(
                DiscretizationConnectionElementGroup(
                    igroup))

    return DiscretizationConnection(
            from_discr, to_discr, groups)

# }}}

# vim: foldmethod=marker
