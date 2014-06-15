from __future__ import division

__copyright__ = "Copyright (C) 2010,2012,2013 Andreas Kloeckner, Michael Tom"

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


#import numpy as np
import modepy as mp
#import numpy.linalg as la
from pytools import Record


# {{{ element group

class MeshElementGroup(Record):
    """A group of elements sharing a common reference element.

    .. attribute:: order

    .. attribute:: vertex_indices

        An array *(nelements, ref_element.nvertices)* of (mesh-wide)
        vertex indices.

    .. attribute:: nodes

        An array of node coordinates with shape
        *(mesh.ambient_dim, nelements, ref_element.nnodes)*.

    .. attribute:: element_nr_base

        Lowest element number in this element group.

    .. attribute:: node_nr_base

        Lowest node number in this element group.

    .. attribute:: dim

        The number of dimensions spanned by the element.
        *Not* the ambient dimension, see :attr:`Mesh.ambient_dim`
        for that.
    """

    def __init__(self, order, vertex_indices, nodes,
            element_nr_base=None, node_nr_base=None,
            unit_nodes=None, dim=None):
        """
        :arg order: the mamximum total degree used for interpolation.
        :arg nodes: ``[ambient_dim, nelements, nunit_nodes]``
            The nodes are assumed to be mapped versions of *unit_nodes*.
        :arg unit_nodes: ``[dim, nunit_nodes]``
            The unit nodes of which *nodes* is a mapped
            version. If unspecified, the nodes from
            :func:`modepy.warp_and_blend_nodes` for *dim*
            are assumed. These must be in unit coordinates
            as defined in :mod:`modepy.nodes`.
        :arg dim: only used if *unit_nodes* is None, to get
            the default unit nodes.

        Do not supply *element_nr_base* and *node_nr_base*, they will be
        automatically assigned.
        """

        if unit_nodes is None:
            unit_nodes = mp.warp_and_blend_nodes(dim, order)

        dims = unit_nodes.shape[0]

        if vertex_indices.shape[-1] != dims+1:
            raise ValueError("vertex_indices has wrong number of vertices per "
                    "element. expected: %d, got: %d" % (dims+1,
                        vertex_indices.shape[-1]))

        Record.__init__(self,
            order=order,
            vertex_indices=vertex_indices,
            nodes=nodes,
            unit_nodes=unit_nodes,
            element_nr_base=None, node_nr_base=None)

    @property
    def dim(self):
        return self.unit_nodes.shape[0]

    def join_mesh(self, element_nr_base, node_nr_base):
        if self.element_nr_base is not None:
            raise RuntimeError("this element group has already joined a mesh, "
                    "cannot join another")

        return self.copy(
                element_nr_base=element_nr_base,
                node_nr_base=node_nr_base)

    @property
    def nelements(self):
        return self.vertex_indices.shape[0]

    @property
    def nnodes(self):
        return self.nelements * self.unit_nodes.shape[-1]

    @property
    def nunit_nodes(self):
        return self.unit_nodes.shape[-1]

# }}}


# {{{ mesh

class Mesh(Record):
    """
    .. attribute:: vertices

        An array of vertex coordinates with shape
        *(ambient_dim, nvertices)*

    .. attribute:: groups

        A list of :class:`MeshElementGroup` instances.
    """

    def __init__(self, vertices, groups):
        el_nr = 0
        node_nr = 0

        new_groups = []
        for g in groups:
            ng = g.join_mesh(el_nr, node_nr)
            new_groups.append(ng)
            el_nr += ng.nelements
            node_nr += ng.nnodes

        Record.__init__(self, vertices=vertices, groups=groups)

    @property
    def ambient_dim(self):
        return self.vertices.shape[0]

    @property
    def dim(self):
        from pytools import single_valued
        return single_valued(grp.dim for grp in self.groups)

    # Design experience: Try not to add too many global data structures
    # to the mesh. Let the element groups be responsible for that.
    # The interpolatory discretization has these big global things.

# }}}


# vim: foldmethod=marker
