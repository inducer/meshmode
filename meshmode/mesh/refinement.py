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


# {{{ simple, no-connectivity refinement

def _refine_simplex_el_group(grp, refine_flags, vertex_nr_base):
    from pytools import (
            generate_nonnegative_integer_tuples_summing_to_at_most as gnitstam)
    from modepy.tools import submesh

    node_tuples = gnitstam(2, grp.dim)
    template = submesh(node_tuples)

    nrefined_elements = np.sum(refine_flags != 0)

    old_vertex_tuples = (
            [(0,) * grp.dim]
            + [(0,)*i + (2,) + (0,)*(grp.dim-1-i) for i in range(grp.dim)])
    new_vertex_tuples = [
            vertex_tuple
            for vertex_tuple in node_tuples
            if vertex_tuple not in old_vertex_tuples]

    new_unit_nodes = np.array(new_vertex_tuples, dtype=grp.nodes.dtype) - 1

    nnew_vertices = nrefined_elements * len(new_vertex_tuples)
    ambient_dim = grp.nodes.shape[0]
    new_vertices = np.empty((nnew_vertices, ambient_dim))

    nnew_elements = (len(template) - 1) * nrefined_elements
    new_vertex_indices = np.empty(
            (nnew_elements, grp.dim+1),
            dtype=grp.vertex_indices.dtype)

    new_nodes = np.empty(
            (ambient_dim, nnew_elements, grp.nunit_nodes),
            dtype=grp.nodes.dtype)

    inew_vertex = vertex_nr_base
    inew_el = 0
    for iel in xrange(grp.nelements):
        if not refine_flags[iel]:
            new_nodes[:, inew_el, :] = grp.nodes[:, iel, :]

            # old vertices always keep their numbers
            new_vertex_indices[inew_el, :] = grp.vertex_indices[iel, :]
            continue

        el_vertex_indices = grp.vertex_indices[iel, :]












    from meshmode.mesh import SimplexElementGroup
    new_grp = SimplexElementGroup(grp.order, new_vertex_indices,
            new_nodes, unit_nodes=grp.unit_nodes)

    return new_vertices, new_grp


def refine_without_connectivity(mesh, refine_flags):
    vertex_chunks = [mesh.vertices]
    nvertices = mesh.vertices.shape[1]

    groups = []

    from meshmode.mesh import SimplexElementGroup, Mesh
    for grp in mesh.groups:
        if isinstance(grp, SimplexElementGroup):
            enb = grp.element_nr_base
            added_vertices, new_grp = _refine_simplex_el_group(
                    grp, refine_flags[enb:enb+grp.nelements], nvertices)

            vertex_chunks.append(added_vertices)
            groups.append(new_grp)
        else:
            raise NotImplementedError("refining element group of type %s"
                    % type(grp).__name__)

    vertices = np.hstack(vertex_chunks)

    if all(grp.dim == 1 for grp in groups):
        # For meshes that are exclusively 1D, deducing connectivity from vertex
        # indices is still OK. For everyone else, not really.

        connectivity = None
    else:
        connectivity = False

    return Mesh(vertices, groups, element_connectivity=connectivity)

# }}}


class _SplitFaceRecord(object):
    """
    .. attribute:: neighboring_elements
    .. attribute:: new_vertex_nr

        integer or None
    """


class Refiner(object):
    def __init__(self, mesh):
        self.last_mesh = mesh

        # Indices in last_mesh that were split in the last round of
        # refinement. Only these elements may be split this time
        # around.
        self.last_split_elements = None

    def get_refine_base_index(self):
        if self.last_split_elements is None:
            return 0
        else:
            return self.last_mesh.nelements - len(self.last_split_elements)

    def get_empty_refine_flags(self):
        return np.zeros(
                self.last_mesh.nelements - self.get_refine_base_index(),
                np.bool)

    def refine(self, refine_flags):
        """
        :return: a refined mesh
        """

        # multi-level mapping:
        # {
        #   dimension of intersecting entity (0=vertex,1=edge,2=face,...)
        #   :
        #   { frozenset of end vertices : _SplitFaceRecord }
        # }
        split_faces = {}

        ibase = self.get_refine_base_index()
        affected_group_indices = set()

        for grp in self.last_mesh.groups:
            iel_base











# vim: foldmethod=marker
