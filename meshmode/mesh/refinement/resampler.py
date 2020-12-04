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

from functools import singledispatch

import numpy as np
import modepy as mp
from pytools import RecordWithoutPickling

import logging
logger = logging.getLogger(__name__)


# {{{ resampling simplex points for refinement

class SimplexResampler:
    @staticmethod
    def get_vertex_pair_to_midpoint_order(dim):
        return get_vertex_pair_to_midpoint_order(mp.Simplex(dim))

    @staticmethod
    def get_midpoints(group, tesselation, elements):
        return get_midpoints(mp.Simplex(group.dim),
                group, tesselation, elements)

    @staticmethod
    def get_tesselated_nodes(group, tesselation, elements):
        return get_tesselated_nodes(mp.Simplex(group.dim),
                group, tesselation, elements)

# }}}


# {{{ interface

# NOTE: internal to refiners: do not make documentation public.

class TesselationInfo(RecordWithoutPickling):
    """
    .. attribute:: ref_vertices
    .. attribute:: children
    """

@singledispatch
def map_unit_nodes_to_children(shape: mp.Shape, tesselation, unit_nodes):
    raise NotImplementedError


@singledispatch
def get_ref_midpoints(shape: mp.Shape, ref_vertices):
    from pytools import add_tuples
    space = mp.space_for_shape(shape, 1)
    orig_vertices = [
            add_tuples(vt, vt) for vt in mp.node_tuples_for_space(space)
            ]
    return [rv for rv in ref_vertices if rv not in orig_vertices]


@singledispatch
def get_midpoints(shape: mp.Shape, group, tesselation, elements):
    """Compute the midpoints of the vertices of the specified elements.

    :arg group: an instance of :class:`meshmode.mesh.MeshElementGroup`.
    :arg tesselation: a :class:`TesselationInfo`.
    :arg elements: a list of (group-relative) element numbers.

    :return: A :class:`dict` mapping element numbers to midpoint
        coordinates, with each value in the map having shape
        ``(ambient_dim, nmidpoints)``. The ordering of the midpoints
        follows their ordering in the tesselation (see also
        :meth:`SimplexResampler.get_vertex_pair_to_midpoint_order`)
    """
    if shape.dim != group.dim:
        raise ValueError("shape and group dimension do not match")

    if group.vertex_indices is not None:
        assert len(group.vertex_indices[0]) == shape.nvertices

    # get midpoints, converted to unit coordinates.
    midpoints = -1 + np.array(get_ref_midpoints(shape, tesselation.ref_vertices))

    space = mp.space_for_shape(shape, group.order)
    resampling_mat = mp.resampling_matrix(
            mp.basis_for_space(space, shape).functions,
            midpoints.T,
            group.unit_nodes)

    resampled_midpoints = np.einsum("mu,deu->edm",
            resampling_mat, group.nodes[:, elements])

    return dict(zip(elements, resampled_midpoints))


@singledispatch
def get_tesselated_nodes(shape: mp.Shape, group, tesselation, elements):
    """Compute the nodes of the child elements according to the tesselation.

    :arg group: An instance of :class:`meshmode.mesh.MeshElementGroup`.
    :arg tesselation: a :class:`TesselationInfo`.
    :arg elements: A list of (group-relative) element numbers.

    :return: A :class:`dict` mapping element numbers to node
        coordinates, with each value in the map having shape
        ``(ambient_dim, nchildren, nunit_nodes)``.
        The ordering of the child nodes follows the ordering
        of ``tesselation.children.``
    """
    if shape.dim != group.dim:
        raise ValueError("shape and group dimension do not match")

    if group.vertex_indices is not None:
        assert len(group.vertex_indices[0]) == shape.nvertices

    # get child unit node coordinates.
    child_unit_nodes = np.hstack(list(
        map_unit_nodes_to_children(shape, tesselation, group.unit_nodes)
        ))

    space = mp.space_for_shape(shape, group.order)
    resampling_mat = mp.resampling_matrix(
            mp.basis_for_space(space, shape).functions,
            child_unit_nodes,
            group.unit_nodes)

    resampled_unit_nodes = np.einsum("cu,deu->edc",
            resampling_mat, group.nodes[:, elements])

    ambient_dim = len(group.nodes)
    nunit_nodes = len(group.unit_nodes[0])

    return {
            el: resampled_unit_nodes[iel].reshape((ambient_dim, -1, nunit_nodes))
            for iel, el in enumerate(elements)
            }
    raise NotImplementedError(type(shape).__name__)

# }}}


# {{{ simplex

@map_unit_nodes_to_children.register(mp.Simplex)
def _(shape: mp.Simplex, tesselation, unit_nodes):
    ref_vertices = np.array(tesselation.ref_vertices, dtype=np.float)
    assert len(unit_nodes.shape) == 2

    for child_element in tesselation.children:
        origin = np.vstack(ref_vertices[child_element[0]])
        basis = ref_vertices.T[:, child_element[1:]] - origin

        yield basis.dot((unit_nodes + 1) / 2) + origin - 1

# }}}


# {{{ hypercube

@map_unit_nodes_to_children.register(mp.Hypercube)
def _(shape: mp.Hypercube, tesselation, unit_nodes):
    ref_vertices = np.array(tesselation.ref_vertices, dtype=np.float)
    assert len(unit_nodes.shape) == 2

    basis_indices = [1, 2, 4][:shape.dim]
    for child_element in tesselation.children:
        origin = np.vstack(ref_vertices[child_element[0]])
        basis = ref_vertices.T[:, np.array(child_element)[indices]] - origin

        yield basis.dot((unit_nodes + 1) / 2) + origin - 1

# }}}

# vim: foldmethod=marker
