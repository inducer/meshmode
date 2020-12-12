__copyright__ = """
Copyright (C) 2018 Andreas Kloeckner
Copyright (C) 2014-6 Shivam Gupta
Copyright (C) 2020 Alexandru Fikl
"""

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

from dataclasses import dataclass
from functools import singledispatch

import numpy as np
import modepy as mp

from pytools import memoize

import logging
logger = logging.getLogger(__name__)

from typing import List, Tuple, Optional


@dataclass(frozen=True)
class TesselationInfo:
    """
    .. attribute:: children

        A tesselation of the reference element, given here by
        :attr:`ref_vertices`.

    .. attribute:: ref_vertices

        A list of tuples (similar to :func:`modepy.node_tuples_for_space`)
        for the reference element containing midpoints. This is equivalent
        to a second-order equidistant element.

    .. attribute:: orig_vertex_indices

        Indices into :attr:`ref_vertices` that select only the vertices, i.e.
        without the midpoints.

    .. attribute:: midpoint_indices

        Indices into :attr:`ref_vertices` that select only the midpoints, i.e.
        without :attr:`orig_vertex_indices`.

    .. attribute:: midpoint_vertex_pairs

        A list of tuples ``(v1, v2)`` of indices into :attr:`orig_vertex_indices`
        that give for each midpoint the two vertices on the same line.
    """

    children: np.ndarray
    ref_vertices: List[Tuple[int, ...]]

    orig_vertex_indices: Optional[np.ndarray] = None
    midpoint_indices: Optional[np.ndarray] = None
    midpoint_vertex_pairs: Optional[List[Tuple[int, int]]] = None


@dataclass(frozen=True)
class GroupRefinementRecord:
    """
    .. attribute:: tesselation

        A :class:`TesselationInfo` that describes the tesselation of the
        element group.

    .. attribute:: element_mapping

        A mapping from the original elements to the refined child elements.
    """

    tesselation: TesselationInfo
    element_mapping: List[List[int]]


def midpoint_tuples(a, b):
    def midpoint(x, y):
        d, r = divmod(x + y, 2)
        if r:
            raise ValueError("%s is not evenly divisible by two" % x)

        return d

    return tuple(midpoint(ai, bi) for ai, bi in zip(a, b))


# {{{ shape-dependent tesselation helpers

@singledispatch
def get_child_basis_vertex_indices(shape: mp.Shape, child):
    assert len(child) == shape.nvertices
    return child[:shape.dim + 1]


@get_child_basis_vertex_indices.register(mp.Hypercube)
def _(shape: mp.Hypercube, child):
    assert len(child) == shape.nvertices

    # * for a cube, we reorder nodes such that the vectors
    #
    #       (0, 1), (0, 2) and (0, 4)
    #
    #   form an orthogonal basis, since (0, 3) is linearly dependent on 1 and 2.
    #
    # * lines and squares don't require any reordering

    return [child[i] for i in [0, 1, 2, 4][:shape.dim + 1]]

# }}}


# {{{ resampling

def get_ref_midpoints(shape, ref_vertices):
    r"""The reference element is considered to be, e.g. for a 2 simplex::

        F
        | \
        |   \
        D----E
        |   /| \
        | /  |   \
        A----B----C

    where the midpoints are ``(B, E, D)``. The same applies to other shapes
    and higher dimensions.

    :arg ref_vertices: a :class:`list` of node index :class:`tuple`\ s
        on :math:`[0, 2]^d`.
    """

    from pytools import add_tuples
    space = mp.space_for_shape(shape, 1)
    orig_vertices = [
            add_tuples(vt, vt) for vt in mp.node_tuples_for_space(space)
            ]
    return [rv for rv in ref_vertices if rv not in orig_vertices]


def map_unit_nodes_to_children(shape, unit_nodes, tesselation):
    """
    :arg unit_nodes: an :class:`~numpy.ndarray` of shape ``(dim, nnodes)``.
    :arg tesselation: a :class:`TesselationInfo`.
    """

    ref_vertices = np.array(tesselation.ref_vertices, dtype=np.float).T
    assert len(unit_nodes.shape) == 2

    for child_element in tesselation.children:
        indices = get_child_basis_vertex_indices(shape, child_element)

        origin = ref_vertices[:, indices[0]].reshape(-1, 1)
        basis = ref_vertices[:, indices[1:]] - origin

        # mapped nodes are on [0, 2], so we subtract 1 to get it to [-1, 1]
        yield basis.dot((unit_nodes + 1.0) / 2.0) + origin - 1.0


def get_group_midpoints(group, tesselation, elements):
    """Compute the midpoints of the vertices of the specified elements.

    :arg group: an instance of :class:`meshmode.mesh.MeshElementGroup`.
    :arg tesselation: a :class:`TesselationInfo`.
    :arg elements: a list of (group-relative) element numbers.

    :return: A :class:`dict` mapping element numbers to midpoint
        coordinates, with each value in the map having shape
        ``(ambient_dim, nmidpoints)``. The ordering of the midpoints
        follows their ordering in the tesselation.
    """
    from meshmode.mesh import _ModepyElementGroup
    if not isinstance(group, _ModepyElementGroup):
        raise TypeError(f"groups of type '{type(group.mesh_el_group).__name__}'"
                " are not supported.")

    shape = group._modepy_shape
    space = mp.space_for_shape(shape, group.order)

    # get midpoints in reference coordinates
    midpoints = -1 + np.array(get_ref_midpoints(shape, tesselation.ref_vertices))

    # resample midpoints to ambient coordinates
    resampling_mat = mp.resampling_matrix(
            mp.basis_for_space(space, shape).functions,
            midpoints.T,
            group.unit_nodes)

    resampled_midpoints = np.einsum("mu,deu->edm",
            resampling_mat, group.nodes[:, elements])

    return dict(zip(elements, resampled_midpoints))


def get_group_tesselated_nodes(group, tesselation, elements):
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
    from meshmode.mesh import _ModepyElementGroup
    if not isinstance(group, _ModepyElementGroup):
        raise TypeError(f"groups of type '{type(group.mesh_el_group).__name__}'"
                " are not supported.")

    shape = group._modepy_shape
    space = mp.space_for_shape(shape, group.order)

    # get child unit node coordinates.
    child_unit_nodes = np.hstack(list(
        map_unit_nodes_to_children(shape, group.unit_nodes, tesselation)
        ))

    # resample child nodes to ambient coordinates
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

# }}}


# {{{ tesselation

def get_shape_tesselation_info(shape):
    space = mp.space_for_shape(shape, 2)
    ref_vertices = mp.node_tuples_for_space(space)
    ref_vertices_to_index = {rv: i for i, rv in enumerate(ref_vertices)}

    from pytools import add_tuples
    space = mp.space_for_shape(shape, 1)
    orig_vertices = tuple([
        add_tuples(vt, vt) for vt in mp.node_tuples_for_space(space)
        ])
    orig_vertex_indices = [ref_vertices_to_index[vt] for vt in orig_vertices]

    midpoints = get_ref_midpoints(shape, ref_vertices)
    midpoint_indices = [ref_vertices_to_index[mp] for mp in midpoints]

    midpoint_to_vertex_pairs = {
            midpoint: (i, j)
            for i, ivt in enumerate(orig_vertices)
            for j, jvt in enumerate(orig_vertices)
            for midpoint in [midpoint_tuples(ivt, jvt)]
            if i < j and midpoint in midpoints
            }
    # ensure order matches the one in midpoint_indices
    midpoint_vertex_pairs = [midpoint_to_vertex_pairs[m] for m in midpoints]

    return TesselationInfo(
            ref_vertices=ref_vertices,
            children=np.array(mp.submesh_for_shape(shape, ref_vertices)),
            orig_vertex_indices=np.array(orig_vertex_indices),
            midpoint_indices=np.array(midpoint_indices),
            midpoint_vertex_pairs=midpoint_vertex_pairs,
            )


@memoize
def get_group_tesselation_info(group_type, dim):
    from meshmode.mesh import _ModepyElementGroup
    if issubclass(group_type, _ModepyElementGroup):
        shape = group_type._modepy_shape_cls(dim)
        return get_shape_tesselation_info(shape)
    else:
        raise NotImplementedError(
                "bisection for element groups of type {group_type.__name__}")

# }}}
