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
from meshmode.mesh import MeshElementGroup, _ModepyElementGroup

import logging
logger = logging.getLogger(__name__)

from typing import List, Tuple, Optional


# {{{ interface

@dataclass(frozen=True)
class ElementTesselationInfo:
    """Describes how one element is split into multiple child elements.

    .. attribute:: children

        An array of shape ``(nchildren, nvertices)`` containing the vertices
        of each child element the reference element was split into.

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
    .. attribute:: el_tess_info

        An instance of :class:`ElementTesselationInfo` that describes the
        tesselation of a single element into multiple child elements.

    .. attribute:: element_mapping

        A mapping from the original elements to the refined child elements.
    """

    el_tess_info: ElementTesselationInfo
    # FIXME: This should really be a CSR data structure.
    element_mapping: List[List[int]]


@singledispatch
def get_group_midpoints(meg: MeshElementGroup, el_tess_info, elements):
    """Compute the midpoints of the vertices of the specified elements.

    :arg group: an instance of :class:`meshmode.mesh.MeshElementGroup`.
    :arg el_tess_info: a :class:`ElementTesselationInfo`.
    :arg elements: a list of (group-relative) element numbers.

    :return: A :class:`dict` mapping element numbers to midpoint
        coordinates, with each value in the map having shape
        ``(ambient_dim, nmidpoints)``. The ordering of the midpoints
        follows their ordering in the tesselation.
    """
    raise NotImplementedError(type(meg).__name__)


@singledispatch
def get_group_tesselated_nodes(meg: MeshElementGroup, el_tess_info, elements):
    """Compute the nodes of the child elements according to the tesselation.

    :arg group: An instance of :class:`meshmode.mesh.MeshElementGroup`.
    :arg el_tess_info: a :class:`ElementTesselationInfo`.
    :arg elements: A list of (group-relative) element numbers.

    :return: A :class:`dict` mapping element numbers to node
        coordinates, with each value in the map having shape
        ``(ambient_dim, nchildren, nunit_nodes)``.
        The ordering of the child nodes follows the ordering
        of ``el_tess_info.children.``
    """
    raise NotImplementedError(type(meg).__name__)


@singledispatch
def get_group_tesselation_info(meg: MeshElementGroup):
    """
    :returns: a :class:`ElementTesselationInfo` for the element group *meg*.
    """
    raise NotImplementedError(type(meg).__name__)

# }}}


# {{{ helpers

def _midpoint_tuples(a, b):
    def midpoint(x, y):
        d, r = divmod(x + y, 2)
        if r:
            raise ValueError("%s is not evenly divisible by two" % x)

        return d

    return tuple(midpoint(ai, bi) for ai, bi in zip(a, b))


def _get_ref_midpoints(shape, ref_vertices):
    r"""The reference element is considered to be, e.g. for a 2-simplex::

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

# }}}


# {{{ modepy.shape tesselation and resampling

@get_group_midpoints.register(_ModepyElementGroup)
def _(meg: _ModepyElementGroup, el_tess_info, elements):
    shape = meg._modepy_shape
    space = meg._modepy_space

    # get midpoints in reference coordinates
    midpoints = -1 + np.array(_get_ref_midpoints(shape, el_tess_info.ref_vertices))

    # resample midpoints to ambient coordinates
    resampling_mat = mp.resampling_matrix(
            mp.basis_for_space(space, shape).functions,
            midpoints.T,
            meg.unit_nodes)

    resampled_midpoints = np.einsum("mu,deu->edm",
            resampling_mat, meg.nodes[:, elements])

    return dict(zip(elements, resampled_midpoints))


@get_group_tesselated_nodes.register(_ModepyElementGroup)
def _(meg: _ModepyElementGroup, el_tess_info, elements):
    shape = meg._modepy_shape
    space = meg._modepy_space

    # get child unit node coordinates.
    from meshmode.mesh.refinement.utils import map_unit_nodes_to_children
    child_unit_nodes = np.hstack(list(
        map_unit_nodes_to_children(meg, meg.unit_nodes, el_tess_info)
        ))

    # resample child nodes to ambient coordinates
    resampling_mat = mp.resampling_matrix(
            mp.basis_for_space(space, shape).functions,
            child_unit_nodes,
            meg.unit_nodes)

    resampled_unit_nodes = np.einsum("cu,deu->edc",
            resampling_mat, meg.nodes[:, elements])

    ambient_dim = len(meg.nodes)
    nunit_nodes = len(meg.unit_nodes[0])

    return {
            el: resampled_unit_nodes[iel].reshape((ambient_dim, -1, nunit_nodes))
            for iel, el in enumerate(elements)
            }


@get_group_tesselation_info.register(_ModepyElementGroup)
def _(meg: _ModepyElementGroup):
    shape = meg._modepy_shape
    space = type(meg._modepy_space)(meg.dim, 2)

    ref_vertices = mp.node_tuples_for_space(space)
    ref_vertices_to_index = {rv: i for i, rv in enumerate(ref_vertices)}

    from pytools import add_tuples
    space = type(meg._modepy_space)(meg.dim, 1)
    orig_vertices = tuple([
        add_tuples(vt, vt) for vt in mp.node_tuples_for_space(space)
        ])
    orig_vertex_indices = [ref_vertices_to_index[vt] for vt in orig_vertices]

    midpoints = _get_ref_midpoints(shape, ref_vertices)
    midpoint_indices = [ref_vertices_to_index[mp] for mp in midpoints]

    midpoint_to_vertex_pairs = {
            midpoint: (i, j)
            for i, ivt in enumerate(orig_vertices)
            for j, jvt in enumerate(orig_vertices)
            for midpoint in [_midpoint_tuples(ivt, jvt)]
            if i < j and midpoint in midpoints
            }
    # ensure order matches the one in midpoint_indices
    midpoint_vertex_pairs = [midpoint_to_vertex_pairs[m] for m in midpoints]

    return ElementTesselationInfo(
            ref_vertices=ref_vertices,
            children=np.array(mp.submesh_for_shape(shape, ref_vertices)),
            orig_vertex_indices=np.array(orig_vertex_indices),
            midpoint_indices=np.array(midpoint_indices),
            midpoint_vertex_pairs=midpoint_vertex_pairs,
            )

# }}}

# vim: foldmethod=marker
