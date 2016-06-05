from __future__ import division
from __future__ import absolute_import
from six.moves import range

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

import numpy as np
from pytools.spatial_btree import SpatialBinaryTreeBucket


# {{{ make_element_lookup_tree

def make_element_lookup_tree(mesh, eps=1e-12):
    from meshmode.mesh.processing import find_bounding_box
    bbox_min, bbox_max = find_bounding_box(mesh)
    bbox_min -= eps
    bbox_max += eps

    tree = SpatialBinaryTreeBucket(bbox_min, bbox_max)

    for igrp, grp in enumerate(mesh.groups):
        for iel_grp in range(grp.nelements):
            el_vertices = mesh.vertices[:, grp.vertex_indices[iel_grp]]

            el_bbox_min = np.min(el_vertices, axis=-1) - eps
            el_bbox_max = np.max(el_vertices, axis=-1) + eps

            tree.insert((igrp, iel_grp), (el_bbox_min, el_bbox_max))

    return tree

# }}}


# {{{ nd_quad_submesh

def nd_quad_submesh(node_tuples):
    """Return a list of tuples of indices into the node list that
    generate a tesselation of the reference element.

    :arg node_tuples: A list of tuples *(i, j, ...)* of integers
        indicating node positions inside the unit element. The
        returned list references indices in this list.

        :func:`pytools.generate_nonnegative_integer_tuples_below`
        may be used to generate *node_tuples*.

    See also :func:`modepy.tools.simplex_submesh`.
    """

    from pytools import single_valued, add_tuples
    dims = single_valued(len(nt) for nt in node_tuples)

    node_dict = dict(
            (ituple, idx)
            for idx, ituple in enumerate(node_tuples))

    from pytools import generate_nonnegative_integer_tuples_below as gnitb

    result = []
    for current in node_tuples:
        try:
            result.append(tuple(
                    node_dict[add_tuples(current, offset)]
                    for offset in gnitb(2, dims)))

        except KeyError:
            pass

    return result

# }}}


# vim: foldmethod=marker
