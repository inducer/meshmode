__copyright__ = "Copyright (C) 2020 Benjamin Sepanski"

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

__doc__ = """
.. autofunction:: get_firedrake_nodal_adjacency_group
.. autofunction:: get_firedrake_boundary_tags
"""

from warnings import warn  # noqa
import numpy as np


# {{{ functions to extract information from Mesh Topology

def get_firedrake_nodal_adjacency_group(fdrake_mesh, cells_to_use=None):
    """
    Create a nodal adjacency object
    representing the nodal adjacency of :param:`fdrake_mesh`

    :param fdrake_mesh: A firedrake mesh (:class:`MeshTopology`
        or :class:`MeshGeometry`)
    :param cells_to_use: Either

            * *None*, in which case this argument is is ignored
            * A numpy array of firedrake cell indices, in which case
              any cell with indices not in the array :param:`cells_to_use`
              is ignored.
              This induces a new order on the cells.
              The *i*th element in the returned :class:`NodalAdjacency`
              object corresponds to the ``cells_to_use[i]``th cell
              in the firedrake mesh.

        This feature has been used when only part of the mesh
        needs to be converted, since firedrake has no concept
        of a "sub-mesh".

    :return: A :class:`meshmode.mesh.NodalAdjacency` instance
        representing the nodal adjacency of :param:`fdrake_mesh`
    """
    mesh_topology = fdrake_mesh.topology
    # TODO... not sure how to get around the private access
    plex = mesh_topology._plex

    # dmplex cell Start/end and vertex Start/end.
    c_start, c_end = plex.getHeightStratum(0)
    v_start, v_end = plex.getDepthStratum(0)

    # TODO... not sure how to get around the private access
    # This maps the dmplex index of a cell to its firedrake index
    to_fd_id = np.vectorize(mesh_topology._cell_numbering.getOffset)(
        np.arange(c_start, c_end, dtype=np.int32))

    element_to_neighbors = {}
    verts_checked = set()  # dmplex ids of vertex checked

    # If using all cells, loop over them all
    if cells_to_use is None:
        range_ = range(c_start, c_end)
    # Otherwise, just the ones you're using
    else:
        assert isinstance(cells_to_use, np.ndarray)
        assert np.size(cells_to_use) == np.size(np.unique(cells_to_use)), \
            "cells_to_use must have unique values"
        assert len(np.shape(cells_to_use)) == 1 and len(cells_to_use) > 0
        isin = np.isin(to_fd_id, cells_to_use)
        range_ = np.arange(c_start, c_end, dtype=np.int32)[isin]

    # For each cell
    for cell_id in range_:
        # For each vertex touching the cell (that haven't already seen)
        for vert_id in plex.getTransitiveClosure(cell_id)[0]:
            if v_start <= vert_id < v_end and vert_id not in verts_checked:
                verts_checked.add(vert_id)
                cells = []
                # Record all cells touching that vertex
                support = plex.getTransitiveClosure(vert_id, useCone=False)[0]
                for other_cell_id in support:
                    if c_start <= other_cell_id < c_end:
                        cells.append(to_fd_id[other_cell_id - c_start])

                # If only using some cells, clean out extraneous ones
                # and relabel them to new id
                cells = set(cells)
                if cells_to_use is not None:
                    cells = set([cells_to_use[fd_ndx] for fd_ndx in cells
                                 if fd_ndx in cells_to_use])

                # mark cells as neighbors
                for cell_one in cells:
                    element_to_neighbors.setdefault(cell_one, set())
                    element_to_neighbors[cell_one] |= cells

    # Count the number of cells
    if cells_to_use is None:
        nelements = mesh_topology.num_cells()
    else:
        nelements = cells_to_use.shape[0]

    # Create neighbors_starts and neighbors
    neighbors = []
    # FIXME : Is this the right integer type to choose?
    neighbors_starts = np.zeros(nelements + 1, dtype=np.int32)
    for iel in range(len(element_to_neighbors)):
        elt_neighbors = element_to_neighbors[iel]
        neighbors += list(elt_neighbors)
        neighbors_starts[iel+1] = len(neighbors)

    neighbors = np.array(neighbors, dtype=np.int32)

    from meshmode.mesh import NodalAdjacency
    return NodalAdjacency(neighbors_starts=neighbors_starts,
                          neighbors=neighbors)


def get_firedrake_boundary_tags(fdrake_mesh):
    """
    Return a tuple of bdy tags as requested in
    the construction of a :mod:`meshmode` :class:`Mesh`

    The tags used are :class:`meshmode.mesh.BTAG_ALL`,
    :class:`meshmode.mesh.BTAG_REALLY_ALL`, and
    any markers in the mesh topology's exterior facets
    (see :attr:`firedrake.mesh.MeshTopology.exterior_facets.unique_markers`)

    :param fdrake_mesh: A :mod:`firedrake` :class:`MeshTopology` or
        :class:`MeshGeometry`

    :return: A tuple of boundary tags
    """
    from meshmode.mesh import BTAG_ALL, BTAG_REALLY_ALL
    bdy_tags = [BTAG_ALL, BTAG_REALLY_ALL]

    unique_markers = fdrake_mesh.topology.exterior_facets.unique_markers
    if unique_markers is not None:
        bdy_tags += list(unique_markers)

    return tuple(bdy_tags)

# }}}
