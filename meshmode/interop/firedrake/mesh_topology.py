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

from warnings import warn  # noqa
import numpy as np

from meshmode.interop import ExternalImportHandler


# {{{ ImportHandler for firedrake's MeshTopology class

class FiredrakeMeshTopologyImporter(ExternalImportHandler):
    """
    An Importer for :class:`firedrake.mesh.MeshTopology`.
    Holds the topological (as opposed to geometric) information
    about a mesh.
    """

    def __init__(self, mesh, cells_to_use=None):
        """
        :param mesh: An instance :mod:`firedrake` :class:`MeshTopology` or
                   :class:`MeshGeometry`. If an instance of
                   :class:`MeshGeometry`, uses its underlying topology.
        :param cells_to_use: Either

            * *None*, in which case this argument is ignored
            * An array of cell ids, in which case those are the
              only cells for which information is gathered/converted

        We require that :param:`mesh` have co-dimesnion
        of 0 or 1.
        Moreover, if :param:`mesh` is a 2-surface embedded in 3-space,
        we _require_ that :function:`init_cell_orientations`
        has been called already.

        :raises TypeError: If :param:`mesh` is not of :mod:`firedrake`
                           :class:`MeshTopology` or :class:`MeshGeometry`
        """
        top = mesh.topological  # convert geometric to topological

        # {{{ Check input types
        from firedrake.mesh import MeshTopology
        if not isinstance(top, MeshTopology):
            raise TypeError(":param:`mesh` must be of type "
                            ":class:`firedrake.mesh.MeshTopology` or "
                            ":class:`firedrake.mesh.MeshGeometry`")
        # }}}

        super(FiredrakeMeshTopologyImporter, self).__init__(top)

        # Ensure has simplex-type elements
        if not top.ufl_cell().is_simplex():
            raise ValueError(":param:`mesh` must have simplex type elements, "
                             "%s is not a simplex" % (mesh.ufl_cell()))

        # Ensure dimensions are in appropriate ranges
        supported_dims = [1, 2, 3]
        if self.cell_dimension() not in supported_dims:
            raise ValueError("Cell dimension is %s. Cell dimension must be one of"
                             " %s" % (self.cell_dimension(), supported_dims))

        self._nodal_adjacency = None
        self.icell_to_fd = cells_to_use  # Map cell index -> fd cell index
        self.fd_to_icell = None          # Map fd cell index -> cell index

        # perform checks on :param:`cells_to_use` if not *None*
        if self.icell_to_fd is not None:
            assert np.unique(self.icell_to_fd).shape == self.icell_to_fd.shape
            self.fd_to_icell = dict(zip(self.icell_to_fd,
                                        np.arange(self.icell_to_fd.shape[0],
                                                  dtype=np.int32)
                                        ))

    @property
    def topology_importer(self):
        """
        A reference to *self*, for compatability with mesh geometry
        importers
        """
        return self

    @property
    def topological_importer(self):
        """
        A reference to *self*, for compatability with mesh geometry
        importers
        """
        return self

    def cell_dimension(self):
        """
        Return the dimension of the cells used by this topology
        """
        return self.data.cell_dimension()

    def nelements(self):
        """
        Return the number of cells in this topology
        """
        if self.icell_to_fd is None:
            num_cells = self.data.num_cells()
        else:
            num_cells = self.icell_to_fd.shape[0]

        return num_cells

    def nunit_vertices(self):
        """
        Return the number of unit vertices on the reference element
        """
        return self.data.ufl_cell().num_vertices()

    def bdy_tags(self):
        """
        Return a tuple of bdy tags as requested in
        the construction of a :mod:`meshmode` :class:`Mesh`

        The tags used are :class:`meshmode.mesh.BTAG_ALL`,
        :class:`meshmode.mesh.BTAG_REALLY_ALL`, and
        any markers in the mesh topology's exterior facets
        (see :attr:`firedrake.mesh.MeshTopology.exterior_facets.unique_markers`)
        """
        from meshmode.mesh import BTAG_ALL, BTAG_REALLY_ALL
        bdy_tags = [BTAG_ALL, BTAG_REALLY_ALL]

        unique_markers = self.data.exterior_facets.unique_markers
        if unique_markers is not None:
            bdy_tags += list(unique_markers)

        return tuple(bdy_tags)

    def nodal_adjacency(self):
        """
        Returns a :class:`meshmode.mesh.NodalAdjacency` object
        representing the nodal adjacency of this mesh
        """
        if self._nodal_adjacency is None:
            # TODO... not sure how to get around the private access
            plex = self.data._plex

            # dmplex cell Start/end and vertex Start/end.
            c_start, c_end = plex.getHeightStratum(0)
            v_start, v_end = plex.getDepthStratum(0)

            # TODO... not sure how to get around the private access
            to_fd_id = np.vectorize(self.data._cell_numbering.getOffset)(
                np.arange(c_start, c_end, dtype=np.int32))

            element_to_neighbors = {}
            verts_checked = set()  # dmplex ids of vertex checked

            # If using all cells, loop over them all
            if self.icell_to_fd is None:
                range_ = range(c_start, c_end)
            # Otherwise, just the ones you're using
            else:
                isin = np.isin(to_fd_id, self.icell_to_fd)
                range_ = np.arange(c_start, c_end, dtype=np.int32)[isin]

            # For each cell
            for cell_id in range_:
                # For each vertex touching the cell (that haven't already seen)
                for vert_id in plex.getTransitiveClosure(cell_id)[0]:
                    if v_start <= vert_id < v_end and vert_id not in verts_checked:
                        verts_checked.add(vert_id)
                        cells = []
                        # Record all cells touching that vertex
                        support = plex.getTransitiveClosure(vert_id,
                                                            useCone=False)[0]
                        for other_cell_id in support:
                            if c_start <= other_cell_id < c_end:
                                cells.append(to_fd_id[other_cell_id - c_start])

                        # If only using some cells, clean out extraneous ones
                        # and relabel them to new id
                        cells = set(cells)
                        if self.fd_to_icell is not None:
                            cells = set([self.fd_to_icell[fd_ndx]
                                         for fd_ndx in cells
                                         if fd_ndx in self.fd_to_icell])

                        # mark cells as neighbors
                        for cell_one in cells:
                            element_to_neighbors.setdefault(cell_one, set())
                            element_to_neighbors[cell_one] |= cells

            # Create neighbors_starts and neighbors
            neighbors = []
            neighbors_starts = np.zeros(self.nelements() + 1, dtype=np.int32)
            for iel in range(len(element_to_neighbors)):
                elt_neighbors = element_to_neighbors[iel]
                neighbors += list(elt_neighbors)
                neighbors_starts[iel+1] = len(neighbors)

            neighbors = np.array(neighbors, dtype=np.int32)

            from meshmode.mesh import NodalAdjacency
            self._nodal_adjacency = NodalAdjacency(neighbors_starts=neighbors_starts,
                                                   neighbors=neighbors)
        return self._nodal_adjacency

# }}}
