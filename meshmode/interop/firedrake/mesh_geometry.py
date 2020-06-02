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
from collections import defaultdict
import numpy as np

from meshmode.interop import ExternalImportHandler
from meshmode.interop.firedrake.function_space_coordless import \
    FiredrakeCoordinatelessFunctionImporter


class FiredrakeMeshGeometryImporter(ExternalImportHandler):
    """
        This takes a :mod:`firedrake` :class:`MeshGeometry`
        and converts its data so that :mod:`meshmode` can handle it.

        .. attribute:: data

            A :mod:`firedrake` :class:`MeshGeometry` instance
    """

    def __init__(self,
                 mesh,
                 coordinates_importer,
                 normals=None,
                 no_normals_warn=True):
        """
            :param mesh: A :mod:`firedrake` :class:`MeshGeometry`.
                We require that :aram:`mesh` have co-dimesnion
                of 0 or 1.
                Moreover, if :param:`mesh` is a 2-surface embedded in 3-space,
                we _require_ that :function:`init_cell_orientations`
                has been called already.

            :param coordinates_importer: An instance of
                class :class:`FiredrakeCoordinatelessFunctionImporter`
                to use, mapping the points of the topological mesh
                to their coordinates (The function is coordinateless
                in the sense that its *domain* has no coordinates)

            For other params see :meth:`orientations`
        """
        super(FiredrakeMeshGeometryImporter, self).__init__(mesh)

        # {{{ Make sure input data is valid

        # Ensure is not a topological mesh
        if mesh.topological == mesh:
            raise TypeError(":param:`mesh` must be of type"
                            " :class:`firedrake.mesh.MeshGeometry`")

        # Ensure dimensions are in appropriate ranges
        supported_dims = [1, 2, 3]
        if mesh.geometric_dimension() not in supported_dims:
            raise ValueError("Geometric dimension is %s. Geometric "
                             " dimension must be one of range %s"
                             % (mesh.geometric_dimension(), supported_dims))

        # Raise warning if co-dimension is not 0 or 1
        co_dimension = mesh.geometric_dimension() - mesh.topological_dimension()
        if co_dimension not in [0, 1]:
            raise ValueError("Codimension is %s, but must be 0 or 1." %
                             (co_dimension))

        # Ensure coordinates are coordinateless
        if not isinstance(coordinates_importer,
                          FiredrakeCoordinatelessFunctionImporter):
            raise ValueError(":param:`coordinates_importer` must be of type"
                             " FiredrakeCoordinatelessFunctionImporter")

        fspace_importer = coordinates_importer.function_space_importer()
        topology_importer = fspace_importer.mesh_importer()

        if topology_importer.data != mesh.topology:
            raise ValueError("Topology :param:`coordinates` lives on must be "
                             "the same "
                             "topology that :param:`mesh` lives on")

        # }}}

        # For sharing data like in firedrake
        self._shared_data_cache = defaultdict(dict)

        # Store input information
        self._coordinates_importer = coordinates_importer
        self._topology_importer = topology_importer

        self._normals = normals
        self._no_normals_warn = no_normals_warn

        # To be computed later
        self._vertex_indices = None
        self._vertices = None
        self._nodes = None
        self._group = None
        self._orient = None
        self._facial_adjacency_groups = None
        self._meshmode_mesh = None

        def callback(cl_ctx):
            """
            Finish initialization by creating a coordinates function importer
            for public access on this mesh which is not "coordinateless" (i.e.
            its domain has coordinates)
            """
            from meshmode.interop.firedrake.function import \
                FiredrakeFunctionImporter
            from meshmode.interop.firedrake.function_space import \
                FiredrakeWithGeometryImporter
            from firedrake import Function

            coordinates_fs = self.data.coordinates.function_space()
            coordinates_fs_importer = \
                self._coordinates_importer.function_space_importer()

            fspace_importer = \
                FiredrakeWithGeometryImporter(cl_ctx,
                                              coordinates_fs,
                                              coordinates_fs_importer,
                                              self)
            f = Function(fspace_importer.data, val=self._coordinates_importer.data)
            self._coordinates_function_importer = \
                FiredrakeFunctionImporter(f, fspace_importer)

            del self._callback

        self._callback = callback

    def initialized(self):
        return not hasattr(self, '_callback')

    def init(self, cl_ctx):
        if not self.initialized():
            self._callback(cl_ctx)

    def __getattr__(self, attr):
        """
        If can't find an attribute, look in the underlying
        topological mesh
        (Done like :class:`firedrake.function.MeshGeometry`)
        """
        return getattr(self._topology_importer, attr)

    @property
    def coordinates_importer(self):
        """
        Return coordinates as a function

        PRECONDITION: Have called :meth:`init`
        """
        try:
            return self._coordinates_function_importer
        except AttributeError:
            raise AttributeError("No coordinates function, have you finished"
                                 " initializing this object"
                                 " (i.e. have you called :meth:`init`)?")

    def _compute_vertex_indices_and_vertices(self):
        if self._vertex_indices is None:
            fspace_importer = self.coordinates_importer.function_space_importer()
            finat_element_importer = fspace_importer.finat_element_importer

            # Convert cell node list of mesh to vertex list
            unit_vertex_indices = finat_element_importer.unit_vertex_indices()
            cfspace = self.data.coordinates.function_space()
            if self.icell_to_fd is not None:
                cell_node_list = cfspace.cell_node_list[self.icell_to_fd]
            else:
                cell_node_list = cfspace.cell_node_list

            vertex_indices = cell_node_list[:, unit_vertex_indices]

            # Get maps newnumbering->old and old->new (new numbering comes
            #                                          from removing the non-vertex
            #                                          nodes)
            vert_ndx_to_fd_ndx = np.unique(vertex_indices.flatten())
            fd_ndx_to_vert_ndx = dict(zip(vert_ndx_to_fd_ndx,
                                          np.arange(vert_ndx_to_fd_ndx.shape[0],
                                                    dtype=np.int32)
                                          ))
            # Get vertices array
            vertices = np.real(
                self.data.coordinates.dat.data[vert_ndx_to_fd_ndx])

            #:mod:`meshmode` wants shape to be [ambient_dim][nvertices]
            if len(vertices.shape) == 1:
                # 1 dim case, (note we're about to transpose)
                vertices = vertices.reshape(vertices.shape[0], 1)
            vertices = vertices.T.copy()

            # Use new numbering on vertex indices
            vertex_indices = np.vectorize(fd_ndx_to_vert_ndx.get)(vertex_indices)

            # store vertex indices and vertices
            self._vertex_indices = vertex_indices
            self._vertices = vertices

    def vertex_indices(self):
        """
        A numpy array of shape *(nelements, nunitvertices)*
        holding the vertex indices associated to each element
        """
        self._compute_vertex_indices_and_vertices()
        return self._vertex_indices

    def vertices(self):
        """
        Return the mesh vertices as a numpy array of shape
        *(dim, nvertices)*
        """
        self._compute_vertex_indices_and_vertices()
        return self._vertices

    def nodes(self):
        """
        Return the mesh nodes as a numpy array of shape
        *(dim, nelements, nunitnodes)*
        """
        if self._nodes is None:
            coords = self.data.coordinates.dat.data
            cfspace = self.data.coordinates.function_space()

            if self.icell_to_fd is not None:
                cell_node_list = cfspace.cell_node_list[self.icell_to_fd]
            else:
                cell_node_list = cfspace.cell_node_list
            self._nodes = np.real(coords[cell_node_list])

            # reshape for 1D so that [nelements][nunit_nodes][dim]
            if len(self._nodes.shape) != 3:
                self._nodes = np.reshape(self._nodes, self._nodes.shape + (1,))

            # Change shape to [dim][nelements][nunit_nodes]
            self._nodes = np.transpose(self._nodes, (2, 0, 1))

        return self._nodes

    def group(self):
        """
        Return an instance of :class:meshmode.mesh.SimplexElementGroup`
        corresponding to the mesh :attr:`data`
        """
        if self._group is None:
            from meshmode.mesh import SimplexElementGroup
            from meshmode.mesh.processing import flip_simplex_element_group

            fspace_importer = self.coordinates_importer.function_space_importer()
            finat_element_importer = fspace_importer.finat_element_importer

            # IMPORTANT that set :attr:`_group` because
            # :meth:`orientations` may call :meth:`group`
            self._group = SimplexElementGroup(
                finat_element_importer.data.degree,
                self.vertex_indices(),
                self.nodes(),
                dim=self.cell_dimension(),
                unit_nodes=finat_element_importer.unit_nodes())

            self._group = flip_simplex_element_group(self.vertices(), self._group,
                                                     self.orientations() < 0)

        return self._group

    def orientations(self):
        """
            Return the orientations of the mesh elements:
            an array, the *i*th element is > 0 if the *ith* element
            is positively oriented, < 0 if negatively oriented

            :param normals: _Only_ used if :param:`mesh` is a 1-surface
                embedded in 2-space. In this case,
                - If *None* then
                  all elements are assumed to be positively oriented.
                - Else, should be a list/array whose *i*th entry
                  is the normal for the *i*th element (*i*th
                  in :param:`mesh`*.coordinate.function_space()*'s
                  :attribute:`cell_node_list`)

            :param no_normals_warn: If *True*, raises a warning
                if :param:`mesh` is a 1-surface embedded in 2-space
                and :param:`normals` is *None*.
        """
        if self._orient is None:
            # compute orientations
            tdim = self.data.topological_dimension()
            gdim = self.data.geometric_dimension()

            orient = None
            if gdim == tdim:
                # We use :mod:`meshmode` to check our orientations
                from meshmode.mesh.processing import \
                    find_volume_mesh_element_group_orientation

                orient = \
                    find_volume_mesh_element_group_orientation(self.vertices(),
                                                               self.group())

            if tdim == 1 and gdim == 2:
                # In this case we have a 1-surface embedded in 2-space
                orient = np.ones(self.nelements())
                if self._normals:
                    for i, (normal, vertices) in enumerate(zip(
                            np.array(self._normals), self.vertices())):
                        if np.cross(normal, vertices) < 0:
                            orient[i] = -1.0
                elif self._no_normals_warn:
                    warn("Assuming all elements are positively-oriented.")

            elif tdim == 2 and gdim == 3:
                # In this case we have a 2-surface embedded in 3-space
                orient = self.data.cell_orientations().dat.data
                r"""
                    Convert (0 \implies negative, 1 \implies positive) to
                    (-1 \implies negative, 1 \implies positive)
                """
                orient *= 2
                orient -= np.ones(orient.shape, dtype=orient.dtype)

            self._orient = orient
            #Make sure the mesh fell into one of the cases
            """
          NOTE : This should be guaranteed by previous checks,
                 but is here anyway in case of future development.
            """
            assert self._orient is not None

        return self._orient

    def face_vertex_indices_to_tags(self):
        """
        Returns a dict mapping frozensets
        of vertex indices (which identify faces) to a
        list of boundary tags
        """
        finat_element = self.data.coordinates.function_space().finat_element
        exterior_facets = self.data.exterior_facets

        # fvi_to_tags maps frozenset(vertex indices) to tags
        fvi_to_tags = {}
        # maps faces to local vertex indices
        connectivity = finat_element.cell.connectivity[(self.cell_dimension()-1, 0)]

        # Compatability for older versions of firedrake
        try:
            local_fac_number = exterior_facets.local_facet_number
        except AttributeError:
            local_fac_number = exterior_facets.local_facet_dat.data

        for i, (icells, ifacs) in enumerate(zip(exterior_facets.facet_cell,
                                                local_fac_number)):
            # Compatability for older versions of firedrake
            try:
                iter(ifacs)
            except TypeError:
                ifacs = [ifacs]

            for icell, ifac in zip(icells, ifacs):
                # If necessary, convert to new cell numbering
                if self.fd_to_icell is not None:
                    if icell not in self.fd_to_icell:
                        continue
                    else:
                        icell = self.fd_to_icell[icell]

                # record face vertex indices to tag map
                cell_vertices = self.vertex_indices()[icell]
                facet_indices = connectivity[ifac]
                fvi = frozenset(cell_vertices[list(facet_indices)])
                fvi_to_tags.setdefault(fvi, [])
                fvi_to_tags[fvi].append(exterior_facets.markers[i])

        # }}}

        return fvi_to_tags

    def facial_adjacency_groups(self):
        """
        Return a list of :mod:`meshmode` :class:`FacialAdjacencyGroups`
        as used in the construction of a :mod:`meshmode` :class:`Mesh`
        """
        # {{{ Compute facial adjacency groups if not already done

        if self._facial_adjacency_groups is None:
            from meshmode.mesh import _compute_facial_adjacency_from_vertices

            self._facial_adjacency_groups = _compute_facial_adjacency_from_vertices(
                [self.group()],
                self.bdy_tags(),
                np.int32, np.int8,
                face_vertex_indices_to_tags=self.face_vertex_indices_to_tags())

        # }}}

        return self._facial_adjacency_groups

    def meshmode_mesh(self):
        """
        PRECONDITION: Have called :meth:`init`
        """
        if self._meshmode_mesh is None:
            assert self.initialized(), \
                "Must call :meth:`init` before :meth:`meshmode_mesh`"

            from meshmode.mesh import Mesh
            self._meshmode_mesh = \
                Mesh(self.vertices(), [self.group()],
                     boundary_tags=self.bdy_tags(),
                     nodal_adjacency=self.nodal_adjacency(),
                     facial_adjacency_groups=self.facial_adjacency_groups())

        return self._meshmode_mesh
