from warnings import warn
import numpy as np

from meshmode.interop import ExternalImportHandler
from meshmode.interop.firedrake.mesh_geometry import \
    FiredrakeMeshGeometryImporter
from meshmode.interop.firedrake.function_space_coordless import \
    FiredrakeFunctionSpaceImporter

from meshmode.interop.firedrake.function_space_shared_data import \
    FiredrakeFunctionSpaceDataImporter


class FiredrakeWithGeometryImporter(ExternalImportHandler):
    def __init__(self,
                 cl_ctx,
                 function_space,
                 function_space_importer,
                 mesh_importer):
        """
        :param cl_ctx: A pyopencl context

        :param function_space: A firedrake :class:`WithGeometry`

        :param function_space_importer: An instance of class
            :class:`FiredrakeFunctionSpaceImporter` for the
            underlying topological function space of :param`function_space`

        :param mesh_importer: An instance of class
            :class:`FiredrakeMeshGeometryImporter` for the mesh
            that :param:`function_space` is built on
        """
        # FIXME use on bdy
        # {{{ Check input
        from firedrake.functionspaceimpl import WithGeometry
        if not isinstance(function_space, WithGeometry):
            raise TypeError(":arg:`function_space` must be of type"
                            " :class:`firedrake.functionspaceimpl.WithGeometry")

        if not isinstance(function_space_importer,
                          FiredrakeFunctionSpaceImporter):
            raise TypeError(":arg:`function_space_importer` must be of type"
                            " FiredrakeFunctionSpaceImporter")

        if not isinstance(mesh_importer, FiredrakeMeshGeometryImporter):
            raise TypeError(":arg:`mesh_importer` must be of type"
                            " FiredrakeMeshGeometryImporter")

        # Make sure importers are good for given function space
        assert function_space_importer.data == function_space.topological
        assert mesh_importer.data == function_space.mesh()

        # }}}

        # Initialize as Analog
        super(FiredrakeWithGeometryImporter, self).__init__(function_space)
        self._topology_importer = function_space_importer
        self._mesh_importer = mesh_importer
        self._cl_ctx = cl_ctx

        finat_element_importer = function_space_importer.finat_element_importer
        self._shared_data = \
            FiredrakeFunctionSpaceDataImporter(cl_ctx,
                                               mesh_importer,
                                               finat_element_importer)

        mesh_order = mesh_importer.data.coordinates.\
            function_space().finat_element.degree
        if mesh_order > self.data.degree:
            warn("Careful! When the mesh order is higher than the element"
                 " order. Conversion MIGHT work, but maybe not..."
                 " To be honest I really don't know.")

        # Used to convert between refernce node sets
        self._resampling_mat_fd2mm = None
        self._resampling_mat_mm2fd = None

    def __getattr__(self, attr):
        return getattr(self._topology_a, attr)

    def mesh_importer(self):
        return self._mesh_importer

    def _reordering_array(self, firedrake_to_meshmode):
        if firedrake_to_meshmode:
            return self._shared_data.firedrake_to_meshmode()
        return self._shared_data.meshmode_to_firedrake()

    def factory(self):
        return self._shared_data.factory()

    def discretization(self):
        return self._shared_data.discretization()

    def resampling_mat(self, firedrake_to_meshmode):
        if self._resampling_mat_fd2mm is None:
            element_grp = self.discretization().groups[0]
            self._resampling_mat_fd2mm = \
                self.finat_element_a.make_resampling_matrix(element_grp)

            self._resampling_mat_mm2fd = np.linalg.inv(self._resampling_mat_fd2mm)

        # return the correct resampling matrix
        if firedrake_to_meshmode:
            return self._resampling_mat_fd2mm
        return self._resampling_mat_mm2fd

    def reorder_nodes(self, nodes, firedrake_to_meshmode=True):
        """
        :arg nodes: An array representing function values at each of the
                    dofs, if :arg:`firedrake_to_meshmode` is *True*, should
                    be of shape (ndofs) or (ndofs, xtra_dims).
                    If *False*, should be of shape (ndofs) or (xtra_dims, ndofs)
        :arg firedrake_to_meshmode: *True* iff firedrake->meshmode, *False*
            if reordering meshmode->firedrake
        """
        # {{{ Case where shape is (ndofs,), just apply reordering

        if len(nodes.shape) == 1:
            return nodes[self._reordering_array(firedrake_to_meshmode)]

        # }}}

        # {{{ Else we have (xtra_dims, ndofs) or (ndofs, xtra_dims):

        # Make sure we have (xtra_dims, ndofs) ordering
        if firedrake_to_meshmode:
            nodes = nodes.T

        reordered_nodes = nodes[:, self._reordering_array(firedrake_to_meshmode)]

        # if converting mm->fd, change to (ndofs, xtra_dims)
        if not firedrake_to_meshmode:
            reordered_nodes = reordered_nodes.T

        return reordered_nodes

        # }}}

    def convert_function(self, function):
        """
        Convert a firedrake function defined on this function space
        to a meshmode form, reordering data as necessary
        and resampling to the unit nodes in meshmode

        :param function: A firedrake function or a
            :class:`FiredrakeFunctionImporter` instance

        :returns: A numpy array
        """
        from meshmode.interop.firedrake.function import FiredrakeFunctionImporter
        if isinstance(function, FiredrakeFunctionImporter):
            function = function.data

        # FIXME: Check that function can be converted! Maybe check the
        #        shared data? We can just do the check below
        #        but it's a little too stiff
        #assert function.function_space() == self.data

        nodes = function.dat.data

        # handle vector function spaces differently, hence the shape checks

        # {{{ Reorder the nodes to have positive orientation
        #     (and if a vector, now have meshmode [dims][nnodes]
        #      instead of firedrake [nnodes][dims] shape)

        if len(nodes.shape) > 1:
            new_nodes = [self.reorder_nodes(nodes.T[i], True) for i in
                         range(nodes.shape[1])]
            nodes = np.array(new_nodes)
        else:
            nodes = self.reorder_nodes(nodes, True)

        # }}}

        # {{{ Now convert to meshmode reference nodes
        node_view = self.discretization().groups[0].view(nodes)
        # Multiply each row (repping an element) by the resampler
        np.matmul(node_view, self.resampling_mat(True).T, out=node_view)

        # }}}

        return nodes
