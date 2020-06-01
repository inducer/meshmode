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
.. autoclass:: FromFiredrakeConnection
.. autofunction:: import_firedrake_mesh
.. autofunction:: import_firedrake_function_space
"""

import numpy as np


# {{{ Helper functions to construct importers for firedrake objects


def _compute_cells_near_bdy(mesh, bdy_id):
    """
    Returns an array of the cell ids with >= 1 vertex on the
    given bdy_id
    """
    cfspace = mesh.coordinates.function_space()
    cell_node_list = cfspace.cell_node_list

    boundary_nodes = cfspace.boundary_nodes(bdy_id, 'topological')
    # Reduce along each cell: Is a vertex of the cell in boundary nodes?
    cell_is_near_bdy = np.any(np.isin(cell_node_list, boundary_nodes), axis=1)

    return np.arange(cell_node_list.shape[0], dtype=np.int32)[cell_is_near_bdy]


def import_firedrake_mesh(fdrake_mesh, near_bdy=None):
    """
    :param fdrake_mesh: an instance of :mod:`firedrake`'s mesh, i.e.
                        of type :class:`firedrake.mesh.MeshGeometry`.

    :param near_bdy: If *None* does nothing. Otherwise should be a
                     boundary id. In this case, only cell ids
                     with >= 1 vertex on the given bdy id are used.

    :returns: A :class:`FiredrakeMeshGeometryImporter` object which
              is created appropriately from :param:`fdrake_mesh`

    :raises TypeError: if :param:`fdrake_mesh` is of the wrong type
    """
    # {{{ Make sure have an initialized firedrake mesh

    from firedrake.mesh import MeshGeometry, MeshTopology
    if not isinstance(fdrake_mesh, MeshGeometry):
        if isinstance(fdrake_mesh, MeshTopology):
            raise TypeError(":param:`fdrake_mesh` must have an associated"
                            " geometry, but is of type MeshTopology")
        raise TypeError(":param:`fdrake_mesh` must be of type"
                        "`firedrake.mesh.MeshGeometry`, not %s "
                        % type(fdrake_mesh))

    fdrake_mesh.init()

    # }}}

    # {{{ Use the coordinates function space to built an importer

    coords_fspace = fdrake_mesh.coordinates.function_space()
    cells_to_use = None
    if near_bdy is not None:
        cells_to_use = _compute_cells_near_bdy(fdrake_mesh, near_bdy)

    from meshmode.interop.FInAT.lagrange_element import \
        FinatLagrangeElementImporter
    from meshmode.interop.firedrake.mesh_topology import \
        FiredrakeMeshTopologyImporter
    topology_importer = FiredrakeMeshTopologyImporter(fdrake_mesh,
                                                      cells_to_use=cells_to_use)
    finat_elt_importer = FinatLagrangeElementImporter(coords_fspace.finat_element)

    from meshmode.interop.firedrake.function_space_coordless import \
        FiredrakeFunctionSpaceImporter, FiredrakeCoordinatelessFunctionImporter

    coords_fspace_importer = FiredrakeFunctionSpaceImporter(coords_fspace,
                                                            topology_importer,
                                                            finat_elt_importer)
    coordinates_importer = \
        FiredrakeCoordinatelessFunctionImporter(fdrake_mesh.coordinates,
                                                coords_fspace_importer)

    # }}}

    # FIXME Allow for normals and no_normals_warn? as in the constructor
    #       for FiredrakeMeshGeometryImporter s
    from meshmode.interop.firedrake.mesh_geometry import \
        FiredrakeMeshGeometryImporter
    return FiredrakeMeshGeometryImporter(fdrake_mesh, coordinates_importer)


def import_firedrake_function_space(cl_ctx, fdrake_fspace, mesh_importer):
    """
    Builds a
    :class:`FiredrakeWithGeometryImporter` built from the given
    :mod:`firedrake` function space :param:`fdrake_fspace`

    :param cl_ctx: A pyopencl computing context. This input
        is not checked.

    :param:`fdrake_fspace` An instance of class :mod:`firedrake`'s
        :class:`WithGeometry` class, representing a function
        space defined on a concrete mesh

    :param mesh_importer: An instance of class
        :class:`FiredrakeMeshGeometryImporter` defined on the
        mesh underlying :param:`fdrake_fspace`.

    :returns: An instance of class
        :mod :`meshmode.interop.firedrake.function_space`
        :class:`FiredrakeWithGeometryImporter` built from the given
        :param:`fdrake_fspace`

    :raises TypeError: If any input is the wrong type
    :raises ValueError: If :param:`mesh_importer` is built on a mesh
        different from the one underlying :param:`fdrake_fspace`.
    """
    # {{{ Input checking

    from firedrake.functionspaceimpl import WithGeometry
    if not isinstance(fdrake_fspace, WithGeometry):
        raise TypeError(":param:`fdrake_fspace` must be of type "
                        ":class:`firedrake.functionspaceimpl.WithGeometry`, "
                        "not %s " % type(fdrake_fspace))

    from meshmode.interop.firedrake.mesh_geometry import \
        FiredrakeMeshGeometryImporter
    if not isinstance(mesh_importer, FiredrakeMeshGeometryImporter):
        raise TypeError(":param:`mesh_importer` must be of type "
                        ":class:`FiredrakeMeshGeometryImporter` not `%s`."
                        % type(mesh_importer))

    if mesh_importer.data != fdrake_fspace.mesh():
        raise ValueError("``mesh_importer.data`` and ``fdrake_fspace.mesh()`` "
                         "must be identical")

    # }}}

    # {{{ Make an importer for the topological function space

    from meshmode.interop.FInAT import FinatLagrangeElementImporter
    from meshmode.interop.firedrake.function_space_coordless import \
        FiredrakeFunctionSpaceImporter
    mesh_importer.init(cl_ctx)
    finat_elt_importer = FinatLagrangeElementImporter(fdrake_fspace.finat_elt)
    topological_importer = FiredrakeFunctionSpaceImporter(fdrake_fspace,
                                                          mesh_importer,
                                                          finat_elt_importer)

    # }}}

    # now we can make the full importer
    from meshmode.interop.firedrake.function_space import \
        FiredrakeWithGeometryImporter
    return FiredrakeWithGeometryImporter(cl_ctx,
                                         fdrake_fspace,
                                         topological_importer,
                                         mesh_importer)

# }}}


class FromFiredrakeConnection:
    """
    Creates a method of transporting functions on a Firedrake mesh
    back and forth between firedrake and meshmode

    .. attribute:: with_geometry_importer

        An instance of class :class:`FiredrakeWithGeometryImporter`
        that converts the firedrake function space into
        meshmode and allows for field conversion.
    """
    def __init__(self, cl_ctx, fdrake_function_space):
        """
        :param cl_ctx: A computing context
        :param fdrake_function_space: A firedrake function space (an instance of
            class :class:`WithGeometry`)
        """
        mesh_importer = import_firedrake_mesh(fdrake_function_space.mesh())
        self.with_geometry_importer = \
            import_firedrake_function_space(cl_ctx,
                                            fdrake_function_space,
                                            mesh_importer)

    def from_function_space(self):
        """
        :returns: the firedrake function space this object was created from
        """
        return self.with_geometry_importer.data

    def to_factory(self):
        """
        :returns: An InterpolatoryQuadratureSimplexGroupFactory
            of the appropriate degree to use
        """
        return self.with_geometry_importer.factory()

    def to_discr(self):
        """
        :returns: A meshmode discretization corresponding to the
             firedrake function space
        """
        return self.with_geometry_importer.discretization()

    def from_firedrake(self, fdrake_function):
        """
        Convert a firedrake function on this function space to a numpy
        array

        :param queue: A CommandQueue
        :param fdrake_function: A firedrake function on the function space
            of this connection

        :returns: A numpy array holding the data of this function as
            a field for the corresponding meshmode mesh
        """
        return self.with_geometry_importer.convert_function(fdrake_function)

    def from_meshmode(self, field, fdrake_function):
        """
        Store a numpy array holding a field on the meshmode mesh into a
        firedrake function

        :param field: A numpy array representing a field on the meshmode version
            of this mesh
        :param fdrake_function: The firedrake function to store the data in
        """
        from meshmode.interop.firedrake.function import \
            FiredrakeFunctionImporter
        importer = FiredrakeFunctionImporter(fdrake_function,
                                             self.with_geometry_importer)
        importer.set_from_field(field)
