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
from pytools import memoize_method, Record
from meshmode.dof_array import DOFArray, flatten, thaw


__doc__ = """

.. autofunction:: make_visualizer

.. autoclass:: Visualizer

.. autofunction:: write_nodal_adjacency_vtk_file
"""


# {{{ helpers

def separate_by_real_and_imag(names_and_fields, real_only):
    """
    :arg names_and_fields: input data array must be already flattened into a
        single :mod:`numpy` array using :func:`resample_to_numpy`.
    """

    for name, field in names_and_fields:
        if isinstance(field, np.ndarray) and field.dtype.char == "O":
            assert len(field.shape) == 1
            from pytools.obj_array import (
                    obj_array_real_copy, obj_array_imag_copy,
                    obj_array_vectorize)

            if field[0].dtype.kind == "c":
                if real_only:
                    yield (name,
                            obj_array_vectorize(obj_array_real_copy, field))
                else:
                    yield (f"{name}_r",
                            obj_array_vectorize(obj_array_real_copy, field))
                    yield (f"{name}_i",
                            obj_array_vectorize(obj_array_imag_copy, field))
            else:
                yield (name, field)
        else:
            if field.dtype.kind == "c":
                if real_only:
                    yield (name, field.real.copy())
                else:
                    yield (f"{name}_r", field.real.copy())
                    yield (f"{name}_i", field.imag.copy())
            else:
                yield (name, field)


def resample_to_numpy(conn, vec):
    if (isinstance(vec, np.ndarray)
            and vec.dtype.char == "O"
            and not isinstance(vec, DOFArray)):
        from pytools.obj_array import obj_array_vectorize
        return obj_array_vectorize(lambda x: resample_to_numpy(conn, x), vec)

    from numbers import Number
    if isinstance(vec, Number):
        nnodes = sum(grp.ndofs for grp in conn.to_discr.groups)
        return np.ones(nnodes) * vec
    else:
        resampled = conn(vec)
        actx = resampled.array_context
        return actx.to_numpy(flatten(resampled))


class _VisConnectivityGroup(Record):
    """
    .. attribute:: vis_connectivity

        an array of shape ``(group.nelements, nsubelements, primitive_element_size)``

    .. attribute:: vtk_cell_type

    .. attribute:: subelement_nr_base
    """

    @property
    def nsubelements(self):
        return self.nelements * self.nsubelements_per_element

    @property
    def nelements(self):
        return self.vis_connectivity.shape[0]

    @property
    def nsubelements_per_element(self):
        return self.vis_connectivity.shape[1]

    @property
    def primitive_element_size(self):
        return self.vis_connectivity.shape[2]

# }}}


# {{{ vtk connectivity

class VTKConnectivity:
    """Connectivity for standard linear VTK element types.

    .. attribute:: version
    .. attribute:: cells
    .. attribute:: groups
    """

    def __init__(self, connection):
        self.connection = connection
        self.discr = connection.from_discr
        self.vis_discr = connection.to_discr

    @property
    def version(self):
        return "0.1"

    @property
    def simplex_cell_types(self):
        import pyvisfile.vtk as vtk
        return {
                1: vtk.VTK_LINE,
                2: vtk.VTK_TRIANGLE,
                3: vtk.VTK_TETRA,
                }

    @property
    def tensor_cell_types(self):
        import pyvisfile.vtk as vtk
        return {
                1: vtk.VTK_LINE,
                2: vtk.VTK_QUAD,
                3: vtk.VTK_HEXAHEDRON,
                }

    def connectivity_for_element_group(self, grp):
        from pytools import (
                generate_nonnegative_integer_tuples_summing_to_at_most as gnitstam,
                generate_nonnegative_integer_tuples_below as gnitb)
        from meshmode.mesh import TensorProductElementGroup, SimplexElementGroup

        if isinstance(grp.mesh_el_group, SimplexElementGroup):
            node_tuples = list(gnitstam(grp.order, grp.dim))

            from modepy.tools import simplex_submesh
            el_connectivity = np.array(
                    simplex_submesh(node_tuples),
                    dtype=np.intp)

            vtk_cell_type = self.simplex_cell_types[grp.dim]

        elif isinstance(grp.mesh_el_group, TensorProductElementGroup):
            node_tuples = list(gnitb(grp.order+1, grp.dim))
            node_tuple_to_index = dict(
                    (nt, i) for i, nt in enumerate(node_tuples))

            def add_tuple(a, b):
                return tuple(ai+bi for ai, bi in zip(a, b))

            el_offsets = {
                    1: [(0,), (1,)],
                    2: [(0, 0), (1, 0), (1, 1), (0, 1)],
                    3: [
                        (0, 0, 0),
                        (1, 0, 0),
                        (1, 1, 0),
                        (0, 1, 0),
                        (0, 0, 1),
                        (1, 0, 1),
                        (1, 1, 1),
                        (0, 1, 1),
                        ]
                    }[grp.dim]

            el_connectivity = np.array([
                    [
                        node_tuple_to_index[add_tuple(origin, offset)]
                        for offset in el_offsets]
                    for origin in gnitb(grp.order, grp.dim)])

            vtk_cell_type = self.tensor_cell_types[grp.dim]

        else:
            raise NotImplementedError("visualization for element groups "
                    "of type '%s'" % type(grp.mesh_el_group).__name__)

        assert len(node_tuples) == grp.nunit_dofs
        return el_connectivity, vtk_cell_type

    @property
    @memoize_method
    def cells(self):
        return np.hstack([
            vgrp.vis_connectivity.reshape(-1) for vgrp in self.groups
            ])

    @property
    @memoize_method
    def groups(self):
        """
        :return: a list of :class:`_VisConnectivityGroup` instances.
        """
        # Assume that we're using modepy's default node ordering.

        result = []
        subel_nr_base = 0
        node_nr_base = 0

        for grp in self.vis_discr.groups:
            el_connectivity, vtk_cell_type = \
                    self.connectivity_for_element_group(grp)

            offsets = node_nr_base + np.arange(
                    0,
                    grp.nelements * grp.nunit_dofs,
                    grp.nunit_dofs).reshape(-1, 1, 1)
            vis_connectivity = (offsets + el_connectivity).astype(np.intp)

            vgrp = _VisConnectivityGroup(
                vis_connectivity=vis_connectivity,
                vtk_cell_type=vtk_cell_type,
                subelement_nr_base=subel_nr_base)
            result.append(vgrp)

            subel_nr_base += vgrp.nsubelements
            node_nr_base += grp.ndofs

        return result


class VTKLagrangeConnectivity(VTKConnectivity):
    """Connectivity for high-order Lagrange elements."""

    @property
    def version(self):
        # NOTE: version 2.2 has an updated ordering for the hexahedron
        # elements that is not supported currently
        # https://gitlab.kitware.com/vtk/vtk/-/merge_requests/6678
        return "2.0"

    @property
    def simplex_cell_types(self):
        import pyvisfile.vtk as vtk
        return {
                1: vtk.VTK_LAGRANGE_CURVE,
                2: vtk.VTK_LAGRANGE_TRIANGLE,
                3: vtk.VTK_LAGRANGE_TETRAHEDRON,
                }

    @property
    def tensor_cell_types(self):
        import pyvisfile.vtk as vtk
        return {
                1: vtk.VTK_LAGRANGE_CURVE,
                2: vtk.VTK_LAGRANGE_QUADRILATERAL,
                3: vtk.VTK_LAGRANGE_HEXAHEDRON,
                }

    def connectivity_for_element_group(self, grp):
        from meshmode.mesh import SimplexElementGroup

        if isinstance(grp.mesh_el_group, SimplexElementGroup):
            from pyvisfile.vtk.vtk_ordering import (
                    vtk_lagrange_simplex_node_tuples,
                    vtk_lagrange_simplex_node_tuples_to_permutation)

            node_tuples = vtk_lagrange_simplex_node_tuples(
                    grp.dim, grp.order, is_consistent=True)
            el_connectivity = np.array(
                    vtk_lagrange_simplex_node_tuples_to_permutation(node_tuples),
                    dtype=np.intp).reshape(1, 1, -1)

            vtk_cell_type = self.simplex_cell_types[grp.dim]

        else:
            raise NotImplementedError("visualization for element groups "
                    "of type '%s'" % type(grp.mesh_el_group).__name__)

        assert len(node_tuples) == grp.nunit_dofs
        return el_connectivity, vtk_cell_type

    @property
    @memoize_method
    def cells(self):
        connectivity = np.hstack([
            grp.vis_connectivity.reshape(-1)
            for grp in self.groups
            ])

        grp_offsets = np.cumsum([0] + [
            grp.ndofs for grp in self.vis_discr.groups
            ])

        offsets = np.hstack([
                grp_offset + np.arange(
                    grp.nunit_dofs,
                    grp.nelements * grp.nunit_dofs + 1,
                    grp.nunit_dofs)
                for grp_offset, grp in zip(grp_offsets, self.vis_discr.groups)
                ])

        from pyvisfile.vtk import DataArray
        return (
                self.vis_discr.mesh.nelements,
                DataArray("connectivity", connectivity),
                DataArray("offsets", offsets)
                )

# }}}


# {{{ visualizer

class Visualizer(object):
    """
    .. automethod:: show_scalar_in_mayavi
    .. automethod:: show_scalar_in_matplotlib_3d
    .. automethod:: write_vtk_file
    .. automethod:: write_high_order_vtk_file
    """

    def __init__(self, connection,
            element_shrink_factor=None, is_equidistant=False):
        self.connection = connection
        self.discr = connection.from_discr
        self.vis_discr = connection.to_discr

        if element_shrink_factor is None:
            element_shrink_factor = 1.0
        self.element_shrink_factor = element_shrink_factor
        self.is_equidistant = is_equidistant

    @memoize_method
    def _vis_nodes_numpy(self):
        actx = self.vis_discr._setup_actx
        return np.array([
            actx.to_numpy(flatten(thaw(actx, ary)))
            for ary in self.vis_discr.nodes()
            ])

    # {{{ mayavi

    def show_scalar_in_mayavi(self, field, **kwargs):
        import mayavi.mlab as mlab

        do_show = kwargs.pop("do_show", True)

        nodes = self._vis_nodes_numpy()
        field = resample_to_numpy(self.connection, field)

        assert nodes.shape[0] == self.vis_discr.ambient_dim
        vis_connectivity = self._vtk_connectivity.groups[0].vis_connectivity

        if self.vis_discr.dim == 1:
            nodes = list(nodes)
            # pad to 3D with zeros
            while len(nodes) < 3:
                nodes.append(0*nodes[0])
            assert len(nodes) == 3

            args = tuple(nodes) + (field,)

            # http://docs.enthought.com/mayavi/mayavi/auto/example_plotting_many_lines.html  # noqa
            src = mlab.pipeline.scalar_scatter(*args)

            src.mlab_source.dataset.lines = vis_connectivity.reshape(-1, 2)
            lines = mlab.pipeline.stripper(src)
            mlab.pipeline.surface(lines, **kwargs)

        elif self.vis_discr.dim == 2:
            nodes = list(nodes)
            # pad to 3D with zeros
            while len(nodes) < 3:
                nodes.append(0*nodes[0])

            args = tuple(nodes) + (vis_connectivity.reshape(-1, 3),)
            kwargs["scalars"] = field

            mlab.triangular_mesh(*args, **kwargs)

        else:
            raise RuntimeError("meshes of bulk dimension %d are currently "
                    "unsupported" % self.vis_discr.dim)

        if do_show:
            mlab.show()

    # }}}

    # {{{ vtk

    @property
    @memoize_method
    def _vtk_connectivity(self):
        return VTKConnectivity(self.connection)

    @property
    @memoize_method
    def _vtk_lagrange_connectivity(self):
        assert self.is_equidistant
        return VTKLagrangeConnectivity(self.connection)

    def write_high_order_vtk_file(self, file_name, names_and_fields,
            compressor=None, real_only=False, overwrite=False):
        """Writes arbitrary order Lagrange VTK elements. These elements are
        described in `this blog post <https://blog.kitware.com/modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit/>`_
        and are available in VTK 8.1 and newer.
        """     # noqa
        if not self.is_equidistant:
            raise RuntimeError("Cannot visualize high-order Lagrange elements "
                    "using a non-equidistant visualizer. "
                    "Call 'make_visualizer' with 'force_equidistant=True'.")

        self._write_vtk_file(file_name, names_and_fields,
                connectivity=self._vtk_lagrange_connectivity,
                compressor=compressor,
                real_only=real_only,
                overwrite=overwrite)

    def write_vtk_file(self, file_name, names_and_fields,
                       par_filename=None, par_namelist=None,
                       compressor=None, mpicomm=None,
                       real_only=False, overwrite=False):
        self._write_vtk_file(file_name, names_and_fields,
                 par_filename=par_filename, par_namelist=par_namelist,
                connectivity=self._vtk_connectivity,
                compressor=compressor,
                mpicomm=mpicomm,
                real_only=real_only,
                overwrite=overwrite)

    def _write_vtk_file(self, file_name, names_and_fields, connectivity,
                        par_filename=None, par_namelist=None,
                        compressor=None, mpicomm=None,
                        real_only=False, overwrite=False):
        from pyvisfile.vtk import (
                UnstructuredGrid, DataArray,
                AppendedDataXMLGenerator,
                ParallelXMLGenerator,
                VF_LIST_OF_COMPONENTS)

        nodes = self._vis_nodes_numpy()
        names_and_fields = [
                (name, resample_to_numpy(self.connection, fld))
                for name, fld in names_and_fields]

        # {{{ create cell_types

        nsubelements = sum(vgrp.nsubelements for vgrp in connectivity.groups)
        cell_types = np.empty(nsubelements, dtype=np.uint8)
        cell_types.fill(255)

        for vgrp in connectivity.groups:
            isubelements = np.s_[
                    vgrp.subelement_nr_base:
                    vgrp.subelement_nr_base + vgrp.nsubelements]
            cell_types[isubelements] = vgrp.vtk_cell_type

        assert (cell_types < 255).all()

        # }}}

        # {{{ shrink elements

        if abs(self.element_shrink_factor - 1.0) > 1.0e-14:
            node_nr_base = 0
            for vgrp in self.vis_discr.groups:
                nodes_view = (
                        nodes[:, node_nr_base:node_nr_base + vgrp.ndofs]
                        .reshape(nodes.shape[0], vgrp.nelements, vgrp.nunit_dofs))

                el_centers = np.mean(nodes_view, axis=-1)
                nodes_view[:] = (
                        (self.element_shrink_factor * nodes_view)
                        + (1-self.element_shrink_factor)
                        * el_centers[:, :, np.newaxis])

                node_nr_base += vgrp.ndofs

        # }}}

        # {{{ create grid

        nodes = nodes.reshape(self.vis_discr.ambient_dim, -1)
        points = DataArray("points", nodes, vector_format=VF_LIST_OF_COMPONENTS)

        grid = UnstructuredGrid(
                (nodes.shape[1], points),
                cells=connectivity.cells,
                cell_types=cell_types)

        for name, field in separate_by_real_and_imag(names_and_fields, real_only):
            grid.add_pointdata(
                    DataArray(name, field, vector_format=VF_LIST_OF_COMPONENTS)
                    )

        # }}}

        # {{{ write

        import os
        from meshmode import FileExistsError
        if os.path.exists(file_name):
            if overwrite:
                os.remove(file_name)
            else:
                raise FileExistsError(f'output file {file_name} already exists')

        with open(file_name, "w") as outf:
            generator = AppendedDataXMLGenerator(
                    compressor=compressor,
                    vtk_file_version=connectivity.version)

            generator(grid).write(outf)

        if mpicomm is not None:
            rank = mpicomm.Get_rank()
            nproc = mpicomm.Get_size()
            if nproc > 1:  # don't bother for serial runs
                if rank == 0:
                    if par_filename is None:
                        par_filename = file_name+'.pvtu'
                    if os.path.exists(par_filename):
                        if overwrite:
                            os.remove(par_filename)
                        else:
                            raise FileExistsError(f'output file {par_filename}'
                                                  f' already exists.')
                    with open(par_filename, "w") as outf:
                        generator = ParallelXMLGenerator(par_namelist)
                        generator(grid).write(outf)

        # }}}

    # }}}

    # {{{ matplotlib 3D

    def show_scalar_in_matplotlib_3d(self, field, **kwargs):
        import matplotlib.pyplot as plt

        # This import also registers the 3D projection.
        import mpl_toolkits.mplot3d.art3d as art3d

        do_show = kwargs.pop("do_show", True)
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)
        norm = kwargs.pop("norm", None)

        nodes = self._vis_nodes_numpy()
        field = resample_to_numpy(self.connection, field)

        assert nodes.shape[0] == self.vis_discr.ambient_dim

        vis_connectivity, = self._vtk_connectivity.groups

        fig = plt.gcf()
        ax = fig.gca(projection="3d")

        had_data = ax.has_data()

        if self.vis_discr.dim == 2:
            nodes = list(nodes)
            # pad to 3D with zeros
            while len(nodes) < 3:
                nodes.append(0*nodes[0])

            from matplotlib.tri.triangulation import Triangulation
            tri, args, kwargs = \
                Triangulation.get_from_args_and_kwargs(
                        *nodes,
                        triangles=vis_connectivity.vis_connectivity.reshape(-1, 3))

            triangles = tri.get_masked_triangles()
            xt = nodes[0][triangles]
            yt = nodes[1][triangles]
            zt = nodes[2][triangles]
            verts = np.stack((xt, yt, zt), axis=-1)

            fieldt = field[triangles]

            polyc = art3d.Poly3DCollection(verts, **kwargs)

            # average over the three points of each triangle
            avg_field = fieldt.mean(axis=1)
            polyc.set_array(avg_field)

            if vmin is not None or vmax is not None:
                polyc.set_clim(vmin, vmax)
            if norm is not None:
                polyc.set_norm(norm)

            ax.add_collection(polyc)
            ax.auto_scale_xyz(xt, yt, zt, had_data)

        else:
            raise RuntimeError("meshes of bulk dimension %d are currently "
                    "unsupported" % self.vis_discr.dim)

        if do_show:
            plt.show()

    # }}}


def make_visualizer(actx, discr, vis_order,
        element_shrink_factor=None, force_equidistant=False):
    """
    :arg vis_order: order of the visualization DOFs.
    :arg element_shrink_factor: number in :math:`(0, 1]`.
    :arg force_equidistant: if *True*, the visualization is done on
        equidistant nodes. If plotting high-order Lagrange VTK elements, this
        needs to be set to *True*.
    """
    from meshmode.discretization import Discretization

    if force_equidistant:
        from meshmode.discretization.poly_element import (
                PolynomialEquidistantSimplexElementGroup as SimplexElementGroup,
                EquidistantTensorProductElementGroup as TensorElementGroup)
    else:
        from meshmode.discretization.poly_element import (
                PolynomialWarpAndBlendElementGroup as SimplexElementGroup,
                LegendreGaussLobattoTensorProductElementGroup as TensorElementGroup)

    from meshmode.discretization.poly_element import OrderAndTypeBasedGroupFactory
    vis_discr = Discretization(
            actx, discr.mesh,
            OrderAndTypeBasedGroupFactory(
                vis_order,
                simplex_group_class=SimplexElementGroup,
                tensor_product_group_class=TensorElementGroup),
            real_dtype=discr.real_dtype)

    from meshmode.discretization.connection import \
            make_same_mesh_connection

    return Visualizer(
            make_same_mesh_connection(actx, vis_discr, discr),
            element_shrink_factor=element_shrink_factor,
            is_equidistant=force_equidistant)

# }}}


# {{{ draw_curve

def draw_curve(discr):
    mesh = discr.mesh

    import matplotlib.pyplot as plt
    plt.plot(mesh.vertices[0], mesh.vertices[1], "o")

    color = plt.cm.rainbow(np.linspace(0, 1, len(discr.groups)))
    for igrp, group in enumerate(discr.groups):
        group_nodes = np.array([
            discr._setup_actx.to_numpy(discr.nodes()[iaxis][igrp])
            for iaxis in range(discr.ambient_dim)
            ])
        artist_handles = plt.plot(
                group_nodes[0].T,
                group_nodes[1].T, "-x",
                color=color[igrp])

        if artist_handles:
            artist_handles[0].set_label("Group %d" % igrp)

# }}}


# {{{ adjacency

def write_nodal_adjacency_vtk_file(file_name, mesh,
                                   compressor=None,
                                   overwrite=False):
    from pyvisfile.vtk import (
            UnstructuredGrid, DataArray,
            AppendedDataXMLGenerator,
            VTK_LINE,
            VF_LIST_OF_COMPONENTS)

    centroids = np.empty(
            (mesh.ambient_dim, mesh.nelements),
            dtype=mesh.vertices.dtype)

    for grp in mesh.groups:
        iel_base = grp.element_nr_base
        centroids[:, iel_base:iel_base+grp.nelements] = (
                np.sum(mesh.vertices[:, grp.vertex_indices], axis=-1)
                / grp.vertex_indices.shape[-1])

    adj = mesh.nodal_adjacency

    nconnections = len(adj.neighbors)
    connections = np.empty((nconnections, 2), dtype=np.int32)

    nb_starts = adj.neighbors_starts
    for iel in range(mesh.nelements):
        connections[nb_starts[iel]:nb_starts[iel+1], 0] = iel

    connections[:, 1] = adj.neighbors

    grid = UnstructuredGrid(
            (mesh.nelements,
                DataArray("points",
                    centroids,
                    vector_format=VF_LIST_OF_COMPONENTS)),
            cells=connections.reshape(-1),
            cell_types=np.asarray([VTK_LINE] * nconnections,
                dtype=np.uint8))

    import os
    from meshmode import FileExistsError
    if os.path.exists(file_name):
        if overwrite:
            os.remove(file_name)
        else:
            raise FileExistsError("output file '%s' already exists" % file_name)

    with open(file_name, "w") as outf:
        AppendedDataXMLGenerator(compressor)(grid).write(outf)

# }}}

# vim: foldmethod=marker
