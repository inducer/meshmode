from __future__ import division, absolute_import

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

from six.moves import range
import numpy as np
from pytools import memoize_method, Record

__doc__ = """

.. autofunction:: make_visualizer

.. autoclass:: Visualizer

.. autofunction:: write_nodal_adjacency_vtk_file
"""


# {{{ visualizer

def separate_by_real_and_imag(data, real_only):
    for name, field in data:
        from pytools.obj_array import log_shape, is_obj_array
        ls = log_shape(field)

        if is_obj_array(field):
            assert len(ls) == 1
            from pytools.obj_array import (
                    oarray_real_copy, oarray_imag_copy,
                    with_object_array_or_scalar)

            if field[0].dtype.kind == "c":
                if real_only:
                    yield (name,
                            with_object_array_or_scalar(oarray_real_copy, field))
                else:
                    yield (name+"_r",
                            with_object_array_or_scalar(oarray_real_copy, field))
                    yield (name+"_i",
                            with_object_array_or_scalar(oarray_imag_copy, field))
            else:
                yield (name, field)
        else:
            # ls == ()
            if field.dtype.kind == "c":
                yield (name+"_r", field.real.copy())
                yield (name+"_i", field.imag.copy())
            else:
                yield (name, field)


class _VisConnectivityGroup(Record):
    """
    .. attribute:: vis_connectivity

        an array of shape ``(group.nelements,nsubelements,primitive_element_size)``

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


class Visualizer(object):
    """
    .. automethod:: show_scalar_in_mayavi
    .. automethod:: show_scalar_in_matplotlib_3d
    .. automethod:: write_vtk_file
    """

    def __init__(self, connection, element_shrink_factor=None):
        self.connection = connection
        self.discr = connection.from_discr
        self.vis_discr = connection.to_discr

        if element_shrink_factor is None:
            element_shrink_factor = 1

        self.element_shrink_factor = element_shrink_factor

    def _resample_and_get(self, queue, vec):
        from pytools.obj_array import with_object_array_or_scalar

        def resample_and_get_one(fld):
            from numbers import Number
            if isinstance(fld, Number):
                return np.ones(self.connection.to_discr.nnodes) * fld
            else:
                return self.connection(queue, fld).get(queue=queue)

        return with_object_array_or_scalar(resample_and_get_one, vec)

    # {{{ vis sub-element connectivity

    @memoize_method
    def _vis_connectivity(self):
        """
        :return: a list of :class:`_VisConnectivityGroup` instances.
        """
        # Assume that we're using modepy's default node ordering.

        from pytools import (
                generate_nonnegative_integer_tuples_summing_to_at_most as gnitstam,
                generate_nonnegative_integer_tuples_below as gnitb)
        from meshmode.mesh import TensorProductElementGroup, SimplexElementGroup

        result = []

        from pyvisfile.vtk import (
                VTK_LINE, VTK_TRIANGLE, VTK_TETRA,
                VTK_QUAD, VTK_HEXAHEDRON)

        subel_nr_base = 0
        node_nr_base = 0

        for group in self.vis_discr.groups:
            if isinstance(group.mesh_el_group, SimplexElementGroup):
                node_tuples = list(gnitstam(group.order, group.dim))

                from modepy.tools import submesh
                el_connectivity = np.array(
                        submesh(node_tuples),
                        dtype=np.intp)
                vtk_cell_type = {
                        1: VTK_LINE,
                        2: VTK_TRIANGLE,
                        3: VTK_TETRA,
                        }[group.dim]

            elif isinstance(group.mesh_el_group, TensorProductElementGroup):
                node_tuples = list(gnitb(group.order+1, group.dim))
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
                        }[group.dim]

                el_connectivity = np.array([
                        [
                            node_tuple_to_index[add_tuple(origin, offset)]
                            for offset in el_offsets]
                        for origin in gnitb(group.order, group.dim)])

                vtk_cell_type = {
                        1: VTK_LINE,
                        2: VTK_QUAD,
                        3: VTK_HEXAHEDRON,
                        }[group.dim]

            else:
                raise NotImplementedError("visualization for element groups "
                        "of type '%s'" % type(group.mesh_el_group).__name__)

            assert len(node_tuples) == group.nunit_dofs
            vis_connectivity = (
                    node_nr_base + np.arange(
                        0, group.nelements*group.nunit_dofs, group.nunit_dofs
                        )[:, np.newaxis, np.newaxis]
                    + el_connectivity).astype(np.intp)

            vgrp = _VisConnectivityGroup(
                vis_connectivity=vis_connectivity,
                vtk_cell_type=vtk_cell_type,
                subelement_nr_base=subel_nr_base)
            result.append(vgrp)

            subel_nr_base += vgrp.nsubelements
            node_nr_base += group.ndofs

        return result

    # }}}

    # {{{ mayavi

    def show_scalar_in_mayavi(self, field, **kwargs):
        import mayavi.mlab as mlab

        do_show = kwargs.pop("do_show", True)

        with cl.CommandQueue(self.vis_discr.cl_context) as queue:
            nodes = self.vis_discr.nodes().with_queue(queue).get()

            field = self._resample_and_get(queue, field)

        assert nodes.shape[0] == self.vis_discr.ambient_dim
        #mlab.points3d(nodes[0], nodes[1], 0*nodes[0])

        vis_connectivity, = self._vis_connectivity()

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

    def write_vtk_file(self, file_name, names_and_fields,
                       compressor=None,
                       real_only=False,
                       overwrite=False):

        from pyvisfile.vtk import (
                UnstructuredGrid, DataArray,
                AppendedDataXMLGenerator,
                VF_LIST_OF_COMPONENTS)

        with cl.CommandQueue(self.vis_discr.cl_context) as queue:
            nodes = self.vis_discr.nodes().with_queue(queue).get()

            names_and_fields = [
                    (name, self._resample_and_get(queue, fld))
                    for name, fld in names_and_fields]

        vc_groups = self._vis_connectivity()

        # {{{ create cell_types

        nsubelements = sum(vgrp.nsubelements for vgrp in vc_groups)
        cell_types = np.empty(nsubelements, dtype=np.uint8)
        cell_types.fill(255)
        for vgrp in vc_groups:
            cell_types[
                    vgrp.subelement_nr_base:
                    vgrp.subelement_nr_base + vgrp.nsubelements] = \
                            vgrp.vtk_cell_type
        assert (cell_types < 255).all()

        # }}}

        if self.element_shrink_factor != 1:
            for vgrp in self.vis_discr.groups:
                nodes_view = vgrp.view(nodes)
                el_centers = np.mean(nodes_view, axis=-1)
                nodes_view[:] = (
                        (self.element_shrink_factor * nodes_view)
                        + (1-self.element_shrink_factor)
                        * el_centers[:, :, np.newaxis])

        grid = UnstructuredGrid(
                (self.vis_discr.groups[0].ndofs,
                    DataArray("points",
                        nodes.reshape(self.vis_discr.ambient_dim, -1),
                        vector_format=VF_LIST_OF_COMPONENTS)),
                cells=np.hstack([
                    vgrp.vis_connectivity.reshape(-1)
                    for vgrp in vc_groups]),
                cell_types=cell_types)

        # for name, field in separate_by_real_and_imag(cell_data, real_only):
        #     grid.add_celldata(DataArray(name, field,
        #         vector_format=VF_LIST_OF_COMPONENTS))

        for name, field in separate_by_real_and_imag(names_and_fields, real_only):
            grid.add_pointdata(DataArray(name, field,
                vector_format=VF_LIST_OF_COMPONENTS))

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

    # {{{ matplotlib 3D

    def show_scalar_in_matplotlib_3d(self, field, **kwargs):
        import matplotlib.pyplot as plt

        # This import also registers the 3D projection.
        import mpl_toolkits.mplot3d.art3d as art3d

        do_show = kwargs.pop("do_show", True)
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)
        norm = kwargs.pop("norm", None)

        with cl.CommandQueue(self.vis_discr.cl_context) as queue:
            nodes = self.vis_discr.nodes().with_queue(queue).get()

            field = self._resample_and_get(queue, field)

        assert nodes.shape[0] == self.vis_discr.ambient_dim

        vis_connectivity, = self._vis_connectivity()

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


def make_visualizer(queue, discr, vis_order, element_shrink_factor=None):
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import (
            PolynomialWarpAndBlendElementGroup,
            LegendreGaussLobattoTensorProductElementGroup,
            OrderAndTypeBasedGroupFactory)
    vis_discr = Discretization(
            discr.cl_context, discr.mesh,
            OrderAndTypeBasedGroupFactory(
                vis_order,
                simplex_group_class=PolynomialWarpAndBlendElementGroup,
                tensor_product_group_class=(
                    LegendreGaussLobattoTensorProductElementGroup)),
            real_dtype=discr.real_dtype)
    from meshmode.discretization.connection import \
            make_same_mesh_connection

    return Visualizer(
            make_same_mesh_connection(vis_discr, discr),
            element_shrink_factor=element_shrink_factor)

# }}}


# {{{ draw_curve

def draw_curve(discr):
    mesh = discr.mesh

    import matplotlib.pyplot as plt
    plt.plot(mesh.vertices[0], mesh.vertices[1], "o")

    color = plt.cm.rainbow(np.linspace(0, 1, len(discr.groups)))
    with cl.CommandQueue(discr.cl_context) as queue:
        for i, group in enumerate(discr.groups):
            group_nodes = group.view(discr.nodes()).get(queue=queue)
            artist_handles = plt.plot(
                    group_nodes[0].T,
                    group_nodes[1].T, "-x",
                    color=color[i])

            if artist_handles:
                artist_handles[0].set_label("Group %d" % i)

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
