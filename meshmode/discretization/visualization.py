from __future__ import division
from __future__ import absolute_import
from six.moves import range

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
from pytools import memoize_method
import pyopencl as cl

__doc__ = """

.. autofunction:: make_visualizer

.. autoclass:: Visualizer

.. autofunction:: write_nodal_adjacency_vtk_file
"""


# {{{ visualizer

def separate_by_real_and_imag(data, real_only):
    for name, field in data:
        from pytools.obj_array import log_shape
        ls = log_shape(field)

        if ls != () and ls[0] > 1:
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


class Visualizer(object):
    """
    .. automethod:: show_scalar_in_mayavi
    .. automethod:: write_vtk_file
    """

    def __init__(self, connection):
        self.connection = connection
        self.discr = connection.from_discr
        self.vis_discr = connection.to_discr

    def _resample_and_get(self, queue, vec):
        from pytools.obj_array import with_object_array_or_scalar

        def resample_and_get_one(fld):
            return self.connection(queue, fld).get(queue=queue)

        return with_object_array_or_scalar(resample_and_get_one, vec)

    @memoize_method
    def _vis_connectivity(self):
        """
        :return: an array of shape
            ``(vis_discr.nelements,nsubelements,primitive_element_size)``
        """
        # Assume that we're using modepy's default node ordering.

        from pytools import generate_nonnegative_integer_tuples_summing_to_at_most \
                as gnitstam, single_valued
        vis_order = single_valued(
                group.order for group in self.vis_discr.groups)
        node_tuples = list(gnitstam(vis_order, self.vis_discr.dim))

        from modepy.tools import submesh
        el_connectivity = np.array(
                submesh(node_tuples),
                dtype=np.intp)

        nelements = sum(group.nelements for group in self.vis_discr.groups)
        vis_connectivity = np.empty(
                (nelements,) + el_connectivity.shape, dtype=np.intp)

        el_nr_base = 0
        for group in self.vis_discr.groups:
            assert len(node_tuples) == group.nunit_nodes
            vis_connectivity[el_nr_base:el_nr_base+group.nelements] = (
                    np.arange(
                        el_nr_base*group.nunit_nodes,
                        (el_nr_base+group.nelements)*group.nunit_nodes,
                        group.nunit_nodes
                        )[:, np.newaxis, np.newaxis]
                    + el_connectivity)
            el_nr_base += group.nelements

        return vis_connectivity

    def show_scalar_in_mayavi(self, field, **kwargs):
        import mayavi.mlab as mlab

        do_show = kwargs.pop("do_show", True)

        with cl.CommandQueue(self.vis_discr.cl_context) as queue:
            nodes = self.vis_discr.nodes().with_queue(queue).get()

            field = self._resample_and_get(queue, field)

        assert nodes.shape[0] == self.vis_discr.ambient_dim
        #mlab.points3d(nodes[0], nodes[1], 0*nodes[0])

        vis_connectivity = self._vis_connectivity()

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

    def write_vtk_file(self, file_name, names_and_fields, compressor=None,
            real_only=False):

        from pyvisfile.vtk import (
                UnstructuredGrid, DataArray,
                AppendedDataXMLGenerator,
                VTK_LINE, VTK_TRIANGLE, VTK_TETRA,
                VF_LIST_OF_COMPONENTS)
        el_types = {
                1: VTK_LINE,
                2: VTK_TRIANGLE,
                3: VTK_TETRA,
                }

        el_type = el_types[self.vis_discr.dim]

        with cl.CommandQueue(self.vis_discr.cl_context) as queue:
            nodes = self.vis_discr.nodes().with_queue(queue).get()

            names_and_fields = [
                    (name, self._resample_and_get(queue, fld))
                    for name, fld in names_and_fields]

        connectivity = self._vis_connectivity()

        nprimitive_elements = (
                connectivity.shape[0]
                * connectivity.shape[1])

        grid = UnstructuredGrid(
                (self.vis_discr.nnodes,
                    DataArray("points",
                        nodes.reshape(self.vis_discr.ambient_dim, -1),
                        vector_format=VF_LIST_OF_COMPONENTS)),
                cells=connectivity.reshape(-1),
                cell_types=np.asarray([el_type] * nprimitive_elements,
                    dtype=np.uint8))

        # for name, field in separate_by_real_and_imag(cell_data, real_only):
        #     grid.add_celldata(DataArray(name, field,
        #         vector_format=VF_LIST_OF_COMPONENTS))

        for name, field in separate_by_real_and_imag(names_and_fields, real_only):
            grid.add_pointdata(DataArray(name, field,
                vector_format=VF_LIST_OF_COMPONENTS))

        from os.path import exists
        if exists(file_name):
            raise RuntimeError("output file '%s' already exists"
                    % file_name)

        with open(file_name, "w") as outf:
            AppendedDataXMLGenerator(compressor)(grid).write(outf)


def make_visualizer(queue, discr, vis_order):
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            PolynomialWarpAndBlendGroupFactory
    vis_discr = Discretization(
            discr.cl_context, discr.mesh,
            PolynomialWarpAndBlendGroupFactory(vis_order),
            real_dtype=discr.real_dtype)
    from meshmode.discretization.connection import \
            make_same_mesh_connection

    return Visualizer(make_same_mesh_connection(vis_discr, discr))

# }}}


# {{{ draw_curve

def draw_curve(discr):
    mesh = discr.mesh

    import matplotlib.pyplot as pt
    pt.plot(mesh.vertices[0], mesh.vertices[1], "o")

    color = pt.cm.rainbow(np.linspace(0, 1, len(discr.groups)))
    with cl.CommandQueue(discr.cl_context) as queue:
        for i, group in enumerate(discr.groups):
            group_nodes = group.view(discr.nodes()).get(queue=queue)
            pt.plot(
                    group_nodes[0].T,
                    group_nodes[1].T, "-x",
                    label="Group %d" % i,
                    color=color[i])

# }}}


# {{{ adjacency

def write_nodal_adjacency_vtk_file(file_name, mesh, compressor=None,):
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

    from os.path import exists
    if exists(file_name):
        raise RuntimeError("output file '%s' already exists"
                % file_name)

    with open(file_name, "w") as outf:
        AppendedDataXMLGenerator(compressor)(grid).write(outf)

# }}}

# vim: foldmethod=marker
