__copyright__ = """
Copyright (C) 2014 Andreas Kloeckner
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

import logging
from dataclasses import dataclass
from functools import singledispatch
from typing import Any

import numpy as np

from arraycontext import flatten
from modepy.shapes import Hypercube, Shape, Simplex
from pytools import memoize_method
from pytools.obj_array import make_obj_array

from meshmode.dof_array import DOFArray
from meshmode.transform_metadata import DiscretizationFlattenedDOFAxisTag


logger = logging.getLogger(__name__)

__doc__ = """

.. autofunction:: make_visualizer

.. autoclass:: Visualizer

.. autofunction:: write_nodal_adjacency_vtk_file
"""


# {{{ helpers

def separate_by_real_and_imag(names_and_fields, real_only):
    """
    :arg names_and_fields: input data array must be already flattened into a
        single :mod:`numpy` array using :func:`_resample_to_numpy`.
    """

    for name, field in names_and_fields:
        if isinstance(field, np.ndarray) and field.dtype.char == "O":
            assert len(field.shape) == 1
            from pytools.obj_array import (
                obj_array_imag_copy,
                obj_array_real_copy,
                obj_array_vectorize,
            )

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


def _stack_object_array(vec, *, by_group=False):
    if not by_group:
        return np.stack(vec)

    return make_obj_array([
        np.stack([ri[igrp] for ri in vec])
        for igrp in range(vec[0].size)
        ])


def _resample_to_numpy(conn, vis_discr, vec, *, stack=False, by_group=False):
    """
    :arg stack: if *True* object arrays are stacked into a single
        :class:`~numpy.ndarray`.
    :arg by_group: if *True*, the per-group arrays in a :class:`DOFArray`
        are flattened separately. This can be used to write each group as a
        separate mesh (in supporting formats).
    """
    # "stack" exists as mainly as a workaround for Xdmf. See here:
    # https://github.com/inducer/pyvisfile/pull/12#discussion_r550959081
    # for (minimal) discussion.
    if isinstance(vec, np.ndarray) and vec.dtype.char == "O":
        from pytools.obj_array import obj_array_vectorize
        r = obj_array_vectorize(
                lambda x: _resample_to_numpy(conn, vis_discr, x, by_group=by_group),
                vec)

        return _stack_object_array(r, by_group=by_group) if stack else r

    if isinstance(vec, DOFArray):
        actx = vec.array_context
        vec = conn(vec)

    from numbers import Number
    if by_group:
        if isinstance(vec, Number):
            return make_obj_array([
                np.full(grp.ndofs, vec) for grp in conn.to_discr.groups
                ])
        elif isinstance(vec, DOFArray):
            if __debug__:
                from meshmode.dof_array import check_dofarray_against_discr
                check_dofarray_against_discr(vis_discr, vec)

            return make_obj_array([
                actx.to_numpy(ivec).reshape(-1) for ivec in vec
                ])
        else:
            raise TypeError(f"unsupported array type: {type(vec).__name__}")
    else:
        if isinstance(vec, Number):
            nnodes = sum(grp.ndofs for grp in conn.to_discr.groups)
            return np.full(nnodes, vec)
        elif isinstance(vec, DOFArray):
            if __debug__:
                from meshmode.dof_array import check_dofarray_against_discr
                check_dofarray_against_discr(vis_discr, vec)

            return actx.to_numpy(actx.tag_axis(0,
                                               DiscretizationFlattenedDOFAxisTag(),
                                               flatten(vec, actx)))
        else:
            raise TypeError(f"unsupported array type: {type(vec).__name__}")


def preprocess_fields(names_and_fields):
    """Gets arrays out of dataclasses and removes empty arrays."""
    from dataclasses import fields, is_dataclass

    def is_empty(field):
        return field is None or (isinstance(field, np.ndarray)
            and field.dtype.char == "O" and len(field) == 0)

    result = []
    for name, field in names_and_fields:
        if is_dataclass(field):
            for attr in fields(field):
                value = getattr(field, attr.name)
                if not is_empty(value):
                    result.append((f"{name}_{attr.name}", value))
        elif not is_empty(field):
            result.append((name, field))

    return result


@dataclass(frozen=True)
class _VisConnectivityGroup:
    """
    .. attribute:: vis_connectivity

        An array of shape ``(nelements, nsubelements, primitive_element_size)``.

    .. attribute:: vtk_cell_type

    .. attribute:: subelement_nr_base

        Starting index for subelements in :attr:`vis_connectivity`.
    """

    vis_connectivity: np.ndarray
    vtk_cell_type: int
    subelement_nr_base: int

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


def _check_discr_same_connectivity(discr, other):
    if len(discr.groups) != len(other.groups):
        return False

    if not all(
            sg.discretization_key() == og.discretization_key()
            and sg.nelements == og.nelements
            for sg, og in zip(discr.groups, other.groups, strict=True)):
        return False

    return True

# }}}


# {{{ vtk submeshes

@singledispatch
def vtk_submesh_for_shape(shape: Shape, node_tuples):
    raise NotImplementedError(type(shape).__name__)


@vtk_submesh_for_shape.register(Simplex)
def _(shape: Simplex, node_tuples):
    import modepy as mp
    return mp.submesh_for_shape(shape, node_tuples)


@vtk_submesh_for_shape.register(Hypercube)
def _(shape: Hypercube, node_tuples):
    node_tuple_to_index = {nt: i for i, nt in enumerate(node_tuples)}

    # NOTE: this can't use mp.submesh_for_shape because VTK vertex order is
    # counterclockwise instead of z order
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
            }[shape.dim]

    from pytools import add_tuples
    elements = []
    for origin in node_tuples:
        try:
            elements.append(tuple(
                node_tuple_to_index[add_tuples(origin, offset)]
                for offset in el_offsets
                ))
        except KeyError:
            pass

    return elements

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
        import modepy as mp

        from meshmode.mesh import ModepyElementGroup

        if isinstance(grp.mesh_el_group, ModepyElementGroup):
            shape = grp.mesh_el_group.shape
            space = mp.space_for_shape(shape, grp.order)
            assert type(space) == type(grp.mesh_el_group.space)  # noqa: E721

            node_tuples = mp.node_tuples_for_space(space)

            el_connectivity = np.array(
                    vtk_submesh_for_shape(shape, node_tuples),
                    dtype=np.intp)

            if isinstance(shape, Simplex):
                vtk_cell_type = self.simplex_cell_types[shape.dim]
            elif isinstance(shape, Hypercube):
                vtk_cell_type = self.tensor_cell_types[shape.dim]
            else:
                raise TypeError(f"unsupported shape: {type(shape)}")

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

    @property
    @memoize_method
    def cell_types(self):
        nsubelements = sum(vgrp.nsubelements for vgrp in self.groups)
        cell_types = np.empty(nsubelements, dtype=np.uint8)
        cell_types.fill(255)

        for vgrp in self.groups:
            isubelements = np.s_[
                    vgrp.subelement_nr_base:
                    vgrp.subelement_nr_base + vgrp.nsubelements]
            cell_types[isubelements] = vgrp.vtk_cell_type

        assert (cell_types < 255).all()

        return cell_types


class VTKLagrangeConnectivity(VTKConnectivity):
    """Connectivity for high-order Lagrange elements."""

    @property
    def version(self):
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
        from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup

        vtk_version = tuple(int(v) for v in self.version.split("."))
        if isinstance(grp.mesh_el_group, SimplexElementGroup):
            from pyvisfile.vtk.vtk_ordering import (
                vtk_lagrange_simplex_node_tuples,
                vtk_lagrange_simplex_node_tuples_to_permutation,
            )

            node_tuples = vtk_lagrange_simplex_node_tuples(
                    grp.dim, grp.order, vtk_version=vtk_version)
            el_connectivity = np.array(
                    vtk_lagrange_simplex_node_tuples_to_permutation(node_tuples),
                    dtype=np.intp).reshape((1, 1, -1))

            vtk_cell_type = self.simplex_cell_types[grp.dim]

        elif isinstance(grp.mesh_el_group, TensorProductElementGroup):
            from pyvisfile.vtk.vtk_ordering import (
                vtk_lagrange_quad_node_tuples,
                vtk_lagrange_quad_node_tuples_to_permutation,
            )

            node_tuples = vtk_lagrange_quad_node_tuples(
                    grp.dim, grp.order, vtk_version=vtk_version)
            el_connectivity = np.array(
                    vtk_lagrange_quad_node_tuples_to_permutation(node_tuples),
                    dtype=np.intp).reshape((1, 1, -1))

            vtk_cell_type = self.tensor_cell_types[grp.dim]

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
                for grp_offset, grp in zip(grp_offsets[:-1], self.vis_discr.groups,
                                           strict=True)
                ])

        return self.vis_discr.mesh.nelements, connectivity, offsets

# }}}


# {{{ visualizer

class Visualizer:
    """
    .. automethod:: show_scalar_in_mayavi
    .. automethod:: show_scalar_in_matplotlib_3d
    .. automethod:: write_vtk_file
    .. automethod:: write_parallel_vtk_file
    .. automethod:: write_vtkhdf_file
    .. automethod:: write_xdmf_file

    .. automethod:: copy_with_same_connectivity
    """

    def __init__(self, connection,
            element_shrink_factor=None,
            is_equidistant=False,
            _vtk_linear_connectivity=None,
            _vtk_lagrange_connectivity=None):

        self.connection = connection
        self.discr = connection.from_discr
        self.vis_discr = connection.to_discr

        if element_shrink_factor is None:
            element_shrink_factor = 1.0
        self.element_shrink_factor = element_shrink_factor
        self.is_equidistant = is_equidistant

        self._cached_vtk_linear_connectivity = _vtk_linear_connectivity
        self._cached_vtk_lagrange_connectivity = _vtk_lagrange_connectivity

    def copy_with_same_connectivity(self, actx, discr, skip_tests=False):
        """Makes a copy of the visualizer for a
        :class:`~meshmode.discretization.Discretization` with the same group
        structure as the original discretization. This can be useful when the
        geometry is mapped (e.g. using :func:`~meshmode.mesh.processing.affine_map`)
        and the connectivity can be reused.

        The *"same group structure"* here means that the two discretizations
        should have the same group types, number of elements, degrees of
        freedom, etc.

        :arg skip_tests: If *True*, no checks in the group structure of the
            discretizations are performed.
        """

        if not skip_tests:
            if not _check_discr_same_connectivity(discr, self.discr):
                raise ValueError("'discr' does not have matching group structures")

        vis_discr = self.vis_discr.copy(actx=actx, mesh=discr.mesh)
        conn = type(self.connection)(
                discr, vis_discr,
                groups=self.connection.groups,
                is_surjective=self.connection.is_surjective)

        return type(self)(
                conn,
                element_shrink_factor=self.element_shrink_factor,
                is_equidistant=self.is_equidistant,
                _vtk_linear_connectivity=self._cached_vtk_linear_connectivity,
                _vtk_lagrange_connectivity=self._cached_vtk_lagrange_connectivity,
                )

    @memoize_method
    def _vis_nodes_numpy(self):
        actx = self.vis_discr._setup_actx
        return np.array([
            actx.to_numpy(actx.tag_axis(
                0,
                DiscretizationFlattenedDOFAxisTag(),
                flatten(actx.thaw(ary), actx)))
            for ary in self.vis_discr.nodes()
            ])

    # {{{ mayavi

    def show_scalar_in_mayavi(self, field, **kwargs):
        # pylint: disable=import-error
        import mayavi.mlab as mlab

        do_show = kwargs.pop("do_show", True)

        nodes = self._vis_nodes_numpy()
        field = _resample_to_numpy(self.connection, self.vis_discr, field)

        assert nodes.shape[0] == self.vis_discr.ambient_dim
        connectivity = self._vtk_connectivity()
        vis_connectivity = connectivity.groups[0].vis_connectivity

        if self.vis_discr.dim == 1:
            nodes = list(nodes)
            # pad to 3D with zeros
            while len(nodes) < 3:
                nodes.append(0*nodes[0])
            assert len(nodes) == 3

            args = (*nodes, field)

            # https://docs.enthought.com/mayavi/mayavi/auto/example_plotting_many_lines.html  # noqa: E501
            src = mlab.pipeline.scalar_scatter(*args)

            src.mlab_source.dataset.lines = vis_connectivity.reshape(-1, 2)
            lines = mlab.pipeline.stripper(src)
            mlab.pipeline.surface(lines, **kwargs)

        elif self.vis_discr.dim == 2:
            nodes = list(nodes)
            # pad to 3D with zeros
            while len(nodes) < 3:
                nodes.append(0*nodes[0])

            args = (*nodes, vis_connectivity.reshape(-1, 3))
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
    def _vtk_linear_connectivity(self):
        if self._cached_vtk_linear_connectivity is None:
            self._cached_vtk_linear_connectivity = VTKConnectivity(self.connection)

        return self._cached_vtk_linear_connectivity

    @property
    def _vtk_lagrange_connectivity(self):
        assert self.is_equidistant

        if self._cached_vtk_lagrange_connectivity is None:
            self._cached_vtk_lagrange_connectivity = \
                    VTKLagrangeConnectivity(self.connection)

        return self._cached_vtk_lagrange_connectivity

    def _vtk_connectivity(self, use_high_order: bool = False) -> VTKConnectivity:
        if use_high_order:
            if not self.is_equidistant:
                raise RuntimeError("Cannot visualize high-order Lagrange elements "
                        "using a non-equidistant visualizer. "
                        "Call 'make_visualizer' with 'force_equidistant=True'.")

            return self._vtk_lagrange_connectivity
        else:
            return self._vtk_linear_connectivity

    def write_parallel_vtk_file(self, mpi_comm, file_name_pattern, names_and_fields,
                compressor=None, real_only=False,
                overwrite=False, use_high_order=None,
                par_manifest_filename=None):
        r"""A convenience wrapper around :meth:`write_vtk_file` for
        distributed-memory visualization.

        :arg mpi_comm: An object that supports ``mpi_comm.Get_rank()``
            and ``mpi_comm.Get_size()`` method calls, typically (but not
            necessarily) an instance of ``mpi4py.Comm``. This is used
            to determine the current rank as well as the total number
            of files being written.
            May also be *None* in which case a unit-size communicator
            is assumed.
        :arg file_name_pattern: A file name pattern (required to end in ``.vtu``)
            that will be used with :meth:`str.format` with an (integer)
            argument of ``rank`` to obtain the per-rank file name. Relative
            path names are also supported.
        :arg par_manifest_filename: as in :meth:`write_vtk_file`.
            If not given, *par_manifest_filename* is synthesized by
            substituting rank 0 into *file_name_pattern* and replacing the file
            extension with ``.pvtu``.

        See :meth:`write_vtk_file` for the meaning of the remainder of the
        arguments.

        .. versionadded:: 2020.2
        """
        if mpi_comm is not None:
            rank = mpi_comm.Get_rank()
            nranks = mpi_comm.Get_size()
        else:
            rank = 0
            nranks = 1

        if par_manifest_filename is None:
            par_manifest_filename = file_name_pattern.format(rank=0)
            if not par_manifest_filename.endswith(".vtu"):
                raise ValueError("file_name_pattern must produce file names "
                        "ending in '.vtu'")

            par_manifest_filename = par_manifest_filename[:-4] + ".pvtu"

        self.write_vtk_file(
                file_name=file_name_pattern.format(rank=rank),
                names_and_fields=names_and_fields,
                compressor=compressor,
                real_only=real_only,
                overwrite=overwrite,
                use_high_order=use_high_order,
                par_manifest_filename=par_manifest_filename,
                par_file_names=[
                    file_name_pattern.format(rank=rank)
                    for rank in range(nranks)
                    ]
                )

    def write_vtk_file(self, file_name, names_and_fields,
            compressor=None, real_only=False, overwrite=False,
            use_high_order=None,
            par_manifest_filename=None, par_file_names=None):
        """Write a Vtk XML file (typical extension ``.vtu``) containing
        the visualization data in *names_and_fields*. Can optionally also write
        manifests for distributed memory simulation (typical extension
        ``.pvtu``). See also :meth:`write_parallel_vtk_file` for a convenience
        wrapper.

        :arg names_and_fields: A list of tuples ``(name, value)``, where
            *name* is a string and *value* is a
            :class:`~meshmode.dof_array.DOFArray` or a constant,
            or an object array of those.
            *value* may also be a data class (see :mod:`dataclasses`),
            whose attributes will be inserted into the visualization
            with their names prefixed by *name*.
            If *value* is *None*, then there is no data to write and the
            corresponding *name* will not appear in the data file.
            If *value* is *None*, it should be *None* collectively across all
            ranks for parallel writes; otherwise the behavior of this routine
            is undefined.
        :arg overwrite: If *True*, silently overwrite existing
            files.
        :arg use_high_order: Writes arbitrary order Lagrange VTK elements.
            These elements are described in
            `this blog post <https://blog.kitware.com/modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit/>`__
            and are available in VTK 8.1 and newer.
        :arg par_manifest_filename: If not *None* write a distributed-memory
            manifest with this file name if *file_name* matches the first entry in
            *par_file_names*.
        :arg par_file_names: A list of file names of visualization files to
            include in the distributed-memory manifest.

        .. versionchanged:: 2020.2

            - Added *par_manifest_filename* and *par_file_names*.
            - Added *use_high_order*.
        """

        if use_high_order is None:
            use_high_order = False

        from pyvisfile.vtk import (
            VF_LIST_OF_COMPONENTS,
            AppendedDataXMLGenerator,
            DataArray,
            ParallelXMLGenerator,
            UnstructuredGrid,
        )

        nodes = self._vis_nodes_numpy()

        names_and_fields = preprocess_fields(names_and_fields)
        names_and_fields = [
                (name, _resample_to_numpy(
                    self.connection, self.vis_discr, fld))
                for name, fld in names_and_fields
                ]

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

        connectivity = self._vtk_connectivity(use_high_order)
        cells = connectivity.cells
        if isinstance(cells, tuple):
            cells = (cells[0],
                     DataArray("connectivity", cells[1]),
                     DataArray("offsets", cells[2]))

        nodes = nodes.reshape(self.vis_discr.ambient_dim, -1)
        points = DataArray("points", nodes, vector_format=VF_LIST_OF_COMPONENTS)

        grid = UnstructuredGrid(
                (nodes.shape[1], points),
                cells=cells,
                cell_types=connectivity.cell_types)

        for name, field in separate_by_real_and_imag(names_and_fields, real_only):
            grid.add_pointdata(
                    DataArray(name, field, vector_format=VF_LIST_OF_COMPONENTS)
                    )

        # }}}

        # {{{ write

        # {{{ write either both the vis file and the manifest, or neither

        import os
        responsible_for_writing_par_manifest = (
                par_file_names
                and par_file_names[0] == file_name)
        if os.path.exists(file_name):
            if overwrite:
                # we simply overwrite below, no need to remove
                pass
            else:
                raise FileExistsError(f"output file '{file_name}' already exists")

        if (responsible_for_writing_par_manifest
                and par_manifest_filename is not None):
            if os.path.exists(par_manifest_filename):
                if overwrite:
                    # we simply overwrite below, no need to remove
                    pass
                else:
                    raise FileExistsError(
                        f"output file '{par_manifest_filename}' already exists")

        # }}}

        with open(file_name, "w") as outf:
            generator = AppendedDataXMLGenerator(
                    compressor=compressor,
                    vtk_file_version=connectivity.version)

            generator(grid).write(outf)

        if par_file_names is not None:
            if par_manifest_filename is None:
                raise ValueError("must specify par_manifest_filename if "
                        "par_file_names are given")

            if responsible_for_writing_par_manifest:
                parfile_relnames = [
                    os.path.relpath(pn, start=os.path.dirname(par_manifest_filename))
                    for pn in par_file_names]
                with open(par_manifest_filename, "w") as outf:
                    generator = ParallelXMLGenerator(parfile_relnames)
                    generator(grid).write(outf)

        # }}}

    # }}}

    # {{{ vtkhdf

    def write_vtkhdf_file(self,
            file_name: str, names_and_fields: list[tuple[str, Any]], *,
            comm=None,
            use_high_order: bool = False,
            real_only: bool = False,
            overwrite: bool = False,
            h5_file_options: dict[str, Any] | None = None,
            dset_options: dict[str, Any] | None = None) -> None:
        """Write a VTK HDF5 file (typical extension ``'.hdf'``) containing
        the visualization fields in *names_and_fields*.

        This function requires ``h5py`` and has support for parallel writes
        through ``mpi4py``.

        :arg comm: an ``mpi4py.Comm``-like interface that supports
            ``Get_rank``, ``Get_size``, ``scan`` and ``reduce``. The last two
            are required to gather global information about the points and
            cells in the discretizations.
        :arg h5_file_options: a :class:`dict` passed directly to
            :class:`h5py.File` that allows controlling chunking, compatibility, etc.
        :arg dataset_options: a :class:`dict` passed directly to
            :meth:`h5py.Group.create_dataset`.
        """
        # {{{ setup

        try:
            import h5py
        except ImportError as exc:
            raise ImportError("'write_vtkhdf_file' requires 'h5py'") from exc

        if h5_file_options is None:
            h5_file_options = {}

        if comm is not None:
            h5_file_options["comm"] = comm
            if "driver" not in h5_file_options:
                h5_file_options["driver"] = "mpio"
            else:
                if h5_file_options["driver"] != "mpio":
                    raise ValueError(
                        "parallel HDF5 requires the 'mpio' driver: "
                        f"driver is '{h5_file_options['driver']}'")

        import os
        if os.path.splitext(file_name)[-1] != ".hdf":
            raise ValueError(f"'file_name' extension must be '.hdf': {file_name}")

        if dset_options is None:
            dset_options = {}

        names_and_fields = preprocess_fields(names_and_fields)
        names_and_fields = [
                (name, _resample_to_numpy(
                    self.connection, self.vis_discr, fld))
                for name, fld in names_and_fields
                ]

        if comm is None:
            mpisize = 1
            mpirank = 0
        else:
            mpisize = comm.Get_size()
            mpirank = comm.Get_rank()

        # }}}

        # {{{ write

        # https://gitlab.kitware.com/vtk/vtk/-/merge_requests/7552/diffs?commit_id=ff63361e1e625bf5f8ff82a4063a9bc5b9f35818#92f6af7573e5302296e4d465fea1d411d4a2611d
        # https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html#vtkhdf-file-format

        def create_dataset(grp, name, data, *, shape, offset):
            if data.ndim == 2 and data.shape[1] < 3:
                # NOTE: Paraview 5.10 (with bundled VTK 9.0.20210922) seems to
                # be hardcoded to 3D somewhere for the VTKHDF format, so we
                # pad the point arrays as well.
                data = np.pad(data, ((0, 0), (0, 3 - shape[1])))
                shape = (shape[0], 3)

            dset = grp.create_dataset(name, shape, dtype=data.dtype, **dset_options)
            dset[offset:offset + data.shape[0]] = data

            return dset

        with h5py.File(file_name, "w", **h5_file_options) as h5:
            root = h5.create_group("VTKHDF")
            root.attrs.create("Version", [1, 0])

            nodes = np.stack(self._vis_nodes_numpy()).T

            # {{{ local connectivity

            connectivity = self._vtk_connectivity(use_high_order)
            cell_types = connectivity.cell_types

            try:
                cell_count, cell_connectivity, cell_offsets = connectivity.cells
                cell_offsets = np.append([0], cell_offsets)
            except ValueError:
                from pyvisfile.vtk import CELL_NODE_COUNT
                cell_count = cell_types.size
                cell_connectivity = connectivity.cells
                cell_offsets = np.cumsum(np.append([0],
                        np.vectorize(CELL_NODE_COUNT.get)(cell_types)),
                        dtype=cell_connectivity.dtype)

            # }}}

            # {{{ determine partitions

            node_count = nodes.shape[0]
            conn_count = cell_connectivity.size

            if comm is None:
                global_cell_offset = 0
                global_node_offset = 0
                global_conn_offset = 0
                global_cell_count = cell_count
                global_node_count = node_count
                global_conn_count = conn_count
            else:
                from mpi4py import MPI

                # FIXME: should be able to do all these in one go
                global_cell_offset = comm.scan(cell_count) - cell_count
                global_node_offset = comm.scan(node_count) - node_count
                global_conn_offset = comm.scan(conn_count) - conn_count
                global_cell_count = comm.allreduce(cell_count, op=MPI.SUM)
                global_node_count = comm.allreduce(node_count, op=MPI.SUM)
                global_conn_count = comm.allreduce(conn_count, op=MPI.SUM)

            # }}}

            # {{{ write mesh

            create_dataset(root, "NumberOfCells", np.array([cell_count]),
                           shape=(mpisize,), offset=mpirank)
            create_dataset(root, "NumberOfPoints", np.array([node_count]),
                           shape=(mpisize,), offset=mpirank)
            create_dataset(root, "NumberOfConnectivityIds", np.array([conn_count]),
                           shape=(mpisize,), offset=mpirank)

            create_dataset(root, "Points", nodes,
                           shape=(global_node_count, nodes.shape[1]),
                           offset=global_node_offset)
            create_dataset(root, "Connectivity", cell_connectivity,
                           shape=(global_conn_count,), offset=global_conn_offset)
            create_dataset(root, "Offsets", cell_offsets,
                           shape=(global_cell_count + mpisize,),
                           offset=global_cell_offset + mpirank)
            create_dataset(root, "Types", cell_types,
                           shape=(global_cell_count,), offset=global_cell_offset)

            # }}}

            # {{{ write point data

            point_data = root.create_group("PointData")
            for name, field in separate_by_real_and_imag(
                    names_and_fields, real_only):
                if field.dtype.char == "O":
                    field = np.stack(field).T

                create_dataset(point_data, name, field,
                               shape=(global_node_count,) + field.shape[1:],
                               offset=global_node_offset)

            # }}}

        # }}}

    # }}}

    # {{{ xdmf

    @memoize_method
    def _xdmf_nodes_numpy(self):
        actx = self.vis_discr._setup_actx
        return _resample_to_numpy(
                lambda x: x, self.vis_discr,
                actx.thaw(self.vis_discr.nodes()),
                stack=True, by_group=True)

    def _vtk_to_xdmf_cell_type(self, cell_type):
        import pyvisfile.vtk as vtk
        from pyvisfile.xdmf import TopologyType
        return {
                vtk.VTK_LINE: TopologyType.Polyline,
                vtk.VTK_TRIANGLE: TopologyType.Triangle,
                vtk.VTK_TETRA: TopologyType.Tetrahedron,
                vtk.VTK_QUAD: TopologyType.Quadrilateral,
                vtk.VTK_HEXAHEDRON: TopologyType.Hexahedron,
                }[cell_type]

    def write_xdmf_file(self, file_name, names_and_fields,
            attrs=None, h5_file_options=None, dataset_options=None,
            real_only=False, overwrite=False):
        """Write an XDMF file (with an ``.xmf`` extension) containing the
        arrays in *names_and_fields*. The heavy data is written to binary
        HDF5 files, which requires installing :ref:`h5py <h5py:install>`.
        Distributed memory visualization is not yet supported.

        :arg names_and_fields: a list of ``(name, array)``, where *array* is
            an array-like object (see :meth:`Visualizer.write_vtk_file`).
        :arg attrs: a :class:`dict` of scalar attributes that will be saved
            in the root HDF5 group.
        :arg h5_file_options: a :class:`dict` passed directly to
            :class:`h5py.File` that allows controlling chunking, compatibility, etc.
        :arg dataset_options: a :class:`dict` passed directly to
            :meth:`h5py.Group.create_dataset`.
        """
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("'write_xdmf_file' requires 'h5py'") from exc

        if attrs is None:
            attrs = {}

        if h5_file_options is None:
            h5_file_options = {}

        dataset_defaults = {"compression": "gzip", "compression_opts": 6}
        if dataset_options is not None:
            dataset_defaults.update(dataset_options)
        dataset_options = dataset_defaults

        if "comm" in h5_file_options \
                or h5_file_options.get("driver", None) == "mpio":
            raise NotImplementedError("distributed memory visualization")

        import os
        h5_file_name = "{}.h5".format(os.path.splitext(file_name)[0])

        # {{{ expand -> filter -> resample -> to_numpy fields

        names_and_fields = preprocess_fields(names_and_fields)
        names_and_fields = [
                (name, _resample_to_numpy(
                    self.connection, self.vis_discr, field,
                    stack=True, by_group=True))
                for name, field in names_and_fields
                ]

        # }}}

        # {{{ write hdf5 + create xml tree

        # NOTE: 01-03-2021 based on Paraview 5.8.1 with (internal) VTK 8.90.0
        #
        # The current setup writes a grid for each element group. The grids
        # are completely separate, i.e. each one gets its own subset of the
        # nodes / connectivity / fields. This seems to work reasonably well.
        #
        # This mostly works with the Xdmf3ReaderS (S for spatial) Paraview
        # plugin. It seems to also work with the XMDFReader (for Xdmf2) plugin,
        # but that's not very tested.
        #
        # A few nice-to-haves / improvements
        #
        # * writing a single grid per meshmode.Mesh. `meshio` actually does this
        #   using the XDMF `TopologyType.Mixed`
        # * writing object ndarrays as separate `DataItem`s. Tried this using
        #   an Xdmf `DataItemType.Function`, but Paraview did not recognize it.
        # * Using `Reference` DataItems to e.g. store the nodes globally on the
        #   domain and just reference it in the individual grids. This crashed
        #   Paraview.

        from pyvisfile.xdmf import (
            DataArray,
            GeometryType,
            Information,
            XdmfUnstructuredGrid,
        )

        if self.vis_discr.ambient_dim == 2:
            geometry_type = GeometryType.XY
        elif self.vis_discr.ambient_dim == 3:
            geometry_type = GeometryType.XYZ
        else:
            raise ValueError(f"unsupported dimension: {self.vis_discr.dim}")

        with h5py.File(h5_file_name, "w", **h5_file_options) as h5:
            tags = []
            for key, value in attrs.items():
                h5.attrs[key] = value
                tags.append(Information(name=key, value=str(value)))

            # {{{ create grids

            nodes = self._xdmf_nodes_numpy()
            connectivity = self._vtk_connectivity()

            # global nodes
            h5grid = h5.create_group("Grid")

            grids = []
            node_nr_base = 0
            for igrp, (vgrp, gnodes) in enumerate(
                    zip(connectivity.groups, nodes, strict=True)):
                grp_name = f"Group_{igrp:05d}"
                h5grp = h5grid.create_group(grp_name)

                # offset connectivity back to local numbering
                visconn = (
                    vgrp.vis_connectivity.reshape(vgrp.nsubelements, -1)
                    - node_nr_base)
                node_nr_base += self.vis_discr.groups[igrp].ndofs

                # hdf5 side
                dset = h5grp.create_dataset("Nodes", data=gnodes.T,
                        **dataset_options)
                gnodes = DataArray.from_dataset(dset)

                dset = h5grp.create_dataset("Connectivity", data=visconn,
                        **dataset_options)
                gconnectivity = DataArray.from_dataset(dset)

                # xdmf side
                topology_type = self._vtk_to_xdmf_cell_type(vgrp.vtk_cell_type)
                grid = XdmfUnstructuredGrid(
                        gnodes, gconnectivity,
                        topology_type=topology_type,
                        geometry_type=geometry_type,
                        name=grp_name)

                # fields
                for name, field in separate_by_real_and_imag(
                        names_and_fields, real_only):
                    dset = h5grp.create_dataset(name, data=field[igrp],
                            **dataset_options)
                    grid.add_attribute(DataArray.from_dataset(dset))

                grids.append(grid)

            # }}}

        # }}}

        # {{{ write xdmf

        from pyvisfile.xdmf import XdmfWriter
        writer = XdmfWriter(tuple(grids), tags=tuple(tags))
        writer.write_pretty(file_name)

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
        field = _resample_to_numpy(self.connection, self.vis_discr, field)
        if not np.can_cast(field.dtype, float, "same_kind"):
            raise ValueError(
                    f"fields of dtype {field.dtype} are not supported: "
                    "cannot be converted to float")

        assert nodes.shape[0] == self.vis_discr.ambient_dim

        vis_connectivity, = self._vtk_connectivity().groups

        ax = plt.axes(projection="3d")
        had_data = ax.has_data()

        if self.vis_discr.dim == 2:
            nodes = list(nodes)
            # pad to 3D with zeros
            while len(nodes) < 3:
                nodes.append(0*nodes[0])

            try:
                from matplotlib.tri import Triangulation
            except ImportError:
                # NOTE: deprecated starting with v3.7
                from matplotlib.tri.triangulation import Triangulation

            tri, _, kwargs = \
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

# }}}


# {{{ make_visualizer

def make_visualizer(actx, discr, vis_order=None,
        element_shrink_factor=None, force_equidistant=False):
    """
    :arg vis_order: order of the visualization DOFs.
    :arg element_shrink_factor: number in :math:`(0, 1]`.
    :arg force_equidistant: if *True*, the visualization is done on
        equidistant nodes. If plotting high-order Lagrange VTK elements, this
        needs to be set to *True*.
    """
    vis_discr = None
    if (element_shrink_factor is None
            and not force_equidistant
            and vis_order is None):
        vis_discr = discr
    else:
        if force_equidistant:
            from meshmode.discretization.poly_element import (
                InterpolatoryEquidistantGroupFactory as VisGroupFactory,
            )
        else:
            from meshmode.discretization.poly_element import (
                InterpolatoryEdgeClusteredGroupFactory as VisGroupFactory,
            )

        vis_discr = discr.copy(actx=actx, group_factory=VisGroupFactory(vis_order))

        if all(grp.discretization_key() == vgrp.discretization_key()
                for grp, vgrp in zip(discr.groups, vis_discr.groups, strict=True)):
            from warnings import warn
            warn("Visualization discretization is identical to base discretization. "
                    "To avoid the creation of a separate discretization for "
                    "visualization, avoid passing vis_order unless needed.",
                    stacklevel=2)

            vis_discr = discr
        else:
            if vis_order is None:
                raise ValueError("A 'vis_order' must be provided for interpolation.")

    from meshmode.discretization.connection import make_same_mesh_connection
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

    # pylint: disable=no-member
    color = plt.cm.rainbow(np.linspace(0, 1, len(discr.groups)))

    for igrp, _ in enumerate(discr.groups):
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
        VF_LIST_OF_COMPONENTS,
        VTK_LINE,
        AppendedDataXMLGenerator,
        DataArray,
        UnstructuredGrid,
    )

    centroids = np.empty(
            (mesh.ambient_dim, mesh.nelements),
            dtype=mesh.vertices.dtype)

    for base_element_nr, grp in zip(mesh.base_element_nrs, mesh.groups, strict=True):
        centroids[:, base_element_nr:base_element_nr + grp.nelements] = (
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
    if os.path.exists(file_name):
        if overwrite:
            os.remove(file_name)
        else:
            raise FileExistsError("output file '%s' already exists" % file_name)

    with open(file_name, "w") as outf:
        AppendedDataXMLGenerator(compressor)(grid).write(outf)

# }}}

# vim: foldmethod=marker
