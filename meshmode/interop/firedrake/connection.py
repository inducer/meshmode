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
"""

import numpy as np
import numpy.linalg as la

from meshmode.interop.firedrake.mesh import import_firedrake_mesh
from meshmode.interop.firedrake.reference_cell import (
    get_affine_reference_simplex_mapping, get_finat_element_unit_nodes)

from meshmode.mesh.processing import get_simplex_element_flip_matrix

from meshmode.discretization.poly_element import \
    InterpolatoryQuadratureSimplexGroupFactory
from meshmode.discretization import Discretization

from modepy import resampling_matrix


def _reorder_nodes(orient, nodes, flip_matrix, unflip=False):
    """
    flips :param:`nodes` in place according to :param:`orient`

    :param orient: An array of shape *(nelements)* of orientations,
                 >0 for positive, <0 for negative
    :param nodes: a *(nelements, nunit_nodes)* or shaped array of nodes
    :param flip_matrix: The matrix used to flip each negatively-oriented
                      element
    :param unflip: If *True*, use transpose of :param:`flip_matrix` to
                 flip negatively-oriented elements
    """
    # reorder nodes (Code adapted from
    # meshmode.mesh.processing.flip_simplex_element_group)

    # ( round to int bc applying on integers)
    flip_mat = np.rint(flip_matrix)
    if unflip:
        flip_mat = flip_mat.T

    # flipping twice should be identity
    assert la.norm(
        np.dot(flip_mat, flip_mat)
        - np.eye(len(flip_mat))) < 1e-13

    # flip nodes that need to be flipped, note that this point we act
    # like we are in a DG space
    nodes[orient < 0] = np.einsum(
        "ij,ej->ei",
        flip_mat, nodes[orient < 0])


class FromFiredrakeConnection:
    """
    A connection created from a :mod:`firedrake`
    ``"CG"`` or ``"DG"`` function space which creates a corresponding
    meshmode discretization and allows
    transfer of functions to and from :mod:`firedrake`.

    .. attribute:: to_discr

        The discretization corresponding to the firedrake function
        space created with a
        :class:`InterpolatoryQuadratureSimplexElementGroup`.
    """
    def __init__(self, cl_ctx, fdrake_fspace):
        """
        :param cl_ctx: A :mod:`pyopencl` computing context
        :param fdrake_fspace: A :mod:`firedrake` ``"CG"`` or ``"DG"``
            function space (of class :class:`WithGeometry`) built on
            a mesh which is importable by :func:`import_firedrake_mesh`.
        """
        # Ensure fdrake_fspace is a function space with appropriate reference
        # element.
        from firedrake.functionspaceimpl import WithGeometry
        if not isinstance(fdrake_fspace, WithGeometry):
            raise TypeError(":param:`fdrake_fspace` must be of firedrake type "
                            ":class:`WithGeometry`, not `%s`."
                            % type(fdrake_fspace))
        ufl_elt = fdrake_fspace.ufl_element()

        if ufl_elt.family() not in ('Lagrange', 'Discontinuous Lagrange'):
            raise ValueError("the ``ufl_element().family()`` of "
                             ":param:`fdrake_fspace` must "
                             "be ``'Lagrange'`` or "
                             "``'Discontinuous Lagrange'``, not %s."
                             % ufl_elt.family())

        # Create to_discr
        mm_mesh, orient = import_firedrake_mesh(fdrake_fspace.mesh())
        factory = InterpolatoryQuadratureSimplexGroupFactory(ufl_elt.degree())
        self.to_discr = Discretization(cl_ctx, mm_mesh, factory)

        # Get meshmode unit nodes
        element_grp = self.to_discr.groups[0]
        mm_unit_nodes = element_grp.unit_nodes
        # get firedrake unit nodes and map onto meshmode reference element
        dim = fdrake_fspace.mesh().topological_dimension()
        fd_ref_cell_to_mm = get_affine_reference_simplex_mapping(dim, True)
        fd_unit_nodes = get_finat_element_unit_nodes(fdrake_fspace.finat_element)
        fd_unit_nodes = fd_ref_cell_to_mm(fd_unit_nodes)

        # compute resampling matrices
        self._resampling_mat_fd2mm = resampling_matrix(element_grp.basis(),
                                                       new_nodes=mm_unit_nodes,
                                                       old_nodes=fd_unit_nodes)
        self._resampling_mat_mm2fd = resampling_matrix(element_grp.basis(),
                                                       new_nodes=fd_unit_nodes,
                                                       old_nodes=mm_unit_nodes)

        # handle reordering fd->mm
        flip_mat = get_simplex_element_flip_matrix(ufl_elt.degree(),
                                                   fd_unit_nodes)
        fd_cell_node_list = fdrake_fspace.cell_node_list
        _reorder_nodes(orient, fd_cell_node_list, flip_mat, unflip=False)
        self._reordering_arr_fd2mm = fd_cell_node_list.flatten()

        # handle reordering mm->fd (this only works in the discontinuous
        #                           case)
        nnodes = self.to_discr.nnodes
        mm_cell_node_list = self.to_discr.groups[0].view(np.arange(nnodes))
        _reorder_nodes(orient, mm_cell_node_list, flip_mat, unflip=True)
        self._reordering_arr_mm2fd = mm_cell_node_list.flatten()

        # Store things that we need for *from_fspace*
        self._ufl_element = ufl_elt
        self._mesh_geometry = fdrake_fspace.mesh()
        self._fspace_cache = {}  # map vector dim -> firedrake fspace

    def from_fspace(self, dim=None):
        """
        Return a firedrake function space of the appropriate vector dimension

        :param dim: Either *None*, in which case a function space which maps
                    to scalar values is returned, or a positive integer *n*,
                    in which case a function space which maps into *\\R^n*
                    is returned
        :return: A :mod:`firedrake` :class:`WithGeometry` which corresponds to
                 :attr:`to_discr` of the appropriate vector dimension
        """
        # Cache the function spaces created to avoid high overhead.
        # Note that firedrake is smart about re-using shared information,
        # so this is not duplicating mesh/reference element information
        if dim not in self._fspace_cache:
            assert (isinstance(dim, int) and dim > 0) or dim is None
            if dim is None:
                from firedrake import FunctionSpace
                self._fspace_cache[dim] = \
                    FunctionSpace(self._mesh_geometry,
                                  self._ufl_element.family(),
                                  degree=self._ufl_element.degree())
            else:
                from firedrake import VectorFunctionSpace
                self._fspace_cache[dim] = \
                    VectorFunctionSpace(self._mesh_geometry,
                                        self._ufl_element.family(),
                                        degree=self._ufl_element.degree(),
                                        dim=dim)
        return self._fspace_cache[dim]

    def from_firedrake(self, function, out=None):
        """
        transport firedrake function onto :attr:`to_discr`

        :param function: A :mod:`firedrake` function to transfer onto
            :attr:`to_discr`. Its function space must have
            the same family, degree, and mesh as ``self.from_fspace()``.
        :param out: If *None* then ignored, otherwise a numpy array of the
            shape *function.dat.data.shape.T* (i.e.
            *(dim, nnodes)* or *(nnodes,)* in which :param:`function`'s
            transported data is stored.

        :return: a numpy array holding the transported function
        """
        # make sure function is a firedrake function in an appropriate
        # function space
        from firedrake.function import Function
        assert isinstance(function, Function), \
            ":param:`function` must be a :mod:`firedrake` Function"
        assert function.function_space().ufl_element().family() \
            == self._ufl_element.family() and \
            function.function_space().ufl_element().degree() \
            == self._ufl_element.degree(), \
            ":param:`function` must live in a function space with the " \
            "same family and degree as ``self.from_fspace()``"
        assert function.function_space().mesh() is self._mesh_geometry, \
            ":param:`function` mesh must be the same mesh as used by " \
            "``self.from_fspace().mesh()``"

        # Get function data as shape [nnodes][dims] or [nnodes]
        function_data = function.dat.data

        if out is None:
            shape = (self.to_discr.nnodes,)
            if len(function_data.shape) > 1:
                shape = (function_data.shape[1],) + shape
            out = np.ndarray(shape, dtype=function_data.dtype)
        # Reorder nodes
        if len(out.shape) > 1:
            out[:] = function_data.T[:, self._reordering_arr_fd2mm]
        else:
            out[:] = function_data[self._reordering_arr_fd2mm]

        # Resample at the appropriate nodes
        out_view = self.to_discr.groups[0].view(out)
        np.matmul(out_view, self._resampling_mat_fd2mm.T, out=out_view)
        return out

    def from_meshmode(self, mm_field, out=None):
        """
        transport meshmode field from :attr:`to_discr` into an
        appropriate firedrake function space.

        :param mm_field: A numpy array of shape *(nnodes,)* or *(dim, nnodes)*
            representing a function on :attr:`to_distr`.
        :param out: If *None* then ignored, otherwise a :mod:`firedrake`
            function of the right function space for the transported data
            to be stored in.

        :return: a :mod:`firedrake` :class:`Function` holding the transported
            data.
        """
        if self._ufl_element.family() == 'Lagrange':
            raise ValueError("Cannot convert functions from discontinuous "
                             " space (meshmode) to continuous firedrake "
                             " space (reference element family %s)."
                             % type(self._ufl_element.family()))
        # make sure out is a firedrake function in an appropriate
        # function space
        if out is not None:
            from firedrake.function import Function
            assert isinstance(out, Function), \
                ":param:`out` must be a :mod:`firedrake` Function or *None*"
            assert out.function_space().ufl_element().family() \
                == self._ufl_element.family() and \
                out.function_space().ufl_element().degree() \
                == self._ufl_element.degree(), \
                ":param:`out` must live in a function space with the " \
                "same family and degree as ``self.from_fspace()``"
            assert out.function_space().mesh() is self._mesh_geometry, \
                ":param:`out` mesh must be the same mesh as used by " \
                "``self.from_fspace().mesh()`` or *None*"
        else:
            if len(mm_field.shape) == 1:
                dim = None
            else:
                dim = mm_field.shape[0]
            out = Function(self.from_fspace(dim))

        # Handle 1-D case
        if len(out.dat.data.shape) == 1 and len(mm_field.shape) > 1:
            mm_field = mm_field.reshape(mm_field.shape[1])

        # resample from nodes
        by_cell_field_view = self.to_discr.groups[0].view(mm_field)
        by_cell_out_view = self.to_discr.groups[0].view(out.dat.data.T)
        np.matmul(by_cell_field_view, self._resampling_mat_mm2fd.T,
                  out=by_cell_out_view)

        # reorder data
        if len(out.dat.data.shape) == 1:
            out.dat.data[:] = out.dat.data[self._reordering_arr_mm2fd]
        else:
            out.dat.data[:] = out.dat.data.T[self._reordering_arr_mm2fd, :]

        return out
