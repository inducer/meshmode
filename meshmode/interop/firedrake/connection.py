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
.. autoclass:: FiredrakeConnection
    :members:
.. autoclass:: FromFiredrakeConnection
.. autoclass:: FromBdyFiredrakeConnection
.. autoclass:: ToFiredrakeConnection
"""

import numpy as np
import numpy.linalg as la
import six

from modepy import resampling_matrix

from meshmode.interop.firedrake.mesh import (
    import_firedrake_mesh, export_mesh_to_firedrake)
from meshmode.interop.firedrake.reference_cell import (
    get_affine_reference_simplex_mapping, get_finat_element_unit_dofs)

from meshmode.mesh.processing import get_simplex_element_flip_matrix

from meshmode.discretization.poly_element import \
    InterpolatoryQuadratureSimplexGroupFactory, \
    InterpolatoryQuadratureSimplexElementGroup
from meshmode.discretization import Discretization


def _reorder_nodes(orient, nodes, flip_matrix, unflip=False):
    """
    flips *nodes* in place according to *orient*

    :arg orient: An array of shape *(nelements)* of orientations,
                 >0 for positive, <0 for negative
    :arg nodes: a *(nelements, nunit_nodes)* shaped array of nodes
    :arg flip_matrix: The matrix used to flip each negatively-oriented
                      element
    :arg unflip: If *True*, use transpose of *flip_matrix* to
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


# {{{ Most basic connection between a fd function space and mm discretization

class FiredrakeConnection:
    """
    A connection between one group of
    a meshmode discretization and a firedrake "CG" or "DG"
    function space.

    Users should instantiate this using a
    :class:`FromFiredrakeConnection` or :class:`ToFiredrakeConnection`.

    .. autoattribute:: discr

        A meshmode discretization

    .. autoattribute:: group_nr

        The group number identifying which element group of
        :attr:`discr` is being connected to a firedrake function space

    .. autoattribute:: mm2fd_node_mapping

        Letting *element_grp = self.discr.groups[self.group_nr]*,
        *mm2fd_node_mapping* is a numpy array of shape
        *(element_grp.nelements, element_grp.nunit_dofs)*
        whose *(i, j)*th entry is the :mod:`firedrake` node
        index associated to the *j*th degree of freedom of the
        *i*th element in *element_grp*.

        degrees of freedom should be associated so that
        the implicit mapping from a reference element to element *i*
        which maps meshmode unit dofs *0,1,...,n-1* to actual
        dofs *(i, 0), (i, 1), ..., (i, n-1)*
        is the same mapping which maps firedrake unit dofs
        *0,1,...,n-1* to firedrake dofs
        *mm2fd_node_mapping[i,0], mm2fd_node_mapping[i,1],...,
        mm2fd_node_mapping[i,n-1]*.

        (See :mod:`meshmode.interop.firedrake.reference_cell` to
         obtain firedrake unit dofs on the meshmode reference cell)

    .. automethod:: __init__
    """
    def __init__(self, discr, fdrake_fspace, mm2fd_node_mapping, group_nr=None):
        """
        :param discr: A :mod:`meshmode` :class:`Discretization`
        :param fdrake_fspace: A :mod:`firedrake`
            :class:`firedrake.functionspaceimpl.WithGeometry`.
            Must have ufl family ``'Lagrange'`` or
            ``'Discontinuous Lagrange'``.
        :param mm2fd_node_mapping: Used as attribute :attr:`mm2fd_node_mapping`.
            A 2-D numpy integer array with the same dtype as
            ``fdrake_fspace.cell_node_list.dtype``
        :param group_nr: The index of the group in *discr* which is
            being connected to *fdrake_fspace*. The group must be
            a :class:`InterpolatoryQuadratureSimplexElementGroup`
            of the same topological dimension as *fdrake_fspace*.
            If *discr* has only one group, *group_nr=None* may be supplied.

        :raises TypeError: If any input arguments are of the wrong type,
            if the designated group is of the wrong type,
            or if *fdrake_fspace* is of the wrong family.
        :raises ValueError: If
            *mm2fd_node_mapping* is of the wrong shape
            or dtype, if *group_nr* is an invalid index, or
            if *group_nr* is *None* when *discr* has more than one group.
        """
        # {{{ Validate input
        if not isinstance(discr, Discretization):
            raise TypeError(":param:`discr` must be of type "
                            ":class:`meshmode.discretization.Discretization`, "
                            "not :class:`%s`." % type(discr))
        from firedrake.functionspaceimpl import WithGeometry
        if not isinstance(fdrake_fspace, WithGeometry):
            raise TypeError(":param:`fdrake_fspace` must be of type "
                            ":class:`firedrake.functionspaceimpl.WithGeometry`, "
                            "not :class:`%s`." % type(fdrake_fspace))
        if not isinstance(mm2fd_node_mapping, np.ndarray):
            raise TypeError(":param:`mm2fd_node_mapping` must be of type "
                            ":class:`np.ndarray`, "
                            "not :class:`%s`." % type(mm2fd_node_mapping))
        if not isinstance(group_nr, int) and group_nr is not None:
            raise TypeError(":param:`group_nr` must be of type *int* or be "
                            "*None*, not of type %s." % type(group_nr))
        # Convert group_nr to an integer if *None*
        if group_nr is None:
            if len(discr.groups) != 1:
                raise ValueError(":param:`group_nr` is *None* but :param:`discr` "
                                 "has %s != 1 groups." % len(discr.groups))
            group_nr = 0
        # store element_grp as variable for convenience
        element_grp = discr.groups[group_nr]

        if group_nr < 0 or group_nr >= len(discr.groups):
            raise ValueError(":param:`group_nr` has value %s, which an invalid "
                             "index into list *discr.groups* of length %s."
                             % (group_nr, len(discr.groups)))
        if not isinstance(element_grp,
                          InterpolatoryQuadratureSimplexElementGroup):
            raise TypeError("*discr.groups[group_nr]* must be of type "
                            ":class:`InterpolatoryQuadratureSimplexElementGroup`"
                            ", not :class:`%s`." % type(element_grp))
        allowed_families = ('Discontinuous Lagrange', 'Lagrange')
        if fdrake_fspace.ufl_element().family() not in allowed_families:
            raise TypeError(":param:`fdrake_fspace` must have ufl family "
                           "be one of %s, not %s."
                            % (allowed_families,
                               fdrake_fspace.ufl_element().family()))
        if mm2fd_node_mapping.shape != (element_grp.nelements,
                                        element_grp.nunit_dofs):
            raise ValueError(":param:`mm2fd_node_mapping` must be of shape ",
                             "(%s,), not %s"
                             % ((discr.groups[group_nr].ndofs,),
                                mm2fd_node_mapping.shape))
        if mm2fd_node_mapping.dtype != fdrake_fspace.cell_node_list.dtype:
            raise ValueError(":param:`mm2fd_node_mapping` must have dtype "
                             "%s, not %s" % (fdrake_fspace.cell_node_list.dtype,
                                             mm2fd_node_mapping.dtype))
        # }}}

        # Get meshmode unit nodes
        mm_unit_nodes = element_grp.unit_nodes()
        # get firedrake unit nodes and map onto meshmode reference element
        tdim = fdrake_fspace.mesh().topological_dimension()
        fd_ref_cell_to_mm = get_affine_reference_simplex_mapping(tdim, True)
        fd_unit_nodes = get_finat_element_unit_dofs(fdrake_fspace.finat_element)
        fd_unit_nodes = fd_ref_cell_to_mm(fd_unit_nodes)

        # compute and store resampling matrices
        self._resampling_mat_fd2mm = resampling_matrix(element_grp.basis(),
                                                       new_nodes=mm_unit_nodes,
                                                       old_nodes=fd_unit_nodes)
        self._resampling_mat_mm2fd = resampling_matrix(element_grp.basis(),
                                                       new_nodes=fd_unit_nodes,
                                                       old_nodes=mm_unit_nodes)

        # Now handle the possibility of multiple meshmode nodes being associated
        # to the same firedrake node
        unique_fd_nodes, counts = np.unique(mm2fd_node_mapping,
                                            return_counts=True)
        # self._duplicate_nodes
        # maps firedrake nodes associated to more than 1 meshmode node
        # to all associated meshmode nodes.
        self._duplicate_nodes = {}
        dup_fd_nodes = set(unique_fd_nodes[counts > 1])
        for mm_inode, fd_inode in enumerate(mm2fd_node_mapping):
            if fd_inode in dup_fd_nodes:
                self._duplicate_nodes.setdefault(fd_inode, [])
                self._duplicate_nodes[fd_inode].append(mm_inode)

        # Store input
        self.discr = discr
        self.group_nr = group_nr
        self.mm2fd_node_mapping = mm2fd_node_mapping
        self._mesh_geometry = fdrake_fspace.mesh()
        self._ufl_element = fdrake_fspace.ufl_element()
        # Cache firedrake function spaces of each vector dimension to
        # avoid overhead. Firedrake takes care of avoiding memory
        # duplication.
        self._fspace_cache = {}

    def firedrake_fspace(self, shape=None):
        """
        Return a firedrake function space that
        *self.discr.groups[self.group_nr]* is connected to
        of the appropriate vector dimension

        :arg shape: Either *None*, in which case a function space which maps
                   to scalar values is returned, a positive integer *n*,
                   in which case a function space which maps into *\\R^n*
                   is returned, or a tuple of integers defining
                   the shape of values in a tensor function space,
                   in which case a tensor function space is returned
        :return: A :mod:`firedrake` :class:`WithGeometry` which corresponds to
                 *self.discr.groups[self.group_nr]* of the appropriate vector
                 dimension

        :raises TypeError: If *shape* is of the wrong type
        """
        # Cache the function spaces created to avoid high overhead.
        # Note that firedrake is smart about re-using shared information,
        # so this is not duplicating mesh/reference element information
        if shape not in self._fspace_cache:
            if shape is None:
                from firedrake import FunctionSpace
                self._fspace_cache[shape] = \
                    FunctionSpace(self._mesh_geometry,
                                  self._ufl_element.family(),
                                  degree=self._ufl_element.degree())
            elif isinstance(shape, int):
                from firedrake import VectorFunctionSpace
                self._fspace_cache[shape] = \
                    VectorFunctionSpace(self._mesh_geometry,
                                        self._ufl_element.family(),
                                        degree=self._ufl_element.degree(),
                                        dim=shape)
            elif isinstance(shape, tuple):
                from firedrake import TensorFunctionSpace
                self._fspace_cache[shape] = \
                    TensorFunctionSpace(self._mesh_geometry,
                                        self._ufl_element.family(),
                                        degree=self._ufl_element.degree(),
                                        shape=shape)
            else:
                raise TypeError(":param:`shape` must be *None*, an integer, "
                                " or a tuple of integers, not of type %s."
                                % type(shape))
        return self._fspace_cache[shape]

    def _validate_function(function, function_name, shape=None, dtype=None):
        """
        Handy helper function to validate that a firedrake function
        is convertible (or can be the recipient of a conversion).
        Raises error messages if wrong types, shape, dtype found
        etc.
        """
        # Validate that *function* is convertible
        from firedrake.function import Function
        if not isinstance(function, Function):
            raise TypeError(function_name + " must be a :mod:`firedrake` "
                            "Function")
        ufl_elt = function.function_space().ufl_element()
        if ufl_elt.family() != self._ufl_element.family():
            raise ValueError(function_name + "'s function_space's ufl element"
                             " must have family %s, not %s"
                             % (self._ufl_element.family(), ufl_elt.family()))
        if ufl_elt.degree() != self._ufl_element.degree():
            raise ValueError(function_name + "'s function_space's ufl element"
                             " must have degree %s, not %s"
                             % (self._ufl_element.degree(), ufl_elt.degree())
        if function.function_space().mesh() is not self._mesh_geometry:
            raise ValueError(function_name + "'s mesh must be the same as "
                            "`self.from_fspace().mesh()``")
        if dtype is not None and function.dat.data.dtype != dtype:
            raise ValueError(function_name + ".dat.dtype must be %s, not %s."
                             % (dtype, function.dat.data.dtype))
        if shape is not None and function.function_space().shape != shape:
            raise ValueError(function_name + ".function_space().shape must be "
                             "%s, not %s" % (shape,
                                             function.function_space().shape))

    def _validate_field(field, field_name, shape=None, dtype=None):
        """
        Handy helper function to validate that a meshmode field
        is convertible (or can be the recipient of a conversion).
        Raises error messages if wrong types, shapes, dtypes found
        etc.
        """
        from meshmode.dof_array import DOFArray
        element_grp = self.discr.groups[self.group_nr]
        group_shape = (element_grp.nelements, element_grp.nunit_dofs)

        def check_dof_array(arr, arr_name):
            if not isinstance(arr, DOFArray):
                raise TypeError(arr_name + " must be of type "
                                ":class:`meshmode.dof_array.DOFArray`, "
                                "not :class:`%s`." % type(arr))
            if arr.shape != self.discr.groups.shape:
                raise ValueError(arr_name + " shape must be %s, not %s."
                                 % (self.discr.groups.shape, arr.shape))
            if arr[self.group_nr].shape != group_shape:
                raise VaueError(arr_name + "[%s].shape must be %s, not %s"
                                % (self.group_nr,
                                   group_shape,
                                   arr[self.group_nr].shape))
            if dtype is not None and arr.entry_dtype != dtype:
                raise ValueError(arr_name + ".entry_dtype must be %s, not %s."
                                 % (dtype, arr.entry_dtype))

        if isinstance(field, DOFArray):
            if shape is not None and shape != tuple():
                raise ValueError("shape != tuple() and %s is of type DOFArray"
                                 " instead of np.ndarray." % field_name)
            check_dof_array(field, field_name)
        elif isinstance(field, np.ndarray):
            if shape is not None and field.shape != shape:
                raise ValueError(field_name + ".shape must be %s, not %s"
                                 % (shape, field.shape))
            for i, arr in np.flatten(field):
                arr_name = "%s[%s]" % (field_name, np.unravel_index(i, shape))
                check_dof_array(arr, arr_name)
        else:
            raise TypeError("field must be of type DOFArray or np.ndarray",
                            "not %s." % type(field))

    def from_firedrake(self, function, out=None, actx=None):
        """
        transport firedrake function onto :attr:`discr`

        :arg function: A :mod:`firedrake` function to transfer onto
            :attr:`discr`. Its function space must have
            the same family, degree, and mesh as ``self.from_fspace()``.
        :arg out: Either
            1.) A :class:`meshmode.dof_array.DOFArray`
            2.) A :class:`np.ndarray` object array, each of whose
                entries is a :class:`meshmode.dof_array.DOFArray`
            3.) *None*
            In the case of (1.), *function* must be in a
            scalar function space
            (i.e. `function.function_space().shape == (,)`).
            In the case of (2.), the shape of *out* must match
            `function.function_space().shape`.

            In either case, each `DOFArray` must be a `DOFArray`
            defined on :attr:`discr` (as described in
            the documentation for :class:`meshmode.dof_array.DOFArray`).
            Also, each `DOFArray`'s *entry_dtype* must match the
            *function.dat.data.dtype*, and be of shape
            *(nelements, nunit_dofs)*.

            In case (3.), an array is created satisfying
            the above requirements.

            The data in *function* is transported to :attr:`discr`
            and stored in *out*, which is then returned.
        :arg actx:
            * If *out* is *None*, then *actx* is a
              :class:`meshmode.array_context.ArrayContext` on which
              to create the :class:`DOFArray`
            * If *out* is not *None*, *actx* must be *None* or *out*'s
              *array_context*.

        :return: *out*, with the converted data from *function* stored in it
        """
        self._validate_function(function, "function")
        # Get function data as shape *(nnodes, ...)* or *(nnodes,)*
        function_data = function.dat.data
        # get the shape needed in each DOFArray and the data shape
        element_grp = self.discr.groups[self.group_nr]
        group_shape = (element_grp.nelements, element_grp.nunit_dofs)
        fspace_shape = function.function_space().shape

        # Handle :arg:`out`
        if out is not None:
            self._validate_field(out, "out", fspace_shape, function_data.dtype)
            # If out is supplied, check type, shape, and dtype
            assert actx in (None, out.array_context), \
                "If :param:`out` is not *None*, :param:`actx` must be" \
                " *None* or *out.array_context*".
        else:
            # If `out` is not supplied, create it
            from meshmode.array_context import ArrayContext
            assert actx isinstance(actx, ArrayContext)
            if fspace_shape == tuple():
                out = self.discr.zeros(actx, dtype=function_data.dtype)
            else:
                out = \
                    np.array([self.discr.zeros(actx, dtype=function_data.dtype)
                              for _ in np.prod(fspace_shape)]
                             ).reshape(fspace_shape)

        def reorder_and_resample(dof_array, fd_data):
            dof_array[self.group_nr] = fd_data[self.mm2fd_node_mapping]
            np.matmul(dof_array[self.group_nr], self._resampling_mat_fd2mm.T,
                      out=dof_array[self.group_nr]
        # If scalar, just reorder and resample out
        if fspace_shape == tuple():
            reorder_and_resample(out, function_data)
        else:
            # otherwise, have to grab each dofarray and the corresponding
            # data from *function_data*
            with np.nditer(out, op_flags=['readwrite', 'multi_index'] as it:
                for dof_array in it:
                    fd_data = function_data[:, it.multi_index]
                    reorder_and_resample(dof_array, function_data)

        return out

    def from_meshmode(self, mm_field, out=None,
                      assert_fdrake_discontinuous=True,
                      continuity_tolerance=None):
        """
        transport meshmode field from :attr:`discr` into an
        appropriate firedrake function space.

        If *out* is *None*, values at any firedrake
        nodes associated to NO meshmode nodes are zeroed out.
        If *out* is supplied, values at nodes associated to NO meshmode nodes
        are not modified.

        :arg mm_field: A numpy array of shape *(nnodes,)* or *(..., nnodes)*
            representing a function on :attr:`to_distr`
            (where nnodes is the number of nodes in *self.discr*. Note
             that only data from group number *self.group_nr* will be
             transported).
        :arg out: If *None* then ignored, otherwise a :mod:`firedrake`
            function of the right function space for the transported data
            to be stored in.
        :arg assert_fdrake_discontinuous: If *True*,
            disallows conversion to a continuous firedrake function space
            (i.e. this function checks that ``self.firedrake_fspace()`` is
             discontinuous and raises a *ValueError* otherwise)
        :arg continuity_tolerance: If converting to a continuous firedrake
            function space (i.e. if ``self.firedrake_fspace()`` is continuous),
            assert that at any two meshmode nodes corresponding to the
            same firedrake node (meshmode is a discontinuous space, so this
            situation will almost certainly happen), the function being transported
            has values at most *continuity_tolerance* distance
            apart. If *None*, no checks are performed. Does nothing if
            the firedrake function space is discontinuous

        :return: a :mod:`firedrake` :class:`Function` holding the transported
            data.
        """
        if self._ufl_element.family() == 'Lagrange' \
                and assert_fdrake_discontinuous:
            raise ValueError("Trying to convert to continuous function space "
                             " with :arg:`assert_fdrake_discontinuous` set "
                             " to *True*")
        dtype = self.firedrake_fspace().mesh().coordinates.dat.dat.dtype
        self._validate_field(mm_field, "mm_field", dtype=dtype)
        # make sure out is a firedrake function in an appropriate
        # function space
        if out is not None:
            if isinstance(mm_field, np.ndarray):
                shape = mm_field.shape
            else:
                shape = tuple()
            self._validate_function(out, "out", shape, dtype)
        else:
            from firedrake.function import Function
            if len(mm_field.shape) == 1:
                shape = None
            elif len(mm_field.shape) == 2:
                shape = mm_field.shape[0]
            else:
                shape = mm_field.shape[:-1]
            out = Function(self.firedrake_fspace(shape))
            out.dat.data[:] = 0.0

        # Handle 1-D case
        if len(out.dat.data.shape) == 1 and len(mm_field.shape) > 1:
            mm_field = mm_field.reshape(mm_field.shape[1])

        # resample from nodes on reordered view. Have to do this in
        # a bit of a roundabout way to be careful about duplicated
        # firedrake nodes.
        el_group = self.discr.groups[self.group_nr]
        by_cell_field_view = el_group.view(mm_field)

        # Get firedrake data from out into meshmode ordering and view by cell
        reordered_outdata = \
            np.moveaxis(out.dat.data, 0, -1)[..., self.mm2fd_node_mapping]
        by_cell_reordered_view = el_group.view(reordered_outdata)
        # Resample this reordered data
        np.matmul(by_cell_field_view, self._resampling_mat_mm2fd.T,
                  out=by_cell_reordered_view)
        # Now store the resampled data back in the firedrake order
        out.dat.data[self.mm2fd_node_mapping] = \
            np.moveaxis(reordered_outdata, -1, 0)

        # Continuity checks if requested
        if self._ufl_element.family() == 'Lagrange' \
                and continuity_tolerance is not None:
            assert isinstance(continuity_tolerance, float)
            assert continuity_tolerance >= 0
            # Check each firedrake node which has been duplicated
            # that all associated values are within the continuity
            # tolerance
            for fd_inode, duplicated_mm_nodes in \
                    six.iteritems(self._duplicate_nodes):
                mm_inode = duplicated_mm_nodes[0]
                # Make sure to compare using reordered_outdata not mm_field,
                # because two meshmode nodes associated to the same firedrake
                # nodes may have been resampled to distinct nodes on different
                # elements. reordered_outdata has undone that resampling.
                for dup_mm_inode in duplicated_mm_nodes[1:]:
                    dist = la.norm(reordered_outdata[..., mm_inode]
                                   - reordered_outdata[..., dup_mm_inode])
                    if dist >= continuity_tolerance:
                        raise ValueError("Meshmode nodes %s and %s represent "
                                         "the same firedrake node %s, but "
                                         ":arg:`mm_field`'s values are "
                                         " %s > %s apart)"
                                         % (mm_inode, dup_mm_inode, fd_inode,
                                            dist, continuity_tolerance))

        return out

# }}}


# {{{ Create connection from firedrake into meshmode

class FromFiredrakeConnection(FiredrakeConnection):
    """
    A connection created from a :mod:`firedrake`
    ``"CG"`` or ``"DG"`` function space which creates a corresponding
    meshmode discretization and allows
    transfer of functions to and from :mod:`firedrake`.
    """
    def __init__(self, cl_ctx, fdrake_fspace):
        """
        :arg cl_ctx: A :mod:`pyopencl` computing context
        :arg fdrake_fspace: A :mod:`firedrake` ``"CG"`` or ``"DG"``
            function space (of class :class:`WithGeometry`) built on
            a mesh which is importable by :func:`import_firedrake_mesh`.
        """
        # Ensure fdrake_fspace is a function space with appropriate reference
        # element.
        from firedrake.functionspaceimpl import WithGeometry
        if not isinstance(fdrake_fspace, WithGeometry):
            raise TypeError(":arg:`fdrake_fspace` must be of firedrake type "
                            ":class:`WithGeometry`, not `%s`."
                            % type(fdrake_fspace))
        ufl_elt = fdrake_fspace.ufl_element()

        if ufl_elt.family() not in ('Lagrange', 'Discontinuous Lagrange'):
            raise ValueError("the ``ufl_element().family()`` of "
                             ":arg:`fdrake_fspace` must "
                             "be ``'Lagrange'`` or "
                             "``'Discontinuous Lagrange'``, not %s."
                             % ufl_elt.family())

        # Create to_discr
        mm_mesh, orient = import_firedrake_mesh(fdrake_fspace.mesh())
        factory = InterpolatoryQuadratureSimplexGroupFactory(ufl_elt.degree())
        to_discr = Discretization(cl_ctx, mm_mesh, factory)

        # get firedrake unit nodes and map onto meshmode reference element
        group = to_discr.groups[0]
        fd_ref_cell_to_mm = get_affine_reference_simplex_mapping(group.dim,
                                                                 True)
        fd_unit_nodes = get_finat_element_unit_nodes(fdrake_fspace.finat_element)
        fd_unit_nodes = fd_ref_cell_to_mm(fd_unit_nodes)
        # Flipping negative elements corresponds to reordering the nodes.
        # We handle reordering by storing the permutation explicitly as
        # a numpy array

        # Get the reordering fd->mm.
        #
        # One should note there is something a bit more subtle going on
        # in the continuous case. All meshmode discretizations use
        # are discontinuous, so nodes are associated with elements(cells)
        # not vertices. In a continuous firedrake space, some nodes
        # are shared between multiple cells. In particular, while the
        # below "reordering" is indeed a permutation if the firedrake space
        # is discontinuous, if the firedrake space is continuous then
        # some firedrake nodes correspond to nodes on multiple meshmode
        # elements, i.e. those nodes appear multiple times
        # in the "reordering" array
        flip_mat = get_simplex_element_flip_matrix(ufl_elt.degree(),
                                                   fd_unit_nodes)
        fd_cell_node_list = fdrake_fspace.cell_node_list
        _reorder_nodes(orient, fd_cell_node_list, flip_mat, unflip=False)
        mm2fd_node_mapping = fd_cell_node_list.flatten()

        super(FromFiredrakeConnection, self).__init__(to_discr,
                                                      fdrake_fspace,
                                                      mm2fd_node_mapping)
        if fdrake_fspace.ufl_element().family() == 'Discontinuous Lagrange':
            assert len(self._duplicate_nodes) == 0, \
                "Somehow a firedrake node in a 'DG' space got duplicated..." \
                "contact the developer."


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

    from pyop2.datatypes import IntType
    return np.nonzero(cell_is_near_bdy)[0].astype(IntType)


class FromBdyFiredrakeConnection(FiredrakeConnection):
    """
    A connection created from a :mod:`firedrake`
    ``"CG"`` or ``"DG"`` function space which creates a
    meshmode discretization corresponding to all cells with at
    least one vertex on the given boundary and allows
    transfer of functions to and from :mod:`firedrake`.

    Use the same bdy_id as one would for a
    :class:`firedrake.bcs.DirichletBC`.
    ``"on_boundary"`` corresponds to the entire boundary.
    """
    def __init__(self, cl_ctx, fdrake_fspace, bdy_id):
        """
        :arg cl_ctx: A :mod:`pyopencl` computing context
        :arg fdrake_fspace: A :mod:`firedrake` ``"CG"`` or ``"DG"``
            function space (of class :class:`WithGeometry`) built on
            a mesh which is importable by :func:`import_firedrake_mesh`.
        :arg bdy_id: A boundary marker of *fdrake_fspace.mesh()* as accepted by
            the *boundary_nodes* method of a firedrake
            :class:`firedrake.functionspaceimpl.WithGeometry`.
        """
        # Ensure fdrake_fspace is a function space with appropriate reference
        # element.
        from firedrake.functionspaceimpl import WithGeometry
        if not isinstance(fdrake_fspace, WithGeometry):
            raise TypeError(":arg:`fdrake_fspace` must be of firedrake type "
                            ":class:`WithGeometry`, not `%s`."
                            % type(fdrake_fspace))
        ufl_elt = fdrake_fspace.ufl_element()

        if ufl_elt.family() not in ('Lagrange', 'Discontinuous Lagrange'):
            raise ValueError("the ``ufl_element().family()`` of "
                             ":arg:`fdrake_fspace` must "
                             "be ``'Lagrange'`` or "
                             "``'Discontinuous Lagrange'``, not %s."
                             % ufl_elt.family())

        # Create to_discr
        cells_to_use = _compute_cells_near_bdy(fdrake_fspace.mesh(), bdy_id)
        mm_mesh, orient = import_firedrake_mesh(fdrake_fspace.mesh(),
                                                cells_to_use=cells_to_use)
        factory = InterpolatoryQuadratureSimplexGroupFactory(ufl_elt.degree())
        to_discr = Discretization(cl_ctx, mm_mesh, factory)

        # get firedrake unit nodes and map onto meshmode reference element
        group = to_discr.groups[0]
        fd_ref_cell_to_mm = get_affine_reference_simplex_mapping(group.dim,
                                                                 True)
        fd_unit_nodes = get_finat_element_unit_nodes(fdrake_fspace.finat_element)
        fd_unit_nodes = fd_ref_cell_to_mm(fd_unit_nodes)

        # Get the reordering fd->mm, see the note in
        # :class:`FromFiredrakeConnection` for a comment on what this is
        # doing in continuous spaces.
        flip_mat = get_simplex_element_flip_matrix(ufl_elt.degree(),
                                                   fd_unit_nodes)
        fd_cell_node_list = fdrake_fspace.cell_node_list[cells_to_use]
        _reorder_nodes(orient, fd_cell_node_list, flip_mat, unflip=False)
        mm2fd_node_mapping = fd_cell_node_list.flatten()

        super(FromBdyFiredrakeConnection, self).__init__(to_discr,
                                                         fdrake_fspace,
                                                         mm2fd_node_mapping)
        if fdrake_fspace.ufl_element().family() == 'Discontinuous Lagrange':
            assert len(self._duplicate_nodes) == 0, \
                "Somehow a firedrake node in a 'DG' space got duplicated..." \
                "contact the developer."

# }}}


# {{{ Create connection to firedrake from meshmode


class ToFiredrakeConnection(FiredrakeConnection):
    """
    Create a connection from a firedrake discretization
    into firedrake. Create a corresponding "DG" function
    space and allow for conversion back and forth
    by resampling at the nodes.

    .. automethod:: __init__
    """
    def __init__(self, discr, group_nr=None, comm=None):
        """
        :param discr: A :class:`Discretization` to intialize the connection with
        :param group_nr: The group number of the discretization to convert.
            If *None* there must be only one group. The selected group
            must be of type :class:`InterpolatoryQuadratureSimplexElementGroup`.
            The mesh group ``discr.mesh.groups[group_nr]`` must have
            order less than or equal to the order of ``discr.groups[group_nr]``.
        :param comm: Communicator to build a dmplex object on for the created
            firedrake mesh
        """
        if group_nr is None:
            assert len(discr.groups) == 1, ":arg:`group_nr` is *None*, but " \
                    ":arg:`discr` has %s != 1 groups." % len(discr.groups)
            group_nr = 0
        if discr.groups[group_nr].order < discr.mesh.groups[group_nr].order:
            raise ValueError("Discretization group order must be greater than "
                             "or equal to the corresponding mesh group's "
                             "order.")
        el_group = discr.groups[group_nr]

        from firedrake.functionspace import FunctionSpace
        fd_mesh, fd_cell_order, perm2cells = \
            export_mesh_to_firedrake(discr.mesh, group_nr, comm)
        fspace = FunctionSpace(fd_mesh, 'DG', el_group.order)
        # get firedrake unit nodes and map onto meshmode reference element
        dim = fspace.mesh().topological_dimension()
        fd_ref_cell_to_mm = get_affine_reference_simplex_mapping(dim, True)
        fd_unit_nodes = get_finat_element_unit_nodes(fspace.finat_element)
        fd_unit_nodes = fd_ref_cell_to_mm(fd_unit_nodes)

        # **_cell_node holds the node nrs in shape *(ncells, nunit_nodes)*
        fd_cell_node = fspace.cell_node_list
        mm_cell_node = el_group.view(np.arange(discr.nnodes))

        # To get the meshmode to firedrake node assocation, we need to handle
        # local vertex reordering and cell reordering.
        from pyop2.datatypes import IntType
        reordering_arr = np.arange(el_group.nnodes, dtype=IntType)
        for perm, cells in six.iteritems(perm2cells):
            # reordering_arr[i] should be the fd node corresponding to meshmode
            # node i
            #
            # The jth meshmode cell corresponds to the fd_cell_order[j]th
            # firedrake cell. If *nodeperm* is the permutation of local nodes
            # applied to the *j*th meshmode cell, the firedrake node
            # fd_cell_node[fd_cell_order[j]][k] corresponds to the
            # mm_cell_node[j, nodeperm[k]]th meshmode node.
            #
            # Note that the permutation on the unit nodes may not be the
            # same as the permutation on the barycentric coordinates (*perm*).
            # Importantly, the permutation is derived from getting a flip
            # matrix from the Firedrake unit nodes, not necessarily the meshmode
            # unit nodes
            #
            flip_mat = get_simplex_element_flip_matrix(el_group.order,
                                                       fd_unit_nodes,
                                                       np.argsort(perm))
            flip_mat = np.rint(flip_mat).astype(IntType)
            fd_permuted_cell_node = np.matmul(fd_cell_node[fd_cell_order[cells]],
                                              flip_mat.T)
            reordering_arr[mm_cell_node[cells]] = fd_permuted_cell_node

        super(ToFiredrakeConnection, self).__init__(discr,
                                                    fspace,
                                                    reordering_arr,
                                                    group_nr=group_nr)
        assert len(self._duplicate_nodes) == 0, \
            "Somehow a firedrake node in a 'DG' space got duplicated..." \
            "contact the developer."

# }}}

# vim: foldmethod=marker
