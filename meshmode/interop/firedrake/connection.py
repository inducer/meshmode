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
.. autoclass:: FromFiredrakeConnection
.. autoclass:: FromBoundaryFiredrakeConnection
.. autoclass:: ToFiredrakeConnection
"""

import numpy as np
import numpy.linalg as la

from modepy import resampling_matrix

from meshmode.interop.firedrake.mesh import (
    import_firedrake_mesh, export_mesh_to_firedrake)
from meshmode.interop.firedrake.reference_cell import (
    get_affine_reference_simplex_mapping, get_finat_element_unit_nodes)

from meshmode.mesh.processing import get_simplex_element_flip_matrix

from meshmode.discretization.poly_element import (
    InterpolatoryQuadratureSimplexGroupFactory,
    PolynomialRecursiveNodesGroupFactory,
    ElementGroupFactory)
from meshmode.discretization import (
    Discretization, InterpolatoryElementGroupBase)

from pytools import memoize_method


def _reorder_nodes(orient, nodes, flip_matrix, unflip=False):
    """
    return a flipped copy of *nodes* according to *orient*

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
    flipped_nodes = np.copy(nodes)
    flipped_nodes[orient < 0] = np.einsum(
        "ij,ej->ei",
        flip_mat, nodes[orient < 0])

    return flipped_nodes


# {{{ Most basic connection between a fd function space and mm discretization

class FiredrakeConnection:
    """
    A connection between one group of
    a meshmode discretization and a firedrake "CG" or "DG"
    function space.

    Users should instantiate this using a
    :class:`FromFiredrakeConnection` or :class:`ToFiredrakeConnection`.

    .. attribute:: discr

        A :class:`meshmode.discretization.Discretization`.

    .. attribute:: group_nr

        The group number identifying which element group of
        :attr:`discr` is being connected to a firedrake function space

    .. attribute:: mm2fd_node_mapping

        Letting *element_grp = self.discr.groups[self.group_nr]*,
        *mm2fd_node_mapping* is a numpy array of shape
        *(element_grp.nelements, element_grp.nunit_dofs)*
        whose *(i, j)*th entry is the :mod:`firedrake` node
        index associated to the *j*th degree of freedom of the
        *i*th element in *element_grp*.

        Degrees of freedom should be associated so that
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
        :param discr: A :class:`meshmode.discretization.Discretization`
        :param fdrake_fspace: A
            :class:`firedrake.functionspaceimpl.WithGeometry`.
            Must have ufl family ``'Lagrange'`` or
            ``'Discontinuous Lagrange'``.
        :param mm2fd_node_mapping: Used as attribute :attr:`mm2fd_node_mapping`.
            A 2-D numpy integer array with the same dtype as
            ``fdrake_fspace.cell_node_list.dtype``
        :param group_nr: The index of the group in *discr* which is
            being connected to *fdrake_fspace*. The group must be a
            :class:`~meshmode.discretization.InterpolatoryElementGroupBase`
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
            raise TypeError("'discr' must be of type "
                            "meshmode.discretization.Discretization, "
                            "not '%s'`." % type(discr))
        from firedrake.functionspaceimpl import WithGeometry
        if not isinstance(fdrake_fspace, WithGeometry):
            raise TypeError("'fdrake_fspace' must be of type "
                            "firedrake.functionspaceimpl.WithGeometry, "
                            "not '%s'." % type(fdrake_fspace))
        if not isinstance(mm2fd_node_mapping, np.ndarray):
            raise TypeError("'mm2fd_node_mapping' must be of type "
                            "numpy.ndarray, "
                            "not '%s'." % type(mm2fd_node_mapping))
        if not isinstance(group_nr, int) and group_nr is not None:
            raise TypeError("'group_nr' must be of type int or be "
                            "*None*, not of type '%s'." % type(group_nr))
        # Convert group_nr to an integer if *None*
        if group_nr is None:
            if len(discr.groups) != 1:
                raise ValueError("'group_nr' is *None* but 'discr' "
                                 "has '%s' != 1 groups." % len(discr.groups))
            group_nr = 0
        # store element_grp as variable for convenience
        element_grp = discr.groups[group_nr]

        if group_nr < 0 or group_nr >= len(discr.groups):
            raise ValueError("'group_nr' has value '%s', which an invalid "
                             "index into list 'discr.groups' of length '%s'."
                             % (group_nr, len(discr.groups)))
        if not isinstance(element_grp, InterpolatoryElementGroupBase):
            raise TypeError("'discr.groups[group_nr]' must be of type "
                            "InterpolatoryElementGroupBase"
                            ", not '%s'." % type(element_grp))
        allowed_families = ('Discontinuous Lagrange', 'Lagrange')
        if fdrake_fspace.ufl_element().family() not in allowed_families:
            raise TypeError("'fdrake_fspace' must have ufl family "
                           "be one of %s, not '%s'."
                            % (allowed_families,
                               fdrake_fspace.ufl_element().family()))
        if mm2fd_node_mapping.shape != (element_grp.nelements,
                                        element_grp.nunit_dofs):
            raise ValueError("'mm2fd_node_mapping' must be of shape ",
                             "(%s,), not '%s'"
                             % ((discr.groups[group_nr].ndofs,),
                                mm2fd_node_mapping.shape))
        if mm2fd_node_mapping.dtype != fdrake_fspace.cell_node_list.dtype:
            raise ValueError("'mm2fd_node_mapping' must have dtype "
                             "%s, not '%s'" % (fdrake_fspace.cell_node_list.dtype,
                                             mm2fd_node_mapping.dtype))
        # }}}

        # Get meshmode unit nodes
        mm_unit_nodes = element_grp.unit_nodes
        # get firedrake unit nodes and map onto meshmode reference element
        tdim = fdrake_fspace.mesh().topological_dimension()
        fd_ref_cell_to_mm = get_affine_reference_simplex_mapping(tdim, True)
        fd_unit_nodes = get_finat_element_unit_nodes(fdrake_fspace.finat_element)
        fd_unit_nodes = fd_ref_cell_to_mm(fd_unit_nodes)

        # compute and store resampling matrices
        self._resampling_mat_fd2mm = resampling_matrix(element_grp.basis(),
                                                       new_nodes=mm_unit_nodes,
                                                       old_nodes=fd_unit_nodes)
        self._resampling_mat_mm2fd = resampling_matrix(element_grp.basis(),
                                                       new_nodes=fd_unit_nodes,
                                                       old_nodes=mm_unit_nodes)

        # {{{ compute meshmode node equivalence classes for continuity check

        # Now handle the possibility of multiple meshmode nodes being associated
        # to the same firedrake node
        unique_fd_nodes, counts = np.unique(mm2fd_node_mapping,
                                            return_counts=True)
        # map firedrake nodes associated to more than 1 meshmode node
        # to all associated meshmode nodes.
        # fd node index -> (meshmode cell index, meshmode local node index)
        fd_to_dupes = {}
        dup_fd_nodes = set(unique_fd_nodes[counts > 1])
        for icell, cell in enumerate(mm2fd_node_mapping):
            for local_mm_inode, fd_inode in enumerate(cell):
                if fd_inode in dup_fd_nodes:
                    fd_to_dupes.setdefault(fd_inode, [])
                    fd_to_dupes[fd_inode] = (icell, local_mm_inode)
        # A list of tuples, each tuple represents an equivalence class
        # of meshmode nodes
        # (represented as tuples
        #  *(meshmode_cell_index, meshmode local node index)*)
        # which associate to the same firedrake node under
        # *mm2fd_node_mapping*. Only equivalence classes of size > 1
        # are included.
        self._mm_node_equiv_classes = [tuple(equiv_class) for equiv_class
                                       in fd_to_dupes.values()]

        # }}}

        # Store input
        self.discr = discr
        self.group_nr = group_nr
        self.mm2fd_node_mapping = mm2fd_node_mapping
        self._mesh_geometry = fdrake_fspace.mesh()
        self._ufl_element = fdrake_fspace.ufl_element()

    @memoize_method
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
        :return: A :class:`firedrake.functionspaceimpl.WithGeometry` which
                corresponds to *self.discr.groups[self.group_nr]* of the appropriate
                vector dimension

        :raises TypeError: If *shape* is of the wrong type
        """
        if shape is None:
            from firedrake import FunctionSpace
            return FunctionSpace(self._mesh_geometry,
                                 self._ufl_element.family(),
                                 degree=self._ufl_element.degree())
        elif isinstance(shape, int):
            from firedrake import VectorFunctionSpace
            return VectorFunctionSpace(self._mesh_geometry,
                                       self._ufl_element.family(),
                                       degree=self._ufl_element.degree(),
                                       dim=shape)
        elif isinstance(shape, tuple):
            from firedrake import TensorFunctionSpace
            return TensorFunctionSpace(self._mesh_geometry,
                                       self._ufl_element.family(),
                                       degree=self._ufl_element.degree(),
                                       shape=shape)
        else:
            raise TypeError("'shape' must be *None*, an integer, "
                            " or a tuple of integers, not of type '%s'."
                            % type(shape))

    def _validate_function(self, function, function_name,
                           shape=None, dtype=None):
        """
        Handy helper function to validate that a firedrake function
        is convertible (or can be the recipient of a conversion).
        Raises error messages if wrong types, shape, dtype found
        etc.
        """
        # Validate that *function* is convertible
        from firedrake.function import Function
        if not isinstance(function, Function):
            raise TypeError(f"'{function_name} must be a firedrake Function"
                            f" but is of unexpected type '{type(function)}'")
        ufl_elt = function.function_space().ufl_element()
        if ufl_elt.family() != self._ufl_element.family():
            raise ValueError(f"'{function_name}.function_space().ufl_element()"
                             f".family()' must be {self._ufl_element.family()}"
                             f", not '{type(ufl_elt.family())}'")
        if ufl_elt.degree() != self._ufl_element.degree():
            raise ValueError(f"'{function_name}.function_space().ufl_element()"
                             f".degree()' must be {self._ufl_element.degree()}"
                             f", not '{ufl_elt.degree()}'")
        if function.function_space().mesh() is not self._mesh_geometry:
            raise ValueError(f"'{function_name}.function_space().mesh()' must"
                             " be the same mesh used by this connection")
        if dtype is not None and function.dat.data.dtype != dtype:
            raise ValueError(f"'{function_name}.dat.dtype' must be "
                             f"{dtype}, not '{function.dat.data.dtype}'")
        if shape is not None and function.function_space().shape != shape:
            raise ValueError("'{function_name}.function_space().shape' must be"
                             " {shape}, not '{function.function_space().shape}"
                             "'")

    def _validate_field(self, field, field_name, shape=None, dtype=None):
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
                raise TypeError(f"'{arr_name}' must be of type "
                                f"meshmode.dof_array.DOFArray, not "
                                f"{type(arr)}")
            if arr.array_context is None:
                raise ValueError(f"'{arr_name}' must have a non-*None* "
                                 "array_context")
            if arr.shape != tuple([len(self.discr.groups)]):
                raise ValueError(f"'{arr_name}' shape must be "
                                 f"{tuple([len(self.discr.groups)])}, not "
                                 f"'{arr.shape}'")
            if arr[self.group_nr].shape != group_shape:
                raise ValueError(f"'{arr_name}[{self.group_nr}].shape' must be"
                                 f" {group_shape}, not "
                                 f"'{arr[self.group_nr].shape}'")
            if dtype is not None and arr.entry_dtype != dtype:
                raise ValueError(f"'{arr_name}.entry_dtype' must be {dtype},"
                                 f" not '{arr.entry_dtype}'")

        if isinstance(field, DOFArray):
            if shape is not None and shape != ():
                raise ValueError("shape != () and '%s' is of type DOFArray"
                                 " instead of np.ndarray." % field_name)
            check_dof_array(field, field_name)
        elif isinstance(field, np.ndarray):
            if shape is not None and field.shape != shape:
                raise ValueError(f"'{field_name}.shape' must be {shape}, not "
                                 f"'{field.shape}'")
            for i, arr in enumerate(field.flatten()):
                arr_name = "%s[%s]" % (field_name, np.unravel_index(i, field.shape))
                try:
                    check_dof_array(arr, arr_name)
                except TypeError as type_err:
                    msg = type_err.args[0]
                    prefix = f"{field_name} is a numpy array of shape " \
                        f"{field.shape}, which is interpreted as a mapping" \
                        f" into a space of sahpe {field.shape}. For each " \
                        " multi-index *mi*, the *mi*th coordinate values " \
                        f" of {field_name} should be represented as a " \
                        f"DOFArray stored in {field_name}[mi]. If you are " \
                        " not trying to represent a mapping into a space of " \
                        f" shape {field.shape}, look at the documentation " \
                        " for FiredrakeConnection.from_meshmode or " \
                        "FiredrakeConnection.from_firedrake to see how " \
                        "fields in a discretization are represented."
                    raise TypeError(prefix + "\n" + msg)
        else:
            raise TypeError("field must be of type DOFArray or a numpy object array "
                            "of those not %s." % type(field))

    def from_firedrake(self, function, out=None, actx=None):
        """
        Transport firedrake function onto :attr:`discr`

        :arg function: A :class:`firedrake.function.Function` to transfer onto
            :attr:`discr`. Its function space must have
            the same family, degree, and mesh as ``self.from_fspace()``.
        :arg out: Either

            1. A :class:`~meshmode.dof_array.DOFArray`
            2. A :class:`numpy.ndarray` object array, each of whose
               entries is a :class:`~meshmode.dof_array.DOFArray`
            3. *None*

            In the case of (1.), *function* must be in a
            scalar function space
            (i.e. `function.function_space().shape == (,)`).
            In the case of (2.), the shape of *out* must match
            `function.function_space().shape`.

            In either case, each :class:`~meshmode.dof_array.DOFArray`
            must be a :class:`~meshmode.dof_array.DOFArray`
            defined on :attr:`discr` (as described in
            the documentation for :class:`~meshmode.dof_array.DOFArray`).
            Also, each :class:`~meshmode.dof_array.DOFArray`'s *entry_dtype* must
            match the *function.dat.data.dtype*, and be of shape
            *(nelements, nunit_dofs)*.

            In case (3.), an array is created satisfying
            the above requirements.

            The data in *function* is transported to :attr:`discr`
            and stored in *out*, which is then returned.
        :arg actx:
            * If *out* is *None*, then *actx* is a
              :class:`~meshmode.array_context.ArrayContext` on which
              to create the :class:`~meshmode.dof_array.DOFArray`
            * If *out* is not *None*, *actx* must be *None* or *out*'s
              :attr:`~meshmode.dof_array.DOFArray.array_context`.

        :return: *out*, with the converted data from *function* stored in it
        """
        self._validate_function(function, "function")
        # get function data and shape of values
        function_data = function.dat.data
        fspace_shape = function.function_space().shape

        # Handle :arg:`out`
        if out is not None:
            self._validate_field(out, "out", fspace_shape, function_data.dtype)
            # If out is supplied, check type, shape, and dtype
            if actx not in (None, out.array_context):
                raise ValueError("If 'out' is not *None*, 'actx' must be"
                                 " *None* or 'out.array_context'")
        else:
            # If 'out' is not supplied, create it
            from meshmode.array_context import ArrayContext
            if not isinstance(actx, ArrayContext):
                raise TypeError("If 'out' is *None*, 'actx' must be of type "
                                "ArrayContext, not '%s'." % type(actx))
            if fspace_shape == ():
                out = self.discr.zeros(actx, dtype=function_data.dtype)
            else:
                out = np.ndarray(fspace_shape, dtype=np.object)
                for multi_index in np.ndindex(fspace_shape):
                    out[multi_index] = \
                        self.discr.zeros(actx, dtype=function_data.dtype)

        def reorder_and_resample(dof_array, fd_data):
            # put the firedrake data in meshmode order and then resample,
            # storing in dof_array
            dof_array[self.group_nr].set(
                    np.matmul(fd_data[self.mm2fd_node_mapping],
                              self._resampling_mat_fd2mm.T)
                    )

        # If scalar, just reorder and resample out
        if fspace_shape == ():
            reorder_and_resample(out, function_data)
        else:
            # firedrake drops extra dimensions
            if len(function_data.shape) != 1 + len(fspace_shape):
                shape = (function_data.shape[0],) + fspace_shape
                function_data = function_data.reshape(shape)
            # otherwise, have to grab each dofarray and the corresponding
            # data from *function_data*
            for multi_index in np.ndindex(fspace_shape):
                dof_array = out[multi_index]
                index = (np.s_[:],) + multi_index
                fd_data = function_data[index]
                reorder_and_resample(dof_array, fd_data)

        return out

    def from_meshmode(self, mm_field, out=None,
                      assert_fdrake_discontinuous=True,
                      continuity_tolerance=None):
        r"""
        Transport meshmode field from :attr:`discr` into an
        appropriate firedrake function space.

        If *out* is *None*, values at any firedrake
        nodes associated to NO meshmode nodes are zeroed out.
        If *out* is supplied, values at nodes associated to NO meshmode nodes
        are not modified.

        :arg mm_field: Either

            * A :class:`~meshmode.dof_array.DOFArray` representing
              a field of shape *tuple()* on :attr:`discr`
            * A :class:`numpy.ndarray` of dtype "object" with
              entries of class :class:`~meshmode.dof_array.DOFArray`
              representing a field of shape *mm_field.shape*
              on :attr:`discr`

            See :class:`~meshmode.dof.DOFArray` for further requirements.
            The :attr:`group_nr` entry of each
            :class:`~meshmode.dof_array.DOFArray`
            must be of shape *(nelements, nunit_dofs)* and
            the *element_dtype* must match that used for
            :class:`firedrake.function.Function`\ s

        :arg out: If *None* then ignored, otherwise a
            :class:`firedrake.function.Function`
            of the right function space for the transported data
            to be stored in. The shape of its function space must
            match the shape of *mm_field*

        :arg assert_fdrake_discontinuous: If *True*,
            disallows conversion to a continuous firedrake function space
            (i.e. this function checks that ``self.firedrake_fspace()`` is
            discontinuous and raises a *ValueError* otherwise)

        :arg continuity_tolerance: If converting to a continuous firedrake
            function space (i.e. if ``self.firedrake_fspace()`` is continuous),
            assert that at any two meshmode nodes corresponding to the
            same firedrake node (meshmode is a discontinuous space, so this
            situation will almost certainly happen), the function being transported
            has values less than *continuity_tolerance* distance (in
            :math:`\\ell^\\infty` distance)
            apart. If *None*, no checks are performed. Does nothing if
            the firedrake function space is discontinuous

        :return: a :class:`firedrake.function.Function` holding the transported
            data (*out*, if *out* was not *None*)
        """
        if self._ufl_element.family() == 'Lagrange' \
                and assert_fdrake_discontinuous:
            raise ValueError("Trying to convert to continuous function space "
                             " with 'assert_fdrake_discontinuous' set "
                             " to *True*")
        # All firedrake functions are the same dtype
        dtype = self.firedrake_fspace().mesh().coordinates.dat.data.dtype
        self._validate_field(mm_field, "mm_field", dtype=dtype)

        # get the shape of mm_field
        from meshmode.dof_array import DOFArray
        if not isinstance(mm_field, DOFArray):
            fspace_shape = mm_field.shape
        else:
            fspace_shape = ()

        # make sure out is a firedrake function in an appropriate
        # function space
        if out is not None:
            self._validate_function(out, "out", fspace_shape, dtype)
        else:
            from firedrake.function import Function
            # Translate shape so that don't always get a TensorFunctionSpace,
            # but instead get FunctionSpace or VectorFunctionSpace when
            # reasonable
            shape = fspace_shape
            if shape == ():
                shape = None
            elif len(shape) == 1:
                shape = shape[0]
            # make a function filled with zeros
            out = Function(self.firedrake_fspace(shape))
            out.dat.data[:] = 0.0

        out_data = out.dat.data
        # Handle firedrake dropping dimensions
        if len(out.dat.data.shape) != 1 + len(fspace_shape):
            shape = (out.dat.data.shape[0],) + fspace_shape
            out_data = out_data.reshape(shape)

        def resample_and_reorder(fd_data, dof_array):
            # pull data into numpy
            dof_np = dof_array.array_context.to_numpy(dof_array[self.group_nr])
            # resample the data data (keep this for continuity checks)
            resampled_data = np.matmul(dof_np, self._resampling_mat_mm2fd.T)
            # store resampled data in firedrake ordering
            fd_data[self.mm2fd_node_mapping] = resampled_data

            # Continuity checks if requested
            if self._ufl_element.family() == 'Lagrange' \
                    and continuity_tolerance is not None:
                assert isinstance(continuity_tolerance, float)
                assert continuity_tolerance >= 0
                # Make sure not to compare using mm_field,
                # because two meshmode nodes associated to the same firedrake
                # nodes may have been resampled to distinct nodes on different
                # elements.
                for mm_equiv_class in self._mm_node_equiv_classes:
                    l_inf = np.ptp(resampled_data[mm_equiv_class])
                    if l_inf >= continuity_tolerance:
                        fd_inode = self.mm2fd_node_mapping[mm_equiv_class[0]]
                        raise ValueError("Meshmode nodes %s (written as "
                                         " *(element index, local node index)*)"
                                         " represent the same firedrake node %s"
                                         ", but 'mm_field's resampled "
                                         " values are %s > %s apart)"
                                         % (mm_equiv_class, fd_inode,
                                            l_inf, continuity_tolerance))

        # If scalar, just reorder and resample out
        if fspace_shape == ():
            resample_and_reorder(out_data, mm_field)
        else:
            # otherwise, have to grab each dofarray and the corresponding
            # data from *function_data*
            for multi_index in np.ndindex(fspace_shape):
                # have to be careful to take view and not copy
                index = (np.s_[:],) + multi_index
                fd_data = out_data[index]
                dof_array = mm_field[multi_index]
                resample_and_reorder(fd_data, dof_array)

        return out

# }}}


# {{{ Create connection from firedrake into meshmode


class FromFiredrakeConnection(FiredrakeConnection):
    """
    A connection created from a :mod:`firedrake`
    ``"CG"`` or ``"DG"`` function space which creates a corresponding
    meshmode discretization and allows
    transfer of functions to and from :mod:`firedrake`.

    .. automethod:: __init__
    """
    def __init__(self, actx, fdrake_fspace, grp_factory=None):
        """
        :arg actx: A :class:`~meshmode.array_context.ArrayContext`
        :arg fdrake_fspace: A :mod:`firedrake` ``"CG"`` or ``"DG"``
            function space (of class
            :class:`~firedrake.functionspaceimpl.WithGeometry`) built on
            a mesh which is importable by
            :func:`~meshmode.interop.firedrake.mesh.import_firedrake_mesh`.
        :arg grp_factory: (optional) If not *None*, should be
            a :class:`~meshmode.discretization.poly_element.ElementGroupFactory`
            whose group class is a subclass of
            :class:`~meshmode.discretization.InterpolatoryElementGroupBase`.
            If *None*, uses

            * A :class:`~meshmode.discretization.poly_element.\
PolynomialRecursiveNodesGroupFactory`
            * A :class:`~meshmode.discretization.poly_element.\
PolynomialWarpAndBlendGroupFactory`
        """
        # Ensure fdrake_fspace is a function space with appropriate reference
        # element.
        from firedrake.functionspaceimpl import WithGeometry
        if not isinstance(fdrake_fspace, WithGeometry):
            raise TypeError("'fdrake_fspace' must be of firedrake type "
                            "WithGeometry, not '%s'."
                            % type(fdrake_fspace))
        ufl_elt = fdrake_fspace.ufl_element()

        if ufl_elt.family() not in ('Lagrange', 'Discontinuous Lagrange'):
            raise ValueError("the 'fdrake_fspace.ufl_element().family()' of "
                             "must be be 'Lagrange' or "
                             "'Discontinuous Lagrange', not '%s'."
                             % ufl_elt.family())
        # Make sure grp_factory is the right type if provided, and
        # uses an interpolatory class.
        if grp_factory is not None:
            if not isinstance(grp_factory, ElementGroupFactory):
                raise TypeError("'grp_factory' must inherit from "
                                "meshmode.discretization.ElementGroupFactory,"
                                "but is instead of type "
                                "'%s'." % type(grp_factory))
                if not issubclass(grp_factory.group_class,
                                  InterpolatoryElementGroupBase):
                    raise TypeError("'grp_factory.group_class' must inherit from"
                                    "meshmode.discretization."
                                    "InterpolatoryElementGroupBase, but"
                                    " is instead of type '%s'"
                                    % type(grp_factory.group_class))
        # If not provided, make one
        else:
            degree = ufl_elt.degree()
            try:
                import recursivenodes  # noqa : F401
                family = 'lgl'  # L-G-Legendre
                grp_factory = PolynomialRecursiveNodesGroupFactory(degree, family)
            except ImportError:
                grp_factory = InterpolatoryQuadratureSimplexGroupFactory(degree)

        # In case this class is really a FromBoundaryFiredrakeConnection,
        # get *cells_to_use*
        cells_to_use = self._get_cells_to_use(fdrake_fspace.mesh())
        # Create to_discr
        mm_mesh, orient = import_firedrake_mesh(fdrake_fspace.mesh(),
                                                cells_to_use=cells_to_use)
        to_discr = Discretization(actx, mm_mesh, grp_factory)

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
        if cells_to_use is not None:
            fd_cell_node_list = fd_cell_node_list[cells_to_use]
        # flip fd_cell_node_list
        flipped_cell_node_list = _reorder_nodes(orient,
                                                fd_cell_node_list,
                                                flip_mat,
                                                unflip=False)

        super(FromFiredrakeConnection, self).__init__(to_discr,
                                                      fdrake_fspace,
                                                      flipped_cell_node_list)
        if fdrake_fspace.ufl_element().family() == 'Discontinuous Lagrange':
            assert len(self._mm_node_equiv_classes) == 0, \
                "A firedrake node in a 'DG' space got duplicated"

    def _get_cells_to_use(self, mesh):
        """
        For compatibility with :class:`FromFiredrakeBdyConnection`
        """
        return None


class FromBoundaryFiredrakeConnection(FromFiredrakeConnection):
    """
    A connection created from a :mod:`firedrake`
    ``"CG"`` or ``"DG"`` function space which creates a
    meshmode discretization corresponding to all cells with at
    least one vertex on the given boundary and allows
    transfer of functions to and from :mod:`firedrake`.

    Use the same bdy_id as one would for a
    :class:`firedrake.bcs.DirichletBC` instance.
    ``"on_boundary"`` corresponds to the entire boundary.

    .. attribute:: bdy_id

        the boundary id of the boundary being connecting from

    .. automethod:: __init__
    """
    def __init__(self, actx, fdrake_fspace, bdy_id, grp_factory=None):
        """
        :arg bdy_id: A boundary marker of *fdrake_fspace.mesh()* as accepted by
            the *boundary_nodes* method of a
            :class:`firedrake.functionspaceimpl.WithGeometry`.

        Other arguments are as in
        :class:`~meshmode.interop.firedrake.connection.FromFiredrakeConnection`.
        """
        self.bdy_id = bdy_id
        super(FromBoundaryFiredrakeConnection, self).__init__(
            actx, fdrake_fspace, grp_factory=grp_factory)

    def _get_cells_to_use(self, mesh):
        """
        Returns an array of the cell ids with >= 1 vertex on the
        given bdy_id
        """
        cfspace = mesh.coordinates.function_space()
        cell_node_list = cfspace.cell_node_list

        boundary_nodes = cfspace.boundary_nodes(self.bdy_id, 'topological')
        # Reduce along each cell: Is a vertex of the cell in boundary nodes?
        cell_is_near_bdy = np.any(np.isin(cell_node_list, boundary_nodes), axis=1)

        from pyop2.datatypes import IntType
        return np.nonzero(cell_is_near_bdy)[0].astype(IntType)

# }}}


# {{{ Create connection to firedrake from meshmode


class ToFiredrakeConnection(FiredrakeConnection):
    """
    Create a connection from a meshmode discretization
    into firedrake. Create a corresponding "DG" function
    space and allow for conversion back and forth
    by resampling at the nodes.

    .. automethod:: __init__
    """
    def __init__(self, discr, group_nr=None, comm=None):
        """
        :param discr: A :class:`~meshmode.discretization.Discretization`
            to intialize the connection with
        :param group_nr: The group number of the discretization to convert.
            If *None* there must be only one group. The selected group
            must be of type
            :class:`~meshmode.discretization.poly_element.\
InterpolatoryQuadratureSimplexElementGroup`.

        :param comm: Communicator to build a dmplex object on for the created
            firedrake mesh
        """
        if group_nr is None:
            if len(discr.groups) != 1:
                raise ValueError("'group_nr' is *None*, but 'discr' has '%s' "
                                 "!= 1 groups." % len(discr.groups))
            group_nr = 0
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

        # To get the meshmode to firedrake node assocation, we need to handle
        # local vertex reordering and cell reordering.
        from pyop2.datatypes import IntType
        mm2fd_node_mapping = np.ndarray((el_group.nelements, el_group.nunit_dofs),
                                        dtype=IntType)
        for perm, cells in perm2cells.items():
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
            mm2fd_node_mapping[cells] = fd_permuted_cell_node

        super(ToFiredrakeConnection, self).__init__(discr,
                                                    fspace,
                                                    mm2fd_node_mapping,
                                                    group_nr=group_nr)
        assert len(self._mm_node_equiv_classes) == 0, \
            "A firedrake node in a 'DG' space got duplicated"

# }}}

# vim: foldmethod=marker
