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
    :members:
.. autoclass:: ToFiredrakeConnection
    :members:
"""

import numpy as np
import numpy.linalg as la
import six

from modepy import resampling_matrix

from meshmode.interop.firedrake.mesh import (
    import_firedrake_mesh, export_mesh_to_firedrake)
from meshmode.interop.firedrake.reference_cell import (
    get_affine_reference_simplex_mapping, get_finat_element_unit_nodes)

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
    :arg nodes: a *(nelements, nunit_nodes)* or shaped array of nodes
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

        A numpy array of shape *(self.discr.groups[group_nr].nnodes,)*
        whose *i*th entry is the :mod:`firedrake` node index associated
        to the *i*th node in *self.discr.groups[group_nr]*.
        It is important to note that, due to :mod:`meshmode`
        and :mod:`firedrake` using different unit nodes, a :mod:`firedrake`
        node associated to a :mod:`meshmode` may have different coordinates.
        However, after resampling to the other system's unit nodes,
        two associated nodes should have identical coordinates.

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
            A numpy integer array with the same dtype as
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
        if group_nr < 0 or group_nr >= len(discr.groups):
            raise ValueError(":param:`group_nr` has value %s, which an invalid "
                             "index into list *discr.groups* of length %s."
                             % (group_nr, len(discr.gropus)))
        if not isinstance(discr.groups[group_nr],
                          InterpolatoryQuadratureSimplexElementGroup):
            raise TypeError("*discr.groups[group_nr]* must be of type "
                            ":class:`InterpolatoryQuadratureSimplexElementGroup`"
                            ", not :class:`%s`." % type(discr.groups[group_nr]))
        allowed_families = ('Discontinuous Lagrange', 'Lagrange')
        if fdrake_fspace.ufl_element().family() not in allowed_families:
            raise TypeError(":param:`fdrake_fspace` must have ufl family "
                           "be one of %s, not %s."
                            % (allowed_families,
                               fdrake_fspace.ufl_element().family()))
        if mm2fd_node_mapping.shape != (discr.groups[group_nr].nnodes,):
            raise ValueError(":param:`mm2fd_node_mapping` must be of shape ",
                             "(%s,), not %s"
                             % ((discr.groups[group_nr].nnodes,),
                                mm2fd_node_mapping.shape))
        if mm2fd_node_mapping.dtype != fdrake_fspace.cell_node_list.dtype:
            raise ValueError(":param:`mm2fd_node_mapping` must have dtype "
                             "%s, not %s" % (fdrake_fspace.cell_node_list.dtype,
                                             mm2fd_node_mapping.dtype))
        # }}}

        # Get meshmode unit nodes
        element_grp = discr.groups[group_nr]
        mm_unit_nodes = element_grp.unit_nodes
        # get firedrake unit nodes and map onto meshmode reference element
        dim = fdrake_fspace.mesh().topological_dimension()
        fd_ref_cell_to_mm = get_affine_reference_simplex_mapping(dim, True)
        fd_unit_nodes = get_finat_element_unit_nodes(fdrake_fspace.finat_element)
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

    def firedrake_fspace(self, vdim=None):
        """
        Return a firedrake function space that
        *self.discr.groups[self.group_nr]* is connected to
        of the appropriate vector dimension

        :arg vdim: Either *None*, in which case a function space which maps
                   to scalar values is returned, a positive integer *n*,
                   in which case a function space which maps into *\\R^n*
                   is returned, or a tuple of integers defining
                   the shape of values in a tensor function space,
                   in which case a tensor function space is returned
        :return: A :mod:`firedrake` :class:`WithGeometry` which corresponds to
                 *self.discr.groups[self.group_nr]* of the appropriate vector
                 dimension

        :raises TypeError: If *vdim* is of the wrong type
        """
        # Cache the function spaces created to avoid high overhead.
        # Note that firedrake is smart about re-using shared information,
        # so this is not duplicating mesh/reference element information
        if vdim not in self._fspace_cache:
            if vdim is None:
                from firedrake import FunctionSpace
                self._fspace_cache[vdim] = \
                    FunctionSpace(self._mesh_geometry,
                                  self._ufl_element.family(),
                                  degree=self._ufl_element.degree())
            elif isinstance(vdim, int):
                from firedrake import VectorFunctionSpace
                self._fspace_cache[vdim] = \
                    VectorFunctionSpace(self._mesh_geometry,
                                        self._ufl_element.family(),
                                        degree=self._ufl_element.degree(),
                                        dim=vdim)
            elif isinstance(vdim, tuple):
                from firedrake import TensorFunctionSpace
                self._fspace_cache[vdim] = \
                    TensorFunctionSpace(self._mesh_geometry,
                                        self._ufl_element.family(),
                                        degree=self._ufl_element.degree(),
                                        shape=vdim)
            else:
                raise TypeError(":param:`vdim` must be *None*, an integer, "
                                " or a tuple of integers, not of type %s."
                                % type(vdim))
        return self._fspace_cache[vdim]

    def from_firedrake(self, function, out=None):
        """
        transport firedrake function onto :attr:`discr`

        :arg function: A :mod:`firedrake` function to transfer onto
            :attr:`discr`. Its function space must have
            the same family, degree, and mesh as ``self.from_fspace()``.
        :arg out: If *None* then ignored, otherwise a numpy array of the
            shape (i.e.
            *(..., num meshmode nodes)* or *(num meshmode nodes,)* and of the
            same dtype in which *function*'s transported data will be stored

        :return: a numpy array holding the transported function
        """
        # Check function is convertible
        from firedrake.function import Function
        if not isinstance(function, Function):
            raise TypeError(":arg:`function` must be a :mod:`firedrake` "
                            "Function")
        assert function.function_space().ufl_element().family() \
            == self._ufl_element.family() and \
            function.function_space().ufl_element().degree() \
            == self._ufl_element.degree(), \
            ":arg:`function` must live in a function space with the " \
            "same family and degree as ``self.from_fspace()``"
        assert function.function_space().mesh() is self._mesh_geometry, \
            ":arg:`function` mesh must be the same mesh as used by " \
            "``self.from_fspace().mesh()``"

        # Get function data as shape *(nnodes, ...)* or *(nnodes,)*
        function_data = function.dat.data

        # Check that out is supplied correctly, or create out if it is
        # not supplied
        shape = (self.discr.groups[self.group_nr].nnodes,)
        if len(function_data.shape) > 1:
            shape = function_data.shape[1:] + shape
        if out is not None:
            if not isinstance(out, np.ndarray):
                raise TypeError(":param:`out` must of type *np.ndarray* or "
                                "be *None*")
                assert out.shape == shape, \
                    ":param:`out` must have shape %s." % shape
                assert out.dtype == function.dat.data.dtype
        else:
            out = np.ndarray(shape, dtype=function_data.dtype)

        # Reorder nodes
        out[:] = np.moveaxis(function_data, 0, -1)[..., self.mm2fd_node_mapping]
        # Resample at the appropriate nodes
        out_view = self.discr.groups[self.group_nr].view(out)
        np.matmul(out_view, self._resampling_mat_fd2mm.T, out=out_view)
        return out

    def from_meshmode(self, mm_field, out=None,
                      assert_fdrake_discontinuous=True,
                      continuity_tolerance=None):
        """
        transport meshmode field from :attr:`discr` into an
        appropriate firedrake function space.

        :arg mm_field: A numpy array of shape *(nnodes,)* or *(..., nnodes)*
            representing a function on :attr:`to_distr`.
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
            apart. If *None*, no checks are performed.

        :return: a :mod:`firedrake` :class:`Function` holding the transported
            data.
        """
        if self._ufl_element.family() == 'Lagrange' \
                and assert_fdrake_discontinuous:
            raise ValueError("Trying to convert to continuous function space "
                             " with :arg:`assert_fdrake_discontinuous` set "
                             " to *True*")
        # make sure out is a firedrake function in an appropriate
        # function space
        if out is not None:
            from firedrake.function import Function
            assert isinstance(out, Function), \
                ":arg:`out` must be a :mod:`firedrake` Function or *None*"
            assert out.function_space().ufl_element().family() \
                == self._ufl_element.family() and \
                out.function_space().ufl_element().degree() \
                == self._ufl_element.degree(), \
                ":arg:`out` must live in a function space with the " \
                "same family and degree as ``self.firedrake_fspace()``"
            assert out.function_space().mesh() is self._mesh_geometry, \
                ":arg:`out` mesh must be the same mesh as used by " \
                "``self.firedrake_fspace().mesh()`` or *None*"
        else:
            if len(mm_field.shape) == 1:
                vdim = None
            elif len(mm_field.shape) == 2:
                vdim = mm_field.shape[0]
            else:
                vdim = mm_field.shape[:-1]
            out = Function(self.firedrake_fspace(vdim))

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
            must be of type :class:`InterpolatoryQuadratureSimplexElementGroup`
        :param comm: Communicator to build a dmplex object on for the created
            firedrake mesh
        """
        if group_nr is None:
            assert len(discr.groups) == 1, ":arg:`group_nr` is *None*, but " \
                    ":arg:`discr` has %s != 1 groups." % len(discr.groups)
            group_nr = 0
        el_group = discr.groups[group_nr]

        from firedrake.functionspace import FunctionSpace
        fd_mesh = export_mesh_to_firedrake(discr.mesh, group_nr, comm)
        fspace = FunctionSpace(fd_mesh, 'DG', el_group.order)
        super(ToFiredrakeConnection, self).__init__(discr,
                                                    fspace,
                                                    np.arange(el_group.nnodes),
                                                    group_nr=group_nr)

# }}}

# vim: foldmethod=marker
