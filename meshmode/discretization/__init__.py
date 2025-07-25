from __future__ import annotations


__copyright__ = """
Copyright (C) 2013-2021 Andreas Kloeckner
Copyright (C) 2021 University of Illinois Board of Trustees
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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable
from warnings import warn

import numpy as np
from typing_extensions import override

import loopy as lp
import modepy as mp
import pytools.obj_array as obj_array
from arraycontext import ArrayContext, make_loopy_program, tag_axes
from pytools import keyed_memoize_in, memoize_in, memoize_method

# underscored because it shouldn't be imported from here.
from meshmode.dof_array import DOFArray as _DOFArray
from meshmode.transform_metadata import (
    ConcurrentDOFInameTag,
    ConcurrentElementInameTag,
    DiscretizationDOFAxisTag,
    DiscretizationElementAxisTag,
    FirstAxisIsElementsTag,
)


if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Sequence

    from numpy.typing import DTypeLike

    from pytools.obj_array import ObjectArray

    from meshmode.mesh import Mesh as _Mesh, MeshElementGroup as _MeshElementGroup


__doc__ = """
Error handling
--------------

.. autoexception:: ElementGroupTypeError
.. autoexception:: NoninterpolatoryElementGroupError

Base classes
------------

.. autoclass:: ElementGroupBase
.. autoclass:: ElementGroupFactory

.. autoclass:: NodalElementGroupBase
.. autoclass:: ElementGroupWithBasis
.. autoclass:: InterpolatoryElementGroupBase
.. autoclass:: ModalElementGroupBase

Discretization class
--------------------

.. autofunction:: num_reference_derivative
.. autoclass:: Discretization

References
----------

.. class:: ObjectArray

    See :class:`pytools.obj_array.ObjectArray`.
"""


class ElementGroupTypeError(TypeError):
    """A :class:`TypeError` specific for handling element
    groups. This exception may be raised to indicate
    whenever an improper operation or function is applied
    to a particular subclass of :class:`~ElementGroupBase`.
    """


class NoninterpolatoryElementGroupError(ElementGroupTypeError):
    """A specialized :class:`~ElementGroupTypeError` that may
    be raised whenever non-interpolatory element groups
    are being used for interpolation.
    """


# {{{ element group base

class ElementGroupBase(ABC):
    """Defines a discrete function space on a homogeneous
    (in terms of element type and order) subset of a :class:`Discretization`.
    These correspond one-to-one with :class:`meshmode.mesh.MeshElementGroup`.
    Responsible for all bulk data handling in :class:`Discretization`.

    .. autoattribute :: mesh_el_group
    .. autoattribute :: order

    .. autoattribute:: is_affine
    .. autoattribute:: nelements
    .. autoattribute:: nunit_dofs
    .. autoattribute:: ndofs
    .. autoattribute:: dim
    .. autoattribute:: shape
    .. autoattribute:: space

    .. automethod:: __init__
    .. automethod:: discretization_key
    """

    mesh_el_group: _MeshElementGroup
    order: int

    def __init__(self,
                 mesh_el_group: _MeshElementGroup,
                 order: int,
                 ) -> None:
        self.mesh_el_group = mesh_el_group
        self.order = order

    @property
    def is_affine(self) -> bool:
        """A :class:`bool` flag that is *True* if the local-to-global
        parametrization of all the elements in the group is affine. Based on
        :attr:`meshmode.mesh.MeshElementGroup.is_affine`.
        """
        return self.mesh_el_group.is_affine

    @property
    def nelements(self) -> int:
        """The total number of polygonal elements in the
        :class:`meshmode.mesh.MeshElementGroup`.
        """
        return self.mesh_el_group.nelements

    @property
    @abstractmethod
    def nunit_dofs(self) -> int:
        """The number of degrees of freedom ("DOFs")
        associated with a single element.
        """

    @property
    def ndofs(self) -> int:
        """The total number of degrees of freedom ("DOFs")
        associated with the entire element group.
        """
        return self.nunit_dofs * self.nelements

    @property
    def dim(self) -> int:
        """The number of spatial dimensions in which the functions
        in :attr:`~space` operate.
        """
        return self.mesh_el_group.dim

    @property
    @abstractmethod
    def shape(self) -> mp.Shape:
        """Returns a subclass of :class:`modepy.Shape` representing
        the reference element defining the element group.
        """

    @property
    @abstractmethod
    def space(self) -> mp.FunctionSpace:
        """Returns a :class:`modepy.FunctionSpace` representing
        the underlying polynomial space defined on the element
        group's reference element.
        """

    def discretization_key(self) -> Sequence[Hashable]:
        """Return a hashable, equality-comparable object that fully describes
        the per-element discretization used by this element group. (This
        should cover all parts of the
        `Ciarlet Triple <https://finite-element.github.io/L2_fespaces.html>`__:
        reference element, shape functions, and the linear functionals defining
        the degrees of freedom.) The object should be independent, however, of
        the (global) elements that make up the group.

        The structure of the element is not specified, but it must be globally
        unique to this element group.
        """
        return (type(self), self.dim, self.order)


@runtime_checkable
class ElementGroupFactory(Protocol):
    """A :class:`typing.Protocol` specifying the interface for group factories.

    .. automethod:: __call__
    """

    def __call__(self, mesh_el_group: _MeshElementGroup, /) -> ElementGroupBase:
        """Create a new :class:`~meshmode.discretization.ElementGroupBase`
        for the given *mesh_el_group*.
        """
        raise NotImplementedError()

# }}}


# {{{ Nodal element group base

class NodalElementGroupBase(ElementGroupBase, ABC):
    """Base class for nodal element groups, defined as finite elements
    equipped with nodes. Nodes are specific locations defined on the
    reference element (:attr:`~ElementGroupBase.shape`)
    defining a degree of freedom by point evaluation at that location.
    Such element groups have an associated quadrature rule to perform
    numerical integration, but are not necessarily usable (unisolvent)
    for interpolation.

    Inherits from :class:`ElementGroupBase`.

    .. autoattribute:: unit_nodes
    .. automethod:: quadrature_rule
    """

    @property
    @override
    def nunit_dofs(self) -> int:
        """The number of (nodal) degrees of freedom ("DOFs")
        associated with a single element.
        """
        return self.unit_nodes.shape[-1]

    @property
    @memoize_method
    def unit_nodes(self) -> np.ndarray:
        """Returns a :class:`numpy.ndarray` of shape ``(dim, nunit_dofs)``
        of reference coordinates of interpolation nodes.

        Note: this method dispatches to the nodes of the underlying
        quadrature rule. This means, for interpolatory element groups,
        interpolation nodes are collocated with quadrature nodes.
        """
        result = self.quadrature_rule().nodes
        if len(result.shape) == 1:
            result = np.array([result])

        dim2, _ = result.shape
        assert dim2 == self.mesh_el_group.dim
        return result

    @abstractmethod
    def quadrature_rule(self) -> mp.Quadrature:
        """Returns a :class:`modepy.Quadrature` object for the
        element group. This quadrature rule shares the nodes
        returned by :meth:`unit_nodes`.
        """

# }}}


# {{{ Element groups with explicit bases

class ElementGroupWithBasis(ElementGroupBase, ABC):
    """Base class for element groups which possess an
    explicit basis for the underlying function space
    :attr:`~ElementGroupBase.space`.

    Inherits from :class:`ElementGroupBase`.

    .. automethod:: basis_obj
    .. automethod:: is_orthonormal_basis
    """

    @abstractmethod
    def basis_obj(self) -> mp.Basis:
        """Returns the `modepy.Basis` which spans the underlying
        :attr:`~ElementGroupBase.space`.
        """

    @memoize_method
    def is_orthonormal_basis(self):
        """Returns a :class:`bool` flag that is *True* if the
        basis corresponding to the element group is orthonormal
        with respect to the :math:`L^2` inner-product.
        """
        import modepy as mp
        try:
            # Check orthonormality weight
            return self.basis_obj().orthonormality_weight() == 1
        except mp.BasisNotOrthonormal:
            return False

# }}}


# {{{ Element groups suitable for interpolation

class InterpolatoryElementGroupBase(
                                    NodalElementGroupBase,
                                    ElementGroupWithBasis,
                                    ABC):
    """An element group equipped with both an explicit basis for the
    underlying :attr:`~ElementGroupBase.space`, and a set of nodal
    locations on the :attr:`~ElementGroupBase.shape`. These element
    groups are unisolvent in the Ciarlet sense, meaning the dimension
    of :attr:`~ElementGroupBase.space` matches the number of
    interpolatory nodal locations. These element groups are therefore
    suitable for interpolation and differentiation.

    Inherits from :class:`NodalElementGroupBase` and
    :class:`ElementGroupWithBasis`.
    """

# }}}


# {{{ modal element group base

class ModalElementGroupBase(ElementGroupWithBasis, ABC):
    """An element group equipped with a function space
    and a hierarchical basis that is orthonormal with
    respect to the :math:`L^2` inner product.

    Inherits from :class:`ElementGroupWithBasis`.
    """

    @property
    @override
    def nunit_dofs(self) -> int:
        """The number of (modal) degrees of freedom ("DOFs")
        associated with a single element.
        """
        return self.space.space_dim

# }}}


# {{{ discretization

class Discretization:
    """An unstructured composite discretization.

    .. autoattribute:: real_dtype
    .. autoattribute:: complex_dtype
    .. autoattribute:: mesh

    .. attribute:: dim
    .. attribute:: ambient_dim
    .. attribute:: ndofs
    .. attribute:: groups

    .. autoattribute:: is_nodal
    .. autoattribute:: is_modal

    .. automethod:: __init__
    .. automethod:: copy
    .. automethod:: zeros
    .. automethod:: zeros_like
    .. automethod:: nodes
    .. automethod:: quad_weights
    """

    mesh: _Mesh
    groups: Sequence[ElementGroupBase]
    real_dtype: np.dtype[np.floating]
    complex_dtype: np.dtype[np.complexfloating]

    _setup_actx: ArrayContext

    def __init__(self,
                 actx: ArrayContext,
                 mesh: _Mesh,
                 group_factory: ElementGroupFactory,
                 real_dtype: DTypeLike = None,
                 _force_actx_clone: bool = True) -> None:
        """
        :arg actx: an :class:`arraycontext.ArrayContext` used to perform
            computation needed during initial set-up of the discretization.
        :arg mesh: a :class:`~meshmode.mesh.Mesh` over which the discretization
            is built.
        :arg group_factory: an :class:`~meshmode.discretization.ElementGroupFactory`.
        :arg real_dtype: The :mod:`numpy` data type used for representing real
            data, either ``numpy.float32`` or ``numpy.float64``.
        """
        if real_dtype is None:
            real_dtype = np.float64

        if not isinstance(actx, ArrayContext):
            raise TypeError("'actx' must be an ArrayContext")

        self.mesh = mesh
        self.groups = [group_factory(mg) for mg in mesh.groups]

        self.real_dtype = cast("np.dtype[np.floating]", np.dtype(real_dtype))
        self.complex_dtype = np.dtype({
                np.float32: np.complex64,
                np.float64: np.complex128
                }[self.real_dtype.type])

        if _force_actx_clone:
            # We're cloning the array context here to make the setup actx
            # distinct from the "ambient" actx. This allows us to catch
            # errors where arrays from both are inadvertently mixed.
            # See https://github.com/inducer/arraycontext/pull/22/files
            # for context.
            # _force_actx clone exists to disable cloning of the array
            # context when copying a discretization, to allow
            # preserving caches.
            # See https://github.com/inducer/meshmode/pull/293
            # for context.
            actx = actx.clone()

        self._setup_actx = actx
        self._group_factory = group_factory
        self._cached_nodes = None

    def copy(self,
             actx: ArrayContext | None = None,
             mesh: _Mesh | None = None,
             group_factory: ElementGroupFactory | None = None,
             real_dtype: np.dtype | None = None) -> Discretization:
        """Creates a new object of the same type with all arguments that are not
        *None* replaced. The copy is not recursive.
        """

        return type(self)(
                self._setup_actx if actx is None else actx.clone(),
                self.mesh if mesh is None else mesh,
                self._group_factory if group_factory is None else group_factory,
                self.real_dtype if real_dtype is None else real_dtype,
                _force_actx_clone=False,
                )

    @property
    def dim(self):
        return self.mesh.dim

    @property
    def ambient_dim(self):
        return self.mesh.ambient_dim

    @property
    def ndofs(self):
        return sum(grp.ndofs for grp in self.groups)

    @property
    @memoize_method
    def is_nodal(self):
        """A :class:`bool` indicating whether the :class:`Discretization`
        is defined over element groups subclasses of :class:`NodalElementGroupBase`.
        """
        return all(isinstance(grp, NodalElementGroupBase)
                   for grp in self.groups)

    @property
    @memoize_method
    def is_modal(self):
        """A :class:`bool` indicating whether the :class:`Discretization`
        is defined over element groups subclasses of :class:`ModalElementGroupBase`.
        """
        return all(isinstance(grp, ModalElementGroupBase)
                   for grp in self.groups)

    def _new_array(self,
                actx: ArrayContext,
                creation_func,
                dtype: DTypeLike = None
            ):
        if dtype is None:
            dtype = self.real_dtype
        elif dtype == "c":
            dtype = self.complex_dtype
        else:
            dtype = np.dtype(dtype)

        return tag_axes(actx, {
                    0: DiscretizationElementAxisTag(),
                    1: DiscretizationDOFAxisTag()},
                _DOFArray(actx,
                           tuple(creation_func(shape=(grp.nelements,
                                                      grp.nunit_dofs),
                                               dtype=dtype)
                                 for grp in self.groups)))

    def empty(self, actx: ArrayContext, dtype: DTypeLike = None) -> _DOFArray:
        """Return an empty :class:`~meshmode.dof_array.DOFArray`.

        :arg dtype: type special value 'c' will result in a
            vector of dtype :attr:`complex_dtype`. If
            *None* (the default), a real vector will be returned.
        """
        if not isinstance(actx, ArrayContext):
            raise TypeError(
                f"'actx' must be an ArrayContext, not '{type(actx).__name__}'")

        warn(f"'{type(self).__name__}.empty' is deprecated and will be removed "
             f"in 2025. Use '{type(self).__name__}.zeros' instead.",
             DeprecationWarning, stacklevel=2)

        return self._new_array(actx, actx.np.zeros, dtype=dtype)

    def zeros(self, actx: ArrayContext, dtype: DTypeLike = None) -> _DOFArray:
        """Return a zero-initialized :class:`~meshmode.dof_array.DOFArray`.

        :arg dtype: type special value 'c' will result in a
            vector of dtype :attr:`complex_dtype`. If
            *None* (the default), a real vector will be returned.
        """
        if not isinstance(actx, ArrayContext):
            raise TypeError(
                f"'actx' must be an ArrayContext, not '{type(actx).__name__}'")

        return self._new_array(actx, actx.np.zeros, dtype=dtype)

    def empty_like(self, array: _DOFArray) -> _DOFArray:
        warn(f"'{type(self).__name__}.empty_like' is deprecated and will be removed "
             f"in 2025. Use '{type(self).__name__}.zeros_like' instead.",
             DeprecationWarning, stacklevel=2)

        actx = array.array_context
        return self._new_array(actx, actx.np.zeros, dtype=array.entry_dtype)

    def zeros_like(self, array: _DOFArray) -> _DOFArray:
        return self.zeros(array.array_context, dtype=array.entry_dtype)

    @memoize_method
    def quad_weights(self) -> _DOFArray:
        """
        :returns: a :class:`~meshmode.dof_array.DOFArray` with quadrature weights.
        """
        actx = self._setup_actx

        if not self.is_nodal:
            raise ElementGroupTypeError("Element groups must be nodal.")

        @memoize_in(actx, (Discretization, "quad_weights_prg"))
        def prg():
            t_unit = make_loopy_program(
                "{[iel,idof]: 0<=iel<nelements and 0<=idof<nunit_dofs}",
                "result[iel,idof] = weights[idof]",
                name="quad_weights")
            return lp.tag_inames(t_unit, {
                "iel": ConcurrentElementInameTag(),
                "idof": ConcurrentDOFInameTag()})

        return _DOFArray(None, tuple(
                actx.freeze(
                    actx.call_loopy(
                        prg(),
                        weights=actx.from_numpy(grp.quadrature_rule().weights),
                        nelements=grp.nelements,
                        )["result"])
                for grp in self.groups))

    def nodes(self, cached: bool = True) -> ObjectArray[tuple[int], _DOFArray]:
        r"""
        :arg cached: A :class:`bool` indicating whether the computed
            nodes should be stored for future use.
        :returns: object array of shape ``(ambient_dim,)`` containing
            :class:`~meshmode.dof_array.DOFArray`\ s of (global) nodal
            locations on the :attr:`~mesh`.
        """

        if self._cached_nodes is not None:
            if not cached:
                from warnings import warn
                warn("It was requested that the computed nodes not be cached, "
                        "but a cached copy of the nodes was already present.",
                        stacklevel=2)
            return self._cached_nodes

        actx = self._setup_actx

        if not self.is_nodal:
            raise ElementGroupTypeError("Element groups must be nodal.")

        def resample_mesh_nodes(grp: NodalElementGroupBase, iaxis: int):
            name_hint = f"nodes{iaxis}_{self.ambient_dim}d"
            # TODO: would be nice to have the mesh use an array context already
            nodes = tag_axes(actx,
                    {0: DiscretizationElementAxisTag(),
                        1: DiscretizationDOFAxisTag()},
                    actx.from_numpy(grp.mesh_el_group.nodes[iaxis]))

            grp_unit_nodes = grp.unit_nodes.reshape(-1)
            meg_unit_nodes = grp.mesh_el_group.unit_nodes.reshape(-1)

            from arraycontext.metadata import NameHint

            tol = 10 * np.finfo(grp_unit_nodes.dtype).eps
            if (grp_unit_nodes.shape == meg_unit_nodes.shape
                    and np.linalg.norm(grp_unit_nodes - meg_unit_nodes) < tol):
                return actx.tag(NameHint(name_hint), nodes)

            return actx.einsum("ij,ej->ei",
                               actx.tag_axis(
                                   0,
                                   DiscretizationDOFAxisTag(),
                                   actx.from_numpy(grp.from_mesh_interp_matrix())),
                               nodes,
                               tagged=(
                                   FirstAxisIsElementsTag(),
                                   NameHint(name_hint)))

        result = obj_array.new_1d([
            _DOFArray(None, tuple(actx.freeze(resample_mesh_nodes(grp, iaxis))
                      for grp in self.groups))
            for iaxis in range(self.ambient_dim)])
        if cached:
            self._cached_nodes = result
        return result


def num_reference_derivative(
        discr: Discretization,
        ref_axes: Iterable[int],
        vec: _DOFArray) -> _DOFArray:
    """
    :arg ref_axes: an :class:`~collections.abc.Iterable` of indices
        that define the sequence of derivatives to *vec*. For example,
        ``(0, 1, 1)`` would take a third partial derivative, one in the first
        axis and two in the second axis.
    """

    if not all(
            isinstance(grp, InterpolatoryElementGroupBase) for grp in discr.groups
            ):
        raise NoninterpolatoryElementGroupError(
            "Element groups must be usable for differentiation and interpolation.")

    if not ref_axes:
        return vec

    actx = vec.array_context
    ref_axes = tuple(sorted(ref_axes))

    if not all(0 <= ref_axis < discr.dim for ref_axis in ref_axes):
        raise ValueError("'ref_axes' exceeds discretization dimensions: "
                f"got {ref_axes} for dimension {discr.dim}")

    @keyed_memoize_in(actx,
            (num_reference_derivative, "num_reference_derivative_matrix"),
            lambda grp, gref_axes: (*grp.discretization_key(), *gref_axes))
    def get_mat(grp: ElementGroupBase, gref_axes: Sequence[int]):
        if not isinstance(grp, InterpolatoryElementGroupBase):
            raise ValueError("element groups must be interpolatory "
                             "to allow taking derivatives")

        matrices = mp.diff_matrices(grp.basis_obj(), grp.unit_nodes)

        mat = None
        for ref_axis in gref_axes:
            next_mat = matrices[ref_axis]
            if mat is None:
                mat = next_mat
            else:
                mat = next_mat @ mat

        assert mat is not None
        return actx.from_numpy(mat)

    return _DOFArray(actx, tuple(
            actx.einsum("ij,ej->ei",
                        actx.tag_axis(0,
                                      DiscretizationDOFAxisTag(),
                                      get_mat(grp, ref_axes)),
                        vec[igrp],
                        tagged=(FirstAxisIsElementsTag(),))
            for igrp, grp in enumerate(discr.groups)))

# }}}

# vim: fdm=marker
