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
from collections.abc import Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar, cast
from warnings import warn

import numpy as np
from typing_extensions import deprecated, override

import modepy as mp
from pytools import memoize_method, memoize_on_first_arg

from meshmode.discretization import (
    ElementGroupBase,
    ElementGroupFactory,
    InterpolatoryElementGroupBase,
    ModalElementGroupBase,
    NodalElementGroupBase,
)
from meshmode.mesh import (
    MeshElementGroup as _MeshElementGroup,
    SimplexElementGroup as _MeshSimplexElementGroup,
    TensorProductElementGroup as _MeshTensorProductElementGroup,
)


if TYPE_CHECKING:
    from modepy.typing import ArrayF

__doc__ = """
Group types
^^^^^^^^^^^

Simplicial group types
----------------------

.. autoclass:: ModalSimplexElementGroup

.. autoclass:: InterpolatoryQuadratureSimplexElementGroup
.. autoclass:: QuadratureSimplexElementGroup
.. autoclass:: PolynomialWarpAndBlend2DRestrictingElementGroup
.. autoclass:: PolynomialWarpAndBlend3DRestrictingElementGroup
.. autoclass:: PolynomialRecursiveNodesElementGroup
.. autoclass:: PolynomialEquidistantSimplexElementGroup
.. autoclass:: PolynomialGivenNodesElementGroup

Tensor product group types
--------------------------

.. autoclass:: ModalTensorProductElementGroup

.. autoclass:: GaussLegendreTensorProductElementGroup
.. autoclass:: LegendreGaussLobattoTensorProductElementGroup
.. autoclass:: EquidistantTensorProductElementGroup

Group factories
^^^^^^^^^^^^^^^

.. autoclass:: HomogeneousOrderBasedGroupFactory
.. autoclass:: TypeMappingGroupFactory

Simplicial group factories
--------------------------

.. autoclass:: ModalSimplexGroupFactory

.. autoclass:: InterpolatoryQuadratureSimplexGroupFactory
.. autoclass:: QuadratureSimplexGroupFactory
.. autoclass:: PolynomialWarpAndBlend2DRestrictingGroupFactory
.. autoclass:: PolynomialWarpAndBlend3DRestrictingGroupFactory
.. autoclass:: PolynomialRecursiveNodesGroupFactory
.. autoclass:: PolynomialEquidistantSimplexGroupFactory
.. autoclass:: PolynomialGivenNodesGroupFactory

Tensor product group factories
------------------------------

.. autoclass:: ModalTensorProductGroupFactory

.. autoclass:: GaussLegendreTensorProductGroupFactory
.. autoclass:: LegendreGaussLobattoTensorProductGroupFactory
.. autoclass:: EquidistantTensorProductGroupFactory

Type-based group factories
--------------------------

.. autoclass:: InterpolatoryEdgeClusteredGroupFactory
.. autoclass:: InterpolatoryQuadratureGroupFactory
.. autoclass:: InterpolatoryEquidistantGroupFactory
.. autoclass:: QuadratureGroupFactory
.. autoclass:: ModalGroupFactory
"""


# {{{ matrices

@memoize_on_first_arg
def from_mesh_interp_matrix(grp: NodalElementGroupBase) -> ArrayF:
    meg = grp.mesh_el_group

    from meshmode.mesh import ModepyElementGroup
    assert isinstance(meg, ModepyElementGroup)

    meg_basis = mp.basis_for_space(meg.space, meg.shape)
    return mp.resampling_matrix(
            meg_basis.functions,
            grp.unit_nodes,
            meg.unit_nodes)


@memoize_on_first_arg
def to_mesh_interp_matrix(grp: InterpolatoryElementGroupBase) -> ArrayF:
    return mp.resampling_matrix(
            grp.basis_obj().functions,
            grp.mesh_el_group.unit_nodes,
            grp.unit_nodes)

# }}}


# {{{ base class for interpolatory polynomial elements

class PolynomialElementGroupBase(InterpolatoryElementGroupBase, ABC):
    pass

# }}}


# {{{ base class for polynomial modal element groups

class PolynomialModalElementGroupBase(ModalElementGroupBase, ABC):
    @memoize_method
    def basis_obj(self) -> mp.Basis:
        return mp.orthonormal_basis_for_space(self.space, self.shape)

# }}}


# {{{ concrete element groups for modal simplices

class ModalSimplexElementGroup(PolynomialModalElementGroupBase):
    @property
    @memoize_method
    def shape(self) -> mp.Shape:
        return mp.Simplex(self.dim)

    @property
    @memoize_method
    def space(self) -> mp.FunctionSpace:
        return mp.PN(self.dim, self.order)

# }}}


# {{{ concrete element groups for nodal and interpolatory simplices

class SimplexElementGroupBase(NodalElementGroupBase, ABC):
    @property
    @memoize_method
    def shape(self) -> mp.Shape:
        return mp.Simplex(self.dim)

    @property
    @memoize_method
    def space(self) -> mp.FunctionSpace:
        return mp.PN(self.dim, self.order)

    @override
    def from_mesh_interp_matrix(self) -> ArrayF:
        return from_mesh_interp_matrix(self)


class PolynomialSimplexElementGroupBase(
        PolynomialElementGroupBase,
        SimplexElementGroupBase,
        ABC):
    @memoize_method
    def basis_obj(self) -> mp.Basis:
        return mp.basis_for_space(self.space, self.shape)


class InterpolatoryQuadratureSimplexElementGroup(PolynomialSimplexElementGroupBase):
    """Elemental discretization supplying a high-order quadrature rule
    with a number of nodes matching the number of polynomials in :math:`P^k`,
    hence usable for differentiation and interpolation.

    No interpolation nodes are present on the boundary of the simplex.
    """

    @memoize_method
    def quadrature_rule(self) -> mp.Quadrature:
        dims = self.mesh_el_group.dim
        if dims == 0:
            return mp.ZeroDimensionalQuadrature()
        elif dims == 1:
            return mp.LegendreGaussQuadrature(self.order, force_dim_axis=True)
        else:
            return mp.VioreanuRokhlinSimplexQuadrature(self.order, dims)


class QuadratureSimplexElementGroup(SimplexElementGroupBase):
    """Elemental discretization supplying a high-order quadrature rule
    with a number of nodes which does not necessarily match the number of
    polynomials in :math:`P^k`. This discretization therefore excels at
    quadarature, but is not necessarily usable for interpolation.

    No interpolation nodes are present on the boundary of the simplex.
    """

    @memoize_method
    def quadrature_rule(self) -> mp.Quadrature:
        dims = self.mesh_el_group.dim
        if dims == 0:
            return mp.ZeroDimensionalQuadrature()
        elif dims == 1:
            return mp.LegendreGaussQuadrature(self.order, force_dim_axis=True)
        else:
            return mp.XiaoGimbutasSimplexQuadrature(self.order, dims)


class _MassMatrixQuadratureElementGroup(PolynomialSimplexElementGroupBase, ABC):
    @memoize_method
    def quadrature_rule(self) -> mp.Quadrature:
        basis = self.basis_obj()
        nodes = self._interp_nodes
        mass_matrix = mp.mass_matrix(basis, nodes)
        weights = np.dot(mass_matrix,
                         np.ones(len(basis.functions)))
        return mp.Quadrature(nodes, weights, exact_to=self.order)

    @property
    @memoize_method
    def unit_nodes(self) -> ArrayF:
        return self._interp_nodes

    @property
    @abstractmethod
    def _interp_nodes(self) -> ArrayF:
        """Returns a :class:`numpy.ndarray` of shape ``(dim, nunit_dofs)``
        of interpolation nodes on the reference cell.
        """


class PolynomialWarpAndBlendElementGroup(_MassMatrixQuadratureElementGroup):
    """Elemental discretization with a number of nodes matching the number of
    polynomials in :math:`P^k`, hence usable for differentiation and
    interpolation. Interpolation nodes edge-clustered for avoidance of Runge
    phenomena. Nodes are present on the boundary of the simplex.

    Uses :func:`modepy.warp_and_blend_nodes`.
    """
    def __init__(self, mesh_el_group: _MeshElementGroup, order: int) -> None:
        warn("PolynomialWarpAndBlendElementGroup is deprecated, since "
                "the facial restrictions of the 3D nodes are not the 2D nodes. "
                "It will go away in 2022. "
                "Use PolynomialWarpAndBlend2DRestrictingElementGroup or "
                "PolynomialWarpAndBlend3DRestrictingElementGroup instead.",
                DeprecationWarning, stacklevel=2)
        super().__init__(mesh_el_group, order)

    @property
    @memoize_method
    def _interp_nodes(self) -> ArrayF:
        dim = self.mesh_el_group.dim
        if self.order == 0:
            result = mp.warp_and_blend_nodes(dim, 1)
            result = np.mean(result, axis=1).reshape(-1, 1)
        else:
            result = mp.warp_and_blend_nodes(dim, self.order)

        dim2, _ = result.shape
        assert dim2 == dim
        return result


class PolynomialWarpAndBlend2DRestrictingElementGroup(
        _MassMatrixQuadratureElementGroup):
    """Elemental discretization with a number of nodes matching the number of
    polynomials in :math:`P^k`, hence usable for differentiation and
    interpolation. Interpolation nodes edge-clustered for avoidance of Runge
    phenomena. Nodes are present on the boundary of the simplex.
    Provides nodes in two and fewer dimensions, based on the 2D
    warp-and-blend nodes and their facial restrictions.

    Uses :func:`modepy.warp_and_blend_nodes`.
    """
    @property
    @memoize_method
    def _interp_nodes(self) -> ArrayF:
        dim = self.mesh_el_group.dim
        if self.order == 0:
            result = mp.warp_and_blend_nodes(dim, 1)
            result = np.mean(result, axis=1).reshape(-1, 1)
        elif dim >= 3:
            raise ValueError(
                    "PolynomialWarpAndBlend2DRestrictingElementGroup does not "
                    f"provide nodes in {dim}D")
        else:
            result = mp.warp_and_blend_nodes(dim, self.order)

        dim2, _ = result.shape
        assert dim2 == dim
        return result


class PolynomialWarpAndBlend3DRestrictingElementGroup(
        _MassMatrixQuadratureElementGroup):
    """Elemental discretization with a number of nodes matching the number of
    polynomials in :math:`P^k`, hence usable for differentiation and
    interpolation. Interpolation nodes edge-clustered for avoidance of Runge
    phenomena. Nodes are present on the boundary of the simplex.
    Provides nodes in two and fewer dimensions, based on the 3D
    warp-and-blend nodes and their facial restrictions.

    Uses :func:`modepy.warp_and_blend_nodes`.
    """
    @property
    @memoize_method
    def _interp_nodes(self) -> ArrayF:
        dim = self.mesh_el_group.dim
        if self.order == 0:
            result = mp.warp_and_blend_nodes(dim, 1)
            result = np.mean(result, axis=1).reshape(-1, 1)
        else:
            result = mp.warp_and_blend_nodes(3, self.order)
            tol = np.finfo(result.dtype).eps * 50
            for d in range(dim, 3):
                wanted = np.abs(result[d] - (-1)) < tol
                result = result[:, wanted]

            result = result[:dim]

        return result


class PolynomialRecursiveNodesElementGroup(_MassMatrixQuadratureElementGroup):
    """Elemental discretization with a number of nodes matching the number of
    polynomials in :math:`P^k`, hence usable for differentiation and
    interpolation. Interpolation nodes edge-clustered for avoidance of Runge
    phenomena. Depending on the *family* argument, nodes may be present on the
    boundary of the simplex. See [Isaac20]_ for details.

    Supports a choice of the base *family* of 1D nodes, see the documentation
    of the *family* argument to :func:`recursivenodes.recursive_nodes`.

    Requires :mod:`recursivenodes` to be installed.

    .. [Isaac20] Tobin Isaac. Recursive, parameter-free, explicitly defined
        interpolation nodes for simplices.
        `Arxiv preprint <https://arxiv.org/abs/2002.09421>`__.

    .. versionadded:: 2020.2
    """

    family: str

    def __init__(self,
                 mesh_el_group: _MeshElementGroup,
                 order: int,
                 family: str) -> None:
        super().__init__(mesh_el_group, order)
        self.family = family

    @property
    @memoize_method
    def _interp_nodes(self) -> ArrayF:
        dim = self.mesh_el_group.dim

        from recursivenodes import recursive_nodes
        result = cast("ArrayF",
                      recursive_nodes(dim, self.order, self.family, domain="biunit")
                      .T.copy())

        dim2, _ = result.shape
        assert dim2 == dim
        return result

    @override
    def discretization_key(self) -> Sequence[Hashable]:
        return (type(self), self.dim, self.order, self.family)


class PolynomialEquidistantSimplexElementGroup(_MassMatrixQuadratureElementGroup):
    """Elemental discretization with a number of nodes matching the number of
    polynomials in :math:`P^k`, hence usable for differentiation and
    interpolation. Interpolation nodes are present on the boundary of the
    simplex.

    .. versionadded:: 2016.1
    """
    @property
    @memoize_method
    def _interp_nodes(self) -> ArrayF:
        dim = self.mesh_el_group.dim
        result = mp.equidistant_nodes(dim, self.order)

        dim2, _ = result.shape
        assert dim2 == dim
        return result


class PolynomialGivenNodesElementGroup(_MassMatrixQuadratureElementGroup):
    """Elemental discretization with a number of nodes matching the number of
    polynomials in :math:`P^k`, hence usable for differentiation and
    interpolation. Uses nodes given by the user.
    """

    _unit_nodes: ArrayF

    def __init__(self,
                 mesh_el_group: _MeshElementGroup,
                 order: int,
                 unit_nodes: ArrayF) -> None:
        super().__init__(mesh_el_group, order)
        self._unit_nodes = unit_nodes

    @property
    @memoize_method
    def _interp_nodes(self) -> ArrayF:
        dim2, nunit_nodes = self._unit_nodes.shape

        if dim2 != self.mesh_el_group.dim:
            raise ValueError("unit nodes supplied to "
                    "PolynomialGivenNodesElementGroup do not have expected "
                    "dimensionality")

        if nunit_nodes != len(self.basis_obj().functions):
            raise ValueError("unit nodes supplied to "
                    "PolynomialGivenNodesElementGroup do not have expected "
                    "node count for provided order")

        return self._unit_nodes

    @override
    def discretization_key(self) -> Sequence[Hashable]:
        # FIXME?
        # The unit_nodes numpy array isn't hashable, and comparisons would
        # be pretty expensive.
        raise NotImplementedError(
            f"'{type(self).__name__}' does not implement 'discretization_key'")

# }}}


# {{{ concrete element groups for modal tensor product (hypercube) elements

class ModalTensorProductElementGroup(PolynomialModalElementGroupBase):
    @property
    @memoize_method
    def shape(self) -> mp.Shape:
        return mp.Hypercube(self.dim)

    @property
    @memoize_method
    def space(self) -> mp.FunctionSpace:
        return mp.QN(self.dim, self.order)

# }}}


# {{{ concrete element groups for nodal tensor product (hypercube) elements

class HypercubeElementGroupBase(NodalElementGroupBase, ABC):
    @property
    @memoize_method
    def shape(self) -> mp.Shape:
        return mp.Hypercube(self.dim)

    @property
    @memoize_method
    def space(self) -> mp.FunctionSpace:
        return mp.QN(self.dim, self.order)

    @override
    def from_mesh_interp_matrix(self) -> ArrayF:
        return from_mesh_interp_matrix(self)


class TensorProductElementGroupBase(
        PolynomialElementGroupBase,
        HypercubeElementGroupBase):
    _basis: mp.TensorProductBasis
    _nodes: ArrayF

    def __init__(self,
                 mesh_el_group: _MeshTensorProductElementGroup,
                 order: int, *,
                 basis: mp.Basis,
                 unit_nodes: ArrayF) -> None:
        """
        :arg basis: a :class:`modepy.TensorProductBasis`.
        :arg unit_nodes: unit nodes for the tensor product, obtained by
            using :func:`modepy.tensor_product_nodes`, for example.
        """
        super().__init__(mesh_el_group, order)

        if basis._dim != mesh_el_group.dim:
            raise ValueError(
                    "basis dimension does not match element group: "
                    f"expected {mesh_el_group.dim}, got {basis._dim}.")

        if isinstance(basis, mp.TensorProductBasis):
            for b in basis.bases:
                if b._dim != 1:
                    raise NotImplementedError(
                        "All bases used to construct the tensor "
                        "product must be of dimension 1. Support "
                        "for higher-dimensional component bases "
                        "does not yet exist.")

        if unit_nodes.shape[0] != mesh_el_group.dim:
            raise ValueError("unit node dimension does not match element group: "
                    f"expected {self.mesh_el_group.dim}, "
                    f"got {unit_nodes.shape[0]}.")

        # NOTE there are cases where basis is a 1D `_SimplexONB` object. We wrap
        # in a TensorProductBasis object if this is the case
        if not isinstance(basis, mp.TensorProductBasis):
            if basis._dim == 1 and unit_nodes.shape[0] == 1:
                basis = mp.TensorProductBasis([basis])
            else:
                raise ValueError("`basis` is not a TensorProductBasis object, "
                                 "and `basis` and `unit_nodes` are not both of "
                                 f"dimension 1. Found `basis` dim = {basis._dim}, "
                                 f"`unit_nodes` dim = {unit_nodes.shape[0]}.")

        self._basis = basis
        self._nodes = unit_nodes

    @override
    def basis_obj(self) -> mp.Basis:
        return self._basis

    @memoize_method
    def quadrature_rule(self) -> mp.Quadrature:
        from modepy.tools import reshape_array_for_tensor_product_space

        quads: list[mp.Quadrature] = []

        if self.dim != 1:
            nodes_tp = cast(
                "ArrayF",
                reshape_array_for_tensor_product_space(self.space, self._nodes))
        else:
            nodes_tp = self._nodes

        for idim, (nodes, basis) in enumerate(
                zip(nodes_tp, self._basis.bases, strict=True)):
            # get current dimension's nodes
            iaxis = (*(0,)*idim, slice(None), *(0,)*(self.dim-idim-1))
            nodes = nodes[iaxis]

            nodes_1d = nodes.reshape(1, -1)
            mass_matrix = mp.mass_matrix(basis, nodes_1d)
            weights = np.dot(mass_matrix, np.ones(len(basis.functions)))

            quads.append(mp.Quadrature(nodes_1d, weights, exact_to=self.order))

        tp_quad = mp.TensorProductQuadrature(quads)
        assert np.allclose(tp_quad.nodes, self._nodes)

        return tp_quad

    @property
    @memoize_method
    def unit_nodes_1d(self) -> ArrayF:
        return self._nodes[0][:self.order + 1].reshape(1, self.order + 1)

    @override
    def discretization_key(self) -> Sequence[Hashable]:
        # FIXME?
        # The unit_nodes numpy array isn't hashable, and comparisons would
        # be pretty expensive.
        raise NotImplementedError(
                f"'{type(self).__name__}' does not implement discretization_key")


class LegendreTensorProductElementGroup(TensorProductElementGroupBase):
    def __init__(self,
                 mesh_el_group: _MeshTensorProductElementGroup,
                 order: int, *,
                 unit_nodes: ArrayF) -> None:
        basis = mp.orthonormal_basis_for_space(
                mp.QN(mesh_el_group.dim, order),
                mp.Hypercube(mesh_el_group.dim))

        super().__init__(mesh_el_group, order, basis=basis, unit_nodes=unit_nodes)


class GaussLegendreTensorProductElementGroup(LegendreTensorProductElementGroup):
    """Elemental discretization supplying a high-order quadrature rule
    with a number of nodes matching the number of polynomials in the tensor
    product basis, hence usable for differentiation and interpolation.

    No interpolation nodes are present on the boundary of the hypercube.
    """

    def __init__(self,
                 mesh_el_group: _MeshTensorProductElementGroup,
                 order: int) -> None:
        self._quadrature_rule: mp.Quadrature = (
            mp.LegendreGaussTensorProductQuadrature(order, mesh_el_group.dim))

        super().__init__(mesh_el_group, order,
                unit_nodes=self._quadrature_rule.nodes)

    @memoize_method
    def quadrature_rule(self) -> mp.Quadrature:
        return self._quadrature_rule

    @override
    def discretization_key(self) -> Sequence[Hashable]:
        return (type(self), self.dim, self.order)


class LegendreGaussLobattoTensorProductElementGroup(
        LegendreTensorProductElementGroup):
    """Elemental discretization supplying a high-order quadrature rule
    with a number of nodes matching the number of polynomials in the tensor
    product basis, hence usable for differentiation and interpolation.
    Nodes sufficient for unisolvency are present on the boundary of the hypercube.

    Uses :func:`~modepy.quadrature.jacobi_gauss.legendre_gauss_lobatto_nodes`.
    """

    def __init__(self,
                 mesh_el_group: _MeshTensorProductElementGroup,
                 order: int) -> None:
        from modepy.quadrature.jacobi_gauss import legendre_gauss_lobatto_nodes
        unit_nodes_1d = legendre_gauss_lobatto_nodes(order)
        unit_nodes = mp.tensor_product_nodes([unit_nodes_1d]*mesh_el_group.dim)

        super().__init__(mesh_el_group, order, unit_nodes=unit_nodes)

    @override
    def discretization_key(self) -> Sequence[Hashable]:
        return (type(self), self.dim, self.order)


class EquidistantTensorProductElementGroup(LegendreTensorProductElementGroup):
    """Elemental discretization supplying a high-order quadrature rule
    with a number of nodes matching the number of polynomials in the tensor
    product basis, hence usable for differentiation and interpolation.
    Nodes sufficient for unisolvency are present on the boundary of the hypercube.

    Uses :func:`~modepy.equidistant_nodes`.
    """

    def __init__(self,
                 mesh_el_group: _MeshTensorProductElementGroup,
                 order: int) -> None:
        from modepy.nodes import equidistant_nodes
        unit_nodes_1d = equidistant_nodes(1, order)[0]
        unit_nodes = mp.tensor_product_nodes([unit_nodes_1d]*mesh_el_group.dim)

        super().__init__(mesh_el_group, order, unit_nodes=unit_nodes)

    @override
    def discretization_key(self) -> Sequence[Hashable]:
        return (type(self), self.dim, self.order)

# }}}


# {{{ group factories

class HomogeneousOrderBasedGroupFactory(ElementGroupFactory):
    """Element group factory for a single type of
    :class:`meshmode.mesh.MeshElementGroup` and fixed order.

    .. autoattribute:: mesh_group_class
    .. autoattribute:: group_class
    .. autoattribute:: order

    .. automethod:: __init__
    .. automethod:: __call__
    """

    mesh_group_class: ClassVar[type[_MeshElementGroup]]
    group_class: ClassVar[type[ElementGroupBase]]
    order: int

    def __init__(self, order: int) -> None:
        """
        :arg order: integer denoting the order of the
            :class:`~meshmode.discretization.ElementGroupBase`. The exact
            interpretation of the order is left to each individual class,
            as given by :attr:`group_class`.
        """

        self.order = order

    @override
    def __call__(self, mesh_el_group: _MeshElementGroup) -> ElementGroupBase:
        """
        :returns: an element group of type :attr:`group_class` and order
            :attr:`order`.
        """
        if not isinstance(mesh_el_group, self.mesh_group_class):
            raise TypeError(
                    f"only mesh element groups of type {self.mesh_group_class} "
                    "are supported")

        return self.group_class(mesh_el_group, self.order)


ElementTypeMapping = Mapping[
    type[_MeshElementGroup],
    type[ElementGroupBase] | ElementGroupFactory]


class TypeMappingGroupFactory(ElementGroupFactory):
    r"""Element group factory that supports multiple types of
    :class:`~meshmode.mesh.MeshElementGroup`\ s, defined through the mapping
    :attr:`mesh_group_class_to_factory`.

    .. autoattribute:: order
    .. autoattribute:: mesh_group_class_to_factory

    .. automethod:: __init__
    .. automethod:: __call__
    """

    order: int
    mesh_group_class_to_factory: ElementTypeMapping

    def __init__(self,
                 order: int,
                 mesh_group_class_to_factory: ElementTypeMapping) -> None:
        """
        :arg mesh_group_class_to_factory: a :class:`dict` from
            :class:`~meshmode.mesh.MeshElementGroup` subclasses to
            :class:`~meshmode.discretization.ElementGroupBase` subclasses or
            :class:`~meshmode.discretization.ElementGroupFactory`
            instances.
        """
        super().__init__()

        self.order = order
        self.mesh_group_class_to_factory = mesh_group_class_to_factory

    @override
    def __call__(self, mesh_el_group: _MeshElementGroup) -> ElementGroupBase:
        cls = self.mesh_group_class_to_factory.get(type(mesh_el_group), None)

        if cls is None:
            raise TypeError(
                    f"mesh group of type '{type(mesh_el_group).__name__}' is "
                    "not supported; available types are: {}".format(
                        {k.__name__ for k in self.mesh_group_class_to_factory}
                        ))

        if isinstance(cls, type) and issubclass(cls, ElementGroupBase):
            return cls(mesh_el_group, self.order)
        elif isinstance(cls, ElementGroupFactory):
            return cls(mesh_el_group)
        else:
            raise TypeError(f"unknown class: '{cls.__name__}'")


@deprecated("long overdue for removal")
class OrderAndTypeBasedGroupFactory(TypeMappingGroupFactory):
    def __init__(self,
                 order: int,
                 simplex_group_class: type[InterpolatoryElementGroupBase],
                 tensor_product_group_class: type[InterpolatoryElementGroupBase],
                 ) -> None:
        warn("OrderAndTypeBasedGroupFactory is deprecated and will go away in 2023. "
                "Use TypeMappingGroupFactory instead.",
                DeprecationWarning, stacklevel=2)

        super().__init__(order, {
            _MeshSimplexElementGroup: simplex_group_class,
            _MeshTensorProductElementGroup: tensor_product_group_class,
            })


# {{{ group factories for simplices

class ModalSimplexGroupFactory(HomogeneousOrderBasedGroupFactory):
    mesh_group_class: ClassVar[type[_MeshElementGroup]] = _MeshSimplexElementGroup
    group_class: ClassVar[type[ElementGroupBase]] = ModalSimplexElementGroup


class InterpolatoryQuadratureSimplexGroupFactory(HomogeneousOrderBasedGroupFactory):
    mesh_group_class: ClassVar[type[_MeshElementGroup]] = _MeshSimplexElementGroup
    group_class: ClassVar[type[ElementGroupBase]] = (
        InterpolatoryQuadratureSimplexElementGroup)


class QuadratureSimplexGroupFactory(HomogeneousOrderBasedGroupFactory):
    mesh_group_class: ClassVar[type[_MeshElementGroup]] = _MeshSimplexElementGroup
    group_class: ClassVar[type[ElementGroupBase]] = QuadratureSimplexElementGroup


@deprecated("long overdue for removal")
class PolynomialWarpAndBlendGroupFactory(HomogeneousOrderBasedGroupFactory):
    mesh_group_class: ClassVar[type[_MeshElementGroup]] = _MeshSimplexElementGroup
    group_class: ClassVar[type[ElementGroupBase]] = PolynomialWarpAndBlendElementGroup

    def __init__(self, order: int) -> None:
        warn(f"'{type(self).__name__}' is deprecated, since the facial restrictions "
             "of the 3D nodes are not the 2D nodes. "
             "It will go away in 2022. "
             "Use 'PolynomialWarpAndBlend2DRestrictingGroupFactory' or "
             "'PolynomialWarpAndBlend3DRestrictingGroupFactory' instead.",
             DeprecationWarning, stacklevel=2)

        super().__init__(order)


class PolynomialWarpAndBlend2DRestrictingGroupFactory(
        HomogeneousOrderBasedGroupFactory):
    mesh_group_class: ClassVar[type[_MeshElementGroup]] = _MeshSimplexElementGroup
    group_class: ClassVar[type[ElementGroupBase]] = (
        PolynomialWarpAndBlend2DRestrictingElementGroup)


class PolynomialWarpAndBlend3DRestrictingGroupFactory(
        HomogeneousOrderBasedGroupFactory):
    mesh_group_class: ClassVar[type[_MeshElementGroup]] = _MeshSimplexElementGroup
    group_class: ClassVar[type[ElementGroupBase]] = (
        PolynomialWarpAndBlend3DRestrictingElementGroup)


class PolynomialRecursiveNodesGroupFactory(HomogeneousOrderBasedGroupFactory):
    mesh_group_class: ClassVar[type[_MeshElementGroup]] = _MeshSimplexElementGroup
    group_class: ClassVar[type[ElementGroupBase]] = PolynomialRecursiveNodesElementGroup
    family: str

    def __init__(self, order: int, family: str) -> None:
        super().__init__(order)
        self.family = family

    @override
    def __call__(self, mesh_el_group: _MeshElementGroup) -> ElementGroupBase:
        if not isinstance(mesh_el_group, _MeshSimplexElementGroup):
            raise TypeError(
                "only mesh element groups of type "
                f"'{_MeshSimplexElementGroup.__name__}' are supported")

        return PolynomialRecursiveNodesElementGroup(
                mesh_el_group, self.order, self.family)


class PolynomialEquidistantSimplexGroupFactory(HomogeneousOrderBasedGroupFactory):
    """
    .. versionadded:: 2016.1
    """

    mesh_group_class: ClassVar[type[_MeshElementGroup]] = _MeshSimplexElementGroup
    group_class: ClassVar[type[ElementGroupBase]] = (
        PolynomialEquidistantSimplexElementGroup)


class PolynomialGivenNodesGroupFactory(HomogeneousOrderBasedGroupFactory):
    unit_nodes: ArrayF

    def __init__(self, order: int, unit_nodes: ArrayF) -> None:
        super().__init__(order)
        self.unit_nodes = unit_nodes

    @override
    def __call__(self, mesh_el_group: _MeshElementGroup) -> ElementGroupBase:
        if not isinstance(mesh_el_group, _MeshSimplexElementGroup):
            raise TypeError(
                "only mesh element groups of type "
                f"'{_MeshSimplexElementGroup.__name__}' are supported")

        return PolynomialGivenNodesElementGroup(
                mesh_el_group, self.order, self.unit_nodes)

# }}}


# {{{ group factories for tensor products

class ModalTensorProductGroupFactory(HomogeneousOrderBasedGroupFactory):
    mesh_group_class: ClassVar[type[_MeshElementGroup]] = _MeshTensorProductElementGroup
    group_class: ClassVar[type[ElementGroupBase]] = ModalTensorProductElementGroup


class GaussLegendreTensorProductGroupFactory(HomogeneousOrderBasedGroupFactory):
    mesh_group_class: ClassVar[type[_MeshElementGroup]] = _MeshTensorProductElementGroup
    group_class: ClassVar[type[ElementGroupBase]] = (
        GaussLegendreTensorProductElementGroup)


class LegendreGaussLobattoTensorProductGroupFactory(
        HomogeneousOrderBasedGroupFactory):
    mesh_group_class: ClassVar[type[_MeshElementGroup]] = _MeshTensorProductElementGroup
    group_class: ClassVar[type[ElementGroupBase]] = (
        LegendreGaussLobattoTensorProductElementGroup)


class EquidistantTensorProductGroupFactory(HomogeneousOrderBasedGroupFactory):
    mesh_group_class: ClassVar[type[_MeshElementGroup]] = _MeshTensorProductElementGroup
    group_class: ClassVar[type[ElementGroupBase]] = EquidistantTensorProductElementGroup

# }}}


# {{{ mesh element group type-based group factories

class _DefaultPolynomialSimplexGroupFactory(ElementGroupFactory):
    order: int

    def __init__(self, order: int) -> None:
        self.order = order

    @override
    def __call__(self, mesh_el_group: _MeshElementGroup) -> ElementGroupBase:
        factory = default_simplex_group_factory(mesh_el_group.dim, self.order)
        return factory(mesh_el_group)


class InterpolatoryEdgeClusteredGroupFactory(TypeMappingGroupFactory):
    r"""Element group factory for all supported
    :class:`~meshmode.mesh.MeshElementGroup`\ s that constructs (recommended)
    element groups with edge-clustered nodes that can be used for interpolation.
    """

    def __init__(self, order: int) -> None:
        super().__init__(order, {
            _MeshSimplexElementGroup: _DefaultPolynomialSimplexGroupFactory(order),
            _MeshTensorProductElementGroup:
                LegendreGaussLobattoTensorProductElementGroup,
            })


class InterpolatoryQuadratureGroupFactory(TypeMappingGroupFactory):
    r"""Element group factory for all supported
    :class:`~meshmode.mesh.MeshElementGroup`\ s that constructs (recommended)
    element groups with nodes that can be used for interpolation and high-order
    quadrature.
    """

    def __init__(self, order: int) -> None:
        super().__init__(order, {
            _MeshSimplexElementGroup: InterpolatoryQuadratureSimplexElementGroup,
            _MeshTensorProductElementGroup: GaussLegendreTensorProductElementGroup,
            })


class InterpolatoryEquidistantGroupFactory(TypeMappingGroupFactory):
    r"""Element group factory for all supported
    :class:`~meshmode.mesh.MeshElementGroup`\ s that constructs (recommended)
    element groups with equidistant nodes that can be used for interpolation.
    """

    def __init__(self, order: int) -> None:
        super().__init__(order, {
            _MeshSimplexElementGroup: PolynomialEquidistantSimplexElementGroup,
            _MeshTensorProductElementGroup: EquidistantTensorProductElementGroup,
            })


class QuadratureGroupFactory(TypeMappingGroupFactory):
    r"""Element group factory for all supported
    :class:`~meshmode.mesh.MeshElementGroup`\ s that constructs (recommended)
    element groups with nodes that can be used for high-order quadrature,
    but (not necessarily) for interpolation.
    """
    def __init__(self, order: int) -> None:
        super().__init__(order, {
            _MeshSimplexElementGroup: QuadratureSimplexElementGroup,
            _MeshTensorProductElementGroup: GaussLegendreTensorProductElementGroup,
            })


class ModalGroupFactory(TypeMappingGroupFactory):
    r"""Element group factory for all supported
    :class:`~meshmode.mesh.MeshElementGroup`\ s that constructs (recommended)
    element groups with modal degrees of freedom.
    """
    def __init__(self, order: int) -> None:
        super().__init__(order, {
            _MeshSimplexElementGroup: ModalSimplexElementGroup,
            _MeshTensorProductElementGroup: ModalTensorProductElementGroup,
            })

# }}}

# }}}


# undocumented for now, mainly for internal use
def default_simplex_group_factory(base_dim: int, order: int) -> ElementGroupFactory:
    """
    :arg base_dim: The dimension of the 'base' discretization to be used.
        The returned group factory will also support creating lower-dimensional
        discretizations.
    """

    try:
        # FIXME: this is a hard dependency (in pyproject.toml) now, so this
        # shouldn't be needed
        import recursivenodes  # noqa: F401
    except ImportError:
        # If it cannot be imported, use warp-and-blend nodes.
        if base_dim <= 2:
            return PolynomialWarpAndBlend2DRestrictingGroupFactory(order)
        elif base_dim == 3:
            return PolynomialWarpAndBlend3DRestrictingGroupFactory(order)
        else:
            raise ValueError(
                f"no usable set of nodes found for {base_dim}D") from None

    return PolynomialRecursiveNodesGroupFactory(order, family="lgl")

# vim: fdm=marker
