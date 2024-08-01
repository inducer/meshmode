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

from abc import abstractmethod
from typing import ClassVar
from warnings import warn

import numpy as np

import modepy as mp
from modepy import Basis
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
def from_mesh_interp_matrix(grp: InterpolatoryElementGroupBase) -> np.ndarray:
    meg = grp.mesh_el_group

    from meshmode.mesh import _ModepyElementGroup
    assert isinstance(meg, _ModepyElementGroup)

    meg_basis = mp.basis_for_space(meg._modepy_space, meg._modepy_shape)
    return mp.resampling_matrix(
            meg_basis.functions,
            grp.unit_nodes,
            meg.unit_nodes)


@memoize_on_first_arg
def to_mesh_interp_matrix(grp: InterpolatoryElementGroupBase) -> np.ndarray:
    return mp.resampling_matrix(
            grp.basis_obj().functions,
            grp.mesh_el_group.unit_nodes,
            grp.unit_nodes)

# }}}


# {{{ base class for interpolatory polynomial elements

class PolynomialElementGroupBase(InterpolatoryElementGroupBase):
    pass

# }}}


# {{{ base class for polynomial modal element groups

class PolynomialModalElementGroupBase(ModalElementGroupBase):
    @memoize_method
    def basis_obj(self):
        return mp.orthonormal_basis_for_space(self.space, self.shape)

# }}}


# {{{ concrete element groups for modal simplices

class ModalSimplexElementGroup(PolynomialModalElementGroupBase):
    @property
    @memoize_method
    def shape(self):
        return mp.Simplex(self.dim)

    @property
    @memoize_method
    def space(self):
        return mp.PN(self.dim, self.order)

# }}}


# {{{ concrete element groups for nodal and interpolatory simplices

class SimplexElementGroupBase(NodalElementGroupBase):
    @property
    @memoize_method
    def shape(self):
        return mp.Simplex(self.dim)

    @property
    @memoize_method
    def space(self):
        return mp.PN(self.dim, self.order)

    def from_mesh_interp_matrix(self):
        return from_mesh_interp_matrix(self)


class PolynomialSimplexElementGroupBase(PolynomialElementGroupBase,
        SimplexElementGroupBase):
    @memoize_method
    def basis_obj(self):
        return mp.basis_for_space(self.space, self.shape)


class InterpolatoryQuadratureSimplexElementGroup(PolynomialSimplexElementGroupBase):
    """Elemental discretization supplying a high-order quadrature rule
    with a number of nodes matching the number of polynomials in :math:`P^k`,
    hence usable for differentiation and interpolation.

    No interpolation nodes are present on the boundary of the simplex.
    """

    @memoize_method
    def quadrature_rule(self):
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
    def quadrature_rule(self):
        dims = self.mesh_el_group.dim
        if dims == 0:
            return mp.ZeroDimensionalQuadrature()
        elif dims == 1:
            return mp.LegendreGaussQuadrature(self.order, force_dim_axis=True)
        else:
            return mp.XiaoGimbutasSimplexQuadrature(self.order, dims)


class _MassMatrixQuadratureElementGroup(PolynomialSimplexElementGroupBase):
    @memoize_method
    def quadrature_rule(self):
        basis = self.basis_obj()
        nodes = self._interp_nodes
        mass_matrix = mp.mass_matrix(basis, nodes)
        weights = np.dot(mass_matrix,
                         np.ones(len(basis.functions)))
        return mp.Quadrature(nodes, weights, exact_to=self.order)

    @property
    @memoize_method
    def unit_nodes(self):
        return self._interp_nodes

    @property
    @abstractmethod
    def _interp_nodes(self):
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
    def __init__(self, mesh_el_group, order):
        warn("PolynomialWarpAndBlendElementGroup is deprecated, since "
                "the facial restrictions of the 3D nodes are not the 2D nodes. "
                "It will go away in 2022. "
                "Use PolynomialWarpAndBlend2DRestrictingElementGroup or "
                "PolynomialWarpAndBlend3DRestrictingElementGroup instead.",
                DeprecationWarning, stacklevel=2)
        super().__init__(mesh_el_group, order)

    @property
    @memoize_method
    def _interp_nodes(self):
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
    def _interp_nodes(self):
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
    def _interp_nodes(self):
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
    def __init__(self, mesh_el_group, order, family):
        super().__init__(mesh_el_group, order)
        self.family = family

    @property
    @memoize_method
    def _interp_nodes(self):
        dim = self.mesh_el_group.dim

        from recursivenodes import recursive_nodes
        result = recursive_nodes(dim, self.order, self.family,
                domain="biunit").T.copy()

        dim2, _ = result.shape
        assert dim2 == dim
        return result

    def discretization_key(self):
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
    def _interp_nodes(self):
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
    def __init__(self, mesh_el_group, order, unit_nodes):
        super().__init__(mesh_el_group, order)
        self._unit_nodes = unit_nodes

    @property
    def _interp_nodes(self):
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

    def discretization_key(self):
        # FIXME?
        # The unit_nodes numpy array isn't hashable, and comparisons would
        # be pretty expensive.
        raise NotImplementedError("PolynomialGivenNodesElementGroup does not "
                "implement discretization_key")

# }}}


# {{{ concrete element groups for modal tensor product (hypercube) elements

class ModalTensorProductElementGroup(PolynomialModalElementGroupBase):
    @property
    @memoize_method
    def shape(self):
        return mp.Hypercube(self.dim)

    @property
    @memoize_method
    def space(self):
        return mp.QN(self.dim, self.order)

# }}}


# {{{ concrete element groups for nodal tensor product (hypercube) elements

class HypercubeElementGroupBase(NodalElementGroupBase):
    @property
    @memoize_method
    def shape(self):
        return mp.Hypercube(self.dim)

    @property
    @memoize_method
    def space(self):
        return mp.QN(self.dim, self.order)

    def from_mesh_interp_matrix(self):
        return from_mesh_interp_matrix(self)


class TensorProductElementGroupBase(PolynomialElementGroupBase,
        HypercubeElementGroupBase):
    def __init__(self, mesh_el_group: _MeshTensorProductElementGroup,
                 order: int, *, basis: Basis,
                 unit_nodes: np.ndarray) -> None:
        """
        :arg basis: a :class:`modepy.TensorProductBasis`.
        :arg unit_nodes: unit nodes for the tensor product, obtained by
            using :func:`modepy.tensor_product_nodes`, for example.
        """
        super().__init__(mesh_el_group, order)

        if basis._dim != mesh_el_group.dim:
            raise ValueError("basis dimension does not match element group: "
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
                                 "dimension 1. Found `basis` dim = {basis._dim}, "
                                 "`unit_nodes` dim = {unit_nodes.shape[0]}.")

        self._basis = basis
        self._nodes = unit_nodes

    def basis_obj(self):
        return self._basis

    @memoize_method
    def quadrature_rule(self):
        basis = self._basis
        nodes = self._nodes
        mass_matrix = mp.mass_matrix(basis, nodes)
        weights = np.dot(mass_matrix,
                         np.ones(len(basis.functions)))
        return mp.Quadrature(nodes, weights, exact_to=self.order)

    @property
    @memoize_method
    def unit_nodes_1d(self):
        return self._nodes[0][:self.order + 1].reshape(1, self.order + 1)

    def discretization_key(self):
        # FIXME?
        # The unit_nodes numpy array isn't hashable, and comparisons would
        # be pretty expensive.
        raise NotImplementedError("TensorProductElementGroup does not "
                "implement discretization_key")


class LegendreTensorProductElementGroup(TensorProductElementGroupBase):
    def __init__(self, mesh_el_group, order, *, unit_nodes):
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

    def __init__(self, mesh_el_group, order):
        self._quadrature_rule = mp.LegendreGaussTensorProductQuadrature(
                order, mesh_el_group.dim)

        super().__init__(mesh_el_group, order,
                unit_nodes=self._quadrature_rule.nodes)

    @memoize_method
    def quadrature_rule(self):
        return self._quadrature_rule

    def discretization_key(self):
        return (type(self), self.dim, self.order)


class LegendreGaussLobattoTensorProductElementGroup(
        LegendreTensorProductElementGroup):
    """Elemental discretization supplying a high-order quadrature rule
    with a number of nodes matching the number of polynomials in the tensor
    product basis, hence usable for differentiation and interpolation.
    Nodes sufficient for unisolvency are present on the boundary of the hypercube.

    Uses :func:`~modepy.quadrature.jacobi_gauss.legendre_gauss_lobatto_nodes`.
    """

    def __init__(self, mesh_el_group, order):
        from modepy.quadrature.jacobi_gauss import legendre_gauss_lobatto_nodes
        unit_nodes_1d = legendre_gauss_lobatto_nodes(order)
        unit_nodes = mp.tensor_product_nodes([unit_nodes_1d]*mesh_el_group.dim)

        super().__init__(mesh_el_group, order, unit_nodes=unit_nodes)

    def discretization_key(self):
        return (type(self), self.dim, self.order)


class EquidistantTensorProductElementGroup(LegendreTensorProductElementGroup):
    """Elemental discretization supplying a high-order quadrature rule
    with a number of nodes matching the number of polynomials in the tensor
    product basis, hence usable for differentiation and interpolation.
    Nodes sufficient for unisolvency are present on the boundary of the hypercube.

    Uses :func:`~modepy.equidistant_nodes`.
    """

    def __init__(self, mesh_el_group, order):
        from modepy.nodes import equidistant_nodes
        unit_nodes_1d = equidistant_nodes(1, order)[0]
        unit_nodes = mp.tensor_product_nodes([unit_nodes_1d]*mesh_el_group.dim)

        super().__init__(mesh_el_group, order, unit_nodes=unit_nodes)

    def discretization_key(self):
        return (type(self), self.dim, self.order)

# }}}


# {{{ group factories

class HomogeneousOrderBasedGroupFactory(ElementGroupFactory):
    """Element group factory for a single type of
    :class:`meshmode.mesh.MeshElementGroup` and fixed order.

    .. attribute:: mesh_group_class
    .. attribute:: group_class
    .. attribute:: order

    .. automethod:: __init__
    .. automethod:: __call__
    """

    mesh_group_class: ClassVar[_MeshElementGroup]
    group_class: ClassVar[ElementGroupBase]

    def __init__(self, order: int) -> None:
        """
        :arg order: integer denoting the order of the
            :class:`~meshmode.discretization.ElementGroupBase`. The exact
            interpretation of the order is left to each individual class,
            as given by :attr:`group_class`.
        """

        self.order = order

    def __call__(self, mesh_el_group):
        """
        :returns: an element group of type :attr:`group_class` and order
            :attr:`order`.
        """
        if not isinstance(mesh_el_group, self.mesh_group_class):
            raise TypeError("only mesh element groups of type '%s' "
                    "are supported" % self.mesh_group_class.__name__)

        return self.group_class(mesh_el_group, self.order)


class TypeMappingGroupFactory(ElementGroupFactory):
    r"""Element group factory that supports multiple types of
    :class:`~meshmode.mesh.MeshElementGroup`\ s, defined through the mapping
    :attr:`mesh_group_class_to_factory`.

    .. attribute:: order
    .. attribute:: mesh_group_class_to_factory

        A :class:`dict` from :class:`~meshmode.mesh.MeshElementGroup`\ s to
        factory callables that return a corresponding
        :class:`~meshmode.discretization.ElementGroupBase` of order
        :attr:`order`.

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, order, mesh_group_class_to_factory):
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

    def __call__(self, mesh_el_group):
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


class OrderAndTypeBasedGroupFactory(TypeMappingGroupFactory):
    def __init__(self, order, simplex_group_class, tensor_product_group_class):
        warn("OrderAndTypeBasedGroupFactory is deprecated and will go away in 2023. "
                "Use TypeMappingGroupFactory instead.",
                DeprecationWarning, stacklevel=2)

        super().__init__(order, {
            _MeshSimplexElementGroup: simplex_group_class,
            _MeshTensorProductElementGroup: tensor_product_group_class,
            })


# {{{ group factories for simplices

class ModalSimplexGroupFactory(HomogeneousOrderBasedGroupFactory):
    mesh_group_class = _MeshSimplexElementGroup
    group_class = ModalSimplexElementGroup


class InterpolatoryQuadratureSimplexGroupFactory(HomogeneousOrderBasedGroupFactory):
    mesh_group_class = _MeshSimplexElementGroup
    group_class = InterpolatoryQuadratureSimplexElementGroup


class QuadratureSimplexGroupFactory(HomogeneousOrderBasedGroupFactory):
    mesh_group_class = _MeshSimplexElementGroup
    group_class = QuadratureSimplexElementGroup


class PolynomialWarpAndBlendGroupFactory(HomogeneousOrderBasedGroupFactory):
    def __init__(self, order):
        warn("PolynomialWarpAndBlendGroupFactory is deprecated, since "
                "the facial restrictions of the 3D nodes are not the 2D nodes. "
                "It will go away in 2022. "
                "Use PolynomialWarpAndBlend2DRestrictingGroupFactory or "
                "PolynomialWarpAndBlend3DRestrictingGroupFactory instead.",
                DeprecationWarning, stacklevel=2)

        super().__init__(order)

    mesh_group_class = _MeshSimplexElementGroup
    group_class = PolynomialWarpAndBlendElementGroup


class PolynomialWarpAndBlend2DRestrictingGroupFactory(
        HomogeneousOrderBasedGroupFactory):
    mesh_group_class = _MeshSimplexElementGroup
    group_class = PolynomialWarpAndBlend2DRestrictingElementGroup


class PolynomialWarpAndBlend3DRestrictingGroupFactory(
        HomogeneousOrderBasedGroupFactory):
    mesh_group_class = _MeshSimplexElementGroup
    group_class = PolynomialWarpAndBlend3DRestrictingElementGroup


class PolynomialRecursiveNodesGroupFactory(HomogeneousOrderBasedGroupFactory):
    def __init__(self, order, family):
        super().__init__(order)
        self.family = family

    def __call__(self, mesh_el_group):
        if not isinstance(mesh_el_group, _MeshSimplexElementGroup):
            raise TypeError("only mesh element groups of type '%s' "
                    "are supported" % _MeshSimplexElementGroup.__name__)

        return PolynomialRecursiveNodesElementGroup(
                mesh_el_group, self.order, self.family)

    mesh_group_class = _MeshSimplexElementGroup
    group_class = PolynomialRecursiveNodesElementGroup


class PolynomialEquidistantSimplexGroupFactory(HomogeneousOrderBasedGroupFactory):
    """
    .. versionadded:: 2016.1
    """

    mesh_group_class = _MeshSimplexElementGroup
    group_class = PolynomialEquidistantSimplexElementGroup


class PolynomialGivenNodesGroupFactory(HomogeneousOrderBasedGroupFactory):
    def __init__(self, order, unit_nodes):
        super().__init__(order)
        self.unit_nodes = unit_nodes

    def __call__(self, mesh_el_group):
        if not isinstance(mesh_el_group, _MeshSimplexElementGroup):
            raise TypeError("only mesh element groups of type '%s' "
                    "are supported" % _MeshSimplexElementGroup.__name__)

        return PolynomialGivenNodesElementGroup(
                mesh_el_group, self.order, self.unit_nodes)

# }}}


# {{{ group factories for tensor products

class ModalTensorProductGroupFactory(HomogeneousOrderBasedGroupFactory):
    mesh_group_class = _MeshTensorProductElementGroup
    group_class = ModalTensorProductElementGroup


class GaussLegendreTensorProductGroupFactory(HomogeneousOrderBasedGroupFactory):
    mesh_group_class = _MeshTensorProductElementGroup
    group_class = GaussLegendreTensorProductElementGroup


class LegendreGaussLobattoTensorProductGroupFactory(
        HomogeneousOrderBasedGroupFactory):
    mesh_group_class = _MeshTensorProductElementGroup
    group_class = LegendreGaussLobattoTensorProductElementGroup


class EquidistantTensorProductGroupFactory(HomogeneousOrderBasedGroupFactory):
    mesh_group_class = _MeshTensorProductElementGroup
    group_class = EquidistantTensorProductElementGroup

# }}}


# {{{ mesh element group type-based group factories

class _DefaultPolynomialSimplexGroupFactory(ElementGroupFactory):
    def __init__(self, order):
        self.order = order

    def __call__(self, mesh_el_group):
        factory = default_simplex_group_factory(mesh_el_group.dim, self.order)
        return factory(mesh_el_group)


class InterpolatoryEdgeClusteredGroupFactory(TypeMappingGroupFactory):
    r"""Element group factory for all supported
    :class:`~meshmode.mesh.MeshElementGroup`\ s that constructs (recommended)
    element groups with edge-clustered nodes that can be used for interpolation.
    """

    def __init__(self, order):
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

    def __init__(self, order):
        super().__init__(order, {
            _MeshSimplexElementGroup: InterpolatoryQuadratureSimplexElementGroup,
            _MeshTensorProductElementGroup: GaussLegendreTensorProductElementGroup,
            })


class InterpolatoryEquidistantGroupFactory(TypeMappingGroupFactory):
    r"""Element group factory for all supported
    :class:`~meshmode.mesh.MeshElementGroup`\ s that constructs (recommended)
    element groups with equidistant nodes that can be used for interpolation.
    """

    def __init__(self, order):
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
    def __init__(self, order):
        super().__init__(order, {
            _MeshSimplexElementGroup: QuadratureSimplexElementGroup,
            _MeshTensorProductElementGroup: GaussLegendreTensorProductElementGroup,
            })


class ModalGroupFactory(TypeMappingGroupFactory):
    r"""Element group factory for all supported
    :class:`~meshmode.mesh.MeshElementGroup`\ s that constructs (recommended)
    element groups with modal degrees of freedom.
    """
    def __init__(self, order):
        super().__init__(order, {
            _MeshSimplexElementGroup: ModalSimplexElementGroup,
            _MeshTensorProductElementGroup: ModalTensorProductElementGroup,
            })

# }}}

# }}}


# undocumented for now, mainly for internal use
def default_simplex_group_factory(base_dim, order):
    """
    :arg base_dim: The dimension of the 'base' discretization to be used.
        The returned group factory will also support creating lower-dimensional
        discretizations.
    """

    try:
        # recursivenodes is only importable in Python 3.8 since
        # it uses :func:`math.comb`, so need to check if it can
        # be imported.
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
