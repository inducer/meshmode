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

from abc import abstractproperty
from typing import Tuple
from warnings import warn

import numpy as np
from pytools import memoize_method, memoize_on_first_arg
from meshmode.mesh import (
        SimplexElementGroup as _MeshSimplexElementGroup,
        TensorProductElementGroup as _MeshTensorProductElementGroup)
from meshmode.discretization import (
        NoninterpolatoryElementGroupError,
        NodalElementGroupBase, ModalElementGroupBase,
        InterpolatoryElementGroupBase)

import modepy as mp

__doc__ = """
Group types
^^^^^^^^^^^

.. autofunction:: mass_matrix
.. autofunction:: diff_matrices

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

.. autoclass:: ElementGroupFactory
.. autoclass:: OrderAndTypeBasedGroupFactory

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
"""


# {{{ matrices

@memoize_on_first_arg
def mass_matrix(grp: InterpolatoryElementGroupBase) -> np.ndarray:
    if not isinstance(grp, InterpolatoryElementGroupBase):
        raise NoninterpolatoryElementGroupError(
                f"cannot construct mass matrix on '{type(grp).__name__}'")

    assert grp.is_orthonormal_basis()
    return mp.mass_matrix(
            grp.basis_obj().functions,
            grp.unit_nodes)


@memoize_on_first_arg
def diff_matrices(grp: InterpolatoryElementGroupBase) -> Tuple[np.ndarray]:
    if not isinstance(grp, InterpolatoryElementGroupBase):
        raise NoninterpolatoryElementGroupError(
                f"cannot construct diff matrices on '{type(grp).__name__}'")

    basis_fcts = grp.basis_obj().functions
    grad_basis_fcts = grp.basis_obj().gradients

    if len(basis_fcts) != grp.unit_nodes.shape[1]:
        raise NoninterpolatoryElementGroupError(
                f"{type(grp).__name__} does not support interpolation because "
                "it is not unisolvent (its unit node count does not match its "
                "number of basis functions). Differentiation requires "
                "the ability to interpolate.")

    result = mp.differentiation_matrices(
            basis_fcts,
            grad_basis_fcts,
            grp.unit_nodes)

    return result if isinstance(result, tuple) else (result,)


@memoize_on_first_arg
def from_mesh_interp_matrix(grp: NodalElementGroupBase) -> np.ndarray:
    meg = grp.mesh_el_group
    meg_space = type(grp.space)(meg.dim, meg.order)

    return mp.resampling_matrix(
            mp.basis_for_space(meg_space, grp.shape).functions,
            grp.unit_nodes,
            meg.unit_nodes)


@memoize_on_first_arg
def to_mesh_interp_matrix(grp: NodalElementGroupBase) -> np.ndarray:
    return mp.resampling_matrix(
            grp.basis_obj().functions,
            grp.mesh_el_group.unit_nodes,
            grp.unit_nodes)

# }}}


# {{{ base class for interpolatory polynomial elements

class PolynomialElementGroupBase(InterpolatoryElementGroupBase):
    def mass_matrix(self):
        warn(
                "This method is deprecated and will go away in 2022.x. "
                "Use 'meshmode.discretization.poly_element.mass_matrix' instead.",
                DeprecationWarning, stacklevel=2)

        return mass_matrix(self)

    def diff_matrices(self):
        warn(
                "This method is deprecated and will go away in 2022.x. "
                "Use 'meshmode.discretization.poly_element.diff_matrices' instead.",
                DeprecationWarning, stacklevel=2)

        return diff_matrices(self)
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
            return mp.LegendreGaussQuadrature(self.order)
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
        basis_fcts = self.basis_obj().functions
        nodes = self._interp_nodes
        mass_matrix = mp.mass_matrix(basis_fcts, nodes)
        weights = np.dot(mass_matrix,
                         np.ones(len(basis_fcts)))
        return mp.Quadrature(nodes, weights, exact_to=self.order)

    @abstractproperty
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
    def __init__(self, mesh_el_group, order, index):
        from warnings import warn
        warn("PolynomialWarpAndBlendElementGroup is deprecated, since "
                "the facial restrictions of the 3D nodes are not the 2D nodes. "
                "It will go away in 2022. "
                "Use PolynomialWarpAndBlend2DRestrictingElementGroup or "
                "PolynomialWarpAndBlend3DRestrictingElementGroup instead.",
                DeprecationWarning, stacklevel=2)
        super().__init__(mesh_el_group, order, index)

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
    def __init__(self, mesh_el_group, order, family, index):
        super().__init__(mesh_el_group, order, index)
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
    def __init__(self, mesh_el_group, order, unit_nodes, index):
        super().__init__(mesh_el_group, order, index)
        self._unit_nodes = unit_nodes

    @property
    def _interp_nodes(self):
        dim2, nunit_nodes = self._unit_nodes.shape

        if dim2 != self.mesh_el_group.dim:
            raise ValueError("unit nodes supplied to "
                    "PolynomialGivenNodesElementGroup do not have expected "
                    "dimensionality")

        if nunit_nodes != len(self.basis()):
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
    def __init__(self, mesh_el_group, order, index, *, basis, unit_nodes):
        """
        :arg basis: a :class:`modepy.TensorProductBasis`.
        :arg unit_nodes: unit nodes for the tensor product, obtained by
            using :func:`modepy.tensor_product_nodes`, for example.
        """
        super().__init__(mesh_el_group, order, index)

        if basis._dim != mesh_el_group.dim:
            raise ValueError("basis dimension does not match element group: "
                    f"expected {mesh_el_group.dim}, got {basis._dim}.")

        if unit_nodes.shape[0] != mesh_el_group.dim:
            raise ValueError("unit node dimension does not match element group: "
                    f"expected {mesh_el_group.dim}, got {unit_nodes.shape[0]}.")

        self._basis = basis
        self._nodes = unit_nodes

    def basis_obj(self):
        return self._basis

    @memoize_method
    def quadrature_rule(self):
        basis_fcts = self._basis.functions
        nodes = self._nodes
        mass_matrix = mp.mass_matrix(basis_fcts, nodes)
        weights = np.dot(mass_matrix,
                         np.ones(len(basis_fcts)))
        return mp.Quadrature(nodes, weights, exact_to=self.order)

    def discretization_key(self):
        # FIXME?
        # The unit_nodes numpy array isn't hashable, and comparisons would
        # be pretty expensive.
        raise NotImplementedError("TensorProductElementGroup does not "
                "implement discretization_key")


class LegendreTensorProductElementGroup(TensorProductElementGroupBase):
    def __init__(self, mesh_el_group, order, index, *, unit_nodes):
        basis = mp.orthonormal_basis_for_space(
                mp.QN(mesh_el_group.dim, order),
                mp.Hypercube(mesh_el_group.dim))

        super().__init__(mesh_el_group, order, index,
                basis=basis,
                unit_nodes=unit_nodes)


class GaussLegendreTensorProductElementGroup(LegendreTensorProductElementGroup):
    """Elemental discretization supplying a high-order quadrature rule
    with a number of nodes matching the number of polynomials in the tensor
    product basis, hence usable for differentiation and interpolation.

    No interpolation nodes are present on the boundary of the hypercube.
    """

    def __init__(self, mesh_el_group, order, index):
        self._quadrature_rule = mp.LegendreGaussTensorProductQuadrature(
                order, mesh_el_group.dim)

        super().__init__(mesh_el_group, order, index,
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

    def __init__(self, mesh_el_group, order, index):
        from modepy.quadrature.jacobi_gauss import legendre_gauss_lobatto_nodes
        unit_nodes_1d = legendre_gauss_lobatto_nodes(order)
        unit_nodes = mp.tensor_product_nodes([unit_nodes_1d] * mesh_el_group.dim)

        super().__init__(mesh_el_group, order, index, unit_nodes=unit_nodes)

    def discretization_key(self):
        return (type(self), self.dim, self.order)


class EquidistantTensorProductElementGroup(LegendreTensorProductElementGroup):
    """Elemental discretization supplying a high-order quadrature rule
    with a number of nodes matching the number of polynomials in the tensor
    product basis, hence usable for differentiation and interpolation.
    Nodes sufficient for unisolvency are present on the boundary of the hypercube.

    Uses :func:`~modepy.equidistant_nodes`.
    """

    def __init__(self, mesh_el_group, order, index):
        from modepy.nodes import equidistant_nodes
        unit_nodes_1d = equidistant_nodes(1, order)[0]
        unit_nodes = mp.tensor_product_nodes([unit_nodes_1d] * mesh_el_group.dim)

        super().__init__(mesh_el_group, order, index, unit_nodes=unit_nodes)

    def discretization_key(self):
        return (type(self), self.dim, self.order)

# }}}


# {{{ group factories

class ElementGroupFactory:
    """
    .. function:: __call__(mesh_ele_group, dof_nr_base)
    """


class HomogeneousOrderBasedGroupFactory(ElementGroupFactory):
    mesh_group_class = type
    group_class = type

    def __init__(self, order):
        self.order = order

    def __call__(self, mesh_el_group, index):
        if not isinstance(mesh_el_group, self.mesh_group_class):
            raise TypeError("only mesh element groups of type '%s' "
                    "are supported" % self.mesh_group_class.__name__)

        return self.group_class(mesh_el_group, self.order, index)


class OrderAndTypeBasedGroupFactory(ElementGroupFactory):
    def __init__(self, order, simplex_group_class, tensor_product_group_class):
        self.order = order
        self.simplex_group_class = simplex_group_class
        self.tensor_product_group_class = tensor_product_group_class

    def __call__(self, mesh_el_group, index):
        if isinstance(mesh_el_group, _MeshSimplexElementGroup):
            group_class = self.simplex_group_class
        elif isinstance(mesh_el_group, _MeshTensorProductElementGroup):
            group_class = self.tensor_product_group_class
        else:
            raise TypeError("only mesh element groups of type '%s' and '%s' "
                    "are supported" % (
                        _MeshSimplexElementGroup.__name__,
                        _MeshTensorProductElementGroup.__name__))

        return group_class(mesh_el_group, self.order, index)


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
        from warnings import warn
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

    def __call__(self, mesh_el_group, index):
        if not isinstance(mesh_el_group, _MeshSimplexElementGroup):
            raise TypeError("only mesh element groups of type '%s' "
                    "are supported" % _MeshSimplexElementGroup.__name__)

        return PolynomialRecursiveNodesElementGroup(
                mesh_el_group, self.order, self.family, index)


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

    def __call__(self, mesh_el_group, index):
        if not isinstance(mesh_el_group, _MeshSimplexElementGroup):
            raise TypeError("only mesh element groups of type '%s' "
                    "are supported" % _MeshSimplexElementGroup.__name__)

        return PolynomialGivenNodesElementGroup(
                mesh_el_group, self.order, self.unit_nodes, index)

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
            raise ValueError(f"no usable set of nodes found for {base_dim}D")

    return PolynomialRecursiveNodesGroupFactory(order, family="lgl")

# vim: fdm=marker
