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


import numpy as np
from pytools import memoize_method
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

Simplicial group types
----------------------

.. autoclass:: ModalSimplexElementGroup

.. autoclass:: InterpolatoryQuadratureSimplexElementGroup
.. autoclass:: QuadratureSimplexElementGroup
.. autoclass:: PolynomialWarpAndBlendElementGroup
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
.. autoclass:: PolynomialWarpAndBlendGroupFactory
.. autoclass:: PolynomialRecursiveNodesGroupFactory
.. autoclass:: PolynomialEquidistantSimplexGroupFactory
.. autoclass:: PolynomialGivenNodesGroupFactory

Tensor product group factories
------------------------------

.. autoclass:: ModalTensorProductGroupFactory

.. autoclass:: GaussLegendreTensorProductGroupFactory
.. autoclass:: LegendreGaussLobattoTensorProductGroupFactory
"""


# {{{ base class for interpolatory polynomial elements

class PolynomialElementGroupBase(InterpolatoryElementGroupBase):
    @memoize_method
    def mass_matrix(self):
        assert self.is_orthonormal_basis()

        return mp.mass_matrix(
                self.get_basis().functions,
                self.unit_nodes)

    @memoize_method
    def diff_matrices(self):
        basis_fcts = self.get_basis().functions
        grad_basis_fcts = self.get_basis().gradients
        if len(basis_fcts) != self.unit_nodes.shape[1]:
            raise NoninterpolatoryElementGroupError(
                    "%s does not support interpolation because it is not "
                    "unisolvent (its unit node count does not match its "
                    "number of basis functions). Differentiation requires "
                    "the ability to interpolate." % type(self).__name__)

        result = mp.differentiation_matrices(
                basis_fcts,
                grad_basis_fcts,
                self.unit_nodes)

        if not isinstance(result, tuple):
            return (result,)
        else:
            return result

# }}}


# {{{ base class for polynomial modal element groups

class PolynomialModalElementGroupBase(ModalElementGroupBase):
    @memoize_method
    def get_basis(self):
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

    @memoize_method
    def from_mesh_interp_matrix(self):
        meg = self.mesh_el_group
        meg_space = mp.PN(meg.dim, meg.order)
        return mp.resampling_matrix(
                mp.basis_for_space(meg_space, self.shape).functions,
                self.unit_nodes,
                meg.unit_nodes)


class PolynomialSimplexElementGroupBase(PolynomialElementGroupBase,
        SimplexElementGroupBase):
    @memoize_method
    def get_basis(self):
        return mp.basis_for_space(self.space, self.shape)


class InterpolatoryQuadratureSimplexElementGroup(PolynomialSimplexElementGroupBase):
    """Elemental discretization supplying a high-order quadrature rule
    with a number of nodes matching the number of polynomials in :math:`P^k`,
    hence usable for differentiation and interpolation.

    No interpolation nodes are present on the boundary of the simplex.
    """

    @memoize_method
    def _quadrature_rule(self):
        dims = self.mesh_el_group.dim
        if dims == 0:
            return mp.Quadrature(np.empty((0, 1)), np.empty((0, 1)))
        elif dims == 1:
            return mp.LegendreGaussQuadrature(self.order)
        else:
            return mp.VioreanuRokhlinSimplexQuadrature(self.order, dims)

    @property
    @memoize_method
    def unit_nodes(self):
        result = self._quadrature_rule().nodes
        if len(result.shape) == 1:
            result = np.array([result])

        dim2, _ = result.shape
        assert dim2 == self.mesh_el_group.dim
        return result

    @property
    @memoize_method
    def weights(self):
        return self._quadrature_rule().weights


class QuadratureSimplexElementGroup(SimplexElementGroupBase):
    """Elemental discretization supplying a high-order quadrature rule
    with a number of nodes which does not necessarily match the number of
    polynomials in :math:`P^k`. This discretization therefore excels at
    quadarature, but is not necessarily usable for interpolation.

    No interpolation nodes are present on the boundary of the simplex.
    """

    @memoize_method
    def _quadrature_rule(self):
        dims = self.mesh_el_group.dim
        if dims == 0:
            return mp.Quadrature(np.empty((0, 1)), np.empty((0, 1)))
        elif dims == 1:
            return mp.LegendreGaussQuadrature(self.order)
        else:
            return mp.XiaoGimbutasSimplexQuadrature(self.order, dims)

    @property
    @memoize_method
    def unit_nodes(self):
        result = self._quadrature_rule().nodes
        if len(result.shape) == 1:
            result = np.array([result])

        dim2, _ = result.shape
        assert dim2 == self.mesh_el_group.dim

        return result

    @property
    @memoize_method
    def weights(self):
        return self._quadrature_rule().weights


class _MassMatrixQuadratureElementGroup(PolynomialSimplexElementGroupBase):
    @property
    @memoize_method
    def weights(self):
        return np.dot(
                self.mass_matrix(),
                np.ones(len(self.get_basis().functions)))


class PolynomialWarpAndBlendElementGroup(_MassMatrixQuadratureElementGroup):
    """Elemental discretization with a number of nodes matching the number of
    polynomials in :math:`P^k`, hence usable for differentiation and
    interpolation. Interpolation nodes edge-clustered for avoidance of Runge
    phenomena. Nodes are present on the boundary of the simplex.

    Uses :func:`modepy.warp_and_blend_nodes`.
    """
    @property
    @memoize_method
    def unit_nodes(self):
        dim = self.mesh_el_group.dim
        if self.order == 0:
            result = mp.warp_and_blend_nodes(dim, 1)
            result = np.mean(result, axis=1).reshape(-1, 1)
        else:
            result = mp.warp_and_blend_nodes(dim, self.order)

        dim2, _ = result.shape
        assert dim2 == dim
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
    def unit_nodes(self):
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
    def unit_nodes(self):
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
    def unit_nodes(self):
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

    @memoize_method
    def from_mesh_interp_matrix(self):
        meg = self.mesh_el_group
        meg_space = mp.QN(meg.dim, meg.order)
        return mp.resampling_matrix(
                mp.basis_for_space(meg_space, self.shape).functions,
                self.unit_nodes,
                meg.unit_nodes)


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
        self._unit_nodes = unit_nodes

    @memoize_method
    def get_basis(self):
        return self._basis

    @property
    def unit_nodes(self):
        return self._unit_nodes

    @property
    @memoize_method
    def weights(self):
        return np.dot(
                self.mass_matrix(),
                np.ones(len(self.get_basis().functions)))

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

    @property
    def weights(self):
        return self._quadrature_rule.weights

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

        super().__init__(mesh_el_group, order, index,
                unit_nodes=mp.tensor_product_nodes(
                    [unit_nodes_1d] * mesh_el_group.dim)
                )

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

        super().__init__(mesh_el_group, order, index,
                unit_nodes=mp.tensor_product_nodes(
                    [unit_nodes_1d] * mesh_el_group.dim)
                )

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
    mesh_group_class = _MeshSimplexElementGroup
    group_class = PolynomialWarpAndBlendElementGroup


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


# vim: fdm=marker
