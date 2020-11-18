__copyright__ = "Copyright (C) 2013 Andreas Kloeckner"

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
#import numpy.linalg as la
from pytools import memoize_method
from meshmode.mesh import (
        SimplexElementGroup as _MeshSimplexElementGroup,
        TensorProductElementGroup as _MeshTensorProductElementGroup)

import modepy as mp

__doc__ = """

Group types
^^^^^^^^^^^

.. autoclass:: InterpolatoryQuadratureSimplexElementGroup
.. autoclass:: QuadratureSimplexElementGroup
.. autoclass:: PolynomialWarpAndBlendElementGroup
.. autoclass:: PolynomialRecursiveNodesElementGroup
.. autoclass:: PolynomialEquidistantSimplexElementGroup
.. autoclass:: PolynomialGivenNodesElementGroup

.. autoclass:: GaussLegendreTensorProductElementGroup
.. autoclass:: LegendreGaussLobattoTensorProductElementGroup
.. autoclass:: EquidistantTensorProductElementGroup

Group factories
^^^^^^^^^^^^^^^

.. autoclass:: ElementGroupFactory
.. autoclass:: OrderAndTypeBasedGroupFactory

.. autoclass:: InterpolatoryQuadratureSimplexGroupFactory
.. autoclass:: QuadratureSimplexGroupFactory
.. autoclass:: PolynomialWarpAndBlendGroupFactory
.. autoclass:: PolynomialRecursiveNodesGroupFactory
.. autoclass:: PolynomialEquidistantSimplexGroupFactory
.. autoclass:: PolynomialGivenNodesGroupFactory

.. autoclass:: GaussLegendreTensorProductGroupFactory
.. autoclass:: LegendreGaussLobattoTensorProductGroupFactory
"""

from meshmode.discretization import ElementGroupBase, InterpolatoryElementGroupBase


# {{{ base class for poynomial elements

class PolynomialElementGroupBase(InterpolatoryElementGroupBase):
    @memoize_method
    def mass_matrix(self):
        assert self.is_orthogonal_basis()

        import modepy as mp
        return mp.mass_matrix(
                self.basis(),
                self.unit_nodes)

    @memoize_method
    def diff_matrices(self):
        if len(self.basis()) != self.unit_nodes.shape[1]:
            from meshmode.discretization import NoninterpolatoryElementGroupError
            raise NoninterpolatoryElementGroupError(
                    "%s does not support interpolation because it is not "
                    "unisolvent (its unit node count does not match its "
                    "number of basis functions). Differentiation requires "
                    "the ability to interpolate." % type(self).__name__)

        result = mp.differentiation_matrices(
                self.basis(),
                self.grad_basis(),
                self.unit_nodes)

        if not isinstance(result, tuple):
            return (result,)
        else:
            return result

# }}}


# {{{ concrete element groups for simplices

class SimplexElementGroupBase(ElementGroupBase):
    @memoize_method
    def from_mesh_interp_matrix(self):
        meg = self.mesh_el_group
        return mp.resampling_matrix(
                mp.simplex_best_available_basis(meg.dim, meg.order),
                self.unit_nodes,
                meg.unit_nodes)


class PolynomialSimplexElementGroupBase(PolynomialElementGroupBase,
        SimplexElementGroupBase):
    def is_orthogonal_basis(self):
        return self.dim <= 3

    @memoize_method
    def _mode_ids_and_basis(self):
        # for now, see https://gitlab.tiker.net/inducer/modepy/-/merge_requests/14
        import modepy.modes as modes
        if self.dim <= 3:
            return modes.simplex_onb_with_mode_ids(self.dim, self.order)
        else:
            return modes.simplex_monomial_basis_with_mode_ids(self.dim, self.order)

    def basis(self):
        mode_ids, basis = self._mode_ids_and_basis()
        return basis

    def mode_ids(self):
        mode_ids, basis = self._mode_ids_and_basis()
        return mode_ids

    def grad_basis(self):
        if self.dim <= 3:
            return mp.grad_simplex_onb(self.dim, self.order)
        else:
            return mp.grad_simplex_monomial_basis(self.dim, self.order)


class InterpolatoryQuadratureSimplexElementGroup(PolynomialSimplexElementGroupBase):
    """Elemental discretization supplying a high-order quadrature rule
    with a number of nodes matching the number of polynomials in :math:`P^k`,
    hence usable for differentiation and interpolation.

    No interpolation nodes are present on the boundary of the simplex.

    The :meth:`~meshmode.discretization.InterpolatoryElementGroupBase.mode_ids`
    are a tuple (one entry per dimension) of directional polynomial degrees
    on the reference element.
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

        dim2, nunit_nodes = result.shape
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

    The :meth:`~meshmode.discretization.InterpolatoryElementGroupBase.mode_ids`
    are a tuple (one entry per dimension) of directional polynomial degrees
    on the reference element.
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

        dim2, nunit_nodes = result.shape
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
                np.ones(len(self.basis())))


class PolynomialWarpAndBlendElementGroup(_MassMatrixQuadratureElementGroup):
    """Elemental discretization with a number of nodes matching the number of
    polynomials in :math:`P^k`, hence usable for differentiation and
    interpolation. Interpolation nodes edge-clustered for avoidance of Runge
    phenomena. Nodes are present on the boundary of the simplex.

    Uses :func:`modepy.warp_and_blend_nodes`.

    The :meth:`~meshmode.discretization.InterpolatoryElementGroupBase.mode_ids`
    are a tuple (one entry per dimension) of directional polynomial degrees
    on the reference element.
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

        dim2, nunit_nodes = result.shape
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

    The :meth:`~meshmode.discretization.InterpolatoryElementGroupBase.mode_ids`
    are a tuple (one entry per dimension) of directional polynomial degrees
    on the reference element.

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

        dim2, nunit_nodes = result.shape
        assert dim2 == dim
        return result


class PolynomialEquidistantSimplexElementGroup(_MassMatrixQuadratureElementGroup):
    """Elemental discretization with a number of nodes matching the number of
    polynomials in :math:`P^k`, hence usable for differentiation and
    interpolation. Interpolation nodes are present on the boundary of the
    simplex.

    The :meth:`~meshmode.discretization.InterpolatoryElementGroupBase.mode_ids`
    are a tuple (one entry per dimension) of directional polynomial degrees
    on the reference element.

    .. versionadded:: 2016.1
    """
    @property
    @memoize_method
    def unit_nodes(self):
        dim = self.mesh_el_group.dim
        result = mp.equidistant_nodes(dim, self.order)

        dim2, nunit_nodes = result.shape
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

# }}}


# {{{ concrete element groups for tensor product elements

class _TensorProductElementGroupBase(PolynomialElementGroupBase):
    def __init__(self, mesh_el_group, order, index, *,
            basis_1d=None, grad_basis_1d=None,
            unit_nodes_1d=None, quad_weights_1d=None):
        super().__init__(mesh_el_group, order, index)

        self._basis_1d = basis_1d
        self._grad_basis_1d = grad_basis_1d
        self._unit_nodes_1d = unit_nodes_1d
        self._quad_weights_1d = quad_weights_1d

    def is_orthogonal_basis(self):
        return True

    def basis(self):
        from modepy.modes import tensor_product_basis
        return tensor_product_basis(self.dim, self._basis_1d)

    def grad_basis(self):
        from modepy.modes import grad_tensor_product_basis
        return grad_tensor_product_basis(self.dim,
                self._basis_1d, self._grad_basis_1d)

    @property
    @memoize_method
    def unit_nodes(self):
        from modepy.nodes import tensor_product_nodes
        return tensor_product_nodes(self.dim, self._unit_nodes_1d)

    @memoize_method
    def from_mesh_interp_matrix(self):
        from modepy.modes import legendre_tensor_product_basis
        meg = self.mesh_el_group
        return mp.resampling_matrix(
                legendre_tensor_product_basis(self.dim, meg.order),
                self.unit_nodes,
                meg.unit_nodes)

    @property
    @memoize_method
    def weights(self):
        if self._quad_weights_1d is None:
            import modepy as mp
            mm = mp.mass_matrix(self._basis_1d, self._unit_nodes_1d)
            weights = mm @ np.ones(len(self._unit_nodes_1d))
        else:
            weights = self._quad_weights_1d

        from itertools import product
        return np.fromiter(
                (np.prod(w) for w in product(weights, repeat=self.dim)),
                dtype=np.float,
                count=(self.order + 1)**self.dim)


class _LegendreTensorProductElementGroup(_TensorProductElementGroupBase):
    def __init__(self, mesh_el_group, order, index, *,
            unit_nodes_1d=None, quad_weights_1d=None):
        from modepy.modes import jacobi, grad_jacobi
        from functools import partial

        super().__init__(mesh_el_group, order, index,
                basis_1d=tuple(
                    partial(jacobi, 0, 0, i) for i in range(order + 1)),
                grad_basis_1d=tuple(
                    partial(grad_jacobi, 0, 0, i) for i in range(order + 1)),
                unit_nodes_1d=unit_nodes_1d,
                quad_weights_1d=quad_weights_1d)


class GaussLegendreTensorProductElementGroup(_LegendreTensorProductElementGroup):
    """Elemental discretization supplying a high-order quadrature rule
    with a number of nodes matching the number of polynomials in the tensor
    product basis, hence usable for differentiation and interpolation.

    No interpolation nodes are present on the boundary of the hypercube.
    """
    def __init__(self, mesh_el_group, order, index):
        import modepy as mp
        quad = mp.LegendreGaussQuadrature(order)

        super().__init__(mesh_el_group, order, index,
                unit_nodes_1d=quad.nodes,
                quad_weights_1d=quad.weights)


class LegendreGaussLobattoTensorProductElementGroup(
        _LegendreTensorProductElementGroup):
    """Elemental discretization supplying a high-order quadrature rule
    with a number of nodes matching the number of polynomials in the tensor
    product basis, hence usable for differentiation and interpolation.
    Nodes sufficient for unisolvency are present on the boundary of the hypercube.

    Uses :func:`~modepy.quadrature.jacobi_gauss.legendre_gauss_lobatto_nodes`.
    """

    def __init__(self, mesh_el_group, order, index):
        from modepy.quadrature.jacobi_gauss import legendre_gauss_lobatto_nodes
        super().__init__(mesh_el_group, order, index,
                unit_nodes_1d=legendre_gauss_lobatto_nodes(order))


class EquidistantTensorProductElementGroup(_LegendreTensorProductElementGroup):
    """Elemental discretization supplying a high-order quadrature rule
    with a number of nodes matching the number of polynomials in the tensor
    product basis, hence usable for differentiation and interpolation.
    Nodes sufficient for unisolvency are present on the boundary of the hypercube.

    Uses :func:`~modepy.equidistant_nodes`.
    """

    def __init__(self, mesh_el_group, order, index):
        from modepy.nodes import equidistant_nodes
        super().__init__(mesh_el_group, order, index,
                unit_nodes_1d=equidistant_nodes(1, order)[0])

# }}}


# {{{ group factories

class ElementGroupFactory:
    """
    .. function:: __call__(mesh_ele_group, node_nr_base)
    """
    pass


class HomogeneousOrderBasedGroupFactory(ElementGroupFactory):
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
        self.order = order
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
        self.order = order
        self.unit_nodes = unit_nodes

    def __call__(self, mesh_el_group, index):
        if not isinstance(mesh_el_group, _MeshSimplexElementGroup):
            raise TypeError("only mesh element groups of type '%s' "
                    "are supported" % _MeshSimplexElementGroup.__name__)

        return PolynomialGivenNodesElementGroup(
                mesh_el_group, self.order, self.unit_nodes, index)

# }}}


# {{{ group factories for tensor products

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
