from __future__ import division

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
.. autoclass:: PolynomialEquidistantSimplexElementGroup
.. autoclass:: LegendreGaussLobattoTensorProductElementGroup

Group factories
^^^^^^^^^^^^^^^

.. autoclass:: InterpolatoryQuadratureSimplexGroupFactory
.. autoclass:: QuadratureSimplexGroupFactory
.. autoclass:: PolynomialWarpAndBlendGroupFactory
.. autoclass:: PolynomialEquidistantSimplexGroupFactory
.. autoclass:: LegendreGaussLobattoTensorProductGroupFactory
"""

# FIXME Most of the loopy kernels will break as soon as we start using multiple
# element groups. That's because then the dimension-to-dimension stride will no
# longer just be the long axis of the array, but something entirely
# independent.  The machinery for this on the loopy end is there, in the form
# of the "stride:auto" dim tag, it just needs to be pushed through all the
# kernels.  Fortunately, this will fail in an obvious and noisy way, because
# loopy sees strides that it doesn't expect and complains.


from meshmode.discretization import ElementGroupBase


# {{{ base class for poynomial elements

class PolynomialElementGroupBase(ElementGroupBase):
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

class PolynomialSimplexElementGroupBase(ElementGroupBase):
    def is_orthogonal_basis(self):
        return self.dim <= 3

    def basis(self):
        if self.dim <= 3:
            return mp.simplex_onb(self.dim, self.order)
        else:
            return mp.simplex_monomial_basis(self.dim, self.order)

    def grad_basis(self):
        if self.dim <= 3:
            return mp.grad_simplex_onb(self.dim, self.order)
        else:
            return mp.grad_simplex_monomial_basis(self.dim, self.order)

    @memoize_method
    def from_mesh_interp_matrix(self):
        meg = self.mesh_el_group
        return mp.resampling_matrix(
                mp.simplex_best_available_basis(meg.dim, meg.order),
                self.unit_nodes,
                meg.unit_nodes)


class InterpolatoryQuadratureSimplexElementGroup(PolynomialSimplexElementGroupBase):
    """Elemental discretization supplying a high-order quadrature rule
    with a number of nodes matching the number of polynomials in :math:`P^k`,
    hence usable for differentiation and interpolation.

    No interpolation nodes are present on the boundary of the simplex.
    """

    @memoize_method
    def _quadrature_rule(self):
        dims = self.mesh_el_group.dim
        if dims == 1:
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


class QuadratureSimplexElementGroup(PolynomialSimplexElementGroupBase):
    """Elemental discretization supplying a high-order quadrature rule
    with a number of nodes which does not necessarily match the number of
    polynomials in :math:`P^k`. This discretization therefore excels at
    quadarature, but is not necessarily usable for interpolation.

    No interpolation nodes are present on the boundary of the simplex.
    """

    @memoize_method
    def _quadrature_rule(self):
        dims = self.mesh_el_group.dim
        if dims == 1:
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
    """
    @property
    @memoize_method
    def unit_nodes(self):
        dim = self.mesh_el_group.dim
        result = mp.warp_and_blend_nodes(dim, self.order)

        dim2, nunit_nodes = result.shape
        assert dim2 == dim
        return result


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

        dim2, nunit_nodes = result.shape
        assert dim2 == dim
        return result

# }}}


# {{{ concrete element groups for tensor product elements

class LegendreGaussLobattoTensorProductElementGroup(PolynomialElementGroupBase):
    def is_orthogonal_basis(self):
        return True

    def basis(self):
        from modepy.modes import tensor_product_basis, jacobi
        from functools import partial
        return tensor_product_basis(
                self.dim, tuple(
                    partial(jacobi, 0, 0, i)
                    for i in range(self.order+1)))

    def grad_basis(self):
        raise NotImplementedError()

    @property
    @memoize_method
    def unit_nodes(self):
        from modepy.nodes import tensor_product_nodes
        from modepy.quadrature.jacobi_gauss import legendre_gauss_lobatto_nodes
        return tensor_product_nodes(
                self.dim, legendre_gauss_lobatto_nodes(self.order))

    @memoize_method
    def from_mesh_interp_matrix(self):
        from modepy.modes import tensor_product_basis, jacobi
        from functools import partial
        meg = self.mesh_el_group

        basis = tensor_product_basis(
                self.dim, tuple(
                    partial(jacobi, 0, 0, i)
                    for i in range(meg.order+1)))

        return mp.resampling_matrix(basis, self.unit_nodes, meg.unit_nodes)

# }}}


# {{{ group factories

class ElementGroupFactory(object):
    pass


class HomogeneousOrderBasedGroupFactory(ElementGroupFactory):
    def __init__(self, order):
        self.order = order

    def __call__(self, mesh_el_group, node_nr_base):
        if not isinstance(mesh_el_group, self.mesh_group_class):
            raise TypeError("only mesh element groups of type '%s' "
                    "are supported" % self.mesh_group_class.__name__)

        return self.group_class(mesh_el_group, self.order, node_nr_base)


class OrderAndTypeBasedGroupFactory(ElementGroupFactory):
    def __init__(self, order, simplex_group_class, tensor_product_group_class):
        self.order = order
        self.simplex_group_class = simplex_group_class
        self.tensor_product_group_class = tensor_product_group_class

    def __call__(self, mesh_el_group, node_nr_base):
        if isinstance(mesh_el_group, _MeshSimplexElementGroup):
            group_class = self.simplex_group_class
        elif isinstance(mesh_el_group, _MeshTensorProductElementGroup):
            group_class = self.tensor_product_group_class
        else:
            raise TypeError("only mesh element groups of type '%s' and '%s' "
                    "are supported" % (
                        _MeshSimplexElementGroup.__name__,
                        _MeshTensorProductElementGroup.__name__))

        return group_class(mesh_el_group, self.order, node_nr_base)


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


class PolynomialEquidistantSimplexGroupFactory(HomogeneousOrderBasedGroupFactory):
    """
    .. versionadded:: 2016.1
    """

    mesh_group_class = _MeshSimplexElementGroup
    group_class = PolynomialEquidistantSimplexElementGroup

# }}}


class LegendreGaussLobattoTensorProductGroupFactory(
        HomogeneousOrderBasedGroupFactory):
    mesh_group_class = _MeshTensorProductElementGroup
    group_class = LegendreGaussLobattoTensorProductElementGroup

# }}}


# vim: fdm=marker
