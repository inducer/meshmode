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
from pytools import memoize_method, memoize_method_nested
from meshmode.discretization import Discretization
from meshmode.mesh import SimplexElementGroup as _MeshSimplexElementGroup

import pyopencl as cl

import loopy as lp
import modepy as mp

__doc__ = """
.. autoclass:: PolynomialElementDiscretization
"""

# FIXME Most of the loopy kernels will break as soon as we start using multiple
# element groups. That's because then the dimension-to-dimension stride will no
# longer just be the long axis of the array, but something entirely
# independent.  The machinery for this on the loopy end is there, in the form
# of the "stride:auto" dim tag, it just needs to be pushed through all the
# kernels.  Fortunately, this will fail in an obvious and noisy way, because
# loopy sees strides that it doesn't expect and complains.


from meshmode.discretization import ElementGroupBase


# {{{ concrete element groups

class PolynomialSimplexElementGroupBase(ElementGroupBase):
    def basis(self):
        return mp.simplex_onb(self.dim, self.order)

    @memoize_method
    def from_mesh_interp_matrix(self):
        meg = self.mesh_el_group
        return mp.resampling_matrix(
                mp.simplex_onb(meg.dim, meg.order),
                self.unit_nodes,
                meg.unit_nodes)

    @memoize_method
    def diff_matrices(self):
        result = mp.differentiation_matrices(
                self.basis(),
                mp.grad_simplex_onb(self.dim, self.order),
                self.unit_nodes)

        if not isinstance(result, tuple):
            return (result,)
        else:
            return result

    @memoize_method
    def resampling_matrix(self):
        meg = self.mesh_el_group
        return mp.resampling_matrix(
                mp.simplex_onb(self.dim, meg.order),
                self.unit_nodes, meg.unit_nodes)


class QuadratureSimplexElementGroup(PolynomialSimplexElementGroupBase):
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


class PolynomialWarpAndBlendElementGroup(PolynomialSimplexElementGroupBase):
    @property
    @memoize_method
    def unit_nodes(self):
        dim = self.mesh_el_group.dim
        result = mp.warp_and_blend_nodes(dim, self.order)

        dim2, nunit_nodes = result.shape
        assert dim2 == dim
        return result

    @property
    @memoize_method
    def weights(self):
        raise NotImplementedError()

# }}}


# {{{ group factories

class ElementGroupFactory(object):
    pass


class OrderBasedGroupFactory(ElementGroupFactory):
    def __init__(self, order):
        self.order = order

    def __call__(self, mesh_el_group, node_nr_base):
        if not isinstance(mesh_el_group, self.mesh_group_class):
            raise TypeError("only mesh element groups of type '%s' "
                    "are supported" % self.mesh_group_class.__name__)

        return self.group_class(mesh_el_group, self.order, node_nr_base)


class QuadratureSimplexGroupFactory(OrderBasedGroupFactory):
    mesh_group_class = _MeshSimplexElementGroup
    group_class = QuadratureSimplexElementGroup


class PolynomialWarpAndBlendGroupFactory(OrderBasedGroupFactory):
    mesh_group_class = _MeshSimplexElementGroup
    group_class = PolynomialWarpAndBlendElementGroup

# }}}


# vim: fdm=marker
