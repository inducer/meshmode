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


# {{{ element group base

class PolynomialElementGroupBase(object):
    """Container for the :class:`QBXGrid` data corresponding to
    one :class:`pytential.mesh.MeshElementGroup`.

    .. attribute :: grid
    .. attribute :: mesh_el_group
    .. attribute :: node_nr_base
    """

    def __init__(self, grid, mesh_el_group, order, node_nr_base):
        """
        :arg grid: an instance of :class:`QBXGridBase`
        :arg mesh_el_group: an instance of
            :class:`pytential.mesh.MeshElementGroup`
        """
        self.grid = grid
        self.mesh_el_group = mesh_el_group
        self.order = order
        self.node_nr_base = node_nr_base

    @property
    def nelements(self):
        return self.mesh_el_group.nelements

    @property
    def nunit_nodes(self):
        return self.unit_nodes.shape[-1]

    @property
    def nnodes(self):
        return self.nunit_nodes * self.nelements

    @memoize_method
    def _from_mesh_interp_matrix(self):
        meg = self.mesh_el_group
        return mp.resampling_matrix(
                mp.simplex_onb(meg.dim, meg.order),
                self.unit_nodes,
                meg.unit_nodes)

    def _nodes(self):
        # Not cached, because the global nodes array is what counts.
        # This is just used to build that.

        return np.tensordot(
                self.mesh_el_group.nodes,
                self._from_mesh_interp_matrix(),
                (-1, -1))

    def view(self, global_array):
        return global_array[
                ..., self.node_nr_base:self.node_nr_base + self.nnodes] \
                .reshape(
                        global_array.shape[:-1]
                        + (self.nelements, self.nunit_nodes))

# }}}


# {{{ element group

class PolynomialElementGroup(PolynomialElementGroupBase):
    @memoize_method
    def _quadrature_rule(self):
        dims = self.mesh_el_group.dim
        if dims == 1:
            return mp.LegendreGaussQuadrature(self.order)
        else:
            return mp.VioreanuRokhlinSimplexQuad(self.order, dims)

    @property
    @memoize_method
    def unit_nodes(self):
        return self._quadrature_rule().nodes

    @property
    @memoize_method
    def weights(self):
        return self._quadrature_rule().weights

# }}}


# {{{ discretization

class PolynomialElementDiscretization(Discretization):
    """An (unstructured) composite polynomial discretization without
    any specific opinion on how to evaluate layer potentials.

    .. attribute :: mesh
    .. attribute :: groups
    .. attribute :: nnodes

    .. autoattribute :: nodes
    """

    def __init__(self, cl_ctx, mesh, order, real_dtype=np.float64):
        """
        :arg order: A polynomial-order-like parameter passed unmodified to
            :attr:`group_class`. See subclasses for more precise definition.
        """

        self.cl_context = cl_ctx

        self.mesh = mesh
        self.nnodes = 0
        self.groups = []
        for mg in mesh.groups:
            ng = self.group_class(self, mg, order, self.nnodes)
            self.groups.append(ng)
            self.nnodes += ng.nnodes

        self.real_dtype = np.dtype(real_dtype)
        self.complex_dtype = {
                np.float32: np.complex64,
                np.float64: np.complex128
                }[self.real_dtype.type]

    @property
    def dim(self):
        return self.mesh.dim

    @property
    def ambient_dim(self):
        return self.mesh.ambient_dim

    def empty(self, dtype, queue=None, extra_dims=None):
        if queue is None:
            first_arg = self.cl_context
        else:
            first_arg = queue

        shape = (self.nnodes,)
        if extra_dims is not None:
            shape = extra_dims + shape

        return cl.array.empty(first_arg, shape, dtype=dtype)

    @memoize_method
    def _diff_matrices(self, grp):
        result = mp.differentiation_matrices(
                mp.simplex_onb(self.dim, grp.order),
                mp.grad_simplex_onb(self.dim, grp.order),
                grp.unit_nodes)

        if not isinstance(result, tuple):
            return (result,)
        else:
            return result

    def num_reference_derivative(
            self, queue, ref_axes, vec):
        @memoize_method_nested
        def knl():
            knl = lp.make_kernel(
                """{[k,i,j]:
                    0<=k<nelements and
                    0<=i,j<ndiscr_nodes}""",
                "result[k,i] = sum(j, diff_mat[i, j] * vec[k, j])",
                default_offset=lp.auto, name="diff")

            knl = lp.split_iname(knl, "i", 16, inner_tag="l.0")
            return lp.tag_inames(knl, dict(k="g.0"))

        result = self.empty(vec.dtype)

        for grp in self.groups:
            mat = None
            for ref_axis in ref_axes:
                next_mat = self._diff_matrices(grp)[ref_axis]
                if mat is None:
                    mat = next_mat
                else:
                    mat = np.dot(next_mat, mat)

            knl()(queue, diff_mat=mat, result=grp.view(result), vec=grp.view(vec))

        return result

    def quad_weights(self, queue):
        @memoize_method_nested
        def knl():
            knl = lp.make_kernel(
                "{[k,i]: 0<=k<nelements and 0<=i<ndiscr_nodes}",
                "result[k,i] = weights[i]",
                name="quad_weights")

            knl = lp.split_iname(knl, "i", 16, inner_tag="l.0")
            return lp.tag_inames(knl, dict(k="g.0"))

        result = self.empty(self.real_dtype)
        for grp in self.groups:
            knl()(queue, result=grp.view(result), weights=grp.weights)
        return result

    @memoize_method
    def _resampling_matrix(self, grp):
        meg = grp.mesh_el_group
        return mp.resampling_matrix(
                mp.simplex_onb(self.dim, meg.order),
                grp.unit_nodes, meg.unit_nodes)

    @memoize_method
    def nodes(self):
        @memoize_method_nested
        def knl():
            knl = lp.make_kernel(
                """{[d,k,i,j]:
                    0<=d<dims and
                    0<=k<nelements and
                    0<=i<ndiscr_nodes and
                    0<=j<nmesh_nodes}""",
                """
                    result[d, k, i] = \
                        sum(j, resampling_mat[i, j] * nodes[d, k, j])
                    """,
                name="nodes")

            knl = lp.split_iname(knl, "i", 16, inner_tag="l.0")
            return lp.tag_inames(knl, dict(k="g.0"))

        result = self.empty(self.real_dtype, extra_dims=(self.ambient_dim,))

        with cl.CommandQueue(self.cl_context) as queue:
            for grp in self.groups:
                meg = grp.mesh_el_group
                knl()(queue,
                        resampling_mat=self._resampling_matrix(grp),
                        result=grp.view(result), nodes=meg.nodes)

        return result

    group_class = PolynomialElementGroup

# }}}


# vim: fdm=marker
