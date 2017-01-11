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
from pytools import memoize_method, memoize_in
import loopy as lp
import pyopencl as cl
import pyopencl.array  # noqa

__doc__ = """
.. autoclass:: ElementGroupBase
.. autoclass:: ElementGroupFactory
.. autoclass:: Discretization
"""


# {{{ element group base

class NoninterpolatoryElementGroupError(TypeError):
    pass


class ElementGroupBase(object):
    """Container for the :class:`Discretization` data corresponding to
    one :class:`meshmode.mesh.MeshElementGroup`.

    .. attribute :: mesh_el_group
    .. attribute :: order
    .. attribute :: node_nr_base

    .. autoattribute:: nelements
    .. autoattribute:: nunit_nodes
    .. autoattribute:: nnodes
    .. autoattribute:: dim
    .. automethod:: view

    .. method:: unit_nodes()

        Returns a :class:`numpy.ndarray` of shape ``(dim, nunit_nodes)``
        of reference coordinates of interpolation nodes.

    .. method:: basis()

        Returns a :class:`list` of basis functions that take arrays
        of shape ``(dim, n)`` and return an array of shape (n,)``
        (which performs evaluation of the basis function).

    .. method:: grad_basis()

        :returns: a :class:`tuple` of functions, each of  which
        accepts arrays of shape *(dims, npts)*
        and returns a :class:`tuple` of length *dims* containing
        the derivatives along each axis as an array of size *npts*.
        'Scalar' evaluation, by passing just one vector of length *dims*,
        is also supported.

    .. method:: diff_matrices()

        Return a :attr:`dim`-long :class:`tuple` of matrices of
        shape ``(nunit_nodes, nunit_nodes)``, each of which,
        when applied to an array of nodal values, take derivatives
        in the reference (r,s,t) directions.

    .. method:: weights()

        Returns an array of length :attr:`nunit_nodes` containing
        quadrature weights.
    """

    def __init__(self, mesh_el_group, order, node_nr_base):
        """
        :arg mesh_el_group: an instance of
            :class:`meshmode.mesh.MeshElementGroup`
        """
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

    @property
    def dim(self):
        return self.mesh_el_group.dim

    def _nodes(self):
        # Not cached, because the global nodes array is what counts.
        # This is just used to build that.

        return np.tensordot(
                self.mesh_el_group.nodes,
                self._from_mesh_interp_matrix(),
                (-1, -1))

    def view(self, global_array):
        """Return a view of *global_array* of shape ``(..., nelements,
        nunit_nodes)`` where *global_array* is of shape ``(..., nnodes)``,
        where *nnodes* is the global (per-discretization) node count.
        """

        return global_array[
                ..., self.node_nr_base:self.node_nr_base + self.nnodes] \
                .reshape(
                        global_array.shape[:-1]
                        + (self.nelements, self.nunit_nodes))

# }}}


# {{{ group factories

class ElementGroupFactory(object):
    """
    .. function:: __call__(mesh_ele_group, node_nr_base)
    """


class OrderBasedGroupFactory(ElementGroupFactory):
    def __init__(self, order):
        self.order = order

    def __call__(self, mesh_el_group, node_nr_base):
        if not isinstance(mesh_el_group, self.mesh_group_class):
            raise TypeError("only mesh element groups of type '%s' "
                    "are supported" % self.mesh_group_class.__name__)

        return self.group_class(mesh_el_group, self.order, node_nr_base)

# }}}


class Discretization(object):
    """An unstructured composite discretization.

    .. attribute:: real_dtype

    .. attribute:: complex_dtype

    .. attribute:: mesh

    .. attribute:: dim

    .. attribute:: ambient_dim

    .. attribute :: nnodes

    .. attribute :: groups

    .. automethod:: empty

    .. automethod:: zeros

    .. method:: nodes()

        shape: ``(ambient_dim, nnodes)``

    .. method:: num_reference_derivative(queue, ref_axes, vec)

    .. method:: quad_weights(queue)

        shape: ``(nnodes)``
    """

    def __init__(self, cl_ctx, mesh, group_factory, real_dtype=np.float64):
        self.cl_context = cl_ctx

        self.mesh = mesh
        self.nnodes = 0
        self.groups = []
        for mg in mesh.groups:
            ng = group_factory(mg, self.nnodes)
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

    def empty(self, queue=None, dtype=None, extra_dims=None, allocator=None):
        """Return an empty DOF vector.

        :arg dtype: type special value 'c' will result in a
            vector of dtype :attr:`self.complex_dtype`. If
            *None* (the default), a real vector will be returned.
        """
        if dtype is None:
            dtype = self.real_dtype
        elif dtype == "c":
            dtype = self.complex_dtype
        else:
            dtype = np.dtype(dtype)

        if queue is None:
            first_arg = self.cl_context
        else:
            first_arg = queue

        shape = (self.nnodes,)
        if extra_dims is not None:
            shape = extra_dims + shape

        return cl.array.empty(first_arg, shape, dtype=dtype, allocator=allocator)

    def zeros(self, queue, dtype=None, extra_dims=None, allocator=None):
        return self.empty(queue, dtype=dtype, extra_dims=extra_dims,
                allocator=allocator).fill(0)

    def num_reference_derivative(self, queue, ref_axes, vec):
        @memoize_in(self, "reference_derivative_knl")
        def knl():
            knl = lp.make_kernel(
                """{[k,i,j]:
                    0<=k<nelements and
                    0<=i,j<ndiscr_nodes}""",
                "result[k,i] = sum(j, diff_mat[i, j] * vec[k, j])",
                default_offset=lp.auto, name="diff")

            knl = lp.split_iname(knl, "i", 16, inner_tag="l.0")
            return lp.tag_inames(knl, dict(k="g.0"))

        result = self.empty(dtype=vec.dtype)

        for grp in self.groups:
            if grp.nelements == 0:
                continue

            mat = None
            for ref_axis in ref_axes:
                next_mat = grp.diff_matrices()[ref_axis]
                if mat is None:
                    mat = next_mat
                else:
                    mat = np.dot(next_mat, mat)

            knl()(queue, diff_mat=mat, result=grp.view(result), vec=grp.view(vec))

        return result

    def quad_weights(self, queue):
        @memoize_in(self, "quad_weights_knl")
        def knl():
            knl = lp.make_kernel(
                "{[k,i]: 0<=k<nelements and 0<=i<ndiscr_nodes}",
                "result[k,i] = weights[i]",
                name="quad_weights",
                default_offset=lp.auto)

            knl = lp.split_iname(knl, "i", 16, inner_tag="l.0")
            return lp.tag_inames(knl, dict(k="g.0"))

        result = self.empty(dtype=self.real_dtype)
        for grp in self.groups:
            if grp.nelements == 0:
                continue

            knl()(queue, result=grp.view(result), weights=grp.weights)
        return result

    @memoize_method
    def nodes(self):
        @memoize_in(self, "nodes_knl")
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
                name="nodes",
                default_offset=lp.auto)

            knl = lp.split_iname(knl, "i", 16, inner_tag="l.0")
            knl = lp.tag_inames(knl, dict(k="g.0"))
            knl = lp.tag_array_axes(knl, "result",
                    "stride:auto,stride:auto,stride:auto")
            return knl

        result = self.empty(dtype=self.real_dtype, extra_dims=(self.ambient_dim,))

        with cl.CommandQueue(self.cl_context) as queue:
            for grp in self.groups:
                if grp.nelements == 0:
                    continue

                meg = grp.mesh_el_group
                knl()(queue,
                        resampling_mat=grp.from_mesh_interp_matrix(),
                        result=grp.view(result), nodes=meg.nodes)

        return result


# vim: fdm=marker
