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


from pytools import memoize_method, memoize_method_nested
import pyopencl as cl
import pyopencl.array  # noqa

__doc__ = """
.. autoclass:: Resampler
"""


class Resampler(object):
    """A container for state needed to perform upsampling between a
    *from_discr* and a *to_discr*.  Requires that *from_discr* and *to_discr*
    are based on the same :class:`meshmode.mesh.Mesh`
    """

    def __init__(self, to_discr, from_discr):
        if from_discr.mesh is not to_discr.mesh:
            raise ValueError("from_discr and to_discr "
                    "must be based on the same mesh.")

        if from_discr.cl_context != to_discr.cl_context:
            raise ValueError("from_discr and to_discr "
                    "must be based on the same OpenCL context.")

        self.cl_context = from_discr.cl_context

        self.from_discr = from_discr
        self.to_discr = to_discr

    @property
    def mesh(self):
        return self.from_discr.mesh

    @property
    def dim(self):
        return self.from_discr.dim

    @property
    def ambient_dim(self):
        return self.from_discr.ambient_dim

    # {{{ oversampling

    @memoize_method
    def _oversample_matrix(self, elgroup_index):
        import modepy as mp
        tgrp = self.from_discr.groups[elgroup_index]
        sgrp = self.to_discr.groups[elgroup_index]

        return mp.resampling_matrix(
                mp.simplex_onb(self.dim, tgrp.order),
                sgrp.unit_nodes, tgrp.unit_nodes)

    def __call__(self, queue, vec):
        @memoize_method_nested
        def knl():
            import loopy as lp
            knl = lp.make_kernel(
                """{[k,i,j]:
                    0<=k<nelements and
                    0<=i<n_to_nodes and
                    0<=j<n_from_nodes}""",
                "result[k,i] = sum(j, oversample_mat[i, j] * vec[k, j])",
                name="oversample")

            knl = lp.split_iname(knl, "i", 16, inner_tag="l.0")
            return lp.tag_inames(knl, dict(k="g.0"))

        if not isinstance(vec, cl.array.Array):
            return vec

        result = self.to_discr.empty(vec.dtype)

        for i_grp, (sgrp, tgrp) in enumerate(
                zip(self.to_discr.groups, self.from_discr.groups)):
            knl()(queue,
                    oversample_mat=self._oversample_matrix(i_grp),
                    result=sgrp.view(result), vec=tgrp.view(vec))

        return result

    # }}}

# vim: fdm=marker
