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
from pytools import memoize_method


__doc__ = """

.. autoclass:: Discretization
"""


class Discretization(object):
    """Abstract interface for discretizations.

    .. attribute:: real_dtype

    .. attribute:: complex_dtype

    .. attribute:: mesh

    .. attribute:: dim

    .. attribute:: ambient_dim

    .. method::  empty(dtype, queue=None, extra_dims=None)

    .. method:: nodes()

        shape: ``(ambient_dim, nnodes)``

    .. method:: num_reference_derivative(queue, ref_axes, vec)

    .. method:: quad_weights(queue)

        shape: ``(nnodes)``

    .. rubric:: Layer potential source discretizations only

    .. method:: preprocess_optemplate(name, expr)

    .. method:: op_group_features(expr)

        Return a characteristic tuple by which operators that can be
        executed together can be grouped.

        *expr* is a subclass of
        :class:`pymbolic.primitives.LayerPotentialOperatorBase`.
    """

    @memoize_method
    def _integral_op(self):
        from pytential import sym, bind
        return bind(self, sym.integral(sym.var("integrand")))

    def integral(self, queue, x):
        return self._integral_op()(queue, integrand=x)

    @memoize_method
    def _norm_op(self, num_components):
        from pytential import sym, bind
        if num_components is not None:
            from pymbolic.primitives import make_sym_vector
            v = make_sym_vector("integrand", num_components)
            integrand = sym.real(np.dot(sym.conj(v), v))
        else:
            integrand = sym.abs(sym.var("integrand"))**2

        return bind(self, sym.integral(integrand))

    def norm(self, queue, x):
        from pymbolic.geometric_algebra import MultiVector
        if isinstance(x, MultiVector):
            x = x.as_vector(np.object)

        num_components = None
        if isinstance(x, np.ndarray):
            num_components, = x.shape

        norm_op = self._norm_op(num_components)
        from math import sqrt
        return sqrt(norm_op(queue, integrand=x))

# vim: fdm=marker
