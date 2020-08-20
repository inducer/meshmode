__copyright__ = "Copyright (C) 2020 Benjamin Sepanski"

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
import numpy.linalg as la


__doc__ = """
.. autofunction:: get_affine_reference_simplex_mapping
.. autofunction:: get_finat_element_unit_nodes
"""


# {{{ Map between reference simplices

def get_affine_reference_simplex_mapping(ambient_dim, firedrake_to_meshmode=True):
    """
    Returns a function which takes a numpy array points
    on one reference cell and maps each
    point to another using a positive affine map.

    :arg ambient_dim: The spatial dimension
    :arg firedrake_to_meshmode: If true, the returned function maps from
        the firedrake reference element to
        meshmode, if false maps from
        meshmode to firedrake. More specifically,
        :mod:`firedrake` uses the standard :mod:`FIAT`
        simplex and :mod:`meshmode` uses
        :mod:`modepy`'s
        `unit coordinates <https://documen.tician.de/modepy/nodes.html>`_.
    :return: A function which takes a numpy array of *n* points with
             shape *(dim, n)* on one reference cell and maps
             each point to another using a positive affine map.
             Note that the returned function performs
             no input validation.
    """
    # validate input
    if not isinstance(ambient_dim, int):
        raise TypeError("'ambient_dim' must be an int, not "
                        f"'{type(ambient_dim)}'")
    if ambient_dim < 0:
        raise ValueError("'ambient_dim' must be non-negative")
    if not isinstance(firedrake_to_meshmode, bool):
        raise TypeError("'firedrake_to_meshmode' must be a bool, not "
                        f"'{type(firedrake_to_meshmode)}'")

    from FIAT.reference_element import ufc_simplex
    from modepy.tools import unit_vertices
    # Get the unit vertices from each system,
    # each stored with shape *(dim, nunit_vertices)*
    firedrake_unit_vertices = np.array(ufc_simplex(ambient_dim).vertices).T
    modepy_unit_vertices = unit_vertices(ambient_dim).T

    if firedrake_to_meshmode:
        from_verts = firedrake_unit_vertices
        to_verts = modepy_unit_vertices
    else:
        from_verts = modepy_unit_vertices
        to_verts = firedrake_unit_vertices

    # Compute matrix A and vector b so that A f_i + b -> t_i
    # for each "from" vertex f_i and corresponding "to" vertex t_i
    assert from_verts.shape == to_verts.shape
    dim, nvects = from_verts.shape

    # If we only have one vertex, have A = I and b = to_vert - from_vert
    if nvects == 1:
        shift = to_verts[:, 0] - from_verts[:, 0]

        def affine_map(points):
            return points + shift[:, np.newaxis]
    # Otherwise, we have to solve for A and b
    else:
        # span verts: v1 - v0, v2 - v0, ...
        from_span_verts = from_verts[:, 1:] - from_verts[:, 0, np.newaxis]
        to_span_verts = to_verts[:, 1:] - to_verts[:, 0, np.newaxis]
        # mat maps (fj - f0) -> (tj - t0), our "A"
        mat = la.solve(from_span_verts, to_span_verts)
        # A f0 + b -> t0 so b = t0 - A f0
        shift = to_verts[:, 0] - np.matmul(mat, from_verts[:, 0])

        # Explicitly ensure A is positive
        if la.det(mat) < 0:
            from meshmode.mesh.processing import get_simplex_element_flip_matrix
            flip_matrix = get_simplex_element_flip_matrix(1, to_verts)
            mat = np.matmul(flip_matrix, mat)

        def affine_map(points):
            return np.matmul(mat, points) + shift[:, np.newaxis]

    return affine_map

# }}}


# {{{ Get firedrake unit nodes

def get_finat_element_unit_nodes(finat_element):
    """
    Returns the unit nodes used by the :mod:`finat` element in firedrake's
    (equivalently, :mod:`finat`/:mod:`FIAT`'s) reference coordinates

    :arg finat_element: An instance of one of the following :mod:`finat`
        elements

        * :class:`finat.fiat_elements.Lagrange`
        * :class:`finat.fiat_elements.DiscontinuousLagrange`
        * :class:`finat.fiat_elements.CrouzeixRaviart`
        * :class:`finat.spectral.GaussLobattoLegendre`
        * :class:`finat.spectral.GaussLegendre`

    :return: A numpy array of shape *(dim, nunit_dofs)* holding the unit
             nodes used by this element. *dim* is the dimension spanned
             by the finat element's reference element
             (see its ``cell`` attribute)
    """
    from finat.fiat_elements import (
        Lagrange, DiscontinuousLagrange, CrouzeixRaviart)
    from finat.spectral import GaussLobattoLegendre, GaussLegendre
    from FIAT.reference_element import Simplex
    allowed_finat_elts = (Lagrange, DiscontinuousLagrange, CrouzeixRaviart,
                          GaussLobattoLegendre, GaussLegendre)
    if not isinstance(finat_element, allowed_finat_elts):
        raise TypeError("'finat_element' is of unexpected type "
                        f"{type(finat_element)}. 'finat_element' must be of "
                        "one of the following types: {allowed_finat_elts}")
    if not isinstance(finat_element.cell, Simplex):
        raise TypeError("Reference element of the finat element MUST be a"
                        " simplex, i.e. 'finat_element's *cell* attribute must"
                        " be of type FIAT.reference_element.Simplex, not "
                        f"'{type(finat_element.cell)}'")
    # We insisted 'finat_element._element' was a
    # FIAT.finite_element.CiarletElement,
    # so the finat_element._element.dual.nodes ought to represent
    # nodal dofs
    #
    # point evaluators is a list of functions *p_0,...,p_{n-1}*.
    # *p_i(f)* evaluates function *f* at node *i* (stored as a tuple),
    # so to recover node *i* we need to evaluate *p_i* at the identity
    # function
    point_evaluators = finat_element._element.dual.nodes
    unit_nodes = [p(lambda x: x) for p in point_evaluators]
    return np.array(unit_nodes).T

# }}}

# vim: foldmethod=marker
