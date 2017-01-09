from __future__ import division, absolute_import

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

from six.moves import range

import numpy as np
import numpy.linalg as la
import modepy as mp


__doc__ = """

Curves
------

.. autofunction:: make_curve_mesh

Curve parametrizations
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: ellipse
.. autofunction:: cloverleaf
.. autofunction:: starfish
.. autofunction:: drop
.. autofunction:: n_gon
.. autofunction:: qbx_peanut

Surfaces
--------

.. autofunction:: generate_icosahedron
.. autofunction:: generate_icosphere
.. autofunction:: generate_torus

Volumes
-------

.. autofunction:: generate_box_mesh
.. autofunction:: generate_regular_rect_mesh
.. autofunction:: generate_warped_rect_mesh

"""


# {{{ test curve parametrizations

def ellipse(aspect_ratio, t):
    """
    :arg t: the parametrization, runs from [0,1)
    :return: an array of shape *(2, npoints)*
    """

    ilength = 2*np.pi
    t = t*ilength
    return np.vstack([
        np.cos(t),
        np.sin(t)/aspect_ratio,
        ])


def cloverleaf(t):
    """
    :arg t: the parametrization, runs from [0,1)
    :return: an array of shape *(2, npoints)*
    """

    ilength = 2*np.pi
    t = t*ilength

    a = 0.3
    b = 3

    return np.vstack([
        np.cos(t)+a*np.sin(b*t),
        np.sin(t)-a*np.cos(b*t)
        ])


def starfish(t):
    """
    :arg t: the parametrization, runs from [0,1)
    :return: an array of shape *(2, npoints)*
    """

    ilength = 2*np.pi
    t = t*ilength

    wave = 1+0.25*np.sin(5*t)

    return np.vstack([
        np.cos(t)*wave,
        np.sin(t)*wave,
        ])


def drop(t):
    """
    :arg t: the parametrization, runs from [0,1)
    :return: an array of shape *(2, npoints)*
    """

    ilength = np.pi
    t = t*ilength

    return 1.7 * np.vstack([
        np.sin(t)-0.5,
        0.5*(np.cos(t)*(t-np.pi)*t),
        ])


def n_gon(n_corners, t):
    """
    :arg t: the parametrization, runs from [0,1)
    :return: an array of shape *(2, npoints)*
    """

    t = t*n_corners

    result = np.empty((2,)+t.shape)

    for side in range(n_corners):
        indices = np.where((side <= t) & (t < side+1))

        startp = np.array([
            np.cos(2*np.pi/n_corners * side),
            np.sin(2*np.pi/n_corners * side),
            ])[:, np.newaxis]
        endp = np.array([
            np.cos(2*np.pi/n_corners * (side+1)),
            np.sin(2*np.pi/n_corners * (side+1)),
            ])[:, np.newaxis]

        tau = t[indices]-side
        result[:, indices] = (1-tau)*startp + tau*endp

    return result


def qbx_peanut(t):
    ilength = 2*np.pi
    t = t*ilength

    sin = np.sin
    cos = np.cos
    pi = np.pi

    return np.vstack([
        0.75*cos(t-0.25*pi)*(1+0.3*sin(2*t)),
        sin(t-0.25*pi)*(1+0.3*sin(2*t))
        ])

# }}}


# {{{ make_curve_mesh

def make_curve_mesh(curve_f, element_boundaries, order,
        unit_nodes=None,
        node_vertex_consistency_tolerance=None,
        return_parametrization_points=False):
    """
    :arg curve_f: A callable representing a parametrization for a curve,
        accepting a vector of point locations and returning
        an array of shape *(2, npoints)*.
    :arg element_boundaries: a vector of element boundary locations in
        :math:`[0,1]`, in order. 0 must be the first entry, 1 the
        last one.
    :arg unit_nodes: if given, the unit nodes to use. Must have shape
        ``(dim, nnoodes)``.
    :returns: a :class:`meshmode.mesh.Mesh`, or if *return_parametrization_points*
        is True, a tuple ``(mesh, par_points)``, where *par_points* is an array of
        parametrization points.
    """

    assert element_boundaries[0] == 0
    assert element_boundaries[-1] == 1
    nelements = len(element_boundaries) - 1

    if unit_nodes is None:
        unit_nodes = mp.warp_and_blend_nodes(1, order)
    nodes_01 = 0.5*(unit_nodes+1)

    vertices = curve_f(element_boundaries)

    el_lengths = np.diff(element_boundaries)
    el_starts = element_boundaries[:-1]

    # (el_nr, node_nr)
    t = el_starts[:, np.newaxis] + el_lengths[:, np.newaxis]*nodes_01
    t = t.ravel()
    nodes = curve_f(t).reshape(vertices.shape[0], nelements, -1)

    from meshmode.mesh import Mesh, SimplexElementGroup
    egroup = SimplexElementGroup(
            order,
            vertex_indices=np.vstack([
                np.arange(nelements, dtype=np.int32),
                np.arange(1, nelements+1, dtype=np.int32) % nelements,
                ]).T,
            nodes=nodes,
            unit_nodes=unit_nodes)

    mesh = Mesh(
            vertices=vertices, groups=[egroup],
            nodal_adjacency=None,
            facial_adjacency_groups=None,
            node_vertex_consistency_tolerance=node_vertex_consistency_tolerance)

    if return_parametrization_points:
        return mesh, t
    else:
        return mesh

# }}}


# {{{ make_group_from_vertices

def make_group_from_vertices(vertices, vertex_indices, order,
        group_factory=None):
    # shape: (dim, nelements, nvertices)
    el_vertices = vertices[:, vertex_indices]

    from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup

    if group_factory is None:
        group_factory = SimplexElementGroup

    if issubclass(group_factory, SimplexElementGroup):
        el_origins = el_vertices[:, :, 0][:, :, np.newaxis]
        # ambient_dim, nelements, nspan_vectors
        spanning_vectors = (
                el_vertices[:, :, 1:] - el_origins)

        nspan_vectors = spanning_vectors.shape[-1]
        dim = nspan_vectors

        # dim, nunit_nodes
        if dim <= 3:
            unit_nodes = mp.warp_and_blend_nodes(dim, order)
        else:
            unit_nodes = mp.equidistant_nodes(dim, order)

        unit_nodes_01 = 0.5 + 0.5*unit_nodes

        nodes = np.einsum(
                "si,des->dei",
                unit_nodes_01, spanning_vectors) + el_origins

    elif issubclass(group_factory, TensorProductElementGroup):
        nelements, nvertices = vertex_indices.shape

        dim = 0
        while True:
            if nvertices == 2**dim:
                break
            if nvertices < 2**dim:
                raise ValueError("invalid number of vertices for tensor-product "
                        "elements, must be power of two")
            dim += 1

        from modepy.quadrature.jacobi_gauss import legendre_gauss_lobatto_nodes
        from modepy.nodes import tensor_product_nodes
        unit_nodes = tensor_product_nodes(dim, legendre_gauss_lobatto_nodes(order))
        # shape: (dim, nnodes)
        unit_nodes_01 = 0.5 + 0.5*unit_nodes

        _, nnodes = unit_nodes.shape

        from pytools import generate_nonnegative_integer_tuples_below as gnitb
        id_tuples = list(gnitb(2, dim))
        assert len(id_tuples) == nvertices

        vdm = np.empty((nvertices, nvertices))
        for i, vertex_tuple in enumerate(id_tuples):
            for j, func_tuple in enumerate(id_tuples):
                vertex_ref = np.array(vertex_tuple, dtype=np.float64)
                vdm[i, j] = np.prod(vertex_ref**func_tuple)

        # shape: (dim, nelements, nvertices)
        coeffs = np.empty((dim, nelements, nvertices))
        for d in range(dim):
            coeffs[d] = la.solve(vdm, el_vertices[d].T).T

        vdm_nodes = np.zeros((nnodes, nvertices))
        for j, func_tuple in enumerate(id_tuples):
            vdm_nodes[:, j] = np.prod(
                    unit_nodes_01 ** np.array(func_tuple).reshape(-1, 1),
                    axis=0)

        nodes = np.einsum("ij,dej->dei", vdm_nodes, coeffs)

    else:
        raise ValueError("unsupported value for 'group_factory': %s"
                % group_factory)

    # make contiguous
    nodes = nodes.copy()

    return group_factory(
            order, vertex_indices, nodes,
            unit_nodes=unit_nodes)

# }}}


# {{{ generate_icosahedron

def generate_icosahedron(r, order):
    # http://en.wikipedia.org/w/index.php?title=Icosahedron&oldid=387737307

    phi = (1+5**(1/2))/2

    from pytools import flatten
    vertices = np.array(sorted(flatten([
            (0, pm1*1, pm2*phi),
            (pm1*1, pm2*phi, 0),
            (pm1*phi, 0, pm2*1)]
            for pm1 in [-1, 1]
            for pm2 in [-1, 1]))).T.copy()

    top_ring = [11, 7, 1, 2, 8]
    bottom_ring = [10, 9, 3, 0, 4]
    bottom_point = 6
    top_point = 5

    tris = []
    l = len(top_ring)
    for i in range(l):
        tris.append([top_ring[i], top_ring[(i+1) % l], top_point])
        tris.append([bottom_ring[i], bottom_point, bottom_ring[(i+1) % l], ])
        tris.append([bottom_ring[i], bottom_ring[(i+1) % l], top_ring[i]])
        tris.append([top_ring[i], bottom_ring[(i+1) % l], top_ring[(i+1) % l]])

    vertices *= r/la.norm(vertices[:, 0])

    vertex_indices = np.array(tris, dtype=np.int32)

    grp = make_group_from_vertices(vertices, vertex_indices, order)

    from meshmode.mesh import Mesh
    return Mesh(
            vertices, [grp],
            nodal_adjacency=None,
            facial_adjacency_groups=None)

# }}}


# {{{ generate_icosphere

def generate_icosphere(r, order):
    mesh = generate_icosahedron(r, order)

    grp, = mesh.groups

    grp = grp.copy(
            nodes=grp.nodes * r / np.sqrt(np.sum(grp.nodes**2, axis=0)))

    from meshmode.mesh import Mesh
    return Mesh(
            mesh.vertices, [grp],
            nodal_adjacency=None,
            facial_adjacency_groups=None)

# }}}


# {{{ generate_torus_and_cycle_vertices

def generate_torus_and_cycle_vertices(r_outer, r_inner,
        n_outer=20, n_inner=10, order=1):
    a = r_outer
    b = r_inner
    u, v = np.mgrid[0:2*np.pi:2*np.pi/n_outer, 0:2*np.pi:2*np.pi/n_inner]

    # http://www.math.hmc.edu/~gu/curves_and_surfaces/surfaces/torus.html
    x = np.cos(u)*(a+b*np.cos(v))
    y = np.sin(u)*(a+b*np.cos(v))
    z = b*np.sin(v)
    vertices = (np.vstack((x[np.newaxis], y[np.newaxis], z[np.newaxis]))
            .transpose(0, 2, 1).copy().reshape(3, -1))

    def idx(i, j):
        return (i % n_outer) + (j % n_inner) * n_outer
    vertex_indices = ([(idx(i, j), idx(i+1, j), idx(i, j+1))
            for i in range(n_outer) for j in range(n_inner)]
            + [(idx(i+1, j), idx(i+1, j+1), idx(i, j+1))
            for i in range(n_outer) for j in range(n_inner)])

    vertex_indices = np.array(vertex_indices, dtype=np.int32)
    grp = make_group_from_vertices(vertices, vertex_indices, order)

    # ambient_dim, nelements, nunit_nodes
    nodes = grp.nodes.copy()

    major_theta = np.arctan2(nodes[1], nodes[0])
    rvec = np.array([
        np.cos(major_theta),
        np.sin(major_theta),
        np.zeros_like(major_theta)])

    #               ^
    #               |
    # --------------+----.
    #           /   |     \
    #          /    |  _-- \
    #         |     |.^  | | y
    #         |     +------+--->
    #         |       x    |
    #          \          /
    #           \        /
    # ------------------'

    x = np.sum(nodes*rvec, axis=0) - a

    minor_theta = np.arctan2(nodes[2], x)

    nodes[0] = np.cos(major_theta)*(a+b*np.cos(
        minor_theta))
    nodes[1] = np.sin(major_theta)*(a+b*np.cos(
        minor_theta))
    nodes[2] = b*np.sin(minor_theta)

    from meshmode.mesh import Mesh
    return (
            Mesh(
                vertices, [grp.copy(nodes=nodes)],
                nodal_adjacency=None,
                facial_adjacency_groups=None),
            [idx(i, 0) for i in range(n_outer)],
            [idx(0, j) for j in range(n_inner)])

# }}}


def generate_torus(r_outer, r_inner, n_outer=20, n_inner=10, order=1):
    mesh, a_cycle, b_cycle = generate_torus_and_cycle_vertices(
            r_outer, r_inner, n_outer, n_inner, order)
    return mesh


# {{{ generate_box_mesh

def generate_box_mesh(axis_coords, order=1, coord_dtype=np.float64,
        group_factory=None):
    """Create a semi-structured mesh.

    :param axis_coords: a tuple with a number of entries corresponding
        to the number of dimensions, with each entry a numpy array
        specifying the coordinates to be used along that axis.
    :param group_factory: One of :class:`meshmode.mesh.SimplexElementGroup`
        or :class:`meshmode.mesh.TensorProductElementGroup`.

    .. versionchanged:: 2017.1

        *group_factory* parameter added.
    """

    for iaxis, axc in enumerate(axis_coords):
        if len(axc) < 2:
            raise ValueError("need at least two points along axis %d"
                    % (iaxis+1))

    dim = len(axis_coords)

    shape = tuple(len(axc) for axc in axis_coords)

    from pytools import product
    nvertices = product(shape)

    vertex_indices = np.arange(nvertices).reshape(*shape, order="F")

    vertices = np.empty((dim,)+shape, dtype=coord_dtype)
    for idim in range(dim):
        vshape = (shape[idim],) + (1,)*idim
        vertices[idim] = axis_coords[idim].reshape(*vshape)

    vertices = vertices.reshape(dim, -1)

    from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
    if group_factory is None:
        group_factory = SimplexElementGroup

    if issubclass(group_factory, SimplexElementGroup):
        is_tp = False
    elif issubclass(group_factory, TensorProductElementGroup):
        is_tp = True
    else:
        raise ValueError("unsupported value for 'group_factory': %s"
                % group_factory)

    el_vertices = []

    if dim == 1:
        for i in range(shape[0]-1):
            # a--b

            a = vertex_indices[i]
            b = vertex_indices[i+1]

            el_vertices.append((a, b,))

    elif dim == 2:
        for i in range(shape[0]-1):
            for j in range(shape[1]-1):

                # c--d
                # |  |
                # a--b

                a = vertex_indices[i, j]
                b = vertex_indices[i+1, j]
                c = vertex_indices[i, j+1]
                d = vertex_indices[i+1, j+1]

                if is_tp:
                    el_vertices.append((a, b, c, d))
                else:
                    el_vertices.append((a, b, c))
                    el_vertices.append((d, c, b))

    elif dim == 3:
        for i in range(shape[0]-1):
            for j in range(shape[1]-1):
                for k in range(shape[2]-1):

                    a000 = vertex_indices[i, j, k]
                    a001 = vertex_indices[i, j, k+1]
                    a010 = vertex_indices[i, j+1, k]
                    a011 = vertex_indices[i, j+1, k+1]

                    a100 = vertex_indices[i+1, j, k]
                    a101 = vertex_indices[i+1, j, k+1]
                    a110 = vertex_indices[i+1, j+1, k]
                    a111 = vertex_indices[i+1, j+1, k+1]

                    if is_tp:
                        el_vertices.append(
                                (a000, a001, a010, a011,
                                    a100, a101, a110, a111))

                    else:
                        el_vertices.append((a000, a100, a010, a001))
                        el_vertices.append((a101, a100, a001, a010))
                        el_vertices.append((a101, a011, a010, a001))

                        el_vertices.append((a100, a010, a101, a110))
                        el_vertices.append((a011, a010, a110, a101))
                        el_vertices.append((a011, a111, a101, a110))

    else:
        raise NotImplementedError("box meshes of dimension %d"
                % dim)

    el_vertices = np.array(el_vertices, dtype=np.int32)

    grp = make_group_from_vertices(
            vertices.reshape(dim, -1), el_vertices, order,
            group_factory=group_factory)

    from meshmode.mesh import Mesh
    return Mesh(vertices, [grp],
            nodal_adjacency=None,
            facial_adjacency_groups=None)

# }}}


# {{{ generate_regular_rect_mesh

def generate_regular_rect_mesh(a=(0, 0), b=(1, 1), n=(5, 5), order=1):
    """Create a semi-structured rectangular mesh.

    :param a: the lower left hand point of the rectangle
    :param b: the upper right hand point of the rectangle
    :param n: a tuple of integers indicating the total number of points
      on [a,b].
    """
    if min(n) < 2:
        raise ValueError("need at least two points in each direction")

    axis_coords = [np.linspace(a_i, b_i, n_i)
            for a_i, b_i, n_i in zip(a, b, n)]

    return generate_box_mesh(axis_coords, order=order)

# }}}


# {{{ generate_warped_rect_mesh

def generate_warped_rect_mesh(dim, order, n):
    """Generate a mesh of a warped line/square/cube. Mainly useful for testing
    functionality with curvilinear meshes.
    """

    assert dim in [1, 2, 3]
    mesh = generate_regular_rect_mesh(
            a=(-0.5,)*dim, b=(0.5,)*dim,
            n=(n,)*dim, order=order)

    def m(x):
        result = np.empty_like(x)
        result[0] = (
                1.5*x[0] + np.cos(x[0])
                + 0.1*np.sin(10*x[1]))
        result[1] = (
                0.05*np.cos(10*x[0])
                + 1.3*x[1] + np.sin(x[1]))
        if len(x) == 3:
            result[2] = x[2] + np.sin(x[0] / 2) / 2
        return result

    from meshmode.mesh.processing import map_mesh
    return map_mesh(mesh, m)

# }}}


# vim: fdm=marker
