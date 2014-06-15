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
import numpy.linalg as la
import modepy as mp


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


def make_curve_mesh(curve_f, element_boundaries, order):
    """
    :arg curve_f: A callable representing a parametrization for a curve,
        accepting a vector of point locations and returning
        an array of shape *(2, npoints)*.
    :arg element_boundaries: a vector of element boundary locations in
        :math:`[0,1]`, in order. 0 must be the first entry, 1 the
        last one.
    :returns: a :class:`meshmode.mesh.Mesh`
    """

    assert element_boundaries[0] == 0
    assert element_boundaries[-1] == 1
    nelements = len(element_boundaries) - 1

    unodes = mp.warp_and_blend_nodes(1, order)
    nodes_01 = 0.5*(unodes+1)

    vertices = curve_f(element_boundaries)

    el_lengths = np.diff(element_boundaries)
    el_starts = element_boundaries[:-1]

    # (el_nr, node_nr)
    t = el_starts[:, np.newaxis] + el_lengths[:, np.newaxis]*nodes_01
    nodes = curve_f(t.ravel()).reshape(vertices.shape[0], nelements, -1)

    from meshmode.mesh import Mesh, MeshElementGroup
    egroup = MeshElementGroup(
            order,
            vertex_indices=np.vstack([
                np.arange(nelements),
                np.arange(1, nelements+1) % nelements,
                ]).T,
            nodes=nodes,
            unit_nodes=unodes)

    return Mesh(vertices=vertices, groups=[egroup])


def generate_icosahedron(r):
    # http://en.wikipedia.org/w/index.php?title=Icosahedron&oldid=387737307

    phi = (1+5**(1/2))/2

    from pytools import flatten
    pts = np.array(sorted(flatten([
            (0, pm1*1, pm2*phi),
            (pm1*1, pm2*phi, 0),
            (pm1*phi, 0, pm2*1)]
            for pm1 in [-1, 1]
            for pm2 in [-1, 1])))

    from meshmode.mesh import Mesh
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

    pts *= r/la.norm(pts[0])

    return Mesh(pts, tris)


def generate_icosphere(r, refinements, order=1):
    mesh = generate_icosahedron(r)

    from meshmode.mesh import Mesh
    from meshmode.mesh import bisect_geometry
    for i in range(refinements):
        mesh, _ = bisect_geometry(mesh)

        pts = mesh.vertex_coordinates
        pt_norms = np.sum(pts**2, axis=-1)**0.5
        pts = pts * r / pt_norms[:, np.newaxis]

        mesh = Mesh(pts, mesh.elements)

    if order > 1:
        from meshmode.mesh.trianglemaps import interv_pts_on_unit
        eq_nodes = interv_pts_on_unit(order, 0, 1)

        # indices: triangle #, node #, xyz axis
        mapped_eq_nodes = np.empty((len(mesh), len(eq_nodes), 3))

        for iel, el_nodes in enumerate(mesh.elements):
            n1, n2, n3 = mesh.vertex_coordinates[el_nodes]
            a = np.vstack([n2-n1, n3-n1]).T
            b = n1

            mapped_eq_nodes[iel] = np.dot(a, eq_nodes.T).T + b

        mn = mapped_eq_nodes
        mapped_eq_nodes = \
                r * mn / np.sqrt(np.sum(mn**2, axis=-1))[:, :, np.newaxis]

        reshaped_nodes = mapped_eq_nodes.reshape(-1, 3)

        mesh = Mesh(
                vertex_coordinates=reshaped_nodes,
                elements=np.arange(len(reshaped_nodes)).reshape(len(mesh), -1),
                order=order)

    return mesh


def generate_torus_and_cycle_vertices(r_outer, r_inner,
        n_outer=20, n_inner=10, order=1):
    a = r_outer
    b = r_inner
    u, v = np.mgrid[0:2*np.pi:2*np.pi/n_outer, 0:2*np.pi:2*np.pi/n_inner]

    # http://www.math.hmc.edu/~gu/curves_and_surfaces/surfaces/torus.html
    x = np.cos(u)*(a+b*np.cos(v))
    y = np.sin(u)*(a+b*np.cos(v))
    z = b*np.sin(v)
    pts = (np.vstack((x[np.newaxis], y[np.newaxis], z[np.newaxis]))
            .T.copy().reshape(-1, 3))

    def idx(i, j):
        return (i % n_outer) + (j % n_inner) * n_outer
    tris = ([(idx(i, j), idx(i+1, j), idx(i, j+1))
            for i in xrange(n_outer) for j in xrange(n_inner)]
            + [(idx(i+1, j), idx(i+1, j+1), idx(i, j+1))
            for i in xrange(n_outer) for j in xrange(n_inner)])

    from meshmode.mesh import Mesh
    mesh = Mesh(pts, tris)

    if order > 1:
        from meshmode.mesh.trianglemaps import interv_pts_on_unit
        eq_nodes = interv_pts_on_unit(order, 0, 1)

        # indices: triangle #, node #, xyz axis
        mapped_eq_nodes = np.empty((len(mesh), len(eq_nodes), 3))

        for iel, el_nodes in enumerate(mesh.elements):
            n1, n2, n3 = mesh.vertex_coordinates[el_nodes]
            a1 = np.vstack([n2-n1, n3-n1]).T
            b1 = n1

            mapped_eq_nodes[iel] = np.dot(a1, eq_nodes.T).T + b1

        mn = mapped_eq_nodes
        major_theta = np.arctan2(mapped_eq_nodes[:, :, 1],
            mapped_eq_nodes[:, :, 0])
        rvec = np.array([
            np.cos(major_theta),
            np.sin(major_theta),
            np.zeros_like(major_theta)]).transpose((1, 2, 0))

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

        x = np.sum(mapped_eq_nodes*rvec, axis=-1) - a

        minor_theta = np.arctan2(mapped_eq_nodes[:, :, 2], x)

        mapped_eq_nodes[:, :, 0] = np.cos(major_theta)*(a+b*np.cos(
            minor_theta))
        mapped_eq_nodes[:, :, 1] = np.sin(major_theta)*(a+b*np.cos(
            minor_theta))
        mapped_eq_nodes[:, :, 2] = b*np.sin(minor_theta)

        #reshaped_nodes = mapped_eq_nodes.reshape(-1, 3)
        reshaped_nodes = mn.reshape(-1, 3)

        mesh = Mesh(
                vertex_coordinates=reshaped_nodes,
                elements=np.arange(len(reshaped_nodes)).reshape(len(mesh), -1),
                order=order)

    return mesh, \
            [idx(i, 0) for i in xrange(n_outer)], \
            [idx(0, j) for j in xrange(n_inner)]


def generate_torus(r_outer, r_inner, n_outer=20, n_inner=10, order=1):
    geo, a_cycle, b_cycle = generate_torus_and_cycle_vertices(
            r_outer, r_inner, n_outer, n_inner, order)
    return geo

# vim: fdm=marker
