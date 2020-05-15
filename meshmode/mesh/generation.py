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

import logging
logger = logging.getLogger(__name__)

from pytools import log_process

__doc__ = """

Curves
------

.. autofunction:: make_curve_mesh

Curve parametrizations
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: ellipse
.. autofunction:: cloverleaf
.. data :: starfish
.. autofunction:: drop
.. autofunction:: n_gon
.. autofunction:: qbx_peanut
.. autofunction:: apple
.. autoclass:: WobblyCircle

Surfaces
--------

.. autofunction:: generate_icosahedron
.. autofunction:: generate_icosphere
.. autofunction:: generate_torus
.. autofunction:: refine_mesh_and_get_urchin_warper
.. autofunction:: generate_urchin

Volumes
-------

.. autofunction:: generate_box_mesh
.. autofunction:: generate_regular_rect_mesh
.. autofunction:: generate_warped_rect_mesh

Tools for Iterative Refinement
------------------------------

.. autofunction:: warp_and_refine_until_resolved
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


def apple(a, t):
    """
    :arg a: 0 <= a <= 1/2; roundedness: 0 returns a circle, 1/2 returns a cardioid
    :arg t: the parametrization, runs from [0,1)
    :return: an array of shape *(2, npoints)*
    """
    ilength = 2*np.pi
    t = t*ilength

    sin = np.sin
    cos = np.cos

    return np.vstack([
        cos(t) + a*cos(2*t),
        sin(t) + a*sin(2*t)
        ])


class WobblyCircle(object):
    """
    .. automethod:: random
    .. automethod:: __call__
    """
    def __init__(self, coeffs):
        self.coeffs = coeffs

    @staticmethod
    def random(ncoeffs, seed):
        rng = np.random.mtrand.RandomState(seed)
        coeffs = rng.rand(ncoeffs)

        coeffs = 0.95*coeffs/np.sum(np.abs(coeffs))

        return WobblyCircle(coeffs)

    def __call__(self, t):
        """
        :arg t: the parametrization, runs from [0,1)
        :return: an array of shape *(2, npoints)*
        """

        ilength = 2*np.pi
        t = t*ilength

        wave = 1
        for i, coeff in enumerate(self.coeffs):
            wave = wave + coeff*np.sin((i+1)*t)

        return np.vstack([
            np.cos(t)*wave,
            np.sin(t)*wave,
            ])


class NArmedStarfish(WobblyCircle):
    def __init__(self, n_arms, amplitude):
        coeffs = np.zeros(n_arms)
        coeffs[-1] = amplitude
        super(NArmedStarfish, self).__init__(coeffs)


starfish = NArmedStarfish(5, 0.25)

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
            node_vertex_consistency_tolerance=node_vertex_consistency_tolerance,
            is_conforming=True)

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
    m = len(top_ring)
    for i in range(m):
        tris.append([top_ring[i], top_ring[(i+1) % m], top_point])
        tris.append([bottom_ring[i], bottom_point, bottom_ring[(i+1) % m], ])
        tris.append([bottom_ring[i], bottom_ring[(i+1) % m], top_ring[i]])
        tris.append([top_ring[i], bottom_ring[(i+1) % m], top_ring[(i+1) % m]])

    vertices *= r/la.norm(vertices[:, 0])

    vertex_indices = np.array(tris, dtype=np.int32)

    grp = make_group_from_vertices(vertices, vertex_indices, order)

    from meshmode.mesh import Mesh
    return Mesh(
            vertices, [grp],
            is_conforming=True)

# }}}


# {{{ generate_icosphere

def generate_icosphere(r, order, uniform_refinement_rounds=0):
    mesh = generate_icosahedron(r, order)

    if uniform_refinement_rounds:
        # These come out conformal, so we're OK to use the faster refiner.
        from meshmode.mesh.refinement import RefinerWithoutAdjacency
        refiner = RefinerWithoutAdjacency(mesh)
        for i in range(uniform_refinement_rounds):
            refiner.refine_uniformly()

        mesh = refiner.get_current_mesh()

    vertices = mesh.vertices * r / np.sqrt(np.sum(mesh.vertices**2, axis=0))
    grp, = mesh.groups
    grp = grp.copy(
            nodes=grp.nodes * r / np.sqrt(np.sum(grp.nodes**2, axis=0)))

    from meshmode.mesh import Mesh
    return Mesh(
            vertices, [grp],
            is_conforming=True)

# }}}


# {{{ generate_torus_and_cycle_vertices

def generate_torus_and_cycle_vertices(r_major, r_minor,
        n_major=20, n_minor=10, order=1):
    a = r_major
    b = r_minor
    u, v = np.mgrid[0:2*np.pi:2*np.pi/n_major, 0:2*np.pi:2*np.pi/n_minor]

    # https://web.archive.org/web/20160410151837/https://www.math.hmc.edu/~gu/curves_and_surfaces/surfaces/torus.html  # noqa
    x = np.cos(u)*(a+b*np.cos(v))
    y = np.sin(u)*(a+b*np.cos(v))
    z = b*np.sin(v)
    vertices = (np.vstack((x[np.newaxis], y[np.newaxis], z[np.newaxis]))
            .transpose(0, 2, 1).copy().reshape(3, -1))

    def idx(i, j):
        return (i % n_major) + (j % n_minor) * n_major
    vertex_indices = ([(idx(i, j), idx(i+1, j), idx(i, j+1))
            for i in range(n_major) for j in range(n_minor)]
            + [(idx(i+1, j), idx(i+1, j+1), idx(i, j+1))
            for i in range(n_major) for j in range(n_minor)])

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
                is_conforming=True),
            [idx(i, 0) for i in range(n_major)],
            [idx(0, j) for j in range(n_minor)])

# }}}


def generate_torus(r_major, r_minor, n_major=20, n_minor=10, order=1):
    r"""Generate a torus.

    .. figure:: images/torus.png
        :align: center

        Shown: A torus with major circle (magenta) and minor circle (red).
        Source: https://commons.wikimedia.org/wiki/File:Torus_cycles.svg
        (public domain image by Krishnavedala).

    The torus is obtained as the image of the parameter domain
    :math:`(u, v) \in [0, 2\pi) \times [0, 2 \pi)` under the map

    .. math::
        \begin{align}
        x &= \cos(u) (r_\text{major} + r_\text{minor} \cos(v)) \\
        y &= \sin(u) (r_\text{major} + r_\text{minor} \sin(v)) \\
        z &= r_\text{minor} \sin(v)
        \end{align}

    where :math:`r_\text{major}` and :math:`r_\text{minor}` are the radii of the
    major and minor circles, respectively. The parameter domain is tiled with
    :math:`n_\text{major} \times n_\text{minor}` contiguous rectangles, and then
    each rectangle is subdivided into two triangles.

    :arg r_major: radius of the major circle
    :arg r_minor: radius of the minor circle
    :arg n_major: number of rectangles along major circle
    :arg n_minor: number of rectangles along minor circle
    :arg order: element order
    :returns: a :class:`meshmode.mesh.Mesh` of a torus

    """
    mesh, a_cycle, b_cycle = generate_torus_and_cycle_vertices(
            r_major, r_minor, n_major, n_minor, order)
    return mesh


# {{{ get_urchin

def refine_mesh_and_get_urchin_warper(order, m, n, est_rel_interp_tolerance,
        min_rad=0.2, uniform_refinement_rounds=0):
    """
    :returns: a tuple ``(refiner, warp_mesh)``, where *refiner* is
        a :class:`meshmode.refinement.Refiner` (from which the unwarped mesh
        may be obtained), and whose
        :meth:`meshmode.refinement.Refiner.get_current_mesh` returns a
        locally-refined :class:`meshmode.mesh.Mesh` of a sphere and *warp_mesh*
        is a callable taking and returning a mesh that warps the unwarped mesh
        into a smooth shape govered by a spherical harmonic of order *(m, n)*.
    :arg order: the polynomial order of the returned mesh
    :arg est_rel_interp_tolerance: a tolerance for the relative
        interpolation error estimates on the warped version of the mesh.

    .. versionadded: 2018.1
    """

    def sph_harm(m, n, pts):
        assert abs(m) <= n
        x, y, z = pts
        r = np.sqrt(np.sum(pts**2, axis=0))
        theta = np.arccos(z/r)
        phi = np.arctan2(y, x)

        import scipy.special as sps
        # Note: This matches the spherical harmonic
        # convention in the QBX3D paper:
        # https://arxiv.org/abs/1805.06106
        #
        # Numpy takes arguments in the order (theta, phi)
        # *and* swaps their meanings, so passing the
        # arguments swapped maintains the intended meaning.
        return sps.sph_harm(m, n, phi, theta)

    def map_coords(pts):
        r = np.sqrt(np.sum(pts**2, axis=0))

        sph = sph_harm(m, n, pts).real
        scaled = min_rad + (sph - lo)/(hi-lo)
        new_rad = scaled

        return pts * new_rad / r

    def warp_mesh(mesh, node_vertex_consistency_tolerance):
        groups = [grp.copy(nodes=map_coords(grp.nodes)) for grp in mesh.groups]

        from meshmode.mesh import Mesh
        return Mesh(
                map_coords(mesh.vertices),
                groups,
                node_vertex_consistency_tolerance=False,
                is_conforming=mesh.is_conforming,
                )

    unwarped_mesh = generate_icosphere(1, order=order)

    from meshmode.mesh.refinement import RefinerWithoutAdjacency

    # These come out conformal, so we're OK to use the faster refiner.
    refiner = RefinerWithoutAdjacency(unwarped_mesh)
    for i in range(uniform_refinement_rounds):
        refiner.refine_uniformly()

    nodes_sph = sph_harm(m, n, unwarped_mesh.groups[0].nodes).real
    lo = np.min(nodes_sph)
    hi = np.max(nodes_sph)
    del nodes_sph

    from functools import partial
    unwarped_mesh = warp_and_refine_until_resolved(
                refiner,
                partial(warp_mesh, node_vertex_consistency_tolerance=False),
                est_rel_interp_tolerance)

    return refiner, partial(
            warp_mesh,
            node_vertex_consistency_tolerance=est_rel_interp_tolerance)


def generate_urchin(order, m, n, est_rel_interp_tolerance, min_rad=0.2):
    """
    :returns: a refined :class:`meshmode.mesh.Mesh` of a smooth shape govered
        by a spherical harmonic of order *(m, n)*.
    :arg order: the polynomial order of the returned mesh
    :arg est_rel_interp_tolerance: a tolerance for the relative
        interpolation error estimates on the warped version of the mesh.

    .. versionadded: 2018.1
    """
    refiner, warper = refine_mesh_and_get_urchin_warper(
            order, m, n, est_rel_interp_tolerance, min_rad)
    return warper(refiner.get_current_mesh())

# }}}


# {{{ generate_box_mesh

def generate_box_mesh(axis_coords, order=1, coord_dtype=np.float64,
        group_factory=None, face_to_boundary_tag={}):
    """Create a semi-structured mesh.

    :param axis_coords: a tuple with a number of entries corresponding
        to the number of dimensions, with each entry a numpy array
        specifying the coordinates to be used along that axis.
    :param group_factory: One of :class:`meshmode.mesh.SimplexElementGroup`
        or :class:`meshmode.mesh.TensorProductElementGroup`.
    :param face_to_boundary_tag: an optional dictionary for boundary configuration.
        The keys correspond to custom boundary tags, with the values giving
        a list of the faces on which they should be applied in terms of coordinate
        directions (+x, -x, +y, -y, +z, -z).

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

    # compute facial adjacency for Mesh if there is tag information
    facial_adjacency_groups = None
    face_vertex_indices_to_tags = {}
    boundary_tags = list(face_to_boundary_tag.keys())
    axes = ["x", "y", "z"]
    face_ids = [1, 0, 0]
    from meshmode.mesh import _compute_facial_adjacency_from_vertices
    if boundary_tags:
        for tag_idx, tag in enumerate(boundary_tags):
            # Need to map the correct face vertices to the boundary tags
            for face in face_to_boundary_tag[tag]:
                for i_ax, axis in enumerate(axes):
                    if face == "-" + axis:
                        if dim < i_ax + 1:
                            raise ValueError("Boundary condition dimension "
                            "mismatch")
                        face_id = face_ids[i_ax]
                        dim_crit = i_ax
                        node_crit = axis_coords[i_ax][0]
                    elif face == "+" + axis:
                        if dim < i_ax + 1:
                            raise ValueError("Boundary condition dimension "
                            "mismatch")
                        face_id = face_ids[i_ax]
                        dim_crit = i_ax
                        node_crit = axis_coords[i_ax][-1]
                for ielem in range(0, grp.nelements):
                    if grp.nodes[dim_crit][ielem][0] == node_crit:
                        fvi = grp.vertex_indices[ielem,
                                                 grp.face_vertex_indices()[
                                                     face_id]]
                        if frozenset(fvi) not in face_vertex_indices_to_tags:
                            face_vertex_indices_to_tags[frozenset(fvi)] = []
                        face_vertex_indices_to_tags[frozenset(fvi)].append(tag)
        # Need to add BTAG ALL and BTAG_REALLY_ALL to list of tags
        from meshmode.mesh import BTAG_ALL, BTAG_REALLY_ALL
        boundary_tags.append(BTAG_ALL)
        boundary_tags.append(BTAG_REALLY_ALL)
        facial_adjacency_groups = _compute_facial_adjacency_from_vertices(
                [grp], boundary_tags, np.int32, np.int8,
                face_vertex_indices_to_tags)

    from meshmode.mesh import Mesh
    return Mesh(vertices, [grp],
            facial_adjacency_groups=facial_adjacency_groups,
            is_conforming=True, boundary_tags=boundary_tags)

# }}}


# {{{ generate_regular_rect_mesh

def generate_regular_rect_mesh(a=(0, 0), b=(1, 1), n=(5, 5), order=1,
                               face_to_boundary_tag={}):
    """Create a semi-structured rectangular mesh.

    :param a: the lower left hand point of the rectangle
    :param b: the upper right hand point of the rectangle
    :param n: a tuple of integers indicating the total number of points
      on [a,b].
    :param face_to_boundary_tag: an optional dictionary for boundary configuration.
        The keys correspond to custom boundary tags, with the values giving
        a list of the faces on which they should be applied in terms of coordinate
        directions (+x, -x, +y, -y, +z, -z).
    """
    if min(n) < 2:
        raise ValueError("need at least two points in each direction")

    axis_coords = [np.linspace(a_i, b_i, n_i)
            for a_i, b_i, n_i in zip(a, b, n)]

    return generate_box_mesh(axis_coords, order=order,
                             face_to_boundary_tag=face_to_boundary_tag)

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


# {{{ warp_and_refine_until_resolved

@log_process(logger)
def warp_and_refine_until_resolved(
        unwarped_mesh_or_refiner, warp_callable, est_rel_interp_tolerance):
    """Given an original ("un-warped") :class:`meshmode.mesh.Mesh` and a
    warping function *warp_callable* that takes and returns a mesh and a
    tolerance to which the mesh should be resolved by the mapping polynomials,
    this function will iteratively refine the *unwarped_mesh* until relative
    interpolation error estimates on the warped version are smaller than
    *est_rel_interp_tolerance* on each element.

    :returns: The refined, un-warped mesh.

    .. versionadded:: 2018.1
    """
    from modepy.modes import simplex_onb
    from modepy.matrices import vandermonde
    from modepy.modal_decay import simplex_interp_error_coefficient_estimator_matrix
    from meshmode.mesh.refinement import Refiner, RefinerWithoutAdjacency

    if isinstance(unwarped_mesh_or_refiner, (Refiner, RefinerWithoutAdjacency)):
        refiner = unwarped_mesh_or_refiner
        unwarped_mesh = refiner.get_current_mesh()
    else:
        unwarped_mesh = unwarped_mesh_or_refiner
        refiner = Refiner(unwarped_mesh)

    iteration = 0

    while True:
        refine_flags = np.zeros(unwarped_mesh.nelements, dtype=np.bool)

        warped_mesh = warp_callable(unwarped_mesh)

        # test whether there are invalid values in warped mesh
        if not np.isfinite(warped_mesh.vertices).all():
            raise FloatingPointError("Warped mesh contains non-finite vertices "
                                     "(NaN or Inf)")

        for group in warped_mesh.groups:
            if not np.isfinite(group.nodes).all():
                raise FloatingPointError("Warped mesh contains non-finite nodes "
                                         "(NaN or Inf)")

        for egrp in warped_mesh.groups:
            dim, nunit_nodes = egrp.unit_nodes.shape

            interp_err_est_mat = simplex_interp_error_coefficient_estimator_matrix(
                    egrp.unit_nodes, egrp.order,
                    n_tail_orders=1 if warped_mesh.dim > 1 else 2)

            vdm_inv = la.inv(
                    vandermonde(simplex_onb(dim, egrp.order), egrp.unit_nodes))

            mapping_coeffs = np.einsum("ij,dej->dei", vdm_inv, egrp.nodes)
            mapping_norm_2 = np.sqrt(np.sum(mapping_coeffs**2, axis=-1))

            interp_error_coeffs = np.einsum(
                    "ij,dej->dei", interp_err_est_mat, egrp.nodes)
            interp_error_norm_2 = np.sqrt(np.sum(interp_error_coeffs**2, axis=-1))

            # max over dimensions
            est_rel_interp_error = np.max(interp_error_norm_2/mapping_norm_2, axis=0)

            refine_flags[
                    egrp.element_nr_base:
                    egrp.element_nr_base+egrp.nelements] = \
                            est_rel_interp_error > est_rel_interp_tolerance

        nrefined_elements = np.sum(refine_flags.astype(np.int32))
        if nrefined_elements == 0:
            break

        logger.info("warp_and_refine_until_resolved: "
                "iteration %d -> splitting %d/%d elements",
                iteration, nrefined_elements, unwarped_mesh.nelements)

        unwarped_mesh = refiner.refine(refine_flags)
        iteration += 1

    return unwarped_mesh

# }}}


# vim: fdm=marker
