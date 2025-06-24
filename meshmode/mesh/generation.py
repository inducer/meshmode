# mypy: disallow-untyped-defs
from __future__ import annotations


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

import logging
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.linalg as la

import modepy as mp
from pytools import deprecate_keyword, log_process

from meshmode.mesh import Mesh, MeshElementGroup, make_mesh
from meshmode.mesh.refinement import Refiner


logger = logging.getLogger(__name__)


__doc__ = """

Curves
------

.. autofunction:: make_curve_mesh

Curve parameterizations
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: circle
.. autofunction:: ellipse
.. autofunction:: cloverleaf
.. autofunction:: drop
.. autofunction:: n_gon
.. autofunction:: qbx_peanut
.. autofunction:: dumbbell
.. autofunction:: wobbly_dumbbell
.. autofunction:: apple
.. autofunction:: clamp_piecewise
.. autoclass:: WobblyCircle
.. autoclass:: NArmedStarfish
.. data:: starfish3
.. data:: starfish5

Surfaces
--------

.. autofunction:: generate_icosahedron
.. autofunction:: generate_cube_surface
.. autofunction:: generate_sphere
.. autofunction:: generate_torus
.. autofunction:: generate_cruller
.. autofunction:: refine_mesh_and_get_urchin_warper
.. autofunction:: generate_urchin
.. autofunction:: generate_surface_of_revolution

Volumes
-------

.. autofunction:: generate_box_mesh
.. autofunction:: generate_regular_rect_mesh
.. autofunction:: generate_warped_rect_mesh
.. autofunction:: generate_annular_cylinder_slice_mesh

Tools for Iterative Refinement
------------------------------

.. autofunction:: warp_and_refine_until_resolved
"""


# {{{ test curve parameterizations

def circle(t: np.ndarray) -> np.ndarray:
    """
    :arg t: the parametrization, runs from :math:`[0, 1]`.
    :return: an array of shape ``(2, t.size)``.
    """
    return ellipse(1.0, t)


def ellipse(aspect_ratio: float, t: np.ndarray) -> np.ndarray:
    """
    :arg t: the parametrization, runs from :math:`[0, 1]`.
    :return: an array of shape ``(2, t.size)``.
    """

    ilength = 2*np.pi
    t = t*ilength
    return np.vstack([
        np.cos(t),
        np.sin(t)/aspect_ratio,
        ])


def cloverleaf(t: np.ndarray) -> np.ndarray:
    """
    :arg t: the parametrization, runs from :math:`[0, 1]`.
    :return: an array of shape ``(2, t.size)``.
    """

    ilength = 2*np.pi
    t = t*ilength

    a = 0.3
    b = 3

    return np.vstack([
        np.cos(t)+a*np.sin(b*t),
        np.sin(t)-a*np.cos(b*t)
        ])


def drop(t: np.ndarray) -> np.ndarray:
    """
    :arg t: the parametrization, runs from :math:`[0, 1]`.
    :return: an array of shape ``(2, t.size)``.
    """

    ilength = np.pi
    t = t*ilength

    return 1.7 * np.vstack([
        np.sin(t)-0.5,
        0.5*(np.cos(t)*(t-np.pi)*t),
        ])


def n_gon(n_corners: int, t: np.ndarray) -> np.ndarray:
    """
    :arg t: the parametrization, runs from :math:`[0, 1]`.
    :return: an array of shape ``(2, t.size)``.
    """

    t = t*n_corners

    result = np.empty((2, *t.shape))

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


def qbx_peanut(t: np.ndarray) -> np.ndarray:
    """
    :arg t: the parametrization, runs from :math:`[0, 1]`.
    :return: an array of shape ``(2, t.size)``.
    """
    t = 2.0 * np.pi * t

    r = (1.0 + 0.3 * np.sin(2 * t))
    return np.vstack([
        3 / 4 * r * np.cos(t - np.pi / 4),
        r * np.sin(t - np.pi / 4),
        ])


def dumbbell(gamma: float, beta: float, t: np.ndarray):
    """
    :arg t: the parametrization, runs from :math:`[0, 1]`.
    :return: an array of shape ``(2, t.size)``.
    """
    return wobbly_dumbbell(gamma, beta, 1, 0, t)


def wobbly_dumbbell(
        gamma: float, beta: float, p: int, wavenumber: int,
        t: np.ndarray):
    """
    :arg t: the parametrization, runs from :math:`[0, 1]`.
    :return: an array of shape ``(2, t.size)``.
    """
    t = 2.0 * np.pi * t
    r = (
        gamma * (1 + beta / (1 - beta) * np.cos(t) ** 2) ** (1 / p)
        + 0.02 * np.sin(wavenumber * t) ** 2)

    return np.stack([
        np.cos(t),
        r * np.sin(t),
        ])


def apple(a: float, t: np.ndarray) -> np.ndarray:
    """
    :arg a: roundness parameter in :math:`[0, 1/2]`, where :math:`0` gives
        a circle and :math:`1/2` gives a cardioid.
    :arg t: the parametrization, runs from :math:`[0, 1]`.
    :return: an array of shape ``(2, t.size)``.
    """
    ilength = 2*np.pi
    t = t*ilength

    sin = np.sin
    cos = np.cos

    return np.vstack([
        cos(t) + a*cos(2*t),
        sin(t) + a*sin(2*t)
        ])


def clamp_piecewise(
        r_major: float, r_minor: float, gap: float,
        t: np.ndarray) -> np.ndarray:
    """
    :arg r_major: radius of the outer shell.
    :arg r_minor: radius of the inner shell.
    :arg gap: half-angle (in radians) of the right-hand side gap.
    :return: an array of shape ``(2, t.size)``.
    """

    def rotation_matrix(angle: float) -> np.ndarray:
        return np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]])

    assert r_major > 0 and r_minor > 0 and r_major > r_minor
    assert gap > 0 and gap < 2 * np.pi
    r_cap = (r_major - r_minor) / 2
    inv_gap = 2 * np.pi - 2 * gap

    # NOTE: Split [0, 1] interval into 5 chunks that have about equal arclength.
    # This is possible because each chunk is a circle arc, so we know its length.

    L = (
        inv_gap * r_major
        + inv_gap * r_minor
        + 2 * np.pi * r_cap)
    t0 = 0.0
    t1 = inv_gap * r_major / L
    t2 = t1 + np.pi * r_cap / L
    t3 = t2 + inv_gap * r_minor / L
    t4 = t3 + np.pi * r_cap / L

    # outer shell
    m_outer = (t < t1).astype(t.dtype)
    theta = m_outer * (np.pi + inv_gap * (t - (t1 + t0) / 2) / (t1 - t0))
    f_outer = np.stack([r_major * np.cos(theta), r_major * np.sin(theta)])

    # first cap
    m_caps0 = np.logical_and(t >= t1, t < t2).astype(t.dtype)
    theta = m_caps0 * (np.pi / 2 + np.pi * (t - (t2 + t1) / 2) / (t2 - t1))
    R = r_cap * rotation_matrix(-gap)
    f_caps0 = R @ np.stack([
        np.cos(theta) - 1, np.sin(theta)
        ]) + r_major * np.stack([[np.cos(-gap), np.sin(-gap)]]).T

    # inner shell
    m_inner = np.logical_and(t >= t2, t < t3).astype(t.dtype)
    theta = -m_inner * (np.pi + inv_gap * (t - (t2 + t3) / 2) / (t3 - t2))
    f_inner = np.stack([r_minor * np.cos(theta), r_minor * np.sin(theta)])

    # second cap
    m_caps1 = (t >= t3).astype(t.dtype)
    theta = m_caps1 * (np.pi / 2 + np.pi * (t - (t3 + t4) / 2) / (t4 - t3))
    R = r_cap * rotation_matrix(np.pi + gap)
    f_caps1 = R @ np.stack([
        np.cos(theta) + 1, np.sin(theta)
        ]) + r_major * np.stack([[np.cos(gap), np.sin(gap)]]).T

    return (
        m_outer * f_outer
        + m_caps0 * f_caps0
        + m_inner * f_inner
        + m_caps1 * f_caps1)


class WobblyCircle:
    """
    .. automethod:: random
    .. automethod:: __call__
    """
    def __init__(self, coeffs: np.ndarray, phase: float = 0.0) -> None:
        self.coeffs = coeffs
        self.phase = phase

    @staticmethod
    def random(ncoeffs: int, seed: int) -> "WobblyCircle":
        rng = np.random.default_rng(seed)
        coeffs = rng.random(ncoeffs)

        coeffs = 0.95*coeffs/np.sum(np.abs(coeffs))

        return WobblyCircle(coeffs)

    def __call__(self, t: np.ndarray) -> np.ndarray:
        """
        :arg t: the parametrization, runs from :math:`[0, 1]`.
        :return: an array of shape ``(2, t.size)``.
        """

        ilength = 2*np.pi
        t = t*ilength

        wave = 1
        for i, coeff in enumerate(self.coeffs):
            wave = wave + coeff*np.sin((i+1)*t + self.phase)

        return np.vstack([
            np.cos(t)*wave,
            np.sin(t)*wave,
            ])


class NArmedStarfish(WobblyCircle):
    """Inherits from :class:`WobblyCircle`.

    .. automethod:: __call__
    """
    def __init__(self, n_arms: int, amplitude: float, phase: float = 0.0) -> None:
        coeffs = np.zeros(n_arms)
        coeffs[-1] = amplitude
        super().__init__(coeffs, phase=phase)

    @staticmethod
    def random(seed: int) -> "NArmedStarfish":
        rng = np.random.default_rng(seed)

        # NOTE: 32 arms should be enough for everybody!
        n_arms = rng.integers(3, 33)
        amplitude = 0.25 + 0.75 * rng.random()

        return NArmedStarfish(n_arms, amplitude)


starfish3 = NArmedStarfish(3, 1 / 2, phase=np.pi / 2)
starfish5 = NArmedStarfish(5, 0.25)

starfish = starfish5

# }}}


# {{{ make_curve_mesh

def make_curve_mesh(
        curve_f: Callable[[np.ndarray], np.ndarray],
        element_boundaries: np.ndarray, order: int, *,
        unit_nodes: np.ndarray | None = None,
        node_vertex_consistency_tolerance: float | bool | None = None,
        closed: bool = True,
        return_parametrization_points: bool = False) -> Mesh:
    """
    :arg curve_f: parametrization for a curve, accepting a vector of
        point locations and returning an array of shape ``(2, npoints)``.
    :arg element_boundaries: a vector of element boundary locations in
        :math:`[0, 1]`, in order. :math:`0` must be the first entry, :math:`1`
        the last one.
    :arg order: order of the (simplex) elements. If *unit_nodes* is also
        provided, the orders should match.
    :arg unit_nodes: if given, the unit nodes to use. Must have shape
        ``(2, nnodes)``.
    :arg node_vertex_consistency_tolerance: passed to the
        :class:`~meshmode.mesh.Mesh` constructor. If *False*, no checks are
        performed.
    :arg closed: if *True*, the curve is assumed closed and the first and
        last of the *element_boundaries* must match.
    :arg return_parametrization_points: if *True*, the parametrization points
        at which all the nodes in the mesh were evaluated are also returned.
    :returns: a :class:`~meshmode.mesh.Mesh`, or if *return_parametrization_points*
        is *True*, a tuple ``(mesh, par_points)``, where *par_points* is an array of
        parametrization points.
    """

    assert element_boundaries[0] == 0
    assert element_boundaries[-1] == 1
    nelements = len(element_boundaries) - 1

    if unit_nodes is None:
        unit_nodes = mp.warp_and_blend_nodes(1, order)
    nodes_01 = 0.5*(unit_nodes+1)

    wrap = nelements
    if not closed:
        wrap += 1

    vertices = curve_f(element_boundaries)[:, :wrap]
    vertex_indices = np.vstack([
        np.arange(0, nelements, dtype=np.int32),
        np.arange(1, nelements + 1, dtype=np.int32) % wrap
        ]).T

    assert vertices.shape[1] == np.max(vertex_indices) + 1
    if closed:
        start_end_par = np.array([0, 1], dtype=np.float64)
        start_end_curve = curve_f(start_end_par)

        assert la.norm(start_end_curve[:, 0] - start_end_curve[:, 1]) < 1.0e-12

    el_lengths = np.diff(element_boundaries)
    el_starts = element_boundaries[:-1]

    # (el_nr, node_nr)
    t = el_starts[:, np.newaxis] + el_lengths[:, np.newaxis]*nodes_01
    t = t.ravel()
    nodes = curve_f(t).reshape(vertices.shape[0], nelements, -1)

    from meshmode.mesh import SimplexElementGroup
    egroup = SimplexElementGroup.make_group(
            order,
            vertex_indices=vertex_indices,
            nodes=nodes,
            unit_nodes=unit_nodes)

    mesh = make_mesh(
            vertices=vertices, groups=[egroup],
            node_vertex_consistency_tolerance=node_vertex_consistency_tolerance,
            is_conforming=True)

    if return_parametrization_points:
        return mesh, t
    else:
        return mesh

# }}}


# {{{ make_group_from_vertices

@deprecate_keyword("group_factory", "group_cls")
def make_group_from_vertices(
        vertices: np.ndarray, vertex_indices: np.ndarray, order: int, *,
        group_cls: type[MeshElementGroup] | None = None,
        unit_nodes: np.ndarray | None = None) -> MeshElementGroup:
    # shape: (ambient_dim, nelements, nvertices)
    ambient_dim = vertices.shape[0]
    el_vertices = vertices[:, vertex_indices]

    from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
    if group_cls is None:
        group_cls = SimplexElementGroup

    if issubclass(group_cls, SimplexElementGroup):
        if order < 1:
            raise ValueError("can't represent simplices with mesh order < 1")

        el_origins = el_vertices[:, :, 0][:, :, np.newaxis]
        # ambient_dim, nelements, nspan_vectors
        spanning_vectors = (
                el_vertices[:, :, 1:] - el_origins)

        nspan_vectors = spanning_vectors.shape[-1]
        dim = nspan_vectors

        # dim, nunit_nodes
        if unit_nodes is None:
            shape = mp.Simplex(dim)
            space = mp.space_for_shape(shape, order)
            unit_nodes = mp.edge_clustered_nodes_for_space(space, shape)

        unit_nodes_01 = 0.5 + 0.5*unit_nodes
        nodes = np.einsum(
                "si,des->dei",
                unit_nodes_01, spanning_vectors) + el_origins

    elif issubclass(group_cls, TensorProductElementGroup):
        nelements, nvertices = vertex_indices.shape

        dim = nvertices.bit_length() - 1
        if nvertices != 2**dim:
            raise ValueError("invalid number of vertices for tensor-product "
                    "elements, must be power of two")

        shape = mp.Hypercube(dim)
        space = mp.space_for_shape(shape, order)

        if unit_nodes is None:
            unit_nodes = mp.edge_clustered_nodes_for_space(space, shape)

        # shape: (dim, nnodes)
        unit_nodes_01 = 0.5 + 0.5*unit_nodes
        _, nnodes = unit_nodes.shape

        vertex_space = mp.space_for_shape(shape, 1)
        vertex_tuples = mp.node_tuples_for_space(vertex_space)
        assert len(vertex_tuples) == nvertices

        vdm = np.empty((nvertices, nvertices))
        for i, vertex_tuple in enumerate(vertex_tuples):
            for j, func_tuple in enumerate(vertex_tuples):
                vertex_ref = np.array(vertex_tuple, dtype=np.float64)
                vdm[i, j] = np.prod(vertex_ref**func_tuple)

        # shape: (ambient_dim, nelements, nvertices)
        coeffs = np.empty((ambient_dim, nelements, nvertices))
        for d in range(ambient_dim):
            coeffs[d] = la.solve(vdm, el_vertices[d].T).T

        vdm_nodes = np.zeros((nnodes, nvertices))
        for j, func_tuple in enumerate(vertex_tuples):
            vdm_nodes[:, j] = np.prod(
                    unit_nodes_01 ** np.array(func_tuple).reshape(-1, 1),
                    axis=0)

        nodes = np.einsum("ij,dej->dei", vdm_nodes, coeffs)
    else:
        raise ValueError(f"unsupported value for 'group_cls': {group_cls}")

    # make contiguous
    nodes = nodes.copy()

    return group_cls.make_group(
            order, vertex_indices, nodes,
            unit_nodes=unit_nodes)

# }}}


# {{{ generate_icosahedron

def generate_icosahedron(
        r: float, order: int, *,
        node_vertex_consistency_tolerance: float | bool | None = None,
        unit_nodes: np.ndarray | None = None) -> Mesh:
    # https://en.wikipedia.org/w/index.php?title=Icosahedron&oldid=387737307

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

    grp = make_group_from_vertices(vertices, vertex_indices, order,
            unit_nodes=unit_nodes)

    return make_mesh(
            vertices, [grp],
            node_vertex_consistency_tolerance=node_vertex_consistency_tolerance,
            is_conforming=True)


def generate_cube_surface(r: float, order: int, *,
        node_vertex_consistency_tolerance: float | bool | None = None,
        unit_nodes: np.ndarray | None = None) -> Mesh:
    shape = mp.Hypercube(3)
    vertices = mp.unit_vertices_for_shape(shape)
    vertices *= r / la.norm(vertices, ord=2, axis=0)
    vertex_indices = np.array([
            face.volume_vertex_indices for face in mp.faces_for_shape(shape)
            ], dtype=np.int32)

    from meshmode.mesh import TensorProductElementGroup
    grp = make_group_from_vertices(
            vertices, vertex_indices, order,
            group_cls=TensorProductElementGroup,
            unit_nodes=unit_nodes)

    return make_mesh(
            vertices, [grp],
            node_vertex_consistency_tolerance=node_vertex_consistency_tolerance,
            is_conforming=True)

# }}}


# {{{ generate_icosphere

def generate_icosphere(r: float, order: int, *,
        uniform_refinement_rounds: int = 0,
        node_vertex_consistency_tolerance: float | bool | None = None,
        unit_nodes: np.ndarray | None = None) -> Mesh:
    from warnings import warn
    warn("'generate_icosphere' is deprecated and will be removed in 2023. "
            "Use 'generate_sphere' instead.",
            DeprecationWarning, stacklevel=2)

    from meshmode.mesh import SimplexElementGroup
    return generate_sphere(r, order,
            uniform_refinement_rounds=uniform_refinement_rounds,
            node_vertex_consistency_tolerance=node_vertex_consistency_tolerance,
            unit_nodes=unit_nodes,
            group_cls=SimplexElementGroup)


def generate_sphere(r: float, order: int, *,
        uniform_refinement_rounds: int = 0,
        node_vertex_consistency_tolerance: float | bool | None = None,
        unit_nodes: np.ndarray | None = None,
        group_cls: type[MeshElementGroup] | None = None) -> Mesh:
    """
    :arg r: radius of the sphere.
    :arg order: order of the group elements. If *unit_nodes* is also
        provided, the orders should match.
    :arg uniform_refinement_rounds: number of uniform refinement rounds to
        perform after the initial mesh was created.
    :arg node_vertex_consistency_tolerance: passed to the
        :class:`~meshmode.mesh.Mesh` constructor. If *False*, no checks are
        performed.
    :arg unit_nodes: if given, the unit nodes to use. Must have shape
        ``(3, nnodes)``.
    :arg group_cls: a :class:`~meshmode.mesh.MeshElementGroup` subclass.
        Based on the class, a different polyhedron is used to construct the
        sphere: simplices use :func:`generate_icosahedron` and tensor
        products use a :func:`generate_cube_surface`.
    """
    from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
    if group_cls is None:
        group_cls = SimplexElementGroup

    if issubclass(group_cls, SimplexElementGroup):
        mesh = generate_icosahedron(r, order,
                node_vertex_consistency_tolerance=node_vertex_consistency_tolerance,
                unit_nodes=unit_nodes)
    elif issubclass(group_cls, TensorProductElementGroup):
        mesh = generate_cube_surface(r, order,
                node_vertex_consistency_tolerance=node_vertex_consistency_tolerance,
                unit_nodes=unit_nodes)
    else:
        raise TypeError(f"unsupported 'group_cls': {group_cls}")

    if uniform_refinement_rounds:
        from meshmode.mesh.refinement import refine_uniformly
        mesh = refine_uniformly(mesh, uniform_refinement_rounds)

    # ensure vertices and nodes are still on the sphere of radius r
    from dataclasses import replace
    vertices = mesh.vertices * r / np.sqrt(np.sum(mesh.vertices**2, axis=0))
    grp, = mesh.groups
    grp = replace(
        grp,
        nodes=grp.nodes * r / np.sqrt(np.sum(grp.nodes**2, axis=0))
        )

    return make_mesh(
            vertices, [grp],
            node_vertex_consistency_tolerance=node_vertex_consistency_tolerance,
            is_conforming=True)

# }}}


# {{{ generate_surface_of_revolution

def generate_surface_of_revolution(
        get_radius: Callable[[np.ndarray, np.ndarray], np.ndarray],
        height_discr: np.ndarray,
        angle_discr: np.ndarray,
        order: int, *,
        node_vertex_consistency_tolerance: float | bool | None = None,
        unit_nodes: np.ndarray | None = None) -> Mesh:
    """Return a cylinder aligned with the "height" axis aligned with the Z axis.

    :arg get_radius: A callable function that takes in a 1D array of heights
        and a 1D array of angles and returns a 1D array of radii.
    :arg height_discr: A discretization of ``[0, 2*pi)``.
    :arg angle_discr: A discretization of ``[0, 2*pi)``.
    :arg order: order of the (simplex) elements. If *unit_nodes* is also
        provided, the orders should match.
    :arg node_vertex_consistency_tolerance: passed to the
        :class:`~meshmode.mesh.Mesh` constructor. If *False*, no checks are
        performed.
    :arg unit_nodes: if given, the unit nodes to use. Must have shape
        ``(3, nnodes)``.
    """
    n = len(angle_discr)
    m = len(height_discr)
    vertices = np.zeros((3, n*m))
    theta, h = np.meshgrid(angle_discr, height_discr)
    theta = theta.flatten()
    h = h.flatten()
    r = get_radius(h, theta)
    vertices[0, :] = np.cos(theta)*r
    vertices[1, :] = np.sin(theta)*r
    vertices[2, :] = h

    tris = []
    for i in range(m-1):
        for j in range(n):
            tris.append([i*n + j, (i + 1)*n + j, (i + 1)*n + (j + 1) % n])
            tris.append([i*n + j, i*n + (j + 1) % n, (i + 1)*n + (j + 1) % n])

    vertex_indices = np.array(tris, dtype=np.int32)

    grp = make_group_from_vertices(vertices, vertex_indices, order,
                unit_nodes=unit_nodes)

    mesh = make_mesh(
            vertices, [grp],
            node_vertex_consistency_tolerance=node_vertex_consistency_tolerance,
            is_conforming=True)

    # ensure vertices and nodes are still on the surface with radius r
    def ensure_radius(arr: np.ndarray) -> np.ndarray:
        res = arr.copy()
        h = res[2, :].flatten()
        theta = np.arctan2(res[1, :].flatten(), res[0, :].flatten())
        r_expected = get_radius(h, theta).reshape(res[0, :].shape)
        res[:2, :] *= r_expected/np.sum(res[:2, :]**2, axis=0)
        return res

    from dataclasses import replace
    vertices = ensure_radius(mesh.vertices)
    grp, = mesh.groups
    grp = replace(grp, nodes=ensure_radius(grp.nodes))

    return make_mesh(
            vertices, [grp],
            node_vertex_consistency_tolerance=node_vertex_consistency_tolerance,
            is_conforming=True)

# }}}


# {{{ generate_torus_and_cycle_vertices

def generate_torus_and_cycle_vertices(
        r_major: float, r_minor: float,
        *,
        n_major: int = 20,
        n_minor: int = 10,
        order: int = 1,
        node_vertex_consistency_tolerance: float | bool | None = None,
        unit_nodes: np.ndarray | None = None,
        r_minor_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        group_cls: type[MeshElementGroup] | None = None,
        ) -> tuple[Mesh, list[int], list[int]]:
    """
    :arg r_major: This and the other arguments match those of :func:`generate_torus`.
    :arg r_minor_func: A callable that takes :math:`(u, v)` coordinates on the
        torus on applies a transformation to them. This is then multiplied into
        the minor radius ``r_minor`` to create a modulation in its amplitude,
        i.e. distort the torus. It is used in :func:`generate_cruller`.

    :returns: a tuple of ``(mesh, major_indices, minor_indices)``. The indices
        are for points around the torus, e.g. ``major_indices`` gives a list of
        *n_major* points around the major radius of the torus.
    """

    def default_func(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return 1.0

    a = r_major
    b = r_minor

    if r_minor_func is None:
        r_minor_func = default_func

    from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup

    if group_cls is None:
        group_cls = SimplexElementGroup

    # {{{ create periodic grid

    def idx(i: int, j: int) -> int:
        return i + j * (n_major + 1)

    if issubclass(group_cls, SimplexElementGroup):
        # NOTE: this makes two triangles from a square like
        #   (i, j+1)    (i+1, j+1)
        #       o---------o
        #       | \       |
        #       |   \     |
        #       |     \   |
        #       |       \ |
        #       o---------o
        #   (i, j)      (i+1, j)

        vertex_indices = ([
            (idx(i, j), idx(i+1, j), idx(i, j+1))
            for i in range(n_major) for j in range(n_minor)
            ] + [
            (idx(i+1, j), idx(i+1, j+1), idx(i, j+1))
            for i in range(n_major) for j in range(n_minor)
            ])
    elif issubclass(group_cls, TensorProductElementGroup):
        # NOTE: this should match the order of the points in modepy
        vertex_indices = [
            (idx(i, j), idx(i+1, j), idx(i, j+1), idx(i+1, j+1))
            for i in range(n_major) for j in range(n_minor)
            ]
    else:
        raise TypeError(f"unsupported 'group_cls': {group_cls}")

    # NOTE: include endpoints first so that `make_group_from_vertices` can
    # actually interpolate the unit nodes to each element
    u = np.linspace(0.0, 2.0 * np.pi, n_major + 1)
    v = np.linspace(0.0, 2.0 * np.pi, n_minor + 1)
    uv = np.stack(np.meshgrid(u, v, copy=False)).reshape(2, -1)

    vertex_indices = np.array(vertex_indices, dtype=np.int32)
    grp = make_group_from_vertices(
            uv, vertex_indices, order,
            unit_nodes=unit_nodes,
            group_cls=group_cls)

    # }}}

    # {{{ evaluate on torus

    # https://web.archive.org/web/20160410151837/https://www.math.hmc.edu/~gu/curves_and_surfaces/surfaces/torus.html  # noqa: E501

    # create new vertices without the endpoints
    u = np.linspace(0.0, 2.0 * np.pi, n_major, endpoint=False)
    v = np.linspace(0.0, 2.0 * np.pi, n_minor, endpoint=False)
    u, v = np.meshgrid(u, v, copy=False)

    # wrap the indices around
    i = vertex_indices % (n_major + 1)
    j = vertex_indices // (n_major + 1)
    vertex_indices = (i % n_major) + (j % n_minor) * n_major

    # evaluate vertices on torus
    bhat = b * r_minor_func(u, v)
    vertices = np.stack([
        np.cos(u) * (a + bhat * np.cos(v)),
        np.sin(u) * (a + bhat * np.cos(v)),
        bhat * np.sin(v)
        ]).reshape(3, -1)

    # evaluate nodes on torus
    u, v = grp.nodes
    bhat = b * r_minor_func(u, v)
    nodes = np.stack([
        np.cos(u) * (a + bhat * np.cos(v)),
        np.sin(u) * (a + bhat * np.cos(v)),
        bhat * np.sin(v)
        ])

    # }}}

    from dataclasses import replace
    grp = replace(grp, vertex_indices=vertex_indices, nodes=nodes)

    return (
            make_mesh(
                vertices, [grp],
                node_vertex_consistency_tolerance=node_vertex_consistency_tolerance,
                is_conforming=True),
            [idx(i, 0) for i in range(n_major)],
            [idx(0, j) for j in range(n_minor)])

# }}}


# {{{ generate_torus

def generate_torus(
        r_major: float, r_minor: float,
        n_major: int = 20,
        n_minor: int = 10,
        *,
        order: int = 1,
        node_vertex_consistency_tolerance: float | bool | None = None,
        unit_nodes: np.ndarray | None = None,
        group_cls: type[MeshElementGroup] | None = None) -> Mesh:
    r"""Generate a torus.

    .. tikz:: A torus with major circle (magenta) and minor circle (red).
        :align: center
        :xscale: 60

        \pgfmathsetmacro{\a}{1.5};
        \pgfmathsetmacro{\b}{0.5};

        \begin{axis}[hide axis, axis equal image]
        \addplot3[
            mesh,
            gray!20,
            samples=20,
            domain=0:2*pi,y domain=0:2*pi,
            z buffer=sort] (
            {(\a + \b*cos(deg(x))) * cos(deg(y+pi/2))},
            {(\a + \b*cos(deg(x))) * sin(deg(y+pi/2))},
            {\b*sin(deg(x))});
        \addplot3 [red, thick, samples=40, domain=0:2*pi] (
            {(\a + \b*cos(deg(x))) * cos(deg(-pi/6))},
            {(\a + \b*cos(deg(x))) * sin(deg(-pi/6))},
            {\b*sin(deg(x))});
        \addplot3 [magenta, thick, samples=80, domain=0:2*pi] (
            {(\a + \b*cos(deg(pi/2))) * cos(deg(x))},
            {(\a + \b*cos(deg(pi/2))) * sin(deg(x))},
            {\b*sin(deg(pi/2))});
        \end{axis}

    The torus is obtained as the image of the parameter domain
    :math:`(u, v) \in [0, 2\pi) \times [0, 2 \pi)` under the map

    .. math::

        \begin{aligned}
        x &= \cos(u) (r_\text{major} + r_\text{minor} \cos(v)), \\
        y &= \sin(u) (r_\text{major} + r_\text{minor} \sin(v)). \\
        z &= r_\text{minor} \sin(v).
        \end{aligned}

    where :math:`r_\text{major}` and :math:`r_\text{minor}` are the radii of the
    major and minor circles, respectively. The parameter domain is tiled with
    :math:`n_\text{major} \times n_\text{minor}` contiguous rectangles, and then
    each rectangle is subdivided into two triangles.

    :arg r_major: radius of the major circle.
    :arg r_minor: radius of the minor circle.
    :arg n_major: number of rectangles along major circle.
    :arg n_minor: number of rectangles along minor circle.
    :arg order: order of the (simplex) elements. If *unit_nodes* is also
        provided, the orders should match.
    :arg node_vertex_consistency_tolerance: passed to the
        :class:`~meshmode.mesh.Mesh` constructor. If *False*, no checks are
        performed.
    :arg unit_nodes: if given, the unit nodes to use. Must have shape
        ``(3, nnodes)``.
    :returns: a :class:`~meshmode.mesh.Mesh` of a torus.

    """

    mesh, _, _ = generate_torus_and_cycle_vertices(
            r_major, r_minor,
            n_major=n_major,
            n_minor=n_minor,
            order=order,
            node_vertex_consistency_tolerance=node_vertex_consistency_tolerance,
            unit_nodes=unit_nodes,
            group_cls=group_cls)

    return mesh


# }}}


# {{{ generate_cruller

def generate_cruller(
        r_major: float, r_minor: float,
        n_major: int = 20,
        n_minor: int = 10,
        *,
        order: int = 1,
        node_vertex_consistency_tolerance: float | bool | None = None,
        unit_nodes: np.ndarray | None = None,
        group_cls: type[MeshElementGroup] | None = None) -> Mesh:
    r"""Generate a "cruller" (as the pastry) like geometry.

    The "cruller" is defined by

    .. math::

        \begin{aligned}
        x &= \cos(u) (r_\text{major} + r_\text{minor} H(u, v) \cos(v)), \\
        y &= \sin(u) (r_\text{major} + r_\text{minor} H(u, v) \sin(v)), \\
        z &= r_\text{minor} H(u, v) \sin(v),
        \end{aligned}

    where :math:`H(u, v) = 1.0 + 0.25 \cos(5 u + 3 v)`. This geometry and
    parameters are taken from [Barnett2020]_.

    .. [Barnett2020] A. Barnett, L. Greengard, T. Hagstrom,
        *High-Order Discretization of a Stable Time-Domain Integral Equation for
        3D Acoustic Scattering*,
        Journal of Computational Physics, Vol. 402, pp. 109047--109047, 2020,
        `DOI <https://doi.org/10.1016/j.jcp.2019.109047>`__.
    """

    mesh, _, _ = generate_torus_and_cycle_vertices(
            r_major, r_minor,
            n_major=n_major,
            n_minor=n_minor,
            order=order,
            r_minor_func=lambda u, v: 1.0 + 0.25 * np.cos(5.0 * u + 3.0 * v),
            node_vertex_consistency_tolerance=node_vertex_consistency_tolerance,
            unit_nodes=unit_nodes,
            group_cls=group_cls)

    return mesh

# }}}


# {{{ get_urchin

def refine_mesh_and_get_urchin_warper(
            order: int, m: int, n: int, est_rel_interp_tolerance: float,
            min_rad: float = 0.2,
            uniform_refinement_rounds: int = 0
        ) -> tuple[Refiner, Callable[[Mesh], Mesh]]:
    """
    :arg order: order of the (simplex) elements.
    :arg m: order of the spherical harmonic :math:`Y^m_n`.
    :arg n: order of the spherical harmonic :math:`Y^m_n`.
    :arg est_rel_interp_tolerance: a tolerance for the relative
        interpolation error estimates on the warped version of the mesh.

    :returns: a tuple ``(refiner, warp_mesh)``, where *refiner* is
        a :class:`~meshmode.mesh.refinement.RefinerWithoutAdjacency` (from
        which the unwarped mesh may be obtained), and whose
        :meth:`~meshmode.mesh.refinement.RefinerWithoutAdjacency.get_current_mesh`
        returns a locally-refined :class:`~meshmode.mesh.Mesh` of a sphere and
        *warp_mesh* is a callable taking and returning a mesh that warps the
        unwarped mesh into a smooth shape covered by a spherical harmonic of
        order :math:`(m, n)`.

    .. versionadded: 2018.1
    """
    try:
        from scipy.special import sph_harm_y
    except ImportError:
        # NOTE: this was deprecated in v1.15 and (will be) removed in v1.17
        from scipy.special import sph_harm as _sph_harm_y

        def sph_harm_y(n: int, m: int, p: np.ndarray, t: np.ndarray) -> np.ndarray:
            # NOTE: this swaps both (n, m) and (theta, phi)
            return _sph_harm_y(m, n, t, p)

    def sph_harm(m: int, n: int, pts: np.ndarray) -> np.ndarray:
        assert abs(m) <= n
        x, y, z = pts
        r = np.sqrt(np.sum(pts**2, axis=0))
        theta = np.arccos(z/r)
        phi = np.arctan2(y, x)

        # NOTE: This matches the spherical harmonic convention in the QBX3D paper:
        # https://arxiv.org/abs/1805.06106
        return sph_harm_y(n, m, theta, phi)

    def map_coords(pts: np.ndarray) -> np.ndarray:
        r = np.sqrt(np.sum(pts**2, axis=0))

        sph = sph_harm(m, n, pts).real
        scaled = min_rad + (sph - lo)/(hi-lo)
        new_rad = scaled

        return pts * new_rad / r

    def warp_mesh(mesh: Mesh) -> Mesh:
        from dataclasses import replace
        groups = [
            replace(grp, nodes=map_coords(grp.nodes))
            for grp in mesh.groups]

        return make_mesh(
                map_coords(mesh.vertices),
                groups,
                node_vertex_consistency_tolerance=False,
                is_conforming=mesh.is_conforming,
                )

    unwarped_mesh = generate_sphere(1, order=order)

    from meshmode.mesh.refinement import RefinerWithoutAdjacency

    # These come out conformal, so we're OK to use the faster refiner.
    refiner = RefinerWithoutAdjacency(unwarped_mesh)
    for _ in range(uniform_refinement_rounds):
        refiner.refine_uniformly()

    nodes_sph = sph_harm(m, n, unwarped_mesh.groups[0].nodes).real
    lo = np.min(nodes_sph)
    hi = np.max(nodes_sph)
    del nodes_sph

    unwarped_mesh = warp_and_refine_until_resolved(
                refiner,
                warp_mesh,
                est_rel_interp_tolerance)

    return refiner, warp_mesh


def generate_urchin(
        order: int, m: int, n: int,
        est_rel_interp_tolerance: float,
        min_rad: float = 0.2) -> Mesh:
    """
    :arg order: order of the (simplex) elements. If *unit_nodes* is also
        provided, the orders should match.
    :arg m: order of the spherical harmonic :math:`Y^m_n`.
    :arg n: order of the spherical harmonic :math:`Y^m_n`.
    :arg est_rel_interp_tolerance: a tolerance for the relative
        interpolation error estimates on the warped version of the mesh.

    :returns: a refined :class:`~meshmode.mesh.Mesh` of a smooth shape covered
        by a spherical harmonic of order :math:`(m, n)`.

    .. versionadded: 2018.1
    """
    refiner, warper = refine_mesh_and_get_urchin_warper(
            order, m, n, est_rel_interp_tolerance,
            min_rad=min_rad,
            uniform_refinement_rounds=0,
            )

    return warper(refiner.get_current_mesh())

# }}}


# {{{ generate_box_mesh

@deprecate_keyword("group_factory", "group_cls")
def generate_box_mesh(
        axis_coords: tuple[np.ndarray, ...],
        order: int = 1, *,
        coord_dtype: Any = np.float64,
        vertex_id_dtype: Any = np.int32,
        element_id_dtype: Any = np.int32,
        face_id_dtype: Any = np.int8,
        periodic: tuple[bool, ...] | None = None,
        group_cls: type[MeshElementGroup] | None = None,
        boundary_tag_to_face: dict[Any, str] | None = None,
        mesh_type: str | None = None,
        unit_nodes: np.ndarray | None = None) -> Mesh:
    r"""Create a semi-structured mesh.

    :arg axis_coords: a tuple with a number of entries corresponding
        to the number of dimensions, with each entry a numpy array
        specifying the coordinates to be used along that axis. The coordinates
        for a given axis must define a nonnegative number of subintervals (in other
        words, the length can be 0 or a number greater than or equal to 2).
    :arg periodic: an optional tuple of :class:`bool` indicating whether
        the mesh is periodic along each axis. Acts as a shortcut for calling
        :func:`meshmode.mesh.processing.glue_mesh_boundaries`.
    :arg group_cls: One of :class:`meshmode.mesh.SimplexElementGroup`
        or :class:`meshmode.mesh.TensorProductElementGroup`.
    :arg boundary_tag_to_face: an optional dictionary for tagging boundaries.
        The keys correspond to custom boundary tags, with the values giving
        a list of the faces on which they should be applied in terms of coordinate
        directions (``+x``, ``-x``, ``+y``, ``-y``, ``+z``, ``-z``, ``+w``, ``-w``).

        For example::

            boundary_tag_to_face = {"bdry_1": ["+x", "+y"], "bdry_2": ["-x"]}

    :arg mesh_type: In two dimensions with non-tensor-product elements,
        *mesh_type* may be set to ``"X"`` to generate this type
        of mesh::

            _______
            |\   /|
            | \ / |
            |  X  |
            | / \ |
            |/   \|
            ^^^^^^^

        instead of the default::

            _______
            |\    |
            | \   |
            |  \  |
            |   \ |
            |    \|
            ^^^^^^^

        Specifying a value other than *None* for all other mesh
        dimensionalities and element types is an error.

    .. versionchanged:: 2017.1

        *group_factory* parameter added.

    .. versionchanged:: 2020.1

        *boundary_tag_to_face* parameter added.

    .. versionchanged:: 2020.3

        *group_factory* deprecated and renamed to *group_cls*.
    """

    if boundary_tag_to_face is None:
        boundary_tag_to_face = {}

    for iaxis, axc in enumerate(axis_coords):
        if len(axc) == 1:
            raise ValueError(f"cannot have a single point along axis {iaxis+1} "
                "(1-based), that would be a surface mesh. (If you want one of "
                "those, make a box mesh of lower topological dimension and map it.)")

    dim = len(axis_coords)

    if periodic is None:
        periodic = (False,)*dim

    shape = tuple(len(axc) for axc in axis_coords)

    from pytools import product
    nvertices = product(shape)

    vertex_indices = np.arange(nvertices, dtype=vertex_id_dtype).reshape(*shape)

    vertices = np.empty((dim, *shape), dtype=coord_dtype)
    for idim in range(dim):
        vshape = (shape[idim],) + (1,)*(dim-1-idim)
        vertices[idim] = axis_coords[idim].reshape(*vshape)

    vertices = vertices.reshape(dim, -1)

    from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
    if group_cls is None:
        group_cls = SimplexElementGroup

    if issubclass(group_cls, SimplexElementGroup):
        is_tp = False
    elif issubclass(group_cls, TensorProductElementGroup):
        is_tp = True
    else:
        raise ValueError(f"unsupported value for 'group_cls': {group_cls}")

    shape_m1 = tuple(max(si-1, 0) for si in shape)

    box_faces = [
        side + str(axis)
        for axis in range(dim)
        for side in ["-", "+"]]

    if dim == 1:
        if mesh_type is not None:
            raise ValueError(f"unsupported mesh type: '{mesh_type}'")

        nelements = shape_m1[0]
        nvertices_per_element = 2

        if nelements > 0:
            a0 = vertex_indices[:shape_m1[0]]
            a1 = vertex_indices[1:shape[0]]

            el_vertices = np.column_stack([a0, a1])
            box_face_to_elements = {
                "-0": np.array([0], dtype=element_id_dtype),
                "+0": np.array([shape_m1[0]-1], dtype=element_id_dtype)}

        else:
            el_vertices = np.empty((0, nvertices_per_element), dtype=vertex_id_dtype)
            box_face_to_elements = {
                box_face: np.array([], dtype=element_id_dtype)
                for box_face in box_faces}

    elif dim == 2:
        if is_tp:
            if mesh_type is not None:
                raise ValueError(f"unsupported mesh type: '{mesh_type}'")

            nsubelements = 1
            nvertices_per_element = 4

        elif mesh_type == "X":
            nmidpoints = product(shape_m1)
            midpoint_indices = (
                    nvertices
                    + np.arange(nmidpoints).reshape(*shape_m1, order="F"))

            midpoints = np.empty((dim, *shape_m1), dtype=coord_dtype)
            for idim in range(dim):
                vshape = (shape_m1[idim],) + (1,)*(1-idim)
                left_axis_coords = axis_coords[idim][:-1]
                right_axis_coords = axis_coords[idim][1:]
                midpoints[idim] = (
                        0.5*(left_axis_coords+right_axis_coords)).reshape(*vshape)

            midpoints = midpoints.reshape((dim, nmidpoints), order="F")
            vertices = np.concatenate((vertices, midpoints), axis=1)
            nvertices += nmidpoints

            nsubelements = 4
            nvertices_per_element = 3

        elif mesh_type is None:
            nsubelements = 2
            nvertices_per_element = 3

        else:
            raise ValueError(f"unsupported mesh type: '{mesh_type}'")

        nelements = nsubelements * product(shape_m1)

        if nelements > 0:
            base_idx_tuples = np.array(list(np.ndindex(shape_m1)))

            i0 = base_idx_tuples[:, 0]
            j0 = base_idx_tuples[:, 1]

            a00 = vertex_indices[i0, j0]
            a01 = vertex_indices[i0, j0+1]
            a10 = vertex_indices[i0+1, j0]
            a11 = vertex_indices[i0+1, j0+1]

            if is_tp:
                el_vertices = np.column_stack([a00, a10, a01, a11])

                box_face_to_elements = {}
                for box_face in box_faces:
                    side, axis = box_face
                    axis = int(axis)
                    idx_face = 0 if side == "-" else shape_m1[axis] - 1
                    elements, = np.where(base_idx_tuples[:, axis] == idx_face)
                    box_face_to_elements[box_face] = elements

            elif mesh_type == "X":
                m = midpoint_indices[i0, j0]
                subelement_to_vertices = [
                    [a00, a10, m],
                    [a10, a11, m],
                    [a11, a01, m],
                    [a01, a00, m]]

                el_vertices = np.empty(
                    (nelements, nvertices_per_element), dtype=vertex_id_dtype)
                for isubelement, verts in enumerate(subelement_to_vertices):
                    for ivert, vert in enumerate(verts):
                        el_vertices[isubelement::nsubelements, ivert] = vert

                box_face_to_subelement_offset = {
                    "-0": 3,
                    "+0": 1,
                    "-1": 0,
                    "+1": 2}

                box_face_to_elements = {}
                for box_face in box_faces:
                    side, axis = box_face
                    axis = int(axis)
                    idx_face = 0 if side == "-" else shape_m1[axis] - 1
                    box_elements, = np.where(base_idx_tuples[:, axis] == idx_face)
                    box_face_to_elements[box_face] = \
                        4*box_elements + box_face_to_subelement_offset[box_face]

            else:
                subelement_to_vertices = [
                    [a00, a10, a01],
                    [a11, a01, a10]]

                el_vertices = np.empty(
                    (nelements, nvertices_per_element), dtype=vertex_id_dtype)
                for isubelement, verts in enumerate(subelement_to_vertices):
                    for ivert, vert in enumerate(verts):
                        el_vertices[isubelement::nsubelements, ivert] = vert

                side_to_subelement_offset = {
                    "-": 0,
                    "+": 1}

                box_face_to_elements = {}
                for box_face in box_faces:
                    side, axis = box_face
                    axis = int(axis)
                    idx_face = 0 if side == "-" else shape_m1[axis] - 1
                    box_elements, = np.where(base_idx_tuples[:, axis] == idx_face)
                    box_face_to_elements[box_face] = \
                        2*box_elements + side_to_subelement_offset[side]
        else:
            el_vertices = np.empty((0, nvertices_per_element), dtype=vertex_id_dtype)
            box_face_to_elements = {
                box_face: np.array([], dtype=element_id_dtype)
                for box_face in box_faces}

    elif dim == 3:
        if is_tp:
            if mesh_type is not None:
                raise ValueError(f"unsupported mesh type: '{mesh_type}'")

            nsubelements = 1
            nvertices_per_element = 8

        elif mesh_type is None:
            nsubelements = 6
            nvertices_per_element = 4

        else:
            raise ValueError(f"unsupported mesh type: '{mesh_type}'")

        nelements = nsubelements * product(shape_m1)

        if nelements > 0:
            base_idx_tuples = np.array(list(np.ndindex(shape_m1)))

            i0 = base_idx_tuples[:, 0]
            j0 = base_idx_tuples[:, 1]
            k0 = base_idx_tuples[:, 2]

            a000 = vertex_indices[i0, j0, k0]
            a001 = vertex_indices[i0, j0, k0+1]
            a010 = vertex_indices[i0, j0+1, k0]
            a011 = vertex_indices[i0, j0+1, k0+1]
            a100 = vertex_indices[i0+1, j0, k0]
            a101 = vertex_indices[i0+1, j0, k0+1]
            a110 = vertex_indices[i0+1, j0+1, k0]
            a111 = vertex_indices[i0+1, j0+1, k0+1]

            if is_tp:
                el_vertices = np.column_stack([
                    a000, a100, a010, a110,
                    a001, a101, a011, a111])

                box_face_to_elements = {}
                for box_face in box_faces:
                    side, axis = box_face
                    axis = int(axis)
                    idx_face = 0 if side == "-" else shape_m1[axis] - 1
                    elements, = np.where(base_idx_tuples[:, axis] == idx_face)
                    box_face_to_elements[box_face] = elements

            else:
                subelement_to_vertices = [
                    [a000, a100, a010, a001],
                    [a101, a100, a001, a010],
                    [a101, a011, a010, a001],
                    [a100, a010, a101, a110],
                    [a011, a010, a110, a101],
                    [a011, a111, a101, a110]]

                el_vertices = np.empty(
                    (nelements, nvertices_per_element), dtype=vertex_id_dtype)
                for isubelement, verts in enumerate(subelement_to_vertices):
                    for ivert, vert in enumerate(verts):
                        el_vertices[isubelement::nsubelements, ivert] = vert

                box_face_to_subelement_offsets = {
                    "-0": (0, 2),
                    "+0": (3, 5),
                    "-1": (0, 1),
                    "+1": (4, 5),
                    "-2": (0, 3),
                    "+2": (2, 5)}

                box_face_to_elements = {}
                for box_face in box_faces:
                    side, axis = box_face
                    axis = int(axis)
                    idx_face = 0 if side == "-" else shape_m1[axis] - 1
                    box_elements, = np.where(base_idx_tuples[:, axis] == idx_face)
                    box_face_to_elements[box_face] = np.concatenate([
                        6*box_elements + offset
                        for offset in box_face_to_subelement_offsets[box_face]])
        else:
            el_vertices = np.empty((0, nvertices_per_element), dtype=vertex_id_dtype)
            box_face_to_elements = {
                box_face: np.array([], dtype=element_id_dtype)
                for box_face in box_faces}

    else:
        raise NotImplementedError("box meshes of dimension %d" % dim)

    grp = make_group_from_vertices(
            vertices.reshape(dim, product(vertices.shape[1:])), el_vertices, order,
            group_cls=group_cls, unit_nodes=unit_nodes)

    axes = ["x", "y", "z", "w"]

    for idim in range(dim):
        if periodic[idim]:
            lower_face = "-" + axes[idim]
            upper_face = "+" + axes[idim]
            boundary_tag_to_face["periodic_" + lower_face] = [lower_face]
            boundary_tag_to_face["periodic_" + upper_face] = [upper_face]

    # {{{ compute facial adjacency for mesh if there is tag information

    facial_adjacency_groups = None
    face_vertex_indices_to_tags = {}

    if boundary_tag_to_face:
        box_face_to_tags = {}
        for tag, tagged_box_faces in boundary_tag_to_face.items():
            for box_face in tagged_box_faces:
                if len(box_face) != 2:
                    raise ValueError(
                        f"face identifier '{box_face}' does not "
                        "consist of exactly two characters")
                side, axis = box_face
                try:
                    axis = axes.index(axis)
                except ValueError as exc:
                    raise ValueError(
                        "unrecognized axis in face identifier "
                        f"'{box_face}'") from exc
                if axis >= dim:
                    raise ValueError(f"axis in face identifier '{box_face}' "
                                     f"does not exist in {dim}D")
                if side not in ("-", "+"):
                    raise ValueError("first character of face identifier"
                                     f" '{box_face}' is not '+' or '-'")
                box_face_to_tags.setdefault(f"{side}{axis}", []).append(tag)

        # Extra vertices added beyond the box vertices don't have corresponding
        # tuples, so assign them tuple(-1, ...)
        vert_index_to_tuple = np.full((nvertices, dim), -1)
        for itup in np.ndindex(shape):
            vert_index_to_tuple[vertex_indices[itup], :] = itup

        for box_face, elements in box_face_to_elements.items():
            try:
                tags = box_face_to_tags[box_face]
            except KeyError:
                continue
            side, axis = box_face
            axis = int(axis)
            vert_face = 0 if side == "-" else shape[axis] - 1

            for ref_fvi in grp.face_vertex_indices():
                fvis = grp.vertex_indices[elements[:, np.newaxis], ref_fvi]
                fvi_tuples = vert_index_to_tuple[fvis, :]
                face_fvis = fvis[
                    np.all(fvi_tuples[:, :, axis] == vert_face, axis=1), :]
                for iface in range(face_fvis.shape[0]):
                    fvi = face_fvis[iface, :]
                    face_vertex_indices_to_tags[frozenset(fvi)] = tags

        from meshmode.mesh import _compute_facial_adjacency_from_vertices
        facial_adjacency_groups = _compute_facial_adjacency_from_vertices(
                [grp], element_id_dtype, face_id_dtype, face_vertex_indices_to_tags)
    else:
        facial_adjacency_groups = None

    # }}}

    mesh = make_mesh(vertices, [grp],
            facial_adjacency_groups=facial_adjacency_groups,
            is_conforming=True,
            vertex_id_dtype=vertex_id_dtype,
            element_id_dtype=element_id_dtype,
            face_id_dtype=face_id_dtype)

    if any(periodic):
        from meshmode import AffineMap
        from meshmode.mesh.processing import BoundaryPairMapping, glue_mesh_boundaries
        bdry_pair_mappings_and_tols = []
        for idim in range(dim):
            if periodic[idim]:
                offset = np.zeros(dim, dtype=coord_dtype)
                offset[idim] = axis_coords[idim][-1] - axis_coords[idim][0]
                bdry_pair_mappings_and_tols.append((
                    BoundaryPairMapping(
                        "periodic_-" + axes[idim],
                        "periodic_+" + axes[idim],
                        AffineMap(offset=offset)),
                    1e-12*offset[idim]))

        periodic_mesh = glue_mesh_boundaries(mesh, bdry_pair_mappings_and_tols)

        return periodic_mesh
    else:
        return mesh

# }}}


# {{{ generate_regular_rect_mesh

@deprecate_keyword("group_factory", "group_cls")
def generate_regular_rect_mesh(
        a: Sequence[float] = (0, 0),
        b: Sequence[float] = (1, 1), *,
        nelements_per_axis: int | None = None,
        npoints_per_axis: int | None = None,
        periodic: bool | None = None,
        order: int = 1,
        boundary_tag_to_face: dict[Any, str] | None = None,
        group_cls: type[MeshElementGroup] | None = None,
        mesh_type: str | None = None,
        n: int | None = None,
        ) -> Mesh:
    """Create a semi-structured rectangular mesh with equispaced elements.

    :arg a: the lower left hand point of the rectangle.
    :arg b: the upper right hand point of the rectangle.
    :arg nelements_per_axis: an optional tuple of integers indicating the
        number of elements along each axis.
    :arg npoints_per_axis: an optional tuple of integers indicating the
        number of points along each axis.
    :arg periodic: an optional tuple of :class:`bool` indicating whether
        the mesh is periodic along each axis. Acts as a shortcut for calling
        :func:`meshmode.mesh.processing.glue_mesh_boundaries`.
    :arg order: the mesh element order.
    :arg boundary_tag_to_face: an optional dictionary for tagging boundaries.
        See :func:`generate_box_mesh`.
    :arg group_cls: see :func:`generate_box_mesh`.
    :arg mesh_type: see :func:`generate_box_mesh`.

    .. note::

        Specify only one of *nelements_per_axis* and *npoints_per_axis*.
    """
    if n is not None:
        from warnings import warn
        warn("n parameter to generate_regular_rect_mesh is deprecated. Use "
                "nelements_per_axis or npoints_per_axis instead. "
                "n will disappear in 2022.",
                DeprecationWarning, stacklevel=2)
        if nelements_per_axis is not None:
            raise TypeError("cannot specify both nelements_per_axis and n")
        if npoints_per_axis is not None:
            raise TypeError("cannot specify both npoints_per_axis and n")
        npoints_per_axis = n
    else:
        if npoints_per_axis is not None:
            if nelements_per_axis is not None:
                raise TypeError("cannot specify both nelements_per_axis and "
                    "npoints_per_axis")
        elif nelements_per_axis is not None:
            npoints_per_axis = tuple(
                nel_i+1 if nel_i > 0 else 0
                for nel_i in nelements_per_axis)
        else:
            raise TypeError("Must specify nelements_per_axis or "
                "npoints_per_axis")

    if any(npoints_i == 1 for npoints_i in npoints_per_axis):
        raise ValueError("cannot have a single point along any axis, that would "
            "be a surface mesh. (If you want one of those, make a box mesh of "
            "lower topological dimension and map it.)")

    axis_coords = [np.linspace(a_i, b_i, npoints_i)
            for a_i, b_i, npoints_i in zip(a, b, npoints_per_axis, strict=False)]

    return generate_box_mesh(axis_coords, order=order,
                             periodic=periodic,
                             boundary_tag_to_face=boundary_tag_to_face,
                             group_cls=group_cls,
                             mesh_type=mesh_type)

# }}}


# {{{ generate_warped_rect_mesh

def generate_warped_rect_mesh(
        dim: int, order: int, *,
        nelements_side: int | None = None,
        npoints_side: int | None = None,
        group_cls: type[MeshElementGroup] | None = None,
        n: int | None = None) -> Mesh:
    """Generate a mesh of a warped square/cube. Mainly useful for testing
    functionality with curvilinear meshes.
    """
    if n is not None:
        from warnings import warn
        warn("n parameter to generate_warped_rect_mesh is deprecated. Use "
                "nelements_side or npoints_side instead. n will disappear "
                "in 2022.", DeprecationWarning, stacklevel=2)
        if nelements_side is not None:
            raise TypeError("cannot specify both nelements_side and n")
        if npoints_side is not None:
            raise TypeError("cannot specify both npoints_side and n")
        npoints_side = n
    else:
        if npoints_side is not None:
            if nelements_side is not None:
                raise TypeError("cannot specify both nelements_side and "
                    "npoints_side")
        elif nelements_side is not None:
            npoints_side = nelements_side + 1

    assert dim in [2, 3]

    npoints_per_axis = (npoints_side,)*dim if npoints_side is not None else None

    mesh = generate_regular_rect_mesh(
            a=(-0.5,)*dim, b=(0.5,)*dim,
            npoints_per_axis=npoints_per_axis, order=order, group_cls=group_cls)

    def m(x: np.ndarray) -> np.ndarray:
        result = np.empty_like(x)
        if len(x) >= 2:
            result[0] = (
                    1.5*x[0] + np.cos(x[0])
                    + 0.1*np.sin(10*x[1]))
            result[1] = (
                    0.05*np.cos(10*x[0])
                    + 1.3*x[1] + np.sin(x[1]))
        else:
            result[0] = 1.5*x[0] + np.cos(x[0])

        if len(x) >= 3:
            result[2] = x[2] + np.sin(x[0] / 2) / 2
        return result

    from meshmode.mesh.processing import map_mesh
    return map_mesh(mesh, m)

# }}}


# {{{ generate_annular_cylinder_slice_mesh

def generate_annular_cylinder_slice_mesh(
        n: int, center: np.ndarray, inner_radius: float, outer_radius: float,
        periodic: bool = False) -> Mesh:
    r"""
    Generate a slice of a 3D annular cylinder for
    :math:`\theta \in [-\frac{\pi}{4}, \frac{\pi}{4}]`. Optionally periodic in
    $\theta$.
    """
    unit_mesh = generate_regular_rect_mesh(
        a=(0,)*3,
        b=(1,)*3,
        nelements_per_axis=(n,)*3,
        boundary_tag_to_face={
            "-r": ["-x"],
            "+r": ["+x"],
            "-theta": ["-y"],
            "+theta": ["+y"],
            "-z": ["-z"],
            "+z": ["+z"],
            })

    def transform(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = inner_radius*(1 - x[0]) + outer_radius*x[0]
        theta = -np.pi/4*(1 - x[1]) + np.pi/4*x[1]
        z = -0.5*(1 - x[2]) + 0.5*x[2]
        return (
            center[0] + r*np.cos(theta),
            center[1] + r*np.sin(theta),
            center[2] + z)

    from meshmode.mesh.processing import map_mesh
    mesh = map_mesh(unit_mesh, lambda x: np.stack(transform(x)))

    if periodic:
        from meshmode.mesh.processing import _get_rotation_matrix_from_angle_and_axis
        matrix = _get_rotation_matrix_from_angle_and_axis(
            np.pi/2, np.array([0, 0, 1]))
        from meshmode.mesh.tools import AffineMap
        aff_map = AffineMap(matrix, center - matrix @ center)

        from meshmode.mesh.processing import BoundaryPairMapping, glue_mesh_boundaries
        periodic_mesh = glue_mesh_boundaries(
            mesh, bdry_pair_mappings_and_tols=[
                (BoundaryPairMapping("-theta", "+theta", aff_map), 1e-12)])

        return periodic_mesh
    else:
        return mesh

# }}}


# {{{ warp_and_refine_until_resolved

@log_process(logger)
def warp_and_refine_until_resolved(
        unwarped_mesh_or_refiner: Mesh | Refiner,
        warp_callable: Callable[[Mesh], Mesh],
        est_rel_interp_tolerance: float) -> Mesh:
    """Given an original ("unwarped") :class:`meshmode.mesh.Mesh` and a
    warping function *warp_callable* that takes and returns a mesh and a
    tolerance to which the mesh should be resolved by the mapping polynomials,
    this function will iteratively refine the *unwarped_mesh* until relative
    interpolation error estimates on the warped version are smaller than
    *est_rel_interp_tolerance* on each element.

    :returns: The refined, unwarped mesh.

    .. versionadded:: 2018.1
    """
    import modepy as mp
    from modepy.modal_decay import simplex_interp_error_coefficient_estimator_matrix

    from meshmode.mesh import SimplexElementGroup
    from meshmode.mesh.refinement import RefinerWithoutAdjacency

    if isinstance(unwarped_mesh_or_refiner, RefinerWithoutAdjacency):
        refiner = unwarped_mesh_or_refiner
        unwarped_mesh = refiner.get_current_mesh()
    elif isinstance(unwarped_mesh_or_refiner, Mesh):
        unwarped_mesh = unwarped_mesh_or_refiner
        refiner = RefinerWithoutAdjacency(unwarped_mesh)
    else:
        raise TypeError(
            f"unsupported type: '{type(unwarped_mesh_or_refiner).__name__}'")

    iteration = 0

    while True:
        refine_flags = np.zeros(unwarped_mesh.nelements, dtype=bool)

        warped_mesh = warp_callable(unwarped_mesh)

        # test whether there are invalid values in warped mesh
        if not np.isfinite(warped_mesh.vertices).all():
            raise FloatingPointError("Warped mesh contains non-finite vertices "
                                     "(NaN or Inf)")

        for group in warped_mesh.groups:
            if not np.isfinite(group.nodes).all():
                raise FloatingPointError("Warped mesh contains non-finite nodes "
                                         "(NaN or Inf)")

        for base_element_nr, egrp in zip(
                warped_mesh.base_element_nrs, warped_mesh.groups,
                strict=True):
            if not isinstance(egrp, SimplexElementGroup):
                raise TypeError(
                    f"Unsupported element group type: '{type(egrp).__name__}'")

            interp_err_est_mat = simplex_interp_error_coefficient_estimator_matrix(
                    egrp.unit_nodes, egrp.order,
                    n_tail_orders=1 if warped_mesh.dim > 1 else 2)

            basis = mp.orthonormal_basis_for_space(
                egrp.space, egrp.shape)
            vdm_inv = la.inv(mp.vandermonde(basis.functions, egrp.unit_nodes))

            mapping_coeffs = np.einsum("ij,dej->dei", vdm_inv, egrp.nodes)
            mapping_norm_2 = np.sqrt(np.sum(mapping_coeffs**2, axis=-1))

            interp_error_coeffs = np.einsum(
                    "ij,dej->dei", interp_err_est_mat, egrp.nodes)
            interp_error_norm_2 = np.sqrt(np.sum(interp_error_coeffs**2, axis=-1))

            # max over dimensions
            est_rel_interp_error = np.max(interp_error_norm_2/mapping_norm_2, axis=0)

            refine_flags[base_element_nr:base_element_nr + egrp.nelements] = (
                est_rel_interp_error > est_rel_interp_tolerance)

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
