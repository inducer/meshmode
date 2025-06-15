__copyright__ = "Copyright (C) 2010,2012,2013 Andreas Kloeckner, Michael Tom"

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

from warnings import warn

import numpy as np
import numpy.linalg as la

from modepy.tools import hypercube_submesh
from pytools import MovedFunctionDeprecationWrapper
from pytools.spatial_btree import SpatialBinaryTreeBucket


__doc__ = """
.. autofunction:: find_point_to_point_mapping

.. currentmodule:: meshmode

.. autoclass:: AffineMap
"""


# {{{ make_element_lookup_tree

def make_element_lookup_tree(mesh, eps=1e-12):
    from meshmode.mesh.processing import find_bounding_box
    bbox_min, bbox_max = find_bounding_box(mesh)
    bbox_min -= eps
    bbox_max += eps

    tree = SpatialBinaryTreeBucket(bbox_min, bbox_max)

    for igrp, grp in enumerate(mesh.groups):
        for iel_grp in range(grp.nelements):
            el_vertices = mesh.vertices[:, grp.vertex_indices[iel_grp]]

            el_bbox_min = np.min(el_vertices, axis=-1) - eps
            el_bbox_max = np.max(el_vertices, axis=-1) + eps

            tree.insert((igrp, iel_grp), (el_bbox_min, el_bbox_max))

    return tree

# }}}


# {{{ nd_quad_submesh

nd_quad_submesh = MovedFunctionDeprecationWrapper(hypercube_submesh)

# }}}


def optional_array_equal(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    if a is None:
        return b is None
    else:
        if b is None:
            assert a is not None
            return False

        return np.array_equal(a, b)


# {{{ random rotation matrix

def rand_rotation_matrix(ambient_dim, deflection=1.0, randnums=None, rng=None):
    """Creates a random rotation matrix.

    :arg deflection: the magnitude of the rotation. For 0, no rotation; for 1,
        completely random rotation. Small deflection => small perturbation.
    :arg randnums: 3 random numbers in the range [0, 1]. If *None*, they will be
        auto-generated.
    """
    # from https://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c  # noqa: E501
    # from https://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html  # noqa: E501

    if ambient_dim != 3:
        raise NotImplementedError("ambient_dim=%d" % ambient_dim)

    if randnums is None:
        if rng is None:
            rng = np.random.default_rng()

        randnums = rng.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M

# }}}


# {{{ AffineMap

class AffineMap:
    """An affine map ``A@x+b`` represented by a matrix *A* and an offset vector *b*.

    .. attribute:: matrix

        A :class:`numpy.ndarray` representing the matrix *A*, or *None*.

    .. attribute:: offset

        A :class:`numpy.ndarray` representing the vector *b*, or *None*.

    .. automethod:: __init__
    .. automethod:: inverted
    .. automethod:: __call__
    .. automethod:: __eq__
    .. automethod:: __ne__
    """

    def __init__(self, matrix=None, offset=None):
        """
        :arg matrix: A :class:`numpy.ndarray` (or something convertible to one),
            or *None*.
        :arg offset: A :class:`numpy.ndarray` (or something convertible to one),
            or *None*.
        """
        def promote_to_numpy(array):
            if array is not None:
                if isinstance(array, np.ndarray):
                    return array
                else:
                    return np.array(array)
            else:
                return None

        self.matrix = promote_to_numpy(matrix)
        self.offset = promote_to_numpy(offset)

    def inverted(self):
        """Return the inverse affine map."""
        if self.matrix is not None:
            inv_matrix = la.inv(self.matrix)
            if self.offset is not None:
                inv_offset = -inv_matrix @ self.offset
            else:
                inv_offset = None
        else:
            inv_matrix = None
            if self.offset is not None:
                inv_offset = -self.offset  # pylint: disable=E1130
            else:
                inv_offset = None
        return AffineMap(inv_matrix, inv_offset)

    def __call__(self, vecs):
        """Apply the affine map to an array *vecs* whose first axis
        length matches ``matrix.shape[1]``.
        """
        if self.matrix is not None:
            result = np.einsum("ij,j...->i...", self.matrix, vecs)
        else:
            result = vecs.copy()
        if self.offset is not None:
            result += self.offset.reshape(-1, *((1,) * (vecs.ndim-1)))
        return result

    def __eq__(self, other):
        def component_equal(array1, array2):
            if isinstance(array1, np.ndarray) and isinstance(array2, np.ndarray):
                return np.array_equal(array1, array2)
            else:
                return array1 == array2

        return (
            component_equal(self.matrix, other.matrix)
            and component_equal(self.offset, other.offset))

    def __ne__(self, other):
        return not self.__eq__(other)

# }}}


# {{{ find_point_permutation

def find_point_permutation(
            targets: np.ndarray,
            permutees: np.ndarray,
            tol_multiplier: float | None = None
        ) -> np.ndarray | None:
    """
    :arg targets: shaped ``(dim, npoints)`` or just ``(dim,)`` if a single point
    :arg permutees: shaped ``(dim, npoints)``
    :returns: a "from"-style permutation, or None if none was found.
    """
    warn("find_point_permutation is deprecated and will be removed in Q3 2025. Use "
         "find_point_to_point_mapping instead.", DeprecationWarning, stacklevel=2)

    single_target = len(targets.shape) == 1
    if single_target:
        targets = targets.reshape(-1, 1)

    target_idx_to_permutee_idx = find_point_to_point_mapping(
        src_points=targets,
        tgt_points=permutees,
        tol_multiplier=tol_multiplier)

    return (
        target_idx_to_permutee_idx
        if np.all(target_idx_to_permutee_idx >= 0)
        else None)


def find_point_to_point_mapping(
        src_points: np.ndarray,
        tgt_points: np.ndarray, *,
        tol: float | None = None,
        tol_multiplier: float | None = None,
        _max_leaf_points: int | None = None) -> np.ndarray:
    """
    Compute a mapping from indices of points in *src_points* to the matching
    indices of points in *tgt_points*.

    :arg src_points: shaped ``(dim, npoints)`` or just ``(dim,)`` if a single point
    :arg tgt_points: shaped ``(dim, npoints)``
    :arg tol: the maximum distance allowed between matching points.
    :arg tol_multiplier: another way of specifying tolerance: as a multiplier of
        machine epsilon. Must not specify both *tol* and *tol_multiplier*.

    :returns: an array storing an index into *tgt_points* for each point in
        *src_points* if a match exists or ``-1`` if not.
    """

    if tol is not None and tol_multiplier is not None:
        raise ValueError("cannot specify both 'tol' and 'tol_multiplier'")

    if tol is None:
        if tol_multiplier is None:
            tol_multiplier = 250

        tol = float(np.finfo(src_points.dtype).eps * tol_multiplier)

    if _max_leaf_points is None:
        _max_leaf_points = 2**8  # *shrug*

    src_dim, n_src_points = src_points.shape
    tgt_dim, n_tgt_points = tgt_points.shape

    if tgt_dim != src_dim:
        raise ValueError(
            "source and target points must have the same ambient dimension: "
            f"got shape {src_points.shape} and {tgt_points.shape}")

    dim = src_dim

    if dim == 0:
        assert n_tgt_points == 1
        return np.array([0])

    if n_src_points + n_tgt_points <= _max_leaf_points:
        displacements = (
            src_points.reshape(dim, -1, 1)
            - tgt_points.reshape(dim, 1, -1))
        distances_sq = np.sum(displacements**2, axis=0)

        src_indices, = np.indices((n_src_points,))
        min_distance_sq_indices = np.argmin(distances_sq, axis=1)
        min_distances_sq = distances_sq[src_indices, min_distance_sq_indices]

        src_idx_to_tgt_idx = np.where(
            min_distances_sq < tol**2,
            min_distance_sq_indices,
            -1)

        return src_idx_to_tgt_idx

    else:
        both_points = np.concatenate((tgt_points, src_points), axis=1)

        idim_largest = np.argmax(
            np.max(both_points, axis=1)
            - np.min(both_points, axis=1))

        median_coord = np.median(both_points[idim_largest, :])

        lower_src_idx_to_full_idx, = np.where(
            src_points[idim_largest, :] <= median_coord + tol)
        lower_tgt_idx_to_full_idx, = np.where(
            tgt_points[idim_largest, :] <= median_coord + tol)

        upper_src_idx_to_full_idx, = np.where(
            src_points[idim_largest, :] >= median_coord - tol)
        upper_tgt_idx_to_full_idx, = np.where(
            tgt_points[idim_largest, :] >= median_coord - tol)

        if __debug__:
            n_lower_points = (
                len(lower_src_idx_to_full_idx)
                + len(lower_tgt_idx_to_full_idx))
            n_upper_points = (
                len(upper_src_idx_to_full_idx)
                + len(upper_tgt_idx_to_full_idx))
            if (
                    # No theoretical justification for using 2/3, it just seems like
                    # a reasonable number
                    n_lower_points > 2*both_points.shape[1]/3
                    or n_upper_points > 2*both_points.shape[1]/3):
                warn(
                    "bad partitioning of points, performance may be degraded.",
                    stacklevel=2)

        lower_src_idx_to_tgt_idx = find_point_to_point_mapping(
            src_points[:, lower_src_idx_to_full_idx],
            tgt_points[:, lower_tgt_idx_to_full_idx],
            tol=tol,
            _max_leaf_points=_max_leaf_points)

        upper_src_idx_to_tgt_idx = find_point_to_point_mapping(
            src_points[:, upper_src_idx_to_full_idx],
            tgt_points[:, upper_tgt_idx_to_full_idx],
            tol=tol,
            _max_leaf_points=_max_leaf_points)

        matched_lower_points, = np.where(lower_src_idx_to_tgt_idx >= 0)
        matched_upper_points, = np.where(upper_src_idx_to_tgt_idx >= 0)

        src_idx_to_tgt_idx = np.full(n_src_points, -1)
        src_idx_to_tgt_idx[lower_src_idx_to_full_idx[matched_lower_points]] = \
            lower_tgt_idx_to_full_idx[lower_src_idx_to_tgt_idx[matched_lower_points]]
        src_idx_to_tgt_idx[upper_src_idx_to_full_idx[matched_upper_points]] = \
            upper_tgt_idx_to_full_idx[upper_src_idx_to_tgt_idx[matched_upper_points]]

        return src_idx_to_tgt_idx

# }}}

# vim: foldmethod=marker
