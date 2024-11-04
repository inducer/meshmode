# mypy: disallow-untyped-defs

__copyright__ = "Copyright (C) 2014-6 Shivam Gupta, Andreas Kloeckner"

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
from abc import ABC, abstractmethod
from functools import singledispatch

import numpy as np

from meshmode.mesh import (
    Mesh,
    MeshElementGroup,
    SimplexElementGroup,
    TensorProductElementGroup,
)


logger = logging.getLogger(__name__)


# {{{ base class

class Refiner(ABC):
    """An abstract refiner class for :class:`meshmode.mesh.Mesh`.

    .. automethod:: __init__
    .. automethod:: get_current_mesh
    .. automethod:: get_previous_mesh

    .. automethod:: refine_uniformly
    .. automethod:: refine
    """

    def __init__(self, mesh: Mesh) -> None:
        self._current_mesh = mesh
        self._previous_mesh = None

    def get_current_mesh(self) -> Mesh:
        return self._current_mesh

    def get_previous_mesh(self) -> Mesh | None:
        return self._previous_mesh

    def refine_uniformly(self):
        flags = np.ones(self._current_mesh.nelements, dtype=bool)
        return self.refine(flags)

    @abstractmethod
    def refine(self, refine_flags: np.ndarray) -> Mesh:
        pass

# }}}


# {{{ map child unit nodes

@singledispatch
def map_unit_nodes_to_children(meg: MeshElementGroup,
        unit_nodes, el_tess_info) -> np.ndarray:
    """
    :arg unit_nodes: an :class:`~numpy.ndarray` of unit nodes on the
        element type described by *meg*.
    :arg el_tess_info: a
        :class:`~meshmode.mesh.refinement.tessellate.ElementTessellationInfo`.
    :returns: an :class:`~numpy.ndarray` of mapped unit nodes for each
        child in the tessellation.
    """
    raise NotImplementedError(type(meg).__name__)


@map_unit_nodes_to_children.register(SimplexElementGroup)
def _(meg: SimplexElementGroup, unit_nodes, el_tess_info):
    ref_vertices = np.array(el_tess_info.ref_vertices, dtype=np.float64).T
    assert len(unit_nodes.shape) == 2

    for child in el_tess_info.children:
        origin = ref_vertices[:, child[0]].reshape(-1, 1)
        basis = ref_vertices[:, child[1:]] - origin

        # mapped nodes are on [0, 2], so we subtract 1 to get it to [-1, 1]
        yield basis.dot((unit_nodes + 1.0) / 2.0) + origin - 1.0


@map_unit_nodes_to_children.register(TensorProductElementGroup)
def _(meg: TensorProductElementGroup, unit_nodes, el_tess_info):
    ref_vertices = np.array(el_tess_info.ref_vertices, dtype=np.float64).T
    assert len(unit_nodes.shape) == 2

    # NOTE: nodes indices in the unit hypercube that form the `e_i` basis
    basis_indices = 2**np.arange(meg.dim)

    for child in el_tess_info.children:
        child_arr = np.array(child)
        origin = ref_vertices[:, child_arr[0]].reshape(-1, 1)
        basis = ref_vertices[:, child_arr[basis_indices]] - origin

        # mapped nodes are on [0, 2], so we subtract 1 to get it to [-1, 1]
        yield basis.dot((unit_nodes + 1.0) / 2.0) + origin - 1.0

# }}}


# {{{ test nodal adjacency against geometry

def is_symmetric(relation, debug=False):
    for a, other_list in enumerate(relation):
        for b in other_list:
            if a not in relation[b]:
                if debug:
                    print("Relation is not symmetric: %s -> %s, but not %s -> %s"
                            % (a, b, b, a))
                return False

    return True


def check_nodal_adj_against_geometry(mesh, tol=1e-12):
    def group_and_iel_to_global_iel(igrp, iel):
        return mesh.base_element_nrs[igrp] + iel

    logger.debug("nodal adj test: tree build")
    from meshmode.mesh.tools import make_element_lookup_tree
    tree = make_element_lookup_tree(mesh, eps=tol)
    logger.debug("nodal adj test: tree build done")

    nadj = mesh.nodal_adjacency
    nvertices_per_element = len(mesh.groups[0].vertex_indices[0])

    connected_to_element_geometry = [set() for i in range(mesh.nelements)]
    connected_to_element_connectivity = [set() for i in range(mesh.nelements)]

    for igrp, grp in enumerate(mesh.groups):
        for iel_grp in range(grp.nelements):
            iel_g = group_and_iel_to_global_iel(igrp, iel_grp)
            nb_starts = nadj.neighbors_starts
            for nb_iel_g in nadj.neighbors[nb_starts[iel_g]:nb_starts[iel_g+1]]:
                connected_to_element_connectivity[iel_g].add(nb_iel_g)

            for vertex_index in grp.vertex_indices[iel_grp]:
                vertex = mesh.vertices[:, vertex_index]

                # check which elements touch this vertex
                for nearby_igrp, nearby_iel in tree.generate_matches(vertex):
                    if nearby_igrp == igrp and nearby_iel == iel_grp:
                        continue
                    nearby_grp = mesh.groups[nearby_igrp]

                    nearby_origin_vertex = mesh.vertices[
                            :, nearby_grp.vertex_indices[nearby_iel][0]]
                    transformation = np.empty(
                            (len(mesh.vertices), nvertices_per_element-1))
                    vertex_transformed = vertex - nearby_origin_vertex

                    for inearby_vertex_index, nearby_vertex_index in enumerate(
                            nearby_grp.vertex_indices[nearby_iel][1:]):
                        nearby_vertex = mesh.vertices[:, nearby_vertex_index]
                        transformation[:, inearby_vertex_index] = \
                                nearby_vertex - nearby_origin_vertex
                    bary_coord, residual = np.linalg.lstsq(
                        transformation, vertex_transformed,
                        rcond=None)[0:2]

                    is_in_element_span = (
                            residual.size == 0
                            or np.linalg.norm(vertex_transformed) == 0
                            or (np.linalg.norm(residual)
                                / np.linalg.norm(vertex_transformed)) <= tol)

                    is_connected = (
                            is_in_element_span
                            and np.sum(bary_coord) <= 1+tol
                            and (bary_coord >= -tol).all())
                    el1 = group_and_iel_to_global_iel(nearby_igrp, nearby_iel)
                    el2 = group_and_iel_to_global_iel(igrp, iel_grp)

                    if is_connected:
                        connected_to_element_geometry[el1].add(el2)
                        connected_to_element_geometry[el2].add(el1)

    assert is_symmetric(connected_to_element_connectivity, debug=True)

    # The geometric adjacency relation isn't necessary symmetric:
    #
    #        /|
    #       / |
    #      /  |\
    #    B \  |/  A
    #       \ |
    #        \|
    #
    # Element A will see element B (its vertices are near B) but not the other
    # way around.

    assert connected_to_element_geometry == connected_to_element_connectivity

# }}}

# vim: foldmethod=marker
