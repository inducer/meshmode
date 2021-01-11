__copyright__ = "Copyright (C) 2014-6 Shivam Gupta"

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

import itertools
from functools import partial

import numpy as np

from meshmode.mesh.refinement.no_adjacency import RefinerWithoutAdjacency
from meshmode.mesh.refinement.tesselate import \
        ElementTesselationInfo, GroupRefinementRecord

import logging
logger = logging.getLogger(__name__)

__doc__ = """
.. autoclass :: Refiner
.. autoclass :: RefinerWithoutAdjacency
.. autofunction :: refine_uniformly
"""

__all__ = [
    "Refiner", "RefinerWithoutAdjacency", "refine_uniformly"
]


# {{{ deprecated

class SimplexResampler:
    @staticmethod
    def get_vertex_pair_to_midpoint_order(dim):
        nmidpoints = dim * (dim + 1) // 2
        return dict(zip(
            ((i, j) for j in range(dim + 1) for i in range(j)),
            range(nmidpoints)
            ))

    @staticmethod
    def get_midpoints(group, el_tess_info, elements):
        import meshmode.mesh.refinement.tesselate as tess
        return tess.get_group_midpoints(group, el_tess_info, elements)

    @staticmethod
    def get_tesselated_nodes(group, el_tess_info, elements):
        import meshmode.mesh.refinement.tesselate as tess
        return tess.get_group_tesselated_nodes(group, el_tess_info, elements)


def tesselate_simplex(dim):
    import modepy as mp
    shape = mp.Simplex(dim)
    space = mp.space_for_shape(shape, 2)

    node_tuples = mp.node_tuples_for_space(space)
    return node_tuples, mp.submesh_for_shape(shape, node_tuples)

# }}}


class TreeRayNode:
    """Describes a ray as a tree, this class represents each node in this tree
    .. attribute:: left
        Left child.
        *None* if ray segment hasn't been split yet.
    .. attribute:: right
        Right child.
        *None* if ray segment hasn't been split yet.
    .. attribute:: midpoint
        Vertex index of midpoint of this ray segment.
        *None* if no midpoint has been assigned yet.
    .. attribute:: adjacent_elements
        List containing elements indices of elements adjacent
        to this ray segment.
    """
    def __init__(self, left_vertex, right_vertex, adjacent_elements=[]):
        import copy
        self.left = None
        self.right = None
        self.parent = None
        self.left_vertex = left_vertex
        self.right_vertex = right_vertex
        self.midpoint = None
        self.adjacent_elements = copy.deepcopy(adjacent_elements)
        self.adjacent_add_diff = []


class Refiner:
    """An older that mostly succeeds at preserving adjacency across
    non-conformal refinement.

    .. note::

        This refiner is currently kind of slow, and not always correct.
        See :class:`RefinerWithoutAdjacency` for a less capable
        but much faster refiner.

    .. automethod:: __init__
    .. automethod:: refine
    .. automethod:: refine_uniformly
    .. automethod:: get_current_mesh
    .. automethod:: get_previous_mesh
    """

    # {{{ constructor

    def __init__(self, mesh):
        from warnings import warn
        warn("Refiner is deprecated and will be removed in 2022.",
                DeprecationWarning, stacklevel=2)

        if mesh.is_conforming is not True:
            raise ValueError("Refiner can only be used with meshes that are known "
                    "to be conforming. If you would like to refine non-conforming "
                    "meshes and do not need adjacency information, consider "
                    "using RefinerWithoutAdjacency.")

        self.lazy = False
        self.seen_tuple = {}
        self.group_refinement_records = []
        seg_node_tuples, seg_result = tesselate_simplex(1)
        tri_node_tuples, tri_result = tesselate_simplex(2)
        tet_node_tuples, tet_result = tesselate_simplex(3)
        #quadrilateral_node_tuples = [
        #print tri_result, tet_result
        self.simplex_node_tuples = [
                None, seg_node_tuples, tri_node_tuples, tet_node_tuples]
        # Dimension-parameterized tesselations for refinement
        self.simplex_result = [None, seg_result, tri_result, tet_result]
        #print tri_node_tuples, tri_result
        #self.simplex_node_tuples, self.simplex_result = tesselatetet()
        self.last_mesh = mesh

        # {{{ initialization

        # mapping: (vertex_1, vertex_2) -> TreeRayNode
        # where vertex_i represents a vertex number
        #
        # Assumption: vertex_1 < vertex_2
        self.pair_map = {}

        nvertices = len(mesh.vertices[0])

        #array containing element whose edge lies on corresponding vertex
        self.hanging_vertex_element = []
        for i in range(nvertices):
            self.hanging_vertex_element.append([])

        # Fill pair_map.
        # Add adjacency information to each TreeRayNode.
        for grp in mesh.groups:
            iel_base = grp.element_nr_base
            for iel_grp in range(grp.nelements):
                vert_indices = grp.vertex_indices[iel_grp]
                for i in range(len(vert_indices)):
                    for j in range(i+1, len(vert_indices)):

                        #minimum and maximum of the two indices for storing
                        #data in vertex_pair
                        min_index = min(vert_indices[i], vert_indices[j])
                        max_index = max(vert_indices[i], vert_indices[j])

                        vertex_pair = (min_index, max_index)
                        #print(vertex_pair)
                        if vertex_pair not in self.pair_map:
                            self.pair_map[vertex_pair] = TreeRayNode(min_index, max_index)
                            self.pair_map[vertex_pair].adjacent_elements.append(iel_base+iel_grp)
                        elif (iel_base+iel_grp) not in self.pair_map[vertex_pair].adjacent_elements:
                            (self.pair_map[vertex_pair].
                                adjacent_elements.append(iel_base+iel_grp))
        # }}}

        #print(vert_indices)
        #generate reference tuples
        self.index_to_node_tuple = []
        self.index_to_midpoint_tuple = []
        for d in range(len(vert_indices)):
            dim = d + 1
            cur_index_to_node_tuple = [()] * dim
            for i in range(0, dim-1):
                cur_index_to_node_tuple[0] = cur_index_to_node_tuple[0] + (0,)
            for i in range(1, dim):
                for j in range(1, dim):
                    if i == j:
                        cur_index_to_node_tuple[i] = cur_index_to_node_tuple[i] + (2,)
                    else:
                        cur_index_to_node_tuple[i] = cur_index_to_node_tuple[i] + (0,)
            cur_index_to_midpoint_tuple = [()] * (int((dim * (dim - 1)) / 2))
            curind = 0
            for ind1 in range(0, len(cur_index_to_node_tuple)):
                for ind2 in range(ind1+1, len(cur_index_to_node_tuple)):
                    i = cur_index_to_node_tuple[ind1]
                    j = cur_index_to_node_tuple[ind2]
                    #print(i, j)
                    for k in range(0, dim-1):
                        cur = int((i[k] + j[k]) / 2)
                        cur_index_to_midpoint_tuple[curind] = cur_index_to_midpoint_tuple[curind] + (cur,)
                    curind += 1
            self.index_to_node_tuple.append(cur_index_to_node_tuple)
            self.index_to_midpoint_tuple.append(cur_index_to_midpoint_tuple)
        '''
        self.ray_vertices = np.empty([len(mesh.groups[0].vertex_indices) *
            len(mesh.groups[0].vertex_indices[0]) * (len(mesh.groups[0].vertex_indices[0]) - 1) / 2, 2],
                dtype=np.int32)
        self.ray_elements = np.zeros([len(mesh.groups[0].vertex_indices) *
            len(mesh.groups[0].vertex_indices[0]) * (len(mesh.groups[0].vertex_indices[0]) - 1)
             / 2, 1, 2], dtype=np.int32)
        self.vertex_to_ray = []
        for i in mesh.vertices[0]:
            self.vertex_to_ray.append([]);
        count = 0
        for grp in mesh.groups:
            for i in range(0, len(grp.vertex_indices)):
                for j in range(0, len(grp.vertex_indices[i])):
                    for k in range(j + 1, len(grp.vertex_indices[i])):
                        self.ray_vertices[count][0] = grp.vertex_indices[i][j]
                        self.ray_vertices[count][1] = grp.vertex_indices[i][k]
                        temp1 = VertexRay(count, 0)
                        self.vertex_to_ray[grp.vertex_indices[i][j]].append(temp1)
                        temp2 = VertexRay(count, 1)
                        self.vertex_to_ray[grp.vertex_indices[i][k]].append(temp2)
                        count += 1
        ind = 0
        #print(self.ray_vertices)
        for i in self.ray_vertices:
            which = 0
            for grp in mesh.groups:
                for j in range(0, len(grp.vertex_indices)):
                    count = 0
                    for k in grp.vertex_indices[j]:
                        if k == i[0] or k == i[1]:
                            count += 1
                    #found an element sharing edge
                    if count == 2:
                        self.ray_elements[ind][0][which] = j
                        which += 1
                        if which == 2:
                            break
            ind += 1
        '''

    # }}}

    # {{{ helper routines

    def get_refine_base_index(self):
        if self.last_split_elements is None:
            return 0
        else:
            return self.last_mesh.nelements - len(self.last_split_elements)

    def get_empty_refine_flags(self):
        return np.zeros(
                self.last_mesh.nelements - self.get_refine_base_index(),
                np.bool)

    def get_previous_mesh(self):
        return self.previous_mesh

    def get_current_mesh(self):
        return self.last_mesh

    def get_leaves(self, cur_node):
        queue = [cur_node]
        res = []
        while queue:
            vertex = queue.pop(0)
            #if leaf node
            if vertex.left is None and vertex.right is None:
                res.append(vertex)
            else:
                queue.append(vertex.left)
                queue.append(vertex.right)
        return res

    def get_subtree(self, cur_node):
        queue = [cur_node]
        res = []
        while queue:
            vertex = queue.pop(0)
            res.append(vertex)
            if not (vertex.left is None and vertex.right is None):
                queue.append(vertex.left)
                queue.append(vertex.right)
        return res

    def apply_diff(self, cur_node, new_hanging_vertex_elements):
        for el in cur_node.adjacent_add_diff:
            if el not in cur_node.adjacent_elements:
                cur_node.adjacent_elements.append(el)
            if el not in new_hanging_vertex_elements[cur_node.left_vertex]:
                new_hanging_vertex_elements[cur_node.left_vertex].append(el)
            if el not in new_hanging_vertex_elements[cur_node.right_vertex]:
                new_hanging_vertex_elements[cur_node.right_vertex].append(el)
            if cur_node.left is not None and cur_node.right is not None:
                if el not in cur_node.left.adjacent_add_diff:
                    cur_node.left.adjacent_add_diff.append(el)
                if el not in cur_node.right.adjacent_add_diff:
                    cur_node.right.adjacent_add_diff.append(el)
        cur_node.adjacent_add_diff = []

#    def propagate(self, cur_node, new_hanging_vertex_elements):
#        if cur_node.parent is not None:
#            parent_node = cur_node.parent
#            self.propagate(parent_node, new_hanging_vertex_elements)
#            self.apply_diff(parent_node, new_hanging_vertex_elements)

    def get_root(self, cur_node):
        while(cur_node.parent is not None):
            cur_node = cur_node.parent
        return cur_node

    def propagate_tree(self, cur_node, new_hanging_vertex_elements, element_to_element):
        vertex_tuple = (min(cur_node.left_vertex, cur_node.right_vertex), max(cur_node.left_vertex, cur_node.right_vertex))
        self.seen_tuple[vertex_tuple] = True
        self.apply_diff(cur_node, new_hanging_vertex_elements)
        if cur_node.left is not None and cur_node.right is not None:
            self.propagate_tree(cur_node.left, new_hanging_vertex_elements, element_to_element)
            self.propagate_tree(cur_node.right, new_hanging_vertex_elements, element_to_element)
        else:
            for el in cur_node.adjacent_elements:
                element_to_element[el].update(cur_node.adjacent_elements)
            for el in new_hanging_vertex_elements[cur_node.left_vertex]:
                element_to_element[el].update(new_hanging_vertex_elements[cur_node.left_vertex])
            for el in new_hanging_vertex_elements[cur_node.right_vertex]:
                element_to_element[el].update(new_hanging_vertex_elements[cur_node.right_vertex])

#    def remove_from_subtree(self, cur_node, new_hanging_vertex_elements, to_remove):
#        if self.lazy:
#            self.propagate(cur_node, new_hanging_vertex_elements)
#            if to_remove in cur_node.adjacent_add_diff:
#                cur_node.adjacent_add_diff.remove(to_remove)
#            if to_remove not in cur_node.adjacent_remove_diff:
#                cur_node.adjacent_remove_diff.append(to_remove)
#        else:
#            subtree = self.get_subtree(cur_node)
#            for node in subtree:
#                if to_remove in node.adjacent_elements:
#                    node.adjacent_elements.remove(to_remove)
#                if to_remove in new_hanging_vertex_elements[node.left_vertex]:
#                    new_hanging_vertex_elements[node.left_vertex].remove(to_remove)
#                if to_remove in new_hanging_vertex_elements[node.right_vertex]:
#                    new_hanging_vertex_elements[node.right_vertex].remove(to_remove)

    def add_to_subtree(self, cur_node, new_hanging_vertex_elements, to_add):
        if self.lazy:
            if to_add not in cur_node.adjacent_add_diff:
                cur_node.adjacent_add_diff.append(to_add)
        else:
            subtree = self.get_subtree(cur_node)
            for node in subtree:
                if to_add not in node.adjacent_elements:
                    node.adjacent_elements.append(to_add)
                if to_add not in new_hanging_vertex_elements[node.left_vertex]:
                    new_hanging_vertex_elements[node.left_vertex].append(to_add)
                if to_add not in new_hanging_vertex_elements[node.right_vertex]:
                    new_hanging_vertex_elements[node.right_vertex].append(to_add)

    # }}}

    def refine_uniformly(self):
        flags = np.ones(self.last_mesh.nelements, dtype=bool)
        self.refine(flags)

    # {{{ refinement

    def refine(self, refine_flags):
        """
        :arg refine_flags: a :class:`numpy.ndarray` of dtype bool of length ``mesh.nelements``
            indicating which elements should be split.
        """

        if len(refine_flags) != self.last_mesh.nelements:
            raise ValueError("length of refine_flags does not match "
                    "element count of last generated mesh")

        #vertices and groups for next generation
        nvertices = len(self.last_mesh.vertices[0])

        groups = []

        midpoint_already = set()
        grpn = 0
        totalnelements = 0

        for grp in self.last_mesh.groups:
            iel_base = grp.element_nr_base
            nelements = 0
            for iel_grp in range(grp.nelements):
                nelements += 1
                vertex_indices = grp.vertex_indices[iel_grp]
                if refine_flags[iel_base+iel_grp]:
                    cur_dim = len(grp.vertex_indices[iel_grp])-1
                    nelements += len(self.simplex_result[cur_dim]) - 1
                    for i in range(len(vertex_indices)):
                        for j in range(i+1, len(vertex_indices)):
                            i_index = vertex_indices[i]
                            j_index = vertex_indices[j]
                            index_tuple = (i_index, j_index) if i_index < j_index else (j_index, i_index)
                            if index_tuple not in midpoint_already and \
                                self.pair_map[index_tuple].midpoint is None:
                                    nvertices += 1
                                    midpoint_already.add(index_tuple)
            groups.append(np.empty([nelements, len(grp.vertex_indices[0])], dtype=np.int32))
            grpn += 1
            totalnelements += nelements

        vertices = np.empty([len(self.last_mesh.vertices), nvertices])

        new_hanging_vertex_element = [
                [] for i in range(nvertices)]

#        def remove_element_from_connectivity(vertices, new_hanging_vertex_elements, to_remove):
#            #print(vertices)
#            import itertools
#            if len(vertices) == 2:
#                min_vertex = min(vertices[0], vertices[1])
#                max_vertex = max(vertices[0], vertices[1])
#                ray = self.pair_map[(min_vertex, max_vertex)]
#                self.remove_from_subtree(ray, new_hanging_vertex_elements, to_remove)
#                return
#
#            cur_dim = len(vertices)-1
#            element_rays = []
#            midpoints = []
#            split_possible = True
#            for i in range(len(vertices)):
#                for j in range(i+1, len(vertices)):
#                    min_vertex = min(vertices[i], vertices[j])
#                    max_vertex = max(vertices[i], vertices[j])
#                    element_rays.append(self.pair_map[(min_vertex, max_vertex)])
#                    if element_rays[len(element_rays)-1].midpoint is not None:
#                        midpoints.append(element_rays[len(element_rays)-1].midpoint)
#                    else:
#                        split_possible = False

            #for node in element_rays:
                #self.remove_from_subtree(node, new_hanging_vertex_elements, to_remove)
            #if split_possible:
#            if split_possible:
#                node_tuple_to_coord = {}
#                for node_index, node_tuple in enumerate(self.index_to_node_tuple[cur_dim]):
#                    node_tuple_to_coord[node_tuple] = vertices[node_index]
#                for midpoint_index, midpoint_tuple in enumerate(self.index_to_midpoint_tuple[cur_dim]):
#                    node_tuple_to_coord[midpoint_tuple] = midpoints[midpoint_index]
#                for i in range(len(self.simplex_result[cur_dim])):
#                    next_vertices = []
#                    for j in range(len(self.simplex_result[cur_dim][i])):
#                        next_vertices.append(node_tuple_to_coord[self.simplex_node_tuples[cur_dim][self.simplex_result[cur_dim][i][j]]])
#                    all_rays_present = True
#                    for v1 in range(len(next_vertices)):
#                        for v2 in range(v1+1, len(next_vertices)):
#                            vertex_tuple = (min(next_vertices[v1], next_vertices[v2]), max(next_vertices[v1], next_vertices[v2]))
#                            if vertex_tuple not in self.pair_map:
#                                all_rays_present = False
#                    if all_rays_present:
#                        remove_element_from_connectivity(next_vertices, new_hanging_vertex_elements, to_remove)
#                    else:
#                        split_possible = False
#            if not split_possible:
#                next_vertices_list = list(itertools.combinations(vertices, len(vertices)-1))
#                for next_vertices in next_vertices_list:
#                    remove_element_from_connectivity(next_vertices, new_hanging_vertex_elements, to_remove)

        # {{{ Add element to connectivity

        def add_element_to_connectivity(vertices, new_hanging_vertex_elements, to_add):
            if len(vertices) == 2:
                min_vertex = min(vertices[0], vertices[1])
                max_vertex = max(vertices[0], vertices[1])
                ray = self.pair_map[(min_vertex, max_vertex)]
                self.add_to_subtree(ray, new_hanging_vertex_elements, to_add)
                return

            cur_dim = len(vertices)-1
            element_rays = []
            midpoints = []
            split_possible = True
            for i in range(len(vertices)):
                for j in range(i+1, len(vertices)):
                    min_vertex = min(vertices[i], vertices[j])
                    max_vertex = max(vertices[i], vertices[j])
                    element_rays.append(self.pair_map[(min_vertex, max_vertex)])
                    if element_rays[len(element_rays)-1].midpoint is not None:
                        midpoints.append(element_rays[len(element_rays)-1].midpoint)
                    else:
                        split_possible = False
            #for node in element_rays:
                #self.add_to_subtree(node, new_hanging_vertex_elements, to_add)
            if split_possible:
                node_tuple_to_coord = {}
                for node_index, node_tuple in enumerate(self.index_to_node_tuple[cur_dim]):
                    node_tuple_to_coord[node_tuple] = vertices[node_index]
                for midpoint_index, midpoint_tuple in enumerate(self.index_to_midpoint_tuple[cur_dim]):
                    node_tuple_to_coord[midpoint_tuple] = midpoints[midpoint_index]
                for i in range(len(self.simplex_result[cur_dim])):
                    next_vertices = []
                    for j in range(len(self.simplex_result[cur_dim][i])):
                        next_vertices.append(node_tuple_to_coord[self.simplex_node_tuples[cur_dim][self.simplex_result[cur_dim][i][j]]])
                    all_rays_present = True
                    for v1 in range(len(next_vertices)):
                        for v2 in range(v1+1, len(next_vertices)):
                            vertex_tuple = (min(next_vertices[v1], next_vertices[v2]), max(next_vertices[v1], next_vertices[v2]))
                            if vertex_tuple not in self.pair_map:
                                all_rays_present = False
                    if all_rays_present:
                        add_element_to_connectivity(next_vertices, new_hanging_vertex_elements, to_add)
                    else:
                        split_possible = False
            if not split_possible:
                next_vertices_list = list(itertools.combinations(vertices, len(vertices)-1))
                for next_vertices in next_vertices_list:
                    add_element_to_connectivity(next_vertices, new_hanging_vertex_elements, to_add)
#            for node in element_rays:
#                self.add_element_to_connectivity(node, new_hanging_vertex_elements, to_add)
 #               leaves = self.get_subtree(node)
 #               for leaf in leaves:
 #                   if to_add not in leaf.adjacent_elements:
 #                       leaf.adjacent_elements.append(to_add)
 #                   if to_add not in new_hanging_vertex_elements[leaf.left_vertex]:
 #                       new_hanging_vertex_elements[leaf.left_vertex].append(to_add)
 #                   if to_add not in new_hanging_vertex_elements[leaf.right_vertex]:
 #                       new_hanging_vertex_elements[leaf.right_vertex].append(to_add)

#            next_element_rays = []
#            for i in range(len(element_rays)):
#                for j in range(i+1, len(element_rays)):
#                    if element_rays[i].midpoint is not None and element_rays[j].midpoint is not None:
#                        min_midpoint = min(element_rays[i].midpoint, element_rays[j].midpoint)
#                        max_midpoint = max(element_rays[i].midpoint, element_rays[j].midpoint)
#                        vertex_pair = (min_midpoint, max_midpoint)
#                        if vertex_pair in self.pair_map:
#                            next_element_rays.append(self.pair_map[vertex_pair])
#                            cur_next_rays = []
#                            if element_rays[i].left_vertex == element_rays[j].left_vertex:
#                                cur_next_rays = [element_rays[i].left, element_rays[j].left, self.pair_map[vertex_pair]]
#                            if element_rays[i].right_vertex == element_rays[j].right_vertex:
#                                cur_next_rays = [element_rays[i].right, element_rays[j].right, self.pair_map[vertex_pair]]
#                            if element_rays[i].left_vertex == element_rays[j].right_vertex:
#                                cur_next_rays = [element_rays[i].left, element_rays[j].right, self.pair_map[vertex_pair]]
#                            if element_rays[i].right_vertex == element_rays[j].left_vertex:
#                                cur_next_rays = [element_rays[i].right, element_rays[j].left, self.pair_map[vertex_pair]]
#                            assert (cur_next_rays != [])
#                            #print cur_next_rays
#                            add_element_to_connectivity(cur_next_rays, new_hanging_vertex_elements, to_add)
#                        else:
#                            return
#                    else:
#                        return
#            add_element_to_connectivity(next_element_rays, new_hanging_vertex_elements, to_add)

        # }}}

        # {{{ Add hanging vertex element

        def add_hanging_vertex_el(v_index, el):
            assert not (v_index == 37 and el == 48)

            new_hanging_vertex_element[v_index].append(el)

        # }}}

#        def remove_ray_el(ray, el):
#            ray.remove(el)

        # {{{ Check adjacent elements

        def check_adjacent_elements(groups, new_hanging_vertex_elements, nelements_in_grp):
            for grp in groups:
                iel_base = 0
                for iel_grp in range(nelements_in_grp):
                    vertex_indices = grp[iel_grp]
                    for i in range(len(vertex_indices)):
                        for j in range(i+1, len(vertex_indices)):
                            min_index = min(vertex_indices[i], vertex_indices[j])
                            max_index = max(vertex_indices[i], vertex_indices[j])
                            cur_node = self.pair_map[(min_index, max_index)]
                            #print iel_base+iel_grp, cur_node.left_vertex, cur_node.right_vertex
                            #if (iel_base + iel_grp) not in cur_node.adjacent_elements:
                                #print min_index, max_index
                                #print iel_base + iel_grp, cur_node.left_vertex, cur_node.right_vertex, cur_node.adjacent_elements
                                #assert (4 in new_hanging_vertex_elements[cur_node.left_vertex] or 4 in new_hanging_vertex_elements[cur_node.right_vertex])
                            assert ((iel_base+iel_grp) in cur_node.adjacent_elements)
                            assert((iel_base+iel_grp) in new_hanging_vertex_elements[cur_node.left_vertex])
                            assert((iel_base+iel_grp) in new_hanging_vertex_elements[cur_node.right_vertex])

        # }}}

        for i in range(len(self.last_mesh.vertices)):
            for j in range(len(self.last_mesh.vertices[i])):
                vertices[i,j] = self.last_mesh.vertices[i,j]
                import copy
                if i == 0:
                    new_hanging_vertex_element[j] = copy.deepcopy(self.hanging_vertex_element[j])
        grpn = 0
        for grp in self.last_mesh.groups:
            for iel_grp in range(grp.nelements):
                for i in range(len(grp.vertex_indices[iel_grp])):
                    groups[grpn][iel_grp][i] = grp.vertex_indices[iel_grp][i]
            grpn += 1

        grpn = 0
        vertices_index = len(self.last_mesh.vertices[0])
        nelements_in_grp = grp.nelements
        del self.group_refinement_records[:]

        from meshmode.mesh import SimplexElementGroup
        for grp_idx, grp in enumerate(self.last_mesh.groups):
            if not isinstance(grp, SimplexElementGroup):
                raise TypeError("refinement not supported for groups of type "
                        f"'{type(grp).__name__}'")

            iel_base = grp.element_nr_base
            # List of lists mapping element number to new element number(s).
            element_mapping = []
            el_tess_info = None

            # {{{ get midpoint coordinates for vertices

            midpoints_to_find = []
            resampler = None
            for iel_grp in range(grp.nelements):
                if refine_flags[iel_base + iel_grp]:
                    # if simplex
                    if len(grp.vertex_indices[iel_grp]) == grp.dim + 1:
                        midpoints_to_find.append(iel_grp)
                        if not resampler:
                            resampler = SimplexResampler()
                            el_tess_info = ElementTesselationInfo(
                                children=self.simplex_result[grp.dim],
                                ref_vertices=self.simplex_node_tuples[grp.dim])
                    else:
                        raise NotImplementedError("unimplemented: midpoint finding"
                                                  "for non simplex elements")

            if midpoints_to_find:
                midpoints = resampler.get_midpoints(
                        grp, el_tess_info, midpoints_to_find)
                midpoint_order = resampler.get_vertex_pair_to_midpoint_order(grp.dim)

            del midpoints_to_find

            # }}}

            for iel_grp in range(grp.nelements):
                element_mapping.append([iel_grp])
                if refine_flags[iel_base+iel_grp]:
                    midpoint_vertices = []
                    vertex_indices = grp.vertex_indices[iel_grp]
                    # if simplex
                    if len(vertex_indices) == grp.dim + 1:
                        for i in range(len(vertex_indices)):
                            for j in range(i+1, len(vertex_indices)):
                                min_index = min(vertex_indices[i], vertex_indices[j])
                                max_index = max(vertex_indices[i], vertex_indices[j])
                                cur_node = self.pair_map[(min_index, max_index)]
                                if cur_node.midpoint is None:
                                    cur_node.midpoint = vertices_index
                                    import copy
                                    cur_node.left = TreeRayNode(min_index, vertices_index,
                                            copy.deepcopy(cur_node.adjacent_elements))
                                    cur_node.left.parent = cur_node
                                    cur_node.right = TreeRayNode(max_index, vertices_index,
                                            copy.deepcopy(cur_node.adjacent_elements))
                                    cur_node.right.parent = cur_node
                                    vertex_pair1 = (min_index, vertices_index)
                                    vertex_pair2 = (max_index, vertices_index)
                                    self.pair_map[vertex_pair1] = cur_node.left
                                    self.pair_map[vertex_pair2] = cur_node.right
                                    midpoint_idx = midpoint_order[(i, j)]
                                    vertices[:, vertices_index] = \
                                            midpoints[iel_grp][:, midpoint_idx]
                                    midpoint_vertices.append(vertices_index)
                                    vertices_index += 1
                                else:
                                    cur_midpoint = cur_node.midpoint
                                    midpoint_vertices.append(cur_midpoint)

                        #generate new rays
                        cur_dim = len(grp.vertex_indices[0])-1
                        for i in range(len(midpoint_vertices)):
                            for j in range(i+1, len(midpoint_vertices)):
                                min_index = min(midpoint_vertices[i], midpoint_vertices[j])
                                max_index = max(midpoint_vertices[i], midpoint_vertices[j])
                                vertex_pair = (min_index, max_index)
                                if vertex_pair in self.pair_map:
                                    continue
                                self.pair_map[vertex_pair] = TreeRayNode(min_index, max_index, [])
                        node_tuple_to_coord = {}
                        for node_index, node_tuple in enumerate(self.index_to_node_tuple[cur_dim]):
                            node_tuple_to_coord[node_tuple] = grp.vertex_indices[iel_grp][node_index]
                        for midpoint_index, midpoint_tuple in enumerate(self.index_to_midpoint_tuple[cur_dim]):
                            node_tuple_to_coord[midpoint_tuple] = midpoint_vertices[midpoint_index]
                        for i in range(len(self.simplex_result[cur_dim])):
                            if i == 0:
                                iel = iel_grp
                            else:
                                iel = nelements_in_grp + i - 1
                                element_mapping[-1].append(iel)
                            for j in range(len(self.simplex_result[cur_dim][i])):
                                groups[grpn][iel][j] = \
                                    node_tuple_to_coord[self.simplex_node_tuples[cur_dim][self.simplex_result[cur_dim][i][j]]]
                        nelements_in_grp += len(self.simplex_result[cur_dim])-1
                    #assuming quad otherwise
                    else:
                        #quadrilateral
                        raise NotImplementedError("unimplemented: "
                                                  "support for quad elements")
#                        node_tuple_to_coord = {}
#                        for node_index, node_tuple in enumerate(self.index_to_node_tuple[cur_dim]):
#                            node_tuple_to_coord[node_tuple] = grp.vertex_indices[iel_grp][node_index]
#                        def generate_all_tuples(cur_list):
#                            if len(cur_list[len(cur_list)-1])

            self.group_refinement_records.append(
                GroupRefinementRecord(
                    el_tess_info=el_tess_info,
                    element_mapping=element_mapping)
                )

        #clear connectivity data
        for grp in self.last_mesh.groups:
            iel_base = grp.element_nr_base
            for iel_grp in range(grp.nelements):
                for i in range(len(grp.vertex_indices[iel_grp])):
                    for j in range(i+1, len(grp.vertex_indices[iel_grp])):
                        min_vert = min(grp.vertex_indices[iel_grp][i], grp.vertex_indices[iel_grp][j])
                        max_vert = max(grp.vertex_indices[iel_grp][i], grp.vertex_indices[iel_grp][j])
                        vertex_pair = (min_vert, max_vert)
                        root_ray = self.get_root(self.pair_map[vertex_pair])
                        if root_ray not in self.seen_tuple:
                            self.seen_tuple[root_ray] = True
                            cur_tree = self.get_subtree(root_ray)
                            for node in cur_tree:
                                node.adjacent_elements = []
                                new_hanging_vertex_element[node.left_vertex] = []
                                new_hanging_vertex_element[node.right_vertex] = []

        self.seen_tuple.clear()

        nelements_in_grp = grp.nelements
        for grp in groups:
            for iel_grp in range(len(grp)):
                add_verts = []
                for i in range(len(grp[iel_grp])):
                    add_verts.append(grp[iel_grp][i])
                add_element_to_connectivity(add_verts, new_hanging_vertex_element, iel_base+iel_grp)
        #assert ray connectivity
        #check_adjacent_elements(groups, new_hanging_vertex_element, nelements_in_grp)

        self.hanging_vertex_element = new_hanging_vertex_element

        # {{{ make new groups

        new_mesh_el_groups = []

        for refinement_record, group, prev_group in zip(
                self.group_refinement_records, groups, self.last_mesh.groups):
            is_simplex = len(prev_group.vertex_indices[0]) == prev_group.dim + 1
            ambient_dim = len(prev_group.nodes)
            nelements = len(group)
            nunit_nodes = len(prev_group.unit_nodes[0])

            nodes = np.empty(
                (ambient_dim, nelements, nunit_nodes),
                dtype=prev_group.nodes.dtype)

            element_mapping = refinement_record.element_mapping

            to_resample = [elem for elem in range(len(element_mapping))
                           if len(element_mapping[elem]) > 1]

            if to_resample:
                # if simplex
                if is_simplex:
                    resampler = SimplexResampler()
                    new_nodes = resampler.get_tesselated_nodes(
                        prev_group, refinement_record.el_tess_info, to_resample)
                else:
                    raise NotImplementedError(
                        "unimplemented: node resampling for non simplex elements")

            for elem, mapped_elems in enumerate(element_mapping):
                if len(mapped_elems) == 1:
                    # No resampling required, just copy over
                    nodes[:, mapped_elems[0]] = prev_group.nodes[:, elem]
                    n = nodes[:, mapped_elems[0]]
                else:
                    nodes[:, mapped_elems] = new_nodes[elem]

            if is_simplex:
                new_mesh_el_groups.append(
                    type(prev_group)(
                        order=prev_group.order,
                        vertex_indices=group,
                        nodes=nodes,
                        unit_nodes=prev_group.unit_nodes))
            else:
                raise NotImplementedError("unimplemented: support for creating"
                                          "non simplex element groups")

        # }}}

        from meshmode.mesh import Mesh

        refine_flags = refine_flags.astype(np.bool)

        self.previous_mesh = self.last_mesh
        self.last_mesh = Mesh(
                vertices, new_mesh_el_groups,
                nodal_adjacency=self.generate_nodal_adjacency(
                    totalnelements, nvertices, groups),
                vertex_id_dtype=self.last_mesh.vertex_id_dtype,
                element_id_dtype=self.last_mesh.element_id_dtype,
                is_conforming=(
                    self.last_mesh.is_conforming
                    and (refine_flags.all() or (~refine_flags).all())))
        return self.last_mesh

    # }}}

    def print_rays(self, ind):
        for i in range(len(self.last_mesh.groups[0].vertex_indices[ind])):
            for j in range(i+1, len(self.last_mesh.groups[0].vertex_indices[ind])):
                mn = min(self.last_mesh.groups[0].vertex_indices[ind][i],
                        self.last_mesh.groups[0].vertex_indices[ind][j])
                mx = max(self.last_mesh.groups[0].vertex_indices[ind][j],
                        self.last_mesh.groups[0].vertex_indices[ind][i])
                vertex_pair = (mn, mx)
                print('LEFT VERTEX:', self.pair_map[vertex_pair].left_vertex)
                print('RIGHT VERTEX:', self.pair_map[vertex_pair].right_vertex)
                print('ADJACENT:')
                rays = self.get_leaves(self.pair_map[vertex_pair])
                for k in rays:
                    print(k.adjacent_elements)
    '''
    def print_rays(self, groups, ind):
        for i in range(len(groups[0][ind])):
            for j in range(i+1, len(groups[0][ind])):
                mn = min(groups[0][ind][i], groups[0][ind][j])
                mx = max(groups[0][ind][i], groups[0][ind][j])
                vertex_pair = (mn, mx)
                print('LEFT VERTEX:', self.pair_map[vertex_pair].left_vertex)
                print('RIGHT VERTEX:', self.pair_map[vertex_pair].right_vertex)
                print('ADJACENT:')
                rays = self.get_leaves(self.pair_map[vertex_pair])
                for k in rays:
                    print(k.adjacent_elements)
    '''

    def print_hanging_elements(self, ind):
        for i in self.last_mesh.groups[0].vertex_indices[ind]:
            print("IND:", i, self.hanging_vertex_element[i])

    # {{{ generate adjacency

    def generate_nodal_adjacency(self, nelements, nvertices, groups):
        # medium-term FIXME: make this an incremental update
        # rather than build-from-scratch
        vertex_to_element = [[] for i in range(nvertices)]
        element_index = 0
        for grp in groups:
            for iel_grp in range(len(grp)):
                for ivertex in grp[iel_grp]:
                    vertex_to_element[ivertex].append(element_index)
                element_index += 1
        element_to_element = [set() for i in range(nelements)]
        element_index = 0
        if self.lazy:
            for grp in groups:
                for iel_grp in range(len(grp)):
                    for i in range(len(grp[iel_grp])):
                        for j in range(i+1, len(grp[iel_grp])):
                            vertex_pair = (min(grp[iel_grp][i], grp[iel_grp][j]), max(grp[iel_grp][i], grp[iel_grp][j]))
                            #print 'iel:', iel_grp, 'pair:', vertex_pair
                            if vertex_pair not in self.seen_tuple:
                                self.propagate_tree(self.get_root(self.pair_map[vertex_pair]), self.hanging_vertex_element, element_to_element)
                            #print self.pair_map[vertex_pair].left_vertex, self.pair_map[vertex_pair].right_vertex, self.pair_map[vertex_pair].adjacent_elements, self.hanging_vertex_element[self.pair_map[vertex_pair].left_vertex], self.hanging_vertex_element[self.pair_map[vertex_pair].right_vertex]

        else:
            for grp in groups:
                for iel_grp in range(len(grp)):
                    for ivertex in grp[iel_grp]:
                        element_to_element[element_index].update(
                                vertex_to_element[ivertex])
                        if self.hanging_vertex_element[ivertex]:
                            for hanging_element in self.hanging_vertex_element[ivertex]:
                                if element_index != hanging_element:
                                    element_to_element[element_index].update([hanging_element])
                                    element_to_element[hanging_element].update([element_index])
                    for i in range(len(grp[iel_grp])):
                        for j in range(i+1, len(grp[iel_grp])):
                            vertex_pair = (min(grp[iel_grp][i], grp[iel_grp][j]),
                                    max(grp[iel_grp][i], grp[iel_grp][j]))
                            #element_to_element[element_index].update(
                                    #self.pair_map[vertex_pair].adjacent_elements)
                            queue = [self.pair_map[vertex_pair]]
                            while queue:
                                vertex = queue.pop(0)
                                #if leaf node
                                if vertex.left is None and vertex.right is None:
                                    assert(element_index in vertex.adjacent_elements)
                                    element_to_element[element_index].update(
                                            vertex.adjacent_elements)
                                else:
                                    queue.append(vertex.left)
                                    queue.append(vertex.right)
                        '''
                        if self.hanging_vertex_element[ivertex] and element_index != self.hanging_vertex_element[ivertex][0]:
                            element_to_element[element_index].update([self.hanging_vertex_element[ivertex][0]])
                            element_to_element[self.hanging_vertex_element[ivertex][0]].update([element_index])
                            '''
                    element_index += 1
        logger.debug("number of new elements: %d" % len(element_to_element))
        for iel, neighbors in enumerate(element_to_element):
            if iel in neighbors:
                neighbors.remove(iel)
        #print(self.ray_elements)
        '''
        for ray in self.rays:
            curnode = ray.first
            while curnode is not None:
                if len(curnode.value.elements) >= 2:
                    if curnode.value.elements[0] is not None:
                        element_to_element[curnode.value.elements[0]].update(curnode.value.elements)
                    if curnode.value.elements[1] is not None:
                        element_to_element[curnode.value.elements[1]].update(curnode.value.elements)
                if len(curnode.value.velements) >= 2:
                    if curnode.value.velements[0] is not None:
                        element_to_element[curnode.value.velements[0]].update(curnode.value.velements)
                    if curnode.value.velements[1] is not None:
                        element_to_element[curnode.value.velements[1]].update(curnode.value.velements)
                curnode = curnode.next
        '''
        '''
        for i in self.ray_elements:
            for j in i:
                #print j[0], j[1]
                element_to_element[j[0]].update(j)
                element_to_element[j[1]].update(j)
        '''
        #print element_to_element
        lengths = [len(el_list) for el_list in element_to_element]
        neighbors_starts = np.cumsum(
                np.array([0] + lengths, dtype=self.last_mesh.element_id_dtype),
                # cumsum silently widens integer types
                dtype=self.last_mesh.element_id_dtype)
        from pytools import flatten
        neighbors = np.array(
                list(flatten(element_to_element)),
                dtype=self.last_mesh.element_id_dtype)

        assert neighbors_starts[-1] == len(neighbors)

        from meshmode.mesh import NodalAdjacency
        return NodalAdjacency(neighbors_starts=neighbors_starts, neighbors=neighbors)

    # }}}


def refine_uniformly(mesh, iterations, with_adjacency=False):
    if with_adjacency:
        # For conforming meshes, even RefinerWithoutAdjacency will reconstruct
        # adjacency from vertex identity.

        if not mesh.is_conforming:
            raise ValueError("mesh must be conforming if adjacency is desired")

    refiner = RefinerWithoutAdjacency(mesh)

    for _ in range(iterations):
        refiner.refine_uniformly()

    return refiner.get_current_mesh()


# vim: foldmethod=marker
