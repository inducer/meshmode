from __future__ import division, print_function

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


import numpy as np
import itertools
from six.moves import range
from pytools import RecordWithoutPickling

from meshmode.mesh.generation import make_group_from_vertices

class Refiner(object):

    class _Tesselation(RecordWithoutPickling):

        def __init__(self, children, ref_vertices):
            RecordWithoutPickling.__init__(self,
                ref_vertices=ref_vertices, children=children)

    class _GroupRefinementRecord(RecordWithoutPickling):

        def __init__(self, tesselation, element_mapping):
            RecordWithoutPickling.__init__(self,
                tesselation=tesselation, element_mapping=element_mapping)

    # {{{ constructor

    #reorder node tuples array so that only one coordinate changes between adjacent indices
    def reorder(self, arr):
        from six.moves import range
        res = []
        res.append(arr.pop(0))
        while len(arr) > 0:
            for i in range(len(arr)):
                count = 0
                for j in range(len(arr[i])):
                    if res[len(res)-1][j] != arr[i][j]:
                        count += 1
                if count == 1:
                    res.append(arr.pop(i))
                    break
        return res

    def __init__(self, mesh):
        from six.moves import range
        from meshmode.mesh.tesselate import tesselatetet, tesselatetri, tesselatesquare, tesselatesegment, tesselatepoint, tesselatecube
        from meshmode.mesh.refinement.resampler import SimplexResampler
        self.simplex_resampler = SimplexResampler()
        self.pair_map = {}
        self.groups = []
        self.group_refinement_records = []
        tri_node_tuples, tri_result = tesselatetri()
        tet_node_tuples, tet_result = tesselatetet()
        segment_node_tuples, segment_result = tesselatesegment()
        square_node_tuples, square_result = tesselatesquare()
        cube_node_tuples, cube_result = tesselatecube()
        point_node_tuples, point_result = tesselatepoint()
        self.simplex_node_tuples = [point_node_tuples, segment_node_tuples, tri_node_tuples, tet_node_tuples]
        # Dimension-parameterized tesselations for refinement
        self.simplex_result = [point_result, segment_result, tri_result, tet_result]
        self.quad_node_tuples = [point_node_tuples, segment_node_tuples, square_node_tuples, cube_node_tuples]
        self.quad_result = [point_result, segment_result, square_result, cube_result]
        self.last_mesh = mesh

        nvertices = len(mesh.vertices[0])
        self.adjacent_set = [set() for _ in range(nvertices)]
        # {{{ initialization

        for grp in mesh.groups:
            iel_base = grp.element_nr_base
            for iel_grp in range(grp.nelements):
                vert_indices = grp.vertex_indices[iel_grp]
                for i in range(len(vert_indices)):
                    self.adjacent_set[vert_indices[i]].add(iel_base+iel_grp)
        # }}}

        self.simplex_index_to_node_tuple = []
        self.simplex_index_to_midpoint_tuple = []
        self.quad_index_to_node_tuple = []
        self.quad_index_to_midpoint_tuple = []
        #put tuples that don't have a 1 in index_to_node_tuple, and put those that do in index_to_midpoint_tuple
        for d in range(4):
            cur_simplex_index_to_node_tuple = []
            cur_simplex_index_to_midpiont_tuple = []
            cur_quad_index_to_node_tuple = []
            cur_quad_index_to_midpoint_tuple = []
            if self.simplex_node_tuples[d] is not None:
                for i in range(len(self.simplex_node_tuples[d])):
                    if 1 not in self.simplex_node_tuples[d][i]:
                        cur_simplex_index_to_node_tuple.append(self.simplex_node_tuples[d][i])
                    else:
                        cur_simplex_index_to_midpiont_tuple.append(self.simplex_node_tuples[d][i])
            if self.quad_node_tuples[d] is not None:
                for i in range(len(self.quad_node_tuples[d])):
                    if 1 not in self.quad_node_tuples[d][i]:
                        cur_quad_index_to_node_tuple.append(self.quad_node_tuples[d][i])
                    else:
                        cur_quad_index_to_midpoint_tuple.append(self.quad_node_tuples[d][i])
            self.simplex_index_to_node_tuple.append(cur_simplex_index_to_node_tuple)
            self.simplex_index_to_midpoint_tuple.append(cur_simplex_index_to_midpiont_tuple)
            self.quad_index_to_node_tuple.append(cur_quad_index_to_node_tuple)
            self.quad_index_to_midpoint_tuple.append(cur_quad_index_to_midpoint_tuple)
        for i in range(len(self.quad_index_to_node_tuple)):
            self.quad_index_to_node_tuple[i] = self.reorder(self.quad_index_to_node_tuple[i])

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


    # }}}

    # {{{ refinement

    def nelements_after_refining(self, group_index, iel_grp):
        from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
        grp = self.last_mesh.groups[group_index]
        element_vertices = grp.vertex_indices[iel_grp]
        dimension = self.last_mesh.groups[group_index].dim
        #simplex
        if isinstance(self.last_mesh.groups[group_index], SimplexElementGroup):
            return (len(self.simplex_result[dimension]))
        #quad
        elif isinstance(self.last_mesh.groups[group_index], TensorProductElementGroup):
            return len(self.quad_result[dimension])

    def midpoint_of_node_tuples(self, tupa, tupb):
        from six.moves import range
        assert(len(tupa) == len(tupb))
        res = ()
        for k in range(len(tupa)):
            res = res + ((tupa[k] + tupb[k])/2,)
        return res

    def node_tuple_to_vertex_index(self, vertices, dimension, index_to_node_tuple):
        from six.moves import range
        node_tuple_to_vertex_index = {}
        for i in range(len(vertices)):
            node_tuple_to_vertex_index[index_to_node_tuple[dimension][i]] = vertices[i]
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                index_i = vertices[i]
                index_j = vertices[j]
                indices_pair = (index_i, index_j) if index_i < index_j else (index_j, index_i)
                if indices_pair in self.pair_map: 
                    midpoint_tuple = self.midpoint_of_node_tuples(index_to_node_tuple[dimension][i], 
                            index_to_node_tuple[dimension][j])
                    node_tuple_to_vertex_index[midpoint_tuple] = self.pair_map[indices_pair]
        return node_tuple_to_vertex_index
    
    def next_vertices_and_dimension(self, vertices, dimension, result_tuples, node_tuples, index_to_node_tuple):
        from six.moves import range
        from itertools import combinations
        node_tuple_to_vertex_index = self.node_tuple_to_vertex_index(vertices, dimension, index_to_node_tuple)
        #can tesselate current element
        if len(node_tuple_to_vertex_index) == len(node_tuples[dimension]):
            result_vertices = []
            for subelement in result_tuples[dimension]:
                current_subelement_vertices = []
                for index in subelement:
                    current_subelement_vertices.append(node_tuple_to_vertex_index[node_tuples[dimension][index]])
                result_vertices.append(current_subelement_vertices)
            return (result_vertices, dimension)
        #move to lower dimension
        result_vertices = list(combinations(vertices, len(index_to_node_tuple[dimension-1])))
        return (result_vertices, dimension-1)

    def remove_from_adjacent_set(self, element_index, vertices, dimension, result_tuples, node_tuples, index_to_node_tuple):
        if len(vertices) == 1:
            if element_index in self.adjacent_set[vertices[0]]:
                self.adjacent_set[vertices[0]].remove(element_index)
        else:
            (next_vertices, next_dimension) = self.next_vertices_and_dimension(vertices, dimension, result_tuples, node_tuples, index_to_node_tuple)
            for cur_vertices in next_vertices:
                self.remove_from_adjacent_set(element_index, cur_vertices, next_dimension, result_tuples, node_tuples, index_to_node_tuple)

    def add_to_adjacent_set(self, element_index, vertices, dimension, result_tuples, node_tuples, index_to_node_tuple):
        if len(vertices) == 1:
            if element_index not in self.adjacent_set[vertices[0]]:
                self.adjacent_set[vertices[0]].add(element_index)
        else:
            (next_vertices, next_dimension) = self.next_vertices_and_dimension(vertices, dimension, result_tuples, node_tuples, index_to_node_tuple)
            for cur_vertices in next_vertices:
                self.add_to_adjacent_set(element_index, cur_vertices, next_dimension, result_tuples, node_tuples, index_to_node_tuple)

    #creates midpoints in result_vertices and updates adjacent_set with midpoint vertices
    def create_midpoints(self, group_index, iel_grp, element_vertices, nvertices, index_to_node_tuple, midpoints, midpoint_order):
        from six.moves import range
        dimension = self.last_mesh.groups[group_index].dim
        midpoint_tuple_to_index = {}
        for i in range(len(element_vertices)):
            for j in range(i+1, len(element_vertices)):
                index_i = element_vertices[i]
                index_j = element_vertices[j]
                indices_pair = (index_i, index_j) if index_i < index_j else (index_j, index_i)
                if indices_pair in self.pair_map:
                    midpoint_tuple = self.midpoint_of_node_tuples(index_to_node_tuple[dimension][i], 
                            index_to_node_tuple[dimension][j])
                    midpoint_tuple_to_index[midpoint_tuple] = self.pair_map[indices_pair]

        for i in range(len(element_vertices)):
            for j in range(i+1, len(element_vertices)):
                index_i = element_vertices[i]
                index_j = element_vertices[j]
                indices_pair = (index_i, index_j) if index_i < index_j else (index_j, index_i)
                midpoint_tuple = self.midpoint_of_node_tuples(index_to_node_tuple[dimension][i], 
                        index_to_node_tuple[dimension][j])
                if indices_pair not in self.pair_map:
                    if midpoint_tuple not in midpoint_tuple_to_index and (i, j) in midpoint_order:
                        self.pair_map[indices_pair] = nvertices
                        for k in range(len(self.last_mesh.vertices)):
                            self.vertices[k, nvertices] = \
                                    midpoints[iel_grp][k][midpoint_order[(i, j)]]
                            (self.last_mesh.vertices[k, index_i] +
                            self.last_mesh.vertices[k, index_j]) / 2.0
                        #update adjacent_set
                        self.adjacent_set.append(self.adjacent_set[index_i].intersection(self.adjacent_set[index_j]))
                        nvertices += 1
                    else:
                        self.pair_map[indices_pair] = midpoint_tuple_to_index[midpoint_tuple]
                midpoint_tuple_to_index[midpoint_tuple] = self.pair_map[indices_pair]
                    
        return nvertices

    #returns element indices and vertices
    def create_elements(self, element_index, element_vertices, dimension, group_index, nelements_in_grp, 
            element_mapping, result_tuples, node_tuples, index_to_node_tuple):
        result = []
        node_tuple_to_vertex_index = self.node_tuple_to_vertex_index(element_vertices, dimension, index_to_node_tuple)
        for subelement_index, subelement in enumerate(result_tuples[dimension]):
            if subelement_index == 0:
                result_vertices = []
                for i, index in enumerate(subelement):
                    
                    cur_node_tuple = node_tuples[dimension][index]
                    self.groups[group_index][element_index][i] = node_tuple_to_vertex_index[cur_node_tuple]
                    result_vertices.append(node_tuple_to_vertex_index[cur_node_tuple])
                element_mapping[-1].append(element_index)
                result.append((element_index, result_vertices))
            else:
                result_vertices = []
                for i, index in enumerate(subelement):
                    cur_node_tuple = node_tuples[dimension][index]
                    self.groups[group_index][nelements_in_grp][i] = node_tuple_to_vertex_index[cur_node_tuple]
                    result_vertices.append(node_tuple_to_vertex_index[cur_node_tuple])
                element_mapping[-1].append(nelements_in_grp)
                result.append((nelements_in_grp, result_vertices))
                nelements_in_grp += 1
        return (result, nelements_in_grp, element_mapping)

    def refine_element(self, group_index, iel_grp, nelements_in_grp, nvertices, element_mapping, result_tuples, node_tuples, index_to_node_tuple, midpoints, midpoint_order):
        from six.moves import range
        grp = self.last_mesh.groups[group_index]
        iel_base = grp.element_nr_base
        element_vertices = grp.vertex_indices[iel_grp]
        dimension = self.last_mesh.groups[group_index].dim
        self.remove_from_adjacent_set(iel_base + iel_grp, element_vertices, dimension, result_tuples, node_tuples, index_to_node_tuple)
        nvertices = self.create_midpoints(group_index, iel_grp, element_vertices, nvertices, index_to_node_tuple, midpoints, midpoint_order)
        element_mapping.append([])
        (subelement_indices_and_vertices, nelements_in_grp, element_mapping) = self.create_elements(iel_grp, 
                element_vertices, dimension, group_index, nelements_in_grp, element_mapping,
                result_tuples, node_tuples, index_to_node_tuple)
        for (index, vertices) in subelement_indices_and_vertices:
            self.add_to_adjacent_set(index, vertices, dimension, result_tuples, node_tuples, index_to_node_tuple)
        return (nelements_in_grp, nvertices, element_mapping)

    def elements_connected_to(self, element_index, vertices, dimension, result_tuples, node_tuples, index_to_node_tuple):
        from six.moves import range
        result = set()
        if len(vertices) == 1:
            if element_index in self.adjacent_set[vertices[0]]:
                result = result.union(self.adjacent_set[vertices[0]])
        else:
            (next_vertices, next_dimension) = self.next_vertices_and_dimension(vertices, dimension, result_tuples, node_tuples, index_to_node_tuple)
            for cur_vertices in next_vertices:
                result = result.union(self.elements_connected_to(element_index, cur_vertices, next_dimension, result_tuples, node_tuples, index_to_node_tuple))
        return result
    
    def perform_refinement(self, group_index, iel_grp, nelements_in_grp, nvertices, element_mapping,
            midpoints, midpoint_order):
        from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
        grp = self.last_mesh.groups[group_index]
        element_vertices = grp.vertex_indices[iel_grp]
        dimension = self.last_mesh.groups[group_index].dim
        #simplex
        if isinstance(self.last_mesh.groups[group_index], SimplexElementGroup):
            return self.refine_element(group_index, iel_grp, nelements_in_grp, nvertices,
                    element_mapping, self.simplex_result, self.simplex_node_tuples,
                    self.simplex_index_to_node_tuple, midpoints, midpoint_order
                    )
        #quad
        elif isinstance(self.last_mesh.groups[group_index], TensorProductElementGroup):
            return self.refine_element(group_index, iel_grp, nelements_in_grp, nvertices,
                    element_mapping, self.quad_result, self.quad_node_tuples,
                    self.quad_index_to_node_tuple, midpoints, midpoint_order
                    )

    def get_elements_connected_to(self, result_groups, group_index, iel_base, iel_grp): 
        import copy
        from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
        grp = result_groups[group_index]
        element_vertices = copy.deepcopy(grp[iel_grp])
        dimension = self.last_mesh.groups[group_index].dim
        #simplex
        if isinstance(self.last_mesh.groups[group_index], SimplexElementGroup):
            return self.elements_connected_to(iel_base+iel_grp, element_vertices, dimension,
                    self.simplex_result, self.simplex_node_tuples, self.simplex_index_to_node_tuple)
        #quad
        elif isinstance(self.last_mesh.groups[group_index], TensorProductElementGroup):
            return self.elements_connected_to(iel_base+iel_grp, element_vertices, dimension,
                    self.quad_result, self.quad_node_tuples, self.quad_index_to_node_tuple)

    def element_index_to_node_tuple(self, grp):
        from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
        #simplex
        if isinstance(grp, SimplexElementGroup):
            return self.simplex_index_to_node_tuple
        #quad
        elif isinstance(grp, TensorProductElementGroup):
            return self.quad_index_to_node_tuple

    def group_index(self, el):
        for grp_index, grp in enumerate(self.last_mesh.groups):
            iel_base = grp.element_nr_base
            if iel_base > el:
                return grp_index - 1
        return len(self.last_mesh.groups)-1

    def update_coarsen_connectivity(self, coarsen_el, new_el_index, old_vertices, new_vertices, dimension, result, node_tuples, index_to_node_tuple):
        from six.moves import range
        for el_index, el in enumerate(coarsen_el):
            self.remove_from_adjacent_set(el, old_vertices[el_index], dimension, result, node_tuples, index_to_node_tuple)
        #remove midpoints from pair_map
        for i in range(len(new_vertices)):
            for j in range(i+1, len(new_vertices)):
                indices_pair = (i, j) if i < j else (i, j)
                if indices_pair in self.pair_map:
                    del self.pair_map[indices_pair]
        self.add_to_adjacent_set(new_el_index, new_vertices, dimension, result, node_tuples, index_to_node_tuple)

    def coarsen(self, coarsen_els):
        from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
        from six.moves import range
        import copy
        nvertices = len(self.last_mesh.vertices[0])
        grp_nelements_to_remove = {}
        self.groups = []
        midpoints = set()
        for to_coarsen in coarsen_els:
            vertices_seen = {}
            grp_index = self.group_index(to_coarsen[0])
            grp = self.last_mesh.groups[grp_index]
            if grp_index not in grp_nelements_to_remove:
                grp_nelements_to_remove[grp_index] = 0
            if isinstance(grp, SimplexElementGroup):
                grp_nelements_to_remove[grp_index] += len(self.simplex_result[grp.dim]) - 1
            elif isinstance(grp, TensorProductElementGroup):
                grp_nelements_to_remove[grp_index] += len(self.quad_result[grp.dim]) - 1
            for el in to_coarsen:
                vertex_indices = grp.vertex_indices[el]
                for i in vertex_indices:
                    if i not in vertices_seen:
                        vertices_seen[i] = 0
                    vertices_seen[i] += 1
            for vertex in vertices_seen:
                if vertices_seen[vertex] > 1:
                    midpoints.add(vertex)
        nvertices -= len(midpoints)
        self.vertices = np.empty([len(self.last_mesh.vertices), nvertices])
        to_coarsen_groups = []
        totalnelements = 0
        for grp_index, grp in enumerate(self.last_mesh.groups):
            nelements = grp.nelements
            to_coarsen_groups.append(np.zeros([nelements], dtype=np.int32))
            if grp_index in grp_nelements_to_remove:
                nelements -= grp_nelements_to_remove[grp_index]
            self.groups.append(np.empty([nelements, len(grp.vertex_indices[0])], dtype=np.int32))
            for el in range(nelements):
                for ind in range(grp.dim):
                    to_coarsen_groups[grp_index][el] = -1
            totalnelements += nelements
        for to_coarsen_index, to_coarsen in enumerate(coarsen_els):
            grp_index = self.group_index(to_coarsen[0])
            grp = self.last_mesh.groups[grp_index]
            iel_base = grp.element_nr_base
            for el in to_coarsen:
                to_coarsen_groups[grp_index][el-iel_base] = to_coarsen_index
        vertex_mapping = {}
        for i in range(len(self.last_mesh.vertices)):
            count = 0
            for j in range(len(self.last_mesh.vertices[i])):
                if j in midpoints:
                    count += 1
                else:
                    self.vertices[i, j-count] = self.last_mesh.vertices[i, j]
                    vertex_mapping[j] = j-count

        coarsen_el_vertices = {}
        coarsen_el_new_index = {}
        for grp_index, grp in enumerate(self.last_mesh.groups):
            cur_index = 0
            seen_coarsen = set()
            for el in range(grp.nelements):
                coarsen_index = to_coarsen_groups[grp_index][el]
                if coarsen_index == -1:
                    self.groups[grp_index][cur_index] = copy.deepcopy(grp.vertex_indices[el])
                    cur_index += 1
                elif coarsen_index not in seen_coarsen:
                    seen_coarsen.add(coarsen_index)
                    coarsen_el_new_index[coarsen_index] = cur_index
                    coarsen_el_vertices[coarsen_index] = []
                    vertices = []
                    for coarsen_el in coarsen_els[coarsen_index]:
                        for i in grp.vertex_indices[coarsen_el]:
                            vertices.append(i)
                    cur_vertex_index = 0
                    for vertex in vertices:
                        if vertex not in midpoints:
                            coarsen_el_vertices[coarsen_index].append(vertex)
                            self.groups[grp_index][cur_index][cur_vertex_index] = vertex_mapping[vertex]
                            cur_vertex_index += 1
                    cur_index += 1
        for coarsen_index, to_coarsen in enumerate(coarsen_els):
            grp_index = self.group_index(to_coarsen[0])
            coarsen_vertex_indices = []
            grp = self.last_mesh.groups[grp_index]
            for el in to_coarsen:
                coarsen_vertex_indices.append(copy.deepcopy(grp.vertex_indices[el]))
            if isinstance(grp, SimplexElementGroup):
                self.update_coarsen_connectivity(to_coarsen, coarsen_el_new_index[coarsen_index], coarsen_vertex_indices,
                        coarsen_el_vertices[coarsen_index], grp.dim, self.simplex_result, 
                        self.simplex_node_tuples, self.simplex_index_to_node_tuple)
            elif isinstance(grp, TensorProductElementGroup):
                self.update_coarsen_connectivity(to_coarsen, coarsen_el_new_index[coarsen_index], coarsen_vertex_indices,
                        coarsen_el_vertices[coarsen_index], grp.dim, self.quad_result, 
                        self.quad_node_tuples, self.quad_index_to_node_tuple)

        grp = []
        for group in self.groups:
            grp.append(make_group_from_vertices(self.vertices, group, 4))

        from meshmode.mesh import Mesh

        self.previous_mesh = self.last_mesh
        self.last_mesh = Mesh(
                self.vertices, grp,
                nodal_adjacency=self.generate_nodal_adjacency(
                    totalnelements, nvertices, self.groups),
                vertex_id_dtype=self.last_mesh.vertex_id_dtype,
                element_id_dtype=self.last_mesh.element_id_dtype)
        return self.last_mesh
        

    def refine(self, refine_flags):
        from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
        from six.moves import range
        """
        :arg refine_flags: a :class:`numpy.ndarray` of dtype bool of length ``mesh.nelements``
            indicating which elements should be split.
        """

        if len(refine_flags) != self.last_mesh.nelements:
            raise ValueError("length of refine_flags does not match "
                    "element count of last generated mesh")

        #vertices and groups for next generation
        nvertices = len(self.last_mesh.vertices[0])
        self.groups = []

        midpoints_to_find = []
        midpoint_already = set()
        totalnelements = 0
        for grp_index, grp in enumerate(self.last_mesh.groups):
            iel_base = grp.element_nr_base
            nelements = 0
            for iel_grp in range(grp.nelements):
                midpoints_to_find.append(iel_grp)
                nelements += 1
                vertex_indices = grp.vertex_indices[iel_grp]
                dimension = self.last_mesh.groups[grp_index].dim
                index_to_node_tuple = self.element_index_to_node_tuple(grp)
                midpoint_tuple_to_index = {}
                if refine_flags[iel_base+iel_grp]:
                    cur_dim = len(grp.vertex_indices[iel_grp])-1
                    nelements += self.nelements_after_refining(grp_index, iel_grp) - 1
                    for i in range(len(vertex_indices)):
                        for j in range(i+1, len(vertex_indices)):
                            i_index = vertex_indices[i]
                            j_index = vertex_indices[j]
                            index_tuple = (i_index, j_index) if i_index < j_index else (j_index, i_index)
                            midpoint_tuple = self.midpoint_of_node_tuples(index_to_node_tuple[dimension][i], 
                                    index_to_node_tuple[dimension][j])
                            if index_tuple not in midpoint_already and \
                                index_tuple not in self.pair_map and \
                                midpoint_tuple not in midpoint_tuple_to_index:
                                        nvertices += 1
                            midpoint_already.add(index_tuple)
                            midpoint_tuple_to_index[midpoint_tuple] = index_tuple
            self.groups.append(np.empty([nelements, len(grp.vertex_indices[iel_grp])], dtype=np.int32))
            totalnelements += nelements
        self.vertices = np.empty([len(self.last_mesh.vertices), nvertices])
        #copy vertices
        for i in range(len(self.last_mesh.vertices)):
            for j in range(len(self.last_mesh.vertices[i])):
                self.vertices[i, j] = self.last_mesh.vertices[i, j]

        nvertices = len(self.last_mesh.vertices[0])
        
        del self.group_refinement_records[:]

        #do actual refinement
        for grp_index, grp in enumerate(self.last_mesh.groups):
            element_mapping = []
            tesselation = None
            midpoints = None
            iel_base = grp.element_nr_base
            nelements_in_grp = grp.nelements
            midpoints_to_find = [iel_grp for iel_grp in range(grp.nelements) if
                    refine_flags[iel_base+iel_grp]]
            if isinstance(grp, SimplexElementGroup):
                tesselation = self._Tesselation(
                    self.simplex_result[grp.dim], self.simplex_node_tuples[grp.dim])
                midpoints = self.simplex_resampler.get_midpoints(
                        grp, tesselation, midpoints_to_find)
                midpoint_order = self.simplex_resampler.get_vertex_pair_to_midpoint_order(grp.dim)
            elif isinstance(grp, TensorProductElementGroup):
                tesselation = self._Tesselation(
                    self.quad_result[dimension], self.quad_node_tuples[dimension])
                #replace with resampler
                num_midpoints = 0
                for vertex in tesselation.ref_vertices:
                    if 1 in vertex:
                        num_midpoints += 1
                midpoints = np.zeros((len(midpoints_to_find), len(self.last_mesh.vertices), 
                    num_midpoints))
                midpoint_order = {}
                cur_el = 0
                for iel_grp in range(grp.nelements):
                    if refine_flags[iel_base+iel_grp]:
                        cur = 0
                        midpoint_tuple_to_index = set()
                        for i in range(len(grp.vertex_indices[iel_grp])):
                            for j in range(i+1, len(grp.vertex_indices[iel_grp])):
                                midpoint_tuple = self.midpoint_of_node_tuples(
                                        index_to_node_tuple[dimension][i], 
                                        index_to_node_tuple[dimension][j])
                                if midpoint_tuple not in midpoint_tuple_to_index:
                                    midpoint_tuple_to_index.add(midpoint_tuple)
                                    for k in range(len(self.last_mesh.vertices)):
                                        midpoints[cur_el,k,cur] = (self.last_mesh.vertices[
                                            k, grp.vertex_indices[iel_grp][i]] + 
                                            self.last_mesh.vertices[
                                                k, grp.vertex_indices[iel_grp][j]]) / 2.0
                                        midpoint_order[(i, j)] = cur
                                    cur += 1
                        cur_el += 1
                midpoints = dict(zip(midpoints_to_find, midpoints))
            for iel_grp in range(grp.nelements):
                if refine_flags[iel_base + iel_grp]:
                    (nelements_in_grp, nvertices, element_mapping) = \
                    self.perform_refinement(grp_index, iel_grp, nelements_in_grp, 
                            nvertices, element_mapping, midpoints, midpoint_order)
                else:
                    for i in range(len(grp.vertex_indices[iel_grp])):
                        self.groups[grp_index][iel_grp][i] = grp.vertex_indices[iel_grp][i]
                    element_mapping.append([iel_grp])
            self.group_refinement_records.append(
                self._GroupRefinementRecord(tesselation, element_mapping))

        grp = []
        for refinement_record, group, prev_group in zip(
                self.group_refinement_records, self.groups, self.last_mesh.groups):
            is_quad = isinstance(prev_group, TensorProductElementGroup)
            if is_quad:
                grp.append(make_group_from_vertices(self.vertices, group, 4))
                continue
            is_simplex = isinstance(prev_group, SimplexElementGroup)
            ambient_dim = self.last_mesh.ambient_dim
            nelements = len(group)
            nunit_nodes = len(prev_group.unit_nodes[0])

            nodes = np.empty(
                    (ambient_dim, nelements, nunit_nodes),
                    dtype=prev_group.nodes.dtype)
            
            element_mapping = refinement_record.element_mapping

            to_resample = [elem for elem in range(len(element_mapping))
                    if len(element_mapping[elem]) > 1]
            
            if to_resample:
                if is_simplex:
                    from meshmode.mesh.refinement.resampler import SimplexResampler
                    resampler = SimplexResampler()
                    new_nodes = resampler.get_tesselated_nodes(
                            prev_group, refinement_record.tesselation, to_resample)
                else:
                    raise NotImplementedError(
                            "unimplemented: node resampling for non simplex and non quad elements")
            for elem, mapped_elems in enumerate(element_mapping):
                if len(mapped_elems) == 1:
                    nodes[:, mapped_elems[0]] = prev_group.nodes[:, elem]
                    n = nodes[:, mapped_elems[0]]
                else:
                    nodes[:, mapped_elems] = new_nodes[elem]
            if is_simplex:
                grp.append(
                        type(prev_group)(
                            order=prev_group.order,
                            vertex_indices=group,
                            nodes=nodes,
                            unit_nodes=prev_group.unit_nodes))
            else:
                raise NotImplementedError("unimplemented: support for creating"
                                          "non simplex and non quad groups")
                
        from meshmode.mesh import Mesh

        self.previous_mesh = self.last_mesh
        self.last_mesh = Mesh(
                self.vertices, grp,
                nodal_adjacency=self.generate_nodal_adjacency(
                    totalnelements, nvertices, self.groups),
                vertex_id_dtype=self.last_mesh.vertex_id_dtype,
                element_id_dtype=self.last_mesh.element_id_dtype)
        return self.last_mesh
    # }}}

    # {{{ generate adjacency

    def generate_nodal_adjacency(self, nelements, nvertices, groups):
        # medium-term FIXME: make this an incremental update
        # rather than build-from-scratch
        element_to_element = []
        iel_base = 0
        for group_index, grp in enumerate(groups):
            for iel_grp in range(len(grp)):
                element_to_element.append(self.get_elements_connected_to(groups, group_index, iel_base, iel_grp))
            iel_base += len(grp)
        for iel, neighbors in enumerate(element_to_element):
            if iel in neighbors:
                neighbors.remove(iel)
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


# vim: foldmethod=marker
