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

    def __init__(self, mesh):
        from six.moves import range
        from meshmode.mesh.tesselate import tesselatetet, tesselatetri, tesselatesquare, tesselatesegment
        self.pair_map = {}
        self.group_refinement_records = []
        tri_node_tuples, tri_result = tesselatetri()
        tet_node_tuples, tet_result = tesselatetet()
        segment_node_tuples, segment_result = tesselatesegment()
        square_node_tuples, square_result = tesselatesquare()
        self.simplex_node_tuples = [None, segment_node_tuples, tri_node_tuples, tet_node_tuples]
        # Dimension-parameterized tesselations for refinement
        self.simplex_result = [None, segment_result, tri_result, tet_result]
        self.quad_node_tuples = [None, segment_node_tuples, square_node_tuples]
        self.quad_result = [None, segment_result, square_result]
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

        self.index_to_node_tuple = []
        self.index_to_midpoint_tuple = []
        #put tuples that don't have a 1 in index_to_node_tuple, and put those that do in index_to_midpoint_tuple
        for d in range(len(vert_indices)):
            #if simplex
            cur_index_to_node_tuple = []
            cur_index_to_midpiont_tuple = []
            if len(grp.vertex_indices[iel_grp]) == len(self.last_mesh.vertices)+1:
                if self.simplex_node_tuples[d] is not None:
                    for i in range(len(self.simplex_node_tuples[d])):
                        if 1 not in self.simplex_node_tuples[d][i]:
                            cur_index_to_node_tuple.append(self.simplex_node_tuples[d][i])
                        else:
                            cur_index_to_midpiont_tuple.append(self.simplex_node_tuples[d][i])
            else:
                if self.quad_node_tuples[d] is not None:
                    for i in range(len(self.quad_node_tuples[d])):
                        if 1 not in self.quad_node_tuples[d][i]:
                            cur_index_to_node_tuple.append(self.quad_node_tuples[d][i])
                        else:
                            cur_index_to_midpiont_tuple.append(self.quad_node_tuples[d][i])
            self.index_to_node_tuple.append(cur_index_to_node_tuple)
            self.index_to_midpoint_tuple.append(cur_index_to_midpiont_tuple)

        print(self.index_to_node_tuple)
        print(self.index_to_midpoint_tuple)

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

        from meshmode.mesh import Mesh
        #return Mesh(vertices, [grp], nodal_adjacency=self.generate_nodal_adjacency(len(self.last_mesh.groups[group].vertex_indices) \
        #            + count*3))
        groups = []
        grpn = 0
        for grp in self.last_mesh.groups:
            groups.append(np.empty([len(grp.vertex_indices),
                len(self.last_mesh.groups[grpn].vertex_indices[0])], dtype=np.int32))
            for iel_grp in range(grp.nelements):
                for i in range(0, len(grp.vertex_indices[iel_grp])):
                    groups[grpn][iel_grp][i] = grp.vertex_indices[iel_grp][i]
            grpn += 1
        grp = []

        for grpn in range(0, len(groups)):
            grp.append(make_group_from_vertices(self.last_mesh.vertices, groups[grpn], 4))
        self.last_mesh = Mesh(
                self.last_mesh.vertices, grp,
                nodal_adjacency=self.generate_nodal_adjacency(
                    len(self.last_mesh.groups[0].vertex_indices),
                    len(self.last_mesh.vertices[0])),
                vertex_id_dtype=self.last_mesh.vertex_id_dtype,
                element_id_dtype=self.last_mesh.element_id_dtype)

        return self.last_mesh


    # }}}

    # {{{ refinement

    def nelements_after_refining(self, group_index, iel_grp):
        grp = self.last_mesh.groups[group_index]
        element_vertices = grp.vertex_indices[iel_grp]
        dimension = len(self.last_mesh.vertices)
        #triangle
        if len(element_vertices) == 3 and dimension == 2:
            return (len(self.simplex_result[dimension]))
        #tet
        if len(element_vertices) == 4 and dimension == 3:
            return (len(self.simplex_result[dimension]))
        #square:
        if len(element_vertices) == 4 and dimension == 2:
            return len(self.quad_result[dimension])

    def midpoint_of_node_tuples(self, tupa, tupb):
        from six.moves import range
        assert(len(tupa) == len(tupb))
        res = ()
        for k in range(len(tupa)):
            res = res + ((tupa[k] + tupb[k])/2,)
        return res

    def simplex_node_tuple_to_vertex_index(self, vertices, dimension):
        from six.moves import range
        node_tuple_to_vertex_index = {}
        for i in range(len(vertices)):
            node_tuple_to_vertex_index[self.index_to_node_tuple[dimension][i]] = vertices[i]
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                index_i = vertices[i]
                index_j = vertices[j]
                indices_pair = (index_i, index_j) if index_i < index_j else (index_j, index_i)
                if indices_pair in self.pair_map: 
                    midpoint_tuple = self.midpoint_of_node_tuples(self.index_to_node_tuple[dimension][i], 
                            self.index_to_node_tuple[dimension][j])
                    node_tuple_to_vertex_index[midpoint_tuple] = self.pair_map[indices_pair]
        return node_tuple_to_vertex_index

    def simplex_next_vertices_and_dimension(self, vertices, dimension):
        from six.moves import range
        from itertools import combinations
        node_tuple_to_vertex_index = self.simplex_node_tuple_to_vertex_index(vertices, dimension)
        #can tesselate current element
        if len(node_tuple_to_vertex_index) == len(self.simplex_node_tuples[dimension]):
            result_vertices = []
            for subelement in self.simplex_result[dimension]:
                current_subelement_vertices = []
                for index in subelement:
                    current_subelement_vertices.append(node_tuple_to_vertex_index[self.simplex_node_tuples[dimension][index]])
                result_vertices.append(current_subelement_vertices)
            return (result_vertices, dimension)
        #move to lower dimension
        result_vertices = list(combinations(vertices, len(vertices)-1))
        return (result_vertices, dimension-1)

    def simplex_remove_from_adjacent_set(self, element_index, vertices, dimension):
        if len(vertices) == 1:
            if element_index in self.adjacent_set[vertices[0]]:
                self.adjacent_set[vertices[0]].remove(element_index)
        else:
            (next_vertices, next_dimension) = self.simplex_next_vertices_and_dimension(vertices, dimension)
            for cur_vertices in next_vertices:
                self.simplex_remove_from_adjacent_set(element_index, cur_vertices, next_dimension)

    def simplex_add_to_adjacent_set(self, element_index, vertices, dimension):
        if len(vertices) == 1:
            if element_index not in self.adjacent_set[vertices[0]]:
                self.adjacent_set[vertices[0]].add(element_index)
        else:
            (next_vertices, next_dimension) = self.simplex_next_vertices_and_dimension(vertices, dimension)
            for cur_vertices in next_vertices:
                self.simplex_add_to_adjacent_set(element_index, cur_vertices, next_dimension)

    #creates midpoints in result_vertices and updates adjacent_set with midpoint vertices
    def simplex_create_midpoints(self, element_vertices, nvertices):
        from six.moves import range
        for i in range(len(element_vertices)):
            for j in range(i+1, len(element_vertices)):
                index_i = element_vertices[i]
                index_j = element_vertices[j]
                indices_pair = (index_i, index_j) if index_i < index_j else (index_j, index_i)
                if indices_pair not in self.pair_map:
                    self.pair_map[indices_pair] = nvertices
                    for k in range(len(self.last_mesh.vertices)):
                        self.vertices[k, nvertices] = \
                        (self.last_mesh.vertices[k, index_i] +
                        self.last_mesh.vertices[k, index_j]) / 2.0
                    #update adjacent_set
                    self.adjacent_set.append(self.adjacent_set[index_i].intersection(self.adjacent_set[index_j]))
                    nvertices += 1
        return nvertices

    #returns element indices and vertices
    def simplex_create_elements(self, element_index, element_vertices, dimension, group_index, nelements_in_grp, element_mapping):
        result = []
        node_tuple_to_vertex_index = self.simplex_node_tuple_to_vertex_index(element_vertices, dimension)
        for subelement_index, subelement in enumerate(self.simplex_result[dimension]):
            if subelement_index == 0:
                result_vertices = []
                for i, index in enumerate(subelement):
                    cur_node_tuple = self.simplex_node_tuples[dimension][index]
                    self.groups[group_index][element_index][i] = node_tuple_to_vertex_index[cur_node_tuple]
                    result_vertices.append(node_tuple_to_vertex_index[cur_node_tuple])
                element_mapping[-1].append(element_index)
                result.append((element_index, result_vertices))
            else:
                result_vertices = []
                for i, index in enumerate(subelement):
                    cur_node_tuple = self.simplex_node_tuples[dimension][index]
                    self.groups[group_index][nelements_in_grp][i] = node_tuple_to_vertex_index[cur_node_tuple]
                    result_vertices.append(node_tuple_to_vertex_index[cur_node_tuple])
                element_mapping[-1].append(nelements_in_grp)
                result.append((nelements_in_grp, result_vertices))
                nelements_in_grp += 1
        return (result, nelements_in_grp, element_mapping)

    def simplex_refinement(self, group_index, iel_grp, nelements_in_grp, nvertices, element_mapping):
        from six.moves import range
        grp = self.last_mesh.groups[group_index]
        iel_base = grp.element_nr_base
        element_vertices = grp.vertex_indices[iel_grp]
        dimension = len(self.last_mesh.vertices)
        self.simplex_remove_from_adjacent_set(iel_base + iel_grp, element_vertices, dimension)
        nvertices = self.simplex_create_midpoints(element_vertices, nvertices)
        element_mapping.append([])
        (subelement_indices_and_vertices, nelements_in_grp, element_mapping) = self.simplex_create_elements(iel_grp, element_vertices, dimension, group_index, nelements_in_grp, element_mapping)
        for (index, vertices) in subelement_indices_and_vertices:
            self.simplex_add_to_adjacent_set(index, vertices, dimension)
        return (nelements_in_grp, nvertices, element_mapping)

    def simplex_elements_connected_to(self, element_index, vertices, dimension):
        from six.moves import range
        result = set()
        if len(vertices) == 1:
            if element_index in self.adjacent_set[vertices[0]]:
                result = result.union(self.adjacent_set[vertices[0]])
        else:
            (next_vertices, next_dimension) = self.simplex_next_vertices_and_dimension(vertices, dimension)
            for cur_vertices in next_vertices:
                result = result.union(self.simplex_elements_connected_to(element_index, cur_vertices, next_dimension))
        return result

    def perform_refinement(self, group_index, iel_grp, nelements_in_grp, nvertices, element_mapping, tesselation):
        grp = self.last_mesh.groups[group_index]
        element_vertices = grp.vertex_indices[iel_grp]
        dimension = len(self.last_mesh.vertices)
        #triangle
        if len(element_vertices) == 3 and dimension == 2:
            tesselation = self._Tesselation(
                self.simplex_result[dimension], self.simplex_node_tuples[dimension])
            return self.simplex_refinement(group_index, iel_grp, nelements_in_grp, nvertices, element_mapping) + (tesselation,)
        #tet
        elif len(element_vertices) == 4 and dimension == 3:
            tesselation = self._Tesselation(
                self.simplex_result[dimension], self.simplex_node_tuples[dimension])
            return self.simplex_refinement(group_index, iel_grp, nelements_in_grp, nvertices, element_mapping) +  (tesselation,)

    def get_elements_connected_to(self, result_groups, group_index, iel_base, iel_grp): 
        import copy
        grp = result_groups[group_index]
        element_vertices = copy.deepcopy(grp[iel_grp])
        dimension = len(self.last_mesh.vertices)
        #triangle
        if len(element_vertices) == 3 and dimension == 2:
            return self.simplex_elements_connected_to(iel_base+iel_grp, element_vertices, dimension)
        #tet
        if len(element_vertices) == 4 and dimension == 3:
            return self.simplex_elements_connected_to(iel_base+iel_grp, element_vertices, dimension)

    def refine(self, refine_flags):
        from six.moves import range
        """
        :arg refine_flags: a :class:`numpy.ndarray` of dtype bool of length ``mesh.nelements``
            indicating which elements should be split.
        """

        #vertices and groups for next generation
        nvertices = len(self.last_mesh.vertices[0])

        self.groups = []

        midpoint_already = set()
        grpn = 0
        totalnelements = 0

        for grp_index, grp in enumerate(self.last_mesh.groups):
            iel_base = grp.element_nr_base
            nelements = 0
            for iel_grp in range(grp.nelements):
                nelements += 1
                vertex_indices = grp.vertex_indices[iel_grp]
                if refine_flags[iel_base+iel_grp]:
                    cur_dim = len(grp.vertex_indices[iel_grp])-1
                    nelements += self.nelements_after_refining(grp_index, iel_grp) - 1
                    for i in range(len(vertex_indices)):
                        for j in range(i+1, len(vertex_indices)):
                            i_index = vertex_indices[i]
                            j_index = vertex_indices[j]
                            index_tuple = (i_index, j_index) if i_index < j_index else (j_index, i_index)
                            if index_tuple not in midpoint_already and \
                                index_tuple not in self.pair_map:
                                    nvertices += 1
                                    midpoint_already.add(index_tuple)
            self.groups.append(np.empty([nelements, len(grp.vertex_indices[iel_grp])], dtype=np.int32))
            grpn += 1
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
            iel_base = grp.element_nr_base
            nelements_in_grp = grp.nelements
            for iel_grp in range(grp.nelements):
                if refine_flags[iel_base + iel_grp]:
                    (nelements_in_grp, nvertices, element_mapping, tesselation) = \
                    self.perform_refinement(grp_index, iel_grp, nelements_in_grp, nvertices, element_mapping, tesselation)
                else:
                    for i in range(len(grp.vertex_indices[iel_grp])):
                        self.groups[grp_index][iel_grp][i] = grp.vertex_indices[iel_grp][i]
                    element_mapping.append([iel_grp])
            self.group_refinement_records.append(
                self._GroupRefinementRecord(tesselation, element_mapping))

        grp = []
        for grpn in range(0, len(self.groups)):
            print (self.groups[grpn])
            grp.append(make_group_from_vertices(self.vertices, self.groups[grpn], 4))

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
