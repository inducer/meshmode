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
        from meshmode.mesh.refinement.resampler import SimplexResampler

        # Generate dimension-parameterized tesselations for refinement
        def generate_tesselations():
            from meshmode.mesh.tesselate import (
                    tesselatetet, tesselatetri, 
                    tesselatesquare, tesselatesegment, 
                    tesselatepoint, tesselatecube)

            # Generate lists of tuples corresponding to original vertices and
            # corresponding to midpoints for a given list of tesselations
            def generate_vertex_and_midpoint_tuples_for_tesselation(tesselations):
                vertex_and_midpoint_tuples = []
                for (node_tuples, _) in tesselations:
                    vertex_tuples = [node_tuple for node_tuple in node_tuples
                            if 1 not in node_tuple]
                    midpoint_tuples = [node_tuple for node_tuple in node_tuples
                            if 1 in node_tuple]
                    vertex_and_midpoint_tuples.append((vertex_tuples, midpoint_tuples))
                return vertex_and_midpoint_tuples

            self.simplex_tesselations = [
                    tesselatepoint(), tesselatesegment(), 
                    tesselatetri(),tesselatetet()]
            self.quad_tesselations = [
                    tesselatepoint(), tesselatesegment(),
                    tesselatesquare(), tesselatecube()]

            self.simplex_vertex_and_midpoint_tuples = generate_vertex_and_midpoint_tuples_for_tesselation(
                    self.simplex_tesselations)
            self.quad_vertex_and_midpoint_tuples = generate_vertex_and_midpoint_tuples_for_tesselation(
                    self.quad_tesselations)

        # Initialize adjacent_set to list of sets, one set for each vertex
        # so that each set contains all the elements that share that vertex
        def initialize_adjacent_set():
            self.adjacent_set = [set() for _ in range(mesh.nvertices)]
            for grp_index, grp in enumerate(mesh.groups):
                for iel_grp in range(grp.nelements):
                    vertex_indices = grp.vertex_indices[iel_grp]
                    for vertex in vertex_indices:
                        self.adjacent_set[vertex].add((grp_index, iel_grp))

        def initialize_vertex_to_generation_and_element():
            self.vertex_to_generation_and_element = [set() for _ in range(mesh.nvertices)]
            for grp_index, grp in enumerate(mesh.groups):
                for iel_grp in range(grp.nelements):
                    vertex_indices = grp.vertex_indices[iel_grp]
                    for vertex in vertex_indices:
                        self.vertex_to_generation_and_element[vertex].add((self.generation, grp_index, iel_grp))
        
        def initialize_generation_and_element_to_vertices():
            import copy
            self.generation_and_element_to_vertices = {}
            for grp_index, grp in enumerate(mesh.groups):
                for iel_grp in range(grp.nelements):
                    vertex_indices = grp.vertex_indices[iel_grp]
                    generation_and_element = (self.generation, grp_index, iel_grp)
                    self.generation_and_element_to_vertices[generation_and_element] = copy.deepcopy(vertex_indices)

        generate_tesselations()
        self.simplex_resampler = SimplexResampler()
        self.vertex_pair_to_midpoint = {}
        self.groups = []
        self.group_refinement_records = []

        self.last_mesh = mesh
        initialize_adjacent_set()
        self.generation = 0
        initialize_vertex_to_generation_and_element()
        initialize_generation_and_element_to_vertices()
        self.generation_templates = []
        self.generation_and_element_to_template_index = {}

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

    def nelements_after_refining_element_in_grp(self, grp):
        from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
        #simplex
        if isinstance(grp, SimplexElementGroup):
            return (len(self.simplex_tesselations[grp.dim][1]))
        #quad
        elif isinstance(grp, TensorProductElementGroup):
            return len(self.quad_tesselations[grp.dim][1])

    def midpoint_of_node_tuples(self, tupa, tupb):
        from six.moves import range
        assert(len(tupa) == len(tupb))
        res = ()
        for k in range(len(tupa)):
            res = res + ((tupa[k] + tupb[k])/2,)
        return res

    def node_tuple_to_vertex_index(self, vertices, node_tuples):
        from six.moves import range
        vertex_tuples = [node_tuple for node_tuple in node_tuples if 1 not in node_tuple]
        node_tuple_to_vertex_index = {}
        for i in range(len(vertices)):
            node_tuple_to_vertex_index[vertex_tuples[i]] = vertices[i]
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                vertex_i = vertices[i]
                vertex_j = vertices[j]
                vertex_pair = (vertex_i, vertex_j) if vertex_i < vertex_j else (vertex_j, vertex_i)
                if vertex_pair in self.vertex_pair_to_midpoint: 
                    midpoint_tuple = self.midpoint_of_node_tuples(vertex_tuples[i], 
                            vertex_tuples[j])
                    if midpoint_tuple in node_tuples:
                        node_tuple_to_vertex_index[midpoint_tuple] = self.vertex_pair_to_midpoint[vertex_pair]
        return node_tuple_to_vertex_index
    
    def next_vertices_and_dimension(self, vertices, dimension):
        from six.moves import range
        from itertools import combinations
        assert(len(vertices) > 1)
        def get_index(nparray, el):
            for ind, cur in enumerate(nparray):
                if cur == el:
                    return ind
            assert(False)
            return -1
        def get_tesselation(base_tesselation, vertex_indices):
            import copy
            (base_node_tuples, base_result_tuples) = base_tesselation
            base_original_node_tuples = [node_tuple for node_tuple in base_node_tuples if 1 not in node_tuple] 
            sub_node_tuples = [base_original_node_tuples[vertex_index] for vertex_index in vertex_indices]
            for i in range(len(sub_node_tuples)):
                for j in range(i+1, len(sub_node_tuples)):
                    midpoint_tuple = self.midpoint_of_node_tuples(sub_node_tuples[i], sub_node_tuples[j])
                    if midpoint_tuple in base_node_tuples and midpoint_tuple not in sub_node_tuples:
                        sub_node_tuples.append(midpoint_tuple)
            sub_result_tuples = []
            max_len_result_tuple = 0
            for result_tuple in base_result_tuples:
                has_sub_tuple = False
                for entry in result_tuple:
                    if base_node_tuples[entry] in sub_node_tuples:
                        has_sub_tuple = True
                if has_sub_tuple:
                    sub_result_tuples.append(tuple(sub_node_tuples.index(base_node_tuples[index]) for index in result_tuple if base_node_tuples[index] in sub_node_tuples))
                    max_len_result_tuple = max(max_len_result_tuple, len(sub_result_tuples[len(sub_result_tuples) - 1]))
            sub_result_tuples = [result_tuple for result_tuple in sub_result_tuples if len(result_tuple) == max_len_result_tuple]
            return (sub_node_tuples, sub_result_tuples)

        def generate_next_dimension_result():
            #FIXME: remove hardcoding
            if dimension == 3 and len(vertices) == 8:
                result_vertices = list(combinations(vertices, 4))
            #tet
            elif dimension == 3 and len(vertices) == 4:
                result_vertices = list(combinations(vertices, 3))
            elif dimension == 2:
                result_vertices = list(combinations(vertices, 2))
            elif dimension == 1:
                result_vertices = list(combinations(vertices, 1))
            return (result_vertices, dimension-1)

        #first element refined whose vertices were *vertices*
        intersect_set = set.intersection(*[self.vertex_to_generation_and_element[vertex] for vertex in vertices])
        intersecting_elements = sorted(list(intersect_set))
        base_tesselation_index = None 
        for intersecting_element in intersecting_elements:
            if intersecting_element in self.generation_and_element_to_template_index:
                base_tesselation_index = self.generation_and_element_to_template_index[intersecting_element]

        if base_tesselation_index == None:
            return generate_next_dimension_result()


        first_element_refined = min(intersect_set)
        (generation, _, _) = first_element_refined
        first_element_refined_vertices = self.generation_and_element_to_vertices[first_element_refined]
        vertex_indices = [get_index(first_element_refined_vertices, vertex) for vertex in vertices]

        base_tesselation = self.generation_templates[generation][base_tesselation_index]

        (node_tuples, result_tuples) = get_tesselation(base_tesselation, vertex_indices)
        node_tuple_to_vertex_index = self.node_tuple_to_vertex_index(vertices, node_tuples)
        #can tesselate current element
        if len(node_tuple_to_vertex_index) == len(node_tuples) and len(result_tuples) > 1:
            result_vertices = []
            for subelement in result_tuples:
                current_subelement_vertices = []
                for index in subelement:
                    current_subelement_vertices.append(node_tuple_to_vertex_index[node_tuples[index]])
                result_vertices.append(current_subelement_vertices)
            return (result_vertices, dimension)
        #move to lower dimension
        return generate_next_dimension_result()

    def split_rect_into_triangles(self, vertices, dimension):
        #can split
        if dimension == 2 and len(vertices) == 4:
            result_vertices = []
            node_tuples_to_index = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
            first = []
            first.append(vertices[node_tuples_to_index[(0, 0)]])
            first.append(vertices[node_tuples_to_index[(1, 0)]])
            first.append(vertices[node_tuples_to_index[(0, 1)]])
            result_vertices.append(first)
            second = []
            second.append(vertices[node_tuples_to_index[(1, 0)]])
            second.append(vertices[node_tuples_to_index[(1, 1)]])
            second.append(vertices[node_tuples_to_index[(0, 1)]])
            result_vertices.append(second)
            third = []
            third.append(vertices[node_tuples_to_index[(0, 0)]])
            third.append(vertices[node_tuples_to_index[(1, 0)]])
            third.append(vertices[node_tuples_to_index[(1, 1)]])
            result_vertices.append(third)
            fourth = []
            fourth.append(vertices[node_tuples_to_index[(0, 0)]])
            fourth.append(vertices[node_tuples_to_index[(1, 1)]])
            fourth.append(vertices[node_tuples_to_index[(0, 1)]])
            result_vertices.append(fourth)
            return (result_vertices, dimension)
        return ([], dimension)

    def remove_from_adjacent_set(self, element_index, vertices, dimension):
        if len(vertices) == 1:
            if element_index in self.adjacent_set[vertices[0]]:
                self.adjacent_set[vertices[0]].remove(element_index)
        else:
            (next_vertices, next_dimension) = self.next_vertices_and_dimension(vertices, dimension)
            for cur_vertices in next_vertices:
                self.remove_from_adjacent_set(element_index, cur_vertices, next_dimension)
            (simplex_vertices, simplex_dimension) = self.split_rect_into_triangles(vertices, dimension)
            for cur_vertices in simplex_vertices:
                self.remove_from_adjacent_set(element_index, cur_vertices, simplex_dimension)

    def add_to_adjacent_set(self, element_index, vertices, dimension):
        if len(vertices) == 1:
            if element_index not in self.adjacent_set[vertices[0]]:
                self.adjacent_set[vertices[0]].add(element_index)
        else:
            (next_vertices, next_dimension) = self.next_vertices_and_dimension(vertices, dimension)
            for cur_vertices in next_vertices:
                self.add_to_adjacent_set(element_index, cur_vertices, next_dimension)
            (simplex_vertices, simplex_dimension) = self.split_rect_into_triangles(vertices, dimension)
            for cur_vertices in simplex_vertices:
                self.add_to_adjacent_set(element_index, cur_vertices, simplex_dimension)

    #creates midpoints in result_vertices and updates adjacent_set with midpoint vertices
    def create_midpoints(self, group_index, iel_grp, element_vertices, nvertices, tesselation, midpoints, midpoint_order, vertex_and_midpoint_tuples):
        from six.moves import range
        dimension = self.last_mesh.groups[group_index].dim
        (vertex_tuples, midpoint_tuples) = vertex_and_midpoint_tuples
        midpoint_tuple_to_index = {}
        for i in range(len(element_vertices)):
            for j in range(i+1, len(element_vertices)):
                vertex_i = element_vertices[i]
                vertex_j = element_vertices[j]
                vertex_pair = (vertex_i, vertex_j) if vertex_i < vertex_j else (vertex_j, vertex_i)
                if vertex_pair in self.vertex_pair_to_midpoint:
                    midpoint_tuple = self.midpoint_of_node_tuples(vertex_tuples[i], 
                            vertex_tuples[j])
                    midpoint_tuple_to_index[midpoint_tuple] = self.vertex_pair_to_midpoint[vertex_pair]

        for i in range(len(element_vertices)):
            for j in range(i+1, len(element_vertices)):
                midpoint_tuple = self.midpoint_of_node_tuples(vertex_tuples[i], 
                        vertex_tuples[j])
                if midpoint_tuple not in midpoint_tuples:
                    continue
                vertex_i = element_vertices[i]
                vertex_j = element_vertices[j]
                vertex_pair = (vertex_i, vertex_j) if vertex_i < vertex_j else (vertex_j, vertex_i)
                if vertex_pair not in self.vertex_pair_to_midpoint:
                    if midpoint_tuple not in midpoint_tuple_to_index and (i, j) in midpoint_order\
                            and midpoint_tuple in midpoint_tuples:
                        self.vertex_pair_to_midpoint[vertex_pair] = nvertices
                        for k in range(len(self.last_mesh.vertices)):
                            self.vertices[k, nvertices] = \
                                    midpoints[iel_grp][k][midpoint_order[(i, j)]]
                        # Update adjacent_set
                        self.adjacent_set.append(self.adjacent_set[vertex_i].intersection(self.adjacent_set[vertex_j]))
                        nvertices += 1
                    else:
                        self.vertex_pair_to_midpoint[vertex_pair] = midpoint_tuple_to_index[midpoint_tuple]

                    assert len(self.adjacent_set) == nvertices

                midpoint_tuple_to_index[midpoint_tuple] = self.vertex_pair_to_midpoint[vertex_pair]
                self.adjacent_set[self.vertex_pair_to_midpoint[vertex_pair]] = self.adjacent_set[self.vertex_pair_to_midpoint[vertex_pair]].union(
                    self.adjacent_set[vertex_i].intersection(self.adjacent_set[vertex_j]))
                    
        return nvertices

    #returns element indices and vertices
    def create_elements(self, element_index, element_vertices, dimension, group_index, nelements_in_grp, tesselation):
        resulting_elements_indices = []
        (node_tuples, result_tuples) = tesselation
        result = []
        node_tuple_to_vertex_index = self.node_tuple_to_vertex_index(element_vertices, node_tuples)
        for subelement_index, subelement in enumerate(result_tuples):
            if subelement_index == 0:
                result_vertices = []
                for i, index in enumerate(subelement):
                    cur_node_tuple = node_tuples[index]
                    self.groups[group_index][element_index][i] = node_tuple_to_vertex_index[cur_node_tuple]
                    result_vertices.append(node_tuple_to_vertex_index[cur_node_tuple])
                resulting_elements_indices.append(element_index)
                result.append(((group_index, element_index), result_vertices))
            else:
                result_vertices = []
                for i, index in enumerate(subelement):
                    cur_node_tuple = node_tuples[index]
                    self.groups[group_index][nelements_in_grp][i] = node_tuple_to_vertex_index[cur_node_tuple]
                    result_vertices.append(node_tuple_to_vertex_index[cur_node_tuple])
                resulting_elements_indices.append(nelements_in_grp)
                result.append(((group_index, nelements_in_grp), result_vertices))
                nelements_in_grp += 1
        return (result, nelements_in_grp, resulting_elements_indices)

    def update_vertex_to_generation_and_element(self, subelement_indices_and_vertices, nvertices):
        cur_len = len(self.vertex_to_generation_and_element)
        for i in range(cur_len, nvertices):
            self.vertex_to_generation_and_element.append(set())
        for ((group_index, element_index), vertices) in subelement_indices_and_vertices:
            for vertex in vertices:
                self.vertex_to_generation_and_element[vertex].add((self.generation, group_index, element_index))

    def update_generation_and_element_to_vertices(self, subelement_indices_and_vertices):
        import copy
        for ((group_index, element_index), vertices) in subelement_indices_and_vertices:
            self.generation_and_element_to_vertices[(self.generation, group_index, element_index)] = copy.deepcopy(vertices)

    def refine_element(self, group_and_el_index, nelements_in_grp, nvertices, tesselation, midpoints, midpoint_order):
        from six.moves import range
        from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
        (group_index, iel_grp) = group_and_el_index
        grp = self.last_mesh.groups[group_index]
        element_vertices = grp.vertex_indices[iel_grp]
        self.remove_from_adjacent_set(group_and_el_index, element_vertices, grp.dim)
        nvertices = self.create_midpoints(group_index, iel_grp, element_vertices, nvertices, tesselation, midpoints, midpoint_order, self.get_vertex_and_midpoint_tuples(tesselation))
        (subelement_indices_and_vertices, nelements_in_grp, resulting_elements_indices) = self.create_elements(iel_grp, 
                element_vertices, grp.dim, group_index, nelements_in_grp, tesselation)
        self.update_vertex_to_generation_and_element(subelement_indices_and_vertices, nvertices)
        self.update_generation_and_element_to_vertices(subelement_indices_and_vertices)
        for (index, vertices) in subelement_indices_and_vertices:
            self.add_to_adjacent_set(index, vertices, grp.dim)
        return (nelements_in_grp, nvertices, resulting_elements_indices)

    def elements_connected_to(self, element_index, vertices, dimension):
        from six.moves import range
        result = set()
        if len(vertices) == 1:
            if element_index in self.adjacent_set[vertices[0]]:
                result = result.union(self.adjacent_set[vertices[0]])
        else:
            (next_vertices, next_dimension) = self.next_vertices_and_dimension(vertices, dimension)
            for cur_vertices in next_vertices:
                result = result.union(self.elements_connected_to(element_index, cur_vertices, next_dimension))
            (simplex_vertices, simplex_dimension) = self.split_rect_into_triangles(vertices, dimension)
            for cur_vertices in simplex_vertices:
                result = result.union(self.elements_connected_to(element_index, cur_vertices, 
                    simplex_dimension))
        return result
    
    # Get vertex and midpoint tuples for a group
    def get_vertex_and_midpoint_tuples(self, tesselation):
        from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
        (node_tuples, _) = tesselation
        vertex_tuples = [node_tuple for node_tuple in node_tuples
                if 1 not in node_tuple]
        midpoint_tuples = [node_tuple for node_tuple in node_tuples
                if 1 in node_tuple]
        return (vertex_tuples, midpoint_tuples)

    # coarsen_els[group_index][i] is a list of elements to be coarsened together, where the element indices are
    # local to the element group
    def coarsen(self, coarsen_els):
        from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
        import copy

        def check_coarsen_els_format():
            if len(coarsen_els) != len(self.last_mesh.groups):
                raise ValueError("length of coarsen_els does not match "
                        "group count of last generated mesh")

            for grp_index, grp in enumerate(self.last_mesh.groups):
                for to_coarsen in coarsen_els[grp_index]:
                    if (isinstance(grp, SimplexElementGroup) and \
                            len(to_coarsen) != len(self.simplex_tesselations[grp.dim][1]))\
                    or (isinstance(grp, TensorProductElementGroup) and \
                            len(to_coarsen) != len(self.quad_tesselations[grp.dim][1])):
                        raise ValueError("length of coarsen_els[group_index][i] does not match "
                                "number of elements resulting from a split of an element in the "
                                "group corresponding to group_index of last generated mesh")
                    for el in to_coarsen:
                        if el >= grp.nelements:
                            raise ValueError("element index in coarsen_els is greater than or "
                                        "equal to number of elements in group")

        # Generate list corresponding to each group, where each entry corresponding to an element is set to -1
        # if it shouldn't be coarsened, and a non-negative index (corresponding to each group to be coarsened together)
        # if it should be coarsened
        def generate_to_coarsen_groups():
            to_coarsen_groups = []
            for grp_index, grp in enumerate(self.last_mesh.groups):
                to_coarsen_group = np.empty([grp.nelements], dtype=np.int32)
                to_coarsen_group.fill(-1)
                for to_coarsen_index, to_coarsen in enumerate(coarsen_els[grp_index]):
                    for el in to_coarsen:
                        to_coarsen_group[el] = to_coarsen_index
                to_coarsen_groups.append(to_coarsen_group)
            return to_coarsen_groups

        def generate_empty_groups():
            groups = []
            for grp_index, grp in enumerate(self.last_mesh.groups):
                if isinstance(grp, SimplexElementGroup):
                    nelements_in_grp = grp.nelements - len(coarsen_els[grp_index]) * (len(self.simplex_tesselations[grp.dim][1]) - 1)
                elif isinstance(grp, TensorProductElementGroup):
                    nelements_in_grp = grp.nelements - len(coarsen_els[grp_index]) * (len(self.quad_tesselations[grp.dim][1]) - 1)
                groups.append(np.empty([nelements_in_grp, grp.vertex_indices.shape[-1]], dtype=np.int32))
            return groups

        # Generate set containing only the vertex indices of the vertices in the new coarsened mesh
        # and ignore vertices that will not appear in coarsened mesh
        def generate_vertex_indices_after_coarsening(to_coarsen_groups):
            vertex_indices_after_coarsening = set()
            # Add vertices of elements resulting from coarsening,
            # i.e., vertices that only appear once in the group of elements
            # to be coarsened together
            for grp_index, grp in enumerate(self.last_mesh.groups):
                for to_coarsen in coarsen_els[grp_index]:
                    times_vertex_seen = {}
                    for el in to_coarsen:
                        for vertex in grp.vertex_indices[el]:
                            if vertex not in times_vertex_seen:
                                times_vertex_seen[vertex] = 0
                            times_vertex_seen[vertex] += 1
                    for vertex in times_vertex_seen:
                        if times_vertex_seen[vertex] == 1:
                            vertex_indices_after_coarsening.add(vertex)

            # Add vertices of elements not being coarsened
            for grp_index, grp in enumerate(self.last_mesh.groups):
                for el in range(grp.nelements):
                    if to_coarsen_groups[grp_index][el] == -1:
                        for vertex in grp.vertex_indices[el]:
                            vertex_indices_after_coarsening.add(vertex)
            return vertex_indices_after_coarsening

        # Generate vertices array with vertices copied from last mesh, and
        # create a mapping from the vertex index in the last mesh to the new
        # vertex index after coarsening
        def generate_vertices_and_vertex_mapping(vertex_indices_after_coarsening):
            self.vertices = np.empty([len(self.last_mesh.vertices), 
                len(vertex_indices_after_coarsening)])
            vertex_mapping = {}
            nvertices_to_ignore = 0
            for vertex_index in range(self.last_mesh.nvertices):
                if vertex_index not in vertex_indices_after_coarsening:
                    nvertices_to_ignore += 1
                else:
                    self.vertices[:, vertex_index-nvertices_to_ignore] = self.last_mesh.vertices[:, vertex_index]
                    vertex_mapping[vertex_index] = vertex_index-nvertices_to_ignore
            return vertex_mapping
            
        check_coarsen_els_format()
        to_coarsen_groups = generate_to_coarsen_groups()
        self.groups = generate_empty_groups()
        vertex_indices_after_coarsening = generate_vertex_indices_after_coarsening(to_coarsen_groups)
        vertex_mapping = generate_vertices_and_vertex_mapping(vertex_indices_after_coarsening)
        coarsen_el_mapping = {}
        element_mapping = {}

        for grp_index, grp in enumerate(self.last_mesh.groups):

            iel_base = grp.element_nr_base
            new_iel_grp = 0
            seen_coarsen = set()

            for iel_grp in range(grp.nelements):

                coarsen_index = to_coarsen_groups[grp_index][iel_grp]

                # If element is not to be coarsened, copy it to new position (grp_index, new_iel_grp)
                if coarsen_index == -1:
                    element_mapping[(grp_index, iel_grp)] = (grp_index, new_iel_grp)
                    for vertex_index, vertex in enumerate(grp.vertex_indices[iel_grp]):
                        self.groups[grp_index][new_iel_grp][vertex_index] = vertex_mapping[vertex]
                    new_iel_grp += 1

                # If this is the first seen element of a collection of elements to be coarsened together, coarsen it
                # and put resulting element into position (grp_index, new_iel_grp)
                elif coarsen_index not in seen_coarsen:
                    seen_coarsen.add(coarsen_index)
                    coarsen_el_mapping[(grp_index, coarsen_index)] = (grp_index, new_iel_grp)

                    # Maintain vertex index local to element, and count of vertex
                    # Have to maintain order of vertices to maintain orientation
                    vertex_to_index_and_count = {}
                    cur_vertex_index = 0
                    for coarsen_el in coarsen_els[grp_index][coarsen_index]:
                        for vertex in grp.vertex_indices[coarsen_el]:
                            if vertex not in vertex_to_index_and_count:
                                vertex_to_index_and_count[vertex] = [cur_vertex_index, 0]
                                cur_vertex_index += 1
                            vertex_to_index_and_count[vertex][1] += 1
                    new_el_vertices = [(vertex_to_index_and_count[vertex][0], vertex) for
                            vertex in vertex_to_index_and_count if vertex_to_index_and_count[vertex][1] == 1]

                    # Retrieve original order to maintain orientation
                    new_el_vertices.sort()

                    # Add element to new groups
                    for index_of_vertex, (_, vertex) in enumerate(new_el_vertices):
                        self.groups[grp_index][new_iel_grp][index_of_vertex] = vertex_mapping[vertex]

                    new_iel_grp += 1

                if coarsen_index != -1:
                    element_mapping[(grp_index, iel_grp)] = coarsen_el_mapping[(grp_index, coarsen_index)]

        # Generate updated vertex_pair_to_midpoint
        new_vertex_pair_to_midpoint = {}
        for (vertex_1, vertex_2) in self.vertex_pair_to_midpoint:
            midpoint = self.vertex_pair_to_midpoint[(vertex_1, vertex_2)]

            assert (vertex_1 in vertex_mapping) and (
                    vertex_2 in vertex_mapping)

            if (midpoint in vertex_indices_after_coarsening):
                new_vertex_1 = vertex_mapping[vertex_1]
                new_vertex_2 = vertex_mapping[vertex_2]
                new_midpoint = vertex_mapping[midpoint]
                new_vertex_pair_to_midpoint[(new_vertex_1, new_vertex_2)] = new_midpoint

        self.vertex_pair_to_midpoint = new_vertex_pair_to_midpoint

        nvertices = len(vertex_indices_after_coarsening)

        # Generate updated adjacent_set
        new_adjacent_set = [set() for _ in range(nvertices)]
        for vertex, adjacent in enumerate(self.adjacent_set):
            if vertex in vertex_indices_after_coarsening:
                for grp_and_iel in adjacent:
                    new_adjacent_set[vertex_mapping[vertex]].add(element_mapping[grp_and_iel])
        self.adjacent_set = new_adjacent_set

        grp = []
        for grp_index, group in enumerate(self.groups):
            if isinstance(self.last_mesh.groups[grp_index], SimplexElementGroup):
                grp.append(make_group_from_vertices(self.vertices, group, 4, SimplexElementGroup))
            elif isinstance(self.last_mesh.groups[grp_index], TensorProductElementGroup):
                grp.append(make_group_from_vertices(self.vertices, group, 4, TensorProductElementGroup))

        from meshmode.mesh import Mesh

        self.previous_mesh = self.last_mesh
        self.last_mesh = Mesh(
                self.vertices, grp,
                nodal_adjacency=self.generate_nodal_adjacency(
                    nvertices, self.groups),
                vertex_id_dtype=self.last_mesh.vertex_id_dtype,
                element_id_dtype=self.last_mesh.element_id_dtype)
        return self.last_mesh
    
    def refine(self, refine_flags):
        from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
        refine_tesselations = []
        refine_indices = []
        for grp_index, grp in enumerate(self.last_mesh.groups):
            iel_base = grp.element_nr_base
            if isinstance(grp, SimplexElementGroup):
                refine_tesselations.append(self.simplex_tesselations[grp.dim])
            elif isinstance(grp, TensorProductElementGroup):
                refine_tesselations.append(self.quad_tesselations[grp.dim])
            for iel_grp in range(grp.nelements):
                if refine_flags[iel_base+iel_grp]:
                    refine_indices.append(grp_index)
                else:
                    refine_indices.append(None)
        return self.refine_using_templates(refine_tesselations, refine_indices)

    #refine_indices is a list where each entry is either None or an index into refine_templates
    def refine_using_templates(self, refine_templates, refine_indices):
        from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
        
        if len(refine_indices) != self.last_mesh.nelements:
            raise ValueError("length of refine_indices does not match "
                    "element count of last generated mesh")

        self.generation += 1
        #generate vertex_and_midpoint_tuples for all the templates
        templates_vertex_and_midpoint_tuples = []
        for template in refine_templates:
            templates_vertex_and_midpoint_tuples.append(self.get_vertex_and_midpoint_tuples(template))

        # Get number of vertices in next generation mesh
        def get_next_gen_nvertices():
            next_gen_nvertices = self.last_mesh.nvertices
            # Pairs of end vertices for the midpoints to be generated in the next generation
            next_gen_midpoint_ends = set()

            for grp_index, grp in enumerate(self.last_mesh.groups):
                iel_base = grp.element_nr_base
                for iel_grp in range(grp.nelements):
                    if refine_indices[iel_base+iel_grp] is not None:
                        vertex_indices = grp.vertex_indices[iel_grp]
                        (vertex_tuples, midpoint_tuples) = templates_vertex_and_midpoint_tuples[refine_indices[iel_base+iel_grp]]
                        # Maintain seen midpoint tuples for this element to avoid repitition (in case of quad meshes)
                        seen_midpoint_tuples = set()
                        for i in range(len(vertex_indices)):
                            for j in range(i+1, len(vertex_indices)):
                                vertex_i = vertex_indices[i]
                                vertex_j = vertex_indices[j]
                                vertex_pair = (vertex_i, vertex_j) if vertex_i < vertex_j else (vertex_j, vertex_i)
                                vertex_i_tuple = vertex_tuples[i]
                                vertex_j_tuple = vertex_tuples[j]
                                if vertex_pair in self.vertex_pair_to_midpoint:
                                    midpoint_tuple = self.midpoint_of_node_tuples(vertex_i_tuple, vertex_j_tuple)
                                    seen_midpoint_tuples.add(midpoint_tuple)
                                
                        # Any vertex pair that doesn't have a midpoint in the current mesh will be refined to create midpoints
                        for i in range(len(vertex_indices)):
                            for j in range(i+1, len(vertex_indices)):
                                vertex_i = vertex_indices[i]
                                vertex_j = vertex_indices[j]
                                vertex_pair = (vertex_i, vertex_j) if vertex_i < vertex_j else (vertex_j, vertex_i)
                                vertex_i_tuple = vertex_tuples[i]
                                vertex_j_tuple = vertex_tuples[j]
                                midpoint_tuple = self.midpoint_of_node_tuples(vertex_i_tuple, vertex_j_tuple)

                                if (vertex_pair not in self.vertex_pair_to_midpoint and
                                    vertex_pair not in next_gen_midpoint_ends and
                                    midpoint_tuple not in seen_midpoint_tuples and
                                    midpoint_tuple in midpoint_tuples):
                                    next_gen_nvertices += 1
                                    next_gen_midpoint_ends.add(vertex_pair)
                                    seen_midpoint_tuples.add(midpoint_tuple)

                                if vertex_pair not in next_gen_midpoint_ends and midpoint_tuple in midpoint_tuples:
                                    next_gen_midpoint_ends.add(vertex_pair)

            return next_gen_nvertices
        
        # Get number of elements in next generation of given group
        def get_next_gen_nelements(grp):
            next_gen_nelements = grp.nelements
            iel_base = grp.element_nr_base
            for iel_grp in range(grp.nelements):
                if refine_indices[iel_base+iel_grp] is not None:
                    next_gen_nelements += len(refine_templates[refine_indices[iel_base+iel_grp]][1])-1
            return next_gen_nelements

        def get_tesselation_record(grp):
            from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
            tesselation_record = None
            if isinstance(grp, SimplexElementGroup):
                tesselation_record = self._Tesselation(self.simplex_tesselations[grp.dim][1],
                        self.simplex_tesselations[grp.dim][0])
            elif isinstance(grp, TensorProductElementGroup):
                tesselation_record = self._Tesselation(self.quad_tesselations[grp.dim][1],
                        self.quad_tesselations[grp.dim][0])
            return tesselation_record
        
        def get_midpoints_and_midpoint_order(grp, iel_grp, tesselation_record, midpoints_to_find):
            midpoints = None
            midpoint_order = None
            if isinstance(grp, SimplexElementGroup):
                midpoints = self.simplex_resampler.get_midpoints(
                        grp, tesselation_record, midpoints_to_find)
                midpoint_order = self.simplex_resampler.get_vertex_pair_to_midpoint_order(grp.dim)
            elif isinstance(grp, TensorProductElementGroup):
                # FIXME: Replace with resampler
                iel_base = grp.element_nr_base
                num_midpoints = 0
                for vertex in tesselation_record.ref_vertices:
                    if 1 in vertex:
                        num_midpoints += 1
                midpoints = np.zeros((len(midpoints_to_find), len(self.last_mesh.vertices), 
                    num_midpoints))
                (vertex_tuples, midpoint_tuples) = templates_vertex_and_midpoint_tuples[refine_indices[iel_base+iel_grp]]
                midpoint_order = {}
                cur_el = 0
                for iel_grp in range(grp.nelements):
                    if refine_indices[iel_base+iel_grp] is not None:
                        cur = 0
                        midpoint_tuple_to_index = set()
                        for i in range(len(grp.vertex_indices[iel_grp])):
                            for j in range(i+1, len(grp.vertex_indices[iel_grp])):
                                midpoint_tuple = self.midpoint_of_node_tuples(
                                        vertex_tuples[i],
                                        vertex_tuples[j])
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

            return (midpoints, midpoint_order)
        
        def update_generation_templates_and_generation_and_element_to_template_index():
            import copy
            from six.moves import range
            self.generation_templates.append(copy.deepcopy(refine_templates))
            for grp in self.last_mesh.groups:
                iel_base = grp.element_nr_base
                for iel_grp in range(grp.nelements):
                    if refine_indices[iel_base+iel_grp] is not None:
                        import pdb
                        self.generation_and_element_to_template_index[(self.generation-1, iel_base, iel_grp)] = refine_indices[iel_base+iel_grp]

        # Create next generation vertices and groups arrays
        self.vertices = np.empty([len(self.last_mesh.vertices), get_next_gen_nvertices()])
        self.groups = []
        for grp in self.last_mesh.groups:
            self.groups.append(np.empty([get_next_gen_nelements(grp), len(grp.vertex_indices[0])], dtype=np.int32))

        # Copy vertices
        for i in range(len(self.last_mesh.vertices)):
            for j in range(len(self.last_mesh.vertices[i])):
                self.vertices[i, j] = self.last_mesh.vertices[i, j]

        nvertices = len(self.last_mesh.vertices[0])
        update_generation_templates_and_generation_and_element_to_template_index()
        del self.group_refinement_records[:]

        # Do actual refinement
        for grp_index, grp in enumerate(self.last_mesh.groups):
            element_mapping = []
            iel_base = grp.element_nr_base
            nelements_in_grp = grp.nelements
            midpoints_to_find = [iel_grp for iel_grp in range(grp.nelements) if
                    refine_indices[iel_base+iel_grp] is not None]
            tesselation_record = get_tesselation_record(grp)
            for iel_grp in range(grp.nelements):
                if refine_indices[iel_base+iel_grp] is not None:
                    (midpoints, midpoint_order) = get_midpoints_and_midpoint_order(grp, iel_grp, tesselation_record, midpoints_to_find)
                    tesselation = refine_templates[refine_indices[iel_base+iel_grp]]
                    vertex_and_midpoint_tuples = templates_vertex_and_midpoint_tuples[refine_indices[iel_base+iel_grp]]
                    (nelements_in_grp, nvertices, resulting_elements_indices) = \
                    self.refine_element((grp_index, iel_grp), nelements_in_grp, 
                            nvertices, tesselation, midpoints, midpoint_order)
                else:
                    for i in range(len(grp.vertex_indices[iel_grp])):
                        self.groups[grp_index][iel_grp][i] = grp.vertex_indices[iel_grp][i]
                    element_mapping.append([iel_grp])
            self.group_refinement_records.append(
                self._GroupRefinementRecord(tesselation_record, element_mapping))

        assert len(self.vertices[0]) == nvertices

        grp = []
        for refinement_record, group, prev_group in zip(
                self.group_refinement_records, self.groups, self.last_mesh.groups):
            is_quad = isinstance(prev_group, TensorProductElementGroup)
            if is_quad:
                grp.append(make_group_from_vertices(self.vertices, group, 4, TensorProductElementGroup))
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
                    nvertices, self.groups),
                vertex_id_dtype=self.last_mesh.vertex_id_dtype,
                element_id_dtype=self.last_mesh.element_id_dtype)


        return self.last_mesh
    # }}}

    # {{{ generate adjacency

    def generate_nodal_adjacency(self, nvertices, groups):
        import copy
        # medium-term FIXME: make this an incremental update
        # rather than build-from-scratch
        element_to_element = []

        iel_base_for_group = []
        cur_iel_base = 0
        for group_index, grp in enumerate(groups):
            iel_base_for_group.append(cur_iel_base);
            cur_iel_base += len(grp)

        for group_index, grp in enumerate(groups):
            for iel_grp in range(len(grp)):
                elements_connected_to = self.elements_connected_to((group_index, iel_grp), copy.deepcopy(grp[iel_grp]), self.last_mesh.groups[group_index].dim)
                global_elements_connected_to = set()
                for el in elements_connected_to:
                    global_elements_connected_to.add(iel_base_for_group[el[0]] + el[1])
                element_to_element.append(global_elements_connected_to)

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
