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
             
        generate_tesselations()
        self.simplex_resampler = SimplexResampler()
        self.vertex_pair_to_midpoint = {}
        self.groups = []
        self.group_refinement_records = []

        self.last_mesh = mesh
        initialize_adjacent_set()


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
        #TODO: use comprehension instead
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

    def remove_from_adjacent_set(self, element_index, vertices, dimension, result_tuples, node_tuples, index_to_node_tuple):
        if len(vertices) == 1:
            if element_index in self.adjacent_set[vertices[0]]:
                self.adjacent_set[vertices[0]].remove(element_index)
        else:
            (next_vertices, next_dimension) = self.next_vertices_and_dimension(vertices, dimension, result_tuples, node_tuples, index_to_node_tuple)
            for cur_vertices in next_vertices:
                self.remove_from_adjacent_set(element_index, cur_vertices, next_dimension, result_tuples, node_tuples, index_to_node_tuple)
            (simplex_vertices, simplex_dimension) = self.split_rect_into_triangles(vertices, dimension)
            for cur_vertices in simplex_vertices:
                self.remove_from_adjacent_set(element_index, cur_vertices, simplex_dimension, self.simplex_result, self.simplex_node_tuples, self.simplex_index_to_node_tuple)

    def add_to_adjacent_set(self, element_index, vertices, dimension, result_tuples, node_tuples, index_to_node_tuple):
        if len(vertices) == 1:
            if element_index not in self.adjacent_set[vertices[0]]:
                self.adjacent_set[vertices[0]].add(element_index)
        else:
            (next_vertices, next_dimension) = self.next_vertices_and_dimension(vertices, dimension, result_tuples, node_tuples, index_to_node_tuple)
            for cur_vertices in next_vertices:
                self.add_to_adjacent_set(element_index, cur_vertices, next_dimension, result_tuples, node_tuples, index_to_node_tuple)
            (simplex_vertices, simplex_dimension) = self.split_rect_into_triangles(vertices, dimension)
            for cur_vertices in simplex_vertices:
                self.add_to_adjacent_set(element_index, cur_vertices, simplex_dimension, self.simplex_result, self.simplex_node_tuples, self.simplex_index_to_node_tuple)

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
                result.append(((group_index, element_index), result_vertices))
            else:
                result_vertices = []
                for i, index in enumerate(subelement):
                    cur_node_tuple = node_tuples[dimension][index]
                    self.groups[group_index][nelements_in_grp][i] = node_tuple_to_vertex_index[cur_node_tuple]
                    result_vertices.append(node_tuple_to_vertex_index[cur_node_tuple])
                element_mapping[-1].append(nelements_in_grp)
                result.append(((group_index, nelements_in_grp), result_vertices))
                nelements_in_grp += 1
        return (result, nelements_in_grp, element_mapping)

    def refine_element(self, group_index, iel_grp, nelements_in_grp, nvertices, element_mapping, tesselations, vertex_and_midpoint_tuples, midpoints, midpoint_order):
        from six.moves import range
        grp = self.last_mesh.groups[group_index]
        from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
        iel_base = grp.element_nr_base
        element_vertices = grp.vertex_indices[iel_grp]
        dimension = self.last_mesh.groups[group_index].dim
        self.remove_from_adjacent_set((group_index, iel_grp), element_vertices, dimension, result_tuples, node_tuples, index_to_node_tuple)
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
            (simplex_vertices, simplex_dimension) = self.split_rect_into_triangles(vertices, dimension)
            for cur_vertices in simplex_vertices:
                result = result.union(self.elements_connected_to(element_index, cur_vertices, simplex_dimension, self.simplex_result, self.simplex_node_tuples, self.simplex_index_to_node_tuple))
        return result
    
    def do_refinement(self, group_index, iel_grp, nelements_in_grp, nvertices, element_mapping,
            midpoints, midpoint_order):
        from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
        grp = self.last_mesh.groups[group_index]
        element_vertices = grp.vertex_indices[iel_grp]
        dimension = self.last_mesh.groups[group_index].dim
        return self.refine_element(group_index, iel_grp, nelements_in_grp, nvertices,
                element_mapping, self.get_tesselations(grp),
                self.get_vertex_and_midpoint_tuples(grp), midpoints, midpoint_order
                )

    def get_elements_connected_to(self, result_groups, group_index, iel_base, iel_grp): 
        import copy
        from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
        grp = result_groups[group_index]
        element_vertices = copy.deepcopy(grp[iel_grp])
        dimension = self.last_mesh.groups[group_index].dim
        #simplex
        if isinstance(self.last_mesh.groups[group_index], SimplexElementGroup):
            return self.elements_connected_to((group_index, iel_grp), element_vertices, dimension,
                    self.simplex_result, self.simplex_node_tuples, self.simplex_index_to_node_tuple)
        #quad
        elif isinstance(self.last_mesh.groups[group_index], TensorProductElementGroup):
            return self.elements_connected_to((group_index, iel_grp), element_vertices, dimension,
                    self.quad_result, self.quad_node_tuples, self.quad_index_to_node_tuple)

    # Get vertex and midpoint tuples for a group
    def get_vertex_and_midpoint_tuples(self, grp):
        from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
        #simplex
        if isinstance(grp, SimplexElementGroup):
            return self.simplex_vertex_and_midpoint_tuples
        #quad
        elif isinstance(grp, TensorProductElementGroup):
            return self.quad_vertex_and_midpoint_tuples

    def get_tesselations(self, grp):
        from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup
        tesselation = None
        if isinstance(grp, SimplexElementGroup):
            tesselation = self._Tesselation(self.simplex_tesselations[grp.dim][1],
                    self.simplex_tesselations[grp.dim][0])
        elif isinstance(grp, TensorProductElementGroup):
            tesselation = self._Tesselation(self.quad_tesselations[grp.dim][1],
                    self.quad_tesselations[grp.dim][0])
        return tesselation

#TODO: erase below if not needed
#    def update_coarsen_connectivity(self, coarsen_el, new_el_index, old_vertices, new_vertices, dimension, result, node_tuples, index_to_node_tuple):
#        from six.moves import range
#        for el_index, el in enumerate(coarsen_el):
#            self.remove_from_adjacent_set(el, old_vertices[el_index], dimension, result, node_tuples, index_to_node_tuple)
#        #remove midpoints from pair_map
#        for i in range(len(new_vertices)):
#            for j in range(i+1, len(new_vertices)):
#                vi = new_vertices[i]
#                vj = new_vertices[j]
#                indices_pair = (vi, vj) if vi < vj else (vj, vi)
#                if indices_pair in self.pair_map:
#                    del self.pair_map[indices_pair]
#        self.add_to_adjacent_set(new_el_index, new_vertices, dimension, result, node_tuples, index_to_node_tuple)
#
#    def update_not_to_coarsen_connectivity(self, old_index, new_index, old_vertices, new_vertices, dimension, result, node_tuples, index_to_node_tuple):
#        self.remove_from_adjacent_set(old_index, old_vertices, dimension, result, node_tuples, index_to_node_tuple)
#        self.add_to_adjacent_set(new_index, new_vertices, dimension, result, node_tuples, index_to_node_tuple)
#

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
                    nelements_in_grp = grp.nelements - len(coarsen_els[grp_index]) * (len(self.simplex_tesselations[grp.dim]) + 1)
                elif isinstance(grp, TensorProductElementGroup):
                    nelements_in_grp = grp.nelements - len(coarsen_els[grp_index]) * (len(self.quad_tesselations[grp.dim] + 1))
                groups.append(np.empty([nelements_in_grp, grp.vertex_indices.shape(-1)]), dtype=np.int32)
            return groups

        # Generate set containing only the vertex indices of the vertices in the new coarsened mesh
        # and ignore vertices that will not appear in coarsened mesh
        def generate_vertex_indices_after_coarsening(to_coarsen_groups):
            vertex_indices_after_coarsening = set()
            # Add vertices of elements resulting from coarsening,
            # i.e., vertices that only appear once in the group of elements
            # to be coarsened together
            for grp_index, grp in enumerate(self.last_mesh.groups):
                for to_coarsen in coarsen_group[grp_index]:
                    for el in to_coarsen:
                        times_vertex_seen = {}
                        for vertex in grp.vertex_indices[el]:
                            if vertex not in times_vertex_seen:
                                times_vertex_seen[vertex] = 0
                            times_vertex_seen[vertex] += 1
                        for vertex in times_vertex_seen:
                            if times_vertex_seen[vertex] == 1:
                                vertex_indices_after_coarsening.add(vertex)

            # Add vertices of elements not being coarsened
            for grp_index, grp in enumerate(self.last_mesh.groups):
                for el in range(len(grp.nelements)):
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
                            vertex in vertex_to_index_and_count if vertices[vertex][1] == 1]

                    # Retrieve original order to maintain orientation
                    new_el_vertices.sort()

                    # Add element to new groups
                    for index_of_vertex, (_, vertex) in enumerate(new_el_vertices):
                        self.groups[grp_index][new_iel_grp][index_of_vertex] = vertex_mapping[vertex]

                    #TODO: below is probably unnecessary
                    # Remove vertices that don't appear in new mesh from vertex_pair_to_midpoint
                    #for i in range(len(new_el_vertices)):
                    #    for j in range(i+1, len(new_el_vertices)):
                    #        vertex_i = new_el_vertices[i][1]
                    #        vertex_j = new_el_vertices[j][1]
                    #        vertex_pair = (vertex_i, vertex_j) if vertex_i < vertex_j else (vertex_j, vertex_i)

                    #        # Midpoint of resulting vertices of element after coarsening should be in vertex_pair_to_midpoint
                    #        assert(vertex_pair in self.vertex_pair_to_midpoint)

                    #        if (vertex_pair in self.vertex_pair_to_midpoint) and (
                    #                self.vertex_pair_to_midpoint[vertex_pair] in vertex_indices_after_coarsening):
                    #            del self.vertex_pair_to_midpoint[vertex_pair]

                    new_iel_grp += 1

                if coarsen_index != -1:
                    element_mapping[(grp_index, iel_grp)] = coarsen_el_mapping[(grp_index, coarsen_index)]

        # Generate updated vertex_pair_to_midpoint
        new_vertex_pair_to_midpoint = {}
        for (vertex_1, vertex_2) in self.vertex_pair_to_midpoint:
            midpoint = self.vertex_pair_to_midpoint[(vertex_1, vertex_2)]

            assert (vertex_1 in vertex_mapping) and (
                    vertex_2 in vertex_mapping) and (midpoint in vertex_mapping)

            new_vertex_1 = vertex_mapping[vertex_1]
            new_vertex_2 = vertex_mapping[vertex_2]
            new_midpoint = vertex_mapping[midpoint]

            if (midpoint in vertex_indices_after_coarsening):
                new_vertex_pair_to_midpoint[(new_vertex_1, new_vertex_2)] = new_midpoint

        self.vertex_pair_to_midpoint = new_vertex_pair_to_midpoint

        nvertices = len(vertex_indices_after_coarsening)

        # Generate updated adjacent_set
        new_adjacent_set = [set() for _ in range(nvertices)]
        for vertex, adjacent in enumerate(self.adjacent_set):
            if vertex in vertex_indices_after_coarsening:
                for (grp_index, iel_grp) in adjacent:
                    new_adjacent_set[vertex_mapping[vertex]].add((grp_index, element_mapping[el]))
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
        """
        :arg refine_flags: a :class:`numpy.ndarray` of dtype bool of length ``mesh.nelements``
            indicating which elements should be split.
        """

        if len(refine_flags) != self.last_mesh.nelements:
            raise ValueError("length of refine_flags does not match "
                    "element count of last generated mesh")

        # Get number of vertices in next generation mesh
        def get_next_gen_nvertices():
            next_gen_nvertices = self.last_mesh.nvertices
            # Pairs of end vertices for the midpoints to be generated in the next generation
            next_gen_midpoint_ends = set()

            for grp_index, grp in enumerate(self.last_mesh.groups):
                iel_base = grp.element_nr_base
                for iel_grp in range(grp.nelements):
                    if refine_flags[iel_base+iel_grp]:
                        vertex_indices = grp.vertex_indices[iel_grp]
                        vertex_tuples = self.get_vertex_and_midpoint_tuples(grp)[grp.dim][0]
                        # Maintain seen midpoint tuples for this element to avoid repitition (in case of quad meshes)
                        seen_midpoint_tuples = set()
                        
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
                                    midpoint_tuple not in seen_midpoint_tuples):
                                    next_gen_nvertices += 1
                                    next_gen_midpoint_ends.add(vertex_pair)
                                    seen_midpoint_tuples.add(midpoint_tuple)

            return next_gen_nvertices
        
        # Get number of elements in next generation of given group
        def get_next_gen_nelements(grp):
            iel_base = grp.element_nr_base
            return (grp.nelements + np.sum(refine_flags[iel_base:iel_base+grp.nelements]) *
                    (self.nelements_after_refining_element_in_grp(grp) - 1))

        
        def get_midpoints_and_midpoint_order(grp, tesselation, midpoints_to_find):
            midpoints = None
            midpoint_order = None
            if isinstance(grp, SimplexElementGroup):
                midpoints = self.simplex_resampler.get_midpoints(
                        grp, tesselation, midpoints_to_find)
                midpoint_order = self.simplex_resampler.get_vertex_pair_to_midpoint_order(grp.dim)
            elif isinstance(grp, TensorProductElementGroup):
                # TODO: Replace with resampler
                num_midpoints = 0
                for vertex in tesselation.ref_vertices:
                    if 1 in vertex:
                        num_midpoints += 1
                midpoints = np.zeros((len(midpoints_to_find), len(self.last_mesh.vertices), 
                    num_midpoints))
                vertex_tuples = self.get_vertex_and_midpoint_tuples(grp)[grp.dim][0]
                midpoint_order = {}
                cur_el = 0
                for iel_grp in range(grp.nelements):
                    if refine_flags[iel_base+iel_grp]:
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
        
        del self.group_refinement_records[:]

        # Do actual refinement
        for grp_index, grp in enumerate(self.last_mesh.groups):
            element_mapping = []
            iel_base = grp.element_nr_base
            nelements_in_grp = grp.nelements
            midpoints_to_find = [iel_grp for iel_grp in range(grp.nelements) if
                    refine_flags[iel_base+iel_grp]]
            tesselation = self.get_tesselations(grp)
            (midpoints, midpoint_order) = get_midpoints_and_midpoint_order(grp, tesselation, midpoints_to_find)
            for iel_grp in range(grp.nelements):
                if refine_flags[iel_base + iel_grp]:
                    (nelements_in_grp, nvertices, element_mapping) = \
                    self.do_refinement(grp_index, iel_grp, nelements_in_grp, 
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
                elements_connected_to = self.get_elements_connected_to(groups, group_index, iel_base_for_group[group_index], iel_grp)
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
