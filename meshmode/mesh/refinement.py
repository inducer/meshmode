from __future__ import division

__copyright__ = "Copyright (C) 2014 Andreas Kloeckner"

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
    .. attribute:: direction
        A :class:`bool` denoting direction in the ray,
        with *True* representing "positive" and *False*
        representing "negative".
    .. attribute:: adjacent_elements
        List containing elements indices of elements adjacent
        to this ray segment.
    """
    def __init__(self, direction = True, adjacent_elements = []):
        self.left = None
        self.right = None
        self.midpoint = None
        self.direction = direction
        self.adjacent_elements = adjacent_elements

class Refiner(object):
    def __init__(self, mesh):
        from llist import dllist, dllistnode
        from meshmode.mesh.tesselate  import tesselatetet
        self.tri_node_tuples, self.tri_result = tesselatetet()
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
        import six
        for i in six.moves.range(nvertices):
            self.hanging_vertex_element.append([])

        import six
        for grp in mesh.groups:
            iel_base = grp.element_nr_base
            for iel_grp in six.moves.range(grp.nelements):
                vert_indices = grp.vertex_indices[iel_grp]
                for i in six.moves.range(len(vert_indices)):
                    for j in six.moves.range(i+1, len(vert_indices)):
                       
                        #minimum and maximum of the two indices for storing 
                        #data in vertex_pair
                        mn_idx = min(vert_indices[i], vert_indices[j])
                        mx_idx = max(vert_indices[i], vert_indices[j])

                        vertex_pair = (mn_idx, mx_idx)
                        if vertex_pair not in self.pair_map:
                            self.pair_map[vertex_pair] = TreeRayNode()

                        (self.pair_map[vertex_pair].
                            adjacent_elements.append(iel_base+iel_grp))
        # }}}

        #generate reference tuples
        self.index_to_node_tuple = [()] * (len(vert_indices))
        for i in six.moves.range(0, len(vert_indices)-1):
            self.index_to_node_tuple[0] = self.index_to_node_tuple[0] + (0,)
        for i in six.moves.range(1, len(vert_indices)):
            for j in six.moves.range(1, len(vert_indices)):
                if i == j:
                    self.index_to_node_tuple[i] = self.index_to_node_tuple[i] + (2,)
                else:
                    self.index_to_node_tuple[i] = self.index_to_node_tuple[i] + (0,)
        self.index_to_midpoint_tuple = [()] * (int((len(vert_indices) * (len(vert_indices) - 1)) / 2))
        curind = 0
        for ind1 in six.moves.range(0, len(self.index_to_node_tuple)):
            for ind2 in six.moves.range(ind1+1, len(self.index_to_node_tuple)):
                i = self.index_to_node_tuple[ind1]
                j = self.index_to_node_tuple[ind2]
                for k in six.moves.range(0, len(vert_indices)-1):
                    cur = int((i[k] + j[k]) / 2)
                    self.index_to_midpoint_tuple[curind] = self.index_to_midpoint_tuple[curind] + (cur,)
                curind += 1
        print "LKJAFLKJASFLJASF"
        print self.index_to_node_tuple
        print self.index_to_midpoint_tuple
        #print mesh.groups[0].vertex_indices
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
        #print self.ray_vertices
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

    def get_refine_base_index(self):
        if self.last_split_elements is None:
            return 0
        else:
            return self.last_mesh.nelements - len(self.last_split_elements)

    def get_empty_refine_flags(self):
        return np.zeros(
                self.last_mesh.nelements - self.get_refine_base_index(),
                np.bool)

    def get_current_mesh(self):
        
        from meshmode.mesh import Mesh
        #return Mesh(vertices, [grp], element_connectivity=self.generate_connectivity(len(self.last_mesh.groups[group].vertex_indices) \
        #            + count*3))
        groups = [] 
        grpn = 0
        for grp in self.last_mesh.groups:
            groups.append(np.empty([len(grp.vertex_indices),
                len(self.last_mesh.groups[grpn].vertex_indices[0])], dtype=np.int32))
            for iel_grp in xrange(grp.nelements):
                for i in range(0, len(grp.vertex_indices[iel_grp])):
                    groups[grpn][iel_grp][i] = grp.vertex_indices[iel_grp][i]
            grpn += 1
        grp = []

        from meshmode.mesh.generation import make_group_from_vertices
        for grpn in range(0, len(groups)):
            grp.append(make_group_from_vertices(self.last_mesh.vertices, groups[grpn], 4))
        self.last_mesh = Mesh(self.last_mesh.vertices, grp,\
                element_connectivity=self.generate_connectivity(len(self.last_mesh.groups[0].vertex_indices),\
                    len(self.last_mesh.vertices[0])))
        
        return self.last_mesh

    #refine_flag tells you which elements to split as a numpy array of bools
    def refine(self, refine_flags):
        import six
        nvertices = self.last_mesh.nvertices
        for grp in self.last_mesh.groups:
            iel_base = grp.element_nr_base
            for iel_grp in six.moves.range(grp.nelements):
                if refine_flags[iel_base+iel_grp]:
                    vertex_indices = grp.vertex_indices[iel_grp]
                    for i in six.moves.range(len(vertex_indices)):
                        for j in six.moves.range(i+1, len(vertex_indices)):
                            min_index = min(vertex_indices[i], vertex_indices[j])
                            max_index = max(vertex_indices[i], vertex_indices[j])
                            cur_node = self.pair_map[(min_index, max_index)]
                            if cur_node.midpoint is None:
                                cur_node.midpoint = nvertices
                                cur_node.left = TreeRayNode(cur_node.direction, cur_node.adjacent_elements)
                                cur_node.right = TreeRayNode(not cur_node.direction, cur_node.adjacent_elements)
                                left_index = cur_node.left.adjacent_elements.index(iel_base+iel_grp)
                                #right_index = cur_node
                                #cur_node.left.adjacent_elements[left_index] = 
                                nvertices += 1


    def print_rays(self, ind):
        import six
        for i in six.moves.range(len(self.last_mesh.groups[0].vertex_indices[ind])):
            for j in six.moves.range(i+1, len(self.last_mesh.groups[0].vertex_indices[ind])):
                mn = min(self.last_mesh.groups[0].vertex_indices[ind][i], self.last_mesh.groups[0].vertex_indices[ind][j])
                mx = max(self.last_mesh.groups[0].vertex_indices[ind][j], self.last_mesh.groups[0].vertex_indices[ind][i])
                d = self.pair_map[(mn, mx)].d
                ray = self.pair_map[(mn, mx)].ray
                if d:
                    print "FROM:", mn, "TO:", mx, self.vertex_to_ray[mn][ray]
                else:
                    print "FROM:", mn, "TO:", mx, self.vertex_to_ray[mx][ray]

    def print_hanging_elements(self, ind):
        import six
        for i in self.last_mesh.groups[0].vertex_indices[ind]:
            print "IND:", i, self.hanging_vertex_element[i]

    def generate_connectivity(self, nelements, nvertices, groups):
        # medium-term FIXME: make this an incremental update
        # rather than build-from-scratch
        vertex_to_element = [[] for i in xrange(nvertices)]
        element_index = 0
        for grp in groups:
            for iel_grp in xrange(len(grp)):
                for ivertex in grp[iel_grp]:
                    vertex_to_element[ivertex].append(element_index)
                element_index += 1
        element_to_element = [set() for i in xrange(nelements)]
        element_index = 0
        for grp in groups:
            for iel_grp in xrange(len(grp)):
                for ivertex in grp[iel_grp]:
                    element_to_element[element_index].update(
                            vertex_to_element[ivertex])
                    if self.hanging_vertex_element[ivertex]:
                        for hanging_element in self.hanging_vertex_element[ivertex]:
                            if element_index != hanging_element:
                                element_to_element[element_index].update([hanging_element])
                                element_to_element[hanging_element].update([element_index])
                    '''
                    if self.hanging_vertex_element[ivertex] and element_index != self.hanging_vertex_element[ivertex][0]:
                        element_to_element[element_index].update([self.hanging_vertex_element[ivertex][0]])
                        element_to_element[self.hanging_vertex_element[ivertex][0]].update([element_index])
                        '''
                element_index += 1
        for iel, neighbors in enumerate(element_to_element):
            neighbors.remove(iel)
        #print self.ray_elements
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
                np.array([0] + lengths, dtype=self.last_mesh.element_id_dtype))
        from pytools import flatten
        neighbors = np.array(
                list(flatten(element_to_element)),
                dtype=self.last_mesh.element_id_dtype)

        assert neighbors_starts[-1] == len(neighbors)
        result = []
        result.append(neighbors_starts)
        result.append(neighbors)
        return result







# vim: foldmethod=marker
