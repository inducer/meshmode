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

class VertexRay:
    def __init__(self, ray, pos):
        self.ray = ray
        self.pos = pos

class _SplitFaceRecord(object):
    """
    .. attribute:: neighboring_elements
    .. attribute:: new_vertex_nr
        integer or None
    """
class Adj:
    """One vertex-associated entry of a ray.
    .. attribute:: elements
        A list of numbers of elements adjacent to
        the edge following :attr:`vertex`
        along the ray.
    """
    def __init__(self, vertex=None, elements=[-1, -1]):
        self.vertex = vertex
        self.elements = elements
    def __str__(self):
        return 'vertex: ' + str(self.vertex) + ' ' + 'elements: ' +\
        str(self.elements)
#map pair of vertices to ray and midpoint
class PairMap:
    """Describes a segment of a ray between two vertices.
    .. attribute:: ray
        Index of the ray in the *rays* list.
    .. attribute:: d
        A :class:`bool` denoting direction in the ray,
        with *True* representing "positive" and *False*
        representing "negative".
    .. attribute:: midpoint
        Vertex index of the midpoint of this segment.
        *None* if no midpoint has been assigned yet.
    """

    def __init__(self, ray=None, d = True, midpoint=None):
        self.ray = ray
        #direction in ray, True means that second node (bigger index
        #) is after first in ray
        self.d = d
        self.midpoint = midpoint
    def __str__(self):
        return 'ray: ' + str(self.ray) + ' d: ' + str(self.d) + \
                ' midpoint: ' + str(self.midpoint)

'''
class ElementRefinementTemplate:
    def __init__(self, dim):
        from meshmode.mesh.tesselate import tesselatetri
        self.node_tuples, self.refined_elements = tesselatetri()
        print self.node_tuples
        #dicionary that maps a pair of vertices (node arrays) to a special element
        self.special_elements = {}
        self.vertices_to_midpoint = {}
                
        for i in self.refined_elements:
            has_two = False
            has_one = False 
            for j in i:
                for k in self.node_tuples[j]:
                    if k == 1:
                        has_one = True
                    if k == 2:
                        has_two = True
            #found special element
            if has_one and not has_two:
                for j in i:
                    has_two = False
                    has_one = False
                    special_elements 
'''

class Refiner(object):
    def __init__(self, mesh):
        from llist import dllist, dllistnode
        from meshmode.mesh.tesselate  import tesselatetet
        self.tri_node_tuples, self.tri_result = tesselatetet()
        print self.tri_node_tuples
        print self.tri_result
        self.last_mesh = mesh
        # Indices in last_mesh that were split in the last round of
        # refinement. Only these elements may be split this time
        # around.
        self.last_split_elements = None

        
        # {{{ initialization

        # a list of dllist instances containing Adj objects
        self.rays = []

        # mapping: (vertex_1, vertex_2) -> PairMap
        # where vertex_i represents a vertex number
        #
        # Assumption: vertex_1 < vertex_2
        self.pair_map = {}

        nvertices = len(mesh.vertices[0])
        
        # list of dictionaries, with each entry corresponding to
        # one vertex.
        # 
        # Each dictionary maps
        #   ray number -> dllist node containing a :class:`Adj`,
        #                 (part of *rays*)
        self.vertex_to_ray = []

        #np array containing element whose edge lies on corresponding vertex
        import six
        self.hanging_vertex_element = []
        for i in six.moves.range(nvertices):
            self.hanging_vertex_element.append([])
#        self.hanging_vertex_element = np.empty([nvertices], dtype=np.int32)
#        self.hanging_vertex_element.fill(-1)

        import six
        for i in six.moves.range(nvertices):
            self.vertex_to_ray.append({})
        for grp in mesh.groups:
            iel_base = grp.element_nr_base
            for iel_grp in six.moves.range(grp.nelements):
                #use six.moves.range

                vert_indices = grp.vertex_indices[iel_grp]

                for i in six.moves.range(len(vert_indices)):
                    for j in six.moves.range(i+1, len(vert_indices)):
                       
                        #minimum and maximum of the two indices for storing 
                        #data in vertex_pair
                        mn_idx = min(vert_indices[i], vert_indices[j])
                        mx_idx = max(vert_indices[i], vert_indices[j])

                        vertex_pair = (mn_idx, mx_idx)
                        if vertex_pair not in self.pair_map:
                            els = []
                            els.append(iel_base+iel_grp)
                            fr = Adj(mn_idx, els)
                            to = Adj(mx_idx, [])
                            self.rays.append(dllist([fr, to]))
                            self.pair_map[vertex_pair] = PairMap(len(self.rays) - 1)
                            self.vertex_to_ray[mn_idx][len(self.rays)-1]\
                                = self.rays[len(self.rays)-1].nodeat(0)
                            self.vertex_to_ray[mx_idx][len(self.rays)-1]\
                                = self.rays[len(self.rays)-1].nodeat(1)
                        else:
                            self.rays[self.pair_map[vertex_pair].ray].nodeat(0).value.elements.append(iel_base+iel_grp)
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
        print "RES: ", self.tri_result
        """
        :return: a refined mesh
        """

        # multi-level mapping:
        # {
        #   dimension of intersecting entity (0=vertex,1=edge,2=face,...)
        #   :
        #   { frozenset of end vertices : _SplitFaceRecord }
        # }
        '''
        import numpy as np
        count = 0
        for i in refine_flags:
            if i:
                count += 1
        
        le = len(self.last_mesh.vertices[0])
        vertices = np.empty([len(self.last_mesh.vertices), len(self.last_mesh.groups[0].vertex_indices[0])
            * count + le])
        vertex_indices = np.empty([len(self.last_mesh.groups[0].vertex_indices)
            + count*3, 
            len(self.last_mesh.groups[0].vertex_indices[0])], dtype=np.int32)
        indices_it = len(self.last_mesh.groups[0].vertex_indices)        
        for i in range(0, len(self.last_mesh.vertices)):
            for j in range(0, len(self.last_mesh.vertices[i])):
                vertices[i][j] = self.last_mesh.vertices[i][j]
        for i in range(0, len(self.last_mesh.groups[0].vertex_indices)):
            for j in range(0, len(self.last_mesh.groups[0].vertex_indices[i])):
                vertex_indices[i][j] = self.last_mesh.groups[0].vertex_indices[i][j]
        
        import itertools
        for i in range(0, len(refine_flags)):
            if refine_flags[i]:
                for subset in itertools.combinations(self.last_mesh.groups[0].vertex_indices[i], 
                    len(self.last_mesh.groups[0].vertex_indices[i]) - 1):
                    for j in range(0, len(self.last_mesh.vertices)):
                        avg = 0
                        for k in subset:
                            avg += self.last_mesh.vertices[j][k]
                        avg /= len(self.last_mesh.vertices)
                        self.last_mesh.vertices[j][le] = avg
                        le++
                vertex_indices[indices_it][0] = self.last_mesh.groups[0].vertex_indices[i][0]
        '''
        '''
        import numpy as np
        count = 0
        for i in refine_flags:
            if i:
                count += 1
        #print count
        #print self.last_mesh.vertices
        #print vertices
        if(len(self.last_mesh.groups[0].vertex_indices[0]) == 3):
            for group in range(0, len(self.last_mesh.groups)):
                le = len(self.last_mesh.vertices[0])
                vertices = np.empty([len(self.last_mesh.vertices), 
                    len(self.last_mesh.groups[group].vertex_indices[0])
                    * count + le])
                vertex_indices = np.empty([len(self.last_mesh.groups[group].vertex_indices)
                    + count*3, 
                    len(self.last_mesh.groups[group].vertex_indices[0])], dtype=np.int32)
                indices_i = 0        
                for i in range(0, len(self.last_mesh.vertices)):
                    for j in range(0, len(self.last_mesh.vertices[i])):
                        vertices[i][j] = self.last_mesh.vertices[i][j]
                #for i in range(0, len(self.last_mesh.groups[group].vertex_indices)):
                    #for j in range(0, len(self.last_mesh.groups[group].vertex_indices[i])):
                        #vertex_indices[i][j] = self.last_mesh.groups[group].vertex_indices[i][j]
                for fl in range(0, len(refine_flags)):
                    if(refine_flags[fl]):
                        i = self.last_mesh.groups[group].vertex_indices[fl]
                        for j in range(0, len(i)):
                            for k in range(j + 1, len(i)):
                                for l in range(0, 3):
                                    #print self.last_mesh.vertices[l][i[j]], ' ', self.last_mesh.vertices[l][i[k]], '=', ((self.last_mesh.vertices[l][i[j]] + self.last_mesh.vertices[l][i[k]]) / 2)
                                    vertices[l][le]=((self.last_mesh.vertices[l][i[j]] + self.last_mesh.vertices[l][i[k]]) / 2)
                                le += 1
                        vertex_indices[indices_i][0] = i[0]
                        vertex_indices[indices_i][1] = le-3
                        vertex_indices[indices_i][2] = le-2
                        indices_i += 1
                        vertex_indices[indices_i][0] = i[1]
                        vertex_indices[indices_i][1] = le-1
                        vertex_indices[indices_i][2] = le-3
                        indices_i += 1
                        vertex_indices[indices_i][0] = i[2]
                        vertex_indices[indices_i][1] = le-2
                        vertex_indices[indices_i][2] = le-1
                        indices_i += 1
                        vertex_indices[indices_i][0] = le-3
                        vertex_indices[indices_i][1] = le-2
                        vertex_indices[indices_i][2] = le-1
                        indices_i += 1
                    else:
                        for j in range(0, len(self.last_mesh.groups[group].vertex_indices[fl])):
                            vertex_indices[indices_i][j] = self.last_mesh.groups[group].vertex_indices[fl][j]
                        indices_i += 1
        '''
        import numpy as np
        import six
        # if (isinstance(self.last_mesh.groups[0], SimplexElementGroup) and 
        #           self.last_mesh.groups[0].dim == 2):
        print 'refining'
        if(len(self.last_mesh.groups[0].vertex_indices[0]) == 3 or len(self.last_mesh.groups[0].vertex_indices[0]) == 4):
            groups = []
            midpoint_already = {}
            nelements = 0
            nvertices = len(self.last_mesh.vertices[0])
            grpn = 0
            totalnelements=0

            # {{{ create new vertices array and each group's vertex_indices
            for grp in self.last_mesh.groups:
                iel_base = grp.element_nr_base
                nelements = 0
                #print grp.nelements
                for iel_grp in six.moves.range(grp.nelements):
                    nelements += 1
                    vert_indices = grp.vertex_indices[iel_grp]
                    if refine_flags[iel_base+iel_grp]:
                        nelements += len(self.tri_result) - 1
                        for i in six.moves.range(0, len(vert_indices)):
                            for j in six.moves.range(i+1, len(vert_indices)):
                                mn_idx = min(vert_indices[i], vert_indices[j])
                                mx_idx = max(vert_indices[i], vert_indices[j])
                                vertex_pair = (mn_idx, mx_idx)
                                if vertex_pair not in midpoint_already and \
                                    self.pair_map[vertex_pair].midpoint is None:
                                    nvertices += 1
                                    midpoint_already[vertex_pair] = True
                groups.append(np.empty([nelements,
                    len(grp.vertex_indices[0])], dtype=np.int32))
                grpn += 1
                totalnelements += nelements

            vertices = np.empty([len(self.last_mesh.vertices), nvertices])
            
            #create new hanging_vertex_element array
#            new_hanging_vertex_element = np.empty([nvertices], dtype=np.int32)
#            new_hanging_vertex_element.fill(-1) 
            new_hanging_vertex_element = []
            for i in range(0, nvertices):
                new_hanging_vertex_element.append([])
            # }}}

            # {{{ bring over hanging_vertex_element, vertex_indices and vertices info from previous generation

            for i in six.moves.range(0, len(self.last_mesh.vertices)):
                for j in six.moves.range(0, len(self.last_mesh.vertices[i])):
                    # always use v[i,j]
                    vertices[i,j] = self.last_mesh.vertices[i,j]
                    import copy
                    if i == 0:
                        new_hanging_vertex_element[j] = copy.deepcopy(self.hanging_vertex_element[j])
            grpn = 0
            for grp in self.last_mesh.groups:
                for iel_grp in six.moves.range(grp.nelements):
                    for i in six.moves.range(0, len(grp.vertex_indices[iel_grp])):
                        groups[grpn][iel_grp][i] = grp.vertex_indices[iel_grp][i]
                grpn += 1
            grpn = 0

            # }}}

            vertices_idx = len(self.last_mesh.vertices[0])
            for grp in self.last_mesh.groups:
                iel_base = grp.element_nr_base
                nelements_in_grp = len(grp.vertex_indices)
                
                # np.where
                for iel_grp in six.moves.range(grp.nelements):
                    if refine_flags[iel_base+iel_grp]:

                        # {{{ split element

                        # {{{ go through vertex pairs in element

                        # stores indices of all midpoints for this element
                        # (in order of vertex pairs in elements)
                        verts = []
                        midpoint_tuples = [] 
                        verts_elements = []
                        vert_indices = grp.vertex_indices[iel_grp]
                        #print vert_indices
                        for i in six.moves.range(0, len(vert_indices)):
                            for j in six.moves.range(i+1, len(vert_indices)):
                                verts_elements.append([])
                                midpoint_tuples.append(tuple([(item1 + item2) / 2 for item1, item2 in zip(self.index_to_node_tuple[i],
                                    self.index_to_node_tuple[j])]))
                                if vert_indices[i] < vert_indices[j]:
                                    mn_idx = vert_indices[i]
                                    mx_idx = vert_indices[j] 
                                    imn_idx = i
                                    imx_idx = j
                                else:
                                    mn_idx = vert_indices[j]
                                    mx_idx = vert_indices[i]
                                    imn_idx = j
                                    imx_idx = i

                                vertex_pair = (mn_idx, mx_idx)
                                
                                # {{{ create midpoint if it doesn't exist already

                                cur_pmap = self.pair_map[vertex_pair]
                                if cur_pmap.midpoint is None:
                                    self.pair_map[vertex_pair].midpoint = vertices_idx
                                    vertex_pair1 = (mn_idx, vertices_idx)
                                    vertex_pair2 = (mx_idx, vertices_idx)
                                    self.pair_map[vertex_pair1] =\
                                        PairMap(cur_pmap.ray, cur_pmap.d, None)
                                    self.pair_map[vertex_pair2] =\
                                        PairMap(cur_pmap.ray, not cur_pmap.d, None)
                                    self.vertex_to_ray.append({})
                                    
                                    # FIXME: Check where the new Adj.elements
                                    # gets populated.

                                    # try and collapse the two branches by setting up variables
                                    # ahead of time
                                    import copy
                                    if self.pair_map[vertex_pair].d:
                                        elements = self.vertex_to_ray[mn_idx][cur_pmap.ray].value.elements
                                        self.rays[cur_pmap.ray].insert(Adj(vertices_idx, copy.deepcopy(elements)),
                                            self.vertex_to_ray[mx_idx][cur_pmap.ray])

                                        #stupid bug: don't use i when already in use
                                        print "ELEMENTS: ", elements
                                        for el in elements:
                                            if el != (iel_base + iel_grp):
                                                verts_elements[len(verts_elements)-1].append(el)
                                                new_hanging_vertex_element[vertices_idx].append(el)
                                        self.vertex_to_ray[vertices_idx][cur_pmap.ray] = \
                                            self.vertex_to_ray[mx_idx][cur_pmap.ray].prev
                                        print iel_base+iel_grp, "NODE: ", self.vertex_to_ray[vertices_idx][cur_pmap.ray]        
                                    else:
                                        elements = self.vertex_to_ray[mx_idx][cur_pmap.ray].value.elements
                                        self.rays[cur_pmap.ray].insert(Adj(vertices_idx, copy.deepcopy(elements)),
                                            self.vertex_to_ray[mn_idx][cur_pmap.ray])
                                        print "ELEMENTS:", elements
                                        for el in elements:
                                            if el != (iel_base + iel_grp):
                                                verts_elements[len(verts_elements)-1].append(el)
                                                new_hanging_vertex_element[vertices_idx].append(el)

                                        self.vertex_to_ray[vertices_idx][cur_pmap.ray] = \
                                            self.vertex_to_ray[mn_idx][cur_pmap.ray].prev
                                    #print len(vert_indices)
                                    # compute location of midpoint
                                    for k in six.moves.range(len(self.last_mesh.vertices)):
                                        vertices[k,vertices_idx] =\
                                            (self.last_mesh.vertices[k,vert_indices[i]] +
                                                    self.last_mesh.vertices[k,vert_indices[j]]) / 2.0
                                    
                                    verts.append(vertices_idx)
                                    vertices_idx += 1
                                else:
                                    cur_midpoint = self.pair_map[vertex_pair].midpoint
                                    elements = self.vertex_to_ray[cur_midpoint][cur_pmap.ray].prev.value.elements
                                    print "ALREADY ELEMENTS:", elements
                                    for el in elements:
                                        if el != (iel_base + iel_grp) and el not in verts_elements[len(verts_elements)-1]:
                                            verts_elements[len(verts_elements)-1].append(el)
                                    elements = self.vertex_to_ray[cur_midpoint][cur_pmap.ray].value.elements
                                    print "ALREADY ELEMENTS:", elements
                                    for el in elements:
                                        if el != (iel_base + iel_grp) and el not in verts_elements[len(verts_elements)-1]:
                                            verts_elements[len(verts_elements)-1].append(el)

                                    for el in new_hanging_vertex_element[cur_midpoint]:
                                        if el != (iel_base + iel_grp) and el not in verts_elements[len(verts_elements)-1]:
                                            verts_elements[len(verts_elements)-1].append(el)
                                    print "HEREEEEE:", cur_midpoint, verts_elements[len(verts_elements)-1]
                                    verts.append(cur_midpoint)
                                    #new_hanging_vertex_element[cur_midpoint] = []
                                #print new_hanging_vertex_element
                                # }}}
                        
                        # }}}

                        # {{{ fix connectivity

                        # new elements will be nels+0 .. nels+2 ...
                        # (While those don't exist yet, we generate connectivity for them
                        # anyhow.)

                        unique_vertex_pairs = [
                                (i,j) for i in range(len(vert_indices)) for j in range(i+1, len(vert_indices))]
                        midpoint_ind = 0
                        for i, j in unique_vertex_pairs:
                            mn_idx = min(vert_indices[i], vert_indices[j]) 
                            mx_idx = max(vert_indices[i], vert_indices[j])
                            element_indices_1 = []
                            element_indices_2 = []
                            for k_ind, k in enumerate(self.tri_result):
                                ituple_ind = self.tri_node_tuples.index(self.index_to_node_tuple[i])
                                jtuple_ind = self.tri_node_tuples.index(self.index_to_node_tuple[j])
                                midpointtuple_ind = self.tri_node_tuples.index(self.index_to_midpoint_tuple[midpoint_ind])
                                if ituple_ind in k and\
                                    midpointtuple_ind in k:
                                        element_indices_1.append(k_ind)
                                if jtuple_ind in k and\
                                    midpointtuple_ind in k:
                                        element_indices_2.append(k_ind)
                            print "INDICES:", element_indices_1, element_indices_2
                            midpoint_ind += 1
                            #print "ELEMENTIDX1: ", element_indices_1
                            #print "ELEMENTIDX2: ", element_indices_2
                            if mn_idx == vert_indices[i]:
                                min_element_index = element_indices_1
                                max_element_index = element_indices_2
                            else:
                                min_element_index = element_indices_2
                                max_element_index = element_indices_1
                            #print "ELEMENTIDX: ", min_element_index, max_element_index
                            vertex_pair = (mn_idx, mx_idx)
                            cur_pmap = self.pair_map[vertex_pair]
                            if cur_pmap.d:
                                start_node =\
                                self.vertex_to_ray[mn_idx][cur_pmap.ray]
                                end_node = self.vertex_to_ray[mx_idx][cur_pmap.ray]
                                first_element_index = min_element_index
                                second_element_index = max_element_index
                            else:
                                start_node =\
                                self.vertex_to_ray[mx_idx][cur_pmap.ray]
                                end_node = self.vertex_to_ray[mn_idx][cur_pmap.ray]
                                first_element_index = max_element_index
                                second_element_index = min_element_index
                            midpoint_node=\
                            self.vertex_to_ray[cur_pmap.midpoint][cur_pmap.ray]
                            # hop along ray from start node to midpoint node
                            #print "Nodes: ", start_node.value, midpoint_node.value
                            while start_node != midpoint_node:
                                # replace old (big) element with new
                                # (refined) element
                                node_els = start_node.value.elements
                                #print "OLD NODE ELS: ", node_els
                                #print node_els
                                print "ORIG: ", node_els
                                node_els.remove(iel_base+iel_grp)
                                print "AFTER: ", node_els
                                for k in first_element_index:
                                    if k == 0:
                                        node_els.append(iel_base+iel_grp)
                                    else:
                                        node_els.append(iel_base+nelements_in_grp+k - 1)
                                print "RESULT: ", node_els
                                print "NODE RESULT:", start_node.value.elements
                                #print "NEW_NODE_ELS: ", node_els
                                #node_els[iold_el] = iel_base+nelements_in_grp+first_element_index
                                #print "HANGING: ", new_hanging_vertex_element[start_node.value.vertex]
                                if new_hanging_vertex_element[start_node.value.vertex] and \
                                    new_hanging_vertex_element[start_node.value.vertex].count(
                                        iel_base+iel_grp):
                                        '''
                                        to_replace_index = new_hanging_vertex_element[start_node.value.vertex].\
                                                index(iel_base+iel_grp)
                                        new_hanging_vertex_element[start_node.value.vertex][to_replace_index] =\
                                                iel_base+nelements_in_grp+first_element_index
                                        '''
                                        print "OLD HANGING VERTEX ELEMENT: ", new_hanging_vertex_element[start_node.value.vertex]
                                        new_hanging_vertex_element[start_node.value.vertex].remove(iel_base+iel_grp)
                                        for k in first_element_index:
                                            if k == 0:
                                                new_hanging_vertex_element[start_node.value.vertex].append(iel_base+iel_grp)
                                            else:
                                                new_hanging_vertex_element[start_node.value.vertex].append(iel_base+nelements_in_grp+k - 1)
                                        print "NEW HANGING VERTEX ELEMENT: ", new_hanging_vertex_element[start_node.value.vertex]
                                start_node = start_node.next
                            # hop along ray from midpoint node to end node
                            while start_node != end_node:
                                #replace old (big) element with new
                                # (refined element
                                node_els = start_node.value.elements
                                #iold_el = node_els.index(iel_base+iel_grp)
                                #node_els[iold_el] = iel_base+nelements_in_grp+second_element_index
                                print "ORIG:", node_els
                                node_els.remove(iel_base+iel_grp)
                                for k in second_element_index:
                                    if k == 0:
                                        node_els.append(iel_base+iel_grp)
                                    else:
                                        node_els.append(iel_base+nelements_in_grp+k-1)
                                print "RESULT:", node_els
                                print "HANGING EL:", new_hanging_vertex_element[start_node.value.vertex]
                                if new_hanging_vertex_element[start_node.value.vertex] and \
                                    new_hanging_vertex_element[start_node.value.vertex].count(
                                        iel_base+iel_grp):
                                        '''
                                        to_replace_index = new_hanging_vertex_element[start_node.value.vertex].\
                                                index(iel_base+iel_grp)
                                        new_hanging_vertex_element[start_node.value.vertex][to_replace_index] =\
                                                iel_base+nelements_in_grp+second_element_index
                                        '''
                                        print "OLD HANGING VERTEX ELEMENT: ", new_hanging_vertex_element[start_node.value.vertex]
                                        new_hanging_vertex_element[start_node.value.vertex].remove(iel_base+iel_grp)
                                        for k in second_element_index:
                                            if k == 0:
                                                new_hanging_vertex_element[start_node.value.vertex].append(iel_base+iel_grp)
                                            else:
                                                new_hanging_vertex_element[start_node.value.vertex].append(iel_base+nelements_in_grp+k-1)
                                        print "NEW HANGING VERTEX ELEMENT: ", new_hanging_vertex_element[start_node.value.vertex]
                                start_node = start_node.next

                        unique_vertex_pairs = [
                                (i,j) for i in range(len(verts)) for j in range(i+1, len(verts))]
                        midpoint_ind = 0
                        for i, j in unique_vertex_pairs:
                            mn_idx = min(verts[i], verts[j]) 
                            mx_idx = max(verts[i], verts[j])
                            vertex_pair = (mn_idx, mx_idx)
                            if not vertex_pair in self.pair_map:
                                continue
                            element_indices = []
                            for k_ind, k in enumerate(self.tri_result):
                                ituple_ind = self.tri_node_tuples.index(self.index_to_midpoint_tuple[i])
                                jtuple_ind = self.tri_node_tuples.index(self.index_to_midpoint_tuple[j])
                                if ituple_ind in k and\
                                    jtuple_ind in k:
                                        element_indices.append(k_ind)
                            print "MID_INDICES:", element_indices
                            #print "ELEMENTIDX1: ", element_indices_1
                            #print "ELEMENTIDX2: ", element_indices_2
                            cur_pmap = self.pair_map[vertex_pair]
                            start_node =\
                            self.vertex_to_ray[mn_idx][cur_pmap.ray]
                            end_node = self.vertex_to_ray[mx_idx][cur_pmap.ray]
                            while start_node != end_node:
                                # replace old (big) element with new
                                # (refined) element
                                node_els = start_node.value.elements
                                #print "OLD NODE ELS: ", node_els
                                #print node_els
                                print "ORIG: ", node_els
                                node_els.remove(iel_base+iel_grp)
                                for k in element_indices:
                                    if k == 0:
                                        node_els.append(iel_base+iel_grp)
                                    else:
                                        node_els.append(iel_base+nelements_in_grp+k - 1)
                                print "RESULT: ", node_els
                                print "NODEELSHERE:", start_node.value
                                #print "NEW_NODE_ELS: ", node_els
                                #node_els[iold_el] = iel_base+nelements_in_grp+first_element_index
                                #print "HANGING: ", new_hanging_vertex_element[start_node.value.vertex]
                                if new_hanging_vertex_element[start_node.value.vertex] and \
                                    new_hanging_vertex_element[start_node.value.vertex].count(
                                        iel_base+iel_grp):
                                        '''
                                        to_replace_index = new_hanging_vertex_element[start_node.value.vertex].\
                                                index(iel_base+iel_grp)
                                        new_hanging_vertex_element[start_node.value.vertex][to_replace_index] =\
                                                iel_base+nelements_in_grp+first_element_index
                                        '''
                                        print "OLD HANGING VERTEX ELEMENT: ", new_hanging_vertex_element[start_node.value.vertex]
                                        new_hanging_vertex_element[start_node.value.vertex].remove(iel_base+iel_grp)
                                        for k in element_indices:
                                            if k == 0:
                                                new_hanging_vertex_element[start_node.value.vertex].append(iel_base+iel_grp)
                                            else:
                                                new_hanging_vertex_element[start_node.value.vertex].append(iel_base+nelements_in_grp+k - 1)
                                        print "NEW HANGING VERTEX ELEMENT: ", new_hanging_vertex_element[start_node.value.vertex]
                                start_node = start_node.next
                        # }}}
                        #TODO: Update existing hanging nodes and elements for rays that may have already been generated by different element
                        #generate new rays
                        from llist import dllist, dllistnode
                        ind = 0
                        for i in six.moves.range(0, len(verts)):
                            for j in six.moves.range(i+1, len(verts)):
                                mn_vert = min(verts[i], verts[j])
                                mx_vert = max(verts[i], verts[j])
                                vertex_pair = (mn_vert, mx_vert)
                                if vertex_pair in self.pair_map:
                                    continue
                                els = []
                                common_elements = list(set(verts_elements[i]).intersection(
                                    verts_elements[j]))
                                for cel in common_elements:
                                    els.append(cel)
                                vert1ind = self.tri_node_tuples.index(self.index_to_midpoint_tuple[i])
                                vert2ind = self.tri_node_tuples.index(self.index_to_midpoint_tuple[j])
                                for kind, k in enumerate(self.tri_result):
                                    if vert1ind in k and vert2ind in k:
                                        if kind == 0:
                                            els.append(iel_base+iel_grp)
                                        else:
                                            els.append(iel_base + nelements_in_grp + kind - 1)
                                print "THIS:", els
                                #print "ELS: ", els
                                #els.append(iel_base+iel_grp)
                                #els.append(iel_base+nelements_in_grp+ind)
                                
                                fr = Adj(mn_vert, els)

                                # We're creating a new ray, and this is the end node
                                # of it.
                                to = Adj(mx_vert, [])

                                self.rays.append(dllist([fr, to]))
                                self.pair_map[vertex_pair] = PairMap(len(self.rays) - 1)
                                self.vertex_to_ray[mn_vert][len(self.rays)-1]\
                                    = self.rays[len(self.rays)-1].nodeat(0)
                                self.vertex_to_ray[mx_vert][len(self.rays)-1]\
                                    = self.rays[len(self.rays)-1].nodeat(1)
                                ind += 1
                        ind = 0
                        #map vertex indices to coordinates
                        print nvertices
                        print nelements_in_grp
                        node_tuple_to_coord = {}
                        for node_ind, node_tuple in enumerate(self.index_to_node_tuple):
                            node_tuple_to_coord[node_tuple] = grp.vertex_indices[iel_grp][node_ind]
                        for midpoint_ind, midpoint_tuple in enumerate(self.index_to_midpoint_tuple):
                            node_tuple_to_coord[midpoint_tuple] = verts[midpoint_ind]
                        o_nelements_in_grp = nelements_in_grp
                        for i in six.moves.range(0, len(self.tri_result)):
                            for j in six.moves.range(0, len(self.tri_result[i])):
                                if i == 0:
                                    groups[grpn][iel_grp][j] = \
                                            node_tuple_to_coord[self.tri_node_tuples[self.tri_result[i][j]]]
                                else:
                                    #print nelements_in_grp
                                    groups[grpn][nelements_in_grp][j] = \
                                        node_tuple_to_coord[self.tri_node_tuples[self.tri_result[i][j]]]
                            if i != 0:
                                nelements_in_grp += 1
                        print nelements
                        print "vertex_indices: ", groups[grpn][385]
                        print new_hanging_vertex_element[130]

                        '''
                        if iel_base + iel_grp == 2 or iel_base+iel_grp == 1:
                            print "LJKASFLKJASFASKL:JFASFLA:SFJ"
                            for i in six.moves.range(len(groups[grpn][385])):
                                for j in six.moves.range(i+1, len(groups[grpn][385])):
                                    mn = min(groups[grpn][385][i], groups[grpn][385][j])
                                    mx = max(groups[grpn][385][j], groups[grpn][385][i])
                                    d = self.pair_map[(mn, mx)].d
                                    ray = self.pair_map[(mn, mx)].ray
                                    if d:
                                        print self.vertex_to_ray[mn][ray]
                                        if 385 not in self.vertex_to_ray[mn][ray].value.elements:
                                            self.vertex_to_ray[mn][ray].value.elements.append(385)
                                            print "FIXED:", self.vertex_to_ray[mn][ray]
                                    else:
                                        print self.vertex_to_ray[mx][ray]
                                        if 385 not in self.vertex_to_ray[mx][ray].value.elements:
                                            self.vertex_to_ray[mx][ray].value.elements.append(385)
                                            print "FIXED:", self.vertex_to_ray[mx][ray]
                        '''
                        '''
                        print "REPAIRING:", iel_base+iel_grp
                        for i in six.moves.range(len(groups[grpn][iel_base+iel_grp])):
                            for j in six.moves.range(i+1, len(groups[grpn][iel_base+iel_grp])):
                                mn = min(groups[grpn][iel_base+iel_grp][i], groups[grpn][iel_base+iel_grp][j])
                                mx = max(groups[grpn][iel_base + iel_grp][i], groups[grpn][iel_base+iel_grp][j])
                                d = self.pair_map[(mn, mx)].d
                                ray = self.pair_map[(mn, mx)].ray
                                if d:
                                    if iel_base+iel_grp not in self.vertex_to_ray[mn][ray].value.elements:
                                        self.vertex_to_ray[mn][ray].value.elements.append(iel_base+iel_grp)
                                else:
                                    if iel_base + iel_grp not in self.vertex_to_ray[mx][ray].value.elements:
                                        self.vertex_to_ray[mx][ray].value.elements.append(iel_base+iel_grp)
                        for elem in six.moves.range(nelements_in_grp):
                            #print "REPAIRING:", elem
                            for i in six.moves.range(len(groups[grpn][elem])):
                                for j in six.moves.range(i+1, len(groups[grpn][elem])):
                                    mn = min(groups[grpn][elem][i], groups[grpn][elem][j])
                                    mx = max(groups[grpn][elem][i], groups[grpn][elem][j])
                                    d = self.pair_map[(mn, mx)].d
                                    ray = self.pair_map[(mn, mx)].ray
                                    if d:
                                        if elem not in self.vertex_to_ray[mn][ray].value.elements:
                                            print "FAILING!!!!!"
                                            self.vertex_to_ray[mn][ray].value.elements.append(elem)
                                    else:
                                        if elem not in self.vertex_to_ray[mx][ray].value.elements:
                                            print "FAILING!!!!!"
                                            self.vertex_to_ray[mx][ray].value.elements.append(elem)
                        '''

                        #print nelements_in_grp
                        #print self.tri_node_tuples
                        #print self.tri_result
                        '''
                        #map vertex indices to coordinates
                        node_tuple_to_coord = {}
                        node_tuple_to_coord[(0, 0)] = grp.vertex_indices[iel_grp][0]
                        node_tuple_to_coord[(2, 0)] = grp.vertex_indices[iel_grp][1]
                        node_tuple_to_coord[(0, 2)] = grp.vertex_indices[iel_grp][2]
                        vertex_pair = (min(grp.vertex_indices[iel_grp][0], grp.vertex_indices[iel_grp][1]), \
                        max(grp.vertex_indices[iel_grp][0], grp.vertex_indices[iel_grp][1]))
                        node_tuple_to_coord[(1, 0)] = self.pair_map[vertex_pair].midpoint 
                        vertex_pair = (min(grp.vertex_indices[iel_grp][0], grp.vertex_indices[iel_grp][2]), \
                        max(grp.vertex_indices[iel_grp][0], grp.vertex_indices[iel_grp][2]))
                        node_tuple_to_coord[(0, 1)] = self.pair_map[vertex_pair].midpoint
                        vertex_pair = (min(grp.vertex_indices[iel_grp][1], grp.vertex_indices[iel_grp][2]), \
                        max(grp.vertex_indices[iel_grp][1], grp.vertex_indices[iel_grp][2]))
                        node_tuple_to_coord[(1, 1)] = self.pair_map[vertex_pair].midpoint
                        #generate actual elements
                        #middle element
                        for i in six.moves.range(0, len(self.tri_result[1])):
                            groups[grpn][iel_grp][i] = \
                            node_tuple_to_coord[self.tri_node_tuples[self.tri_result[1][i]]]
                        for i in six.moves.range(0, 4):
                            if i == 1:
                                continue
                            for j in six.moves.range(0, len(self.tri_result[i])):
                                groups[grpn][nelements_in_grp][j] = \
                                        node_tuple_to_coord[self.tri_node_tuples[self.tri_result[i][j]]]
                            nelements_in_grp += 1
                        '''
                        '''
                        #middle element
                        vertex_pair = (min(grp.vertex_indices[iel_grp][0], grp.vertex_indices[iel_grp][1]), \
                        max(grp.vertex_indices[iel_grp][0], grp.vertex_indices[iel_grp][1]))
                        groups[grpn][iel_grp][0] = self.pair_map[vertex_pair].midpoint
                        vertex_pair = (min(grp.vertex_indices[iel_grp][1], grp.vertex_indices[iel_grp][2]), \
                        max(grp.vertex_indices[iel_grp][1], grp.vertex_indices[iel_grp][2]))
                        groups[grpn][iel_grp][1] = self.pair_map[vertex_pair].midpoint
                        vertex_pair = (min(grp.vertex_indices[iel_grp][0], grp.vertex_indices[iel_grp][2]), \
                        max(grp.vertex_indices[iel_grp][0], grp.vertex_indices[iel_grp][2]))
                        groups[grpn][iel_grp][2] = self.pair_map[vertex_pair].midpoint
                        #element 0
                        groups[grpn][nelements_in_grp][0] = grp.vertex_indices[iel_grp][0]
                        vertex_pair = (min(grp.vertex_indices[iel_grp][0], grp.vertex_indices[iel_grp][1]), \
                        max(grp.vertex_indices[iel_grp][0], grp.vertex_indices[iel_grp][1]))
                        groups[grpn][nelements_in_grp][1] = self.pair_map[vertex_pair].midpoint
                        vertex_pair = (min(grp.vertex_indices[iel_grp][0], grp.vertex_indices[iel_grp][2]), \
                        max(grp.vertex_indices[iel_grp][0], grp.vertex_indices[iel_grp][2]))
                        groups[grpn][nelements_in_grp][2] = self.pair_map[vertex_pair].midpoint
                        nelements_in_grp += 1
                        #element 1
                        groups[grpn][nelements_in_grp][0] = grp.vertex_indices[iel_grp][1]
                        vertex_pair = (min(grp.vertex_indices[iel_grp][1], grp.vertex_indices[iel_grp][0]), \
                        max(grp.vertex_indices[iel_grp][1], grp.vertex_indices[iel_grp][0]))
                        groups[grpn][nelements_in_grp][1] = self.pair_map[vertex_pair].midpoint
                        vertex_pair = (min(grp.vertex_indices[iel_grp][1], grp.vertex_indices[iel_grp][2]), \
                        max(grp.vertex_indices[iel_grp][1], grp.vertex_indices[iel_grp][2]))
                        groups[grpn][nelements_in_grp][2] = self.pair_map[vertex_pair].midpoint
                        nelements_in_grp += 1
                        #element 2
                        groups[grpn][nelements_in_grp][0] = grp.vertex_indices[iel_grp][2]
                        vertex_pair = (min(grp.vertex_indices[iel_grp][2], grp.vertex_indices[iel_grp][0]), \
                        max(grp.vertex_indices[iel_grp][2], grp.vertex_indices[iel_grp][0]))
                        groups[grpn][nelements_in_grp][1] = self.pair_map[vertex_pair].midpoint
                        vertex_pair = (min(grp.vertex_indices[iel_grp][2], grp.vertex_indices[iel_grp][1]), \
                        max(grp.vertex_indices[iel_grp][2], grp.vertex_indices[iel_grp][1]))
                        groups[grpn][nelements_in_grp][2] = self.pair_map[vertex_pair].midpoint
                        nelements_in_grp += 1
                        '''
                        # }}}

                grpn += 1

        self.hanging_vertex_element = new_hanging_vertex_element
        #print vertices
        #print vertex_indices
        from meshmode.mesh.generation import make_group_from_vertices
        #grp = make_group_from_vertices(vertices, vertex_indices, 4)
        grp = []
        grpn = 0
        for grpn in range(0, len(groups)):
            grp.append(make_group_from_vertices(vertices, groups[grpn], 4))

        from meshmode.mesh import Mesh
        #return Mesh(vertices, [grp], element_connectivity=self.generate_connectivity(len(self.last_mesh.groups[group].vertex_indices) \
        #            + count*3))
        
        self.last_mesh = Mesh(vertices, grp, element_connectivity=self.generate_connectivity(totalnelements, nvertices, groups))
        return self.last_mesh
        split_faces = {}

        ibase = self.get_refine_base_index()
        affected_group_indices = set()

        for grp in self.last_mesh.groups:
            iel_base



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
# vim: shiftwidth=4
# vim: softtabstop=4
