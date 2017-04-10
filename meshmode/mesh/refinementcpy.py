
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
