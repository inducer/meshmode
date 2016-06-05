from __future__ import division, print_function

from six.moves import range
import numpy as np  # noqa
import pyopencl as cl
import random
import os
order = 1


#construct vertex vertex_index
def get_vertex(mesh, vertex_index):
    vertex = np.empty([len(mesh.vertices)])
    for i_cur_dim, cur_dim_coords in enumerate(mesh.vertices):
        vertex[i_cur_dim] = cur_dim_coords[vertex_index]
    return vertex


def remove_if_exists(name):
    from errno import ENOENT
    try:
        os.remove(name)
    except OSError as e:
        if e.errno == ENOENT:
            pass
        else:
            raise


def test_refiner_connectivity(mesh):
    def group_and_iel_to_global_iel(igrp, iel):
        return mesh.groups[igrp].element_nr_base + iel

    from meshmode.mesh.tools import make_element_lookup_tree
    tree = make_element_lookup_tree(mesh)

    from meshmode.mesh.processing import find_bounding_box
    bbox_min, bbox_max = find_bounding_box(mesh)

    cnx = mesh.element_connectivity
    nvertices_per_element = len(mesh.groups[0].vertex_indices[0])
    #stores connectivity obtained from geometry using barycentric coordinates
    connected_to_element_geometry = np.zeros(
            (mesh.nelements, mesh.nelements), dtype=bool)
    #store connectivity obtained from mesh's connectivity
    connected_to_element_connectivity = np.zeros(
            (mesh.nelements, mesh.nelements), dtype=bool)

    for igrp, grp in enumerate(mesh.groups):
        for iel_grp in range(grp.nelements):
            iel_g = group_and_iel_to_global_iel(igrp, iel_grp)
            nb_starts = cnx.neighbors_starts
            for nb_iel_g in cnx.neighbors[nb_starts[iel_g]:nb_starts[iel_g+1]]:
                connected_to_element_connectivity[nb_iel_g, iel_g] = True
                connected_to_element_connectivity[iel_g, nb_iel_g] = True

            for vertex_index in grp.vertex_indices[iel_grp]:
                vertex = get_vertex(mesh, vertex_index)
                #check which elements touch this vertex
                for bounding_igrp, bounding_iel in tree.generate_matches(vertex):
                    if bounding_igrp == igrp and bounding_iel == iel_grp:
                        continue
                    bounding_grp = mesh.groups[bounding_igrp]

                    last_bounding_vertex = get_vertex(mesh,
                            bounding_grp.vertex_indices[bounding_iel][nvertices_per_element-1])
                    transformation = np.empty([len(mesh.vertices), nvertices_per_element-1])
                    vertex_transformed = vertex - last_bounding_vertex
                    for ibounding_vertex_index, bounding_vertex_index in enumerate(
                            bounding_grp.vertex_indices[bounding_iel][:nvertices_per_element-1]):
                        bounding_vertex = get_vertex(mesh, bounding_vertex_index)
                        transformation[:,ibounding_vertex_index] = bounding_vertex - last_bounding_vertex
                    barycentric_coordinates = np.linalg.solve(transformation, vertex_transformed)
                    is_connected = True
                    sum_of_coords = 0.0
                    for coord in barycentric_coordinates:
                        if coord < 0:
                            is_connected = False
                        sum_of_coords += coord
                    if sum_of_coords > 1:
                        is_connected = False
                    if is_connected:
                        connected_to_element_geometry[group_and_iel_to_global_iel(bounding_igrp, bounding_iel),
                                group_and_iel_to_global_iel(igrp, iel_grp)] = True
                        connected_to_element_geometry[group_and_iel_to_global_iel(igrp, iel_grp),
                                group_and_iel_to_global_iel(bounding_igrp, bounding_iel)] = True
    print("GEOMETRY: ")
    print(connected_to_element_geometry)
    print("CONNECTIVITY: ")
    print(connected_to_element_connectivity)
    cmpmatrix = (connected_to_element_geometry == connected_to_element_connectivity)
    print(cmpmatrix)
    print(np.where(cmpmatrix == False))  # noqa

    if not (connected_to_element_geometry == connected_to_element_connectivity).all():
        cmpmatrix = connected_to_element_connectivity == connected_to_element_geometry
        for ii, i in enumerate(cmpmatrix):
            for ij, j in enumerate(i):
                if not j:
                    print(
                            ii, ij,
                            "GEOMETRY: ",
                            connected_to_element_geometry[ii][ij],
                            "CONNECTIVITY: ",
                            connected_to_element_connectivity[ii][ij])
    assert (connected_to_element_geometry == connected_to_element_connectivity).all()


def test_refiner_connectivity_efficient(mesh):
    def group_and_iel_to_global_iel(igrp, iel):
        return mesh.groups[igrp].element_nr_base + iel

    from meshmode.mesh.tools import make_element_lookup_tree
    tree = make_element_lookup_tree(mesh)

    from meshmode.mesh.processing import find_bounding_box
    bbox_min, bbox_max = find_bounding_box(mesh)

    cnx = mesh.element_connectivity
    nvertices_per_element = len(mesh.groups[0].vertex_indices[0])
    #stores connectivity obtained from geometry using barycentric coordinates
    #connected_to_element_geometry = np.zeros([mesh.nelements, mesh.nelements], dtype=bool)
    #store connectivity obtained from mesh's connectivity
    #connected_to_element_connectivity = np.zeros([mesh.nelements, mesh.nelements], dtype=bool)
    connected_to_element_geometry = []
    connected_to_element_connectivity = []

    for i in range(mesh.nelements):
        connected_to_element_geometry.append([])
        connected_to_element_connectivity.append([])

    for igrp, grp in enumerate(mesh.groups):
        for iel_grp in range(grp.nelements):

            iel_g = group_and_iel_to_global_iel(igrp, iel_grp)
            nb_starts = cnx.neighbors_starts
            for nb_iel_g in cnx.neighbors[nb_starts[iel_g]:nb_starts[iel_g+1]]:
                if iel_g not in connected_to_element_connectivity[nb_iel_g]:
                    connected_to_element_connectivity[nb_iel_g].append(iel_g)
                if nb_iel_g not in connected_to_element_connectivity[iel_g]:
                    connected_to_element_connectivity[iel_g].append(nb_iel_g)
                #connected_to_element_connectivity[nb_iel_g, iel_g] = True
                #connected_to_element_connectivity[iel_g, nb_iel_g] = True

            for vertex_index in grp.vertex_indices[iel_grp]:
                vertex = get_vertex(mesh, vertex_index)
                #check which elements touch this vertex
                for bounding_igrp, bounding_iel in tree.generate_matches(vertex):
                    if bounding_igrp == igrp and bounding_iel == iel_grp:
                        continue
                    bounding_grp = mesh.groups[bounding_igrp]

                    last_bounding_vertex = get_vertex(mesh, \
                            bounding_grp.vertex_indices[bounding_iel][nvertices_per_element-1])
                    transformation = np.empty([len(mesh.vertices), nvertices_per_element-1])
                    vertex_transformed = vertex - last_bounding_vertex
                    for ibounding_vertex_index, bounding_vertex_index in \
                        enumerate(bounding_grp.vertex_indices[bounding_iel][:nvertices_per_element-1]):
                        bounding_vertex = get_vertex(mesh, bounding_vertex_index)
                        transformation[:,ibounding_vertex_index] = bounding_vertex - last_bounding_vertex
                    barycentric_coordinates = np.linalg.solve(transformation, vertex_transformed)
                    is_connected = True
                    sum_of_coords = 0.0
                    for coord in barycentric_coordinates:
                        if coord < 0:
                            is_connected = False
                        sum_of_coords += coord
                    if sum_of_coords > 1:
                        is_connected = False
                    if is_connected:
                        el1 = group_and_iel_to_global_iel(bounding_igrp, bounding_iel)
                        el2 = group_and_iel_to_global_iel(igrp, iel_grp)
                        if el2 not in connected_to_element_geometry[el1]:
                            connected_to_element_geometry[el1].append(el2)
                        if el1 not in connected_to_element_geometry[el2]:
                            connected_to_element_geometry[el2].append(el1)

    from collections import Counter
    for i in range(mesh.nelements):
        assert Counter(connected_to_element_connectivity[i]) \
                == Counter(connected_to_element_geometry[i])


def linear_func(vert):
    csum = 0
    for i in vert:
        csum += i
        #print csum
    return csum


def sine_func(vert):
    #print vert
    import math
    res = 1
    for i in vert:
        res *= math.sin(i*2.0*math.pi)
    #print res
    return abs(res) * 0.2 + 0.2


def quadratic_func(vert):
    csum = 0
    for i in vert:
        csum += i * i
    return csum * 1.5


def get_function_flags(mesh, function):
    from math import sqrt
    flags = np.zeros(len(mesh.groups[0].vertex_indices))
    for grp in mesh.groups:
        for iel_grp in range(grp.nelements):
            vertex_indices = grp.vertex_indices[iel_grp]
            max_edge_len = 0
            for i in range(len(vertex_indices)):
                for j in range(i+1, len(vertex_indices)):
                    edge_len = 0
                    for k in range(len(mesh.vertices)):
                        edge_len += (
                                (mesh.vertices[k, vertex_indices[i]]
                                    - mesh.vertices[k, vertex_indices[j]])
                                * (mesh.vertices[k, vertex_indices[i]]
                                    - mesh.vertices[k, vertex_indices[j]]))
                    edge_len = sqrt(edge_len)
                    max_edge_len = max(max_edge_len, edge_len)
                #print(edge_lens[0], mesh.vertices[0, vertex_indices[i]], mesh.vertices[1, vertex_indices[i]], mesh.vertices[2, vertex_indices[i]])  # noqa
                centroid = [0] * len(mesh.vertices)
                for j in range(len(mesh.vertices)):
                    centroid[j] += mesh.vertices[j, vertex_indices[i]]
            for i in range(len(mesh.vertices)):
                centroid[i] /= len(vertex_indices)
            val = function(centroid)
            if max_edge_len > val:
                flags[iel_grp] = True
    return flags


def get_corner_flags(mesh):
    flags = np.zeros(len(mesh.groups[0].vertex_indices))
    for grp in mesh.groups:
        for iel_grp in range(grp.nelements):
            is_corner_el = False
            vertex_indices = grp.vertex_indices[iel_grp]
            for i in range(len(vertex_indices)):
                cur_vertex_corner = True
                for j in range(len(mesh.vertices)):
                    print(iel_grp, i, mesh.vertices[j, vertex_indices[i]])
                    if mesh.vertices[j, vertex_indices[i]] != 0.0:
                        cur_vertex_corner = False
                if cur_vertex_corner:
                    is_corner_el = True
                    break
            if is_corner_el:
                print(iel_grp)
                flags[iel_grp] = True
    return flags


def get_random_flags(mesh):
    flags = np.zeros(len(mesh.groups[0].vertex_indices))
    for i in range(0, len(flags)):
            flags[i] = random.randint(0, 1)
    return flags


def refine_and_generate_chart_function(mesh, filename, function):
    from time import clock
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    print("NELEMENTS: ", mesh.nelements)
    #print mesh
    for i in range(len(mesh.groups[0].vertex_indices[0])):
        for k in range(len(mesh.vertices)):
            print(mesh.vertices[k, i])

    #test_refiner_connectivity(mesh);
    from meshmode.mesh.refinement import Refiner
    r = Refiner(mesh)
    #random.seed(0)
    poss_flags = np.ones(len(mesh.groups[0].vertex_indices))
    #times = 3
    num_elements = []
    time_t = []
    #nelements = mesh.nelements
    while True:
        print("NELS:", mesh.nelements)
        #flags = get_corner_flags(mesh)
        flags = get_function_flags(mesh, function)
        nels = 0
        for i in flags:
            if i:
                nels += 1
        if nels == 0:
            break
        print("LKJASLFKJALKASF:", nels)
        num_elements.append(nels)
        #flags = get_corner_flags(mesh)
        beg = clock()
        mesh = r.refine(flags)
        end = clock()
        time_taken = end - beg
        time_t.append(time_taken)
        #if nelements == mesh.nelements:
            #break
        #nelements = mesh.nelements
        #from meshmode.mesh.visualization import draw_2d_mesh
        #draw_2d_mesh(mesh, True, True, True, fill=None)
        #import matplotlib.pyplot as pt
        #pt.show()

        #poss_flags = np.zeros(len(mesh.groups[0].vertex_indices))
        #for i in range(0, len(flags)):
        #    poss_flags[i] = flags[i]
        #for i in range(len(flags), len(poss_flags)):
        #    poss_flags[i] = 1

    import matplotlib.pyplot as pt
    pt.xlabel('Number of elements being refined')
    pt.ylabel('Time taken')
    pt.plot(num_elements, time_t, "o")
    pt.savefig(filename, format='pdf')
    pt.clf()
    print('DONE REFINING')
    '''
    flags = np.zeros(len(mesh.groups[0].vertex_indices))
    flags[0] = 1
    flags[1] = 1
    mesh = r.refine(flags)
    flags = np.zeros(len(mesh.groups[0].vertex_indices))
    flags[0] = 1
    flags[1] = 1
    flags[2] = 1
    mesh = r.refine(flags)
    '''
    #test_refiner_connectivity_efficient(mesh)
    #r.print_rays(70)
    #r.print_rays(117)
    #r.print_hanging_elements(10)
    #r.print_hanging_elements(117)
    #r.print_hanging_elements(757)
    #from meshmode.mesh.visualization import draw_2d_mesh
    #draw_2d_mesh(mesh, False, False, False, fill=None)
    #import matplotlib.pyplot as pt
    #pt.show()

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            PolynomialWarpAndBlendGroupFactory
    discr = Discretization(
            cl_ctx, mesh, PolynomialWarpAndBlendGroupFactory(order))
    from meshmode.discretization.visualization import make_visualizer
    vis = make_visualizer(queue, discr, order)
    remove_if_exists("connectivity2.vtu")
    remove_if_exists("geometry2.vtu")
    vis.write_vtk_file("geometry2.vtu", [
        ("f", discr.nodes()[0]),
        ])

    from meshmode.discretization.visualization import \
            write_nodal_adjacency_vtk_file

    write_nodal_adjacency_vtk_file("connectivity2.vtu",
            mesh)


def main2():
    from meshmode.mesh.generation import (  # noqa
            generate_icosphere, generate_icosahedron,
            generate_torus, generate_regular_rect_mesh,
            generate_box_mesh)
#    mesh = generate_icosphere(1, order=order)
    #mesh = generate_icosahedron(1, order=order)
    #mesh = generate_torus(3, 1, order=order)
    #mesh = generate_regular_rect_mesh()
    #mesh =  generate_box_mesh(3*(np.linspace(0, 3, 5),))
    #mesh =  generate_box_mesh(3*(np.linspace(0, 1, 3),))
    mesh = generate_box_mesh(3*(np.linspace(0, 1, 5),))
    refine_and_generate_chart_function(mesh, "plot.pdf", sine_func)


def all_refine(num_mesh, depth, fname):
    from meshmode.mesh.generation import (  # noqa
            generate_icosphere, generate_icosahedron,
            generate_torus, generate_regular_rect_mesh,
            generate_box_mesh)
    import timeit
    nelements = []
    runtimes = []
    for el_fact in range(2, num_mesh+2):
        mesh = generate_box_mesh(3*(np.linspace(0, 1, el_fact),))
        from meshmode.mesh.refinement import Refiner
        r = Refiner(mesh)
        for time in range(depth):
            flags = np.ones(len(mesh.groups[0].vertex_indices))
            if time < depth-1:
                mesh = r.refine(flags)
            else:
                start = timeit.default_timer()
                mesh = r.refine(flags)
                stop = timeit.default_timer()
                nelements.append(mesh.nelements)
                runtimes.append(stop-start)
        #test_refiner_connectivity_efficient(mesh)
    import matplotlib.pyplot as pt
    pt.plot(nelements, runtimes, "o")
    pt.savefig(fname)
    pt.clf()
    #pt.show()


def uniform_refine(num_mesh, fract, depth, fname):
    from meshmode.mesh.generation import (  # noqa
            generate_icosphere, generate_icosahedron,
            generate_torus, generate_regular_rect_mesh,
            generate_box_mesh)
    import timeit
    nelements = []
    runtimes = []
    for el_fact in range(2, num_mesh+2):
        mesh = generate_box_mesh(3*(np.linspace(0, 1, el_fact),))
        from meshmode.mesh.refinement import Refiner
        r = Refiner(mesh)
        all_els = range(mesh.nelements)
        for time in range(depth):
            flags = np.zeros(mesh.nelements)
            from random import shuffle
            shuffle(all_els)
            nels_this_round = 0
            for i in range(len(all_els)):
                if i / len(flags) > fract:
                    break
                flags[all_els[i]] = 1
                nels_this_round += 1

            if time < depth-1:
                mesh = r.refine(flags)
            else:
                start = timeit.default_timer()
                mesh = r.refine(flags)
                stop = timeit.default_timer()
                nelements.append(mesh.nelements)
                runtimes.append(stop-start)
            all_els = []
            for i in range(len(flags)):
                if flags[i]:
                    all_els.append(i)
            for i in range(len(flags), mesh.nelements):
                all_els.append(i)
        test_refiner_connectivity(mesh)
    import matplotlib.pyplot as pt
    pt.plot(nelements, runtimes, "o")
    pt.savefig(fname)
    pt.clf()


if __name__ == "__main__":
    print("HEREERERE")
    #all_refine(3, 2, 'all_a.pdf')
    #all_refine(3, 3, 'all_b.pdf')
    #uniform_refine(3, 0.2, 3, 'uniform_a.pdf')
    main2()
