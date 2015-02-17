from __future__ import division

import numpy as np  # noqa
import pyopencl as cl
import random
import os
order = 4


def main():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    from meshmode.mesh.generation import (  # noqa
            generate_icosphere, generate_icosahedron,
            generate_torus)
    #mesh = generate_icosphere(1, order=order)
    #mesh = generate_icosahedron(1, order=order)
    mesh = generate_torus(3, 1, order=order)
    from meshmode.mesh.refinement import Refiner
    r = Refiner(mesh)
    #mesh = r.refine(0)
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            PolynomialWarpAndBlendGroupFactory

    discr = Discretization(
            cl_ctx, mesh, PolynomialWarpAndBlendGroupFactory(order))

    from meshmode.discretization.visualization import make_visualizer
    vis = make_visualizer(queue, discr, order=1)
    os.remove("geometry.vtu")
    os.remove("connectivity.vtu")
    vis.write_vtk_file("geometry.vtu", [
        ("f", discr.nodes()[0]),
        ])

    from meshmode.discretization.visualization import \
            write_mesh_connectivity_vtk_file

    write_mesh_connectivity_vtk_file("connectivity.vtu",
            mesh)


def main2():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    from meshmode.mesh.generation import (  # noqa
            generate_icosphere, generate_icosahedron,
            generate_torus)
    #mesh = generate_icosphere(1, order=order)
    #mesh = generate_icosahedron(1, order=order)
    mesh = generate_torus(3, 1, order=order)
    from meshmode.mesh.refinement import Refiner
    r = Refiner(mesh)
    
    times = random.randint(1, 1)
    for time in xrange(times):
        flags = np.zeros(len(mesh.groups[0].vertex_indices))
        for i in xrange(0, len(flags)):
            flags[i] = random.randint(0, 1)
        mesh = r.refine(flags)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            PolynomialWarpAndBlendGroupFactory

    discr = Discretization(
            cl_ctx, mesh, PolynomialWarpAndBlendGroupFactory(order))

    from meshmode.discretization.visualization import make_visualizer
    vis = make_visualizer(queue, discr, order)
    os.remove("geometry2.vtu")
    os.remove("connectivity2.vtu")
    vis.write_vtk_file("geometry2.vtu", [
        ("f", discr.nodes()[0]),
        ])

    from meshmode.discretization.visualization import \
            write_mesh_connectivity_vtk_file

    write_mesh_connectivity_vtk_file("connectivity2.vtu",
            mesh)

if __name__ == "__main__":
    main()
    main2()
