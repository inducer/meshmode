import numpy as np  # noqa
import pyopencl as cl
from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import thaw

order = 4


def main():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    from meshmode.mesh.generation import (  # noqa
            generate_icosphere, generate_icosahedron,
            generate_torus)
    #mesh = generate_icosphere(1, order=order)
    mesh = generate_icosahedron(1, order=order)
    #mesh = generate_torus(3, 1, order=order)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            PolynomialWarpAndBlendGroupFactory

    discr = Discretization(
            actx, mesh, PolynomialWarpAndBlendGroupFactory(order))

    from meshmode.discretization.visualization import make_visualizer
    vis = make_visualizer(actx, discr, order)

    vis.write_vtk_file("geometry.vtu", [
        ("f", thaw(actx, discr.nodes()[0])),
        ])

    from meshmode.discretization.visualization import \
            write_nodal_adjacency_vtk_file

    write_nodal_adjacency_vtk_file("adjacency.vtu",
            mesh)


if __name__ == "__main__":
    main()
