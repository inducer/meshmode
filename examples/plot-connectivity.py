import numpy as np  # noqa

import pyopencl as cl

from meshmode.array_context import PyOpenCLArrayContext


order = 4


def main():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue, force_device_scalars=True)

    from meshmode.mesh.generation import (  # noqa: F401
        generate_icosahedron,
        generate_sphere,
        generate_torus,
    )

    #mesh = generate_sphere(1, order=order)
    mesh = generate_icosahedron(1, order=order)
    #mesh = generate_torus(3, 1, order=order)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import (
        PolynomialWarpAndBlend3DRestrictingGroupFactory,
    )

    discr = Discretization(
            actx, mesh, PolynomialWarpAndBlend3DRestrictingGroupFactory(order))

    from meshmode.discretization.visualization import make_visualizer
    vis = make_visualizer(actx, discr, order)

    vis.write_vtk_file("geometry.vtu", [
        ("f", actx.thaw(discr.nodes()[0])),
        ])

    from meshmode.discretization.visualization import write_nodal_adjacency_vtk_file

    write_nodal_adjacency_vtk_file("adjacency.vtu",
            mesh)


if __name__ == "__main__":
    main()
