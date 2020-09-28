__copyright__ = "Copyright (C) 2020 Benjamin Sepanski"

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

import pyopencl as cl


# This example provides a brief template for bringing information in
# from firedrake and makes some plots to help you better understand
# what a FromBoundaryFiredrakeConnection does
def main():
    # If can't import firedrake, do nothing
    #
    # filename MUST include "firedrake" (i.e. match *firedrake*.py) in order
    # to be run during CI
    try:
        import firedrake  # noqa : F401
    except ImportError:
        return 0

    from meshmode.interop.firedrake import build_connection_from_firedrake
    from firedrake import (
        UnitSquareMesh, FunctionSpace, SpatialCoordinate, Function, cos
        )

    # Create a firedrake mesh and interpolate cos(x+y) onto it
    fd_mesh = UnitSquareMesh(10, 10)
    fd_fspace = FunctionSpace(fd_mesh, "DG", 2)
    spatial_coord = SpatialCoordinate(fd_mesh)
    fd_fntn = Function(fd_fspace).interpolate(cos(sum(spatial_coord)))

    # Make connections
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    from meshmode.array_context import PyOpenCLArrayContext
    actx = PyOpenCLArrayContext(queue)

    fd_connection = build_connection_from_firedrake(actx, fd_fspace)
    fd_bdy_connection = \
        build_connection_from_firedrake(actx,
                                        fd_fspace,
                                        restrict_to_boundary="on_boundary")

    # Plot the meshmode meshes that the connections connect to
    import matplotlib.pyplot as plt
    from meshmode.mesh.visualization import draw_2d_mesh
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title("FiredrakeConnection")
    plt.sca(ax1)
    draw_2d_mesh(fd_connection.discr.mesh,
                 draw_vertex_numbers=False,
                 draw_element_numbers=False,
                 set_bounding_box=True)
    ax2.set_title("FiredrakeConnection 'on_boundary'")
    plt.sca(ax2)
    draw_2d_mesh(fd_bdy_connection.discr.mesh,
                 draw_vertex_numbers=False,
                 draw_element_numbers=False,
                 set_bounding_box=True)
    plt.show()

    # Plot fd_fntn using unrestricted FiredrakeConnection
    from meshmode.discretization.visualization import make_visualizer
    discr = fd_connection.discr
    vis = make_visualizer(actx, discr, discr.groups[0].order+3)
    field = fd_connection.from_firedrake(fd_fntn, actx=actx)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.set_title("cos(x+y) in\nFiredrakeConnection")
    vis.show_scalar_in_matplotlib_3d(field, do_show=False)

    # Now repeat using FiredrakeConnection restricted to "on_boundary"
    bdy_discr = fd_bdy_connection.discr
    bdy_vis = make_visualizer(actx, bdy_discr, bdy_discr.groups[0].order+3)
    bdy_field = fd_bdy_connection.from_firedrake(fd_fntn, actx=actx)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    plt.sca(ax2)
    ax2.set_title("cos(x+y) in\nFiredrakeConnection 'on_boundary'")
    bdy_vis.show_scalar_in_matplotlib_3d(bdy_field, do_show=False)

    import matplotlib.cm as cm
    fig.colorbar(cm.ScalarMappable())
    plt.show()


if __name__ == "__main__":
    main()

# vim: foldmethod=marker
