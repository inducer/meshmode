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


import numpy as np
import pyopencl as cl


# Nb: Some of the initial setup was adapted from meshmode/examplse/simple-dg.py
#     written by Andreas Klockner:
#    https://gitlab.tiker.net/inducer/meshmode/-/blob/7826fa5e13854bf1dae425b4226865acc10ee01f/examples/simple-dg.py  # noqa : E501
def main():
    # If can't import firedrake, do nothing
    #
    # filename MUST include "firedrake" (i.e. match *firedrake*.py) in order
    # to be run during CI
    try:
        import firedrake  # noqa : F401
    except ImportError:
        return 0

    # For this example, imagine we wish to solve the Laplace equation
    # on a meshmode mesh with some given Dirichlet boundary conditions,
    # and decide to use firedrake.
    #
    # To verify this is working, we use a solution to the wave equation
    # to get our boundary conditions

    # {{{ First we make a discretization in meshmode and get our bcs

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    from meshmode.array_context import PyOpenCLArrayContext
    actx = PyOpenCLArrayContext(queue)

    nel_1d = 16
    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(-0.5, -0.5),
            b=(0.5, 0.5),
            n=(nel_1d, nel_1d))

    order = 3

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory
    group_factory = InterpolatoryQuadratureSimplexGroupFactory(order=order)
    discr = Discretization(actx, mesh, group_factory)

    # Get our solution: we will use
    # Real(e^z) = Real(e^{x+iy})
    #           = e^x Real(e^{iy})
    #           = e^x cos(y)
    nodes = discr.nodes()
    from meshmode.dof_array import thaw
    for i in range(len(nodes)):
        nodes[i] = thaw(actx, nodes[i])
    # First index is dimension
    candidate_sol = actx.np.exp(nodes[0]) * actx.np.cos(nodes[1])

    # }}}

    # {{{ Now send candidate_sol into firedrake and use it for boundary conditions

    from meshmode.interop.firedrake import build_connection_to_firedrake
    fd_connection = build_connection_to_firedrake(discr, group_nr=0)
    # convert candidate_sol to firedrake
    fd_candidate_sol = fd_connection.from_meshmode(candidate_sol)
    # get the firedrake function space
    fd_fspace = fd_connection.firedrake_fspace()

    # set up dirichlet laplace problem in fd and solve
    from firedrake import (
        FunctionSpace, TrialFunction, TestFunction, Function, inner, grad, dx,
        Constant, project, DirichletBC, solve)

    # because it's easier to write down the variational problem,
    # we're going to project from our "DG" space
    # into a continuous one.
    cfd_fspace = FunctionSpace(fd_fspace.mesh(), "CG", order)
    u = TrialFunction(cfd_fspace)
    v = TestFunction(cfd_fspace)
    sol = Function(cfd_fspace)

    a = inner(grad(u), grad(v)) * dx
    rhs = Constant(0.0) * v * dx
    bc_value = project(fd_candidate_sol, cfd_fspace)
    bc = DirichletBC(cfd_fspace, bc_value, "on_boundary")
    params = {"ksp_monitor": None}
    solve(a == rhs, sol, bcs=[bc], solver_parameters=params)

    # project back into our "DG" space
    sol = project(sol, fd_fspace)

    # }}}

    # {{{ Take the solution from firedrake and compare it to candidate_sol

    true_sol = fd_connection.from_firedrake(sol, actx=actx)
    # pull back into numpy
    true_sol = actx.to_numpy(true_sol[0])
    candidate_sol = actx.to_numpy(candidate_sol[0])
    print("l^2 difference between candidate solution and firedrake solution=",
          np.linalg.norm(true_sol - candidate_sol))

    # }}}


if __name__ == "__main__":
    main()

# vim: foldmethod=marker
