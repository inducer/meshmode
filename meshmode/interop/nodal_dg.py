"""Provides interoperability with the `Matlab/Octave Codes
<https://github.com/tcew/nodal-dg>`__ complementing the
book "Nodal Discontinuous Galerkin Methods" by Jan Hesthaven
and Tim Warburton (Springer, 2008).

.. autoclass:: NodalDGContext

.. autofunction:: download_nodal_dg_if_not_present
"""

__copyright__ = "Copyright (C) 2020 Andreas Kloeckner"

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
import meshmode.mesh
import meshmode.discretization
import meshmode.dof_array
import meshmode.array_context


class NodalDGContext(object):
    """Should be used as a context manager to ensure proper cleanup.

    .. automethod:: __init__
    .. automethod:: set_mesh
    .. automethod:: get_discr
    .. automethod:: push_dof_array
    .. automethod:: pull_dof_array
    """

    def __init__(self, path):
        """
        :arg path: The path to the ``Codes1.1`` folder of the nodal DG codes.
        """
        self.path = path
        self.octave = None

    def __enter__(self):
        import oct2py
        self.octave = oct2py.Oct2Py()
        self.octave.eval(f'cd "{self.path}"')
        self.octave.eval("mypath")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Work around https://github.com/pexpect/pexpect/issues/462
        self.octave._engine.repl.delayafterterminate = 2

        self.octave.exit()

    REF_AXES = ["r", "s", "t"]
    AXES = ["x", "y", "z"]

    def set_mesh(self, mesh: meshmode.mesh.Mesh, order):
        """Set the mesh information in the nodal DG Octave instance to
        the one given by *mesh*.

        The mesh must only have a single element group of simplices.

        .. warning::

            High-order geometry information is currently silently ignored.
        """
        if len(mesh.groups) != 1:
            raise ValueError("mesh must have exactly one element group")

        elgrp, = mesh.groups

        self.octave.eval(f"Globals{mesh.dim}D;")
        self.octave.push("Nv", mesh.nvertices)
        self.octave.push("K", mesh.nelements)
        for ax in range(mesh.ambient_dim):
            self.octave.push(f"V{self.AXES[ax].upper()}", mesh.vertices[ax])

        self.octave.push(f"V{self.AXES[ax].upper()}", mesh.vertices[ax])
        self.octave.push("EToV", elgrp.vertex_indices+1)

        self.octave.push("N", order)

        self.octave.eval(f"StartUp{mesh.dim}D;")

    def get_discr(self, actx) -> meshmode.discretization.Discretization:
        """Get a discretization with nodes exactly matching the ones used
        by the nodal-DG code.

        The returned discretization contains a new :class:`~meshmode.mesh.Mesh`
        object constructed from the global Octave state.
        """
        # find dim as number of vertices in the simplex - 1
        etov_size = self.octave.eval("size(EToV)", verbose=False)
        dim = int(etov_size[0, 1]-1)

        if dim == 1:
            unit_nodes = self.octave.eval("JacobiGL(0, 0, N)", verbose=False).T
        else:
            unit_nodes_arrays = self.octave.eval(
                    f"Nodes{dim}D(N)", nout=dim, verbose=False)

            equilat_to_unit_func_name = (
                    "".join(self.AXES[:dim] + ["to"] + self.REF_AXES[:dim]))

            unit_nodes_arrays = self.octave.feval(
                    equilat_to_unit_func_name, *unit_nodes_arrays,
                    nout=dim, verbose=False)

            unit_nodes = np.array([a.reshape(-1) for a in unit_nodes_arrays])

        vertices = np.array([
                self.octave.pull(f"V{self.AXES[ax].upper()}").reshape(-1)
                for ax in range(dim)])
        nodes = np.array([self.octave.pull(self.AXES[ax]).T for ax in range(dim)])
        vertex_indices = (self.octave.pull("EToV")).astype(np.int32)-1

        from meshmode.mesh import Mesh, SimplexElementGroup
        order = int(self.octave.pull("N"))
        egroup = SimplexElementGroup(
                order,
                vertex_indices=vertex_indices,
                nodes=nodes,
                unit_nodes=unit_nodes)

        mesh = Mesh(vertices=vertices, groups=[egroup], is_conforming=True)

        from meshmode.discretization import Discretization
        from meshmode.discretization.poly_element import (
                PolynomialGivenNodesGroupFactory)
        return Discretization(actx, mesh,
                PolynomialGivenNodesGroupFactory(order, unit_nodes))

    def push_dof_array(self, name, ary: meshmode.dof_array.DOFArray):
        """
        """
        grp_array, = ary
        ary = ary.array_context.to_numpy(grp_array)
        self.octave.push(name, ary.T)

    def pull_dof_array(
            self, actx: meshmode.array_context.ArrayContext, name
            ) -> meshmode.dof_array.DOFArray:
        ary = self.octave.pull(name).T

        return meshmode.dof_array.DOFArray(actx, (actx.from_numpy(ary),))


def download_nodal_dg_if_not_present(path="nodal-dg"):
    """Download the nodal-DG source code.

    :arg path: The destination path.
    """
    import os
    if os.path.exists(path):
        return

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        filename = os.path.join(tmp, "master.zip")

        from pytools import download_from_web_if_not_present
        download_from_web_if_not_present(
                url="https://github.com/tcew/nodal-dg/archive/master.zip",
                local_name=filename)

        import zipfile
        with zipfile.ZipFile(filename, "r") as zp:
            zp.extractall(tmp)

        if not os.path.exists(path):
            import shutil
            shutil.move(os.path.join(tmp, "nodal-dg-master"), path)
