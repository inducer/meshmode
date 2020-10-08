"""Provides interoperability with the `Matlab/Octave Codes
<https://github.com/tcew/nodal-dg>`__ complementing the
book "Nodal Discontinuous Galerkin Methods" by Jan Hesthaven
and Tim Warburton (Springer, 2008).

.. autoclass:: NodalDGContext
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
        import os.path
        self.octave.feval(os.path.join(self.path, "mypath.m"))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.octave.kill_octave()

    def set_mesh(self, mesh: meshmode.mesh.Mesh):
        """Set the mesh information in the nodal DG Octave instance to
        the one given by *mesh*.

        High-order information is silently ignored.
        """
        pass

    def get_discr(self) -> meshmode.discretization.Discretization:
        """Get a discretization with nodes exactly matching the ones used
        by the book code.

        The returned discretization contains a new :class:`~meshmode.mesh.Mesh`
        object constructed from the global Octave state.
        """

    def push_dof_array(self, name, ary: meshmode.dof_array.DOFArray):
        """
        """
        pass

    def pull_dof_array(
            self, actx: meshmode.array_context.ArrayContext, name
            ) -> meshmode.dof_array.DOFArray:
        pass
