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
from meshmode.interop.firedrake.connection import (
    FromBdyFiredrakeConnection, FromFiredrakeConnection)
from meshmode.interop.firedrake.mesh import import_firedrake_mesh

__all__ = ["FromBdyFiredrakeConnection", "FromFiredrakeConnection",
           "import_firedrake_mesh",
           ]


def _compute_cells_near_bdy(mesh, bdy_id):
    """
    Returns an array of the cell ids with >= 1 vertex on the
    given bdy_id
    """
    cfspace = mesh.coordinates.function_space()
    cell_node_list = cfspace.cell_node_list

    boundary_nodes = cfspace.boundary_nodes(bdy_id, 'topological')
    # Reduce along each cell: Is a vertex of the cell in boundary nodes?
    cell_is_near_bdy = np.any(np.isin(cell_node_list, boundary_nodes), axis=1)

    return np.arange(cell_node_list.shape[0], dtype=np.int32)[cell_is_near_bdy]
