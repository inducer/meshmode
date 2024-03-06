__copyright__ = "Copyright (C) 2014-6 Shivam Gupta"

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

import logging

from meshmode.mesh.refinement.no_adjacency import RefinerWithoutAdjacency
from meshmode.mesh.refinement.utils import Refiner


logger = logging.getLogger(__name__)

__doc__ = """
.. autoclass:: Refiner
.. autoclass :: RefinerWithoutAdjacency
.. autofunction :: refine_uniformly
"""

__all__ = [
    "Refiner",
    "RefinerWithoutAdjacency", "refine_uniformly"
]


def refine_uniformly(mesh, iterations, with_adjacency=False):
    if with_adjacency:
        # For conforming meshes, even RefinerWithoutAdjacency will reconstruct
        # adjacency from vertex identity.

        if not mesh.is_conforming:
            raise ValueError("mesh must be conforming if adjacency is desired")

    refiner = RefinerWithoutAdjacency(mesh)

    for _ in range(iterations):
        refiner.refine_uniformly()

    return refiner.get_current_mesh()


# vim: foldmethod=marker
