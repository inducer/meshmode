from __future__ import division
from __future__ import absolute_import
from six.moves import range

__copyright__ = "Copyright (C) 2014 Andreas Kloeckner"

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


# {{{ draw_2d_mesh

def draw_2d_mesh(mesh, draw_numbers=True, **kwargs):
    assert mesh.ambient_dim == 2

    import matplotlib.pyplot as pt
    import matplotlib.patches as mpatches
    from matplotlib.path import Path

    for igrp, grp in enumerate(mesh.groups):
        for iel, el in enumerate(grp.vertex_indices):
            elverts = mesh.vertices[:, el]

            pathdata = [
                (Path.MOVETO, (elverts[0, 0], elverts[1, 0])),
                ]
            for i in range(1, elverts.shape[1]):
                pathdata.append(
                    (Path.LINETO, (elverts[0, i], elverts[1, i]))
                    )
            pathdata.append(
                (Path.CLOSEPOLY, (elverts[0, 0], elverts[1, 0])))

            codes, verts = zip(*pathdata)
            path = Path(verts, codes)
            patch = mpatches.PathPatch(path, **kwargs)
            pt.gca().add_patch(patch)

            if draw_numbers:
                centroid = (np.sum(elverts, axis=1)
                        / elverts.shape[1])

                if len(mesh.groups) == 1:
                    el_label = str(iel)
                else:
                    el_label = "%d:%d" % (igrp, iel)

                pt.text(centroid[0], centroid[1], el_label, fontsize=17,
                        ha="center", va="center",
                        bbox=dict(facecolor='white', alpha=0.5, lw=0))

# }}}

# vim: foldmethod=marker
