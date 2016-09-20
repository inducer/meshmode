from __future__ import division, absolute_import

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
from six.moves import range


# {{{ draw_2d_mesh

def draw_2d_mesh(mesh, draw_vertex_numbers=True, draw_element_numbers=True,
        draw_nodal_adjacency=False, draw_face_numbers=False,
        set_bounding_box=False, **kwargs):
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

            if draw_element_numbers:
                centroid = (np.sum(elverts, axis=1) / elverts.shape[1])

                if len(mesh.groups) == 1:
                    el_label = str(iel)
                else:
                    el_label = "%d:%d" % (igrp, iel)

                pt.text(centroid[0], centroid[1], el_label, fontsize=17,
                        ha="center", va="center",
                        bbox=dict(facecolor='white', alpha=0.5, lw=0))

    if draw_vertex_numbers:
        for ivert, vert in enumerate(mesh.vertices.T):
            pt.text(vert[0], vert[1], str(ivert), fontsize=15,
                    ha="center", va="center", color="blue",
                    bbox=dict(facecolor='white', alpha=0.5, lw=0))

    if draw_nodal_adjacency:
        def global_iel_to_group_and_iel(global_iel):
            for igrp, grp in enumerate(mesh.groups):
                if global_iel < grp.nelements:
                    return grp, global_iel
                global_iel -= grp.nelements

            raise ValueError("invalid element nr")

        cnx = mesh.nodal_adjacency

        nb_starts = cnx.neighbors_starts
        for iel_g in range(mesh.nelements):
            for nb_iel_g in cnx.neighbors[nb_starts[iel_g]:nb_starts[iel_g+1]]:
                assert iel_g != nb_iel_g

                grp, iel = global_iel_to_group_and_iel(iel_g)
                nb_grp, nb_iel = global_iel_to_group_and_iel(nb_iel_g)

                elverts = mesh.vertices[:, grp.vertex_indices[iel]]
                nb_elverts = mesh.vertices[:, nb_grp.vertex_indices[nb_iel]]

                centroid = (np.sum(elverts, axis=1) / elverts.shape[1])
                nb_centroid = (np.sum(nb_elverts, axis=1) / nb_elverts.shape[1])

                dx = nb_centroid - centroid
                start = centroid + 0.15*dx

                mag = np.max(np.abs(dx))
                start += 0.05*(np.random.rand(2)-0.5)*mag
                dx += 0.05*(np.random.rand(2)-0.5)*mag

                pt.arrow(start[0], start[1], 0.7*dx[0], 0.7*dx[1],
                        length_includes_head=True,
                        color="green", head_width=1e-2, lw=1e-2)

    if draw_face_numbers:
        for igrp, grp in enumerate(mesh.groups):
            for iel, el in enumerate(grp.vertex_indices):
                elverts = mesh.vertices[:, el]
                el_center = np.mean(elverts, axis=-1)

                for iface, fvi in enumerate(grp.face_vertex_indices()):
                    face_center = (
                            0.3*el_center
                            +
                            0.7*np.mean(elverts[:, fvi], axis=-1))

                    pt.text(face_center[0], face_center[1], str(iface), fontsize=12,
                            ha="center", va="center", color="purple",
                            bbox=dict(facecolor='white', alpha=0.5, lw=0))

    if set_bounding_box:
        from meshmode.mesh.processing import find_bounding_box
        lower, upper = find_bounding_box(mesh)
        pt.xlim([lower[0], upper[0]])
        pt.ylim([lower[1], upper[1]])

# }}}


# {{{ draw_curve

def draw_curve(mesh):
    import matplotlib.pyplot as pt
    pt.plot(mesh.vertices[0], mesh.vertices[1], "o")

    for i, group in enumerate(mesh.groups):
        pt.plot(
                group.nodes[0].ravel(),
                group.nodes[1].ravel(), "-x", label="Group %d" % i)

# }}}

# vim: foldmethod=marker
