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

__doc__ = """
.. autofunction:: draw_2d_mesh
.. autofunction:: draw_curve
.. autofunction:: write_vertex_vtk_file
.. autofunction:: mesh_to_tikz
"""


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

            from meshmode.mesh import TensorProductElementGroup
            if isinstance(grp, TensorProductElementGroup) and grp.dim == 2:
                elverts = elverts[:,
                        np.array([0, 1, 3, 2])]

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
                        bbox=dict(facecolor="white", alpha=0.5, lw=0))

    if draw_vertex_numbers:
        for ivert, vert in enumerate(mesh.vertices.T):
            pt.text(vert[0], vert[1], str(ivert), fontsize=15,
                    ha="center", va="center", color="blue",
                    bbox=dict(facecolor="white", alpha=0.5, lw=0))

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
                            + 0.7*np.mean(elverts[:, fvi], axis=-1))

                    pt.text(face_center[0], face_center[1], str(iface), fontsize=12,
                            ha="center", va="center", color="purple",
                            bbox=dict(facecolor="white", alpha=0.5, lw=0))

    if set_bounding_box:
        from meshmode.mesh.processing import find_bounding_box
        lower, upper = find_bounding_box(mesh)
        pt.xlim([lower[0], upper[0]])
        pt.ylim([lower[1], upper[1]])

# }}}


# {{{ draw_curve

def draw_curve(mesh,
        el_bdry_style="o", el_bdry_kwargs=None,
        node_style="x-", node_kwargs=None):
    import matplotlib.pyplot as plt

    if el_bdry_kwargs is None:
        el_bdry_kwargs = {}
    if node_kwargs is None:
        node_kwargs = {}

    plt.plot(mesh.vertices[0], mesh.vertices[1], el_bdry_style, **el_bdry_kwargs)

    for i, group in enumerate(mesh.groups):
        plt.plot(
                group.nodes[0].T,
                group.nodes[1].T, node_style, label="Group %d" % i,
                **node_kwargs)

# }}}


# {{{ write_vtk_file

def write_vertex_vtk_file(mesh, file_name,
                          compressor=None,
                          overwrite=False):
    from pyvisfile.vtk import (
            UnstructuredGrid, DataArray,
            AppendedDataXMLGenerator,
            VF_LIST_OF_COMPONENTS)

    # {{{ create cell_types

    from pyvisfile.vtk import (
            VTK_LINE, VTK_TRIANGLE, VTK_TETRA,
            VTK_QUAD, VTK_HEXAHEDRON)

    from meshmode.mesh import TensorProductElementGroup, SimplexElementGroup

    cell_types = np.empty(mesh.nelements, dtype=np.uint8)
    cell_types.fill(255)
    for egrp in mesh.groups:
        if isinstance(egrp, SimplexElementGroup):
            vtk_cell_type = {
                    1: VTK_LINE,
                    2: VTK_TRIANGLE,
                    3: VTK_TETRA,
                    }[egrp.dim]
        elif isinstance(egrp, TensorProductElementGroup):
            vtk_cell_type = {
                    1: VTK_LINE,
                    2: VTK_QUAD,
                    3: VTK_HEXAHEDRON,
                    }[egrp.dim]
        else:
            raise NotImplementedError("mesh vtk file writing for "
                    "element group of type '%s'" % type(egrp).__name__)

        cell_types[
                egrp.element_nr_base:
                egrp.element_nr_base + egrp.nelements] = \
                        vtk_cell_type

    assert (cell_types != 255).all()

    # }}}

    # {{{ create cell connectivity

    cells = np.empty(
            sum(egrp.vertex_indices.size for egrp in mesh.groups),
            dtype=mesh.vertex_id_dtype)

    # NOTE: vtk uses z-order for the linear quads
    tensor_order = {
            1: (0, 1),
            2: (0, 1, 3, 2),
            3: (0, 1, 3, 2, 4, 5, 7, 6)
            }

    vertex_nr_base = 0
    for egrp in mesh.groups:
        i = np.s_[vertex_nr_base:vertex_nr_base + egrp.vertex_indices.size]
        if isinstance(egrp, SimplexElementGroup):
            cells[i] = egrp.vertex_indices.reshape(-1)
        elif isinstance(egrp, TensorProductElementGroup):
            cells[i] = egrp.vertex_indices[:, tensor_order[egrp.dim]].reshape(-1)
        else:
            raise TypeError("unsupported group type")

        vertex_nr_base += egrp.vertex_indices.size

    # }}}

    grid = UnstructuredGrid(
            (mesh.nvertices,
                DataArray("points",
                    mesh.vertices,
                    vector_format=VF_LIST_OF_COMPONENTS)),
            cells=cells,
            cell_types=cell_types)

    import os
    if os.path.exists(file_name):
        if overwrite:
            os.remove(file_name)
        else:
            raise FileExistsError("output file '%s' already exists" % file_name)

    with open(file_name, "w") as outf:
        AppendedDataXMLGenerator(compressor)(grid).write(outf)

# }}}


# {{{ mesh_to_tikz

def mesh_to_tikz(mesh):
    lines = []

    lines.append(r"\def\nelements{%s}" % (sum(grp.nelements for grp in mesh.groups)))
    lines.append(r"\def\nvertices{%s}" % mesh.nvertices)
    lines.append("")

    drawel_lines = []
    drawel_lines.append(r"\def\drawelements#1{")

    for grp in mesh.groups:
        for iel, el in enumerate(grp.vertex_indices):
            el_nr = grp.element_nr_base+iel+1
            elverts = mesh.vertices[:, el]

            centroid = np.average(elverts, axis=1)
            lines.append(r"\coordinate (cent%d) at (%s);"
                    % (el_nr,
                        ", ".join("%.5f" % (vi) for vi in centroid)))

            for ivert, vert in enumerate(elverts.T):
                lines.append(r"\coordinate (v%d-%d) at (%s);"
                        % (el_nr, ivert+1, ", ".join("%.5f" % vi for vi in vert)))
            drawel_lines.append(
                    r"\draw [#1] %s -- cycle;"
                    % " -- ".join(
                        "(v%d-%d)" % (el_nr, vi+1)
                        for vi in range(elverts.shape[1])))
    drawel_lines.append("}")
    lines.append("")

    lines.extend(drawel_lines)

    return "\n".join(lines)

# }}}

# vim: foldmethod=marker
