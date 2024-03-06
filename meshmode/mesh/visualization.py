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

from typing import Any, Dict, Optional

import numpy as np

from arraycontext import ArrayContext

from meshmode.mesh import Mesh


__doc__ = """
.. autofunction:: draw_2d_mesh
.. autofunction:: draw_curve
.. autofunction:: write_vertex_vtk_file
.. autofunction:: mesh_to_tikz
.. autofunction:: vtk_visualize_mesh
.. autofunction:: write_stl_file

.. autofunction:: visualize_mesh_vertex_resampling_error
"""


# {{{ draw_2d_mesh

def draw_2d_mesh(
        mesh: Mesh, *,
        draw_vertex_numbers: bool = True,
        draw_element_numbers: bool = True,
        draw_nodal_adjacency: bool = False,
        draw_face_numbers: bool = False,
        set_bounding_box: bool = False, **kwargs: Any) -> None:
    """Draw the mesh and its connectivity using ``matplotlib``.

    :arg set_bounding_box: if *True*, the plot limits are set to the mesh
        bounding box. This can help if some of the actors are not visible.
    :arg kwargs: additional arguments passed to ``PathPatch`` when drawing
        the mesh group elements.
    """
    assert mesh.ambient_dim == 2

    import matplotlib.patches as mpatches
    import matplotlib.pyplot as pt
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
                        bbox={"facecolor": "white", "alpha": 0.5, "lw": 0})

    if draw_vertex_numbers:
        for ivert, vert in enumerate(mesh.vertices.T):
            pt.text(vert[0], vert[1], str(ivert), fontsize=15,
                    ha="center", va="center", color="blue",
                    bbox={"facecolor": "white", "alpha": 0.5, "lw": 0})

    if draw_nodal_adjacency:
        def global_iel_to_group_and_iel(global_iel):
            for grp in mesh.groups:
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
        for grp in mesh.groups:
            for el in grp.vertex_indices:
                elverts = mesh.vertices[:, el]
                el_center = np.mean(elverts, axis=-1)

                for iface, fvi in enumerate(grp.face_vertex_indices()):
                    face_center = (
                            0.3*el_center
                            + 0.7*np.mean(elverts[:, fvi], axis=-1))

                    pt.text(face_center[0], face_center[1], str(iface), fontsize=12,
                            ha="center", va="center", color="purple",
                            bbox={"facecolor": "white", "alpha": 0.5, "lw": 0})

    if set_bounding_box:
        from meshmode.mesh.processing import find_bounding_box
        lower, upper = find_bounding_box(mesh)
        pt.xlim([lower[0], upper[0]])
        pt.ylim([lower[1], upper[1]])

# }}}


# {{{ draw_curve

def draw_curve(
        mesh: Mesh, *,
        el_bdry_style: str = "o",
        el_bdry_kwargs: Optional[Dict[str, Any]] = None,
        node_style: str = "x-",
        node_kwargs: Optional[Dict[str, Any]] = None) -> None:
    """Draw a curve mesh.

    :arg el_bdry_kwargs: passed to ``plot`` when drawing elements.
    :arg node_kwargs: passed to ``plot`` when drawing group nodes.
    """

    if not (mesh.ambient_dim == 2 and mesh.dim == 1):
        raise ValueError(
            f"cannot draw a mesh of ambient dimension {mesh.ambient_dim} "
            f"and dimension {mesh.dim}")

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

def write_vertex_vtk_file(
        mesh: Mesh, file_name: str, *,
        compressor: Optional[str] = None,
        overwrite: bool = False) -> None:
    # {{{ create cell_types
    from pyvisfile.vtk import (
        VF_LIST_OF_COMPONENTS, VTK_HEXAHEDRON, VTK_LINE, VTK_QUAD, VTK_TETRA,
        VTK_TRIANGLE, AppendedDataXMLGenerator, DataArray, UnstructuredGrid)

    from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup

    cell_types = np.empty(mesh.nelements, dtype=np.uint8)
    cell_types.fill(255)
    for base_element_nr, egrp in zip(mesh.base_element_nrs, mesh.groups):
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

        cell_types[base_element_nr:base_element_nr + egrp.nelements] = vtk_cell_type

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

def mesh_to_tikz(mesh: Mesh) -> str:
    lines = []

    lines.append(r"\def\nelements{%s}" % mesh.nelements)
    lines.append(r"\def\nvertices{%s}" % mesh.nvertices)
    lines.append("")

    drawel_lines = []
    drawel_lines.append(r"\def\drawelements#1{")

    for base_element_nr, grp in zip(mesh.base_element_nrs, mesh.groups):
        for iel, el in enumerate(grp.vertex_indices):
            el_nr = base_element_nr + iel + 1
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


# {{{ visualize_mesh

def vtk_visualize_mesh(
        actx: ArrayContext, mesh: Mesh, filename: str, *,
        vtk_high_order: bool = True,
        overwrite: bool = False) -> None:
    order = vis_order = max(mgrp.order for mgrp in mesh.groups)
    if not vtk_high_order:
        vis_order = None

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import (
        InterpolatoryEdgeClusteredGroupFactory)
    discr = Discretization(actx, mesh, InterpolatoryEdgeClusteredGroupFactory(order))

    from meshmode.discretization.visualization import make_visualizer
    vis = make_visualizer(actx, discr,
            vis_order=vis_order,
            force_equidistant=vtk_high_order)

    vis.write_vtk_file(filename, [],
            use_high_order=vtk_high_order,
            overwrite=overwrite)

# }}}


# {{{ write_stl_file

def write_stl_file(mesh: Mesh, stl_name: str, *, overwrite: bool = False) -> None:
    """Writes a `STL <https://en.wikipedia.org/wiki/STL_(file_format)>`__ file
    from a triangular mesh in 3D. Requires the
    `numpy-stl <https://pypi.org/project/numpy-stl/>`__ package.
    """

    import stl.mesh

    if len(mesh.groups) != 1:
        raise NotImplementedError("meshes with more than one group are "
                "not yet supported")
    if mesh.ambient_dim != 3:
        raise ValueError("STL export requires a mesh in 3D ambient space")

    grp, = mesh.groups

    from meshmode.mesh import SimplexElementGroup
    if not isinstance(grp, SimplexElementGroup) or grp.dim != 2:
        raise ValueError("STL export requires the mesh to consist of "
                "triangular elements")

    faces = mesh.vertices[:, grp.vertex_indices]

    stl_mesh = stl.mesh.Mesh(
            np.zeros(mesh.nelements, dtype=stl.mesh.Mesh.dtype))
    for iface in range(mesh.nelements):
        for ivertex in range(3):
            stl_mesh.vectors[iface][ivertex] = faces[:, iface, ivertex]

    import os
    if os.path.exists(stl_name):
        if overwrite:
            os.remove(stl_name)
        else:
            raise FileExistsError(f"output file '{stl_name}' already exists")

    stl_mesh.save(stl_name)

# }}}


# {{{ visualize_mesh_vertex_resampling_error

def visualize_mesh_vertex_resampling_error(
        actx: ArrayContext, mesh: Mesh, filename: str, *,
        overwrite: bool = False) -> None:
    # {{{ comput resampling errors

    from meshmode.dof_array import DOFArray
    from meshmode.mesh import _mesh_group_node_vertex_error
    error = DOFArray(actx, tuple([
        actx.from_numpy(
            np.sqrt(np.sum(_mesh_group_node_vertex_error(mesh, mgrp)**2, axis=0))
        )
        for mgrp in mesh.groups
    ]))

    # }}}

    # {{{ visualize

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import (
        InterpolatoryEdgeClusteredGroupFactory)
    discr = Discretization(actx, mesh, InterpolatoryEdgeClusteredGroupFactory(1))

    from meshmode.discretization.visualization import make_visualizer
    vis = make_visualizer(actx, discr)
    vis.write_vtk_file(filename, [("error", error)], overwrite=overwrite)

    # }}}

# }}}

# vim: foldmethod=marker
