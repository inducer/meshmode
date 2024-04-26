__copyright__ = """
Copyright (C) 2018 Andreas Kloeckner
Copyright (C) 2014-6 Shivam Gupta
Copyright (C) 2016 Matt Wala
"""

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

import numpy as np

from meshmode.mesh.refinement.utils import Refiner


logger = logging.getLogger(__name__)


class RefinerWithoutAdjacency(Refiner):
    """A refiner that may be applied to non-conforming
    :class:`meshmode.mesh.Mesh` instances. It does not generate adjacency
    information.

    .. note::

        If the input meshes to this refiner are not conforming, then
        the resulting meshes may contain duplicated vertices.
        (I.e. two different numbers referring to the same geometric
        vertex.)

    .. automethod:: __init__
    .. automethod:: refine
    .. automethod:: refine_uniformly
    .. automethod:: get_current_mesh
    .. automethod:: get_previous_mesh
    """

    def __init__(self, mesh):
        self._current_mesh = mesh
        self._previous_mesh = None
        self.group_refinement_records = None
        self.global_vertex_pair_to_midpoint = {}

    def refine_uniformly(self):
        flags = np.ones(self._current_mesh.nelements, dtype=bool)
        return self.refine(flags)

    # {{{ refinement top-level

    def refine(self, refine_flags):
        """
        :arg refine_flags: an :class:`~numpy.ndarray` of :class:`~numpy.dtype`
            :class:`bool` and length :attr:`meshmode.mesh.Mesh.nelements`
            indicating which elements should be split.
        """

        mesh = self._current_mesh
        refine_flags = np.asarray(refine_flags, dtype=bool)

        if len(refine_flags) != mesh.nelements:
            raise ValueError("length of refine_flags does not match "
                    "element count of last generated mesh")

        perform_vertex_updates = mesh.vertices is not None

        new_el_groups = []
        group_refinement_records = []
        additional_vertices = []
        if perform_vertex_updates:
            inew_vertex = mesh.nvertices

        from meshmode.mesh.refinement.tessellate import (
            get_group_midpoints, get_group_tessellated_nodes,
            get_group_tessellation_info)

        for base_element_nr, grp in zip(mesh.base_element_nrs, mesh.groups):
            el_tess_info = get_group_tessellation_info(grp)

            # {{{ compute counts and index arrays

            grp_flags = refine_flags[base_element_nr:base_element_nr + grp.nelements]

            nchildren = len(el_tess_info.children)
            nchild_elements = np.ones(grp.nelements, dtype=mesh.element_id_dtype)
            nchild_elements[grp_flags] = nchildren

            child_el_indices = np.empty(grp.nelements+1, dtype=mesh.element_id_dtype)
            child_el_indices[0] = 0
            child_el_indices[1:] = np.cumsum(nchild_elements)

            unrefined_el_new_indices = child_el_indices[:-1][~grp_flags]
            refining_el_old_indices, = np.where(grp_flags)

            new_nelements = child_el_indices[-1]

            # }}}

            from meshmode.mesh.refinement.tessellate import GroupRefinementRecord
            group_refinement_records.append(
                    GroupRefinementRecord(
                        el_tess_info=el_tess_info,
                        element_mapping=[
                            list(range(
                                child_el_indices[iel],
                                child_el_indices[iel]+nchild_elements[iel]))
                            for iel in range(grp.nelements)]))

            # {{{ get new vertices together

            if perform_vertex_updates:
                midpoints = get_group_midpoints(
                        grp, el_tess_info, refining_el_old_indices)

                new_vertex_indices = np.empty(
                    (new_nelements, grp.vertex_indices.shape[1]),
                    dtype=mesh.vertex_id_dtype)
                new_vertex_indices.fill(-17)

                # copy over unchanged vertices
                new_vertex_indices[unrefined_el_new_indices] = \
                        grp.vertex_indices[~grp_flags]

                for old_iel in refining_el_old_indices:
                    new_iel_base = child_el_indices[old_iel]

                    refining_vertices = np.empty(len(el_tess_info.ref_vertices),
                        dtype=mesh.vertex_id_dtype)
                    refining_vertices.fill(-17)

                    # carry over old vertices
                    refining_vertices[el_tess_info.orig_vertex_indices] = \
                            grp.vertex_indices[old_iel]

                    for imidpoint, (iref_midpoint, (v1, v2)) in enumerate(zip(
                            el_tess_info.midpoint_indices,
                            el_tess_info.midpoint_vertex_pairs)):

                        global_v1 = grp.vertex_indices[old_iel, v1]
                        global_v2 = grp.vertex_indices[old_iel, v2]

                        if global_v1 > global_v2:
                            global_v1, global_v2 = global_v2, global_v1

                        try:
                            global_midpoint = self.global_vertex_pair_to_midpoint[
                                    global_v1, global_v2]
                        except KeyError:
                            global_midpoint = inew_vertex
                            additional_vertices.append(
                                    midpoints[old_iel][:, imidpoint])
                            self.global_vertex_pair_to_midpoint[
                                    global_v1, global_v2] = global_midpoint
                            inew_vertex += 1

                        refining_vertices[iref_midpoint] = global_midpoint

                    assert (refining_vertices >= 0).all()

                    new_vertex_indices[new_iel_base:new_iel_base+nchildren] = \
                            refining_vertices[el_tess_info.children]

                assert (new_vertex_indices >= 0).all()
            else:
                new_vertex_indices = None

            # }}}

            # {{{ get new nodes together

            new_nodes = np.empty(
                (mesh.ambient_dim, new_nelements, grp.nunit_nodes),
                dtype=grp.nodes.dtype)

            new_nodes.fill(float("nan"))

            # copy over unchanged nodes
            new_nodes[:, unrefined_el_new_indices] = grp.nodes[:, ~grp_flags]

            tessellated_nodes = get_group_tessellated_nodes(
                    grp, el_tess_info, refining_el_old_indices)

            for old_iel in refining_el_old_indices:
                new_iel_base = child_el_indices[old_iel]
                new_nodes[:, new_iel_base:new_iel_base+nchildren, :] = \
                        tessellated_nodes[old_iel]

            assert (~np.isnan(new_nodes)).all()

            # }}}

            new_el_groups.append(
                grp.make_group(
                    order=grp.order,
                    vertex_indices=new_vertex_indices,
                    nodes=new_nodes,
                    unit_nodes=grp.unit_nodes))

        if perform_vertex_updates:
            new_vertices = np.empty(
                    (mesh.ambient_dim, mesh.nvertices + len(additional_vertices)),
                    mesh.vertices.dtype)
            new_vertices[:, :mesh.nvertices] = mesh.vertices
            new_vertices[:, mesh.nvertices:] = np.array(additional_vertices).T
        else:
            new_vertices = None

        from meshmode.mesh import make_mesh
        new_mesh = make_mesh(new_vertices, new_el_groups, is_conforming=(
            mesh.is_conforming
            and (refine_flags.all() or (~refine_flags).all())))

        self.group_refinement_records = group_refinement_records
        self._current_mesh = new_mesh
        self._previous_mesh = mesh

        return new_mesh

    # }}}

    def get_current_mesh(self):
        return self._current_mesh

    def get_previous_mesh(self):
        return self._previous_mesh


# vim: foldmethod=marker
