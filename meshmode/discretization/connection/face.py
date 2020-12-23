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

from pytools import Record

import numpy as np
import modepy as mp

import logging
logger = logging.getLogger(__name__)


class FACE_RESTR_INTERIOR:  # noqa
    """A special value to pass to
    :func:`meshmode.discretization.connection.make_face_restriction`
    to produce a discretization consisting of all interior faces
    in a discretization.
    """


class FACE_RESTR_ALL:  # noqa
    """A special value to pass to
    :func:`meshmode.discretization.connection.make_face_restriction`
    to produce a discretization consisting of all faces (interior and boundary)
    faces in a discretization.
    """


# deprecated names for compatibility
FRESTR_ALL_FACES = FACE_RESTR_ALL
FRESTR_INTERIOR_FACES = FACE_RESTR_INTERIOR


# {{{ boundary connection

class _ConnectionBatchData(Record):
    pass


def _build_boundary_connection(actx, vol_discr, bdry_discr, connection_data,
        per_face_groups):
    from meshmode.discretization.connection.direct import (
            InterpolationBatch,
            DiscretizationConnectionElementGroup,
            DirectDiscretizationConnection)

    ibdry_grp = 0
    batches = []

    connection_groups = []
    for igrp, vol_grp in enumerate(vol_discr.groups):
        mgrp = vol_grp.mesh_el_group

        for face_id in range(mgrp.nfaces):
            bdry_grp = bdry_discr.groups[ibdry_grp]
            data = connection_data[igrp, face_id]

            result_unit_nodes = data.face.map_to_volume(bdry_grp.unit_nodes)

            batches.append(
                InterpolationBatch(
                    from_group_index=igrp,
                    from_element_indices=actx.freeze(actx.from_numpy(
                        data.group_source_element_indices)),
                    to_element_indices=actx.freeze(actx.from_numpy(
                        data.group_target_element_indices)),
                    result_unit_nodes=result_unit_nodes,
                    to_element_face=face_id
                    ))

            is_last_face = face_id + 1 == mgrp.nfaces

            if per_face_groups or is_last_face:
                connection_groups.append(
                    DiscretizationConnectionElementGroup(batches))
                batches = []

                ibdry_grp += 1

    assert ibdry_grp == len(bdry_discr.groups)

    return DirectDiscretizationConnection(
            vol_discr, bdry_discr, connection_groups,
            is_surjective=True)

# }}}


# {{{ pull together boundary vertices

def _get_face_vertices(mesh, boundary_tag):
    # a set of volume vertex numbers
    bdry_vertex_vol_nrs = set()

    if boundary_tag not in [FACE_RESTR_INTERIOR, FACE_RESTR_ALL]:
        # {{{ boundary faces

        btag_bit = mesh.boundary_tag_bit(boundary_tag)

        for fagrp_map in mesh.facial_adjacency_groups:
            bdry_grp = fagrp_map.get(None)
            if bdry_grp is None:
                continue

            assert (bdry_grp.neighbors < 0).all()

            grp = mesh.groups[bdry_grp.igroup]

            nb_el_bits = -bdry_grp.neighbors
            face_relevant_flags = (nb_el_bits & btag_bit) != 0

            for iface, fvi in enumerate(grp.face_vertex_indices()):
                bdry_vertex_vol_nrs.update(
                        grp.vertex_indices
                        [bdry_grp.elements[face_relevant_flags]]
                        [:, np.array(fvi, dtype=np.intp)]
                        .flat)

        return np.array(sorted(bdry_vertex_vol_nrs), dtype=np.intp)

        # }}}
    else:
        # For FACE_RESTR_INTERIOR, this is likely every vertex in the book.
        # Don't ever bother trying to cut the list down.
        # For FACE_RESTR_ALL, it literally is every single vertex.

        return np.arange(mesh.nvertices, dtype=np.intp)

# }}}


def make_face_restriction(actx, discr, group_factory, boundary_tag,
        per_face_groups=False):
    """Create a mesh, a discretization and a connection to restrict
    a function on *discr* to its values on the edges of element faces
    denoted by *boundary_tag*.

    :arg boundary_tag: The boundary tag for which to create a face
        restriction. May be
        :class:`FACE_RESTR_INTERIOR`
        to indicate interior faces, or
        :class:`FACE_RESTR_ALL`
        to make a discretization consisting of all (interior and
        boundary) faces.

    :arg per_face_groups: If *True*, the resulting discretization is
        guaranteed to have groups organized as::

            (grp0, face0), (grp0, face1), ... (grp0, faceN),
            (grp1, face0), (grp1, face1), ... (grp1, faceN), ...

        each with the elements in the same order as the originating
        group. If *False*, volume and boundary groups correspond with
        each other one-to-one, and an interpolation batch is created
        per face.

    :return: a
        :class:`meshmode.discretization.connection.DirectDiscretizationConnection`
        representing the new connection. The new boundary discretization can be
        obtained from the
        :attr:`meshmode.discretization.connection.DirectDiscretizationConnection.to_discr`
        attribute of the return value, and the corresponding new boundary mesh
        from that.

    """

    if boundary_tag is None:
        boundary_tag = FACE_RESTR_INTERIOR
        from warnings import warn
        warn("passing *None* for boundary_tag is deprecated--pass "
                "FACE_RESTR_INTERIOR instead",
                DeprecationWarning, stacklevel=2)

    logger.info("building face restriction: start")

    # {{{ gather boundary vertices

    bdry_vertex_vol_nrs = _get_face_vertices(discr.mesh, boundary_tag)

    vol_to_bdry_vertices = np.empty(
            discr.mesh.vertices.shape[-1],
            discr.mesh.vertices.dtype)
    vol_to_bdry_vertices.fill(-1)
    vol_to_bdry_vertices[bdry_vertex_vol_nrs] = np.arange(
            len(bdry_vertex_vol_nrs), dtype=np.intp)

    bdry_vertices = discr.mesh.vertices[:, bdry_vertex_vol_nrs]

    # }}}

    from meshmode.mesh import Mesh, _ModepyElementGroup
    bdry_mesh_groups = []
    connection_data = {}

    if boundary_tag not in [FACE_RESTR_ALL, FACE_RESTR_INTERIOR]:
        btag_bit = discr.mesh.boundary_tag_bit(boundary_tag)

    for igrp, (grp, fagrp_map) in enumerate(
            zip(discr.groups, discr.mesh.facial_adjacency_groups)):

        mgrp = grp.mesh_el_group

        if not isinstance(mgrp, _ModepyElementGroup):
            raise NotImplementedError("can only take boundary of "
                    "meshes based on SimplexElementGroup and "
                    "TensorProductElementGroup")

        # {{{ pull together per-group face lists

        group_boundary_faces = []

        if boundary_tag is FACE_RESTR_INTERIOR:
            for fagrp in fagrp_map.values():
                if fagrp.ineighbor_group is None:
                    # boundary faces -> not looking for those
                    continue

                group_boundary_faces.extend(
                        zip(fagrp.elements, fagrp.element_faces))

        elif boundary_tag is FACE_RESTR_ALL:
            group_boundary_faces.extend(
                    (iel, iface)
                    for iface in range(grp.mesh_el_group.nfaces)
                    for iel in range(grp.nelements)
                    )

        else:
            bdry_grp = fagrp_map.get(None)
            if bdry_grp is not None:
                nb_el_bits = -bdry_grp.neighbors
                face_relevant_flags = (nb_el_bits & btag_bit) != 0

                group_boundary_faces.extend(
                            zip(
                                bdry_grp.elements[face_relevant_flags],
                                bdry_grp.element_faces[face_relevant_flags]))

        # }}}

        batch_base = 0

        # group by face_index

        for face in mgrp._modepy_faces:
            batch_boundary_el_numbers_in_grp = np.array([
                ibface_el
                for ibface_el, ibface_face in group_boundary_faces
                if ibface_face == face.face_index
                ], dtype=np.intp)

            # {{{ preallocate arrays for mesh group

            nbatch_elements = len(batch_boundary_el_numbers_in_grp)

            if per_face_groups or face.face_index == 0:
                if per_face_groups:
                    ngroup_bdry_elements = nbatch_elements
                else:
                    ngroup_bdry_elements = len(group_boundary_faces)

                # make up some not-terrible nodes for the boundary Mesh
                space = mp.space_for_shape(face, mgrp.order)
                bdry_unit_nodes = mp.edge_clustered_nodes_for_space(space, face)

                vol_basis = mp.basis_for_space(
                        mgrp._modepy_space, mgrp._modepy_shape).functions

                vertex_indices = np.empty(
                        (ngroup_bdry_elements, face.nvertices),
                        mgrp.vertex_indices.dtype)

                nbdry_unit_nodes = bdry_unit_nodes.shape[-1]
                nodes = np.empty(
                        (discr.ambient_dim, ngroup_bdry_elements, nbdry_unit_nodes),
                        dtype=np.float64)
            # }}}

            new_el_numbers = batch_base + np.arange(nbatch_elements)
            if not per_face_groups:
                batch_base += nbatch_elements

            # {{{ no per-element axes in these computations

            face_unit_nodes = face.map_to_volume(bdry_unit_nodes)
            resampling_mat = mp.resampling_matrix(
                    vol_basis,
                    face_unit_nodes, mgrp.unit_nodes)

            # }}}

            # {{{ build information for mesh element group

            # Find vertex_indices
            glob_face_vertices = mgrp.vertex_indices[
                    batch_boundary_el_numbers_in_grp][:, face.volume_vertex_indices]
            vertex_indices[new_el_numbers] = \
                    vol_to_bdry_vertices[glob_face_vertices]

            # Find nodes
            nodes[:, new_el_numbers, :] = np.einsum(
                    "ij,dej->dei",
                    resampling_mat,
                    mgrp.nodes[:, batch_boundary_el_numbers_in_grp, :])

            # }}}

            connection_data[igrp, face.face_index] = _ConnectionBatchData(
                    group_source_element_indices=batch_boundary_el_numbers_in_grp,
                    group_target_element_indices=new_el_numbers,
                    face=face,
                    )

            is_last_face = face.face_index + 1 == mgrp.nfaces

            if per_face_groups or is_last_face:
                bdry_mesh_group = type(mgrp)(
                        mgrp.order, vertex_indices, nodes,
                        unit_nodes=bdry_unit_nodes)
                bdry_mesh_groups.append(bdry_mesh_group)

    bdry_mesh = Mesh(bdry_vertices, bdry_mesh_groups)

    from meshmode.discretization import Discretization
    bdry_discr = Discretization(
            actx, bdry_mesh, group_factory)

    connection = _build_boundary_connection(
            actx, discr, bdry_discr, connection_data,
            per_face_groups)

    logger.info("building face restriction: done")

    return connection

# }}}


# {{{ face -> all_faces connection

def make_face_to_all_faces_embedding(actx, faces_connection, all_faces_discr,
        from_discr=None):
    """Return a
    :class:`meshmode.discretization.connection.DiscretizationConnection`
    connecting a discretization containing some faces of a discretization
    to one containing all faces.

    :arg faces_connection: must be the (connection) result of calling
        :func:`meshmode.discretization.connection.make_face_restriction`
        with
        :class:`meshmode.discretization.connection.FACE_RESTR_INTERIOR`
        or a boundary tag.
    :arg all_faces_discr: must be the (discretization) result of calling
        :func:`meshmode.discretization.connection.make_face_restriction`
        with
        :class:`meshmode.discretization.connection.FACE_RESTR_ALL`
        for the same volume discretization as the one from which
        *faces_discr* was obtained.
    :arg from_discr: Allows substituting in a different origin
        discretization for the returned connection. This discretization
        must use the same mesh as ``faces_connection.to_discr``.
    """

    vol_discr = faces_connection.from_discr
    faces_discr = faces_connection.to_discr

    if from_discr is None:
        from_discr = faces_discr

    assert from_discr.mesh is faces_discr.mesh

    per_face_groups = (
            len(vol_discr.groups) != len(faces_discr.groups))

    if len(faces_discr.groups) != len(all_faces_discr.groups):
        raise ValueError("faces_discr and all_faces_discr must have the "
                "same number of groups")
    if len(faces_connection.groups) != len(all_faces_discr.groups):
        raise ValueError("faces_connection and all_faces_discr must have the "
                "same number of groups")

    from meshmode.discretization.connection import (
            DirectDiscretizationConnection,
            DiscretizationConnectionElementGroup,
            InterpolationBatch)

    i_faces_grp = 0

    groups = []
    for ivol_grp, vol_grp in enumerate(vol_discr.groups):
        batches = []

        nfaces = vol_grp.mesh_el_group.nfaces
        for iface in range(nfaces):
            all_faces_grp = all_faces_discr.groups[i_faces_grp]

            if per_face_groups:
                assert len(faces_connection.groups[i_faces_grp].batches) == 1
            else:
                assert (len(faces_connection.groups[i_faces_grp].batches)
                        == nfaces)

            assert np.array_equal(
                    from_discr.groups[i_faces_grp].unit_nodes,
                    all_faces_grp.unit_nodes)

            # {{{ find src_batch

            src_batches = faces_connection.groups[i_faces_grp].batches
            if per_face_groups:
                src_batch, = src_batches
            else:
                src_batch = src_batches[iface]
            del src_batches

            # }}}

            if per_face_groups:
                to_element_indices = src_batch.from_element_indices
            else:
                assert all_faces_grp.nelements == nfaces * vol_grp.nelements

                to_element_indices = actx.freeze(
                        vol_grp.nelements*iface
                        + actx.thaw(src_batch.from_element_indices))

            batches.append(
                    InterpolationBatch(
                        from_group_index=i_faces_grp,
                        from_element_indices=src_batch.to_element_indices,
                        to_element_indices=to_element_indices,
                        result_unit_nodes=all_faces_grp.unit_nodes,
                        to_element_face=None))

            is_last_face = iface + 1 == nfaces
            if per_face_groups or is_last_face:
                groups.append(
                        DiscretizationConnectionElementGroup(batches=batches))
                batches = []

                i_faces_grp += 1

    return DirectDiscretizationConnection(
            from_discr,
            all_faces_discr,
            groups,
            is_surjective=False)

# }}}

# vim: foldmethod=marker
