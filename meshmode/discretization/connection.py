from __future__ import division
from __future__ import absolute_import
import six
from six.moves import range
from six.moves import zip

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
import modepy as mp
import pyopencl as cl
import pyopencl.array  # noqa
from pytools import memoize_method, memoize_method_nested, Record

import logging
logger = logging.getLogger(__name__)


__doc__ = """
.. autoclass:: DiscretizationConnection

.. autofunction:: make_same_mesh_connection

.. autofunction:: make_boundary_restriction

Implementation details
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: InterpolationBatch

.. autoclass:: DiscretizationConnectionElementGroup
"""


class InterpolationBatch(object):
    """One interpolation batch captures how a batch of elements *within* an
    element group should be an interpolated. Note that while it's possible that
    an interpolation batch takes care of interpolating an entire element group
    from source to target, that's not *necessarily* the case. Consider the case
    of extracting boundary values of a discretization. For, say, a triangle, at
    least three different interpolation batches are needed to cover boundary
    edges that fall onto each of the three edges of the unit triangle.

    .. attribute:: source_element_indices

        A :class:`pyopencl.array.Arrays` of length ``nelements``, containing the
        element index from which this "*to*" element's data will be
        interpolated.

    .. attribute:: target_element_indices

        A :class:`pyopencl.array.Arrays` of length ``nelements``, containing the
        element index to which this "*to*" element's data will be
        interpolated.

    .. attribute:: result_unit_nodes

        A :class:`numpy.ndarray` of shape
        ``(from_group.dim,to_group.nelements,to_group.nunit_nodes)``
        storing the coordinates of the nodes (in unit coordinates
        of the *from* reference element) from which the node
        locations of this element should be interpolated.
    """

    def __init__(self, source_element_indices,
            target_element_indices, result_unit_nodes):
        self.source_element_indices = source_element_indices
        self.target_element_indices = target_element_indices
        self.result_unit_nodes = result_unit_nodes

    @property
    def nelements(self):
        return len(self.source_element_indices)


class DiscretizationConnectionElementGroup(object):
    """
    .. attribute:: batches

        A list of :class:`InterpolationBatch` instances.
    """
    def __init__(self, batches):
        self.batches = batches


class DiscretizationConnection(object):
    """
    .. attribute:: from_discr

    .. attribute:: to_discr

    .. attribute:: groups

        a list of :class:`MeshConnectionGroup` instances, with
        a one-to-one correspondence to the groups in
        :attr:`from_discr` and :attr:`to_discr`.
    """

    def __init__(self, from_discr, to_discr, groups):
        if from_discr.cl_context != to_discr.cl_context:
            raise ValueError("from_discr and to_discr must live in the "
                    "same OpenCL context")

        self.cl_context = from_discr.cl_context

        self.from_discr = from_discr
        self.to_discr = to_discr
        self.groups = groups

    @memoize_method
    def _resample_matrix(self, elgroup_index, ibatch_index):
        import modepy as mp
        ibatch = self.groups[elgroup_index].batches[ibatch_index]
        from_grp = self.from_discr.groups[elgroup_index]

        return mp.resampling_matrix(
                mp.simplex_onb(self.from_discr.dim, from_grp.order),
                ibatch.result_unit_nodes, from_grp.unit_nodes)

    def __call__(self, queue, vec):
        @memoize_method_nested
        def knl():
            import loopy as lp
            knl = lp.make_kernel(
                """{[k,i,j]:
                    0<=k<nelements and
                    0<=i<n_to_nodes and
                    0<=j<n_from_nodes}""",
                "result[target_element_indices[k], i] \
                    = sum(j, resample_mat[i, j] \
                    * vec[source_element_indices[k], j])",
                [
                    lp.GlobalArg("result", None,
                        shape="nelements_result, n_to_nodes",
                        offset=lp.auto),
                    lp.GlobalArg("vec", None,
                        shape="nelements_vec, n_from_nodes",
                        offset=lp.auto),
                    lp.ValueArg("nelements_result", np.int32),
                    lp.ValueArg("nelements_vec", np.int32),
                    "...",
                    ],
                name="oversample")

            knl = lp.split_iname(knl, "i", 16, inner_tag="l.0")
            return lp.tag_inames(knl, dict(k="g.0"))

        if not isinstance(vec, cl.array.Array):
            return vec

        result = self.to_discr.empty(vec.dtype)

        if vec.shape != (self.from_discr.nnodes,):
            raise ValueError("invalid shape of incoming resampling data")

        for i_grp, (sgrp, tgrp, cgrp) in enumerate(
                zip(self.to_discr.groups, self.from_discr.groups, self.groups)):
            for i_batch, batch in enumerate(cgrp.batches):
                if len(batch.source_element_indices):
                    knl()(queue,
                            resample_mat=self._resample_matrix(i_grp, i_batch),
                            result=sgrp.view(result), vec=tgrp.view(vec),
                            source_element_indices=batch.source_element_indices,
                            target_element_indices=batch.target_element_indices)

        return result

    # }}}


# {{{ same-mesh constructor

def make_same_mesh_connection(queue, to_discr, from_discr):
    if from_discr.mesh is not to_discr.mesh:
        raise ValueError("from_discr and to_discr must be based on "
                "the same mesh")

    assert queue.context == from_discr.cl_context
    assert queue.context == to_discr.cl_context

    groups = []
    for fgrp, tgrp in zip(from_discr.groups, to_discr.groups):
        all_elements = cl.array.arange(queue,
                fgrp.nelements,
                dtype=np.intp).with_queue(None)
        ibatch = InterpolationBatch(
                source_element_indices=all_elements,
                target_element_indices=all_elements,
                result_unit_nodes=tgrp.unit_nodes)

        groups.append(
                DiscretizationConnectionElementGroup([ibatch]))

    return DiscretizationConnection(
            from_discr, to_discr, groups)

# }}}


# {{{ boundary restriction constructor

class _ConnectionBatchData(Record):
    pass


def _build_boundary_connection(queue, vol_discr, bdry_discr, connection_data):
    connection_groups = []
    for igrp, (vol_grp, bdry_grp) in enumerate(
            zip(vol_discr.groups, bdry_discr.groups)):
        connection_batches = []
        mgrp = vol_grp.mesh_el_group

        for face_id in range(len(mgrp.face_vertex_indices())):
            data = connection_data[igrp, face_id]

            bdry_unit_nodes_01 = (bdry_grp.unit_nodes + 1)*0.5
            result_unit_nodes = (np.dot(data.A, bdry_unit_nodes_01).T + data.b).T

            connection_batches.append(
                    InterpolationBatch(
                        source_element_indices=cl.array.to_device(
                            queue,
                            vol_grp.mesh_el_group.element_nr_base
                            + data.group_source_element_indices)
                        .with_queue(None),
                        target_element_indices=cl.array.to_device(
                            queue,
                            bdry_grp.mesh_el_group.element_nr_base
                            + data.group_target_element_indices)
                        .with_queue(None),
                        result_unit_nodes=result_unit_nodes,
                        ))

        connection_groups.append(
                DiscretizationConnectionElementGroup(
                    connection_batches))

    return DiscretizationConnection(
            vol_discr, bdry_discr, connection_groups)


def make_boundary_restriction(queue, discr, group_factory):
    """
    :return: a tuple ``(bdry_mesh, bdry_discr, connection)``
    """

    logger.info("building boundary connection: start")

    # {{{ build face_map

    # maps (igrp, el_grp, face_id) to a frozenset of vertex IDs
    face_map = {}

    for igrp, mgrp in enumerate(discr.mesh.groups):
        grp_face_vertex_indices = mgrp.face_vertex_indices()

        for iel_grp in range(mgrp.nelements):
            for fid, loc_face_vertices in enumerate(grp_face_vertex_indices):
                face_vertices = frozenset(
                        mgrp.vertex_indices[iel_grp, fvi]
                        for fvi in loc_face_vertices
                        )
                face_map.setdefault(face_vertices, []).append(
                        (igrp, iel_grp, fid))

    del face_vertices

    # }}}

    boundary_faces = [
            face_ids[0]
            for face_vertices, face_ids in six.iteritems(face_map)
            if len(face_ids) == 1]

    from pytools import flatten
    bdry_vertex_vol_nrs = sorted(set(flatten(six.iterkeys(face_map))))

    vol_to_bdry_vertices = np.empty(
            discr.mesh.vertices.shape[-1],
            discr.mesh.vertices.dtype)
    vol_to_bdry_vertices.fill(-1)
    vol_to_bdry_vertices[bdry_vertex_vol_nrs] = np.arange(
            len(bdry_vertex_vol_nrs))

    bdry_vertices = discr.mesh.vertices[:, bdry_vertex_vol_nrs]

    from meshmode.mesh import Mesh, SimplexElementGroup
    bdry_mesh_groups = []
    connection_data = {}

    for igrp, grp in enumerate(discr.groups):
        mgrp = grp.mesh_el_group
        group_boundary_faces = [
                (ibface_el, ibface_face)
                for ibface_group, ibface_el, ibface_face in boundary_faces
                if ibface_group == igrp]

        if not isinstance(mgrp, SimplexElementGroup):
            raise NotImplementedError("can only take boundary of "
                    "SimplexElementGroup-based meshes")

        # {{{ Preallocate arrays for mesh group

        ngroup_bdry_elements = len(group_boundary_faces)
        vertex_indices = np.empty(
                (ngroup_bdry_elements, mgrp.dim+1-1),
                mgrp.vertex_indices.dtype)

        bdry_unit_nodes = mp.warp_and_blend_nodes(mgrp.dim-1, mgrp.order)
        bdry_unit_nodes_01 = (bdry_unit_nodes + 1)*0.5

        vol_basis = mp.simplex_onb(mgrp.dim, mgrp.order)
        nbdry_unit_nodes = bdry_unit_nodes_01.shape[-1]
        nodes = np.empty(
                (discr.ambient_dim, ngroup_bdry_elements, nbdry_unit_nodes),
                dtype=np.float64)

        # }}}

        grp_face_vertex_indices = mgrp.face_vertex_indices()
        grp_vertex_unit_coordinates = mgrp.vertex_unit_coordinates()

        # batch by face_id

        batch_base = 0

        for face_id in range(len(grp_face_vertex_indices)):
            batch_boundary_el_numbers_in_grp = np.array(
                    [
                        ibface_el
                        for ibface_el, ibface_face in group_boundary_faces
                        if ibface_face == face_id],
                    dtype=np.intp)

            new_el_numbers = np.arange(
                    batch_base,
                    batch_base + len(batch_boundary_el_numbers_in_grp))

            # {{{ no per-element axes in these computations

            # Find boundary vertex indices
            loc_face_vertices = list(grp_face_vertex_indices[face_id])

            # Find unit nodes for boundary element
            face_vertex_unit_coordinates = \
                    grp_vertex_unit_coordinates[loc_face_vertices]

            # Find A, b such that A [e_1 e_2] + b = [r_1 r_2]
            # (Notation assumes that the volume is 3D and the face is 2D.
            # Code does not.)

            b = face_vertex_unit_coordinates[0]
            A = (  # noqa
                    face_vertex_unit_coordinates[1:]
                    - face_vertex_unit_coordinates[0]).T

            face_unit_nodes = (np.dot(A, bdry_unit_nodes_01).T + b).T

            resampling_mat = mp.resampling_matrix(
                    vol_basis,
                    face_unit_nodes, mgrp.unit_nodes)

            # }}}

            # {{{ build information for mesh element group

            # Find vertex_indices
            glob_face_vertices = mgrp.vertex_indices[
                    batch_boundary_el_numbers_in_grp][:, loc_face_vertices]
            vertex_indices[new_el_numbers] = \
                    vol_to_bdry_vertices[glob_face_vertices]

            # Find nodes
            nodes[:, new_el_numbers, :] = np.einsum(
                    "ij,dej->dei",
                    resampling_mat,
                    mgrp.nodes[:, batch_boundary_el_numbers_in_grp, :])

            # }}}

            connection_data[igrp, face_id] = _ConnectionBatchData(
                    group_source_element_indices=batch_boundary_el_numbers_in_grp,
                    group_target_element_indices=new_el_numbers,
                    A=A,
                    b=b,
                    )

            batch_base += len(batch_boundary_el_numbers_in_grp)

        bdry_mesh_group = SimplexElementGroup(
                mgrp.order, vertex_indices, nodes, unit_nodes=bdry_unit_nodes)
        bdry_mesh_groups.append(bdry_mesh_group)

    bdry_mesh = Mesh(bdry_vertices, bdry_mesh_groups)

    from meshmode.discretization import Discretization
    bdry_discr = Discretization(
            discr.cl_context, bdry_mesh, group_factory)

    connection = _build_boundary_connection(
            queue, discr, bdry_discr, connection_data)

    logger.info("building boundary connection: done")

    return bdry_mesh, bdry_discr, connection

# }}}


# {{{ refinement connection

def make_refinement_connection(refiner, coarse_discr):
    pass

# }}}

# vim: foldmethod=marker
