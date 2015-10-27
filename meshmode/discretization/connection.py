from __future__ import division, print_function, absolute_import

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

import six
from six.moves import range, zip

import numpy as np
import numpy.linalg as la
import modepy as mp
import pyopencl as cl
import pyopencl.array  # noqa
from pytools import memoize_method, memoize_in, Record

import logging
logger = logging.getLogger(__name__)


__doc__ = """
.. autoclass:: DiscretizationConnection

.. autofunction:: make_same_mesh_connection

.. autofunction:: make_face_restriction

.. autofunction:: make_opposite_face_connection

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

    .. attribute:: from_group_index

        An integer indicating from which element group in the *from* discretization
        the data should be interpolated.

    .. attribute:: from_element_indices

        ``element_id_t [nelements]``. (a :class:`pyopencl.array.Array`)
        This contains the (group-local) element index (relative to
        :attr:`from_group_index` from which this "*to*" element's data will be
        interpolated.

    .. attribute:: to_element_indices

        ``element_id_t [nelements]``. (a :class:`pyopencl.array.Array`)
        This contains the (group-local) element index to which this "*to*"
        element's data will be interpolated.

    .. attribute:: result_unit_nodes

        A :class:`numpy.ndarray` of shape
        ``(from_group.dim,to_group.nelements,to_group.nunit_nodes)``
        storing the coordinates of the nodes (in unit coordinates
        of the *from* reference element) from which the node
        locations of this element should be interpolated.

    .. autoattribute:: nelements

    .. attribute:: to_element_face

        *int* or *None*. (a :class:`pyopencl.array.Array` if existent) If this
        interpolation batch targets interpolation *to* a face, then this number
        captures the face number (on all elements referenced by
        :attr:`from_element_indices` to which this batch interpolates. (Since
        there is a fixed set of "from" unit nodes per batch, one batch will
        always go to a single face index.)
    """

    def __init__(self, from_group_index, from_element_indices,
            to_element_indices, result_unit_nodes, to_element_face):
        self.from_group_index = from_group_index
        self.from_element_indices = from_element_indices
        self.to_element_indices = to_element_indices
        self.result_unit_nodes = result_unit_nodes
        self.to_element_face = to_element_face

    @property
    def nelements(self):
        return len(self.from_element_indices)


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

        a list of :class:`DiscretizationConnectionElementGroup`
        instances, with a one-to-one correspondence to the groups in
        :attr:`to_discr`.

    .. automethod:: __call__

    .. automethod:: full_resample_matrix

    """

    def __init__(self, from_discr, to_discr, groups):
        if from_discr.cl_context != to_discr.cl_context:
            raise ValueError("from_discr and to_discr must live in the "
                    "same OpenCL context")

        self.cl_context = from_discr.cl_context

        if from_discr.mesh.vertex_id_dtype != to_discr.mesh.vertex_id_dtype:
            raise ValueError("from_discr and to_discr must agree on the "
                    "vertex_id_dtype")

        if from_discr.mesh.element_id_dtype != to_discr.mesh.element_id_dtype:
            raise ValueError("from_discr and to_discr must agree on the "
                    "element_id_dtype")

        self.cl_context = from_discr.cl_context

        self.from_discr = from_discr
        self.to_discr = to_discr
        self.groups = groups

    @memoize_method
    def _resample_matrix(self, to_group_index, ibatch_index):
        import modepy as mp
        ibatch = self.groups[to_group_index].batches[ibatch_index]
        from_grp = self.from_discr.groups[ibatch.from_group_index]

        result = mp.resampling_matrix(
                mp.simplex_onb(self.from_discr.dim, from_grp.order),
                ibatch.result_unit_nodes, from_grp.unit_nodes)

        with cl.CommandQueue(self.cl_context) as queue:
            return cl.array.to_device(queue, result).with_queue(None)

    @memoize_method
    def _resample_point_pick_indices(self, to_group_index, ibatch_index,
            tol_multiplier=None):
        """If :meth:`_resample_matrix` *R* is a row subset of a permutation matrix *P*,
        return the index subset I so that, loosely, ``x[I] == R @ x``.

        Will return *None* if no such index array exists, or a
        :class:`pyopencl.array.Array` containing the index subset.
        """

        with cl.CommandQueue(self.cl_context) as queue:
            mat = self._resample_matrix(to_group_index, ibatch_index).get(
                    queue=queue)

        nrows, ncols = mat.shape
        result = np.zeros(nrows, dtype=self.to_discr.mesh.element_id_dtype)

        if tol_multiplier is None:
            tol_multiplier = 50

        tol = np.finfo(mat.dtype).eps * tol_multiplier

        for irow in range(nrows):
            one_indices, = np.where(np.abs(mat[irow] - 1) < tol)
            zero_indices, = np.where(np.abs(mat[irow]) < tol)

            if len(one_indices) != 1:
                return None
            if len(zero_indices) != ncols - 1:
                return None

            one_index, = one_indices
            result[irow] = one_index

        with cl.CommandQueue(self.cl_context) as queue:
            return cl.array.to_device(queue, result).with_queue(None)

    def full_resample_matrix(self, queue):
        """Build a dense matrix representing this discretization connection.

        .. warning::

            On average, this will be exceedingly expensive (:math:`O(N^2)` in
            the number *N* of discretization points) in terms of memory usage
            and thus not what you'd typically want.
        """

        @memoize_in(self, "oversample_mat_knl")
        def knl():
            import loopy as lp
            knl = lp.make_kernel(
                """{[k,i,j]:
                    0<=k<nelements and
                    0<=i<n_to_nodes and
                    0<=j<n_from_nodes}""",
                "result[itgt_base + to_element_indices[k]*n_to_nodes + i, \
                        isrc_base + from_element_indices[k]*n_from_nodes + j] \
                    = resample_mat[i, j]",
                [
                    lp.GlobalArg("result", None,
                        shape="nnodes_tgt, nnodes_src",
                        offset=lp.auto),
                    lp.ValueArg("itgt_base,isrc_base", np.int32),
                    lp.ValueArg("nnodes_tgt,nnodes_src", np.int32),
                    "...",
                    ],
                name="oversample_mat")

            knl = lp.split_iname(knl, "i", 16, inner_tag="l.0")
            return lp.tag_inames(knl, dict(k="g.0"))

        result = cl.array.zeros(
                queue,
                (self.to_discr.nnodes, self.from_discr.nnodes),
                dtype=self.to_discr.real_dtype)

        for i_tgrp, (tgrp, cgrp) in enumerate(
                zip(self.to_discr.groups, self.groups)):
            for i_batch, batch in enumerate(cgrp.batches):
                if len(batch.from_element_indices):
                    if not len(batch.from_element_indices):
                        continue

                    sgrp = self.from_discr.groups[batch.from_group_index]

                    knl()(queue,
                            resample_mat=self._resample_matrix(i_tgrp, i_batch),
                            result=result,
                            itgt_base=tgrp.node_nr_base,
                            isrc_base=sgrp.node_nr_base,
                            from_element_indices=batch.from_element_indices,
                            to_element_indices=batch.to_element_indices)

        return result

    def __call__(self, queue, vec):
        @memoize_in(self, "resample_by_mat_knl")
        def mat_knl():
            import loopy as lp
            knl = lp.make_kernel(
                """{[k,i,j]:
                    0<=k<nelements and
                    0<=i<n_to_nodes and
                    0<=j<n_from_nodes}""",
                "result[to_element_indices[k], i] \
                    = sum(j, resample_mat[i, j] \
                    * vec[from_element_indices[k], j])",
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
                name="resample_by_mat")

            knl = lp.split_iname(knl, "i", 16, inner_tag="l.0")
            return lp.tag_inames(knl, dict(k="g.0"))

        @memoize_in(self, "resample_by_picking_knl")
        def pick_knl():
            import loopy as lp
            knl = lp.make_kernel(
                """{[k,i,j]:
                    0<=k<nelements and
                    0<=i<n_to_nodes}""",
                "result[to_element_indices[k], i] \
                    = vec[from_element_indices[k], pick_list[i]]",
                [
                    lp.GlobalArg("result", None,
                        shape="nelements_result, n_to_nodes",
                        offset=lp.auto),
                    lp.GlobalArg("vec", None,
                        shape="nelements_vec, n_from_nodes",
                        offset=lp.auto),
                    lp.ValueArg("nelements_result", np.int32),
                    lp.ValueArg("nelements_vec", np.int32),
                    lp.ValueArg("n_from_nodes", np.int32),
                    "...",
                    ],
                name="resample_by_picking")

            knl = lp.split_iname(knl, "i", 16, inner_tag="l.0")
            return lp.tag_inames(knl, dict(k="g.0"))

        if not isinstance(vec, cl.array.Array):
            return vec

        result = self.to_discr.empty(dtype=vec.dtype)

        if vec.shape != (self.from_discr.nnodes,):
            raise ValueError("invalid shape of incoming resampling data")

        for i_tgrp, (tgrp, sgrp, cgrp) in enumerate(
                zip(self.to_discr.groups, self.from_discr.groups, self.groups)):
            for i_batch, batch in enumerate(cgrp.batches):
                if not len(batch.from_element_indices):
                    continue

                point_pick_indices = self._resample_point_pick_indices(
                        i_tgrp, i_batch)

                if point_pick_indices is None:
                    mat_knl()(queue,
                            resample_mat=self._resample_matrix(i_tgrp, i_batch),
                            result=tgrp.view(result), vec=sgrp.view(vec),
                            from_element_indices=batch.from_element_indices,
                            to_element_indices=batch.to_element_indices)

                else:
                    pick_knl()(queue,
                            pick_list=point_pick_indices,
                            result=tgrp.view(result), vec=sgrp.view(vec),
                            from_element_indices=batch.from_element_indices,
                            to_element_indices=batch.to_element_indices)

        return result

    # }}}


# {{{ same-mesh constructor

def make_same_mesh_connection(to_discr, from_discr):
    if from_discr.mesh is not to_discr.mesh:
        raise ValueError("from_discr and to_discr must be based on "
                "the same mesh")

    assert to_discr.cl_context == from_discr.cl_context

    with cl.CommandQueue(to_discr.cl_context) as queue:
        groups = []
        for igrp, (fgrp, tgrp) in enumerate(zip(from_discr.groups, to_discr.groups)):
            all_elements = cl.array.arange(queue,
                    fgrp.nelements,
                    dtype=np.intp).with_queue(None)
            ibatch = InterpolationBatch(
                    from_group_index=igrp,
                    from_element_indices=all_elements,
                    to_element_indices=all_elements,
                    result_unit_nodes=tgrp.unit_nodes,
                    to_element_face=None)

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
                        from_group_index=igrp,
                        from_element_indices=cl.array.to_device(
                            queue,
                            vol_grp.mesh_el_group.element_nr_base
                            + data.group_source_element_indices)
                        .with_queue(None),
                        to_element_indices=cl.array.to_device(
                            queue,
                            bdry_grp.mesh_el_group.element_nr_base
                            + data.group_target_element_indices)
                        .with_queue(None),
                        result_unit_nodes=result_unit_nodes,
                        to_element_face=face_id
                        ))

        connection_groups.append(
                DiscretizationConnectionElementGroup(
                    connection_batches))

    return DiscretizationConnection(
            vol_discr, bdry_discr, connection_groups)


# {{{ pull together boundary vertices

def _get_face_vertices(mesh, boundary_tag):
    # a set of volume vertex numbers
    bdry_vertex_vol_nrs = set()

    if boundary_tag is not None:
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
        # For interior faces, this is likely every vertex in the book.
        # Don't ever bother trying to cut the list down.

        return np.arange(mesh.nvertices, dtype=np.intp)

# }}}


def make_face_restriction(discr, group_factory, boundary_tag):
    """Create a mesh, a discretization and a connection to restrict
    a function on *discr* to its values on the edges of element faces
    denoted by *boundary_tag*.

    :arg boundary_tag: The boundary tag for which to create a face
        restriction. May be *None* to indicate interior faces.

    :return: a tuple ``(bdry_mesh, bdry_discr, connection)``
    """

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

    from meshmode.mesh import Mesh, SimplexElementGroup
    bdry_mesh_groups = []
    connection_data = {}

    btag_bit = discr.mesh.boundary_tag_bit(boundary_tag)

    for igrp, (grp, fagrp_map) in enumerate(
            zip(discr.groups, discr.mesh.facial_adjacency_groups)):

        mgrp = grp.mesh_el_group

        if not isinstance(mgrp, SimplexElementGroup):
            raise NotImplementedError("can only take boundary of "
                    "SimplexElementGroup-based meshes")

        # {{{ pull together per-group face lists

        group_boundary_faces = []

        if boundary_tag is not None:
            bdry_grp = fagrp_map.get(None)
            if bdry_grp is not None:
                nb_el_bits = -bdry_grp.neighbors
                face_relevant_flags = (nb_el_bits & btag_bit) != 0

                group_boundary_faces.extend(
                            zip(
                                bdry_grp.elements[face_relevant_flags],
                                bdry_grp.element_faces[face_relevant_flags]))

        else:
            for fagrp in six.itervalues(fagrp_map):
                if fagrp.ineighbor_group is None:
                    # boundary faces -> not looking for those
                    continue

                group_boundary_faces.extend(
                        zip(fagrp.elements, fagrp.element_faces))

        # }}}

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

    with cl.CommandQueue(discr.cl_context) as queue:
        connection = _build_boundary_connection(
                queue, discr, bdry_discr, connection_data)

    logger.info("building face restriction: done")

    return bdry_mesh, bdry_discr, connection

# }}}


# {{{ opposite-face connection

def _make_cross_face_batches(
        queue, vol_discr, bdry_discr,
        i_tgt_grp, i_src_grp,
        i_face_tgt,
        adj_grp,
        vbc_tgt_grp_face_batch, src_grp_el_lookup):

    # {{{ index wrangling

    # Assert that the adjacency group and the restriction
    # interpolation batch and the adjacency group have the same
    # element ordering.

    adj_grp_tgt_flags = adj_grp.element_faces == i_face_tgt

    assert (
            np.array_equal(
                adj_grp.elements[adj_grp_tgt_flags],
                vbc_tgt_grp_face_batch.from_element_indices
                .get(queue=queue)))

    # find to_element_indices

    to_bdry_element_indices = (
            vbc_tgt_grp_face_batch.to_element_indices
            .get(queue=queue))

    # find from_element_indices

    from_vol_element_indices = adj_grp.neighbors[adj_grp_tgt_flags]
    from_element_faces = adj_grp.neighbor_faces[adj_grp_tgt_flags]

    from_bdry_element_indices = src_grp_el_lookup[
            from_vol_element_indices, from_element_faces]

    # }}}

    # {{{ visualization (for debugging)

    if 0:
        print("TVE", adj_grp.elements[adj_grp_tgt_flags])
        print("TBE", to_bdry_element_indices)
        print("FVE", from_vol_element_indices)
        from meshmode.mesh.visualization import draw_2d_mesh
        import matplotlib.pyplot as pt
        draw_2d_mesh(vol_discr.mesh, draw_element_numbers=True,
                set_bounding_box=True,
                draw_vertex_numbers=False,
                draw_face_numbers=True,
                fill=None)
        pt.figure()

        draw_2d_mesh(bdry_discr.mesh, draw_element_numbers=True,
                set_bounding_box=True,
                draw_vertex_numbers=False,
                draw_face_numbers=True,
                fill=None)

        pt.show()
    # }}}

    # {{{ invert face map (using Gauss-Newton)

    to_bdry_nodes = (
            bdry_discr.groups[i_tgt_grp].view(bdry_discr.nodes())
            .get(queue=queue)
            [:, to_bdry_element_indices])

    tol = 1e3 * np.finfo(to_bdry_nodes.dtype).eps

    from_mesh_grp = bdry_discr.mesh.groups[i_src_grp]
    from_grp = bdry_discr.groups[i_src_grp]

    dim = from_grp.dim
    ambient_dim, nelements, nto_unit_nodes = to_bdry_nodes.shape

    initial_guess = np.mean(from_mesh_grp.vertex_unit_coordinates(), axis=0)
    from_unit_nodes = np.empty((dim, nelements, nto_unit_nodes))
    from_unit_nodes[:] = initial_guess.reshape(-1, 1, 1)

    import modepy as mp
    from_vdm = mp.vandermonde(from_grp.basis(), from_grp.unit_nodes)
    from_inv_t_vdm = la.inv(from_vdm.T)
    from_nfuncs = len(from_grp.basis())

    # (ambient_dim, nelements, nfrom_unit_nodes)
    from_bdry_nodes = (
            bdry_discr.groups[i_src_grp].view(bdry_discr.nodes())
            .get(queue=queue)
            [:, from_bdry_element_indices])

    def apply_map(unit_nodes):
        # unit_nodes: (dim, nelements, nto_unit_nodes)

        # basis_at_unit_nodes
        basis_at_unit_nodes = np.empty((from_nfuncs, nelements, nto_unit_nodes))

        for i, f in enumerate(from_grp.basis()):
            basis_at_unit_nodes[i] = (
                    f(unit_nodes.reshape(dim, -1))
                    .reshape(nelements, nto_unit_nodes))

        intp_coeffs = np.einsum("fj,jet->fet", from_inv_t_vdm, basis_at_unit_nodes)

        # If we're interpolating 1, we had better get 1 back.
        one_deviation = np.abs(np.sum(intp_coeffs, axis=0) - 1)
        assert (one_deviation < tol).all(), np.max(one_deviation)

        return np.einsum("fet,aef->aet", intp_coeffs, from_bdry_nodes)

    def get_map_jacobian(unit_nodes):
        # unit_nodes: (dim, nelements, nto_unit_nodes)

        # basis_at_unit_nodes
        dbasis_at_unit_nodes = np.empty(
                (dim, from_nfuncs, nelements, nto_unit_nodes))

        for i, df in enumerate(from_grp.grad_basis()):
            df_result = df(unit_nodes.reshape(dim, -1))

            for rst_axis, df_r in enumerate(df_result):
                dbasis_at_unit_nodes[rst_axis, i] = (
                        df_r.reshape(nelements, nto_unit_nodes))

        dintp_coeffs = np.einsum(
                "fj,rjet->rfet", from_inv_t_vdm, dbasis_at_unit_nodes)

        return np.einsum("rfet,aef->raet", dintp_coeffs, from_bdry_nodes)

    # {{{ test map applier and jacobian

    if 0:
        u = from_unit_nodes
        f = apply_map(u)
        for h in [1e-1, 1e-2]:
            du = h*np.random.randn(*u.shape)

            f_2 = apply_map(u+du)

            jf = get_map_jacobian(u)

            f2_2 = f + np.einsum("raet,ret->aet", jf, du)

            print(h, la.norm((f_2-f2_2).ravel()))

    # }}}

    # {{{ visualize initial guess

    if 0:
        import matplotlib.pyplot as pt
        guess = apply_map(from_unit_nodes)
        goals = to_bdry_nodes

        from meshmode.discretization.visualization import draw_curve
        draw_curve(bdry_discr)

        pt.plot(guess[0].reshape(-1), guess[1].reshape(-1), "or")
        pt.plot(goals[0].reshape(-1), goals[1].reshape(-1), "og")
        pt.plot(from_bdry_nodes[0].reshape(-1), from_bdry_nodes[1].reshape(-1), "o",
                color="purple")
        pt.show()

    # }}}

    logger.info("make_opposite_face_connection: begin gauss-newton")

    niter = 0
    while True:
        resid = apply_map(from_unit_nodes) - to_bdry_nodes

        df = get_map_jacobian(from_unit_nodes)
        df_inv_resid = np.empty_like(from_unit_nodes)
        # FIXME: Should look for a way to batch this
        for e in range(nelements):
            for t in range(nto_unit_nodes):
                df_inv_resid[:, e, t], _, _, _ = \
                        la.lstsq(df[:, :, e, t].T, resid[:, e, t])

        from_unit_nodes = from_unit_nodes - df_inv_resid

        max_resid = np.max(np.abs(resid))
        logger.debug("gauss-newton residual: %g" % max_resid)

        if max_resid < tol:
            logger.info("make_opposite_face_connection: gauss-newton: done, "
                    "final residual: %g" % max_resid)
            break

        niter += 1
        if niter > 10:
            raise RuntimeError("Gauss-Newton (for finding opposite-face reference "
                    "coordinates) did not converge")

    # }}}

    # {{{ find groups of from_unit_nodes

    def to_dev(ary):
        return cl.array.to_device(queue, ary, array_queue=None)

    done_elements = np.zeros(nelements, dtype=np.bool)
    while True:
        todo_elements, = np.where(~done_elements)
        if not len(todo_elements):
            return

        template_unit_nodes = from_unit_nodes[:, todo_elements[0], :]

        unit_node_dist = np.max(np.max(np.abs(
                from_unit_nodes[:, todo_elements, :]
                -
                template_unit_nodes.reshape(dim, 1, -1)),
                axis=2), axis=0)

        close_els = todo_elements[unit_node_dist < tol]
        done_elements[close_els] = True

        unit_node_dist = np.max(np.max(np.abs(
                from_unit_nodes[:, todo_elements, :]
                -
                template_unit_nodes.reshape(dim, 1, -1)),
                axis=2), axis=0)

        yield InterpolationBatch(
                from_group_index=i_src_grp,
                from_element_indices=to_dev(from_bdry_element_indices[close_els]),
                to_element_indices=to_dev(to_bdry_element_indices[close_els]),
                result_unit_nodes=template_unit_nodes,
                to_element_face=None)

    # }}}


def _find_ibatch_for_face(vbc_tgt_grp_batches, iface):
    vbc_tgt_grp_face_batches = [
            batch
            for batch in vbc_tgt_grp_batches
            if batch.to_element_face == iface]

    assert len(vbc_tgt_grp_face_batches) == 1

    vbc_tgt_grp_face_batch, = vbc_tgt_grp_face_batches

    return vbc_tgt_grp_face_batch


def _make_el_lookup_table(queue, connection, igrp):
    from_nelements = connection.from_discr.groups[igrp].nelements
    from_nfaces = connection.from_discr.mesh.groups[igrp].nfaces

    iel_lookup = np.empty((from_nelements, from_nfaces),
            dtype=connection.from_discr.mesh.element_id_dtype)
    iel_lookup.fill(-1)

    for ibatch, batch in enumerate(connection.groups[igrp].batches):
        from_element_indices = batch.from_element_indices.get(queue=queue)
        iel_lookup[from_element_indices, batch.to_element_face] = \
                batch.to_element_indices.get(queue=queue)

    return iel_lookup


def make_opposite_face_connection(volume_to_bdry_conn):
    """Given a boundary restriction connection *volume_to_bdry_conn*,
    return a :class:`DiscretizationConnection` that performs data
    exchange across opposite faces.
    """

    vol_discr = volume_to_bdry_conn.from_discr
    vol_mesh = vol_discr.mesh
    bdry_discr = volume_to_bdry_conn.to_discr

    # make sure we were handed a volume-to-boundary connection
    for i_tgrp, conn_grp in enumerate(volume_to_bdry_conn.groups):
        for batch in conn_grp.batches:
            assert batch.from_group_index == i_tgrp
            assert batch.to_element_face is not None

    ngrps = len(volume_to_bdry_conn.groups)
    assert ngrps == len(vol_discr.groups)
    assert ngrps == len(bdry_discr.groups)

    # One interpolation batch in this connection corresponds
    # to a key (i_tgt_grp,)  (i_src_grp, i_face_tgt,)

    with cl.CommandQueue(vol_discr.cl_context) as queue:
        # a list of batches for each group
        groups = [[] for i_tgt_grp in range(ngrps)]

        for i_src_grp in range(ngrps):
            src_grp_el_lookup = _make_el_lookup_table(
                    queue, volume_to_bdry_conn, i_src_grp)

            for i_tgt_grp in range(ngrps):
                vbc_tgt_grp_batches = volume_to_bdry_conn.groups[i_tgt_grp].batches

                adj_grp = vol_mesh.facial_adjacency_groups[i_tgt_grp][i_src_grp]

                for i_face_tgt in range(vol_mesh.groups[i_tgt_grp].nfaces):
                    vbc_tgt_grp_face_batch = _find_ibatch_for_face(
                            vbc_tgt_grp_batches, i_face_tgt)

                    groups[i_tgt_grp].extend(
                        _make_cross_face_batches(
                            queue, vol_discr, bdry_discr,
                            i_tgt_grp, i_src_grp,
                            i_face_tgt,
                            adj_grp,
                            vbc_tgt_grp_face_batch, src_grp_el_lookup))

    return DiscretizationConnection(
            from_discr=bdry_discr,
            to_discr=bdry_discr,
            groups=[
                DiscretizationConnectionElementGroup(batches=batches)
                for batches in groups])

# }}}


# {{{ refinement connection

def make_refinement_connection(refiner, coarse_discr):
    pass

# }}}

# vim: foldmethod=marker
