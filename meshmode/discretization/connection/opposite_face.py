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

from six.moves import range

import numpy as np
import numpy.linalg as la
import pyopencl as cl
import pyopencl.array  # noqa

import logging
logger = logging.getLogger(__name__)


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
            # FIXME: This should view-then-transfer (but PyOpenCL doesn't do
            # non-contiguous transfers for now).
            bdry_discr.groups[i_tgt_grp].view(
                bdry_discr.nodes().get(queue=queue))
            [:, to_bdry_element_indices])

    tol = 1e4 * np.finfo(to_bdry_nodes.dtype).eps

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
            # FIXME: This should view-then-transfer (but PyOpenCL doesn't do
            # non-contiguous transfers for now).
            bdry_discr.groups[i_src_grp].view(
                bdry_discr.nodes().get(queue=queue))
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

        # For the 1D/2D accelerated versions, we'll use the normal
        # equations and Cramer's rule. If you're looking for high-end
        # numerics, look no further than meshmode.

        if dim == 1:
            # A is df.T
            ata = np.einsum("iket,jket->ijet", df, df)
            atb = np.einsum("iket,ket->iet", df, resid)

            df_inv_resid = atb / ata[0, 0]

        elif dim == 2:
            # A is df.T
            ata = np.einsum("iket,jket->ijet", df, df)
            atb = np.einsum("iket,ket->iet", df, resid)

            det = ata[0, 0]*ata[1, 1] - ata[0, 1]*ata[1, 0]

            df_inv_resid = np.empty_like(from_unit_nodes)
            df_inv_resid[0] = 1/det * (ata[1, 1] * atb[0] - ata[1, 0]*atb[1])
            df_inv_resid[1] = 1/det * (-ata[0, 1] * atb[0] + ata[0, 0]*atb[1])

        else:
            # The boundary of a 3D mesh is 2D, so that's the
            # highest-dimensional case we genuinely care about.
            #
            # This stinks, performance-wise, because it's not vectorized.
            # But we'll only hit it for boundaries of 4+D meshes, in which
            # case... good luck. :)
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

        from meshmode.discretization.connection import InterpolationBatch
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
    return a :class:`DirectDiscretizationConnection` that performs data
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

    from meshmode.discretization.connection import (
            DirectDiscretizationConnection, DiscretizationConnectionElementGroup)
    return DirectDiscretizationConnection(
            from_discr=bdry_discr,
            to_discr=bdry_discr,
            groups=[
                DiscretizationConnectionElementGroup(batches=batches)
                for batches in groups],
            is_surjective=True)

# }}}

# vim: foldmethod=marker
