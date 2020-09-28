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
import numpy.linalg as la

import logging
logger = logging.getLogger(__name__)


def freeze_from_numpy(actx, array):
    return actx.freeze(actx.from_numpy(array))


def thaw_to_numpy(actx, array):
    return actx.to_numpy(actx.thaw(array))


# {{{ _make_cross_face_batches

def _make_cross_face_batches(actx,
        tgt_bdry_discr, src_bdry_discr,
        i_tgt_grp, i_src_grp,
        tgt_bdry_element_indices, src_bdry_element_indices):

    from meshmode.discretization.connection.direct import InterpolationBatch
    if tgt_bdry_discr.dim == 0:
        yield InterpolationBatch(
            from_group_index=i_src_grp,
            from_element_indices=freeze_from_numpy(actx, src_bdry_element_indices),
            to_element_indices=freeze_from_numpy(actx, tgt_bdry_element_indices),
            result_unit_nodes=src_bdry_discr.groups[i_src_grp].unit_nodes,
            to_element_face=None)
        return

    tgt_bdry_nodes = np.array([
        thaw_to_numpy(actx, ary[i_tgt_grp])[tgt_bdry_element_indices]
        for ary in tgt_bdry_discr.nodes()
        ])

    src_bdry_nodes = np.array([
        thaw_to_numpy(actx, ary[i_src_grp])[src_bdry_element_indices]
        for ary in src_bdry_discr.nodes()
        ])

    tol = 1e4 * np.finfo(tgt_bdry_nodes.dtype).eps

    src_mesh_grp = src_bdry_discr.mesh.groups[i_src_grp]
    src_grp = src_bdry_discr.groups[i_src_grp]

    dim = src_grp.dim
    ambient_dim, nelements, ntgt_unit_nodes = tgt_bdry_nodes.shape
    assert tgt_bdry_nodes.shape == src_bdry_nodes.shape

    # {{{ invert face map (using Gauss-Newton)

    initial_guess = np.mean(src_mesh_grp.vertex_unit_coordinates(), axis=0)
    src_unit_nodes = np.empty((dim, nelements, ntgt_unit_nodes))
    src_unit_nodes[:] = initial_guess.reshape(-1, 1, 1)

    import modepy as mp
    vdm = mp.vandermonde(src_grp.basis(), src_grp.unit_nodes)
    inv_t_vdm = la.inv(vdm.T)
    nsrc_funcs = len(src_grp.basis())

    def apply_map(unit_nodes):
        # unit_nodes: (dim, nelements, ntgt_unit_nodes)

        # basis_at_unit_nodes
        basis_at_unit_nodes = np.empty((nsrc_funcs, nelements, ntgt_unit_nodes))

        for i, f in enumerate(src_grp.basis()):
            basis_at_unit_nodes[i] = (
                    f(unit_nodes.reshape(dim, -1))
                    .reshape(nelements, ntgt_unit_nodes))

        intp_coeffs = np.einsum("fj,jet->fet", inv_t_vdm, basis_at_unit_nodes)

        # If we're interpolating 1, we had better get 1 back.
        one_deviation = np.abs(np.sum(intp_coeffs, axis=0) - 1)
        assert (one_deviation < tol).all(), np.max(one_deviation)

        return np.einsum("fet,aef->aet", intp_coeffs, src_bdry_nodes)

    def get_map_jacobian(unit_nodes):
        # unit_nodes: (dim, nelements, ntgt_unit_nodes)

        # basis_at_unit_nodes
        dbasis_at_unit_nodes = np.empty(
                (dim, nsrc_funcs, nelements, ntgt_unit_nodes))

        for i, df in enumerate(src_grp.grad_basis()):
            df_result = df(unit_nodes.reshape(dim, -1))

            for rst_axis, df_r in enumerate(df_result):
                dbasis_at_unit_nodes[rst_axis, i] = (
                        df_r.reshape(nelements, ntgt_unit_nodes))

        dintp_coeffs = np.einsum(
                "fj,rjet->rfet", inv_t_vdm, dbasis_at_unit_nodes)

        return np.einsum("rfet,aef->raet", dintp_coeffs, src_bdry_nodes)

    # {{{ test map applier and jacobian

    if 0:
        u = src_unit_nodes
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
        guess = apply_map(src_unit_nodes)
        goals = tgt_bdry_nodes

        from meshmode.discretization.visualization import draw_curve
        pt.figure(0)
        draw_curve(tgt_bdry_discr)
        pt.figure(1)
        draw_curve(src_bdry_discr)
        pt.figure(2)

        pt.plot(guess[0].reshape(-1), guess[1].reshape(-1), "or")
        pt.plot(goals[0].reshape(-1), goals[1].reshape(-1), "og")
        pt.plot(src_bdry_nodes[0].reshape(-1), src_bdry_nodes[1].reshape(-1), "xb")
        pt.show()

    # }}}

    logger.debug("_make_cross_face_batches: begin gauss-newton")

    niter = 0
    while True:
        resid = apply_map(src_unit_nodes) - tgt_bdry_nodes

        df = get_map_jacobian(src_unit_nodes)
        df_inv_resid = np.empty_like(src_unit_nodes)

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

            df_inv_resid = np.empty_like(src_unit_nodes)
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
                for t in range(ntgt_unit_nodes):
                    df_inv_resid[:, e, t], _, _, _ = \
                            la.lstsq(df[:, :, e, t].T, resid[:, e, t])

        src_unit_nodes = src_unit_nodes - df_inv_resid

        # {{{ visualize next guess

        if 0:
            import matplotlib.pyplot as pt
            guess = apply_map(src_unit_nodes)
            goals = tgt_bdry_nodes

            pt.plot(guess[0].reshape(-1), guess[1].reshape(-1), "rx")
            pt.plot(goals[0].reshape(-1), goals[1].reshape(-1), "go")
            pt.show()

        # }}}

        max_resid = np.max(np.abs(resid))

        if max_resid < tol:
            logger.debug("_make_cross_face_batches: gauss-newton: done, "
                    "final residual: %g" % max_resid)
            break

        niter += 1
        if niter > 10:
            raise RuntimeError("Gauss-Newton (for finding opposite-face reference "
                    "coordinates) did not converge (residual: %g)" % max_resid)

    # }}}

    # {{{ find groups of src_unit_nodes

    done_elements = np.zeros(nelements, dtype=np.bool)
    while True:
        todo_elements, = np.where(~done_elements)
        if not len(todo_elements):
            return

        template_unit_nodes = src_unit_nodes[:, todo_elements[0], :]

        unit_node_dist = np.max(np.max(np.abs(
                src_unit_nodes[:, todo_elements, :]
                - template_unit_nodes.reshape(dim, 1, -1)),
                axis=2), axis=0)

        close_els = todo_elements[unit_node_dist < tol]
        done_elements[close_els] = True

        from meshmode.discretization.connection.direct import InterpolationBatch
        yield InterpolationBatch(
                from_group_index=i_src_grp,
                from_element_indices=freeze_from_numpy(
                    actx, src_bdry_element_indices[close_els]),
                to_element_indices=freeze_from_numpy(
                    actx, tgt_bdry_element_indices[close_els]),
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


def _make_bdry_el_lookup_table(actx, connection, igrp):
    """Given a volume-to-boundary connection as *connection*, return
    a table of shape ``(from_nelements, nfaces)`` to look up the
    element number of the boundary element for that face.
    """
    from_nelements = connection.from_discr.groups[igrp].nelements
    from_nfaces = connection.from_discr.mesh.groups[igrp].nfaces

    iel_lookup = np.full((from_nelements, from_nfaces), -1,
            dtype=connection.from_discr.mesh.element_id_dtype)

    for ibatch, batch in enumerate(connection.groups[igrp].batches):
        from_element_indices = thaw_to_numpy(actx, batch.from_element_indices)
        iel_lookup[from_element_indices, batch.to_element_face] = \
                thaw_to_numpy(actx, batch.to_element_indices)

    return iel_lookup

# }}}


# {{{ make_opposite_face_connection

def make_opposite_face_connection(actx, volume_to_bdry_conn):
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

    # a list of batches for each group
    groups = [[] for i_tgt_grp in range(ngrps)]

    for i_src_grp in range(ngrps):
        src_grp_el_lookup = _make_bdry_el_lookup_table(
                actx, volume_to_bdry_conn, i_src_grp)

        for i_tgt_grp in range(ngrps):
            vbc_tgt_grp_batches = volume_to_bdry_conn.groups[i_tgt_grp].batches

            adj = vol_mesh.facial_adjacency_groups[i_tgt_grp][i_src_grp]

            for i_face_tgt in range(vol_mesh.groups[i_tgt_grp].nfaces):
                vbc_tgt_grp_face_batch = _find_ibatch_for_face(
                        vbc_tgt_grp_batches, i_face_tgt)

                # {{{ index wrangling

                # The elements in the adjacency group will be a subset of
                # the elements in the restriction interpolation batch:
                # Imagine an inter-group boundary. The volume-to-boundary
                # connection will include all faces as targets, whereas
                # there will be separate adjacency groups for intra- and
                # inter-group connections.

                adj_tgt_flags = adj.element_faces == i_face_tgt
                adj_els = adj.elements[adj_tgt_flags]
                if adj_els.size == 0:
                    # NOTE: this case can happen for inter-group boundaries
                    # when all elements are adjacent on the same face
                    # index, so all other ones will be empty
                    continue

                vbc_els = thaw_to_numpy(actx,
                        vbc_tgt_grp_face_batch.from_element_indices)

                if len(adj_els) == len(vbc_els):
                    # Same length: assert (below) that the two use the same
                    # ordering.
                    vbc_used_els = slice(None)

                else:
                    # Genuine subset: figure out an index mapping.
                    vbc_els_sort_idx = np.argsort(vbc_els)
                    vbc_used_els = vbc_els_sort_idx[np.searchsorted(
                        vbc_els, adj_els, sorter=vbc_els_sort_idx
                        )]

                assert np.array_equal(vbc_els[vbc_used_els], adj_els)

                # find to_element_indices

                tgt_bdry_element_indices = thaw_to_numpy(
                        actx,
                        vbc_tgt_grp_face_batch.to_element_indices
                        )[vbc_used_els]

                # find from_element_indices

                src_vol_element_indices = adj.neighbors[adj_tgt_flags]
                src_element_faces = adj.neighbor_faces[adj_tgt_flags]

                src_bdry_element_indices = src_grp_el_lookup[
                        src_vol_element_indices, src_element_faces]

                # }}}

                # {{{ visualization (for debugging)

                if 0:
                    print("TVE", adj.elements[adj_tgt_flags])
                    print("TBE", tgt_bdry_element_indices)
                    print("FVE", src_vol_element_indices)
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

                batches = _make_cross_face_batches(actx,
                        bdry_discr, bdry_discr,
                        i_tgt_grp, i_src_grp,
                        tgt_bdry_element_indices,
                        src_bdry_element_indices)
                groups[i_tgt_grp].extend(batches)

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


# {{{ make_partition_connection

def make_partition_connection(actx, *, local_bdry_conn, i_local_part,
        remote_bdry_discr, remote_group_infos):
    """
    Connects ``local_bdry_conn`` to a neighboring partition.

    :arg local_bdry_conn: A :class:`DiscretizationConnection` of the local
        partition.
    :arg i_local_part: The partition number of the local partition.
    :arg remote_bdry_discr: A :class:`~meshmode.discretization.Discretization`
        of the boundary of the remote partition.
    :arg remote_group_infos: An array of
        :class:`meshmode.distributed.RemoteGroupInfo` instances, one per remote
        volume element group.
    :returns: A :class:`DirectDiscretizationConnection` that performs data
        exchange across faces from the remote partition to partition `i_local_part`.

    .. versionadded:: 2017.1

    .. warning:: Interface is not final.
    """

    from meshmode.mesh.processing import find_group_indices
    from meshmode.discretization.connection import (
            DirectDiscretizationConnection, DiscretizationConnectionElementGroup)

    local_vol_groups = local_bdry_conn.from_discr.mesh.groups

    part_batches = [[] for _ in local_vol_groups]

    assert len(local_vol_groups) == len(local_bdry_conn.to_discr.groups)

    # We need a nested loop over remote and local groups here.
    # The code assumes that there is the same number of volume and surface groups.
    #
    # A weak reason to choose remote as the outer loop is because
    # InterPartitionAdjacency refers to neighbors by global volume element
    # numbers, and we only have enough information to resolve those to (group,
    # group_local_el_nr) for local elements (whereas we have no information
    # about remote volume elements).
    #
    # (See the find_group_indices below.)

    for rgi in remote_group_infos:
        rem_ipag = rgi.inter_partition_adj_group

        indices = (i_local_part == rem_ipag.neighbor_partitions)
        if not np.any(indices):
            # Skip because remote group is not connected to i_local_part.
            continue

        i_remote_vol_elems = rem_ipag.elements[indices]
        i_remote_faces = rem_ipag.element_faces[indices]
        i_local_vol_elems = rem_ipag.partition_neighbors[indices]
        i_local_faces = rem_ipag.neighbor_faces[indices]

        del indices

        i_local_grps = find_group_indices(local_vol_groups, i_local_vol_elems)

        # {{{ make remote_vol_to_bdry

        remote_approx_vol_nelements = np.max(rgi.vol_elem_indices)+1
        remote_approx_nfaces = np.max(rgi.bdry_faces)+1
        remote_vol_to_bdry = np.full(
                (remote_approx_vol_nelements, remote_approx_nfaces),
                -1, dtype=remote_bdry_discr.mesh.element_id_dtype)
        remote_vol_to_bdry[rgi.vol_elem_indices, rgi.bdry_faces] = \
                rgi.bdry_elem_indices

        # }}}

        for i_local_grp in np.unique(i_local_grps):

            # {{{ come up with matched_{local,remote}_bdry_el_indices

            local_vol_to_bdry = _make_bdry_el_lookup_table(actx, local_bdry_conn,
                        i_local_grp)

            local_indices = np.where(i_local_grps == i_local_grp)[0]
            if len(local_indices) == 0:
                continue

            local_grp_vol_elems = (
                    i_local_vol_elems[local_indices]
                    - local_vol_groups[i_local_grp].element_nr_base)
            # These are group-local.
            remote_grp_vol_elems = i_remote_vol_elems[local_indices]

            matched_local_bdry_el_indices = local_vol_to_bdry[
                    local_grp_vol_elems,
                    i_local_faces[local_indices]]
            assert (matched_local_bdry_el_indices >= 0).all()
            matched_remote_bdry_el_indices = remote_vol_to_bdry[
                    remote_grp_vol_elems,
                    i_remote_faces[local_indices]]
            assert (matched_remote_bdry_el_indices >= 0).all()

            # }}}

            grp_batches = _make_cross_face_batches(actx,
                        local_bdry_conn.to_discr, remote_bdry_discr,
                        i_local_grp, rem_ipag.igroup,
                        matched_local_bdry_el_indices,
                        matched_remote_bdry_el_indices)

            part_batches[i_local_grp].extend(grp_batches)

    return DirectDiscretizationConnection(
            from_discr=remote_bdry_discr,
            to_discr=local_bdry_conn.to_discr,
            groups=[DiscretizationConnectionElementGroup(batches=grp_batches)
                        for grp_batches in part_batches],
            is_surjective=True)

# }}}


# vim: foldmethod=marker
