from __future__ import division, absolute_import, print_function

__copyright__ = """
Copyright (C) 2017 Ellis Hoag
Copyright (C) 2017 Andreas Kloeckner
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

from six.moves import range
import numpy as np
import numpy.linalg as la
import pyopencl as cl
import pyopencl.array  # noqa
import pyopencl.clmath  # noqa

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)

from meshmode.discretization.poly_element import (
        PolynomialWarpAndBlendGroupFactory)
from meshmode.mesh import BTAG_ALL

import pytest
import os

import logging
logger = logging.getLogger(__name__)

# Is there a smart way of choosing this number?
# Currenly it is the same as the base from MPIBoundaryCommunicator
TAG_BASE = 83411
TAG_SEND_REMOTE_NODES = TAG_BASE + 3
TAG_SEND_LOCAL_NODES = TAG_BASE + 4


# {{{ partition_interpolation


# FIXME: Getting some warning on some of these tests. Need to look into this later.
@pytest.mark.parametrize("group_factory", [PolynomialWarpAndBlendGroupFactory])
@pytest.mark.parametrize("num_parts", [2, 3])
@pytest.mark.parametrize("num_groups", [1, 2])
@pytest.mark.parametrize("scramble_partitions", [False])
@pytest.mark.parametrize(("dim", "mesh_pars"),
        [
         (2, [3, 4, 7]),
         (3, [3, 4])
        ])
def test_partition_interpolation(ctx_factory, group_factory, dim, mesh_pars,
                                 num_parts, num_groups, scramble_partitions):
    np.random.seed(42)
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    order = 4

    from pytools.convergence import EOCRecorder
    eoc_rec = dict()
    for i in range(num_parts):
        for j in range(num_parts):
            if i == j:
                continue
            eoc_rec[i, j] = EOCRecorder()

    def f(x):
        return 10.*cl.clmath.sin(500.*x)

    for n in mesh_pars:
        from meshmode.mesh.generation import generate_warped_rect_mesh
        meshes = [generate_warped_rect_mesh(dim, order=order, n=n)
                                for _ in range(num_groups)]

        if num_groups > 1:
            from meshmode.mesh.processing import merge_disjoint_meshes
            mesh = merge_disjoint_meshes(meshes)
        else:
            mesh = meshes[0]

        if scramble_partitions:
            part_per_element = np.random.randint(num_parts, size=mesh.nelements)
        else:
            from pymetis import part_graph
            _, p = part_graph(num_parts,
                              xadj=mesh.nodal_adjacency.neighbors_starts.tolist(),
                              adjncy=mesh.nodal_adjacency.neighbors.tolist())
            part_per_element = np.array(p)

        from meshmode.mesh.processing import partition_mesh
        part_meshes = [
            partition_mesh(mesh, part_per_element, i)[0] for i in range(num_parts)]

        from meshmode.discretization import Discretization
        vol_discrs = [Discretization(cl_ctx, part_meshes[i], group_factory(order))
                        for i in range(num_parts)]

        from meshmode.mesh import BTAG_PARTITION
        from meshmode.discretization.connection import (make_face_restriction,
                                                        make_partition_connection,
                                                        check_connection)

        for i_local_part, i_remote_part in eoc_rec.keys():
            if eoc_rec[i_local_part, i_remote_part] is None:
                continue

            # Mark faces within local_mesh that are connected to remote_mesh
            local_bdry_conn = make_face_restriction(vol_discrs[i_local_part],
                                                    group_factory(order),
                                                    BTAG_PARTITION(i_remote_part))

            # If these parts are not connected, don't bother checking the error
            bdry_nodes = local_bdry_conn.to_discr.nodes()
            if bdry_nodes.size == 0:
                eoc_rec[i_local_part, i_remote_part] = None
                continue

            # Mark faces within remote_mesh that are connected to local_mesh
            remote_bdry_conn = make_face_restriction(vol_discrs[i_remote_part],
                                                     group_factory(order),
                                                     BTAG_PARTITION(i_local_part))

            assert bdry_nodes.size == remote_bdry_conn.to_discr.nodes().size, \
                        "partitions do not have the same number of connected nodes"

            # Gather just enough information for the connection
            local_bdry = local_bdry_conn.to_discr
            local_mesh = part_meshes[i_local_part]
            local_adj_groups = [local_mesh.facial_adjacency_groups[i][None]
                                for i in range(len(local_mesh.groups))]
            local_batches = [local_bdry_conn.groups[i].batches
                                for i in range(len(local_mesh.groups))]
            local_to_elem_faces = [[batch.to_element_face
                                            for batch in grp_batches]
                                        for grp_batches in local_batches]
            local_to_elem_indices = [[batch.to_element_indices.get(queue=queue)
                                            for batch in grp_batches]
                                        for grp_batches in local_batches]

            remote_bdry = remote_bdry_conn.to_discr
            remote_mesh = part_meshes[i_remote_part]
            remote_adj_groups = [remote_mesh.facial_adjacency_groups[i][None]
                                for i in range(len(remote_mesh.groups))]
            remote_batches = [remote_bdry_conn.groups[i].batches
                                for i in range(len(remote_mesh.groups))]
            remote_to_elem_faces = [[batch.to_element_face
                                            for batch in grp_batches]
                                        for grp_batches in remote_batches]
            remote_to_elem_indices = [[batch.to_element_indices.get(queue=queue)
                                            for batch in grp_batches]
                                        for grp_batches in remote_batches]

            # Connect from local_mesh to remote_mesh
            local_to_remote_conn = make_partition_connection(local_bdry_conn,
                                                             i_local_part,
                                                             remote_bdry,
                                                             remote_adj_groups,
                                                             remote_to_elem_faces,
                                                             remote_to_elem_indices)
            # Connect from remote mesh to local mesh
            remote_to_local_conn = make_partition_connection(remote_bdry_conn,
                                                             i_remote_part,
                                                             local_bdry,
                                                             local_adj_groups,
                                                             local_to_elem_faces,
                                                             local_to_elem_indices)
            check_connection(local_to_remote_conn)
            check_connection(remote_to_local_conn)

            true_local_points = f(local_bdry.nodes()[0].with_queue(queue))
            remote_points = local_to_remote_conn(queue, true_local_points)
            local_points = remote_to_local_conn(queue, remote_points)

            err = la.norm((true_local_points - local_points).get(), np.inf)
            eoc_rec[i_local_part, i_remote_part].add_data_point(1./n, err)

    for (i, j), e in eoc_rec.items():
        if e is not None:
            print("Error of connection from part %i to part %i." % (i, j))
            print(e)
            assert(e.order_estimate() >= order - 0.5 or e.max_error() < 1e-14)

# }}}


# {{{ partition_mesh

@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("num_parts", [4, 5, 7])
@pytest.mark.parametrize("num_meshes", [1, 2, 7])
@pytest.mark.parametrize("scramble_partitions", [True, False])
def test_partition_mesh(num_parts, num_meshes, dim, scramble_partitions):
    np.random.seed(42)
    n = (5,) * dim
    from meshmode.mesh.generation import generate_regular_rect_mesh
    meshes = [generate_regular_rect_mesh(a=(0 + i,) * dim, b=(1 + i,) * dim, n=n)
                        for i in range(num_meshes)]

    from meshmode.mesh.processing import merge_disjoint_meshes
    mesh = merge_disjoint_meshes(meshes)

    if scramble_partitions:
        part_per_element = np.random.randint(num_parts, size=mesh.nelements)
    else:
        from pymetis import part_graph
        _, p = part_graph(num_parts,
                          xadj=mesh.nodal_adjacency.neighbors_starts.tolist(),
                          adjncy=mesh.nodal_adjacency.neighbors.tolist())
        part_per_element = np.array(p)

    from meshmode.mesh.processing import partition_mesh
    # TODO: The same part_per_element array must be used to partition each mesh.
    # Maybe the interface should be changed to guarantee this.
    new_meshes = [
        partition_mesh(mesh, part_per_element, i) for i in range(num_parts)]

    assert mesh.nelements == np.sum(
        [new_meshes[i][0].nelements for i in range(num_parts)]), \
        "part_mesh has the wrong number of elements"

    assert count_tags(mesh, BTAG_ALL) == np.sum(
        [count_tags(new_meshes[i][0], BTAG_ALL) for i in range(num_parts)]), \
        "part_mesh has the wrong number of BTAG_ALL boundaries"

    from meshmode.mesh import BTAG_PARTITION, InterPartitionAdjacencyGroup
    from meshmode.mesh.processing import find_group_indices
    num_tags = np.zeros((num_parts,))

    index_lookup_table = dict()
    for ipart, (m, _) in enumerate(new_meshes):
        for igrp in range(len(m.groups)):
            adj = m.facial_adjacency_groups[igrp][None]
            if not isinstance(adj, InterPartitionAdjacencyGroup):
                # This group is not connected to another partition.
                continue
            for i, (elem, face) in enumerate(zip(adj.elements, adj.element_faces)):
                index_lookup_table[ipart, igrp, elem, face] = i

    for part_num in range(num_parts):
        part, part_to_global = new_meshes[part_num]
        for grp_num in range(len(part.groups)):
            adj = part.facial_adjacency_groups[grp_num][None]
            tags = -part.facial_adjacency_groups[grp_num][None].neighbors
            assert np.all(tags >= 0)
            if not isinstance(adj, InterPartitionAdjacencyGroup):
                # This group is not connected to another partition.
                continue
            elem_base = part.groups[grp_num].element_nr_base
            for idx in range(len(adj.elements)):
                if adj.global_neighbors[idx] == -1:
                    continue
                elem = adj.elements[idx]
                face = adj.element_faces[idx]
                n_part_num = adj.neighbor_partitions[idx]
                n_meshwide_elem = adj.global_neighbors[idx]
                n_face = adj.neighbor_faces[idx]
                num_tags[n_part_num] += 1
                n_part, n_part_to_global = new_meshes[n_part_num]
                # Hack: find_igrps expects a numpy.ndarray and returns
                #       a numpy.ndarray. But if a single integer is fed
                #       into find_igrps, an integer is returned.
                n_grp_num = int(find_group_indices(n_part.groups, n_meshwide_elem))
                n_adj = n_part.facial_adjacency_groups[n_grp_num][None]
                n_elem_base = n_part.groups[n_grp_num].element_nr_base
                n_elem = n_meshwide_elem - n_elem_base
                n_idx = index_lookup_table[n_part_num, n_grp_num, n_elem, n_face]
                assert (part_num == n_adj.neighbor_partitions[n_idx]
                        and elem + elem_base == n_adj.global_neighbors[n_idx]
                        and face == n_adj.neighbor_faces[n_idx]),\
                        "InterPartitionAdjacencyGroup is not consistent"
                _, n_part_to_global = new_meshes[n_part_num]
                p_meshwide_elem = part_to_global[elem + elem_base]
                p_meshwide_n_elem = n_part_to_global[n_elem + n_elem_base]

                p_grp_num = find_group_indices(mesh.groups, p_meshwide_elem)
                p_n_grp_num = find_group_indices(mesh.groups, p_meshwide_n_elem)

                p_elem_base = mesh.groups[p_grp_num].element_nr_base
                p_n_elem_base = mesh.groups[p_n_grp_num].element_nr_base
                p_elem = p_meshwide_elem - p_elem_base
                p_n_elem = p_meshwide_n_elem - p_n_elem_base

                f_groups = mesh.facial_adjacency_groups[p_grp_num]
                for p_bnd_adj in f_groups.values():
                    for idx in range(len(p_bnd_adj.elements)):
                        if (p_elem == p_bnd_adj.elements[idx] and
                                 face == p_bnd_adj.element_faces[idx]):
                            assert p_n_elem == p_bnd_adj.neighbors[idx],\
                                    "Tag does not give correct neighbor"
                            assert n_face == p_bnd_adj.neighbor_faces[idx],\
                                    "Tag does not give correct neighbor"

    for i_tag in range(num_parts):
        tag_sum = 0
        for mesh, _ in new_meshes:
            tag_sum += count_tags(mesh, BTAG_PARTITION(i_tag))
        assert num_tags[i_tag] == tag_sum,\
                "part_mesh has the wrong number of BTAG_PARTITION boundaries"


def count_tags(mesh, tag):
    num_bnds = 0
    for adj_dict in mesh.facial_adjacency_groups:
        for neighbors in adj_dict[None].neighbors:
            if neighbors < 0:
                if -neighbors & mesh.boundary_tag_bit(tag) != 0:
                    num_bnds += 1
    return num_bnds

# }}}


# {{{ MPI test rank entrypoint

def mpi_test_rank_entrypoint():
    from meshmode.distributed import MPIMeshDistributor, MPIBoundaryCommunicator

    from mpi4py import MPI
    mpi_comm = MPI.COMM_WORLD
    i_local_part = mpi_comm.Get_rank()
    num_parts = mpi_comm.Get_size()

    mesh_dist = MPIMeshDistributor(mpi_comm)

    if mesh_dist.is_mananger_rank():
        np.random.seed(42)
        from meshmode.mesh.generation import generate_warped_rect_mesh
        meshes = [generate_warped_rect_mesh(3, order=4, n=4) for _ in range(2)]

        from meshmode.mesh.processing import merge_disjoint_meshes
        mesh = merge_disjoint_meshes(meshes)

        part_per_element = np.random.randint(num_parts, size=mesh.nelements)

        local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)
    else:
        local_mesh = mesh_dist.receive_mesh_part()

    group_factory = PolynomialWarpAndBlendGroupFactory(4)
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    from meshmode.discretization import Discretization
    vol_discr = Discretization(cl_ctx, local_mesh, group_factory)

    from meshmode.distributed import get_connected_partitions
    connected_parts = get_connected_partitions(local_mesh)
    assert i_local_part not in connected_parts
    bdry_conn_futures = {}
    local_bdry_conns = {}
    for i_remote_part in connected_parts:
        bdry_conn_futures[i_remote_part] = MPIBoundaryCommunicator(mpi_comm,
                                                                   queue,
                                                                   vol_discr,
                                                                   group_factory,
                                                                   i_remote_part)
        local_bdry_conns[i_remote_part] =\
                bdry_conn_futures[i_remote_part].local_bdry_conn

    remote_to_local_bdry_conns = {}
    from meshmode.discretization.connection import check_connection
    while len(bdry_conn_futures) > 0:
        for i_remote_part, future in bdry_conn_futures.items():
            if future.is_ready():
                conn, _ = bdry_conn_futures.pop(i_remote_part)()
                check_connection(conn)
                remote_to_local_bdry_conns[i_remote_part] = conn
                break
    _test_data_transfer(mpi_comm,
                        queue,
                        local_bdry_conns,
                        remote_to_local_bdry_conns,
                        connected_parts)

    logger.debug("Rank %d exiting", i_local_part)


def _test_data_transfer(mpi_comm, queue, local_bdry_conns,
                        remote_to_local_bdry_conns, connected_parts):
    from mpi4py import MPI

    def f(x):
        return 10*cl.clmath.sin(60.*x)

    '''
    Here is a simplified example of what happens from
    the point of view of the local rank.

    Local rank:
        1. Transfer local points from local boundary to remote boundary
            to get remote points.
        2. Send remote points to remote rank.
    Remote rank:
        3. Receive remote points from local rank.
        4. Transfer remote points from remote boundary to local boundary
            to get local points.
        5. Send local points to local rank.
    Local rank:
        6. Receive local points from remote rank.
        7. Check if local points are the same as the original local points.
    '''

    send_reqs = []
    for i_remote_part in connected_parts:
        conn = remote_to_local_bdry_conns[i_remote_part]
        bdry_discr = local_bdry_conns[i_remote_part].to_discr
        bdry_x = bdry_discr.nodes()[0].with_queue(queue=queue)

        true_local_f = f(bdry_x)
        remote_f = conn(queue, true_local_f)

        send_reqs.append(mpi_comm.isend(remote_f.get(queue=queue),
                                        dest=i_remote_part,
                                        tag=TAG_SEND_REMOTE_NODES))

    buffers = {}
    for i_remote_part in connected_parts:
        status = MPI.Status()
        mpi_comm.probe(source=i_remote_part,
                       tag=TAG_SEND_REMOTE_NODES,
                       status=status)
        buffers[i_remote_part] = np.empty(status.count, dtype=bytes)

    recv_reqs = {}
    for i_remote_part, buf in buffers.items():
        recv_reqs[i_remote_part] = mpi_comm.irecv(buf=buf,
                                                  source=i_remote_part,
                                                  tag=TAG_SEND_REMOTE_NODES)
    remote_to_local_f_data = {}
    for i_remote_part, req in recv_reqs.items():
        remote_to_local_f_data[i_remote_part] = req.wait()
        buffers[i_remote_part] = None   # free buffer

    for req in send_reqs:
        req.wait()

    send_reqs = []
    for i_remote_part in connected_parts:
        conn = remote_to_local_bdry_conns[i_remote_part]
        local_f_np = remote_to_local_f_data[i_remote_part]
        local_f_cl = cl.array.Array(queue,
                                    shape=local_f_np.shape,
                                    dtype=local_f_np.dtype)
        local_f_cl.set(local_f_np)
        remote_f = conn(queue, local_f_cl).get(queue=queue)

        send_reqs.append(mpi_comm.isend(remote_f,
                                        dest=i_remote_part,
                                        tag=TAG_SEND_LOCAL_NODES))

    buffers = {}
    for i_remote_part in connected_parts:
        status = MPI.Status()
        mpi_comm.probe(source=i_remote_part,
                       tag=TAG_SEND_LOCAL_NODES,
                       status=status)
        buffers[i_remote_part] = np.empty(status.count, dtype=bytes)

    recv_reqs = {}
    for i_remote_part, buf in buffers.items():
        recv_reqs[i_remote_part] = mpi_comm.irecv(buf=buf,
                                                  source=i_remote_part,
                                                  tag=TAG_SEND_LOCAL_NODES)
    local_f_data = {}
    for i_remote_part, req in recv_reqs.items():
        local_f_data[i_remote_part] = req.wait()
        buffers[i_remote_part] = None   # free buffer

    for req in send_reqs:
        req.wait()

    for i_remote_part in connected_parts:
        bdry_discr = local_bdry_conns[i_remote_part].to_discr
        bdry_x = bdry_discr.nodes()[0].with_queue(queue=queue)

        true_local_f = f(bdry_x).get(queue=queue)
        local_f = local_f_data[i_remote_part]

        from numpy.linalg import norm
        err = norm(true_local_f - local_f, np.inf)
        assert err < 1e-13, "Error = %f is too large" % err

# }}}


# {{{ MPI test pytest entrypoint

@pytest.mark.mpi
@pytest.mark.parametrize("num_partitions", [3, 6])
def test_mpi_communication(num_partitions):
    pytest.importorskip("mpi4py")

    num_ranks = num_partitions
    from subprocess import check_call
    import sys
    newenv = os.environ.copy()
    newenv["RUN_WITHIN_MPI"] = "1"
    check_call([
        "mpiexec", "-np", str(num_ranks), "-x", "RUN_WITHIN_MPI",
        sys.executable, __file__],
        env=newenv)

# }}}


if __name__ == "__main__":
    if "RUN_WITHIN_MPI" in os.environ:
        mpi_test_rank_entrypoint()
    else:
        import sys
        if len(sys.argv) > 1:
            exec(sys.argv[1])
        else:
            from py.test.cmdline import main
            main([__file__])

# vim: fdm=marker
