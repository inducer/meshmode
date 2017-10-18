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

import six

import numpy as np
from mpi4py import MPI

import logging
logger = logging.getLogger(__name__)

TAG_BASE = 83411
TAG_DISTRIBUTE_MESHES = TAG_BASE + 1
TAG_SEND_BOUNDARY = TAG_BASE + 2
TAG_SEND_REMOTE_NODES = TAG_BASE + 3
TAG_SEND_LOCAL_NODES = TAG_BASE + 4


# {{{ mesh distributor

class MPIMeshDistributor(object):
    def __init__(self, mpi_comm, manager_rank=0):
        self.mpi_comm = mpi_comm
        self.manager_rank = manager_rank

    def is_mananger_rank(self):
        return self.mpi_comm.Get_rank() == self.manager_rank

    def send_mesh_parts(self, mesh, part_per_element, num_parts):
        mpi_comm = self.mpi_comm
        rank = mpi_comm.Get_rank()
        assert num_parts <= mpi_comm.Get_size()

        assert self.is_mananger_rank()

        from meshmode.mesh.processing import partition_mesh
        parts = [partition_mesh(mesh, part_per_element, i)[0]
                        for i in range(num_parts)]

        local_part = None

        reqs = []
        for r, part in enumerate(parts):
            if r == self.manager_rank:
                local_part = part
            else:
                reqs.append(mpi_comm.isend(part, dest=r, tag=TAG_DISTRIBUTE_MESHES))

        logger.info('rank %d: sent all mesh partitions', rank)
        for req in reqs:
            req.wait()

        return local_part

    def receive_mesh_part(self):
        mpi_comm = self.mpi_comm
        rank = mpi_comm.Get_rank()

        status = MPI.Status()
        result = self.mpi_comm.recv(
                source=self.manager_rank, tag=TAG_DISTRIBUTE_MESHES,
                status=status)
        logger.info('rank %d: recieved local mesh (size = %d)', rank, status.count)

        return result

# }}}


# {{{ boundary communicator

class MPIBoundaryCommunicator(object):
    def __init__(self, mpi_comm, queue, part_discr, bdry_group_factory):
        self.mpi_comm = mpi_comm
        self.part_discr = part_discr

        self.i_local_part = mpi_comm.Get_rank()

        self.bdry_group_factory = bdry_group_factory

        from meshmode.mesh import InterPartitionAdjacencyGroup
        self.connected_parts = set()
        for adj in part_discr.mesh.facial_adjacency_groups:
            if isinstance(adj[None], InterPartitionAdjacencyGroup):
                indices = (adj[None].neighbor_partitions >= 0)
                self.connected_parts = self.connected_parts.union(
                                            adj[None].neighbor_partitions[indices])
        assert self.i_local_part not in self.connected_parts

        from meshmode.discretization.connection import make_face_restriction
        from meshmode.mesh import BTAG_PARTITION
        self.local_bdry_conns = {}
        for i_remote_part in self.connected_parts:
            bdry_conn = make_face_restriction(part_discr,
                                              bdry_group_factory,
                                              BTAG_PARTITION(i_remote_part))

            # Assert that everything in self.connected_parts is truly connected
            assert bdry_conn.to_discr.nnodes > 0
            self.local_bdry_conns[i_remote_part] = bdry_conn

        self._setup(queue)

    def _post_boundary_data_sends(self, queue):
        send_reqs = []
        for i_remote_part in self.connected_parts:
            local_bdry = self.local_bdry_conns[i_remote_part].to_discr
            local_mesh = self.local_bdry_conns[i_remote_part].from_discr.mesh
            local_adj_groups = [local_mesh.facial_adjacency_groups[i][None]
                                for i in range(len(local_mesh.groups))]
            local_batches = [self.local_bdry_conns[i_remote_part].groups[i].batches
                                for i in range(len(local_mesh.groups))]
            local_to_elem_faces = [[batch.to_element_face for batch in grp_batches]
                                        for grp_batches in local_batches]
            local_to_elem_indices = [[batch.to_element_indices.get(queue=queue)
                                            for batch in grp_batches]
                                        for grp_batches in local_batches]

            local_data = {'bdry_mesh': local_bdry.mesh,
                          'adj': local_adj_groups,
                          'to_elem_faces': local_to_elem_faces,
                          'to_elem_indices': local_to_elem_indices}
            send_reqs.append(self.mpi_comm.isend(
                local_data, dest=i_remote_part, tag=TAG_SEND_BOUNDARY))

        return send_reqs

    def _receive_boundary_data(self, queue):
        rank = self.mpi_comm.Get_rank()
        i_local_part = rank

        remote_buf = {}
        for i_remote_part in self.connected_parts:
            status = MPI.Status()
            self.mpi_comm.probe(
                    source=i_remote_part, tag=TAG_SEND_BOUNDARY, status=status)
            remote_buf[i_remote_part] = np.empty(status.count, dtype=bytes)

        recv_reqs = {}
        for i_remote_part, buf in remote_buf.items():
            recv_reqs[i_remote_part] = self.mpi_comm.irecv(buf=buf,
                                                  source=i_remote_part,
                                                  tag=TAG_SEND_BOUNDARY)

        remote_data = {}
        total_bytes_recvd = 0
        for i_remote_part, req in recv_reqs.items():
            status = MPI.Status()
            remote_data[i_remote_part] = req.wait(status=status)

            # Free the buffer
            remote_buf[i_remote_part] = None
            logger.debug('rank %d: Received rank %d data (%d bytes)',
                    rank, i_remote_part, status.count)

            total_bytes_recvd += status.count

        logger.debug('rank %d: recieved %d bytes in total', rank, total_bytes_recvd)

        self.remote_to_local_bdry_conns = {}

        from meshmode.discretization import Discretization

        for i_remote_part, data in remote_data.items():
            remote_bdry_mesh = data['bdry_mesh']
            remote_bdry = Discretization(
                    queue.context,
                    remote_bdry_mesh,
                    self.bdry_group_factory)
            remote_adj_groups = data['adj']
            remote_to_elem_faces = data['to_elem_faces']
            remote_to_elem_indices = data['to_elem_indices']

            # Connect local_mesh to remote_mesh
            from meshmode.discretization.connection import make_partition_connection
            self.remote_to_local_bdry_conns[i_remote_part] = \
                    make_partition_connection(
                        self.local_bdry_conns[i_remote_part],
                        i_local_part,
                        remote_bdry,
                        remote_adj_groups,
                        remote_to_elem_faces,
                        remote_to_elem_indices)

    def _setup(self, queue):
        logger.info("bdry comm rank %d send begin", self.mpi_comm.Get_rank())

        send_reqs = self._post_boundary_data_sends(queue)
        self._receive_boundary_data(queue)

        for req in send_reqs:
            req.wait()

        logger.info("bdry comm rank %d send completed", self.mpi_comm.Get_rank())

    def check(self):
        from meshmode.discretization.connection import check_connection

        for i, conn in six.iteritems(self.remote_to_local_bdry_conns):
            check_connection(conn)

    def test_data_transfer(self, queue):
        import pyopencl as cl

        def f(x):
            return 0.1*cl.clmath.sin(30.*x)

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
            6. Recieve local points from remote rank.
            7. Check if local points are the same as the original local points.
        '''

        send_reqs = []
        for i_remote_part in self.connected_parts:
            conn = self.remote_to_local_bdry_conns[i_remote_part]
            bdry_discr = self.local_bdry_conns[i_remote_part].to_discr
            bdry_x = bdry_discr.nodes()[0].with_queue(queue=queue)

            true_local_f = f(bdry_x)
            remote_f = conn(queue, true_local_f)

            send_reqs.append(self.mpi_comm.isend(remote_f.get(queue=queue),
                                                 dest=i_remote_part,
                                                 tag=TAG_SEND_REMOTE_NODES))

        buffers = {}
        for i_remote_part in self.connected_parts:
            status = MPI.Status()
            self.mpi_comm.probe(source=i_remote_part,
                                tag=TAG_SEND_REMOTE_NODES,
                                status=status)
            buffers[i_remote_part] = np.empty(status.count, dtype=bytes)

        recv_reqs = {}
        for i_remote_part, buf in buffers.items():
            recv_reqs[i_remote_part] = self.mpi_comm.irecv(buf=buf,
                                                           source=i_remote_part,
                                                           tag=TAG_SEND_REMOTE_NODES)
        remote_to_local_f_data = {}
        for i_remote_part, req in recv_reqs.items():
            remote_to_local_f_data[i_remote_part] = req.wait()
            buffers[i_remote_part] = None   # free buffer

        for req in send_reqs:
            req.wait()

        send_reqs = []
        for i_remote_part in self.connected_parts:
            conn = self.remote_to_local_bdry_conns[i_remote_part]
            local_f_np = remote_to_local_f_data[i_remote_part]
            local_f_cl = cl.array.Array(queue,
                                        shape=local_f_np.shape,
                                        dtype=local_f_np.dtype)
            local_f_cl.set(local_f_np)
            remote_f = conn(queue, local_f_cl).get(queue=queue)

            send_reqs.append(self.mpi_comm.isend(remote_f,
                                                 dest=i_remote_part,
                                                 tag=TAG_SEND_LOCAL_NODES))

        buffers = {}
        for i_remote_part in self.connected_parts:
            status = MPI.Status()
            self.mpi_comm.probe(source=i_remote_part,
                                tag=TAG_SEND_LOCAL_NODES,
                                status=status)
            buffers[i_remote_part] = np.empty(status.count, dtype=bytes)

        recv_reqs = {}
        for i_remote_part, buf in buffers.items():
            recv_reqs[i_remote_part] = self.mpi_comm.irecv(buf=buf,
                                                           source=i_remote_part,
                                                           tag=TAG_SEND_LOCAL_NODES)
        local_f_data = {}
        for i_remote_part, req in recv_reqs.items():
            local_f_data[i_remote_part] = req.wait()
            buffers[i_remote_part] = None   # free buffer

        for req in send_reqs:
            req.wait()

        for i_remote_part in self.connected_parts:
            bdry_discr = self.local_bdry_conns[i_remote_part].to_discr
            bdry_x = bdry_discr.nodes()[0].with_queue(queue=queue)

            true_local_f = f(bdry_x).get(queue=queue)
            local_f = local_f_data[i_remote_part]

            from numpy.linalg import norm
            err = norm(true_local_f - local_f, np.inf)
            assert err < 1e-13, "Error = %f is too large" % err

# }}}


# vim: foldmethod=marker
