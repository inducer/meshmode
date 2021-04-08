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

from dataclasses import dataclass
import numpy as np

from meshmode.mesh import InterPartitionAdjacencyGroup

# This file needs to be importable without mpi4py. So don't be tempted to add
# that import here--push it into individual functions instead.

import logging
logger = logging.getLogger(__name__)

TAG_BASE = 83411
TAG_DISTRIBUTE_MESHES = TAG_BASE + 1

__doc__ = """
.. autoclass:: MPIMeshDistributor
.. autoclass:: MPIBoundaryCommSetupHelper

.. autofunction:: get_partition_by_pymetis
.. autofunction:: get_connected_partitions
"""


# {{{ mesh distributor

class MPIMeshDistributor:
    """
    .. automethod:: is_mananger_rank
    .. automethod:: send_mesh_parts
    .. automethod:: recv_mesh_part
    """
    def __init__(self, mpi_comm, manager_rank=0):
        """
        :arg mpi_comm: A :class:`MPI.Intracomm`
        """
        self.mpi_comm = mpi_comm
        self.manager_rank = manager_rank

    def is_mananger_rank(self):
        return self.mpi_comm.Get_rank() == self.manager_rank

    def send_mesh_parts(self, mesh, part_per_element, num_parts):
        """
        :arg mesh: A :class:`Mesh` to distribute to other ranks.
        :arg part_per_element: A :class:`numpy.ndarray` containing one
            integer per element of *mesh* indicating which part of the
            partitioned mesh the element is to become a part of.
        :arg num_parts: The number of partitions to divide the mesh into.

        Sends each partition to a different rank.
        Returns one partition that was not sent to any other rank.
        """
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

        logger.info("rank %d: sent all mesh partitions", rank)
        for req in reqs:
            req.wait()

        return local_part

    def receive_mesh_part(self):
        """
        Returns the mesh sent by the manager rank.
        """
        mpi_comm = self.mpi_comm
        rank = mpi_comm.Get_rank()

        assert not self.is_mananger_rank(), "Manager rank cannot receive mesh"

        from mpi4py import MPI
        status = MPI.Status()
        result = self.mpi_comm.recv(
                source=self.manager_rank, tag=TAG_DISTRIBUTE_MESHES,
                status=status)
        logger.info("rank %d: received local mesh (size = %d)", rank, status.count)

        return result

# }}}


# {{{ boundary communication setup helper

@dataclass
class RemoteGroupInfo:
    inter_partition_adj_group: InterPartitionAdjacencyGroup
    vol_elem_indices: np.ndarray
    bdry_elem_indices: np.ndarray
    bdry_faces: np.ndarray


def make_remote_group_infos(actx, bdry_conn):
    local_vol_mesh = bdry_conn.from_discr.mesh

    assert len(local_vol_mesh.groups) == len(bdry_conn.to_discr.groups)

    return [
            RemoteGroupInfo(
                inter_partition_adj_group=(
                    local_vol_mesh.facial_adjacency_groups[igrp][None]),
                vol_elem_indices=np.concatenate([
                    actx.to_numpy(batch.from_element_indices)
                    for batch in bdry_conn.groups[igrp].batches]),
                bdry_elem_indices=np.concatenate([
                    actx.to_numpy(batch.to_element_indices)
                    for batch in bdry_conn.groups[igrp].batches]),
                bdry_faces=np.concatenate(
                    [np.full(batch.nelements, batch.to_element_face)
                        for batch in bdry_conn.groups[igrp].batches]))
            for igrp in range(len(bdry_conn.from_discr.groups))]


class MPIBoundaryCommSetupHelper:
    """
    Helper for setting up inter-partition facial data exchange.

    .. automethod:: __init__
    .. automethod:: __enter__
    .. automethod:: __exit__
    .. automethod:: complete_some
    """
    def __init__(self, mpi_comm, actx, local_bdry_conns, bdry_grp_factory):
        """
        :arg mpi_comm: A :class:`MPI.Intracomm`
        :arg actx: An array context
        :arg local_bdry_conns: A :class:`dict` mapping remote partition to
            `local_bdry_conn`, where `local_bdry_conn` is a
            :class:`DirectDiscretizationConnection` that performs data exchange from
            the volume to the faces adjacent to partition `i_remote_part`.
        :arg bdry_grp_factory: Group factory to use when creating the remote-to-local
            boundary connections
        """
        self.mpi_comm = mpi_comm
        self.array_context = actx
        self.i_local_part = mpi_comm.Get_rank()
        self.connected_parts = list(local_bdry_conns.keys())
        self.local_bdry_conns = local_bdry_conns
        self.bdry_grp_factory = bdry_grp_factory
        self._internal_mpi_comm = None
        self.send_reqs = None
        self.recv_reqs = None

    def __enter__(self):
        self._internal_mpi_comm = self.mpi_comm.Dup()

        logger.info("bdry comm rank %d comm begin", self.i_local_part)

        self.recv_reqs = [
            self._internal_mpi_comm.irecv(source=i_remote_part)
            for i_remote_part in self.connected_parts]

        self.send_reqs = [
            self._internal_mpi_comm.isend((
                self.local_bdry_conns[i_remote_part].to_discr.mesh,
                make_remote_group_infos(
                    self.array_context, self.local_bdry_conns[i_remote_part])),
                dest=i_remote_part)
            for i_remote_part in self.connected_parts]

        return self

    def __exit__(self, type, value, traceback):
        self._internal_mpi_comm.Free()

    def complete_some(self):
        """
        Returns a :class:`dict` mapping a subset of remote partitions to
        remote-to-local boundary connections, where a remote-to-local boundary
        connection is a :class:`DirectDiscretizationConnection` that performs data
        exchange across faces from partition `i_remote_part` to the local mesh. When
        an empty dictionary is returned, setup is complete.
        """
        from mpi4py import MPI

        # FIXME: when waitsome makes it into an mpi4py release
        # indices, data = MPI.Request.waitsome(self.recv_reqs)
        index, msg = MPI.Request.waitany(self.recv_reqs)

        if index == MPI.UNDEFINED:
            # Already completed, nothing more to do
            return {}

        indices = [index]
        data = [msg]

        remote_to_local_bdry_conns = {}

        for irecv, (remote_bdry_mesh, remote_group_infos) in zip(indices, data):
            i_remote_part = self.connected_parts[irecv]
            logger.debug("rank %d: Received rank %d data",
                         self.i_local_part, i_remote_part)

            # Connect local_mesh to remote_mesh
            from meshmode.discretization.connection import make_partition_connection
            local_bdry_conn = self.local_bdry_conns[i_remote_part]
            remote_to_local_bdry_conns[i_remote_part] = make_partition_connection(
                    self.array_context,
                    local_bdry_conn=local_bdry_conn,
                    i_local_part=self.i_local_part,
                    remote_bdry_discr=local_bdry_conn.to_discr.copy(
                        actx=self.array_context,
                        mesh=remote_bdry_mesh,
                        group_factory=self.bdry_grp_factory),
                    remote_group_infos=remote_group_infos)

        all_recvs_completed = not any([bool(req) for req in self.recv_reqs])
        if all_recvs_completed:
            MPI.Request.waitall(self.send_reqs)
            logger.info("bdry comm rank %d comm end", self.i_local_part)

        return remote_to_local_bdry_conns

# }}}


def get_partition_by_pymetis(mesh, num_parts, *, connectivity="facial", **kwargs):
    """Return a mesh partition created by :mod:`pymetis`.

    :arg mesh: A :class:`meshmode.mesh.Mesh` instance
    :arg num_parts: the number of parts in the mesh partition
    :arg connectivity: the adjacency graph to be used for partitioning. Either
        ``"facial"`` or ``"nodal"`` (based on vertices).
    :arg kwargs: Passed unmodified to :func:`pymetis.part_graph`.
    :returns: a :class:`numpy.ndarray` with one entry per element indicating
        to which partition each element belongs, with entries between ``0`` and
        ``num_parts-1``.

    .. versionchanged:: 2020.2

        *connectivity* was added.
    """

    if connectivity == "facial":
        # shape: (2, n_el_pairs)
        neighbor_el_pairs = np.hstack([
                np.array([
                    fagrp.elements
                    + mesh.groups[fagrp.igroup].element_nr_base,
                    fagrp.neighbors
                    + mesh.groups[fagrp.ineighbor_group].element_nr_base])
                for fadj in mesh.facial_adjacency_groups
                for to_grp, fagrp in fadj.items()
                if fagrp.ineighbor_group is not None
                ])
        sorted_neighbor_el_pairs = neighbor_el_pairs[
                :, np.argsort(neighbor_el_pairs[0])]
        xadj = np.searchsorted(
                sorted_neighbor_el_pairs[0],
                np.arange(mesh.nelements+1))
        adjncy = sorted_neighbor_el_pairs[1]

    elif connectivity == "nodal":
        xadj = mesh.nodal_adjacency.neighbors_starts.tolist()
        adjncy = mesh.nodal_adjacency.neighbors.tolist()

    else:
        raise ValueError("invalid value of connectivity")

    from pymetis import part_graph
    _, p = part_graph(num_parts, xadj=xadj, adjncy=adjncy, **kwargs)

    return np.array(p)


def get_connected_partitions(mesh):
    """For a local mesh part in *mesh*, determine the set of numbers
    of remote partitions to which this mesh piece is connected.

    :arg mesh: A :class:`meshmode.mesh.Mesh` instance
    :returns: the set of partition numbers that are connected to `mesh`
    """
    connected_parts = set()
    for adj in mesh.facial_adjacency_groups:
        grp = adj.get(None, None)
        if isinstance(grp, InterPartitionAdjacencyGroup):
            indices = grp.neighbor_partitions >= 0
            connected_parts = connected_parts.union(
                    grp.neighbor_partitions[indices])

    return connected_parts


# vim: foldmethod=marker
