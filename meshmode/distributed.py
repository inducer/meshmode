"""
.. autoclass:: InterRankBoundaryInfo
.. autoclass:: MPIBoundaryCommSetupHelper

.. autofunction:: get_partition_by_pymetis
.. autofunction:: membership_list_to_map
.. autofunction:: get_connected_parts

.. autoclass:: RemoteGroupInfo
.. autoclass:: make_remote_group_infos
"""

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
from typing import TYPE_CHECKING, Any, Hashable, List, Mapping, Sequence, Union, cast
from warnings import warn

import numpy as np

from arraycontext import ArrayContext

from meshmode.discretization import ElementGroupFactory
from meshmode.discretization.connection import DirectDiscretizationConnection
from meshmode.mesh import (
    InteriorAdjacencyGroup, InterPartAdjacencyGroup, Mesh, PartID)


# This file needs to be importable without mpi4py. So don't be tempted to add
# that import here--push it into individual functions instead.


if TYPE_CHECKING:
    import mpi4py.MPI

import logging


logger = logging.getLogger(__name__)

TAG_BASE = 83411
TAG_DISTRIBUTE_MESHES = TAG_BASE + 1


# {{{ mesh distributor

class MPIMeshDistributor:
    """
    .. automethod:: is_mananger_rank
    .. automethod:: send_mesh_parts
    .. automethod:: receive_mesh_part
    """
    def __init__(self, mpi_comm, manager_rank=0):
        """
        :arg mpi_comm: An ``MPI.Intracomm``
        """
        self.mpi_comm = mpi_comm
        self.manager_rank = manager_rank

        warn("MPIMeshDistributor is deprecated and will be removed in 2024. "
             "Directly call partition_mesh and use mpi_comm.scatter instead.",
             DeprecationWarning, stacklevel=2)

    def is_mananger_rank(self):
        return self.mpi_comm.Get_rank() == self.manager_rank

    def send_mesh_parts(self, mesh, part_per_element, num_parts):
        """
        :arg mesh: A :class:`~meshmode.mesh.Mesh` to distribute to other ranks.
        :arg part_per_element: A :class:`numpy.ndarray` containing one
            integer per element of *mesh* indicating which part of the
            partitioned mesh the element is to become a part of.
        :arg num_parts: The number of parts to divide the mesh into.

        Sends each part to a different rank.
        Returns one part that was not sent to any other rank.
        """
        mpi_comm = self.mpi_comm
        rank = mpi_comm.Get_rank()
        assert num_parts <= mpi_comm.Get_size()

        assert self.is_mananger_rank()

        part_num_to_elements = membership_list_to_map(part_per_element)

        from meshmode.mesh.processing import partition_mesh
        parts = partition_mesh(mesh, part_num_to_elements)

        local_part = None

        reqs = []
        for r, part in parts.items():
            if r == self.manager_rank:
                local_part = part
            else:
                reqs.append(mpi_comm.isend(part, dest=r, tag=TAG_DISTRIBUTE_MESHES))

        logger.info("rank %d: sent all mesh parts", rank)
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


# {{{ remote group info

# FIXME: "Remote" is perhaps not the best naming convention for this. For example,
# in a multi-volume context it may be used when constructing inter-part connections
# between two parts on the same rank.
@dataclass
class RemoteGroupInfo:
    inter_part_adj_groups: List[InterPartAdjacencyGroup]
    vol_elem_indices: np.ndarray
    bdry_elem_indices: np.ndarray
    bdry_faces: np.ndarray


def make_remote_group_infos(
        actx: ArrayContext,
        remote_part_id: PartID,
        bdry_conn: DirectDiscretizationConnection
        ) -> Sequence[RemoteGroupInfo]:
    local_vol_mesh = bdry_conn.from_discr.mesh

    assert len(local_vol_mesh.groups) == len(bdry_conn.to_discr.groups)

    return [
            RemoteGroupInfo(
                inter_part_adj_groups=[
                    fagrp for fagrp in local_vol_mesh.facial_adjacency_groups[igrp]
                    if isinstance(fagrp, InterPartAdjacencyGroup)
                    and fagrp.part_id == remote_part_id],
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

# }}}


# {{{ boundary communication setup helper

@dataclass(init=True, frozen=True)
class InterRankBoundaryInfo:
    """
    .. attribute:: local_part_id

        An opaque, hashable, picklable identifier for the local part.

    .. attribute:: remote_part_id

        An opaque, hashable, picklable identifier for the remote part.

    .. attribute:: remote_rank

        The MPI rank with which this boundary communicates.

    .. attribute:: local_boundary_connection

        A :class:`~meshmode.discretization.connection.DirectDiscretizationConnection`
        from the volume onto the boundary described by
        ``BTAG_PARTITION(remote_part_id)``.

    .. automethod:: __init__
    """

    local_part_id: PartID
    remote_part_id: PartID
    remote_rank: int
    local_boundary_connection: DirectDiscretizationConnection


class MPIBoundaryCommSetupHelper:
    """
    Helper for setting up inter-part facial data exchange.

    .. automethod:: __init__
    .. automethod:: __enter__
    .. automethod:: __exit__
    .. automethod:: complete_some
    """
    def __init__(self,
            mpi_comm: "mpi4py.MPI.Intracomm",
            actx: ArrayContext,
            inter_rank_bdry_info: Union[
                # new-timey
                Sequence[InterRankBoundaryInfo],
                # old-timey, for compatibility
                Mapping[int, DirectDiscretizationConnection],
                ],
            bdry_grp_factory: ElementGroupFactory):
        """
        :arg bdry_grp_factory: Group factory to use when creating the remote-to-local
            boundary connections
        """
        self.mpi_comm = mpi_comm
        self.array_context = actx
        self.i_local_rank = mpi_comm.Get_rank()

        # {{{ normalize inter_rank_bdry_info

        self._using_old_timey_interface = False

        if isinstance(inter_rank_bdry_info, dict):
            self._using_old_timey_interface = True
            warn("Using the old-timey interface of MPIBoundaryCommSetupHelper. "
                    "That's deprecated and will stop working in July 2022. "
                    "Use the currently documented interface instead.",
                    DeprecationWarning, stacklevel=2)

            inter_rank_bdry_info = [
                    InterRankBoundaryInfo(
                        local_part_id=self.i_local_rank,
                        remote_part_id=remote_rank,
                        remote_rank=remote_rank,
                        local_boundary_connection=conn
                        )
                    for remote_rank, conn in inter_rank_bdry_info.items()]

        # }}}

        self.inter_rank_bdry_info = cast(
                Sequence[InterRankBoundaryInfo], inter_rank_bdry_info)

        self.bdry_grp_factory = bdry_grp_factory

    def __enter__(self):
        self._internal_mpi_comm = self.mpi_comm.Dup()

        logger.info("bdry comm rank %d comm begin", self.i_local_rank)

        # Not using irecv because mpi4py only allocates 32KB per receive buffer
        # when receiving pickled objects. We could pass buffers to irecv explicitly,
        # but in order to know the required buffer sizes we would have to do all of
        # the pickling ourselves.

        # to know when we're done
        self.pending_recv_identifiers = {
                (irbi.local_part_id, irbi.remote_part_id): i
                for i, irbi in enumerate(self.inter_rank_bdry_info)}

        assert len(self.pending_recv_identifiers) \
                == len(self.inter_rank_bdry_info)

        self.send_reqs = [
            self._internal_mpi_comm.isend(
                (
                    irbi.local_part_id,
                    irbi.remote_part_id,
                    irbi.local_boundary_connection.to_discr.mesh,
                    make_remote_group_infos(
                        self.array_context, irbi.remote_part_id,
                        irbi.local_boundary_connection)),
                dest=irbi.remote_rank)
            for irbi in self.inter_rank_bdry_info]

        return self

    def __exit__(self, type, value, traceback):
        self._internal_mpi_comm.Free()

    def complete_some(self):
        """
        Returns a :class:`dict` mapping a subset of remote parts to
        remote-to-local boundary connections, where a remote-to-local boundary
        connection is a
        :class:`~meshmode.discretization.connection.DirectDiscretizationConnection`
        that performs data exchange across faces from part `i_remote_part` to the
        local mesh. When an empty dictionary is returned, setup is complete.
        """
        from mpi4py import MPI

        if not self.pending_recv_identifiers:
            # Already completed, nothing more to do
            return {}

        status = MPI.Status()

        # Wait for all receives
        # Note: This is inefficient, but ensures a deterministic order of
        # boundary setup.
        nrecvs = len(self.pending_recv_identifiers)
        data = [None] * nrecvs
        source_ranks = [None] * nrecvs

        while nrecvs > 0:
            r = self._internal_mpi_comm.recv(status=status)
            key = (r[1], r[0])
            loc = self.pending_recv_identifiers[key]
            assert data[loc] is None
            assert source_ranks[loc] is None
            data[loc] = r
            source_ranks[loc] = status.source
            nrecvs -= 1

        remote_to_local_bdry_conns = {}

        part_ids_to_irbi = {
                (irbi.local_part_id, irbi.remote_part_id): irbi
                for irbi in self.inter_rank_bdry_info}
        if len(part_ids_to_irbi) < len(self.inter_rank_bdry_info):
            raise ValueError(
                "duplicate local/remote part pair in inter_rank_bdry_info")

        for i_src_rank, recvd in zip(source_ranks, data):
            (remote_part_id, local_part_id,
                    remote_bdry_mesh, remote_group_infos) = recvd

            logger.debug("rank %d: Received part id '%s' data from rank %d",
                         self.i_local_rank, remote_part_id, i_src_rank)

            # Connect local_mesh to remote_mesh
            from meshmode.discretization.connection import make_partition_connection
            irbi = part_ids_to_irbi[local_part_id, remote_part_id]
            assert i_src_rank == irbi.remote_rank

            if self._using_old_timey_interface:
                key = remote_part_id
            else:
                key = (remote_part_id, local_part_id)

            remote_to_local_bdry_conns[key] = (
                make_partition_connection(
                    self.array_context,
                    local_bdry_conn=irbi.local_boundary_connection,
                    remote_bdry_discr=irbi.local_boundary_connection.to_discr.copy(
                        actx=self.array_context,
                        mesh=remote_bdry_mesh,
                        group_factory=self.bdry_grp_factory),
                    remote_group_infos=remote_group_infos))

            del self.pending_recv_identifiers[(local_part_id, remote_part_id)]

        assert not self.pending_recv_identifiers
        MPI.Request.waitall(self.send_reqs)
        logger.info("bdry comm rank %d comm end", self.i_local_rank)

        return remote_to_local_bdry_conns

# }}}


# FIXME: Move somewhere else, since it's not strictly limited to distributed?
def get_partition_by_pymetis(mesh, num_parts, *, connectivity="facial", **kwargs):
    """Return a mesh partition created by :mod:`pymetis`.

    :arg mesh: A :class:`meshmode.mesh.Mesh` instance
    :arg num_parts: the number of parts in the mesh partition
    :arg connectivity: the adjacency graph to be used for partitioning. Either
        ``"facial"`` or ``"nodal"`` (based on vertices).
    :arg kwargs: Passed unmodified to :func:`pymetis.part_graph`.
    :returns: a :class:`numpy.ndarray` with one entry per element indicating
        to which part each element belongs, with entries between ``0`` and
        ``num_parts-1``.

    .. versionchanged:: 2020.2

        *connectivity* was added.
    """

    if connectivity == "facial":
        # shape: (2, n_el_pairs)
        neighbor_el_pairs = np.hstack([
                np.array([
                    fagrp.elements + mesh.base_element_nrs[fagrp.igroup],
                    fagrp.neighbors + mesh.base_element_nrs[fagrp.ineighbor_group]
                    ])
                for fagrp_list in mesh.facial_adjacency_groups
                for fagrp in fagrp_list
                if isinstance(fagrp, InteriorAdjacencyGroup)
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


def membership_list_to_map(
            membership_list: np.ndarray[Any, Any]
        ) -> Mapping[Hashable, np.ndarray]:
    """
    Convert a :class:`numpy.ndarray` that maps an index to a key into a
    :class:`dict` that maps a key to a set of indices (with each set of indices
    stored as a sorted :class:`numpy.ndarray`).
    """
    from pytools import unique

    # FIXME: not clear why the sorted() call is necessary here
    return {
        entry: np.where(membership_list == entry)[0]
        for entry in sorted(unique(membership_list))}


# FIXME: Move somewhere else, since it's not strictly limited to distributed?
def get_connected_parts(mesh: Mesh) -> "Sequence[PartID]":
    """For a local mesh part in *mesh*, determine the set of connected parts."""
    assert mesh.facial_adjacency_groups is not None

    from pytools import unique

    return tuple(unique(
            grp.part_id
            for fagrp_list in mesh.facial_adjacency_groups
            for grp in fagrp_list
            if isinstance(grp, InterPartAdjacencyGroup)))


def get_connected_partitions(mesh: Mesh) -> "Sequence[PartID]":
    warn(
        "get_connected_partitions is deprecated and will stop working in June 2023. "
        "Use get_connected_parts instead.", DeprecationWarning, stacklevel=2)
    return get_connected_parts(mesh)

# vim: foldmethod=marker
