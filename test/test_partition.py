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

import numpy as np
import pyopencl as cl

from meshmode.dof_array import thaw, flatten, unflatten

from meshmode.array_context import (  # noqa
        pytest_generate_tests_for_pyopencl_array_context
        as pytest_generate_tests)

from meshmode.discretization.poly_element import (
        PolynomialWarpAndBlendGroupFactory)
from meshmode.mesh import BTAG_ALL

import pytest
import os

import logging
logger = logging.getLogger(__name__)

# Is there a smart way of choosing this number?
# Currenly it is the same as the base from MPIBoundaryTransceiver
TAG_BASE = 83411
TAG_SEND_REMOTE_NODES = TAG_BASE + 3
TAG_SEND_LOCAL_NODES = TAG_BASE + 4


# {{{ partition_interpolation

@pytest.mark.parametrize("num_parts", [3])
@pytest.mark.parametrize("num_groups", [2])
@pytest.mark.parametrize("part_method", ["random", "facial", "nodal"])
@pytest.mark.parametrize(("dim", "mesh_pars"),
        [
         (2, [3, 4, 7]),
         (3, [3, 4])
        ])
def test_partition_interpolation(actx_factory, dim, mesh_pars,
                                 num_parts, num_groups, part_method):
    np.random.seed(42)
    group_factory = PolynomialWarpAndBlendGroupFactory
    actx = actx_factory()

    order = 4

    def f(x):
        return 10.*actx.np.sin(50.*x)

    for n in mesh_pars:
        from meshmode.mesh.generation import generate_warped_rect_mesh
        base_mesh = generate_warped_rect_mesh(dim, order=order, n=n)

        if num_groups > 1:
            from meshmode.mesh.processing import split_mesh_groups
            # Group every Nth element
            element_flags = np.arange(base_mesh.nelements,
                        dtype=base_mesh.element_id_dtype) % num_groups
            mesh = split_mesh_groups(base_mesh, element_flags)
        else:
            mesh = base_mesh

        if part_method == "random":
            part_per_element = np.random.randint(num_parts, size=mesh.nelements)
        else:
            pytest.importorskip("pymetis")

            from meshmode.distributed import get_partition_by_pymetis
            part_per_element = get_partition_by_pymetis(mesh, num_parts,
                    connectivity=part_method)

        from meshmode.mesh.processing import partition_mesh
        part_meshes = [
            partition_mesh(mesh, part_per_element, i)[0] for i in range(num_parts)]

        connected_parts = set()
        for i_local_part, part_mesh in enumerate(part_meshes):
            from meshmode.distributed import get_connected_partitions
            neighbors = get_connected_partitions(part_mesh)
            for i_remote_part in neighbors:
                connected_parts.add((i_local_part, i_remote_part))

        from meshmode.discretization import Discretization
        vol_discrs = [Discretization(actx, part_meshes[i], group_factory(order))
                        for i in range(num_parts)]

        from meshmode.mesh import BTAG_PARTITION
        from meshmode.discretization.connection import (make_face_restriction,
                                                        make_partition_connection,
                                                        check_connection)

        for i_local_part, i_remote_part in connected_parts:
            # Mark faces within local_mesh that are connected to remote_mesh
            local_bdry_conn = make_face_restriction(actx, vol_discrs[i_local_part],
                                                    group_factory(order),
                                                    BTAG_PARTITION(i_remote_part))

            # Mark faces within remote_mesh that are connected to local_mesh
            remote_bdry_conn = make_face_restriction(actx, vol_discrs[i_remote_part],
                                                     group_factory(order),
                                                     BTAG_PARTITION(i_local_part))

            bdry_nelements = sum(
                    grp.nelements for grp in local_bdry_conn.to_discr.groups)
            remote_bdry_nelements = sum(
                    grp.nelements for grp in remote_bdry_conn.to_discr.groups)
            assert bdry_nelements == remote_bdry_nelements, \
                    "partitions do not have the same number of connected elements"

            local_bdry = local_bdry_conn.to_discr

            remote_bdry = remote_bdry_conn.to_discr

            from meshmode.distributed import make_remote_group_infos
            remote_to_local_conn = make_partition_connection(
                    actx,
                    local_bdry_conn=local_bdry_conn,
                    i_local_part=i_local_part,
                    remote_bdry_discr=remote_bdry,
                    remote_group_infos=make_remote_group_infos(
                        actx, remote_bdry_conn))

            # Connect from local mesh to remote mesh
            local_to_remote_conn = make_partition_connection(
                    actx,
                    local_bdry_conn=remote_bdry_conn,
                    i_local_part=i_remote_part,
                    remote_bdry_discr=local_bdry,
                    remote_group_infos=make_remote_group_infos(
                        actx, local_bdry_conn))

            check_connection(actx, remote_to_local_conn)
            check_connection(actx, local_to_remote_conn)

            true_local_points = f(thaw(actx, local_bdry.nodes()[0]))
            remote_points = local_to_remote_conn(true_local_points)
            local_points = remote_to_local_conn(remote_points)

            err = actx.np.linalg.norm(true_local_points - local_points, np.inf)

            # Can't currently expect exact results due to limitations of
            # interpolation "snapping" in DirectDiscretizationConnection's
            # _resample_point_pick_indices
            assert err < 1e-11

# }}}


# {{{ partition_mesh

@pytest.mark.parametrize(("dim", "mesh_size", "num_parts", "scramble_partitions"),
        [
            (2, 5, 4, False),
            (2, 5, 4, True),
            (2, 5, 5, False),
            (2, 5, 5, True),
            (2, 5, 7, False),
            (2, 5, 7, True),
            (2, 10, 32, False),
            (3, 8, 32, False),
        ])
@pytest.mark.parametrize("num_groups", [1, 2, 7])
def test_partition_mesh(mesh_size, num_parts, num_groups, dim, scramble_partitions):
    np.random.seed(42)
    n = (mesh_size,) * dim
    from meshmode.mesh.generation import generate_regular_rect_mesh
    meshes = [generate_regular_rect_mesh(a=(0 + i,) * dim, b=(1 + i,) * dim, n=n)
                        for i in range(num_groups)]

    from meshmode.mesh.processing import merge_disjoint_meshes
    mesh = merge_disjoint_meshes(meshes)

    if scramble_partitions:
        part_per_element = np.random.randint(num_parts, size=mesh.nelements)
    else:
        pytest.importorskip("pymetis")

        from meshmode.distributed import get_partition_by_pymetis
        part_per_element = get_partition_by_pymetis(mesh, num_parts)

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

    connected_parts = set()
    for i_local_part, (part_mesh, _) in enumerate(new_meshes):
        from meshmode.distributed import get_connected_partitions
        neighbors = get_connected_partitions(part_mesh)
        for i_remote_part in neighbors:
            connected_parts.add((i_local_part, i_remote_part))

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
                if adj.partition_neighbors[idx] == -1:
                    continue
                elem = adj.elements[idx]
                face = adj.element_faces[idx]
                n_part_num = adj.neighbor_partitions[idx]
                n_meshwide_elem = adj.partition_neighbors[idx]
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
                        and elem + elem_base == n_adj.partition_neighbors[n_idx]
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
                        if (p_elem == p_bnd_adj.elements[idx]
                                 and face == p_bnd_adj.element_faces[idx]):
                            assert p_n_elem == p_bnd_adj.neighbors[idx],\
                                    "Tag does not give correct neighbor"
                            assert n_face == p_bnd_adj.neighbor_faces[idx],\
                                    "Tag does not give correct neighbor"

    for i_remote_part in range(num_parts):
        tag_sum = 0
        for i_local_part, (mesh, _) in enumerate(new_meshes):
            if (i_local_part, i_remote_part) in connected_parts:
                tag_sum += count_tags(mesh, BTAG_PARTITION(i_remote_part))
        assert num_tags[i_remote_part] == tag_sum,\
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


# {{{ MPI test boundary swap

def _test_mpi_boundary_swap(dim, order, num_groups):
    from meshmode.distributed import MPIMeshDistributor, MPIBoundaryCommSetupHelper

    from mpi4py import MPI
    mpi_comm = MPI.COMM_WORLD
    i_local_part = mpi_comm.Get_rank()
    num_parts = mpi_comm.Get_size()

    mesh_dist = MPIMeshDistributor(mpi_comm)

    if mesh_dist.is_mananger_rank():
        np.random.seed(42)
        from meshmode.mesh.generation import generate_warped_rect_mesh
        meshes = [generate_warped_rect_mesh(dim, order=order, n=4)
                        for _ in range(num_groups)]

        if num_groups > 1:
            from meshmode.mesh.processing import merge_disjoint_meshes
            mesh = merge_disjoint_meshes(meshes)
        else:
            mesh = meshes[0]

        part_per_element = np.random.randint(num_parts, size=mesh.nelements)

        local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)
    else:
        local_mesh = mesh_dist.receive_mesh_part()

    group_factory = PolynomialWarpAndBlendGroupFactory(order)

    from meshmode.array_context import PyOpenCLArrayContext
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    from meshmode.discretization import Discretization
    vol_discr = Discretization(actx, local_mesh, group_factory)

    from meshmode.distributed import get_connected_partitions
    connected_parts = get_connected_partitions(local_mesh)
    assert i_local_part not in connected_parts
    bdry_setup_helpers = {}
    local_bdry_conns = {}

    from meshmode.discretization.connection import make_face_restriction
    from meshmode.mesh import BTAG_PARTITION
    for i_remote_part in connected_parts:
        local_bdry_conns[i_remote_part] = make_face_restriction(
                actx, vol_discr, group_factory, BTAG_PARTITION(i_remote_part))

        setup_helper = bdry_setup_helpers[i_remote_part] = \
                MPIBoundaryCommSetupHelper(
                        mpi_comm, actx, local_bdry_conns[i_remote_part],
                        i_remote_part, bdry_grp_factory=group_factory)

        setup_helper.post_sends()

    remote_to_local_bdry_conns = {}
    from meshmode.discretization.connection import check_connection
    while bdry_setup_helpers:
        for i_remote_part, setup_helper in bdry_setup_helpers.items():
            if setup_helper.is_setup_ready():
                assert bdry_setup_helpers.pop(i_remote_part) is setup_helper
                conn = setup_helper.complete_setup()
                check_connection(actx, conn)
                remote_to_local_bdry_conns[i_remote_part] = conn
                break

        # FIXME: Not ideal, busy-waits

    _test_data_transfer(mpi_comm,
                        actx,
                        local_bdry_conns,
                        remote_to_local_bdry_conns,
                        connected_parts)

    logger.debug("Rank %d exiting", i_local_part)


# TODO
def _test_data_transfer(mpi_comm, actx, local_bdry_conns,
                        remote_to_local_bdry_conns, connected_parts):
    from mpi4py import MPI

    def f(x):
        return 10*actx.np.sin(20.*x)

    """
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
    """

    # 1.
    send_reqs = []
    for i_remote_part in connected_parts:
        conn = remote_to_local_bdry_conns[i_remote_part]
        bdry_discr = local_bdry_conns[i_remote_part].to_discr
        bdry_x = thaw(actx, bdry_discr.nodes()[0])

        true_local_f = f(bdry_x)
        remote_f = conn(true_local_f)

        # 2.
        send_reqs.append(mpi_comm.isend(actx.to_numpy(flatten(remote_f)),
                                        dest=i_remote_part,
                                        tag=TAG_SEND_REMOTE_NODES))

    # 3.
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

    # 4.
    send_reqs = []
    for i_remote_part in connected_parts:
        conn = remote_to_local_bdry_conns[i_remote_part]
        local_f = unflatten(actx, conn.from_discr,
                actx.from_numpy(remote_to_local_f_data[i_remote_part]))
        remote_f = actx.to_numpy(flatten(conn(local_f)))

        # 5.
        send_reqs.append(mpi_comm.isend(remote_f,
                                        dest=i_remote_part,
                                        tag=TAG_SEND_LOCAL_NODES))

    # 6.
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

    # 7.
    for i_remote_part in connected_parts:
        bdry_discr = local_bdry_conns[i_remote_part].to_discr
        bdry_x = thaw(actx, bdry_discr.nodes()[0])

        true_local_f = actx.to_numpy(flatten(f(bdry_x)))
        local_f = local_f_data[i_remote_part]

        from numpy.linalg import norm
        err = norm(true_local_f - local_f, np.inf)
        assert err < 1e-11, "Error = %f is too large" % err

# }}}


# {{{ MPI pytest entrypoint

@pytest.mark.mpi
@pytest.mark.parametrize("num_partitions", [3, 4])
@pytest.mark.parametrize("order", [2, 3])
def test_mpi_communication(num_partitions, order):
    pytest.importorskip("mpi4py")

    num_ranks = num_partitions
    from subprocess import check_call
    import sys
    check_call([
        "mpiexec", "-np", str(num_ranks),
        "-x", "RUN_WITHIN_MPI=1",
        "-x", "order=%d" % order,

        # https://mpi4py.readthedocs.io/en/stable/mpi4py.run.html
        sys.executable, "-m", "mpi4py.run", __file__],
        )

# }}}


if __name__ == "__main__":
    if "RUN_WITHIN_MPI" in os.environ:
        dim = 2
        order = int(os.environ["order"])
        num_groups = 2
        _test_mpi_boundary_swap(dim, order, num_groups)
    else:
        import sys
        if len(sys.argv) > 1:
            exec(sys.argv[1])
        else:
            from pytest import main
            main([__file__])

# vim: fdm=marker
