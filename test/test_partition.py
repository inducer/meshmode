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

from meshmode.dof_array import flat_norm

from arraycontext import thaw, flatten, unflatten
from meshmode.array_context import PytestPyOpenCLArrayContextFactory
from arraycontext import pytest_generate_tests_for_array_contexts
pytest_generate_tests = pytest_generate_tests_for_array_contexts(
        [PytestPyOpenCLArrayContextFactory])

from meshmode.discretization.poly_element import default_simplex_group_factory
from meshmode.mesh import (
    BTAG_ALL,
    InteriorAdjacencyGroup,
    BoundaryAdjacencyGroup,
    InterPartitionAdjacencyGroup
)

import pytest
import os

import logging
logger = logging.getLogger(__name__)

# Is there a smart way of choosing this number?
# Currenly it is the same as the base from MPIBoundaryCommSetupHelper
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
    order = 4

    np.random.seed(42)
    group_factory = default_simplex_group_factory(base_dim=dim, order=order)
    actx = actx_factory()

    def f(x):
        return 10.*actx.np.sin(50.*x)

    for n in mesh_pars:
        from meshmode.mesh.generation import generate_warped_rect_mesh
        base_mesh = generate_warped_rect_mesh(dim, order=order, nelements_side=n)

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
        vol_discrs = [Discretization(actx, part_meshes[i], group_factory)
                        for i in range(num_parts)]

        from meshmode.mesh import BTAG_PARTITION
        from meshmode.discretization.connection import (make_face_restriction,
                                                        make_partition_connection,
                                                        check_connection)

        for i_local_part, i_remote_part in connected_parts:
            # Mark faces within local_mesh that are connected to remote_mesh
            local_bdry_conn = make_face_restriction(actx, vol_discrs[i_local_part],
                                                    group_factory,
                                                    BTAG_PARTITION(i_remote_part))

            # Mark faces within remote_mesh that are connected to local_mesh
            remote_bdry_conn = make_face_restriction(actx, vol_discrs[i_remote_part],
                                                     group_factory,
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
                    remote_bdry_discr=remote_bdry,
                    remote_group_infos=make_remote_group_infos(
                        actx, BTAG_PARTITION(i_local_part), remote_bdry_conn))

            # Connect from local mesh to remote mesh
            local_to_remote_conn = make_partition_connection(
                    actx,
                    local_bdry_conn=remote_bdry_conn,
                    remote_bdry_discr=local_bdry,
                    remote_group_infos=make_remote_group_infos(
                        actx, BTAG_PARTITION(i_remote_part), local_bdry_conn))

            check_connection(actx, remote_to_local_conn)
            check_connection(actx, local_to_remote_conn)

            true_local_points = f(thaw(local_bdry.nodes()[0], actx))
            remote_points = local_to_remote_conn(true_local_points)
            local_points = remote_to_local_conn(remote_points)

            err = flat_norm(true_local_points - local_points, np.inf)

            # Can't currently expect exact results due to limitations of
            # interpolation "snapping" in DirectDiscretizationConnection's
            # _resample_point_pick_indices
            assert err < 1e-11

# }}}


# {{{ partition_mesh

def _check_for_cross_rank_adj(mesh, part_per_element):
    for igrp, grp in enumerate(mesh.groups):
        fagrp_list = mesh.facial_adjacency_groups[igrp]
        int_grps = [
            grp for grp in fagrp_list
            if isinstance(grp, InteriorAdjacencyGroup)]
        for fagrp in int_grps:
            ineighbor_grp = fagrp.ineighbor_group
            for iface in range(len(fagrp.elements)):
                part = part_per_element[
                    fagrp.elements[iface] + mesh.base_element_nrs[igrp]]
                neighbor_part = part_per_element[
                    fagrp.neighbors[iface] + mesh.base_element_nrs[ineighbor_grp]]
                if part != neighbor_part:
                    return True
    return False


@pytest.mark.parametrize(("dim", "mesh_size", "num_parts", "scramble_partitions"),
        [
            (2, 4, 4, False),
            (2, 4, 4, True),
            (2, 4, 5, False),
            (2, 4, 5, True),
            (2, 4, 7, False),
            (2, 4, 7, True),
            (2, 9, 32, False),
            (3, 7, 32, False),
        ])
@pytest.mark.parametrize("num_groups", [1, 2, 7])
def test_partition_mesh(mesh_size, num_parts, num_groups, dim, scramble_partitions):
    np.random.seed(42)
    nelements_per_axis = (mesh_size,) * dim
    from meshmode.mesh.generation import generate_regular_rect_mesh
    meshes = [generate_regular_rect_mesh(a=(0 + i,) * dim, b=(1 + i,) * dim,
              nelements_per_axis=nelements_per_axis) for i in range(num_groups)]

    from meshmode.mesh.processing import merge_disjoint_meshes
    mesh = merge_disjoint_meshes(meshes)

    if scramble_partitions:
        part_per_element = np.random.randint(num_parts, size=mesh.nelements)
    else:
        pytest.importorskip("pymetis")

        from meshmode.distributed import get_partition_by_pymetis
        part_per_element = get_partition_by_pymetis(mesh, num_parts)

    # For certain combinations of parameters, partitioned mesh has no cross-rank
    # adjacency (e.g., when #groups == #parts)
    has_cross_rank_adj = _check_for_cross_rank_adj(mesh, part_per_element)

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

    from meshmode.mesh import BTAG_PARTITION
    from meshmode.mesh.processing import find_group_indices
    num_tags = np.zeros((num_parts,))

    index_lookup_table = dict()
    for ipart, (m, _) in enumerate(new_meshes):
        for igrp in range(len(m.groups)):
            ipagrps = [
                fagrp for fagrp in m.facial_adjacency_groups[igrp]
                if isinstance(fagrp, InterPartitionAdjacencyGroup)]
            for ipagrp in ipagrps:
                for i, (elem, face) in enumerate(
                        zip(ipagrp.elements, ipagrp.element_faces)):
                    index_lookup_table[ipart, igrp, elem, face] = i

    ipagrp_count = 0

    for part_num in range(num_parts):
        part, part_to_global = new_meshes[part_num]
        for grp_num in range(len(part.groups)):
            ipagrps = [
                fagrp for fagrp in part.facial_adjacency_groups[grp_num]
                if isinstance(fagrp, InterPartitionAdjacencyGroup)]
            ipagrp_count += len(ipagrps)
            for ipagrp in ipagrps:
                n_part_num = ipagrp.boundary_tag.part_nr
                num_tags[n_part_num] += len(ipagrp.elements)
                elem_base = part.base_element_nrs[grp_num]
                for idx in range(len(ipagrp.elements)):
                    elem = ipagrp.elements[idx]
                    meshwide_elem = elem_base + elem
                    face = ipagrp.element_faces[idx]
                    n_meshwide_elem = ipagrp.neighbors[idx]
                    n_face = ipagrp.neighbor_faces[idx]
                    n_part, n_part_to_global = new_meshes[n_part_num]
                    # Hack: find_igrps expects a numpy.ndarray and returns
                    #       a numpy.ndarray. But if a single integer is fed
                    #       into find_igrps, an integer is returned.
                    n_grp_num = int(find_group_indices(
                        n_part.groups, n_meshwide_elem))
                    n_ipagrps = [
                        fagrp for fagrp in n_part.facial_adjacency_groups[n_grp_num]
                        if isinstance(fagrp, InterPartitionAdjacencyGroup)
                        and fagrp.boundary_tag.part_nr == part_num]
                    found_reverse_adj = False
                    for n_ipagrp in n_ipagrps:
                        n_elem_base = n_part.base_element_nrs[n_grp_num]
                        n_elem = n_meshwide_elem - n_elem_base
                        n_idx = index_lookup_table[
                            n_part_num, n_grp_num, n_elem, n_face]
                        found_reverse_adj = found_reverse_adj or (
                            meshwide_elem == n_ipagrp.neighbors[n_idx]
                            and face == n_ipagrp.neighbor_faces[n_idx])
                        if found_reverse_adj:
                            _, n_part_to_global = new_meshes[n_part_num]
                            p_meshwide_elem = part_to_global[elem + elem_base]
                            p_meshwide_n_elem = n_part_to_global[n_meshwide_elem]
                    assert found_reverse_adj, ("InterPartitionAdjacencyGroup is not "
                        "consistent")

                    p_grp_num = find_group_indices(mesh.groups, p_meshwide_elem)
                    p_n_grp_num = find_group_indices(mesh.groups, p_meshwide_n_elem)

                    p_elem_base = mesh.base_element_nrs[p_grp_num]
                    p_n_elem_base = mesh.base_element_nrs[p_n_grp_num]
                    p_elem = p_meshwide_elem - p_elem_base
                    p_n_elem = p_meshwide_n_elem - p_n_elem_base

                    f_groups = mesh.facial_adjacency_groups[p_grp_num]
                    int_groups = [
                        grp for grp in f_groups
                        if isinstance(grp, InteriorAdjacencyGroup)]
                    for adj in int_groups:
                        for idx in range(len(adj.elements)):
                            if (p_elem == adj.elements[idx]
                                     and face == adj.element_faces[idx]):
                                assert p_n_elem == adj.neighbors[idx],\
                                        "Tag does not give correct neighbor"
                                assert n_face == adj.neighbor_faces[idx],\
                                        "Tag does not give correct neighbor"

    assert ipagrp_count > 0 or not has_cross_rank_adj,\
        "expected at least one InterPartitionAdjacencyGroup"

    for i_remote_part in range(num_parts):
        tag_sum = 0
        for i_local_part, (mesh, _) in enumerate(new_meshes):
            if (i_local_part, i_remote_part) in connected_parts:
                tag_sum += count_tags(mesh, BTAG_PARTITION(i_remote_part))
        assert num_tags[i_remote_part] == tag_sum,\
                "part_mesh has the wrong number of BTAG_PARTITION boundaries"


def count_tags(mesh, tag):
    num_bnds = 0
    for fagrp_list in mesh.facial_adjacency_groups:
        matching_bdry_grps = [
            fagrp for fagrp in fagrp_list
            if isinstance(fagrp, BoundaryAdjacencyGroup)
            and fagrp.boundary_tag == tag]
        for bdry_grp in matching_bdry_grps:
            num_bnds += len(bdry_grp.elements)
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
        meshes = [generate_warped_rect_mesh(dim, order=order, nelements_side=4)
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

    group_factory = default_simplex_group_factory(base_dim=dim, order=order)

    from arraycontext import PyOpenCLArrayContext
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue, force_device_scalars=True)

    from meshmode.discretization import Discretization
    vol_discr = Discretization(actx, local_mesh, group_factory)

    from meshmode.distributed import get_connected_partitions
    connected_parts = get_connected_partitions(local_mesh)

    # Check that the connectivity makes sense before doing any communication
    _test_connected_parts(mpi_comm, connected_parts)

    from meshmode.discretization.connection import make_face_restriction
    from meshmode.mesh import BTAG_PARTITION
    local_bdry_conns = {}
    for i_remote_part in connected_parts:
        local_bdry_conns[i_remote_part] = make_face_restriction(
                actx, vol_discr, group_factory, BTAG_PARTITION(i_remote_part))

    remote_to_local_bdry_conns = {}
    with MPIBoundaryCommSetupHelper(mpi_comm, actx, local_bdry_conns,
            bdry_grp_factory=group_factory) as bdry_setup_helper:
        from meshmode.discretization.connection import check_connection
        while True:
            conns = bdry_setup_helper.complete_some()
            if not conns:
                break
            for i_remote_part, conn in conns.items():
                check_connection(actx, conn)
                remote_to_local_bdry_conns[i_remote_part] = conn

    _test_data_transfer(mpi_comm,
                        actx,
                        local_bdry_conns,
                        remote_to_local_bdry_conns,
                        connected_parts)

    logger.debug("Rank %d exiting", i_local_part)


def _test_connected_parts(mpi_comm, connected_parts):
    num_parts = mpi_comm.Get_size()
    i_local_part = mpi_comm.Get_rank()

    assert i_local_part not in connected_parts

    # Get the full adjacency
    connected_mask = np.empty(num_parts, dtype=bool)
    connected_mask[:] = False
    for i_remote_part in connected_parts:
        connected_mask[i_remote_part] = True
    all_connected_masks = mpi_comm.allgather(connected_mask)

    # Construct a list of parts that have the local part in their adjacency and
    # make sure it agrees with connected_parts
    parts_connected_to_me = set()
    for i_remote_part in range(num_parts):
        if all_connected_masks[i_remote_part][i_local_part]:
            parts_connected_to_me.add(i_remote_part)
    assert parts_connected_to_me == connected_parts


# TODO
def _test_data_transfer(mpi_comm, actx, local_bdry_conns,
                        remote_to_local_bdry_conns, connected_parts):
    from mpi4py import MPI

    def f(x):
        return 10*actx.np.sin(20.*x)

    # Here is a simplified example of what happens from
    # the point of view of the local rank.
    #
    # Local rank:
    #     1. Transfer local points from local boundary to remote boundary
    #         to get remote points.
    #     2. Send remote points to remote rank.
    # Remote rank:
    #     3. Receive remote points from local rank.
    #     4. Transfer remote points from remote boundary to local boundary
    #         to get local points.
    #     5. Send local points to local rank.
    # Local rank:
    #     6. Receive local points from remote rank.
    #     7. Check if local points are the same as the original local points.

    # 1.
    send_reqs = []
    for i_remote_part in connected_parts:
        conn = remote_to_local_bdry_conns[i_remote_part]
        bdry_discr = local_bdry_conns[i_remote_part].to_discr
        bdry_x = thaw(bdry_discr.nodes()[0], actx)

        true_local_f = f(bdry_x)
        remote_f = conn(true_local_f)

        # 2.
        send_reqs.append(mpi_comm.isend(
            actx.to_numpy(flatten(remote_f, actx)),
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

        local_f = unflatten(
                thaw(conn.from_discr.nodes()[0], actx),
                actx.from_numpy(remote_to_local_f_data[i_remote_part]),
                actx)
        remote_f = actx.to_numpy(flatten(conn(local_f), actx))

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
        bdry_x = thaw(bdry_discr.nodes()[0], actx)

        true_local_f = actx.to_numpy(flatten(f(bdry_x), actx))
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
        "mpiexec",
        "--oversubscribe",
        "-np", str(num_ranks),
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
