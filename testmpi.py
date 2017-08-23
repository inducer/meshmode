import numpy as np

num_parts = 3

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    np.random.seed(42)
    from meshmode.mesh.generation import generate_warped_rect_mesh
    meshes = [generate_warped_rect_mesh(3, order=4, n=5) for _ in range(2)]

    from meshmode.mesh.processing import merge_disjoint_meshes
    mesh = merge_disjoint_meshes(meshes)

    part_per_element = np.random.randint(num_parts, size=mesh.nelements)

    from meshmode.mesh.processing import partition_mesh
    parts = [partition_mesh(mesh, part_per_element, i)[0]
                    for i in range(num_parts)]

    reqs = []
    for r in range(num_parts):
        reqs.append(comm.isend(parts[r], dest=r+1, tag=1))
    print('Sent all mesh partitions.')
    for req in reqs:
        req.wait()

elif (rank - 1) in range(num_parts):
    status = MPI.Status()
    local_mesh = comm.recv(source=0, tag=1, status=status)
    print('Recieved mesh (size = {0})'.format(status.count))

    from meshmode.discretization.poly_element\
                    import PolynomialWarpAndBlendGroupFactory
    group_factory = PolynomialWarpAndBlendGroupFactory(4)
    import pyopencl as cl
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    from meshmode.discretization import Discretization
    vol_discr = Discretization(cl_ctx, local_mesh, group_factory)

    send_reqs = []
    i_local_part = rank - 1
    local_bdry_conns = {}
    for i_remote_part in range(num_parts):
        if i_local_part == i_remote_part:
            continue
        # Mark faces within local_mesh that are connected to remote_mesh
        from meshmode.discretization.connection import make_face_restriction
        from meshmode.mesh import BTAG_PARTITION
        local_bdry_conns[i_remote_part] =\
                make_face_restriction(vol_discr, group_factory,
                                      BTAG_PARTITION(i_remote_part))

    for i_remote_part in range(num_parts):
        if i_local_part == i_remote_part:
            continue
        bdry_nodes = local_bdry_conns[i_remote_part].to_discr.nodes()
        if bdry_nodes.size == 0:
            # local_mesh is not connected to remote_mesh, send None
            send_reqs.append(comm.isend(None, dest=i_remote_part+1, tag=2))
            continue

        # Gather information to send to other ranks
        local_bdry = local_bdry_conns[i_remote_part].to_discr
        local_adj_groups = [local_mesh.facial_adjacency_groups[i][None]
                            for i in range(len(local_mesh.groups))]
        local_batches = [local_bdry_conns[i_remote_part].groups[i].batches
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
        send_reqs.append(comm.isend(local_data, dest=i_remote_part+1, tag=2))

    recv_reqs = {}
    for i_remote_part in range(num_parts):
        if i_local_part == i_remote_part:
            continue
        status = MPI.Status()
        #TODO: Send size of data before to allocate a buffer.
        recv_reqs[i_remote_part] = comm.irecv(buf=1000000,
                                              source=i_remote_part+1,
                                              tag=2)
    remote_data = {}
    for i_part, req in recv_reqs.items():
        remote_data[i_part] = req.wait(status=status)
        print('Received remote data (size = {0})'.format(status.count))
    for req in send_reqs:
        req.wait()

    connection = {}
    for i_remote_part, data in remote_data.items():
        if data is None:
            # Local mesh is not connected to remote mesh
            continue
        remote_bdry_mesh = data['bdry_mesh']
        remote_bdry = Discretization(cl_ctx, remote_bdry_mesh, group_factory)
        remote_adj_groups = data['adj']
        remote_to_elem_faces = data['to_elem_faces']
        remote_to_elem_indices = data['to_elem_indices']
        # Connect local_mesh to remote_mesh
        from meshmode.discretization.connection import make_partition_connection
        connection[i_remote_part] =\
                    make_partition_connection(local_bdry_conns[i_remote_part],
                                              i_local_part,
                                              remote_bdry,
                                              remote_adj_groups,
                                              remote_to_elem_faces,
                                              remote_to_elem_indices)
        from meshmode.discretization.connection import check_connection
        check_connection(connection[i_remote_part])
