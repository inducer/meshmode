from mpi4py import MPI
import numpy as np
import pyopencl

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

num_parts = 3
if rank == 0:
    np.random.seed(42)
    from meshmode.mesh.generation import generate_warped_rect_mesh
    meshes = [generate_warped_rect_mesh(3, order=4, n=5) for _ in range(2)]

    from meshmode.mesh.processing import merge_disjoint_meshes
    mesh = merge_disjoint_meshes(meshes)

    part_per_element = np.random.randint(num_parts, size=mesh.nelements)

    from meshmode.mesh.processing import partition_mesh
    parts = [partition_mesh(mesh, part_per_element, i)[0] for i in range(num_parts)]

    reqs = []
    for r in range(num_parts):
        reqs.append(comm.isend(parts[r], dest=r+1, tag=1))
    print('Sent all mesh parts.')
    for req in reqs:
        req.wait()

elif (rank - 1) in range(num_parts):
    mesh = comm.recv(source=0, tag=1)
    print('Recieved mesh')

    cl_ctx = pyopencl.create_some_context()

    from meshmode.discretization.poly_element\
                    import PolynomialWarpAndBlendGroupFactory
    group_factory = PolynomialWarpAndBlendGroupFactory(4)

    from meshmode.discretization import Discretization
    vol_discr = Discretization(cl_ctx, mesh, group_factory)

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
        local_mesh = local_bdry_conns[i_remote_part].from_discr.mesh
        local_adj_groups = [local_mesh.facial_adjacency_groups[i][None]
                            for i in range(len(local_mesh.groups))]
        local_batches = [local_bdry_conns[i_remote_part].groups[i].batches
                            for i in range(len(local_mesh.groups))]
        local_data = {'bdry': local_bdry,
                      'adj': local_adj_groups,
                      'batches': local_batches}
        send_reqs.append(comm.isend(local_data, dest=i_remote_part+1, tag=2))

    recv_reqs = {}
    for i_remote_part in range(num_parts):
        if i_local_part == i_remote_part:
            continue
        recv_reqs[i_remote_part] = comm.irecv(source=i_remote_part+1, tag=2)

    remote_data = {}
    for i_part, req in recv_reqs.items():
        remote_data[i_part] = req.wait()
    for req in send_reqs:
        req.wait()


    connection = {}
    for i_remote_part, data in remote_data.items():
        if data is None:
            # Local mesh is not connected to remote mesh
            continue
        remote_bdry = data['bdry']
        remote_adj_groups =data['adj']
        remote_batches = data['batches']
        # Connect local_mesh to remote_mesh
        from meshmode.discretization.connection import make_partition_connection
        connection[i_remote_part] =\
                    make_partition_connection(local_bdry_conns[i_remote_part],
                                              i_local_part,
                                              remote_bdry,
                                              remote_adj_groups,
                                              remote_batches)
        from meshmode.discretization.connection import check_connection
        check_connection(connection[i_remote_part])

