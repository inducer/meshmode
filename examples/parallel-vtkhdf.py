import logging
import numpy as np
from meshmode.mesh import Mesh

logger = logging.getLogger(__file__)


def make_example_mesh(ambient_dim: int, nelements: int, order: int) -> Mesh:
    import meshmode.mesh.generation as mgen
    if ambient_dim == 2:
        return mgen.make_curve_mesh(
            mgen.starfish,
            np.linspace(0.0, 1.0, nelements + 1),
            order=order)
    else:
        return mgen.generate_torus(4.0, 2.0,
            n_major=nelements, n_minor=nelements // 2,
            order=order)


def main(*, ambient_dim: int) -> None:
    logging.basicConfig(level=logging.INFO)

    import h5py
    if "mpio" not in h5py.registered_drivers():
        logger.info("h5py does not have the 'mpio' driver")
        return

    from meshmode import _acf
    actx = _acf()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    mpisize = comm.Get_size()
    mpirank = comm.Get_rank()

    from meshmode.distributed import MPIMeshDistributor
    dist = MPIMeshDistributor(comm)

    order = 5
    nelements = 64 if ambient_dim == 3 else 256

    logger.info("[%4d] distributing mesh: started", mpirank)

    if dist.is_mananger_rank():
        mesh = make_example_mesh(ambient_dim, nelements, order=order)
        logger.info("[%4d] mesh: nelements %d nvertices %d",
                    mpirank, mesh.nelements, mesh.nvertices)

        rng = np.random.default_rng()
        part_per_element = rng.integers(mpisize, size=mesh.nelements)

        local_mesh = dist.send_mesh_parts(mesh, part_per_element, mpisize)
    else:
        local_mesh = dist.receive_mesh_part()

    logger.info("[%4d] distributing mesh: finished", mpirank)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import default_simplex_group_factory
    discr = Discretization(actx, local_mesh,
        default_simplex_group_factory(local_mesh.dim, order=order))

    logger.info("[%4d] discretization: finished", mpirank)

    from arraycontext import thaw
    vector_field = thaw(discr.nodes(), actx)
    scalar_field = actx.np.sin(thaw(discr.nodes()[0], actx))
    part_id = 1 + mpirank + discr.zeros(actx)
    logger.info("[%4d] fields: finished", mpirank)

    from meshmode.discretization.visualization import make_visualizer
    vis = make_visualizer(actx, discr, vis_order=order, force_equidistant=False)
    logger.info("[%4d] make_visualizer: finished", mpirank)

    filename = f"parallel-vtkhdf-example-{ambient_dim}d.hdf"
    vis.write_vtkhdf_file(filename, [
        ("scalar", scalar_field),
        ("vector", vector_field),
        ("part_id", part_id)
        ], comm=comm, overwrite=True, use_high_order=False)

    logger.info("[%4d] write: finished: %s", mpirank, filename)


if __name__ == "__main__":
    main(ambient_dim=2)
    main(ambient_dim=3)
