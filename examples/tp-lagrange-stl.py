# Requires numpy-stl
# pip install numpy-stl

import numpy as np
import numpy.linalg as la

import modepy
import pyopencl as cl

import meshmode.discretization.connection as conn
import meshmode.discretization.poly_element as poly
import meshmode.mesh.generation as mgen
from meshmode.array_context import PyOpenCLArrayContext
from meshmode.discretization import Discretization
from meshmode.mesh import BTAG_ALL, Mesh


def main():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue, force_device_scalars=True)

    npts1d = 100  # use 300 for actual print
    rs_coords = np.linspace(-1, 1, npts1d)
    mesh = mgen.generate_box_mesh((
                rs_coords,
                rs_coords,
                np.linspace(-0.1, 0, 10),
                ))

    group_factory = poly.PolynomialWarpAndBlend2DRestrictingGroupFactory(1)
    discr = Discretization(actx, mesh, group_factory)

    frestr = conn.make_face_restriction(actx, discr, group_factory, BTAG_ALL)

    bdry_mesh = frestr.to_discr.mesh

    order = 10
    hc_shape = modepy.Hypercube(2)
    space = modepy.QN(2, order)
    basis = modepy.basis_for_space(space, hc_shape)

    from modepy.quadrature.jacobi_gauss import legendre_gauss_lobatto_nodes
    nodes_1d, = legendre_gauss_lobatto_nodes(order, force_dim_axis=True)

    nodes = modepy.tensor_product_nodes(2, nodes_1d)
    vdm_inv = la.inv(modepy.vandermonde(basis.functions, nodes))

    vertices = bdry_mesh.vertices

    top_surface = np.abs(vertices[2] - 0) < 1e-12

    rs = vertices[:2, top_surface]

    val = 0*rs[0]

    for i, bf in enumerate(basis.functions):
        val += 0.5*vdm_inv[i, len(nodes_1d)*4 + 3] * bf(rs)

    for i, bf in enumerate(basis.functions):
        val += 0.5*vdm_inv[i, len(nodes_1d)*6 + 4] * bf(rs)

    for r1d in nodes_1d:
        r1d_discrete = rs_coords[np.argmin(np.abs(r1d - rs_coords))]
        assert abs(r1d - r1d_discrete) < 2/npts1d

        dimple = np.zeros_like(val)
        at_node_r = np.abs(rs[0] - r1d_discrete) < 1e-12
        dimple[at_node_r] = 1

        at_node_s = np.abs(rs[1] - r1d_discrete) < 1e-12
        dimple[at_node_s] = 1

        val -= 0.005 * dimple

    vertices[2, top_surface] = 0.1+val

    grp, = bdry_mesh.groups
    from meshmode.mesh.generation import make_group_from_vertices
    mod_grp = make_group_from_vertices(vertices, grp.vertex_indices, order=grp.order)

    mod_mesh = Mesh(
            vertices=vertices, groups=[mod_grp],
            is_conforming=bdry_mesh.is_conforming)

    from meshmode.mesh.visualization import write_stl_file
    write_stl_file(mod_mesh, "tp-lagrange.stl")


if __name__ == "__main__":
    main()

# vim: foldmethod=marker
