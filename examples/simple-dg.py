__copyright__ = "Copyright (C) 2020 Andreas Kloeckner"

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
import numpy.linalg as la  # noqa

import pyopencl as cl
import pyopencl.array as cla  # noqa

from pytools import memoize_method, log_process
from pytools.obj_array import flat_obj_array, make_obj_array

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.dof_array import DOFArray, flat_norm
from meshmode.array_context import (PyOpenCLArrayContext,
        SingleGridWorkBalancingPytatoArrayContext as PytatoPyOpenCLArrayContext)
from arraycontext import (
        ArrayContainer,
        map_array_container,
        with_container_arithmetic,
        dataclass_array_container,
        )
from meshmode.transform_metadata import FirstAxisIsElementsTag

import logging
logger = logging.getLogger(__name__)


# Features lost vs. https://github.com/inducer/grudge:
# - dimension independence / differential geometry
# - overintegration
# - operator fusion
# - distributed-memory


# {{{ discretization

def parametrization_derivative(actx, discr):
    thawed_nodes = actx.thaw(discr.nodes())

    from meshmode.discretization import num_reference_derivative
    result = np.zeros((discr.ambient_dim, discr.dim), dtype=object)
    for iambient in range(discr.ambient_dim):
        for idim in range(discr.dim):
            result[iambient, idim] = num_reference_derivative(discr,
                    (idim,), thawed_nodes[iambient])

    return result


class DGDiscretization:
    def __init__(self, actx, mesh, order):
        self.order = order

        from meshmode.discretization import Discretization
        from meshmode.discretization.poly_element import \
                PolynomialWarpAndBlend2DRestrictingGroupFactory
        self.group_factory = PolynomialWarpAndBlend2DRestrictingGroupFactory(
                order=order)
        self.volume_discr = Discretization(actx, mesh, self.group_factory)

        assert self.volume_discr.dim == 2

    @property
    def _setup_actx(self):
        return self.volume_discr._setup_actx

    @property
    def dim(self):
        return self.volume_discr.dim

    # {{{ discretizations/connections

    @memoize_method
    def boundary_connection(self, boundary_tag):
        from meshmode.discretization.connection import make_face_restriction
        return make_face_restriction(
                        self.volume_discr._setup_actx,
                        self.volume_discr,
                        self.group_factory,
                        boundary_tag=boundary_tag)

    @memoize_method
    def interior_faces_connection(self):
        from meshmode.discretization.connection import (
                make_face_restriction, FACE_RESTR_INTERIOR)
        return make_face_restriction(
                        self.volume_discr._setup_actx,
                        self.volume_discr,
                        self.group_factory,
                        FACE_RESTR_INTERIOR,
                        per_face_groups=False)

    @memoize_method
    def opposite_face_connection(self):
        from meshmode.discretization.connection import \
                make_opposite_face_connection

        return make_opposite_face_connection(
                self._setup_actx, self.interior_faces_connection())

    @memoize_method
    def all_faces_connection(self):
        from meshmode.discretization.connection import (
                make_face_restriction, FACE_RESTR_ALL)
        return make_face_restriction(
                        self.volume_discr._setup_actx,
                        self.volume_discr,
                        self.group_factory,
                        FACE_RESTR_ALL,
                        per_face_groups=False)

    @memoize_method
    def get_to_all_face_embedding(self, where):
        from meshmode.discretization.connection import \
                make_face_to_all_faces_embedding

        faces_conn = self.get_connection("vol", where)
        return make_face_to_all_faces_embedding(
                self._setup_actx, faces_conn, self.get_discr("all_faces"))

    def get_connection(self, src, tgt):
        src_tgt = (src, tgt)

        if src_tgt == ("vol", "int_faces"):
            return self.interior_faces_connection()
        elif src_tgt == ("vol", "all_faces"):
            return self.all_faces_connection()
        elif src_tgt == ("vol", BTAG_ALL):
            return self.boundary_connection(tgt)
        elif src_tgt == ("int_faces", "all_faces"):
            return self.get_to_all_face_embedding(src)
        elif src_tgt == (BTAG_ALL, "all_faces"):
            return self.get_to_all_face_embedding(src)
        else:
            raise ValueError(f"locations '{src}'->'{tgt}' not understood")

    def interp(self, src, tgt, vec):
        return self.get_connection(src, tgt)(vec)

    def get_discr(self, where):
        if where == "vol":
            return self.volume_discr
        elif where == "all_faces":
            return self.all_faces_connection().to_discr
        elif where == "int_faces":
            return self.interior_faces_connection().to_discr
        elif where == BTAG_ALL:
            return self.boundary_connection(where).to_discr
        else:
            raise ValueError(f"location '{where}' not understood")

    # }}}

    @memoize_method
    def parametrization_derivative(self):
        return self._setup_actx.freeze(
                parametrization_derivative(self._setup_actx, self.volume_discr))

    @memoize_method
    def vol_jacobian(self):
        [a, b], [c, d] = self._setup_actx.thaw(self.parametrization_derivative())
        return self._setup_actx.freeze(a*d - b*c)

    @memoize_method
    def inverse_parametrization_derivative(self):
        [a, b], [c, d] = self._setup_actx.thaw(self.parametrization_derivative())

        result = np.zeros((2, 2), dtype=object)
        det = a*d-b*c
        result[0, 0] = d/det
        result[0, 1] = -b/det
        result[1, 0] = -c/det
        result[1, 1] = a/det

        return self._setup_actx.freeze(result)

    def zeros(self, actx):
        return self.volume_discr.zeros(actx)

    def grad(self, vec):
        ipder = vec.array_context.thaw(self.inverse_parametrization_derivative())

        from meshmode.discretization import num_reference_derivative
        dref = [
                num_reference_derivative(self.volume_discr, (idim,), vec)
                for idim in range(self.volume_discr.dim)]

        return make_obj_array([
            sum(dref_i*ipder_i for dref_i, ipder_i in zip(dref, ipder[iambient]))
            for iambient in range(self.volume_discr.ambient_dim)])

    def div(self, vecs):
        return sum(
                self.grad(vec_i)[i] for i, vec_i in enumerate(vecs))

    @memoize_method
    def normal(self, where):
        bdry_discr = self.get_discr(where)

        ((a,), (b,)) = parametrization_derivative(self._setup_actx, bdry_discr)

        nrm = 1/(a**2+b**2)**0.5
        return self._setup_actx.freeze(flat_obj_array(b*nrm, -a*nrm))

    @memoize_method
    def face_jacobian(self, where):
        bdry_discr = self.get_discr(where)

        ((a,), (b,)) = parametrization_derivative(self._setup_actx, bdry_discr)

        return self._setup_actx.freeze((a**2 + b**2)**0.5)

    @memoize_method
    def get_inverse_mass_matrix(self, grp, dtype):
        import modepy as mp
        matrix = mp.inverse_mass_matrix(
                grp.basis_obj().functions,
                grp.unit_nodes)

        actx = self._setup_actx
        return actx.freeze(actx.from_numpy(matrix))

    def inverse_mass(self, vec):
        if not isinstance(vec, DOFArray):
            return map_array_container(self.inverse_mass, vec)

        actx = vec.array_context
        dtype = vec.entry_dtype
        discr = self.volume_discr

        return DOFArray(
            actx,
            data=tuple(
                actx.einsum(
                    "ij,ej->ei",
                    self.get_inverse_mass_matrix(grp, dtype),
                    vec_i,
                    arg_names=("mass_inv_mat", "vec"),
                    tagged=(FirstAxisIsElementsTag(),)
                ) for grp, vec_i in zip(discr.groups, vec)
            )
        ) / actx.thaw(self.vol_jacobian())

    @memoize_method
    def get_local_face_mass_matrix(self, afgrp, volgrp, dtype):
        nfaces = volgrp.mesh_el_group.nfaces
        assert afgrp.nelements == nfaces * volgrp.nelements

        matrix = np.empty(
                (volgrp.nunit_dofs,
                    nfaces,
                    afgrp.nunit_dofs),
                dtype=dtype)

        import modepy as mp
        shape = mp.Simplex(volgrp.dim)
        unit_vertices = mp.unit_vertices_for_shape(shape).T

        for face in mp.faces_for_shape(shape):
            face_vertices = unit_vertices[np.array(face.volume_vertex_indices)].T
            matrix[:, face.face_index, :] = mp.nodal_face_mass_matrix(
                    volgrp.basis_obj().functions,
                    volgrp.unit_nodes, afgrp.unit_nodes,
                    volgrp.order,
                    face_vertices)

        actx = self._setup_actx
        return actx.freeze(actx.from_numpy(matrix))

    def face_mass(self, vec):
        if not isinstance(vec, DOFArray):
            return map_array_container(self.face_mass, vec)

        actx = vec.array_context
        dtype = vec.entry_dtype

        all_faces_conn = self.get_connection("vol", "all_faces")
        all_faces_discr = all_faces_conn.to_discr
        vol_discr = all_faces_conn.from_discr

        fj = vec.array_context.thaw(self.face_jacobian("all_faces"))
        vec = vec*fj

        assert len(all_faces_discr.groups) == len(vol_discr.groups)

        return DOFArray(
            actx,
            data=tuple(
                actx.einsum("ifj,fej->ei",
                            self.get_local_face_mass_matrix(afgrp, volgrp, dtype),
                            vec_i.reshape(
                                volgrp.mesh_el_group.nfaces,
                                volgrp.nelements,
                                afgrp.nunit_dofs
                            ),
                            tagged=(FirstAxisIsElementsTag(),))
                for afgrp, volgrp, vec_i in zip(all_faces_discr.groups,
                                                vol_discr.groups, vec)
            )
        )

# }}}


# {{{ trace pair

@with_container_arithmetic(
        bcast_obj_array=False, eq_comparison=False, rel_comparison=False)
@dataclass_array_container
@dataclass(frozen=True)
class TracePair:
    where: str
    interior: ArrayContainer
    exterior: ArrayContainer

    def __getattr__(self, name):
        return map_array_container(
                lambda ary: getattr(ary, name),
                self)

    @property
    def int(self):
        return self.interior

    @property
    def ext(self):
        return self.exterior

    @property
    def avg(self):
        return 0.5*(self.int + self.ext)


def interior_trace_pair(discr, vec):
    i = discr.interp("vol", "int_faces", vec)
    e = discr.opposite_face_connection()(i)
    return TracePair("int_faces", interior=i, exterior=e)

# }}}


# {{{ wave equation bits

def wave_flux(actx, discr, c, q_tpair):
    u = q_tpair.u
    v = q_tpair.v

    normal = actx.thaw(discr.normal(q_tpair.where))

    flux_weak = WaveState(
            u=np.dot(v.avg, normal),
            v=normal * u.avg)

    # upwind
    v_jump = np.dot(normal, v.ext-v.int)
    flux_weak += WaveState(
            u=0.5*(u.ext-u.int),
            v=0.5*normal*v_jump)

    flux_strong = WaveState(
            u=np.dot(v.int, normal),
            v=u.int * normal,
            ) - flux_weak

    return discr.interp(q_tpair.where, "all_faces", c*flux_strong)


def wave_operator(actx, discr, c, q):
    dir_q = discr.interp("vol", BTAG_ALL, q)
    dir_bc = WaveState(u=-dir_q.u, v=dir_q.v)

    return (
            WaveState(
                u=c*discr.div(q.v),
                v=c*discr.grad(q.u)
                )
            -  # noqa: W504
            discr.inverse_mass(
                discr.face_mass(
                    wave_flux(actx, discr, c=c,
                        q_tpair=interior_trace_pair(discr, q))
                    + wave_flux(actx, discr, c=c,
                        q_tpair=TracePair(BTAG_ALL, dir_q, dir_bc))
                    ))
                )

# }}}


def rk4_step(y, t, h, f):
    k1 = f(t, y)
    k2 = f(t+h/2, y + h/2*k1)
    k3 = f(t+h/2, y + h/2*k2)
    k4 = f(t+h, y + h*k3)
    return y + h/6*(k1 + 2*k2 + 2*k3 + k4)


def bump(actx, discr, t=0):
    source_center = np.array([0.0, 0.05])
    source_width = 0.05
    source_omega = 3

    nodes = actx.thaw(discr.volume_discr.nodes())
    center_dist = flat_obj_array([
        nodes[0] - source_center[0],
        nodes[1] - source_center[1],
        ])

    return (
        np.cos(source_omega*t)
        * actx.np.exp(
            -np.dot(center_dist, center_dist)
            / source_width**2))


@with_container_arithmetic(bcast_obj_array=True, rel_comparison=True)
@dataclass_array_container
@dataclass(frozen=True)
class WaveState:
    u: DOFArray
    v: np.ndarray  # [object]

    def __post_init__(self):
        assert isinstance(self.v, np.ndarray) and self.v.dtype.char == "O"

    @property
    def array_context(self):
        return self.u.array_context


@log_process(logger)
def main(lazy=False):
    logging.basicConfig(level=logging.INFO)

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    if lazy:
        actx = PytatoPyOpenCLArrayContext(queue)
    else:
        actx = PyOpenCLArrayContext(queue, force_device_scalars=True)

    nel_1d = 16
    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(-0.5, -0.5),
            b=(0.5, 0.5),
            nelements_per_axis=(nel_1d, nel_1d))

    order = 3

    # no deep meaning here, just a fudge factor
    dt = 0.7/(nel_1d*order**2)

    logger.info("%d elements", mesh.nelements)

    discr = DGDiscretization(actx, mesh, order=order)

    fields = WaveState(
            u=bump(actx, discr),
            v=make_obj_array([discr.zeros(actx) for i in range(discr.dim)]),
            )

    from meshmode.discretization.visualization import make_visualizer
    vis = make_visualizer(actx, discr.volume_discr)

    def rhs(t, q):
        return wave_operator(actx, discr, c=1, q=q)

    compiled_rhs = actx.compile(rhs)

    t = np.float64(0)
    t_final = 3
    istep = 0
    while t < t_final:
        fields = actx.thaw(actx.freeze(fields,))
        fields = rk4_step(fields, t, dt, compiled_rhs)

        if istep % 10 == 0:
            # FIXME: Maybe an integral function to go with the
            # DOFArray would be nice?
            assert len(fields.u) == 1
            logger.info("[%05d] t %.5e / %.5e norm %.5e",
                    istep, t, t_final, actx.to_numpy(flat_norm(fields.u, 2)))
            vis.write_vtk_file("fld-wave-min-%04d.vtu" % istep, [
                ("q", fields),
                ])

        t += dt
        istep += 1

    assert actx.to_numpy(flat_norm(fields.u, 2)) < 100


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Wave Equation Solver")
    parser.add_argument("--lazy", action="store_true",
                        help="switch to a lazy computation mode")
    args = parser.parse_args()
    main(lazy=args.lazy)

# vim: foldmethod=marker
