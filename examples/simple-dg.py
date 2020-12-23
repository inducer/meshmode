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


import numpy as np
import numpy.linalg as la  # noqa
import pyopencl as cl
import pyopencl.array as cla  # noqa
from pytools import memoize_method, memoize_in
from pytools.obj_array import (
        flat_obj_array, make_obj_array,
        obj_array_vectorize)
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.dof_array import freeze, thaw
from meshmode.array_context import PyOpenCLArrayContext, make_loopy_program


# Features lost vs. https://github.com/inducer/grudge:
# - dimension independence / differential geometry
# - overintegration
# - operator fusion
# - distributed-memory


# {{{ discretization

def parametrization_derivative(actx, discr):
    thawed_nodes = thaw(actx, discr.nodes())

    result = np.zeros((discr.ambient_dim, discr.dim), dtype=object)
    for iambient in range(discr.ambient_dim):
        for idim in range(discr.dim):
            result[iambient, idim] = discr.num_reference_derivative(
                    (idim,), thawed_nodes[iambient])

    return result


class DGDiscretization:
    def __init__(self, actx, mesh, order):
        self.order = order

        from meshmode.discretization import Discretization
        from meshmode.discretization.poly_element import \
                PolynomialWarpAndBlendGroupFactory
        self.group_factory = PolynomialWarpAndBlendGroupFactory(order=order)
        self.volume_discr = Discretization(actx, mesh, self.group_factory)

        assert self.volume_discr.dim == 2

    @property
    def _setup_actx(self):
        return self.volume_discr._setup_actx

    @property
    def array_context(self):
        return self.volume_discr.array_context

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
        if isinstance(vec, np.ndarray):
            return obj_array_vectorize(
                    lambda el: self.interp(src, tgt, el), vec)

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
        return freeze(
                parametrization_derivative(self._setup_actx, self.volume_discr))

    @memoize_method
    def vol_jacobian(self):
        [a, b], [c, d] = thaw(self._setup_actx, self.parametrization_derivative())
        return freeze(a*d-b*c)

    @memoize_method
    def inverse_parametrization_derivative(self):
        [a, b], [c, d] = thaw(self._setup_actx, self.parametrization_derivative())

        result = np.zeros((2, 2), dtype=object)
        det = a*d-b*c
        result[0, 0] = d/det
        result[0, 1] = -b/det
        result[1, 0] = -c/det
        result[1, 1] = a/det

        return freeze(result)

    def zeros(self, actx):
        return self.volume_discr.zeros(actx)

    def grad(self, vec):
        ipder = self.inverse_parametrization_derivative()

        dref = [
                self.volume_discr.num_reference_derivative((idim,), vec)
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
        return freeze(flat_obj_array(b*nrm, -a*nrm))

    @memoize_method
    def face_jacobian(self, where):
        bdry_discr = self.get_discr(where)

        ((a,), (b,)) = parametrization_derivative(self._setup_actx, bdry_discr)

        return freeze((a**2 + b**2)**0.5)

    @memoize_method
    def get_inverse_mass_matrix(self, grp, dtype):
        import modepy as mp
        matrix = mp.inverse_mass_matrix(
                grp.basis(),
                grp.unit_nodes)

        actx = self._setup_actx
        return actx.freeze(actx.from_numpy(matrix))

    def inverse_mass(self, vec):
        if isinstance(vec, np.ndarray):
            return obj_array_vectorize(
                    lambda el: self.inverse_mass(el), vec)

        @memoize_in(self, "elwise_linear_knl")
        def knl():
            return make_loopy_program(
                """{[iel,idof,j]:
                    0<=iel<nelements and
                    0<=idof<ndiscr_nodes_out and
                    0<=j<ndiscr_nodes_in}""",
                "result[iel,idof] = sum(j, mat[idof, j] * vec[iel, j])",
                name="diff")

        discr = self.volume_discr

        result = discr.empty_like(vec)

        for grp in discr.groups:
            matrix = self.get_inverse_mass_matrix(grp, vec.entry_dtype)

            vec.array_context.call_loopy(
                    knl(),
                    mat=matrix, result=result[grp.index], vec=vec[grp.index])

        return result/self.vol_jacobian()

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
                    volgrp.basis(), volgrp.unit_nodes, afgrp.unit_nodes,
                    volgrp.order,
                    face_vertices)

        actx = self._setup_actx
        return actx.freeze(actx.from_numpy(matrix))

    def face_mass(self, vec):
        if isinstance(vec, np.ndarray):
            return obj_array_vectorize(lambda el: self.face_mass(el), vec)

        @memoize_in(self, "face_mass_knl")
        def knl():
            return make_loopy_program(
                """{[iel,idof,f,j]:
                    0<=iel<nelements and
                    0<=f<nfaces and
                    0<=idof<nvol_nodes and
                    0<=j<nface_nodes}""",
                "result[iel,idof] = "
                "sum(f, sum(j, mat[idof, f, j] * vec[f, iel, j]))",
                name="face_mass")

        all_faces_conn = self.get_connection("vol", "all_faces")
        all_faces_discr = all_faces_conn.to_discr
        vol_discr = all_faces_conn.from_discr

        result = vol_discr.empty_like(vec)

        fj = self.face_jacobian("all_faces")
        vec = vec*fj

        assert len(all_faces_discr.groups) == len(vol_discr.groups)

        for afgrp, volgrp in zip(all_faces_discr.groups, vol_discr.groups):
            nfaces = volgrp.mesh_el_group.nfaces

            matrix = self.get_local_face_mass_matrix(afgrp, volgrp, vec.entry_dtype)

            vec.array_context.call_loopy(knl(),
                    mat=matrix, result=result[volgrp.index],
                    vec=vec[afgrp.index].reshape(
                        nfaces, volgrp.nelements, afgrp.nunit_dofs))

        return result

# }}}


# {{{ trace pair

class TracePair:
    def __init__(self, where, interior, exterior):
        self.where = where
        self.interior = interior
        self.exterior = exterior

    def __getitem__(self, index):
        return TracePair(
                self.where,
                self.exterior[index],
                self.interior[index])

    def __len__(self):
        assert len(self.exterior) == len(self.interior)
        return len(self.exterior)

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
    e = obj_array_vectorize(
            lambda el: discr.opposite_face_connection()(el),
            i)
    return TracePair("int_faces", i, e)

# }}}


# {{{ wave equation bits

def wave_flux(actx, discr, c, w_tpair):
    u = w_tpair[0]
    v = w_tpair[1:]

    normal = thaw(actx, discr.normal(w_tpair.where))

    flux_weak = flat_obj_array(
            np.dot(v.avg, normal),
            normal[0] * u.avg,
            normal[1] * u.avg)

    # upwind
    v_jump = np.dot(normal, v.int-v.ext)
    flux_weak -= flat_obj_array(
            0.5*(u.int-u.ext),
            0.5*normal[0]*v_jump,
            0.5*normal[1]*v_jump,
            )

    flux_strong = flat_obj_array(
            np.dot(v.int, normal),
            u.int * normal[0],
            u.int * normal[1],
            ) - flux_weak

    return discr.interp(w_tpair.where, "all_faces", c*flux_strong)


def wave_operator(actx, discr, c, w):
    u = w[0]
    v = w[1:]

    dir_u = discr.interp("vol", BTAG_ALL, u)
    dir_v = discr.interp("vol", BTAG_ALL, v)
    dir_bval = flat_obj_array(dir_u, dir_v)
    dir_bc = flat_obj_array(-dir_u, dir_v)

    return (
            - flat_obj_array(
                -c*discr.div(v),
                -c*discr.grad(u)
                )
            +  # noqa: W504
            discr.inverse_mass(
                discr.face_mass(
                    wave_flux(actx, discr, c=c,
                        w_tpair=interior_trace_pair(discr, w))
                    + wave_flux(actx, discr, c=c,
                        w_tpair=TracePair(BTAG_ALL, dir_bval, dir_bc))
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

    nodes = thaw(actx, discr.volume_discr.nodes())
    center_dist = flat_obj_array([
        nodes[0] - source_center[0],
        nodes[1] - source_center[1],
        ])

    return (
        np.cos(source_omega*t)
        * actx.np.exp(
            -np.dot(center_dist, center_dist)
            / source_width**2))


def main():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    actx = PyOpenCLArrayContext(queue)

    nel_1d = 16
    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(-0.5, -0.5),
            b=(0.5, 0.5),
            n=(nel_1d, nel_1d))

    order = 3

    # no deep meaning here, just a fudge factor
    dt = 0.75/(nel_1d*order**2)

    print("%d elements" % mesh.nelements)

    discr = DGDiscretization(actx, mesh, order=order)

    fields = flat_obj_array(
            bump(actx, discr),
            [discr.zeros(actx) for i in range(discr.dim)]
            )

    from meshmode.discretization.visualization import make_visualizer
    vis = make_visualizer(actx, discr.volume_discr, discr.order+3)

    def rhs(t, w):
        return wave_operator(actx, discr, c=1, w=w)

    t = 0
    t_final = 3
    istep = 0
    while t < t_final:
        fields = rk4_step(fields, t, dt, rhs)

        if istep % 10 == 0:
            # FIXME: Maybe an integral function to go with the
            # DOFArray would be nice?
            assert len(fields[0]) == 1
            print(istep, t, la.norm(actx.to_numpy(fields[0][0])))
            vis.write_vtk_file("fld-wave-min-%04d.vtu" % istep,
                    [
                        ("u", fields[0]),
                        ("v", fields[1:]),
                        ])

        t += dt
        istep += 1


if __name__ == "__main__":
    main()

# vim: foldmethod=marker
