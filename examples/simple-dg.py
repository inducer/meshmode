from __future__ import division, print_function

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
import pyopencl.clmath as clmath
from pytools import memoize_method, memoize_in
from pytools.obj_array import (
        join_fields, make_obj_array,
        with_object_array_or_scalar,
        is_obj_array)
import loopy as lp
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.discretization import DOFArray  # noqa: F401


# Features lost vs. https://github.com/inducer/grudge:
# - dimension independence / differential geometry
# - overintegration
# - operator fusion
# - distributed-memory


def with_queue(queue, ary):
    def dof_array_with_queue(dof_array):
        return dof_array.copy(group_arrays=[
            grp_ary.with_queue(queue)
            for grp_ary in dof_array.group_arrays])

    return with_object_array_or_scalar(dof_array_with_queue, ary)


def without_queue(ary):
    return with_queue(None, ary)


# {{{ discretization

def parametrization_derivative(queue, discr):
    result = np.zeros((discr.ambient_dim, discr.dim), dtype=object)
    for iambient in range(discr.ambient_dim):
        for idim in range(discr.dim):
            result[iambient, idim] = discr.num_reference_derivative(
                    queue, (idim,), discr.nodes()[iambient])

    return result


class DGDiscretization:
    def __init__(self, cl_ctx, mesh, order):
        self.order = order

        from meshmode.discretization import Discretization
        from meshmode.discretization.poly_element import \
                PolynomialWarpAndBlendGroupFactory
        self.group_factory = PolynomialWarpAndBlendGroupFactory(order=order)
        self.volume_discr = Discretization(cl_ctx, mesh, self.group_factory)

        assert self.volume_discr.dim == 2

    @property
    def cl_context(self):
        return self.volume_discr.cl_context

    @property
    def dim(self):
        return self.volume_discr.dim

    # {{{ discretizations/connections

    @memoize_method
    def boundary_connection(self, boundary_tag):
        from meshmode.discretization.connection import make_face_restriction
        return make_face_restriction(
                        self.volume_discr,
                        self.group_factory,
                        boundary_tag=boundary_tag)

    @memoize_method
    def interior_faces_connection(self):
        from meshmode.discretization.connection import (
                make_face_restriction, FACE_RESTR_INTERIOR)
        return make_face_restriction(
                        self.volume_discr,
                        self.group_factory,
                        FACE_RESTR_INTERIOR,
                        per_face_groups=False)

    @memoize_method
    def opposite_face_connection(self):
        from meshmode.discretization.connection import \
                make_opposite_face_connection

        return make_opposite_face_connection(self.interior_faces_connection())

    @memoize_method
    def all_faces_connection(self):
        from meshmode.discretization.connection import (
                make_face_restriction, FACE_RESTR_ALL)
        return make_face_restriction(
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
                faces_conn, self.get_discr("all_faces"))

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
        if is_obj_array(vec):
            return with_object_array_or_scalar(
                    lambda el: self.interp(src, tgt, el), vec)

        return self.get_connection(src, tgt)(vec.queue, vec)

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
        with cl.CommandQueue(self.cl_context) as queue:
            return without_queue(
                    parametrization_derivative(queue, self.volume_discr))

    @memoize_method
    def vol_jacobian(self):
        with cl.CommandQueue(self.cl_context) as queue:
            [a, b], [c, d] = with_queue(queue, self.parametrization_derivative())
            return (a*d-b*c).with_queue(None)

    @memoize_method
    def inverse_parametrization_derivative(self):
        with cl.CommandQueue(self.cl_context) as queue:
            [a, b], [c, d] = with_queue(queue, self.parametrization_derivative())

            result = np.zeros((2, 2), dtype=object)
            det = a*d-b*c
            result[0, 0] = d/det
            result[0, 1] = -b/det
            result[1, 0] = -c/det
            result[1, 1] = a/det

            return without_queue(result)

    def zeros(self, queue):
        return self.volume_discr.zeros(queue)

    def grad(self, vec):
        ipder = self.inverse_parametrization_derivative()

        queue = vec.queue
        dref = [
                self.volume_discr.num_reference_derivative(
                    queue, (idim,), vec).with_queue(queue)
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

        with cl.CommandQueue(self.cl_context) as queue:
            ((a,), (b,)) = with_queue(
                    queue, parametrization_derivative(queue, bdry_discr))

            nrm = 1/(a**2+b**2)**0.5
            return without_queue(join_fields(b*nrm, -a*nrm))

    @memoize_method
    def face_jacobian(self, where):
        bdry_discr = self.get_discr(where)

        with cl.CommandQueue(self.cl_context) as queue:
            ((a,), (b,)) = with_queue(queue,
                    parametrization_derivative(queue, bdry_discr))

            return ((a**2 + b**2)**0.5).with_queue(None)

    @memoize_method
    def get_inverse_mass_matrix(self, grp, dtype):
        import modepy as mp
        matrix = mp.inverse_mass_matrix(
                grp.basis(),
                grp.unit_nodes)

        with cl.CommandQueue(self.cl_context) as queue:
            return (cla.to_device(queue, matrix)
                    .with_queue(None))

    def inverse_mass(self, vec):
        if is_obj_array(vec):
            return with_object_array_or_scalar(
                    lambda el: self.inverse_mass(el), vec)

        @memoize_in(self, "elwise_linear_knl")
        def knl():
            knl = lp.make_kernel(
                """{[k,i,j]:
                    0<=k<nelements and
                    0<=i<ndiscr_nodes_out and
                    0<=j<ndiscr_nodes_in}""",
                "result[k,i] = sum(j, mat[i, j] * vec[k, j])",
                default_offset=lp.auto, name="diff")

            knl = lp.split_iname(knl, "i", 16, inner_tag="l.0")
            return lp.tag_inames(knl, dict(k="g.0"))

        discr = self.volume_discr

        result = discr.empty(queue=vec.queue, dtype=vec.dtype)

        for grp in discr.groups:
            matrix = self.get_inverse_mass_matrix(grp, vec.dtype)

            knl()(vec.queue, mat=matrix, result=grp.view(result),
                    vec=grp.view(vec))

        return result/self.vol_jacobian()

    @memoize_method
    def get_local_face_mass_matrix(self, afgrp, volgrp, dtype):
        nfaces = volgrp.mesh_el_group.nfaces
        assert afgrp.nelements == nfaces * volgrp.nelements

        matrix = np.empty(
                (volgrp.nunit_nodes,
                    nfaces,
                    afgrp.nunit_nodes),
                dtype=dtype)

        from modepy.tools import UNIT_VERTICES
        import modepy as mp
        for iface, fvi in enumerate(
                volgrp.mesh_el_group.face_vertex_indices()):
            face_vertices = UNIT_VERTICES[volgrp.dim][np.array(fvi)].T
            matrix[:, iface, :] = mp.nodal_face_mass_matrix(
                    volgrp.basis(), volgrp.unit_nodes, afgrp.unit_nodes,
                    volgrp.order,
                    face_vertices)

        with cl.CommandQueue(self.cl_context) as queue:
            return (cla.to_device(queue, matrix)
                    .with_queue(None))

    def face_mass(self, vec):
        if is_obj_array(vec):
            return with_object_array_or_scalar(
                    lambda el: self.face_mass(el), vec)

        @memoize_in(self, "face_mass_knl")
        def knl():
            knl = lp.make_kernel(
                """{[k,i,f,j]:
                    0<=k<nelements and
                    0<=f<nfaces and
                    0<=i<nvol_nodes and
                    0<=j<nface_nodes}""",
                "result[k,i] = sum(f, sum(j, mat[i, f, j] * vec[f, k, j]))",
                default_offset=lp.auto, name="face_mass")

            knl = lp.split_iname(knl, "i", 16, inner_tag="l.0")
            return lp.tag_inames(knl, dict(k="g.0"))

        all_faces_conn = self.get_connection("vol", "all_faces")
        all_faces_discr = all_faces_conn.to_discr
        vol_discr = all_faces_conn.from_discr

        result = vol_discr.empty(queue=vec.queue, dtype=vec.dtype)

        fj = self.face_jacobian("all_faces")
        vec = vec*fj

        assert len(all_faces_discr.groups) == len(vol_discr.groups)

        for afgrp, volgrp in zip(all_faces_discr.groups, vol_discr.groups):
            nfaces = volgrp.mesh_el_group.nfaces

            matrix = self.get_local_face_mass_matrix(afgrp, volgrp, vec.dtype)

            input_view = afgrp.view(vec).reshape(
                    nfaces, volgrp.nelements, afgrp.nunit_nodes)
            knl()(vec.queue, mat=matrix, result=volgrp.view(result),
                    vec=input_view)

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
    e = with_object_array_or_scalar(
            lambda el: discr.opposite_face_connection()(el.queue, el),
            i)
    return TracePair("int_faces", i, e)

# }}}


# {{{ wave equation bits

def wave_flux(discr, c, w_tpair):
    u = w_tpair[0]
    v = w_tpair[1:]

    normal = with_queue(u.int.queue, discr.normal(w_tpair.where))

    flux_weak = join_fields(
            np.dot(v.avg, normal),
            normal[0] * u.avg,
            normal[1] * u.avg)

    # upwind
    v_jump = np.dot(normal, v.int-v.ext)
    flux_weak -= join_fields(
            0.5*(u.int-u.ext),
            0.5*normal[0]*v_jump,
            0.5*normal[1]*v_jump,
            )

    flux_strong = join_fields(
            np.dot(v.int, normal),
            u.int * normal[0],
            u.int * normal[1],
            ) - flux_weak

    return discr.interp(w_tpair.where, "all_faces", c*flux_strong)


def wave_operator(discr, c, w):
    u = w[0]
    v = w[1:]

    dir_u = discr.interp("vol", BTAG_ALL, u)
    dir_v = discr.interp("vol", BTAG_ALL, v)
    dir_bval = join_fields(dir_u, dir_v)
    dir_bc = join_fields(-dir_u, dir_v)

    return (
            - join_fields(
                -c*discr.div(v),
                -c*discr.grad(u)
                )
            +  # noqa: W504
            discr.inverse_mass(
                discr.face_mass(
                    wave_flux(discr, c=c, w_tpair=interior_trace_pair(discr, w))
                    + wave_flux(discr, c=c, w_tpair=TracePair(
                        BTAG_ALL, dir_bval, dir_bc))
                    ))
                )


# }}}


def rk4_step(y, t, h, f):
    k1 = f(t, y)
    k2 = f(t+h/2, y + h/2*k1)
    k3 = f(t+h/2, y + h/2*k2)
    k4 = f(t+h, y + h*k3)
    return y + h/6*(k1 + 2*k2 + 2*k3 + k4)


def bump(discr, queue, t=0):
    source_center = np.array([0.0, 0.05])
    source_width = 0.05
    source_omega = 3

    nodes = with_queue(queue, discr.volume_discr.nodes())
    center_dist = join_fields([
        nodes[0] - source_center[0],
        nodes[1] - source_center[1],
        ])

    return (
        np.cos(source_omega*t)
        * clmath.exp(
            -np.dot(center_dist, center_dist)
            / source_width**2))


def main():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

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

    discr = DGDiscretization(cl_ctx, mesh, order=order)

    fields = join_fields(
            bump(discr, queue),
            [discr.zeros(queue) for i in range(discr.dim)]
            )

    from meshmode.discretization.visualization import make_visualizer
    vis = make_visualizer(queue, discr.volume_discr, discr.order+3)

    def rhs(t, w):
        return wave_operator(discr, c=1, w=w)

    t = 0
    t_final = 3
    istep = 0
    while t < t_final:
        fields = rk4_step(fields, t, dt, rhs)

        if istep % 10 == 0:
            print(istep, t, la.norm(fields[0].get()))
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
