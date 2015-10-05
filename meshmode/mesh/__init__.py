from __future__ import division
from __future__ import absolute_import
from six.moves import range

__copyright__ = "Copyright (C) 2010,2012,2013 Andreas Kloeckner, Michael Tom"

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
import modepy as mp
import numpy.linalg as la
from pytools import Record

__doc__ = """

.. autoclass:: MeshElementGroup
    :members:
    :undoc-members:

.. autoclass:: Mesh
    :members:
    :undoc-members:

.. autoclass:: NodalAdjacency

.. autofunction:: as_python

"""


# {{{ element group

class MeshElementGroup(Record):
    """A group of elements sharing a common reference element.

    .. attribute:: order

    .. attribute:: vertex_indices

        An array *(nelements, ref_element.nvertices)* of (mesh-wide)
        vertex indices.

    .. attribute:: nodes

        An array of node coordinates with shape
        *(mesh.ambient_dim, nelements, nunit_nodes)*.

    .. attribute:: unit_nodes

        *(dim, nunit_nodes)*

    .. attribute:: element_nr_base

        Lowest element number in this element group.

    .. attribute:: node_nr_base

        Lowest node number in this element group.

    .. attribute:: dim

        The number of dimensions spanned by the element.
        *Not* the ambient dimension, see :attr:`Mesh.ambient_dim`
        for that.

    .. automethod:: __eq__
    .. automethod:: __ne__
    """

    def __init__(self, order, vertex_indices, nodes,
            element_nr_base=None, node_nr_base=None,
            unit_nodes=None, dim=None):
        """
        :arg order: the mamximum total degree used for interpolation.
        :arg nodes: ``[ambient_dim, nelements, nunit_nodes]``
            The nodes are assumed to be mapped versions of *unit_nodes*.
        :arg unit_nodes: ``[dim, nunit_nodes]``
            The unit nodes of which *nodes* is a mapped
            version.

        Do not supply *element_nr_base* and *node_nr_base*, they will be
        automatically assigned.
        """

        Record.__init__(self,
            order=order,
            vertex_indices=vertex_indices,
            nodes=nodes,
            unit_nodes=unit_nodes,
            element_nr_base=element_nr_base, node_nr_base=node_nr_base)

    def copy(self, **kwargs):
        if "element_nr_base" not in kwargs:
            kwargs["element_nr_base"] = None
        if "node_nr_base" not in kwargs:
            kwargs["node_nr_base"] = None
        return Record.copy(self, **kwargs)

    @property
    def dim(self):
        return self.unit_nodes.shape[0]

    def join_mesh(self, element_nr_base, node_nr_base):
        if self.element_nr_base is not None:
            raise RuntimeError("this element group has already joined a mesh, "
                    "cannot join another")

        return self.copy(
                element_nr_base=element_nr_base,
                node_nr_base=node_nr_base)

    @property
    def nelements(self):
        return self.vertex_indices.shape[0]

    @property
    def nnodes(self):
        return self.nelements * self.unit_nodes.shape[-1]

    @property
    def nunit_nodes(self):
        return self.unit_nodes.shape[-1]

    def __eq__(self, other):
        return (
                type(self) == type(other)
                and self.order == other.order
                and np.array_equal(self.vertex_indices, other.vertex_indices)
                and np.array_equal(self.nodes, other.nodes)
                and np.array_equal(self.unit_nodes, other.unit_nodes)
                and self.element_nr_base == other.element_nr_base
                and self.node_nr_base == other.node_nr_base)

    def __ne__(self, other):
        return not self.__eq__(other)


class SimplexElementGroup(MeshElementGroup):
    def __init__(self, order, vertex_indices, nodes,
            element_nr_base=None, node_nr_base=None,
            unit_nodes=None, dim=None):
        """
        :arg order: the mamximum total degree used for interpolation.
        :arg nodes: ``[ambient_dim, nelements, nunit_nodes]``
            The nodes are assumed to be mapped versions of *unit_nodes*.
        :arg unit_nodes: ``[dim, nunit_nodes]``
            The unit nodes of which *nodes* is a mapped
            version. If unspecified, the nodes from
            :func:`modepy.warp_and_blend_nodes` for *dim*
            are assumed. These must be in unit coordinates
            as defined in :mod:`modepy.nodes`.
        :arg dim: only used if *unit_nodes* is None, to get
            the default unit nodes.

        Do not supply *element_nr_base* and *node_nr_base*, they will be
        automatically assigned.
        """

        if not issubclass(vertex_indices.dtype.type, np.integer):
            raise TypeError("vertex_indices must be integral")

        if unit_nodes is None:
            if dim is None:
                raise TypeError("'dim' must be passed "
                        "if 'unit_nodes' is not passed")

            unit_nodes = mp.warp_and_blend_nodes(dim, order)

        dims = unit_nodes.shape[0]

        if vertex_indices.shape[-1] != dims+1:
            raise ValueError("vertex_indices has wrong number of vertices per "
                    "element. expected: %d, got: %d" % (dims+1,
                        vertex_indices.shape[-1]))

        MeshElementGroup.__init__(self, order, vertex_indices, nodes,
                element_nr_base, node_nr_base, unit_nodes, dim)

    def face_vertex_indices(self):
        if self.dim == 1:
            return ((0, 1),)
        elif self.dim == 2:
            return (
                (0, 1),
                (2, 0),
                (1, 2),
                )
        elif self.dim == 3:
            return (
                (0, 1, 2),
                (0, 3, 1),
                (0, 2, 3),
                (1, 3, 2)
                )
        else:
            raise NotImplementedError("dim=%d" % self.dim)

    def vertex_unit_coordinates(self):
        if self.dim == 1:
            return np.array([
                [-1], [1]
                ], dtype=np.float64)
        elif self.dim == 2:
            return np.array([
                [-1, -1],
                [1, -1],
                [-1, 1],
                ], dtype=np.float64)
        elif self.dim == 3:
            return np.array([
                [-1, -1, -1],
                [1, -1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                ], dtype=np.float64)
        else:
            raise NotImplementedError("dim=%d" % self.dim)

# }}}


# {{{ mesh

class NodalAdjacency(Record):
    """Describes element adjacency information, i.e. information about
    elements that touch in at least one point.

    .. attribute:: neighbors_starts

        ``element_id_t [nelments+1]``

        Use together with :attr:`neighbors`.  ``neighbors_starts[iel]`` and
        ``neighbors_starts[iel+1]`` together indicate a ranges of element indices
        :attr:`neighbors` which are adjacent to *iel*.

    .. attribute:: neighbors

        ``element_id_t []``

        See :attr:`neighbors_starts`.

    .. automethod:: __eq__
    .. automethod:: __ne__
    """

    def __eq__(self, other):
        return (
                type(self) == type(other)
                and np.array_equal(self.neighbors_starts,
                    other.neighbors_starts)
                and np.array_equal(self.neighbors, other.neighbors))

    def __ne__(self, other):
        return not self.__eq__(other)


class Mesh(Record):
    """
    .. attribute:: vertices

        An array of vertex coordinates with shape
        *(ambient_dim, nvertices)*

    .. attribute:: groups

        A list of :class:`MeshElementGroup` instances.

    .. attribute:: nodal_adjacency

        An instance of :class:`NodalAdjacency`.

        Referencing this attribute may raise
        :exc:`meshmode.DataUnavailable`.

    .. attribute:: vertex_id_dtype

    .. attribute:: element_id_dtype

    .. automethod:: __eq__
    .. automethod:: __ne__
    """

    def __init__(self, vertices, groups, skip_tests=False,
            nodal_adjacency=False,
            vertex_id_dtype=np.int32,
            element_id_dtype=np.int32):
        """
        The following are keyword-only:

        :arg skip_tests: Skip mesh tests, in case you want to load a broken
            mesh anyhow and then fix it inside of this data structure.
        :arg nodal_adjacency: One of three options:
            *None*, in which case this information
            will be deduced from vertex adjacency. *False*, in which case
            this information will be marked unavailable (such as if there are
            hanging nodes in the geometry, so that vertex adjacency does not convey
            the full picture), and references to
            :attr:`element_neighbors_starts` and :attr:`element_neighbors`
            will result in exceptions. Lastly, a tuple
            *(element_neighbors_starts, element_neighbors)*, representing the
            correspondingly-named attributes.
        """
        el_nr = 0
        node_nr = 0

        new_groups = []
        for g in groups:
            ng = g.join_mesh(el_nr, node_nr)
            new_groups.append(ng)
            el_nr += ng.nelements
            node_nr += ng.nnodes

        if nodal_adjacency is not False and nodal_adjacency is not None:
            nb_starts, nbs = nodal_adjacency
            nodal_adjacency = NodalAdjacency(
                    neighbors_starts=nb_starts,
                    neighbors=nbs)

            del nb_starts
            del nbs

        Record.__init__(
                self, vertices=vertices, groups=new_groups,
                _nodal_adjacency=nodal_adjacency,
                vertex_id_dtype=np.dtype(vertex_id_dtype),
                element_id_dtype=np.dtype(element_id_dtype),
                )

        if not skip_tests:
            assert _test_node_vertex_consistency(self)
            for g in self.groups:
                assert g.vertex_indices.dtype == self.vertex_id_dtype

            from meshmode.mesh.processing import \
                    test_volume_mesh_element_orientations

            if self.dim == self.ambient_dim:
                # only for volume meshes, for now
                assert test_volume_mesh_element_orientations(self), \
                        "negatively oriented elements found"

    @property
    def ambient_dim(self):
        return self.vertices.shape[0]

    @property
    def dim(self):
        from pytools import single_valued
        return single_valued(grp.dim for grp in self.groups)

    @property
    def nelements(self):
        return sum(grp.nelements for grp in self.groups)

    @property
    def nodal_adjacency(self):
        if self._nodal_adjacency is False:
            from meshmode import DataUnavailable
            raise DataUnavailable("nodal_adjacency")
        elif self._nodal_adjacency is None:
            self._nodal_adjacency = _compute_nodal_adjacency_from_vertices(self)

        return self._nodal_adjacency

    def nodal_adjacency_init_arg(self):
        """Returns an 'nodal_adjacency' argument that can be
        passed to a Mesh constructor.
        """

        if isinstance(self._nodal_adjacency, NodalAdjacency):
            return (self._nodal_adjacency.neighbors_starts,
                    self._nodal_adjacency.neighbors)
        else:
            return self._nodal_adjacency

    def __eq__(self, other):
        return (
                type(self) == type(other)
                and np.array_equal(self.vertices, other.vertices)
                and self.groups == other.groups
                and self.vertex_id_dtype == other.vertex_id_dtype
                and self.element_id_dtype == other.element_id_dtype
                and (self._nodal_adjacency
                        == other._nodal_adjacency))

    def __ne__(self, other):
        return not self.__eq__(other)

    # Design experience: Try not to add too many global data structures to the
    # mesh. Let the element groups be responsible for that at the mesh level.
    #
    # There are more big, global structures on the discretization level.

# }}}


# {{{ node-vertex consistency test

def _test_node_vertex_consistency_simplex(mesh, mgrp):
    from modepy.tools import UNIT_VERTICES
    resampling_mat = mp.resampling_matrix(
            mp.simplex_onb(mgrp.dim, mgrp.order),
            UNIT_VERTICES[mgrp.dim].T.copy(), mgrp.unit_nodes)

    # dim, nelments, nnvertices
    map_vertices = np.einsum(
            "ij,dej->dei", resampling_mat, mgrp.nodes)

    grp_vertices = mesh.vertices[:, mgrp.vertex_indices]

    per_element_vertex_errors = np.sqrt(np.sum(
            np.sum((map_vertices - grp_vertices)**2, axis=0),
            axis=-1))

    tol = 1e3 * np.finfo(per_element_vertex_errors.dtype).eps

    from meshmode.mesh.processing import find_bounding_box

    bbox_min, bbox_max = find_bounding_box(mesh)
    size = la.norm(bbox_max-bbox_min)

    assert np.max(per_element_vertex_errors) < tol*size, \
            np.max(per_element_vertex_errors)

    return True


def _test_node_vertex_consistency(mesh):
    """Ensure that order of by-index vertices matches that of mapped
    unit vertices.
    """

    for mgrp in mesh.groups:
        if isinstance(mgrp, SimplexElementGroup):
            assert _test_node_vertex_consistency_simplex(mesh, mgrp)
        else:
            from warnings import warn
            warn("not implemented: node-vertex consistency check for '%s'"
                    % type(mgrp).__name__)

    return True

# }}}


# {{{ vertex-based adjacency

def _compute_nodal_adjacency_from_vertices(mesh):
    # FIXME Native code would make this faster

    _, nvertices = mesh.vertices.shape
    vertex_to_element = [[] for i in range(nvertices)]

    for grp in mesh.groups:
        iel_base = grp.element_nr_base
        for iel_grp in range(grp.nelements):
            for ivertex in grp.vertex_indices[iel_grp]:
                vertex_to_element[ivertex].append(iel_base + iel_grp)

    element_to_element = [set() for i in range(mesh.nelements)]
    for grp in mesh.groups:
        iel_base = grp.element_nr_base
        for iel_grp in range(grp.nelements):
            for ivertex in grp.vertex_indices[iel_grp]:
                element_to_element[iel_base + iel_grp].update(
                        vertex_to_element[ivertex])

    for iel, neighbors in enumerate(element_to_element):
        neighbors.remove(iel)

    lengths = [len(el_list) for el_list in element_to_element]
    neighbors_starts = np.cumsum(
            np.array([0] + lengths, dtype=mesh.element_id_dtype))
    from pytools import flatten
    neighbors = np.array(
            list(flatten(element_to_element)),
            dtype=mesh.element_id_dtype)

    assert neighbors_starts[-1] == len(neighbors)

    return NodalAdjacency(
            neighbors_starts=neighbors_starts,
            neighbors=neighbors)

# }}}


# {{{ as_python

def _numpy_array_as_python(array):
    return "np.array(%s, dtype=np.%s)" % (
            repr(array.tolist()),
            array.dtype.name)


def as_python(mesh, function_name="make_mesh"):
    """Return a snippet of Python code (as a string) that will
    recreate the mesh given as an input parameter.
    """

    from pytools.py_codegen import PythonCodeGenerator, Indentation
    cg = PythonCodeGenerator()
    cg("# generated by meshmode.mesh.as_python")
    cg("")
    cg("import numpy as np")
    cg("from meshmode.mesh import Mesh, MeshElementGroup")
    cg("")

    cg("def %s():" % function_name)
    with Indentation(cg):
        cg("vertices = " + _numpy_array_as_python(mesh.vertices))
        cg("")
        cg("groups = []")
        cg("")
        for group in mesh.groups:
            cg("import %s" % type(group).__module__)
            cg("groups.append(%s.%s(" % (
                type(group).__module__,
                type(group).__name__))
            cg("    order=%s," % group.order)
            cg("    vertex_indices=%s,"
                    % _numpy_array_as_python(group.vertex_indices))
            cg("    nodes=%s,"
                    % _numpy_array_as_python(group.nodes))
            cg("    unit_nodes=%s))"
                    % _numpy_array_as_python(group.unit_nodes))

        cg("return Mesh(vertices, groups, skip_tests=True,")
        cg("    vertex_id_dtype=np.%s," % mesh.vertex_id_dtype.name)
        cg("    element_id_dtype=np.%s," % mesh.element_id_dtype.name)

        if isinstance(mesh._nodal_adjacency, NodalAdjacency):
            el_con_str = "(%s, %s)" % (
                    _numpy_array_as_python(
                        mesh._nodal_adjacency.neighbors_starts),
                    _numpy_array_as_python(
                        mesh._nodal_adjacency.neighbors),
                    )
        else:
            el_con_str = repr(mesh._nodal_adjacency)

        cg("    nodal_adjacency=%s)" % el_con_str)

    return cg.get()

# }}}


# vim: foldmethod=marker
