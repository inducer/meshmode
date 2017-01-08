from __future__ import division, absolute_import

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

from six.moves import range
import six

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
.. autoclass:: FacialAdjacencyGroup

.. autofunction:: as_python
.. autofunction:: check_bc_coverage
.. autofunction:: is_boundary_tag_empty

Predefined Boundary tags
------------------------

.. autoclass:: BTAG_NONE
.. autoclass:: BTAG_ALL
.. autoclass:: BTAG_REALLY_ALL
.. autoclass:: BTAG_NO_BOUNDARY
"""


# {{{ element tags

class BTAG_NONE(object):  # noqa
    """A boundary tag representing an empty boundary or volume."""
    pass


class BTAG_ALL(object):  # noqa
    """A boundary tag representing the entire boundary or volume.

    In the case of the boundary, TAG_ALL does not include rank boundaries,
    or, more generally, anything tagged with TAG_NO_BOUNDARY."""
    pass


class BTAG_REALLY_ALL(object):  # noqa
    """A boundary tag representing the entire boundary.

    Unlike :class:`TAG_ALL`, this includes rank boundaries,
    or, more generally, everything tagged with :class:`TAG_NO_BOUNDARY`."""
    pass


class BTAG_NO_BOUNDARY(object):  # noqa
    """A boundary tag indicating that this edge should not fall under
    :class:`TAG_ALL`. Among other things, this is used to keep rank boundaries
    out of :class:`BTAG_ALL`.
    """
    pass


SYSTEM_TAGS = set([BTAG_NONE, BTAG_ALL, BTAG_REALLY_ALL, BTAG_NO_BOUNDARY])

# }}}


# {{{ element group

# {{{ base class

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

    .. automethod:: face_vertex_indices
    .. automethod:: vertex_unit_coordinates

    .. attribute:: nfaces

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

    def face_vertex_indices(self):
        """Return a tuple of tuples indicating which vertices
        (in mathematically positive ordering) make up each face
        of an element in this group.
        """
        raise NotImplementedError()

    def vertex_unit_coordinates(self):
        """Return an array of shape ``(nfaces, dim)`` with the unit
        coordinates of each vertex.
        """
        raise NotImplementedError()

    @property
    def nfaces(self):
        return len(self.face_vertex_indices())

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

# }}}


# {{{ simplex

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

            if dim <= 3:
                unit_nodes = mp.warp_and_blend_nodes(dim, order)
            else:
                unit_nodes = mp.equidistant_nodes(dim, order)

        dims = unit_nodes.shape[0]

        if vertex_indices.shape[-1] != dims+1:
            raise ValueError("vertex_indices has wrong number of vertices per "
                    "element. expected: %d, got: %d" % (dims+1,
                        vertex_indices.shape[-1]))

        super(SimplexElementGroup, self).__init__(order, vertex_indices, nodes,
                element_nr_base, node_nr_base, unit_nodes, dim)

    def face_vertex_indices(self):
        if self.dim == 1:
            return (
                (0,),
                (1,),
                )
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
        from modepy.tools import unit_vertices
        return unit_vertices(self.dim)

# }}}


# {{{ tensor-product

class TensorProductElementGroup(MeshElementGroup):
    def __init__(self, order, vertex_indices, nodes,
            element_nr_base=None, node_nr_base=None,
            unit_nodes=None):
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

        if not issubclass(vertex_indices.dtype.type, np.integer):
            raise TypeError("vertex_indices must be integral")

        dims = unit_nodes.shape[0]

        if vertex_indices.shape[-1] != 2**dims:
            raise ValueError("vertex_indices has wrong number of vertices per "
                    "element. expected: %d, got: %d" % (2**dims,
                        vertex_indices.shape[-1]))

        super(TensorProductElementGroup, self).__init__(order, vertex_indices, nodes,
                element_nr_base, node_nr_base, unit_nodes)

    def face_vertex_indices(self):
        raise NotImplementedError()

    def vertex_unit_coordinates(self):
        raise NotImplementedError()

# }}}

# }}}


# {{{ nodal adjacency

class NodalAdjacency(Record):
    """Describes nodal element adjacency information, i.e. information about
    elements that touch in at least one point.

    .. attribute:: neighbors_starts

        ``element_id_t [nelements+1]``

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

# }}}


# {{{ facial adjacency

class FacialAdjacencyGroup(Record):
    """Describes facial element adjacency information for one
    :class:`MeshElementGroup`, i.e. information about elements that share (part
    of) a face.

    .. attribute:: igroup

    .. attribute:: ineighbor_group

        ID of neighboring group, or *None* for boundary faces. If identical
        to :attr:`igroup`, then this contains the self-connectivity in this
        group.

    .. attribute:: elements

        ``element_id_t [nfagrp_elements]``. ``elements[i]``
        Group-local element numbers.

    .. attribute:: element_faces

        ``face_id_t [nfagrp_elements]``. ``element_faces[i]``
        indicate what face index of the opposite element indicated in
        ``neighbors[iel_grp][iface]`` touches face number *iface* of element
        number *iel_grp* in this element group.

    .. attribute:: neighbors

        ``element_id_t [nfagrp_elements]``. ``neighbors[i]``
        gives the element number within :attr:`ineighbor_group` of the element
        opposite ``elements[i]``.

        If this number is negative, then this indicates that this is a
        boundary face, and the bits set in ``-neighbors[i]``
        should be interpreted according to :attr:`Mesh.boundary_tags`.

    .. attribute:: neighbor_faces

        ``face_id_t [nfagrp_elements]``. ``neighbor_faces[i]`` indicate what
        face index of the opposite element indicated in ``neighbors[i]`` has
        facial contact with face number ``element_faces[i]`` of element number
        ``elements[i]`` in element group *igroup*.

        Zero if ``neighbors[i]`` is negative.

    .. automethod:: __eq__
    .. automethod:: __ne__
    """

    def __eq__(self, other):
        return (
                type(self) == type(other)
                and self.igroup == other.igroup
                and self.ineighbor_group == other.ineighbor_group
                and np.array_equal(self.elements, other.elements)
                and np.array_equal(self.element_faces, other.element_faces)
                and np.array_equal(self.neighbors, other.neighbors)
                and np.array_equal(self.neighbor_faces, other.neighbor_faces)
                )

    def __ne__(self, other):
        return not self.__eq__(other)

# }}}


# {{{ mesh

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

    .. attribute:: facial_adjacency_groups

        A list of mappings from neighbor group IDs to instances of
        :class:`FacialAdjacencyGroup`.

        ``facial_adjacency_groups[igrp][ineighbor_group]`` gives
        the set of facial adjacency relations between group *igrp*
        and *ineighbor_group*. *ineighbor_group* and *igrp* may be
        identical, or *ineighbor_group* may be *None*, in which case
        a group containing boundary faces is returned.

        Referencing this attribute may raise
        :exc:`meshmode.DataUnavailable`.

    .. attribute:: boundary_tags

        A tuple of boundary tag identifiers. :class:`BTAG_ALL` and
        :class:`BTAG_REALLY_ALL` are guranateed to exist.

    .. attribute:: btag_to_index

        A mapping that maps boundary tag identifiers to their
        corresponding index.

    .. attribute:: vertex_id_dtype

    .. attribute:: element_id_dtype

    .. automethod:: __eq__
    .. automethod:: __ne__
    """

    face_id_dtype = np.int8

    def __init__(self, vertices, groups, skip_tests=False,
            node_vertex_consistency_tolerance=None,
            nodal_adjacency=False,
            facial_adjacency_groups=False,
            boundary_tags=None,
            vertex_id_dtype=np.int32,
            element_id_dtype=np.int32):
        """
        The following are keyword-only:

        :arg skip_tests: Skip mesh tests, in case you want to load a broken
            mesh anyhow and then fix it inside of this data structure.
        :arg node_vertex_consistency_tolerance: If *False*, do not check
            for consistency between vertex and nodal data. If *None*, use
            the (small, near FP-epsilon) tolerance.
        :arg nodal_adjacency: One of three options:
            *None*, in which case this information
            will be deduced from vertex adjacency. *False*, in which case
            this information will be marked unavailable (such as if there are
            hanging nodes in the geometry, so that vertex adjacency does not convey
            the full picture), and references to
            :attr:`element_neighbors_starts` and :attr:`element_neighbors`
            will result in exceptions. Lastly, a tuple
            :class:`NodalAdjacency` object.
        :arg facial_adjacency_groups: One of three options:
            *None*, in which case this information
            will be deduced from vertex adjacency. *False*, in which case
            this information will be marked unavailable (such as if there are
            hanging nodes in the geometry, so that vertex adjacency does not convey
            the full picture), and references to
            :attr:`element_neighbors_starts` and :attr:`element_neighbors`
            will result in exceptions. Lastly, a data structure as described in
            :attr:`facial_adjacency_groups` may be passed.
        """
        el_nr = 0
        node_nr = 0

        new_groups = []
        for g in groups:
            ng = g.join_mesh(el_nr, node_nr)
            new_groups.append(ng)
            el_nr += ng.nelements
            node_nr += ng.nnodes

        # {{{ boundary tags

        if boundary_tags is None:
            boundary_tags = []
        else:
            boundary_tags = boundary_tags[:]

        if BTAG_NONE in boundary_tags:
            raise ValueError("BTAG_NONE is not allowed to be part of "
                    "boundary_tags")
        if BTAG_ALL not in boundary_tags:
            boundary_tags.append(BTAG_ALL)
        if BTAG_REALLY_ALL not in boundary_tags:
            boundary_tags.append(BTAG_REALLY_ALL)

        max_boundary_tag_count = int(
                np.log(np.iinfo(element_id_dtype).max)/np.log(2))
        if len(boundary_tags) > max_boundary_tag_count:
            raise ValueError("too few bits in element_id_dtype to represent all "
                    "boundary tags")

        btag_to_index = dict(
                (btag, i) for i, btag in enumerate(boundary_tags))

        # }}}

        if nodal_adjacency is not False and nodal_adjacency is not None:
            if not isinstance(nodal_adjacency, NodalAdjacency):
                nb_starts, nbs = nodal_adjacency
                nodal_adjacency = NodalAdjacency(
                        neighbors_starts=nb_starts,
                        neighbors=nbs)

                del nb_starts
                del nbs

        Record.__init__(
                self, vertices=vertices, groups=new_groups,
                _nodal_adjacency=nodal_adjacency,
                _facial_adjacency_groups=facial_adjacency_groups,
                boundary_tags=boundary_tags,
                btag_to_index=btag_to_index,
                vertex_id_dtype=np.dtype(vertex_id_dtype),
                element_id_dtype=np.dtype(element_id_dtype),
                )

        if not skip_tests:
            assert _test_node_vertex_consistency(
                    self, node_vertex_consistency_tolerance)
            for g in self.groups:
                assert g.vertex_indices.dtype == self.vertex_id_dtype

            if nodal_adjacency:
                assert nodal_adjacency.neighbors_starts.shape == (self.nelements+1,)
                assert len(nodal_adjacency.neighbors.shape) == 1

                assert (nodal_adjacency.neighbors_starts.dtype
                        == self.element_id_dtype)
                assert nodal_adjacency.neighbors.dtype == self.element_id_dtype

            if facial_adjacency_groups:
                assert len(facial_adjacency_groups) == len(self.groups)
                for fagrp_map in facial_adjacency_groups:
                    for fagrp in six.itervalues(fagrp_map):
                        grp = self.groups[fagrp.igroup]

                        fvi = grp.face_vertex_indices()
                        assert fagrp.neighbors.dtype == self.element_id_dtype
                        assert fagrp.neighbors.shape == (
                                grp.nelements, len(fvi))
                        assert fagrp.neighbors.dtype == self.face_id_dtype
                        assert fagrp.neighbor_faces.shape == (
                                grp.nelements, len(fvi))

                        is_bdry = fagrp.neighbors < 0
                        assert ((1 << btag_to_index[BTAG_REALLY_ALL])
                                & fagrp.neighbors[is_bdry]).all(), \
                                    "boundary faces without BTAG_REALLY_ALL found"

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
    def nvertices(self):
        return self.vertices.shape[-1]

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

        return self._nodal_adjacency

    @property
    def facial_adjacency_groups(self):
        if self._facial_adjacency_groups is False:
            from meshmode import DataUnavailable
            raise DataUnavailable("facial_adjacency_groups")
        elif self._facial_adjacency_groups is None:
            self._facial_adjacency_groups = \
                    _compute_facial_adjacency_from_vertices(self)

        return self._facial_adjacency_groups

    def boundary_tag_bit(self, boundary_tag):
        try:
            return 1 << self.btag_to_index[boundary_tag]
        except KeyError:
            return 0

    def __eq__(self, other):
        return (
                type(self) == type(other)
                and np.array_equal(self.vertices, other.vertices)
                and self.groups == other.groups
                and self.vertex_id_dtype == other.vertex_id_dtype
                and self.element_id_dtype == other.element_id_dtype
                and (self._nodal_adjacency
                        == other._nodal_adjacency)
                and (self._facial_adjacency_groups
                        == other._facial_adjacency_groups)
                and self.boundary_tags == other.boundary_tags)

    def __ne__(self, other):
        return not self.__eq__(other)

    # Design experience: Try not to add too many global data structures to the
    # mesh. Let the element groups be responsible for that at the mesh level.
    #
    # There are more big, global structures on the discretization level.

# }}}


# {{{ node-vertex consistency test

def _test_node_vertex_consistency_simplex(mesh, mgrp, tol):
    if mgrp.nelements == 0:
        return True

    resampling_mat = mp.resampling_matrix(
            mp.simplex_best_available_basis(mgrp.dim, mgrp.order),
            mgrp.vertex_unit_coordinates().T,
            mgrp.unit_nodes)

    # dim, nelments, nnvertices
    map_vertices = np.einsum(
            "ij,dej->dei", resampling_mat, mgrp.nodes)

    grp_vertices = mesh.vertices[:, mgrp.vertex_indices]

    per_element_vertex_errors = np.sqrt(np.sum(
            np.sum((map_vertices - grp_vertices)**2, axis=0),
            axis=-1))

    if tol is None:
        tol = 1e3 * np.finfo(per_element_vertex_errors.dtype).eps

    from meshmode.mesh.processing import find_bounding_box

    bbox_min, bbox_max = find_bounding_box(mesh)
    size = la.norm(bbox_max-bbox_min)

    assert np.max(per_element_vertex_errors) < tol*size, \
            np.max(per_element_vertex_errors)

    return True


def _test_node_vertex_consistency(mesh, tol):
    """Ensure that order of by-index vertices matches that of mapped
    unit vertices.
    """

    if tol is False:
        return

    for mgrp in mesh.groups:
        if isinstance(mgrp, SimplexElementGroup):
            assert _test_node_vertex_consistency_simplex(mesh, mgrp, tol)
        else:
            from warnings import warn
            warn("not implemented: node-vertex consistency check for '%s'"
                    % type(mgrp).__name__)

    return True

# }}}


# {{{ vertex-based nodal adjacency

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


# {{{ vertex-based facial adjacency

def _compute_facial_adjacency_from_vertices(mesh):
    # FIXME Native code would make this faster

    # create face_map, which is a mapping of
    # (vertices on a face) ->
    #  [(igrp, iel_grp, face_idx) for elements bordering that face]
    face_map = {}
    for igrp, grp in enumerate(mesh.groups):
        for fid, face_vertex_indices in enumerate(grp.face_vertex_indices()):
            all_fvi = grp.vertex_indices[:, face_vertex_indices]

            for iel_grp, fvi in enumerate(all_fvi):
                face_map.setdefault(
                        frozenset(fvi), []).append((igrp, iel_grp, fid))

    # maps tuples (igrp, ineighbor_group) to number of elements
    group_count = {}
    for face_tuples in six.itervalues(face_map):
        if len(face_tuples) == 2:
            (igrp, _, _), (inb_grp, _, _) = face_tuples
            group_count[igrp, inb_grp] = group_count.get((igrp, inb_grp), 0) + 1
            group_count[inb_grp, igrp] = group_count.get((inb_grp, igrp), 0) + 1
        elif len(face_tuples) == 1:
            (igrp, _, _), = face_tuples
            group_count[igrp, None] = group_count.get((igrp, None), 0) + 1
        else:
            raise RuntimeError("unexpected number of adjacent faces")

    # {{{ build facial_adjacency_groups data structure, still empty

    facial_adjacency_groups = []
    for igroup in range(len(mesh.groups)):
        grp_map = {}
        facial_adjacency_groups.append(grp_map)

        bdry_count = group_count.get((igroup, None))
        if bdry_count is not None:
            elements = np.empty(bdry_count, dtype=mesh.element_id_dtype)
            element_faces = np.empty(bdry_count, dtype=mesh.face_id_dtype)
            neighbors = np.empty(bdry_count, dtype=mesh.element_id_dtype)
            neighbor_faces = np.zeros(bdry_count, dtype=mesh.face_id_dtype)

            neighbors.fill(-(
                    mesh.boundary_tag_bit(BTAG_ALL)
                    | mesh.boundary_tag_bit(BTAG_REALLY_ALL)))

            grp_map[None] = FacialAdjacencyGroup(
                    igroup=igroup,
                    ineighbor_group=None,
                    elements=elements,
                    element_faces=element_faces,
                    neighbors=neighbors,
                    neighbor_faces=neighbor_faces)

        for ineighbor_group in range(len(mesh.groups)):
            nb_count = group_count.get((igroup, ineighbor_group))
            if nb_count is not None:
                elements = np.empty(nb_count, dtype=mesh.element_id_dtype)
                element_faces = np.empty(nb_count, dtype=mesh.face_id_dtype)
                neighbors = np.empty(nb_count, dtype=mesh.element_id_dtype)
                neighbor_faces = np.empty(nb_count, dtype=mesh.face_id_dtype)

                grp_map[ineighbor_group] = FacialAdjacencyGroup(
                        igroup=igroup,
                        ineighbor_group=ineighbor_group,
                        elements=elements,
                        element_faces=element_faces,
                        neighbors=neighbors,
                        neighbor_faces=neighbor_faces)

    # }}}

    # maps tuples (igrp, ineighbor_group) to number of elements filled in group
    fill_count = {}
    for face_tuples in six.itervalues(face_map):
        if len(face_tuples) == 2:
            for (igroup, iel, iface), (inb_group, inb_el, inb_face) in [
                    (face_tuples[0], face_tuples[1]),
                    (face_tuples[1], face_tuples[0]),
                    ]:
                idx = fill_count.get((igrp, inb_grp), 0)
                fill_count[igrp, inb_grp] = idx + 1

                fagrp = facial_adjacency_groups[igroup][inb_grp]
                fagrp.elements[idx] = iel
                fagrp.element_faces[idx] = iface
                fagrp.neighbors[idx] = inb_el
                fagrp.neighbor_faces[idx] = inb_face

        elif len(face_tuples) == 1:
            (igroup, iel, iface), = face_tuples

            idx = fill_count.get((igrp, None), 0)
            fill_count[igrp, None] = idx + 1

            fagrp = facial_adjacency_groups[igroup][None]
            fagrp.elements[idx] = iel
            fagrp.element_faces[idx] = iface

        else:
            raise RuntimeError("unexpected number of adjacent faces")

    return facial_adjacency_groups

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
    cg("""
        # generated by meshmode.mesh.as_python

        import numpy as np
        from meshmode.mesh import (
            Mesh, MeshElementGroup, FacialAdjacencyGroup,
            BTAG_ALL, BTAG_REALLY_ALL)

        """)

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

        # {{{ facial adjacency groups

        def fagrp_params_str(fagrp):
            params = {
                    "igroup": fagrp.igroup,
                    "ineighbor_group": repr(fagrp.ineighbor_group),
                    "elements": _numpy_array_as_python(fagrp.elements),
                    "element_faces": _numpy_array_as_python(fagrp.element_faces),
                    "neighbors": _numpy_array_as_python(fagrp.neighbors),
                    "neighbor_faces": _numpy_array_as_python(fagrp.neighbor_faces),
                    }
            return ",\n    ".join("%s=%s" % (k, v) for k, v in six.iteritems(params))

        if mesh._facial_adjacency_groups:
            cg("facial_adjacency_groups = []")

            for igrp, fagrp_map in enumerate(mesh.facial_adjacency_groups):
                cg("facial_adjacency_groups.append({%s})" % ",\n    ".join(
                    "%r: FacialAdjacencyGroup(%s)" % (
                        inb_grp, fagrp_params_str(fagrp))
                    for inb_grp, fagrp in six.iteritems(fagrp_map)))

        else:
            cg("facial_adjacency_groups = %r" % mesh._facial_adjacency_groups)

        # }}}

        # {{{ boundary tags

        def strify_boundary_tag(btag):
            if isinstance(btag, type):
                return btag.__name__
            else:
                return repr(btag)

        btags_str = ", ".join(
                strify_boundary_tag(btag) for btag in mesh.boundary_tags)

        # }}}

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

        cg("    nodal_adjacency=%s," % el_con_str)
        cg("    facial_adjacency_groups=facial_adjacency_groups,")
        cg("    boundary_tags=[%s])" % btags_str)

        # FIXME: Handle facial adjacency, boundary tags

    return cg.get()

# }}}


# {{{ check_bc_coverage

def check_bc_coverage(mesh, boundary_tags, incomplete_ok=False):
    """Verify boundary condition coverage.

    Given a list of boundary tags as *boundary_tags*, this function verifies
    that

     1. the union of all these boundaries gives the complete boundary,
     1. all these boundaries are disjoint.

    :arg incomplete_ok: Do not report an error if some faces are not covered
      by the boundary conditions.
    """

    for igrp, fagrp_map in enumerate(mesh.facial_adjacency_groups):
        bdry_grp = fagrp_map.get(None)
        if bdry_grp is None:
            continue

        nb_elements = bdry_grp.neighbors
        assert (nb_elements < 0).all()

        nb_el_bits = -nb_elements

        seen = np.zeros_like(nb_el_bits, dtype=np.bool)

        for btag in boundary_tags:
            tag_bit = mesh.boundary_tag_bit(btag)
            tag_set = (nb_el_bits & tag_bit) != 0

            if (seen & tag_set).any():
                raise RuntimeError("faces with multiple boundary conditions found")

            seen = seen | tag_set

        if not incomplete_ok and not seen.all():
            raise RuntimeError("found faces without boundary conditions")

# }}}


# {{{ is_boundary_tag_empty

def is_boundary_tag_empty(mesh, boundary_tag):
    """Return *True* if the corresponding boundary tag does not occur as part of
    *mesh*.
    """

    btag_bit = mesh.boundary_tag_bit(boundary_tag)
    if not btag_bit:
        return True

    for igrp in range(len(mesh.groups)):
        bdry_fagrp = mesh.facial_adjacency_groups[igrp].get(None, None)
        if bdry_fagrp is None:
            continue

        neg = bdry_fagrp.neighbors < 0
        if (-bdry_fagrp.neighbors[neg] & btag_bit).any():
            return False

    return True

# }}}


# vim: foldmethod=marker
