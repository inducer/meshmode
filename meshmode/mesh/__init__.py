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
import numpy.linalg as la

import modepy as mp
from pytools import Record, memoize_method

__doc__ = """

.. autoclass:: MeshElementGroup
.. autoclass:: SimplexElementGroup
.. autoclass:: TensorProductElementGroup

.. autoclass:: Mesh

.. autoclass:: NodalAdjacency
.. autoclass:: FacialAdjacencyGroup
.. autoclass:: InterPartitionAdjacencyGroup

.. autofunction:: as_python
.. autofunction:: check_bc_coverage
.. autofunction:: is_boundary_tag_empty

Predefined Boundary tags
------------------------

.. autoclass:: BTAG_NONE
.. autoclass:: BTAG_ALL
.. autoclass:: BTAG_REALLY_ALL
.. autoclass:: BTAG_NO_BOUNDARY
.. autoclass:: BTAG_PARTITION
.. autoclass:: BTAG_INDUCED_BOUNDARY
"""


# {{{ element tags

class BTAG_NONE:  # noqa: N801
    """A boundary tag representing an empty boundary or volume."""
    pass


class BTAG_ALL:  # noqa: N801
    """A boundary tag representing the entire boundary or volume.

    In the case of the boundary, :class:`BTAG_ALL` does not include rank boundaries,
    or, more generally, anything tagged with :class:`BTAG_NO_BOUNDARY`.

    In the case of a mesh representing an element-wise subset of another,
    :class:`BTAG_ALL` does not include boundaries induced by taking the subset.
    Instead, these boundaries will be tagged with
    :class:`BTAG_INDUCED_BOUNDARY`.
    """
    pass


class BTAG_REALLY_ALL:  # noqa: N801
    """A boundary tag representing the entire boundary.

    Unlike :class:`BTAG_ALL`, this includes rank boundaries,
    or, more generally, everything tagged with :class:`BTAG_NO_BOUNDARY`.

    In the case of a mesh representing an element-wise subset of another,
    this tag includes boundaries induced by taking the subset, or, more generally,
    everything tagged with
    :class:`BTAG_INDUCED_BOUNDARY`
    """
    pass


class BTAG_NO_BOUNDARY:  # noqa: N801
    """A boundary tag indicating that this edge should not fall under
    :class:`BTAG_ALL`. Among other things, this is used to keep rank boundaries
    out of :class:`BTAG_ALL`.
    """
    pass


class BTAG_PARTITION:  # noqa: N801
    """
    A boundary tag indicating that this edge is adjacent to an element of
    another :class:`Mesh`. The partition number of the adjacent mesh
    is given by ``part_nr``.

    .. attribute:: part_nr

    .. versionadded:: 2017.1
    """
    def __init__(self, part_nr):
        self.part_nr = int(part_nr)

    def __hash__(self):
        return hash((type(self), self.part_nr))

    def __eq__(self, other):
        if isinstance(other, BTAG_PARTITION):
            return self.part_nr == other.part_nr
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "<{}({})>".format(type(self).__name__, repr(self.part_nr))


class BTAG_INDUCED_BOUNDARY(BTAG_NO_BOUNDARY):  # noqa: N801
    """When a :class:`Mesh` is created as an element-by-element subset of another
    (as, for example, when using the Firedrake interop features
    while passing *restrict_to_boundary*), boundaries may arise where there
    were none in the original mesh. This boundary tag is used to indicate
    such boundaries.
    """
    # Don't be tempted to add a sphinx ref to the Firedrake stuff here.
    # This is unavailable in the Github doc build because
    # firedrakeproject.org seems to reject connections from Github.

    pass


SYSTEM_TAGS = {BTAG_NONE, BTAG_ALL, BTAG_REALLY_ALL, BTAG_NO_BOUNDARY,
                   BTAG_PARTITION, BTAG_INDUCED_BOUNDARY}

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

    .. attribute:: is_affine

        A :class:`bool` flag that is *True* if the local-to-global
        parametrization of all the elements in the group is affine.

    .. automethod:: face_vertex_indices
    .. automethod:: vertex_unit_coordinates

    .. attribute:: nfaces

    .. automethod:: __eq__
    .. automethod:: __ne__
    """

    def __init__(self, order, vertex_indices, nodes,
            element_nr_base=None, node_nr_base=None,
            unit_nodes=None, dim=None, **kwargs):
        """
        :arg order: the maximum total degree used for interpolation.
        :arg nodes: ``(ambient_dim, nelements, nunit_nodes)``
            The nodes are assumed to be mapped versions of *unit_nodes*.
        :arg unit_nodes: ``(dim, nunit_nodes)`` The unit nodes of which *nodes*
            is a mapped version.

        Do not supply *element_nr_base* and *node_nr_base*, they will be
        automatically assigned.
        """

        super().__init__(
            order=order,
            vertex_indices=vertex_indices,
            nodes=nodes,
            unit_nodes=unit_nodes,
            element_nr_base=element_nr_base, node_nr_base=node_nr_base,
            **kwargs)

    def get_copy_kwargs(self, **kwargs):
        if "element_nr_base" not in kwargs:
            kwargs["element_nr_base"] = None
        if "node_nr_base" not in kwargs:
            kwargs["node_nr_base"] = None

        return super().get_copy_kwargs(**kwargs)

    @property
    def dim(self):
        return self.unit_nodes.shape[0]

    def join_mesh(self, element_nr_base, node_nr_base):
        if self.element_nr_base is not None:
            raise RuntimeError("this element group has already joined a mesh, "
                    "cannot join another (The element group's element_nr_base "
                    "is already assigned, and that typically happens when a "
                    "group joins a Mesh instance.)")

        return self.copy(
                element_nr_base=element_nr_base,
                node_nr_base=node_nr_base)

    @property
    def nelements(self):
        return self.nodes.shape[1]

    @property
    def nnodes(self):
        return self.nelements * self.unit_nodes.shape[-1]

    @property
    def nunit_nodes(self):
        return self.unit_nodes.shape[-1]

    @property
    def is_affine(self):
        raise NotImplementedError

    def face_vertex_indices(self):
        """Return a tuple of tuples indicating which vertices
        (in mathematically positive ordering) make up each face
        of an element in this group.
        """
        raise NotImplementedError

    def vertex_unit_coordinates(self):
        """Return an array of shape ``(nfaces, dim)`` with the unit
        coordinates of each vertex.
        """
        raise NotImplementedError

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


# {{{ modepy-based element group

class _ModepyElementGroup(MeshElementGroup):
    def __init__(self, order, vertex_indices, nodes,
            element_nr_base=None, node_nr_base=None,
            unit_nodes=None, dim=None, **kwargs):
        """
        :arg order: the maximum total degree used for interpolation.
        :arg nodes: ``(ambient_dim, nelements, nunit_nodes)``
            The nodes are assumed to be mapped versions of *unit_nodes*.
        :arg unit_nodes: ``(dim, nunit_nodes)`` The unit nodes of which
            *nodes* is a mapped version. If unspecified, the nodes from
            :func:`modepy.edge_clustered_nodes_for_space` are assumed.
            These must be in unit coordinates as defined in :mod:`modepy`.
        :arg dim: only used if *unit_nodes* is *None*, to get
            the default unit nodes.

        Do not supply *element_nr_base* and *node_nr_base*, they will be
        automatically assigned.
        """

        if unit_nodes is not None:
            _dim = unit_nodes.shape[0]
            if dim is not None and _dim != dim:
                raise ValueError("'dim' does not match 'unit_nodes' dimension")
            else:
                dim = _dim
        else:
            if dim is None:
                raise TypeError("'dim' must be passed if 'unit_nodes' is not passed")

        # dim is now usable
        shape = self._modepy_shape_cls(dim)
        space = mp.space_for_shape(shape, order)

        if unit_nodes is None:
            unit_nodes = mp.edge_clustered_nodes_for_space(space, shape)

        if nodes is not None:
            if unit_nodes.shape[-1] != nodes.shape[-1]:
                raise ValueError(
                        "'nodes' has wrong number of unit nodes per element."
                        f" expected {unit_nodes.shape[-1]}, "
                        f" but got {nodes.shape[-1]}.")

        if vertex_indices is not None:
            if not issubclass(vertex_indices.dtype.type, np.integer):
                raise TypeError("'vertex_indices' must be integral")

            if vertex_indices.shape[-1] != shape.nvertices:
                raise ValueError(
                        "'vertex_indices' has wrong number of vertices per element."
                        f" expected {shape.nvertices},"
                        f" got {vertex_indices.shape[-1]}")

        super().__init__(order, vertex_indices, nodes,
                element_nr_base=element_nr_base,
                node_nr_base=node_nr_base,
                unit_nodes=unit_nodes,
                dim=dim,
                _modepy_shape=shape,
                _modepy_space=space)

    @property
    @memoize_method
    def _modepy_faces(self):
        return mp.faces_for_shape(self._modepy_shape)

    def face_vertex_indices(self):
        return tuple(face.volume_vertex_indices for face in self._modepy_faces)

    def vertex_unit_coordinates(self):
        return mp.unit_vertices_for_shape(self._modepy_shape).T

# }}}


class SimplexElementGroup(_ModepyElementGroup):
    _modepy_shape_cls = mp.Simplex

    @property
    @memoize_method
    def is_affine(self):
        return is_affine_simplex_group(self)


class TensorProductElementGroup(_ModepyElementGroup):
    _modepy_shape_cls = mp.Hypercube

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

    .. image:: images/facial-adjacency-group.png
        :align: center
        :width: 60%

    Represents (for example) *one* of the (colored) interfaces between
    :class:`MeshElementGroup` instances, or an interface between
    :class:`MeshElementGroup` and a boundary. (Note that element groups are not
    necessarily contiguous like the figure may suggest.)

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
        ``neighbors[iface]`` touches face number *iface* of element
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


# {{{ partition adjacency

class InterPartitionAdjacencyGroup(FacialAdjacencyGroup):
    """
    Describes boundary adjacency information of elements in one
    :class:`MeshElementGroup`.

    .. attribute:: igroup

        The group number of this group.

    .. attribute:: ineighbor_group

        *None* for boundary faces.

    .. attribute:: elements

        Group-local element numbers.
        Element ``element_id_dtype elements[i]`` and face
        ``face_id_dtype element_faces[i]`` is connected to neighbor element
        ``element_id_dtype partition_neighbors[i]`` with face
        ``face_id_dtype global_neighbor_faces[i]``. The partition number it connects
        to is ``neighbor_partitions[i]``.

    .. attribute:: element_faces

        ``face_id_dtype element_faces[i]`` gives the face of
        ``element_id_dtype elements[i]`` that is connected to
        ``partition_neighbors[i]``.

    .. attribute:: neighbors

        Since this is a boundary, ``element_id_dtype neighbors[i]`` is interpreted
        as a boundary tag. ``-neighbors[i]`` should be interpreted according to
        :class:``Mesh.boundary_tags``.

    .. attribute:: partition_neighbors

        ``element_id_dtype partition_neighbors[i]`` gives the volume element number
        within the neighboring partition of the element connected to
        ``element_id_dtype elements[i]`` (which is a boundary element index). Use
        `~meshmode.mesh.processing.find_group_indices` to find the group that
        the element belongs to, then subtract ``element_nr_base`` to find the
        element of the group.

        If ``partition_neighbors[i]`` is negative, ``elements[i]`` is on a true
        boundary and is not connected to any other :class:``Mesh``.

    .. attribute:: neighbor_faces

        ``face_id_dtype global_neighbor_faces[i]`` gives face index within the
        neighboring partition of the face connected to
        ``element_id_dtype elements[i]``

        If ``neighbor_partitions[i]`` is negative, ``elements[i]`` is on a true
        boundary and is not connected to any other :class:``Mesh``.

    .. attribute:: neighbor_partitions

        ``neighbor_partitions[i]`` gives the partition number that ``elements[i]``
        is connected to.

        If ``neighbor_partitions[i]`` is negative, ``elements[i]`` is on a true
        boundary and is not connected to any other :class:``Mesh``.

    .. versionadded:: 2017.1
    """

    def __eq__(self, other):
        return (super.__eq__(self, other)
            and np.array_equal(self.partition_neighbors, other.partition_neighbors)
            and np.array_equal(self.neighbor_partitions, other.neighbor_partitions))

# }}}


# {{{ mesh

class Mesh(Record):
    """
    .. attribute:: ambient_dim

    .. attribute:: dim

    .. attribute:: vertices

        *None* or an array of vertex coordinates with shape
        *(ambient_dim, nvertices)*. If *None*, vertices are not
        known for this mesh.

    .. attribute:: nvertices

    .. attribute:: groups

        A list of :class:`MeshElementGroup` instances.

    .. attribute:: nelements

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
        an :class:`InterPartitionAdjacencyGroup` group containing boundary
        faces is returned.

        Referencing this attribute may raise
        :exc:`meshmode.DataUnavailable`.

        .. image:: images/facial-adjacency-group.png
            :align: center
            :width: 60%

        For example for the mesh in the figure, the following data structure
        would be present::

            [
                {...},  # connectivity for group 0
                {...},  # connectivity for group 1
                {...},  # connectivity for group 2
                {       # connectivity for group 3
                    1: FacialAdjacencyGroup(...)  # towards group 1, green
                    2: FacialAdjacencyGroup(...)  # towards group 2, pink
                    None: FacialAdjacencyGroup(...)  # towards the boundary, orange
                }
            ]

        (Note that element groups are not necessarily geometrically contiguous
        like the figure may suggest.)

    .. attribute:: boundary_tags

        A list of boundary tag identifiers. :class:`BTAG_ALL` and
        :class:`BTAG_REALLY_ALL` are guaranteed to exist.

    .. attribute:: btag_to_index

        A mapping that maps boundary tag identifiers to their
        corresponding index.

        .. note::

            Elements of :attr:`boundary_tags` that do not cover any
            part of the boundary will not be keys in this dictionary.

    .. attribute:: vertex_id_dtype

    .. attribute:: element_id_dtype

    .. attribute:: is_conforming

        *True* if it is known that all element interfaces are conforming.
        *False* if it is known that some element interfaces are non-conforming.
        *None* if neither of the two is known.

    .. automethod:: __eq__
    .. automethod:: __ne__
    """

    face_id_dtype = np.int8

    def __init__(self, vertices, groups, skip_tests=False,
            node_vertex_consistency_tolerance=None,
            skip_element_orientation_test=False,
            nodal_adjacency=None,
            facial_adjacency_groups=None,
            boundary_tags=None,
            vertex_id_dtype=np.int32,
            element_id_dtype=np.int32,
            is_conforming=None):
        """
        The following are keyword-only:

        :arg skip_tests: Skip mesh tests, in case you want to load a broken
            mesh anyhow and then fix it inside of this data structure.
        :arg node_vertex_consistency_tolerance: If *False*, do not check
            for consistency between vertex and nodal data. If *None*, use
            the (small, near FP-epsilon) default tolerance.
        :arg skip_element_orientation_test: If *False*, check that
            element orientation is positive in volume meshes
            (i.e. ones where ambient and topological dimension match).
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

        btag_to_index = {
                btag: i for i, btag in enumerate(boundary_tags)}

        # }}}

        if vertices is None:
            is_conforming = None

        if not is_conforming:
            if nodal_adjacency is None:
                nodal_adjacency = False
            if facial_adjacency_groups is None:
                facial_adjacency_groups = False

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
                is_conforming=is_conforming,
                )

        if not skip_tests:
            if node_vertex_consistency_tolerance is not False:
                assert _test_node_vertex_consistency(
                        self, node_vertex_consistency_tolerance)

            for g in self.groups:
                if g.vertex_indices is not None:
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
                    for fagrp in fagrp_map.values():
                        nfagrp_elements, = fagrp.elements.shape

                        assert fagrp.element_faces.dtype == self.face_id_dtype
                        assert fagrp.element_faces.shape == (nfagrp_elements,)

                        assert fagrp.neighbors.dtype == self.element_id_dtype
                        assert fagrp.neighbors.shape == (nfagrp_elements,)

                        assert fagrp.neighbor_faces.dtype == self.face_id_dtype
                        assert fagrp.neighbor_faces.shape == (nfagrp_elements,)

                        if fagrp.ineighbor_group is None:
                            is_bdry = fagrp.neighbors < 0
                            assert ((1 << btag_to_index[BTAG_REALLY_ALL])
                                    & -fagrp.neighbors[is_bdry]).all(), \
                                    "boundary faces without BTAG_REALLY_ALL found"

            from meshmode.mesh.processing import \
                    test_volume_mesh_element_orientations

            if self.dim == self.ambient_dim and not skip_element_orientation_test:
                # only for volume meshes, for now
                assert test_volume_mesh_element_orientations(self), \
                        "negatively oriented elements found"

    def get_copy_kwargs(self, **kwargs):
        def set_if_not_present(name, from_name=None):
            if from_name is None:
                from_name = name
            if name not in kwargs:
                kwargs[name] = getattr(self, from_name)

        set_if_not_present("vertices")
        if "groups" not in kwargs:
            kwargs["groups"] = [group.copy() for group in self.groups]
        set_if_not_present("boundary_tags")
        set_if_not_present("nodal_adjacency", "_nodal_adjacency")
        set_if_not_present("facial_adjacency_groups", "_facial_adjacency_groups")
        set_if_not_present("boundary_tags")
        set_if_not_present("vertex_id_dtype")
        set_if_not_present("element_id_dtype")
        set_if_not_present("is_conforming")

        return kwargs

    @property
    def ambient_dim(self):
        from pytools import single_valued
        return single_valued(grp.nodes.shape[0] for grp in self.groups)

    @property
    def dim(self):
        from pytools import single_valued
        return single_valued(grp.dim for grp in self.groups)

    @property
    def nvertices(self):
        if self.vertices is None:
            from meshmode import DataUnavailable
            raise DataUnavailable("vertices")

        return self.vertices.shape[-1]

    @property
    def nelements(self):
        return sum(grp.nelements for grp in self.groups)

    @property
    def nodal_adjacency(self):
        from meshmode import DataUnavailable
        if self._nodal_adjacency is False:
            raise DataUnavailable("nodal_adjacency")

        elif self._nodal_adjacency is None:
            if not self.is_conforming:
                raise DataUnavailable("nodal_adjacency can only "
                        "be computed for known-conforming meshes")

            self._nodal_adjacency = _compute_nodal_adjacency_from_vertices(self)

        return self._nodal_adjacency

    def nodal_adjacency_init_arg(self):
        """Returns a *nodal_adjacency* argument that can be
        passed to a :class:`Mesh` constructor.
        """

        return self._nodal_adjacency

    @property
    def facial_adjacency_groups(self):
        from meshmode import DataUnavailable
        if self._facial_adjacency_groups is False:
            raise DataUnavailable("facial_adjacency_groups")

        elif self._facial_adjacency_groups is None:
            if not self.is_conforming:
                raise DataUnavailable("facial_adjacency_groups can only "
                        "be computed for known-conforming meshes")

            self._facial_adjacency_groups = _compute_facial_adjacency_from_vertices(
                                                self.groups,
                                                self.boundary_tags,
                                                self.element_id_dtype,
                                                self.face_id_dtype)

        return self._facial_adjacency_groups

    def boundary_tag_bit(self, boundary_tag):
        return _boundary_tag_bit(self.boundary_tags, self.btag_to_index,
                        boundary_tag)

    def __eq__(self, other):
        return (
                type(self) == type(other)
                and np.array_equal(self.vertices, other.vertices)
                and self.groups == other.groups
                and self.vertex_id_dtype == other.vertex_id_dtype
                and self.element_id_dtype == other.element_id_dtype
                and self._nodal_adjacency == other._nodal_adjacency
                and self._facial_adjacency_groups == other._facial_adjacency_groups
                and self.boundary_tags == other.boundary_tags
                and self.is_conforming == other.is_conforming)

    def __ne__(self, other):
        return not self.__eq__(other)

    # Design experience: Try not to add too many global data structures to the
    # mesh. Let the element groups be responsible for that at the mesh level.
    #
    # There are more big, global structures on the discretization level.

# }}}


# {{{ node-vertex consistency test

def _test_node_vertex_consistency_resampling(mesh, mgrp, tol):
    if mesh.vertices is None:
        return True

    if mgrp.nelements == 0:
        return True

    if isinstance(mgrp, _ModepyElementGroup):
        basis = mp.basis_for_space(mgrp._modepy_space, mgrp._modepy_shape).functions
    else:
        raise TypeError(f"unsupported group type: {type(mgrp).__name__}")

    resampling_mat = mp.resampling_matrix(
            basis,
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

    max_el_vertex_error = np.max(per_element_vertex_errors)
    assert max_el_vertex_error < tol*size, max_el_vertex_error

    return True


def _test_node_vertex_consistency(mesh, tol):
    """Ensure that order of by-index vertices matches that of mapped
    unit vertices.
    """

    for mgrp in mesh.groups:
        if isinstance(mgrp, _ModepyElementGroup):
            assert _test_node_vertex_consistency_resampling(mesh, mgrp, tol)
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


# {{{ boundary tag to bit

def _boundary_tag_bit(boundary_tags, btag_to_index, boundary_tag):
    if boundary_tag is BTAG_NONE:
        return 0

    if boundary_tag not in boundary_tags:
        raise ValueError("boundary tag '%s' is not known" % boundary_tag)

    try:
        return 1 << btag_to_index[boundary_tag]
    except KeyError:
        return 0

# }}}


# {{{ vertex-based facial adjacency

def _compute_facial_adjacency_from_vertices(groups, boundary_tags,
                                     element_id_dtype,
                                     face_id_dtype,
                                     face_vertex_indices_to_tags=None):
    if not groups:
        return None
    boundary_tag_to_index = {tag: i for i, tag in enumerate(boundary_tags)}

    def boundary_tag_bit(boundary_tag):
        return _boundary_tag_bit(boundary_tags, boundary_tag_to_index, boundary_tag)

    # FIXME Native code would make this faster

    # create face_map, which is a mapping of
    # (vertices on a face) ->
    #  [(igrp, iel_grp, face_idx) for elements bordering that face]
    face_map = {}
    for igrp, grp in enumerate(groups):
        for fid, face_vertex_indices in enumerate(grp.face_vertex_indices()):
            all_fvi = grp.vertex_indices[:, face_vertex_indices]

            for iel_grp, fvi in enumerate(all_fvi):
                face_map.setdefault(
                        frozenset(fvi), []).append((igrp, iel_grp, fid))

    del igrp
    del grp

    # maps tuples (igrp, ineighbor_group) to number of elements
    group_count = {}
    for face_tuples in face_map.values():
        if len(face_tuples) == 2:
            (igrp, _, _), (inb_grp, _, _) = face_tuples
            group_count[igrp, inb_grp] = group_count.get((igrp, inb_grp), 0) + 1
            group_count[inb_grp, igrp] = group_count.get((inb_grp, igrp), 0) + 1
        elif len(face_tuples) == 1:
            (igrp, _, _), = face_tuples
            group_count[igrp, None] = group_count.get((igrp, None), 0) + 1
        else:
            raise RuntimeError("unexpected number of adjacent faces")

    del face_tuples
    del igrp

    # {{{ build facial_adjacency_groups data structure, still empty
    from meshmode.mesh import FacialAdjacencyGroup, BTAG_ALL, BTAG_REALLY_ALL

    facial_adjacency_groups = []
    for igroup in range(len(groups)):
        grp_map = {}
        facial_adjacency_groups.append(grp_map)

        bdry_count = group_count.get((igroup, None))
        if bdry_count is not None:
            elements = np.empty(bdry_count, dtype=element_id_dtype)
            element_faces = np.empty(bdry_count, dtype=face_id_dtype)
            neighbors = np.empty(bdry_count, dtype=element_id_dtype)
            neighbor_faces = np.zeros(bdry_count, dtype=face_id_dtype)

            # Ensure uninitialized entries get noticed
            elements.fill(-1)
            element_faces.fill(-1)
            neighbor_faces.fill(-1)

            neighbors.fill(-(
                    boundary_tag_bit(BTAG_ALL)
                    | boundary_tag_bit(BTAG_REALLY_ALL)))

            grp_map[None] = FacialAdjacencyGroup(
                    igroup=igroup,
                    ineighbor_group=None,
                    elements=elements,
                    element_faces=element_faces,
                    neighbors=neighbors,
                    neighbor_faces=neighbor_faces)

        for ineighbor_group in range(len(groups)):
            nb_count = group_count.get((igroup, ineighbor_group))
            if nb_count is not None:
                elements = np.empty(nb_count, dtype=element_id_dtype)
                element_faces = np.empty(nb_count, dtype=face_id_dtype)
                neighbors = np.empty(nb_count, dtype=element_id_dtype)
                neighbor_faces = np.empty(nb_count, dtype=face_id_dtype)

                # Ensure uninitialized entries get noticed
                elements.fill(-1)
                element_faces.fill(-1)
                neighbors.fill(-1)
                neighbor_faces.fill(-1)

                grp_map[ineighbor_group] = FacialAdjacencyGroup(
                        igroup=igroup,
                        ineighbor_group=ineighbor_group,
                        elements=elements,
                        element_faces=element_faces,
                        neighbors=neighbors,
                        neighbor_faces=neighbor_faces)

    del igroup
    del ineighbor_group
    del grp_map

    # }}}

    # maps tuples (igrp, ineighbor_group) to number of elements filled in group
    fill_count = {}
    for face_tuples in face_map.values():
        if len(face_tuples) == 2:
            for (igroup, iel, iface), (ineighbor_group, inb_el, inb_face) in [
                    (face_tuples[0], face_tuples[1]),
                    (face_tuples[1], face_tuples[0]),
                    ]:
                idx = fill_count.get((igroup, ineighbor_group), 0)
                fill_count[igroup, ineighbor_group] = idx + 1

                fagrp = facial_adjacency_groups[igroup][ineighbor_group]
                fagrp.elements[idx] = iel
                fagrp.element_faces[idx] = iface
                fagrp.neighbors[idx] = inb_el
                fagrp.neighbor_faces[idx] = inb_face

        elif len(face_tuples) == 1:
            (igroup, iel, iface), = face_tuples

            idx = fill_count.get((igroup, None), 0)
            fill_count[igroup, None] = idx + 1

            fagrp = facial_adjacency_groups[igroup][None]
            fagrp.elements[idx] = iel
            fagrp.element_faces[idx] = iface
            # mark tags if present
            if face_vertex_indices_to_tags:
                face_vertex_indices = groups[igroup].face_vertex_indices()[iface]
                fvi = frozenset(groups[igroup].vertex_indices[
                        iel, face_vertex_indices])
                tags = face_vertex_indices_to_tags.get(fvi, None)
                if tags is not None:
                    tag_mask = 0
                    for tag in tags:
                        tag_mask |= boundary_tag_bit(tag)
                    fagrp.neighbors[idx] = -(-(fagrp.neighbors[idx]) | tag_mask)

        else:
            raise RuntimeError("unexpected number of adjacent faces")

    return facial_adjacency_groups

# }}}


# {{{ as_python

def _numpy_array_as_python(array):
    return "np.array({}, dtype=np.{})".format(
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
            cg("groups.append({}.{}(".format(
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
            return ",\n    ".join(f"{k}={v}" for k, v in params.items())

        if mesh._facial_adjacency_groups:
            cg("facial_adjacency_groups = []")

            for igrp, fagrp_map in enumerate(mesh.facial_adjacency_groups):
                cg("facial_adjacency_groups.append({%s})" % ",\n    ".join(
                    "{!r}: FacialAdjacencyGroup({})".format(
                        inb_grp, fagrp_params_str(fagrp))
                    for inb_grp, fagrp in fagrp_map.items()))

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
            el_con_str = "({}, {})".format(
                    _numpy_array_as_python(
                        mesh._nodal_adjacency.neighbors_starts),
                    _numpy_array_as_python(
                        mesh._nodal_adjacency.neighbors),
                    )
        else:
            el_con_str = repr(mesh._nodal_adjacency)

        cg("    nodal_adjacency=%s," % el_con_str)
        cg("    facial_adjacency_groups=facial_adjacency_groups,")
        cg("    boundary_tags=[%s]," % btags_str)
        cg("    is_conforming=%s)" % repr(mesh.is_conforming))

        # FIXME: Handle facial adjacency, boundary tags

    return cg.get()

# }}}


# {{{ check_bc_coverage

def check_bc_coverage(mesh, boundary_tags, incomplete_ok=False,
        true_boundary_only=True):
    """Verify boundary condition coverage.

    Given a list of boundary tags as *boundary_tags*, this function verifies
    that

     1. the union of all these boundaries gives the complete boundary,
     1. all these boundaries are disjoint.

    :arg incomplete_ok: Do not report an error if some faces are not covered
      by the boundary conditions.
    :arg true_boundary_only: only verify for faces tagged with :class:`BTAG_ALL`.
    """

    for igrp, fagrp_map in enumerate(mesh.facial_adjacency_groups):
        bdry_grp = fagrp_map.get(None)
        if bdry_grp is None:
            continue

        nb_elements = bdry_grp.neighbors
        assert (nb_elements < 0).all()

        nb_el_bits = -nb_elements

        # An array of flags for each face indicating whether we have encountered
        # a boundary condition for that face.
        seen = np.zeros_like(nb_el_bits, dtype=np.bool)

        if true_boundary_only:
            tag_bit = mesh.boundary_tag_bit(BTAG_ALL)
            tag_set = (nb_el_bits & tag_bit) != 0

            # Consider non-boundary faces 'seen'
            seen = seen | ~tag_set

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


# {{{ is_affine_simplex_group

def is_affine_simplex_group(group, abs_tol=None):
    if abs_tol is None:
        abs_tol = 1.0e-13

    if not isinstance(group, SimplexElementGroup):
        raise TypeError("expected a 'SimplexElementGroup' not '%s'" %
                type(group).__name__)

    # get matrices
    basis = mp.basis_for_space(group._modepy_space, group._modepy_shape)
    vinv = la.inv(mp.vandermonde(basis.functions, group.unit_nodes))
    diff = mp.differentiation_matrices(
            basis.functions, basis.gradients, group.unit_nodes)
    if not isinstance(diff, tuple):
        diff = (diff,)

    # construct all second derivative matrices (including cross terms)
    from itertools import product
    mats = []
    for n in product(range(group.dim), repeat=2):
        if n[0] > n[1]:
            continue
        mats.append(vinv.dot(diff[n[0]].dot(diff[n[1]])))

    # check just the first element for a non-affine local-to-global mapping
    ddx_coeffs = np.einsum("aij,bj->abi", mats, group.nodes[:, 0, :])
    norm_inf = np.max(np.abs(ddx_coeffs))
    if norm_inf > abs_tol:
        return False

    # check all elements for a non-affine local-to-global mapping
    ddx_coeffs = np.einsum("aij,bcj->abci", mats, group.nodes)
    norm_inf = np.max(np.abs(ddx_coeffs))
    return norm_inf < abs_tol

# }}}

# vim: foldmethod=marker
