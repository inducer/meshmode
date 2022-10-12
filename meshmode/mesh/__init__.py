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

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace, field
from typing import Any, ClassVar, Hashable, Optional, Tuple, Type, Sequence

import numpy as np
import numpy.linalg as la

import modepy as mp
from pytools import Record, memoize_method

from meshmode.mesh.tools import AffineMap

__doc__ = """

.. autoclass:: MeshElementGroup
.. autoclass:: SimplexElementGroup
.. autoclass:: TensorProductElementGroup

.. autoclass:: Mesh

.. autoclass:: NodalAdjacency
.. autoclass:: FacialAdjacencyGroup
.. autoclass:: InteriorAdjacencyGroup
.. autoclass:: BoundaryAdjacencyGroup
.. autoclass:: InterPartAdjacencyGroup

.. autofunction:: as_python
.. autofunction:: is_true_boundary
.. autofunction:: mesh_has_boundary
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

BoundaryTag = Hashable
PartID = Hashable


class BTAG_NONE:  # noqa: N801
    """A boundary tag representing an empty boundary or volume."""


class BTAG_ALL:  # noqa: N801
    """A boundary tag representing the entire boundary or volume.

    In the case of the boundary, :class:`BTAG_ALL` does not include rank boundaries,
    or, more generally, anything tagged with :class:`BTAG_NO_BOUNDARY`.

    In the case of a mesh representing an element-wise subset of another,
    :class:`BTAG_ALL` does not include boundaries induced by taking the subset.
    Instead, these boundaries will be tagged with
    :class:`BTAG_INDUCED_BOUNDARY`.
    """


class BTAG_REALLY_ALL:  # noqa: N801
    """A boundary tag representing the entire boundary.

    Unlike :class:`BTAG_ALL`, this includes rank boundaries,
    or, more generally, everything tagged with :class:`BTAG_NO_BOUNDARY`.

    In the case of a mesh representing an element-wise subset of another,
    this tag includes boundaries induced by taking the subset, or, more generally,
    everything tagged with
    :class:`BTAG_INDUCED_BOUNDARY`
    """


class BTAG_NO_BOUNDARY:  # noqa: N801
    """A boundary tag indicating that this edge should not fall under
    :class:`BTAG_ALL`. Among other things, this is used to keep rank boundaries
    out of :class:`BTAG_ALL`.
    """


class BTAG_PARTITION(BTAG_NO_BOUNDARY):  # noqa: N801
    """
    A boundary tag indicating that this edge is adjacent to an element of
    another :class:`Mesh`. The part identifier of the adjacent mesh is given
    by ``part_id``.

    .. attribute:: part_id

    .. versionadded:: 2017.1
    """
    def __init__(self, part_id: PartID, part_nr=None):
        if part_nr is not None:
            from warnings import warn
            warn("part_nr is deprecated and will stop working in March 2023. "
                 "Use part_id instead.",
                 DeprecationWarning, stacklevel=2)
            self.part_id = int(part_nr)
        else:
            self.part_id = part_id

    @property
    def part_nr(self):
        from warnings import warn
        warn("part_nr is deprecated and will stop working in March 2023. "
             "Use part_id instead.",
             DeprecationWarning, stacklevel=2)
        return self.part_id

    def __hash__(self):
        return hash((type(self), self.part_id))

    def __eq__(self, other):
        if isinstance(other, BTAG_PARTITION):
            return self.part_id == other.part_id
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "<{}({})>".format(type(self).__name__, repr(self.part_id))

    def as_python(self):
        return f"{self.__class__.__name__}({self.part_id})"


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


SYSTEM_TAGS = {BTAG_NONE, BTAG_ALL, BTAG_REALLY_ALL, BTAG_NO_BOUNDARY,
                   BTAG_PARTITION, BTAG_INDUCED_BOUNDARY}

# }}}


# {{{ element group

# {{{ base class

@dataclass(frozen=True, eq=False)
class MeshElementGroup(ABC):
    """A group of elements sharing a common reference element.

    .. attribute:: order

        The maximum degree used for interpolation. The exact meaning depends
        on the element type, e.g. for :class:`SimplexElementGroup` this is
        the total degree.

    .. attribute:: dim

        The number of dimensions spanned by the element.
        *Not* the ambient dimension, see :attr:`Mesh.ambient_dim`
        for that.

    .. attribute:: nvertices

        Number of vertices in the reference element.

    .. attribute:: nfaces

        Number of faces of the reference element.

    .. attribute:: nunit_nodes

        Number of nodes on the reference element.

    .. attribute:: nelements

        Number of elements in the group.

    .. attribute:: nnodes

        Total number of nodes in the group (equivalent to
        ``nelements * nunit_nodes``).

    .. attribute:: vertex_indices

        An array of shape ``(nelements, nvertices)`` of (mesh-wide)
        vertex indices.

    .. attribute:: nodes

        An array of node coordinates with shape
        ``(mesh.ambient_dim, nelements, nunit_nodes)``.

    .. attribute:: unit_nodes

        An array with shape ``(dim, nunit_nodes)`` of nodes on the reference
        element. The coordinates :attr:`nodes` are a mapped version
        of these reference nodes.

    .. attribute:: is_affine

        A :class:`bool` flag that is *True* if the local-to-global
        parametrization of all the elements in the group is affine.

    Element groups can also be compared for equality using the following
    methods. Note that these are *very* expensive, as they compare all
    the :attr:`nodes`.

    .. automethod:: __eq__
    .. automethod:: __ne__

    .. automethod:: __init__

    The following abstract methods must be implemented by subclasses.

    .. automethod:: make_group
    .. automethod:: face_vertex_indices
    .. automethod:: vertex_unit_coordinates
    """

    order: int

    # NOTE: the mesh supports not having vertices if no facial or nodal
    # adjacency is required, so we can mark this as optional
    vertex_indices: Optional[np.ndarray]
    nodes: np.ndarray

    # TODO: Remove ` = None` when everything is constructed through the factory
    unit_nodes: np.ndarray = None

    # FIXME: these should be removed!
    # https://github.com/inducer/meshmode/issues/224
    element_nr_base: Optional[int] = None
    node_nr_base: Optional[int] = None

    # TODO: remove when everything has been constructed through the factory
    _factory_constructed: bool = False

    def __post_init__(self):
        if not self._factory_constructed:
            from warnings import warn
            warn(f"Calling the constructor of '{type(self).__name__}' is "
                 "deprecated and will stop working in July 2022. "
                 f"Use '{type(self).__name__}.make_group' instead",
                 DeprecationWarning, stacklevel=2)

    def __getattribute__(self, name):
        if name in ("element_nr_base", "node_nr_base"):
            new_name = ("base_element_nrs"
                        if name == "element_nr_base" else
                        "base_node_nrs")

            from warnings import warn
            warn(f"'{type(self).__name__}.{name}' is deprecated and will be "
                 f"removed in July 2022. Use 'Mesh.{new_name}' instead",
                 DeprecationWarning, stacklevel=2)

        return super().__getattribute__(name)

    @property
    def dim(self):
        return self.unit_nodes.shape[0]

    @property
    def nvertices(self):
        return self.vertex_unit_coordinates().shape[-1]

    @property
    def nfaces(self):
        return len(self.face_vertex_indices())

    @property
    def nunit_nodes(self):
        return self.unit_nodes.shape[-1]

    @property
    def nelements(self):
        return self.nodes.shape[1]

    @property
    def nnodes(self):
        return self.nelements * self.unit_nodes.shape[-1]

    def copy(self, **kwargs: Any) -> "MeshElementGroup":
        from warnings import warn
        warn(f"{type(self).__name__}.copy is deprecated and will be removed in "
                f"July 2022. {type(self).__name__} is now a dataclass, so "
                "standard functions such as dataclasses.replace should be used "
                "instead.",
                DeprecationWarning, stacklevel=2)

        if "element_nr_base" not in kwargs:
            kwargs["element_nr_base"] = None
        if "node_nr_base" not in kwargs:
            kwargs["node_nr_base"] = None

        return replace(self, **kwargs)

    def __eq__(self, other):
        return (
                type(self) == type(other)
                and self.order == other.order
                and np.array_equal(self.vertex_indices, other.vertex_indices)
                and np.array_equal(self.nodes, other.nodes)
                and np.array_equal(self.unit_nodes, other.unit_nodes))

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def is_affine(self):
        raise NotImplementedError

    @abstractmethod
    def face_vertex_indices(self) -> Tuple[Tuple[int, ...], ...]:
        """
        :returns: a :class:`tuple` of tuples indicating which vertices
            (in mathematically positive ordering) make up each face
            of an element in this group.
        """

    @abstractmethod
    def vertex_unit_coordinates(self) -> np.ndarray:
        """
        :returns: an array of shape ``(nfaces, dim)`` with the unit
            coordinates of each vertex.
        """

    @classmethod
    @abstractmethod
    def make_group(cls, **kwargs: Any) -> "MeshElementGroup":
        """Instantiate a new group of class *cls*.

        Unlike the constructor, this factory function performs additional
        consistency checks and should be used instead.
        """

# }}}


# {{{ modepy-based element group

@dataclass(frozen=True, eq=False)
class _ModepyElementGroup(MeshElementGroup):
    """
    .. attribute:: _modepy_shape_cls

        Must be set by subclasses to generate the correct shape and spaces
        attributes for the group.

    .. attribute:: _modepy_shape
    .. attribute:: _modepy_space
    """

    # TODO: remove once `make_group` is used everywhere
    dim: Optional[int] = None

    _modepy_shape_cls: ClassVar[Type[mp.Shape]] = mp.Shape
    _modepy_shape: mp.Shape = field(default=None, repr=False)
    _modepy_space: mp.FunctionSpace = field(default=None, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if self._factory_constructed:
            return

        # {{{ duplicates make_group below, keep in sync

        if self.unit_nodes is None:
            if self.dim is None:
                raise TypeError("either 'dim' or 'unit_nodes' must be provided")
        else:
            if self.dim is None:
                object.__setattr__(self, "dim", self.unit_nodes.shape[0])

            if self.unit_nodes.shape[0] != self.dim:
                raise ValueError("'dim' does not match 'unit_nodes' dimension")

        # dim is now usable
        assert self._modepy_shape_cls is not mp.Shape
        object.__setattr__(self, "_modepy_shape",
                # pylint: disable=abstract-class-instantiated
                self._modepy_shape_cls(self.dim))
        object.__setattr__(self, "_modepy_space",
                mp.space_for_shape(self._modepy_shape, self.order))

        if self.unit_nodes is None:
            unit_nodes = mp.edge_clustered_nodes_for_space(
                    self._modepy_space, self._modepy_shape)
            object.__setattr__(self, "unit_nodes", unit_nodes)

        if self.nodes is not None:
            if self.unit_nodes.shape[-1] != self.nodes.shape[-1]:
                raise ValueError(
                        "'nodes' has wrong number of unit nodes per element."
                        f" expected {self.unit_nodes.shape[-1]}, "
                        f" but got {self.nodes.shape[-1]}.")

        if self.vertex_indices is not None:
            if not issubclass(self.vertex_indices.dtype.type, np.integer):
                raise TypeError("'vertex_indices' must be integral")

            if self.vertex_indices.shape[-1] != self.nvertices:
                raise ValueError(
                        "'vertex_indices' has wrong number of vertices per element."
                        f" expected {self.nvertices},"
                        f" got {self.vertex_indices.shape[-1]}")

        # }}}

    @property
    def nvertices(self):
        return self._modepy_shape.nvertices     # pylint: disable=no-member

    @property
    @memoize_method
    def _modepy_faces(self):
        return mp.faces_for_shape(self._modepy_shape)

    @memoize_method
    def face_vertex_indices(self):
        return tuple([face.volume_vertex_indices for face in self._modepy_faces])

    @memoize_method
    def vertex_unit_coordinates(self):
        return mp.unit_vertices_for_shape(self._modepy_shape).T

    @classmethod
    def make_group(cls, order: int,
                   vertex_indices: Optional[np.ndarray],
                   nodes: np.ndarray,
                   unit_nodes: Optional[np.ndarray] = None,
                   dim: Optional[int] = None) -> "_ModepyElementGroup":
        # {{{ duplicates __post_init__ above, keep in sync

        if unit_nodes is None:
            if dim is None:
                raise TypeError("either 'dim' or 'unit_nodes' must be provided")
        else:
            if dim is None:
                dim = unit_nodes.shape[0]

            if unit_nodes.shape[0] != dim:
                raise ValueError("'dim' does not match 'unit_nodes' dimension")

        # pylint: disable=abstract-class-instantiated
        shape = cls._modepy_shape_cls(dim)
        space = mp.space_for_shape(shape, order)

        if unit_nodes is None:
            unit_nodes = mp.edge_clustered_nodes_for_space(space, shape)

        if unit_nodes.shape[1] != space.space_dim:
            raise ValueError("'unit_nodes' size does not match the dimension "
                             f"of a '{type(space).__name__}' space of order {order}")

        return cls(order=order, vertex_indices=vertex_indices, nodes=nodes,
                   dim=dim,
                   unit_nodes=unit_nodes,
                   _modepy_shape=shape, _modepy_space=space,
                   _factory_constructed=True)

        # }}}

# }}}


@dataclass(frozen=True, eq=False)
class SimplexElementGroup(_ModepyElementGroup):
    r"""Inherits from :class:`MeshElementGroup`."""
    _modepy_shape_cls: ClassVar[Type[mp.Shape]] = mp.Simplex

    @property
    @memoize_method
    def is_affine(self):
        return is_affine_simplex_group(self)


@dataclass(frozen=True, eq=False)
class TensorProductElementGroup(_ModepyElementGroup):
    r"""Inherits from :class:`MeshElementGroup`."""
    _modepy_shape_cls: ClassVar[Type[mp.Shape]] = mp.Hypercube

# }}}


# {{{ nodal adjacency

@dataclass(frozen=True, eq=False)
class NodalAdjacency:
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

    neighbors_starts: np.ndarray
    neighbors: np.ndarray

    def copy(self, **kwargs: Any) -> "NodalAdjacency":
        from warnings import warn
        warn(f"{type(self).__name__}.copy is deprecated and will be removed in "
                f"July 2022. {type(self).__name__} is now a dataclass, so "
                "standard functions such as dataclasses.replace should be used "
                "instead.",
                DeprecationWarning, stacklevel=2)

        return replace(self, **kwargs)

    def __eq__(self, other):
        return (
                type(self) == type(other)
                and np.array_equal(self.neighbors_starts, other.neighbors_starts)
                and np.array_equal(self.neighbors, other.neighbors))

    def __ne__(self, other):
        return not self.__eq__(other)

# }}}


# {{{ facial adjacency

@dataclass(frozen=True, eq=False)
class FacialAdjacencyGroup:
    r"""
    Describes facial element adjacency information for one
    :class:`MeshElementGroup`, i.e. information about elements that share (part
    of) a face or elements that lie on a boundary.

    .. tikz:: Facial Adjacency Group
        :align: center
        :xscale: 40

        \draw [thick] (0, 2) rectangle node {$0$} (3, 4);
        \draw [thick] (3, 2) rectangle node {$1$} (6, 4);
        \draw [thick] (0, 0) rectangle node {$2$} (4, 2);
        \draw [thick] (4, 0) rectangle node {$3$} (6, 2);
        \draw [line width=3pt, line cap=round, orange]
            (4, 0) -- (6, 0) -- (6, 2);
        \draw [line width=3pt, line cap=round, magenta]
            (4, 0) -- (4, 2);
        \draw [line width=3pt, line cap=round, green!60!black]
            (4, 2) -- (6, 2);

    Represents (for example) *one* of the (colored) interfaces between
    :class:`MeshElementGroup` instances, or an interface between
    :class:`MeshElementGroup` and a boundary. (Note that element groups are not
    necessarily contiguous like the figure may suggest.)

    .. attribute:: igroup
    """

    igroup: int

    def copy(self, **kwargs: Any) -> "FacialAdjacencyGroup":
        from warnings import warn
        warn(f"{type(self).__name__}.copy is deprecated and will be removed in "
                f"July 2022. {type(self).__name__} is now a dataclass, so "
                "standard functions such as dataclasses.replace should be used "
                "instead.",
                DeprecationWarning, stacklevel=2)

        return replace(self, **kwargs)

    def __eq__(self, other):
        return (
                type(self) == type(other)
                and self.igroup == other.igroup)

    def __ne__(self, other):
        return not self.__eq__(other)

    def _as_python(self, **kwargs):
        return "{cls}({args})".format(
                cls=self.__class__.__name__,
                args=",\n    ".join(f"{k}={v}" for k, v in kwargs.items())
                )

    def as_python(self) -> str:
        """
        :returns: a string that can be evaluated to reconstruct the class.
        """

        if type(self) != FacialAdjacencyGroup:
            raise NotImplementedError(
                    f"Not implemented for '{type(self).__name__}'.")

        return self._as_python(igroup=self.igroup)

# }}}


# {{{ interior adjacency

@dataclass(frozen=True, eq=False)
class InteriorAdjacencyGroup(FacialAdjacencyGroup):
    """Describes interior facial element adjacency information for one
    :class:`MeshElementGroup`.

    .. attribute:: igroup

        The mesh element group number of this group.

    .. attribute:: ineighbor_group

        ID of neighboring group, or *None* for boundary faces. If identical
        to :attr:`igroup`, then this contains the self-connectivity in this
        group.

    .. attribute:: elements

        ``element_id_t [nfagrp_elements]``. ``elements[i]`` gives the
        element number within :attr:`igroup` of the interior face.

    .. attribute:: element_faces

        ``face_id_t [nfagrp_elements]``. ``element_faces[i]`` gives the face
        index of the interior face in element ``elements[i]``.

    .. attribute:: neighbors

        ``element_id_t [nfagrp_elements]``. ``neighbors[i]`` gives the element
        number within :attr:`ineighbor_group` of the element opposite
        ``elements[i]``.

    .. attribute:: neighbor_faces

        ``face_id_t [nfagrp_elements]``. ``neighbor_faces[i]`` gives the
        face index of the opposite face in element ``neighbors[i]``

    .. attribute:: aff_map

        An :class:`~meshmode.AffineMap` representing the mapping from the group's
        faces to their corresponding neighbor faces.

    .. automethod:: __eq__
    .. automethod:: __ne__
    """

    ineighbor_group: int
    elements: np.ndarray
    element_faces: np.ndarray
    neighbors: np.ndarray
    neighbor_faces: np.ndarray
    aff_map: AffineMap

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and self.ineighbor_group == other.ineighbor_group
            and np.array_equal(self.elements, other.elements)
            and np.array_equal(self.element_faces, other.element_faces)
            and np.array_equal(self.neighbors, other.neighbors)
            and np.array_equal(self.neighbor_faces, other.neighbor_faces)
            and self.aff_map == other.aff_map)

    def as_python(self):
        if type(self) != InteriorAdjacencyGroup:
            raise NotImplementedError(f"Not implemented for {type(self)}.")

        return self._as_python(
            igroup=self.igroup,
            ineighbor_group=self.ineighbor_group,
            elements=_numpy_array_as_python(self.elements),
            element_faces=_numpy_array_as_python(self.element_faces),
            neighbors=_numpy_array_as_python(self.neighbors),
            neighbor_faces=_numpy_array_as_python(self.neighbor_faces),
            aff_map=_affine_map_as_python(self.aff_map))

# }}}


# {{{ boundary adjacency

@dataclass(frozen=True, eq=False)
class BoundaryAdjacencyGroup(FacialAdjacencyGroup):
    """Describes boundary adjacency information for one :class:`MeshElementGroup`.

    .. attribute:: igroup

        The mesh element group number of this group.

    .. attribute:: boundary_tag

        The boundary tag identifier of this group.

    .. attribute:: elements

        ``element_id_t [nfagrp_elements]``. ``elements[i]`` gives the
        element number within :attr:`igroup` of the boundary face.

    .. attribute:: element_faces

        ``face_id_t [nfagrp_elements]``. ``element_faces[i]`` gives the face
        index of the boundary face in element ``elements[i]``.
    """

    boundary_tag: Hashable
    elements: np.ndarray
    element_faces: np.ndarray

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and self.boundary_tag == other.boundary_tag
            and np.array_equal(self.elements, other.elements)
            and np.array_equal(self.element_faces, other.element_faces))

    def as_python(self):
        if type(self) != BoundaryAdjacencyGroup:
            raise NotImplementedError(f"Not implemented for {type(self)}.")

        return self._as_python(
            igroup=self.igroup,
            boundary_tag=_boundary_tag_as_python(self.boundary_tag),
            elements=_numpy_array_as_python(self.elements),
            element_faces=_numpy_array_as_python(self.element_faces))

# }}}


# {{{ partition adjacency

@dataclass(frozen=True, eq=False)
class InterPartAdjacencyGroup(BoundaryAdjacencyGroup):
    """
    Describes inter-part adjacency information for one
    :class:`MeshElementGroup`.

    .. attribute:: igroup

        The mesh element group number of this group.

    .. attribute:: boundary_tag

        The boundary tag identifier of this group. Will be an instance of
        :class:`~meshmode.mesh.BTAG_PARTITION`.

    .. attribute:: part_id

        The identifier of the neighboring part.

    .. attribute:: elements

        Group-local element numbers.
        Element ``element_id_dtype elements[i]`` and face
        ``face_id_dtype element_faces[i]`` is connected to neighbor element
        ``element_id_dtype neighbors[i]`` with face
        ``face_id_dtype neighbor_faces[i]``.

    .. attribute:: element_faces

        ``face_id_dtype element_faces[i]`` gives the face of
        ``element_id_dtype elements[i]`` that is connected to ``neighbors[i]``.

    .. attribute:: neighbors

        ``element_id_dtype neighbors[i]`` gives the volume element number
        within the neighboring part of the element connected to
        ``element_id_dtype elements[i]`` (which is a boundary element index). Use
        `~meshmode.mesh.processing.find_group_indices` to find the group that
        the element belongs to, then subtract ``element_nr_base`` to find the
        element of the group.

    .. attribute:: neighbor_faces

        ``face_id_dtype global_neighbor_faces[i]`` gives face index within the
        neighboring part of the face connected to ``element_id_dtype elements[i]``

    .. attribute:: aff_map

        An :class:`~meshmode.AffineMap` representing the mapping from the group's
        faces to their corresponding neighbor faces.

    .. versionadded:: 2017.1
    """

    part_id: PartID
    neighbors: np.ndarray
    neighbor_faces: np.ndarray
    aff_map: AffineMap

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and np.array_equal(self.part_id, other.part_id)
            and np.array_equal(self.neighbors, other.neighbors)
            and np.array_equal(self.neighbor_faces, other.neighbor_faces)
            and self.aff_map == other.aff_map)

    def as_python(self):
        if type(self) != InterPartAdjacencyGroup:
            raise NotImplementedError(f"Not implemented for {type(self)}.")

        return self._as_python(
            igroup=self.igroup,
            boundary_tag=_boundary_tag_as_python(self.boundary_tag),
            part_id=self.part_id,
            elements=_numpy_array_as_python(self.elements),
            element_faces=_numpy_array_as_python(self.element_faces),
            neighbors=_numpy_array_as_python(self.neighbors),
            neighbor_faces=_numpy_array_as_python(self.neighbor_faces),
            aff_map=_affine_map_as_python(self.aff_map))

# }}}


# {{{ mesh

class Mesh(Record):
    r"""
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

    .. attribute:: base_element_nrs

        An array of size ``(len(groups),)`` of starting element indices for
        each group in the mesh.

    .. attribute:: base_node_nrs

        An array of size ``(len(groups),)`` of starting node indices for
        each group in the mesh.

    .. attribute:: nodal_adjacency

        An instance of :class:`NodalAdjacency`.

        Referencing this attribute may raise
        :exc:`meshmode.DataUnavailable`.

    .. attribute:: facial_adjacency_groups

        A list of lists of instances of :class:`FacialAdjacencyGroup`.

        ``facial_adjacency_groups[igrp]`` gives the facial adjacency relations for
        group *igrp*, expressed as a list of :class:`FacialAdjacencyGroup` instances.

        Referencing this attribute may raise
        :exc:`meshmode.DataUnavailable`.

        .. tikz:: Facial Adjacency Group
            :align: center
            :xscale: 40

            \draw [thick] (0, 2) rectangle node {$0$} (3, 4);
            \draw [thick] (3, 2) rectangle node {$1$} (6, 4);
            \draw [thick] (0, 0) rectangle node {$2$} (4, 2);
            \draw [thick] (4, 0) rectangle node {$3$} (6, 2);
            \draw [line width=3pt, line cap=round, orange]
                (4, 0) -- (6, 0) -- (6, 2);
            \draw [line width=3pt, line cap=round, magenta]
                (4, 0) -- (4, 2);
            \draw [line width=3pt, line cap=round, green!60!black]
                (4, 2) -- (6, 2);


        For example for the mesh in the figure, the following data structure
        could be present::

            [
                [...],  # connectivity for group 0
                [...],  # connectivity for group 1
                [...],  # connectivity for group 2
                # connectivity for group 3
                [
                    # towards group 1, green
                    InteriorAdjacencyGroup(ineighbor_group=1, ...),
                    # towards group 2, pink
                    InteriorAdjacencyGroup(ineighbor_group=2, ...),
                    # towards the boundary, orange
                    BoundaryAdjacencyGroup(...)
                ]
            ]

        (Note that element groups are not necessarily geometrically contiguous
        like the figure may suggest.)

    .. attribute:: vertex_id_dtype

    .. attribute:: element_id_dtype

    .. attribute:: is_conforming

        *True* if it is known that all element interfaces are conforming.
        *False* if it is known that some element interfaces are non-conforming.
        *None* if neither of the two is known.

    .. automethod:: copy
    .. automethod:: __eq__
    .. automethod:: __ne__
    """

    groups: Sequence[MeshElementGroup]

    face_id_dtype = np.int8

    def __init__(self, vertices, groups, *, skip_tests=False,
            node_vertex_consistency_tolerance=None,
            skip_element_orientation_test=False,
            nodal_adjacency=None,
            facial_adjacency_groups=None,
            vertex_id_dtype=np.int32,
            element_id_dtype=np.int32,
            is_conforming=None):
        """
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
            ng = replace(g, element_nr_base=el_nr, node_nr_base=node_nr)
            new_groups.append(ng)
            el_nr += ng.nelements
            node_nr += ng.nnodes

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

        if (
                facial_adjacency_groups is not False
                and facial_adjacency_groups is not None):
            facial_adjacency_groups = _complete_facial_adjacency_groups(
                facial_adjacency_groups,
                np.dtype(element_id_dtype),
                self.face_id_dtype)

        Record.__init__(
                self, vertices=vertices, groups=new_groups,
                _nodal_adjacency=nodal_adjacency,
                _facial_adjacency_groups=facial_adjacency_groups,
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
                for fagrp_list in facial_adjacency_groups:
                    for fagrp in fagrp_list:
                        nfagrp_elements, = fagrp.elements.shape
                        assert fagrp.element_faces.dtype == self.face_id_dtype
                        assert fagrp.element_faces.shape == (nfagrp_elements,)
                        if isinstance(fagrp, InteriorAdjacencyGroup):
                            assert fagrp.neighbors.dtype == self.element_id_dtype
                            assert fagrp.neighbors.shape == (nfagrp_elements,)
                            assert fagrp.neighbor_faces.dtype == self.face_id_dtype
                            assert fagrp.neighbor_faces.shape == (nfagrp_elements,)

            from meshmode.mesh.processing import \
                    test_volume_mesh_element_orientations

            if self.dim == self.ambient_dim and not skip_element_orientation_test:
                # only for volume meshes, for now
                if not test_volume_mesh_element_orientations(self):
                    raise ValueError("negatively oriented elements found")

    def get_copy_kwargs(self, **kwargs):
        def set_if_not_present(name, from_name=None):
            if from_name is None:
                from_name = name
            if name not in kwargs:
                kwargs[name] = getattr(self, from_name)

        set_if_not_present("vertices")
        if "groups" not in kwargs:
            kwargs["groups"] = [
                replace(group, element_nr_base=None, node_nr_base=None)
                for group in self.groups]
        set_if_not_present("nodal_adjacency", "_nodal_adjacency")
        set_if_not_present("facial_adjacency_groups", "_facial_adjacency_groups")
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
    @memoize_method
    def base_element_nrs(self):
        return np.cumsum([0] + [grp.nelements for grp in self.groups[:-1]])

    @property
    @memoize_method
    def base_node_nrs(self):
        return np.cumsum([0] + [grp.nnodes for grp in self.groups[:-1]])

    @property
    def nodal_adjacency(self):
        from meshmode import DataUnavailable

        # pylint: disable=access-member-before-definition
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
    def facial_adjacency_groups(self) -> Sequence[Sequence[FacialAdjacencyGroup]]:
        from meshmode import DataUnavailable

        # pylint: disable=access-member-before-definition
        if self._facial_adjacency_groups is False:
            raise DataUnavailable("facial_adjacency_groups")

        elif self._facial_adjacency_groups is None:
            if not self.is_conforming:
                raise DataUnavailable("facial_adjacency_groups can only "
                        "be computed for known-conforming meshes")

            self._facial_adjacency_groups = _compute_facial_adjacency_from_vertices(
                                                self.groups,
                                                self.element_id_dtype,
                                                self.face_id_dtype)

        return self._facial_adjacency_groups

    def __eq__(self, other):
        return (
                type(self) == type(other)
                and np.array_equal(self.vertices, other.vertices)
                and self.groups == other.groups
                and self.vertex_id_dtype == other.vertex_id_dtype
                and self.element_id_dtype == other.element_id_dtype
                and self._nodal_adjacency == other._nodal_adjacency
                and self._facial_adjacency_groups == other._facial_adjacency_groups
                and self.is_conforming == other.is_conforming)

    def __ne__(self, other):
        return not self.__eq__(other)

    # Design experience: Try not to add too many global data structures to the
    # mesh. Let the element groups be responsible for that at the mesh level.
    #
    # There are more big, global structures on the discretization level.

# }}}


# {{{ node-vertex consistency test

def _mesh_group_node_vertex_error(mesh, mgrp):
    if isinstance(mgrp, _ModepyElementGroup):
        basis = mp.basis_for_space(mgrp._modepy_space, mgrp._modepy_shape).functions
    else:
        raise TypeError(f"unsupported group type: {type(mgrp).__name__}")

    resampling_mat = mp.resampling_matrix(
            basis,
            mgrp.vertex_unit_coordinates().T,
            mgrp.unit_nodes)

    # dim, nelments, nvertices
    map_vertices = np.einsum(
            "ij,dej->dei", resampling_mat, mgrp.nodes)
    grp_vertices = mesh.vertices[:, mgrp.vertex_indices]

    return map_vertices - grp_vertices


def _test_node_vertex_consistency_resampling(mesh, igrp, tol):
    if mesh.vertices is None:
        return True

    mgrp = mesh.groups[igrp]

    if mgrp.nelements == 0:
        return True

    per_vertex_errors = _mesh_group_node_vertex_error(mesh, mgrp)
    per_element_vertex_errors = np.max(
        np.max(np.abs(per_vertex_errors), axis=-1), axis=0)

    if tol is None:
        tol = 1e3 * np.finfo(per_element_vertex_errors.dtype).eps

    grp_vertices = mesh.vertices[:, mgrp.vertex_indices]

    coord_scales = np.max(np.max(np.abs(grp_vertices), axis=-1), axis=0)

    per_element_tols = tol + tol * coord_scales

    elements_above_tol, = np.where(per_element_vertex_errors >= per_element_tols)
    if len(elements_above_tol) > 0:
        i_grp_elem = elements_above_tol[0]
        ielem = i_grp_elem + mesh.base_element_nrs[igrp]
        from meshmode import InconsistentVerticesError
        raise InconsistentVerticesError(
            f"vertex consistency check failed for element {ielem}; "
            f"{per_element_vertex_errors[i_grp_elem]} >= "
            f"{per_element_tols[i_grp_elem]}")

    return True


def _test_node_vertex_consistency(mesh, tol):
    """Ensure that order of by-index vertices matches that of mapped
    unit vertices.
    """
    if not __debug__:
        return True

    for igrp, mgrp in enumerate(mesh.groups):
        if isinstance(mgrp, _ModepyElementGroup):
            assert _test_node_vertex_consistency_resampling(mesh, igrp, tol)
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

    for base_element_nr, grp in zip(mesh.base_element_nrs, mesh.groups):
        for iel_grp in range(grp.nelements):
            for ivertex in grp.vertex_indices[iel_grp]:
                vertex_to_element[ivertex].append(base_element_nr + iel_grp)

    element_to_element = [set() for i in range(mesh.nelements)]
    for base_element_nr, grp in zip(mesh.base_element_nrs, mesh.groups):
        for iel_grp in range(grp.nelements):
            for ivertex in grp.vertex_indices[iel_grp]:
                element_to_element[base_element_nr + iel_grp].update(
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

@dataclass(frozen=True)
class _FaceIDs:
    """
    Data structure for storage of a list of face identifiers (group, element, face).
    Each attribute is a :class:`numpy.ndarray` of shape ``(nfaces,)``.

    .. attribute:: groups

        The index of the group containing the face.

    .. attribute:: elements

        The group-relative index of the element containing the face.

    .. attribute:: faces

        The element-relative index of face.
    """
    groups: np.ndarray
    elements: np.ndarray
    faces: np.ndarray


def _concatenate_face_ids(face_ids_list):
    return _FaceIDs(
        groups=np.concatenate([ids.groups for ids in face_ids_list]),
        elements=np.concatenate([ids.elements for ids in face_ids_list]),
        faces=np.concatenate([ids.faces for ids in face_ids_list]))


def _match_faces_by_vertices(groups, face_ids, vertex_index_map_func=None):
    """
    Return matching faces in *face_ids* (expressed as pairs of indices into
    *face_ids*), where two faces match if they have the same vertices.

    :arg groups: A list of :class:`~meshmode.mesh.MeshElementGroup` to which the
        faces in *face_ids* belong.
    :arg face_ids: A :class:`~meshmode.mesh._FaceIDs` containing the faces selected
        for matching.
    :arg vertex_index_map_func: An optional function that maps a set of vertex
        indices (stored in a :class:`numpy.ndarray`) to another set of vertex
        indices. Must accept multidimensional arrays as input and return an array
        of the same shape.
    :returns: A :class:`numpy.ndarray` of shape ``(2, nmatches)`` of indices into
        *face_ids*. The ordering of the matches returned is unspecified. For a given
        match, however, the first index will correspond to the face that occurs first
        in *face_ids*.
    """
    if vertex_index_map_func is None:
        def vertex_index_map_func(vertices):
            return vertices

    from pytools import single_valued
    vertex_id_dtype = single_valued(grp.vertex_indices.dtype for grp in groups)

    nfaces = len(face_ids.groups)

    max_face_vertices = max(len(ref_fvi) for grp in groups
        for ref_fvi in grp.face_vertex_indices())

    face_vertex_indices = np.empty((max_face_vertices, nfaces),
        dtype=vertex_id_dtype)
    face_vertex_indices[:, :] = -1

    for igrp, grp in enumerate(groups):
        for fid, ref_fvi in enumerate(grp.face_vertex_indices()):
            indices, = np.where((face_ids.groups == igrp) & (face_ids.faces == fid))
            grp_fvi = grp.vertex_indices[face_ids.elements[indices], :][:, ref_fvi]
            face_vertex_indices[:len(ref_fvi), indices] = (
                vertex_index_map_func(grp_fvi).T)

    # Normalize vertex-based "face identifiers" by sorting
    face_vertex_indices_increasing = np.sort(face_vertex_indices, axis=0)
    # Lexicographically sort the face vertex indices, then diff the result to find
    # faces with the same vertices
    order = np.lexsort(face_vertex_indices_increasing)
    diffs = np.diff(face_vertex_indices_increasing[:, order], axis=1)
    match_indices, = (~np.any(diffs, axis=0)).nonzero()

    return np.stack((order[match_indices], order[match_indices+1]))


def _compute_facial_adjacency_from_vertices(
        groups, element_id_dtype, face_id_dtype, tag_to_faces=None
        ) -> Sequence[Sequence[FacialAdjacencyGroup]]:
    if not groups:
        return []

    if tag_to_faces is None:
        tag_to_faces = [{} for grp in groups]

    # Match up adjacent faces according to their vertex indices

    face_ids_per_group = []
    for igrp, grp in enumerate(groups):
        indices = np.indices((grp.nfaces, grp.nelements), dtype=element_id_dtype)
        face_ids_per_group.append(_FaceIDs(
            groups=np.full(grp.nelements * grp.nfaces, igrp),
            elements=indices[1].flatten(),
            faces=indices[0].flatten().astype(face_id_dtype)))
    face_ids = _concatenate_face_ids(face_ids_per_group)

    face_index_pairs = _match_faces_by_vertices(groups, face_ids)

    del igrp
    del grp

    # Get ((grp#, elem#, face#), (neighbor grp#, neighbor elem#, neighbor face#))
    # for every face (both ways)

    face_index_pairs_both_ways = np.stack((
        np.concatenate((
            face_index_pairs[0, :],
            face_index_pairs[1, :])),
        np.concatenate((
            face_index_pairs[1, :],
            face_index_pairs[0, :]))))
    # Accomplish a sort by group, then neighbor group. This is done by sorting by
    # the indices in face_ids. Realize that those are already ordered by group by
    # construction.
    order = np.lexsort((
        face_index_pairs_both_ways[1, :],
        face_index_pairs_both_ways[0, :]))
    face_index_pairs_both_ways_sorted = face_index_pairs_both_ways[:, order]

    face_id_pairs = (
        _FaceIDs(
            groups=face_ids.groups[face_index_pairs_both_ways_sorted[0, :]],
            elements=face_ids.elements[face_index_pairs_both_ways_sorted[0, :]],
            faces=face_ids.faces[face_index_pairs_both_ways_sorted[0, :]]),
        _FaceIDs(
            groups=face_ids.groups[face_index_pairs_both_ways_sorted[1, :]],
            elements=face_ids.elements[face_index_pairs_both_ways_sorted[1, :]],
            faces=face_ids.faces[face_index_pairs_both_ways_sorted[1, :]]))

    # {{{ build facial_adjacency_groups data structure

    facial_adjacency_groups = []
    for igrp, grp in enumerate(groups):
        grp_list = []

        face_has_neighbor = np.full((grp.nfaces, grp.nelements), False)

        is_grp_adj = face_id_pairs[0].groups == igrp
        connected_groups = np.unique(face_id_pairs[1].groups[is_grp_adj])
        for i_neighbor_grp in connected_groups:
            is_neighbor_adj = (
                is_grp_adj & (face_id_pairs[1].groups == i_neighbor_grp))
            grp_list.append(
                InteriorAdjacencyGroup(
                    igroup=igrp,
                    ineighbor_group=i_neighbor_grp,
                    elements=face_id_pairs[0].elements[is_neighbor_adj],
                    element_faces=face_id_pairs[0].faces[is_neighbor_adj],
                    neighbors=face_id_pairs[1].elements[is_neighbor_adj],
                    neighbor_faces=face_id_pairs[1].faces[is_neighbor_adj],
                    aff_map=AffineMap(),
                    ))
            face_has_neighbor[
                face_id_pairs[0].faces[is_neighbor_adj],
                face_id_pairs[0].elements[is_neighbor_adj]] = True

        has_bdry = not np.all(face_has_neighbor)
        if has_bdry:
            bdry_element_faces, bdry_elements = np.where(~face_has_neighbor)
            bdry_element_faces = bdry_element_faces.astype(face_id_dtype)
            bdry_elements = bdry_elements.astype(element_id_dtype)

            is_tagged = np.full(len(bdry_elements), False)

            for tag, tagged_elements_and_faces in tag_to_faces[igrp].items():
                # Combine known tagged faces and current boundary faces into a
                # single array, lexicographically sort them, and find identical
                # neighboring entries to get tags
                extended_elements_and_faces = np.concatenate((
                    tagged_elements_and_faces,
                    np.stack(
                        (bdry_elements, bdry_element_faces),
                        axis=-1)))
                order = np.lexsort(extended_elements_and_faces.T)
                diffs = np.diff(extended_elements_and_faces[order, :], axis=0)
                match_indices, = (~np.any(diffs, axis=1)).nonzero()
                # lexsort is stable, so the second entry in each match corresponds
                # to the yet-to-be-tagged boundary face
                face_indices = (
                    order[match_indices+1]
                    - len(tagged_elements_and_faces))
                if len(face_indices) > 0:
                    elements = bdry_elements[face_indices]
                    element_faces = bdry_element_faces[face_indices]
                    grp_list.append(
                        BoundaryAdjacencyGroup(
                            igroup=igrp,
                            boundary_tag=tag,
                            elements=elements,
                            element_faces=element_faces))
                    is_tagged[face_indices] = True

            if np.any(~is_tagged):
                grp_list.append(
                    BoundaryAdjacencyGroup(
                        igroup=igrp,
                        boundary_tag=BTAG_ALL,
                        elements=bdry_elements,
                        element_faces=bdry_element_faces))

        facial_adjacency_groups.append(grp_list)

    # }}}

    return _complete_facial_adjacency_groups(
        facial_adjacency_groups, element_id_dtype, face_id_dtype)

# }}}


# {{{ complete facial adjacency groups

def _merge_boundary_adjacency_groups(
        igrp, bdry_grps, merged_btag, element_id_dtype, face_id_dtype):
    """
    Create a new :class:`~meshmode.mesh.BoundaryAdjacencyGroup` containing all of
    the entries from a list of existing boundary adjacency groups.
    """

    if len(bdry_grps) == 0:
        return BoundaryAdjacencyGroup(
            igroup=igrp,
            boundary_tag=merged_btag,
            elements=np.empty((0,), dtype=element_id_dtype),
            element_faces=np.empty((0,), dtype=face_id_dtype))

    max_ielem = max([
        np.max(grp.elements, initial=0)
        for grp in bdry_grps])
    max_iface = max([
        np.max(grp.element_faces, initial=0)
        for grp in bdry_grps])

    face_has_adj = np.full((max_iface+1, max_ielem+1), False)

    for grp in bdry_grps:
        face_has_adj[grp.element_faces, grp.elements] = True

    faces, elements = np.where(face_has_adj)
    merged_elements = elements.astype(element_id_dtype)
    merged_element_faces = faces.astype(face_id_dtype)

    return BoundaryAdjacencyGroup(
        igroup=igrp,
        boundary_tag=merged_btag,
        elements=merged_elements,
        element_faces=merged_element_faces)


def _complete_facial_adjacency_groups(
        facial_adjacency_groups, element_id_dtype, face_id_dtype):
    """
    Add :class:`~meshmode.mesh.BoundaryAdjacencyGroup` instances for
    :class:`~meshmode.mesh.BTAG_NONE`, :class:`~meshmode.mesh.BTAG_ALL`, and
    :class:`~meshmode.mesh.BTAG_REALLY_ALL` to a facial adjacency group list if
    they are not present.
    """

    completed_facial_adjacency_groups = facial_adjacency_groups.copy()

    for igrp, fagrp_list in enumerate(facial_adjacency_groups):
        completed_fagrp_list = completed_facial_adjacency_groups[igrp]

        bdry_grps = [
            grp for grp in fagrp_list
            if isinstance(grp, BoundaryAdjacencyGroup)]

        bdry_tags = {grp.boundary_tag for grp in bdry_grps}

        if BTAG_NONE not in bdry_tags:
            completed_fagrp_list.append(
                BoundaryAdjacencyGroup(
                    igroup=igrp,
                    boundary_tag=BTAG_NONE,
                    elements=np.empty((0,), dtype=element_id_dtype),
                    element_faces=np.empty((0,), dtype=face_id_dtype)))

        if BTAG_ALL not in bdry_tags:
            true_bdry_grps = [
                grp for grp in bdry_grps
                if is_true_boundary(grp.boundary_tag)]

            completed_fagrp_list.append(
                _merge_boundary_adjacency_groups(
                    igrp, true_bdry_grps, BTAG_ALL, element_id_dtype, face_id_dtype))

        if BTAG_REALLY_ALL not in bdry_tags:
            completed_fagrp_list.append(
                _merge_boundary_adjacency_groups(
                    igrp, bdry_grps, BTAG_REALLY_ALL, element_id_dtype,
                    face_id_dtype))

    return completed_facial_adjacency_groups

# }}}


# {{{ as_python

def _boundary_tag_as_python(boundary_tag):
    if isinstance(boundary_tag, type):
        return boundary_tag.__name__
    elif isinstance(boundary_tag, str):
        return boundary_tag
    else:
        return boundary_tag.as_python()


def _numpy_array_as_python(array):
    if array is not None:
        return "np.array({}, dtype=np.{})".format(
                repr(array.tolist()),
                array.dtype.name)
    else:
        return "None"


def _affine_map_as_python(aff_map):
    return ("AffineMap("
        + _numpy_array_as_python(aff_map.matrix) + ", "
        + _numpy_array_as_python(aff_map.offset) + ")")


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
            Mesh,
            MeshElementGroup,
            FacialAdjacencyGroup,
            InteriorAdjacencyGroup,
            BoundaryAdjacencyGroup,
            InterPartAdjacencyGroup,
            BTAG_NONE,
            BTAG_ALL,
            BTAG_REALLY_ALL)
        from meshmode.mesh.tools import AffineMap

        """)

    cg("def %s():" % function_name)
    with Indentation(cg):
        cg("vertices = " + _numpy_array_as_python(mesh.vertices))
        cg("")
        cg("groups = []")
        cg("")
        for group in mesh.groups:
            cg("import %s" % type(group).__module__)
            cg("groups.append({}.{}.make_group(".format(
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

        if mesh._facial_adjacency_groups:
            cg("facial_adjacency_groups = []")

            for fagrp_list in mesh.facial_adjacency_groups:
                cg("facial_adjacency_groups.append([")
                with Indentation(cg):
                    for fagrp in fagrp_list:
                        cg(fagrp.as_python() + ",")
                cg("])")

        else:
            cg("facial_adjacency_groups = %r" % mesh._facial_adjacency_groups)

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
        cg("    is_conforming=%s)" % repr(mesh.is_conforming))

        # FIXME: Handle facial adjacency, boundary tags

    return cg.get()

# }}}


# {{{ is_true_boundary

def is_true_boundary(boundary_tag):
    if boundary_tag == BTAG_REALLY_ALL:
        return False
    elif isinstance(boundary_tag, type):
        return not issubclass(boundary_tag, BTAG_NO_BOUNDARY)
    else:
        return not isinstance(boundary_tag, BTAG_NO_BOUNDARY)

# }}}


# {{{ mesh_has_boundary

def mesh_has_boundary(mesh, boundary_tag):
    for fagrp_list in mesh.facial_adjacency_groups:
        matching_bdry_grps = [
            fagrp for fagrp in fagrp_list
            if isinstance(fagrp, BoundaryAdjacencyGroup)
            and fagrp.boundary_tag == boundary_tag]
        if len(matching_bdry_grps) > 0:
            return True
    return False

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
    :arg true_boundary_only: only verify for faces whose tags do not
        inherit from `BTAG_NO_BOUNDARY`.
    """

    for boundary_tag in boundary_tags:
        if not mesh_has_boundary(mesh, boundary_tag):
            raise ValueError(f"invalid boundary tag {boundary_tag}.")

    for igrp, grp in enumerate(mesh.groups):
        fagrp_list = mesh.facial_adjacency_groups[igrp]
        if true_boundary_only:
            all_btag = BTAG_ALL
        else:
            all_btag = BTAG_REALLY_ALL

        all_bdry_grp, = [
            fagrp for fagrp in fagrp_list
            if isinstance(fagrp, BoundaryAdjacencyGroup)
            and fagrp.boundary_tag == all_btag]

        matching_bdry_grps = [
            fagrp for fagrp in fagrp_list
            if isinstance(fagrp, BoundaryAdjacencyGroup)
            and fagrp.boundary_tag in boundary_tags]

        def get_bdry_counts(bdry_grp):
            counts = np.full((grp.nfaces, grp.nelements), 0)  # noqa: B023
            counts[bdry_grp.element_faces, bdry_grp.elements] += 1
            return counts

        all_bdry_counts = get_bdry_counts(all_bdry_grp)

        if len(matching_bdry_grps) > 0:
            from functools import reduce
            matching_bdry_counts = reduce(np.add, [
                get_bdry_counts(bdry_grp)
                for bdry_grp in matching_bdry_grps])

            if np.any(matching_bdry_counts > 1):
                raise RuntimeError("found faces with multiple boundary conditions")

            if not incomplete_ok and np.any(matching_bdry_counts < all_bdry_counts):
                raise RuntimeError("found faces without boundary conditions")

        else:
            if not incomplete_ok and np.any(all_bdry_counts):
                raise RuntimeError("found faces without boundary conditions")

# }}}


# {{{ is_boundary_tag_empty

def is_boundary_tag_empty(mesh, boundary_tag):
    """Return *True* if the corresponding boundary tag does not occur as part of
    *mesh*.
    """
    if not mesh_has_boundary(mesh, boundary_tag):
        raise ValueError(f"invalid boundary tag {boundary_tag}.")

    for igrp in range(len(mesh.groups)):
        nfaces = sum([
            len(grp.elements) for grp in mesh.facial_adjacency_groups[igrp]
            if isinstance(grp, BoundaryAdjacencyGroup)
            and grp.boundary_tag == boundary_tag])
        if nfaces > 0:
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

    if group.nelements == 0:
        # All zero of them are affine! :)
        return True

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
