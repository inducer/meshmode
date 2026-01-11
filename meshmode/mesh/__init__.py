from __future__ import annotations


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
from collections.abc import Callable, Collection, Hashable, Iterable, Mapping, Sequence
from dataclasses import InitVar, dataclass, field, replace
from functools import partial
from typing import (
    Any,
    ClassVar,
    Literal,
    TypeAlias,
    TypeVar,
)
from warnings import warn

import numpy as np
import numpy.linalg as la
from typing_extensions import override

import modepy as mp
from pytools import memoize_method, module_getattr_for_deprecations

from meshmode.mesh.tools import AffineMap, optional_array_equal


__doc__ = """
.. autoclass:: MeshElementGroup
.. autoclass:: ModepyElementGroup
.. autoclass:: SimplexElementGroup
.. autoclass:: TensorProductElementGroup

.. autoclass:: Mesh
.. autofunction:: make_mesh
.. autofunction:: check_mesh_consistency
.. autofunction:: is_mesh_consistent

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

.. class:: BoundaryTag

    A type alias of :class:`typing.Hashable`.

.. class:: PartID

    A type alias of :class:`typing.Hashable`.

.. autoclass:: BTAG_NONE
.. autoclass:: BTAG_ALL
.. autoclass:: BTAG_REALLY_ALL
.. autoclass:: BTAG_NO_BOUNDARY
.. autoclass:: BTAG_PARTITION
.. autoclass:: BTAG_INDUCED_BOUNDARY
"""

IndexArray: TypeAlias = np.ndarray[tuple[int], np.dtype[np.integer[Any]]]

VertexIndices: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.integer[Any]]]
VertexArray: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.floating[Any]]]

NodesArray: TypeAlias = np.ndarray[tuple[int, int, int], np.dtype[np.floating[Any]]]
RefNodesArray: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.floating[Any]]]

# {{{ element tags

BoundaryTag: TypeAlias = Hashable
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

    .. autoattribute:: part_id
    .. automethod:: as_python

    .. versionadded:: 2017.1
    """

    part_id: PartID

    def __init__(self, part_id: PartID) -> None:
        self.part_id = part_id

    @override
    def __hash__(self) -> int:
        return hash((type(self), self.part_id))

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, BTAG_PARTITION):
            return self.part_id == other.part_id
        else:
            return False

    @override
    def __repr__(self) -> str:
        return "<{}({})>".format(type(self).__name__, repr(self.part_id))

    def as_python(self) -> str:
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

    .. autoattribute:: order
    .. autoattribute:: vertex_indices
    .. autoattribute:: nodes
    .. autoattribute:: unit_nodes

    .. autoproperty:: dim
    .. autoproperty:: nvertices
    .. autoproperty:: nfaces
    .. autoproperty:: nunit_nodes
    .. autoproperty:: nelements
    .. autoproperty:: nnodes
    .. autoproperty:: is_affine

    Element groups can also be compared for equality using the following
    methods. Note that these are *very* expensive, as they compare all
    the :attr:`nodes`.

    .. automethod:: __eq__
    .. automethod:: __init__

    The following abstract methods must be implemented by subclasses.

    .. automethod:: make_group
    .. automethod:: face_vertex_indices
    .. automethod:: vertex_unit_coordinates
    """

    order: int
    """"The maximum degree used for interpolation. The exact meaning depends on
    the element type, e.g. for :class:`SimplexElementGroup` this is the total degree.
    """
    vertex_indices: VertexIndices | None
    """An array of shape ``(nelements, nvertices)`` of (mesh-wide) vertex indices.
    This can also be *None* to support the case where the associated mesh does
    not have any :attr:`~Mesh.vertices`.
    """
    nodes: NodesArray
    """An array of node coordinates with shape
    ``(mesh.ambient_dim, nelements, nunit_nodes)``.
    """
    unit_nodes: RefNodesArray
    """An array with shape ``(dim, nunit_nodes)`` of nodes on the reference
    element. The coordinates :attr:`nodes` are a mapped version
    of these reference nodes.
    """

    @property
    def dim(self) -> int:
        """The number of dimensions spanned by the element. *Not* the ambient
        dimension, see :attr:`Mesh.ambient_dim` for that.
        """
        return self.unit_nodes.shape[0]

    @property
    def nvertices(self) -> int:
        """Number of vertices in the reference element."""
        return self.vertex_unit_coordinates().shape[-1]

    @property
    def nfaces(self) -> int:
        """Number of faces of the reference element."""
        return len(self.face_vertex_indices())

    @property
    def nunit_nodes(self) -> int:
        """Number of nodes on the reference element."""
        return self.unit_nodes.shape[-1]

    @property
    def nelements(self) -> int:
        """Number of elements in the group."""
        return self.nodes.shape[1]

    @property
    def nnodes(self) -> int:
        """Total number of nodes in the group (equivalent to
        ``nelements * nunit_nodes``).
        """
        return self.nelements * self.unit_nodes.shape[-1]

    @override
    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return False
        assert isinstance(other, MeshElementGroup)

        return (
                self.order == other.order
                and optional_array_equal(self.vertex_indices, other.vertex_indices)
                and np.array_equal(self.nodes, other.nodes)
                and np.array_equal(self.unit_nodes, other.unit_nodes))

    @property
    def is_affine(self) -> bool:
        """A :class:`bool` flag that is *True* if the local-to-global
        parametrization of all the elements in the group is affine.
        """
        raise NotImplementedError

    @abstractmethod
    def face_vertex_indices(self) -> tuple[tuple[int, ...], ...]:
        """
        :returns: a :class:`tuple` of tuples indicating which vertices
            (in mathematically positive ordering) make up each face
            of an element in this group.
        """

    @abstractmethod
    def vertex_unit_coordinates(self) -> RefNodesArray:
        """
        :returns: an array of shape ``(nfaces, dim)`` with the unit
            coordinates of each vertex.
        """

    @classmethod
    @abstractmethod
    def make_group(cls, *args: Any, **kwargs: Any) -> MeshElementGroup:
        """Instantiate a new group of class *cls*.

        Unlike the constructor, this factory function performs additional
        consistency checks and should be used instead.
        """

# }}}


# {{{ modepy-based element group

# https://stackoverflow.com/a/13624858
class _classproperty(property):  # noqa: N801
    @override
    def __get__(self, owner_self: Any, owner_cls: type | None = None) -> Any:
        assert self.fget is not None
        return self.fget(owner_cls)


@dataclass(frozen=True, eq=False)
class ModepyElementGroup(MeshElementGroup):
    """
    .. autoattribute:: shape_cls
    .. autoattribute:: shape
    .. autoattribute:: space
    """

    shape_cls: ClassVar[type[mp.Shape]]
    """Must be set by subclasses to generate the correct shape and spaces
    attributes for the group.
    """
    shape: mp.Shape = field(repr=False)
    space: mp.FunctionSpace = field(repr=False)

    @property
    @override
    def nvertices(self) -> int:
        return self.shape.nvertices     # pylint: disable=no-member

    @property
    @memoize_method
    def _modepy_faces(self) -> Sequence[mp.Face]:
        return mp.faces_for_shape(self.shape)

    @memoize_method
    def face_vertex_indices(self) -> tuple[tuple[int, ...], ...]:
        return tuple(face.volume_vertex_indices for face in self._modepy_faces)

    @memoize_method
    def vertex_unit_coordinates(self) -> RefNodesArray:
        return mp.unit_vertices_for_shape(self.shape).T

    @classmethod
    @override
    def make_group(cls,
                   order: int,
                   vertex_indices: VertexIndices | None,
                   nodes: NodesArray,
                   *,
                   unit_nodes: RefNodesArray | None = None,
                   dim: int | None = None) -> ModepyElementGroup:

        if unit_nodes is None:
            if dim is None:
                raise TypeError("either 'dim' or 'unit_nodes' must be provided")
        else:
            if dim is None:
                dim = unit_nodes.shape[0]

            if unit_nodes.shape[0] != dim:
                raise ValueError("'dim' does not match 'unit_nodes' dimension")

        assert dim is not None

        # pylint: disable=abstract-class-instantiated
        shape = cls.shape_cls(dim)
        space = mp.space_for_shape(shape, order)

        if unit_nodes is None:
            unit_nodes = mp.edge_clustered_nodes_for_space(space, shape)

        if unit_nodes.shape[1] != space.space_dim:
            raise ValueError("'unit_nodes' size does not match the dimension "
                             f"of a '{type(space).__name__}' space of order {order}")

        return cls(order=order,
                   vertex_indices=vertex_indices,
                   nodes=nodes,
                   unit_nodes=unit_nodes,
                   shape=shape,
                   space=space)

    @_classproperty
    def _modepy_shape_cls(cls) -> type[mp.Shape]:  # noqa: N805  # pylint: disable=no-self-argument
        return cls.shape_cls

    @property
    def _modepy_shape(self) -> mp.Shape:
        return self.shape

    @property
    def _modepy_space(self) -> mp.FunctionSpace:
        return self.space

# }}}


@dataclass(frozen=True, eq=False)
class SimplexElementGroup(ModepyElementGroup):
    r"""Inherits from :class:`MeshElementGroup`."""

    shape_cls: ClassVar[type[mp.Shape]] = mp.Simplex

    @property
    @memoize_method
    def is_affine(self) -> bool:
        return is_affine_simplex_group(self)


@dataclass(frozen=True, eq=False)
class TensorProductElementGroup(ModepyElementGroup):
    r"""Inherits from :class:`MeshElementGroup`."""

    shape_cls: ClassVar[type[mp.Shape]] = mp.Hypercube

    @property
    @override
    def is_affine(self) -> bool:
        # Tensor product mappings are generically bilinear.
        # FIXME: Are affinely mapped ones a 'juicy' enough special case?
        return False

# }}}


# {{{ nodal adjacency

@dataclass(frozen=True, eq=False)
class NodalAdjacency:
    """Describes nodal element adjacency information, i.e. information about
    elements that touch in at least one point.

    .. autoattribute:: neighbors_starts
    .. autoattribute:: neighbors

    .. automethod:: __eq__
    """

    neighbors_starts: IndexArray
    """"
    ``element_id_t [nelements+1]``

    Use together with :attr:`neighbors`.  ``neighbors_starts[iel]`` and
    ``neighbors_starts[iel+1]`` together indicate a ranges of element indices
    :attr:`neighbors` which are adjacent to *iel*.
    """

    neighbors: IndexArray
    """
    ``element_id_t []``

    See :attr:`neighbors_starts`.
    """

    @override
    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return False
        assert isinstance(other, NodalAdjacency)

        return (
                np.array_equal(self.neighbors_starts, other.neighbors_starts)
                and np.array_equal(self.neighbors, other.neighbors))

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

    .. autoattribute:: igroup
    .. autoattribute:: elements
    .. autoattribute:: element_faces

    .. automethod:: as_python
    """

    igroup: int
    """The mesh element group number of this group."""
    elements: IndexArray
    element_faces: IndexArray

    @override
    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return False
        assert isinstance(other, FacialAdjacencyGroup)

        return self.igroup == other.igroup

    def _as_python(self, **kwargs: Any) -> str:
        return "{cls}({args})".format(
                cls=self.__class__.__name__,
                args=",\n    ".join(f"{k}={v}" for k, v in kwargs.items())
                )

    def as_python(self) -> str:
        """
        :returns: a string that can be evaluated to reconstruct the class.
        """

        if type(self) is not FacialAdjacencyGroup:
            raise NotImplementedError(
                    f"Not implemented for '{type(self).__name__}'.")

        return self._as_python(igroup=self.igroup)

# }}}


# {{{ interior adjacency

@dataclass(frozen=True, eq=False)
class InteriorAdjacencyGroup(FacialAdjacencyGroup):
    """Describes interior facial element adjacency information for one
    :class:`MeshElementGroup`.

    .. autoattribute:: igroup
    .. autoattribute:: ineighbor_group
    .. autoattribute:: elements
    .. autoattribute:: element_faces
    .. autoattribute:: neighbors
    .. autoattribute:: neighbor_faces
    .. autoattribute:: aff_map

    .. automethod:: __eq__
    .. automethod:: as_python
    """

    ineighbor_group: int
    """ID of neighboring group, or *None* for boundary faces. If identical
    to :attr:`igroup`, then this contains the self-connectivity in this group.
    """

    elements: IndexArray
    """``element_id_t [nfagrp_elements]``. ``elements[i]`` gives the
    element number within :attr:`igroup` of the interior face."""

    element_faces: IndexArray
    """``face_id_t [nfagrp_elements]``. ``element_faces[i]`` gives the face
    index of the interior face in element ``elements[i]``.
    """

    neighbors: IndexArray
    """``element_id_t [nfagrp_elements]``. ``neighbors[i]`` gives the element
    number within :attr:`ineighbor_group` of the element opposite
    ``elements[i]``.
    """

    neighbor_faces: IndexArray
    """``face_id_t [nfagrp_elements]``. ``neighbor_faces[i]`` gives the
    face index of the opposite face in element ``neighbors[i]``
    """

    aff_map: AffineMap
    """An :class:`~meshmode.AffineMap` representing the mapping from the group's
    faces to their corresponding neighbor faces.
    """

    @override
    def __eq__(self, other: object) -> bool:
        if not super().__eq__(other):
            return False
        assert isinstance(other, InteriorAdjacencyGroup)

        return (
            self.ineighbor_group == other.ineighbor_group
            and np.array_equal(self.elements, other.elements)
            and np.array_equal(self.element_faces, other.element_faces)
            and np.array_equal(self.neighbors, other.neighbors)
            and np.array_equal(self.neighbor_faces, other.neighbor_faces)
            and self.aff_map == other.aff_map)

    @override
    def as_python(self) -> str:
        if type(self) is not InteriorAdjacencyGroup:
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

    .. autoattribute:: igroup
    .. autoattribute:: boundary_tag
    .. autoattribute:: elements
    .. autoattribute:: element_faces

    .. automethod:: as_python
    """

    boundary_tag: BoundaryTag
    """"The boundary tag identifier of this group."""

    elements: IndexArray
    """"
    ``element_id_t [nfagrp_elements]``. ``elements[i]`` gives the
    element number within :attr:`igroup` of the boundary face.
    """

    element_faces: IndexArray
    """"
    ``face_id_t [nfagrp_elements]``. ``element_faces[i]`` gives the face
    index of the boundary face in element ``elements[i]``.
    """

    @override
    def __eq__(self, other: object) -> bool:
        if not super().__eq__(other):
            return False
        assert isinstance(other, BoundaryAdjacencyGroup)

        return (
            self.boundary_tag == other.boundary_tag
            and np.array_equal(self.elements, other.elements)
            and np.array_equal(self.element_faces, other.element_faces))

    @override
    def as_python(self) -> str:
        if type(self) is not BoundaryAdjacencyGroup:
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
    Describes inter-part adjacency information for one :class:`MeshElementGroup`.

    .. autoattribute:: igroup
    .. autoattribute:: boundary_tag
    .. autoattribute:: part_id
    .. autoattribute:: elements
    .. autoattribute:: element_faces
    .. autoattribute:: neighbors
    .. autoattribute:: neighbor_faces
    .. autoattribute:: aff_map

    .. automethod:: as_python

    .. versionadded:: 2017.1
    """

    igroup: int
    """The mesh element group number of this group.
    """

    boundary_tag: BoundaryTag
    """The boundary tag identifier of this group. Will be an instance of
    :class:`~meshmode.mesh.BTAG_PARTITION`.
    """

    part_id: PartID
    """The identifier of the neighboring part.
    """

    elements: IndexArray
    """Group-local element numbers.
    Element ``element_id_dtype elements[i]`` and face
    ``face_id_dtype element_faces[i]`` is connected to neighbor element
    ``element_id_dtype neighbors[i]`` with face
    ``face_id_dtype neighbor_faces[i]``.
    """

    element_faces: IndexArray
    """``face_id_dtype element_faces[i]`` gives the face of
    ``element_id_dtype elements[i]`` that is connected to ``neighbors[i]``.
    """

    neighbors: IndexArray
    """``element_id_dtype neighbors[i]`` gives the volume element number
    within the neighboring part of the element connected to
    ``element_id_dtype elements[i]`` (which is a boundary element index). Use
    `~meshmode.mesh.processing.find_group_indices` to find the group that
    the element belongs to, then subtract ``element_nr_base`` to find the
    element of the group.
    """

    neighbor_faces: IndexArray
    """``face_id_dtype global_neighbor_faces[i]`` gives face index within the
    neighboring part of the face connected to ``element_id_dtype elements[i]``
    """

    aff_map: AffineMap
    """An :class:`~meshmode.AffineMap` representing the mapping from the group's
    faces to their corresponding neighbor faces.
    """

    @override
    def __eq__(self, other: object) -> bool:
        if not super().__eq__(other):
            return False
        assert isinstance(other, InterPartAdjacencyGroup)

        return (
            self.part_id == other.part_id
            and np.array_equal(self.neighbors, other.neighbors)
            and np.array_equal(self.neighbor_faces, other.neighbor_faces)
            and self.aff_map == other.aff_map)

    @override
    def as_python(self) -> str:
        if type(self) is not InterPartAdjacencyGroup:
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

DTypeLike = np.dtype | np.generic
NodalAdjacencyLike = (
    Literal[False] | Iterable[IndexArray] | NodalAdjacency
    )
FacialAdjacencyLike = (
    Literal[False] | Sequence[Sequence[FacialAdjacencyGroup]]
    )


def check_mesh_consistency(
        mesh: Mesh,
        *,
        node_vertex_consistency_tolerance: Literal[False] | float | None = None,
        skip_element_orientation_test: bool = False,
        ) -> None:
    """Check the mesh for consistency between the vertices, nodes, and their
    adjacency.

    This function checks:

    * The node to vertex consistency, by interpolation.
    * The :class:`~numpy.dtype` of the various arrays matching the ones in
      :class:`Mesh`.
    * The nodal adjacency shapes and dtypes.
    * The facial adjacency shapes and dtypes.
    * The mesh orientation using
      :func:`~meshmode.mesh.processing.find_volume_mesh_element_orientations`.

    :arg node_vertex_consistency_tolerance: If *False*, do not check for
        consistency between vertex and nodal data. If *None*, a default tolerance
        based on the :class:`~numpy.dtype` of the *vertices* array will be used.
        Otherwise, the given value is used as the tolerance.
    :arg skip_element_orientation_test: If *False*, check that element
        orientation is positive in volume meshes (i.e. ones where ambient and
        topological dimension match).

    :raises InconsistentMeshError: when the mesh is found to be inconsistent in
        some fashion.
    """
    from meshmode import (
        InconsistentAdjacencyError,
        InconsistentArrayDTypeError,
        InconsistentMeshError,
    )

    if node_vertex_consistency_tolerance is not False:
        _test_node_vertex_consistency(mesh, tol=node_vertex_consistency_tolerance)

    for i, g in enumerate(mesh.groups):
        if g.vertex_indices is None:
            continue

        if g.vertex_indices.dtype != mesh.vertex_id_dtype:
            raise InconsistentArrayDTypeError(
                f"Group '{i}' attribute 'vertex_indices' has incorrect dtype: "
                f"{g.vertex_indices.dtype!r} (expected mesh 'vertex_id_dtype' = "
                f"{mesh.vertex_id_dtype!r})")

    nodal_adjacency = mesh._nodal_adjacency
    if nodal_adjacency:
        if nodal_adjacency.neighbors_starts.shape != (mesh.nelements + 1,):
            raise InconsistentAdjacencyError(
                "Nodal adjacency 'neighbors_starts' has incorrect shape: "
                f"'{nodal_adjacency.neighbors_starts.shape}' (expected "
                f"nelements + 1 = {mesh.nelements + 1})")

        if len(nodal_adjacency.neighbors.shape) != 1:
            raise InconsistentAdjacencyError(
                "Nodal adjacency 'neighbors' have incorrect dim: "
                f"{nodal_adjacency.neighbors.shape} (expected ndim = 1)")

        if nodal_adjacency.neighbors_starts.dtype != mesh.element_id_dtype:
            raise InconsistentArrayDTypeError(
                "Nodal adjacency 'neighbors_starts' has incorrect dtype: "
                f"{nodal_adjacency.neighbors_starts.dtype!r} (expected mesh "
                f"'element_id_dtype' = {mesh.element_id_dtype!r})")

        if nodal_adjacency.neighbors.dtype != mesh.element_id_dtype:
            raise InconsistentArrayDTypeError(
                "Nodal adjacency 'neighbors' has incorrect dtype: "
                f"{nodal_adjacency.neighbors.dtype!r} (expected mesh "
                f"'element_id_dtype' = {mesh.element_id_dtype!r})")

    facial_adjacency_groups = mesh._facial_adjacency_groups
    if facial_adjacency_groups:
        if len(facial_adjacency_groups) != len(mesh.groups):
            raise InconsistentAdjacencyError(
                "Facial adjacency groups do not match mesh groups: "
                f"{len(facial_adjacency_groups)} (expected {len(mesh.groups)})")

        for igrp, fagrp_list in enumerate(facial_adjacency_groups):
            for ifagrp, fagrp in enumerate(fagrp_list):
                if len(fagrp.elements.shape) != 1:
                    raise InconsistentAdjacencyError(
                        f"Facial adjacency {ifagrp} for group {igrp} has incorrect "
                        f"'elements' shape: {fagrp.elements.shape} "
                        "(expected ndim = 1)")

                nfagrp_elements, = fagrp.elements.shape
                if fagrp.element_faces.shape != (nfagrp_elements,):
                    raise InconsistentAdjacencyError(
                        f"Facial adjacency {ifagrp} for group {igrp} has incorrect "
                        f"'element_faces' shape: {fagrp.element_faces.shape} "
                        f"(expected 'elements.shape' = {fagrp.elements.shape})")

                if fagrp.element_faces.dtype != mesh.face_id_dtype:
                    raise InconsistentArrayDTypeError(
                        f"Facial adjacency {ifagrp} for group {igrp} has "
                        "incorrect 'element_faces' dtype: "
                        f"{fagrp.element_faces.dtype!r} (expected mesh "
                        f"'face_id_dtype' = {mesh.face_id_dtype!r})")

                if isinstance(fagrp, InteriorAdjacencyGroup):
                    if fagrp.neighbors.dtype != mesh.element_id_dtype:
                        raise InconsistentArrayDTypeError(
                            f"Facial adjacency {ifagrp} for group {igrp} has "
                            "incorrect 'neighbors' dtype: "
                            f"{fagrp.neighbors.dtype!r} (expected mesh "
                            f"'element_id_dtype' = {mesh.element_id_dtype!r})")

                    if fagrp.neighbor_faces.dtype != mesh.face_id_dtype:
                        raise InconsistentArrayDTypeError(
                            f"Facial adjacency {ifagrp} for group {igrp} has "
                            "incorrect 'neighbor_faces' dtype: "
                            f"{fagrp.neighbor_faces.dtype!r} (expected mesh "
                            f"'face_id_dtype' = {mesh.face_id_dtype!r})")

                    if fagrp.neighbors.shape != (nfagrp_elements,):
                        raise InconsistentAdjacencyError(
                            f"Facial adjacency {ifagrp} for group {igrp} has "
                            "incorrect 'neighbors' shape: "
                            f"{fagrp.neighbors.shape} (expected "
                            f"'elements.shape' = {fagrp.elements.shape})")

                    if fagrp.neighbor_faces.shape != (nfagrp_elements,):
                        raise InconsistentAdjacencyError(
                            f"Facial adjacency {ifagrp} for group {igrp} has "
                            "incorrect 'neighbor_faces' shape: "
                            f"{fagrp.neighbor_faces.shape} (expected "
                            f"'elements.shape' = {fagrp.elements.shape})")

    from meshmode.mesh.processing import find_volume_mesh_element_orientations

    if not skip_element_orientation_test:
        if mesh.dim == mesh.ambient_dim:
            area_elements = find_volume_mesh_element_orientations(
                    mesh, tolerate_unimplemented_checks=True)
            valid = ~np.isnan(area_elements)
            if (~valid).any():
                warn("Some element orientations could not be checked due to "
                     "unimplemented orientation computations.", stacklevel=2)

            if not bool(np.all(area_elements[valid] > 0)):
                raise InconsistentMeshError(
                    "Mesh has negatively oriented elements. "
                    "To address this problem, create the mesh while providing the "
                    "parameter force_positive_orientation=True to make_mesh().")
        else:
            warn("Unimplemented: Cannot check element orientation for a mesh with "
                 "mesh.dim != mesh.ambient_dim", stacklevel=2)


def is_mesh_consistent(
        mesh: Mesh,
        *,
        node_vertex_consistency_tolerance: Literal[False] | float | None = None,
        skip_element_orientation_test: bool = False,
        ) -> bool:
    """A boolean version of :func:`check_mesh_consistency`."""

    from meshmode import InconsistentMeshError

    try:
        check_mesh_consistency(
            mesh,
            node_vertex_consistency_tolerance=node_vertex_consistency_tolerance,
            skip_element_orientation_test=skip_element_orientation_test)
    except InconsistentMeshError:
        return False
    else:
        return True


def make_mesh(
        vertices: VertexArray | None,
        groups: Iterable[MeshElementGroup],
        *,
        nodal_adjacency: NodalAdjacencyLike | None = None,
        facial_adjacency_groups: FacialAdjacencyLike | None = None,
        is_conforming: bool | None = None,
        # dtypes
        vertex_id_dtype: DTypeLike = np.dtype("int32"),  # noqa: B008
        element_id_dtype: DTypeLike = np.dtype("int32"),  # noqa: B008
        face_id_dtype: DTypeLike = np.dtype("int8"),  # noqa: B008
        # tests
        skip_tests: bool = False,
        node_vertex_consistency_tolerance: float | None = None,
        skip_element_orientation_test: bool = False,
        force_positive_orientation: bool = False,
        ) -> Mesh:
    """Construct a new mesh from a given list of *groups*.

    This constructor performs additional checks on the mesh once constructed and
    should be preferred to calling the constructor of the :class:`Mesh` class
    directly.

    :arg vertices: an array of vertices that match the given element *groups*.
        These can be *None* for meshes where adjacency is not required
        (e.g. non-conforming meshes).
    :arg nodal_adjacency: a definition of the nodal adjacency of the mesh.
        This argument can take one of four values:

        * *False*, in which case the information is marked as unavailable for
          this mesh and will not be computed at all. This should be used if the
          vertex adjacency does not convey the full picture, e.g if there are
          hanging nodes in the geometry.
        * *None*, in which case the nodal adjacency will be deduced from the
          vertex adjacency on demand (this requires the *vertices*).
        * a tuple of ``(element_neighbors_starts, element_neighbors)`` from which
          a :class:`NodalAdjacency` object can be constructed.
        * a :class:`NodalAdjacency` object.

    :arg facial_adjacency_groups: a definition of the facial adjacency for
        each group in the mesh. This argument can take one of three values:

        * *False*, in which case the information is marked as unavailable for
          this mesh and will not be computed.
        * *None*, in which case the facial adjacency will be deduced from the
          vertex adjacency on demand (this requires *vertices*).
        * an iterable of :class:`FacialAdjacencyGroup` objects.

    :arg is_conforming: *True* if the mesh is known to be conforming.

    :arg vertex_id_dtype: an integer :class:`~numpy.dtype` for the vertex indices.
    :arg element_id_dtype: an integer :class:`~numpy.dtype` for the element indices
        (relative to an element group).
    :arg face_id_dtype: an integer :class:`~numpy.dtype` for the face indices
        (relative to an element).

    :arg skip_tests: a flag used to skip any mesh consistency checks. This can
        be set to *True* in special situation, e.g. when loading a broken mesh
        that will be fixed later.
    :arg node_vertex_consistency_tolerance: see :func:`check_mesh_consistency`.
    :arg skip_element_orientation_test: see :func:`check_mesh_consistency`.
    """
    vertex_id_dtype = np.dtype(vertex_id_dtype)
    if vertex_id_dtype.kind not in {"i", "u"}:
        raise ValueError(
            f"'vertex_id_dtype' expected to be an integer kind: {vertex_id_dtype}"
            )

    element_id_dtype = np.dtype(element_id_dtype)
    if element_id_dtype.kind not in {"i", "u"}:
        raise ValueError(
            f"'element_id_dtype' expected to be an integer kind: {element_id_dtype}"
            )

    face_id_dtype = np.dtype(face_id_dtype)
    if face_id_dtype.kind not in {"i", "u"}:
        raise ValueError(
            f"'face_id_dtype' expected to be an integer kind: {face_id_dtype}"
            )

    if vertices is None:
        if is_conforming is not None:
            warn("No vertices provided and 'is_conforming' is set to "
                 f"'{is_conforming}'. Setting to 'None' instead, since no "
                 "adjacency can be known.",
                 UserWarning, stacklevel=2)

        is_conforming = None

    if not is_conforming:
        if nodal_adjacency is None:
            nodal_adjacency = False

        if facial_adjacency_groups is None:
            facial_adjacency_groups = False

    if (
            nodal_adjacency is not False
            and nodal_adjacency is not None
            and not isinstance(nodal_adjacency, NodalAdjacency)):
        nb_starts, nbs = nodal_adjacency
        nodal_adjacency = (
            NodalAdjacency(neighbors_starts=nb_starts, neighbors=nbs))

    if (
            facial_adjacency_groups is not False
            and facial_adjacency_groups is not None):
        facial_adjacency_groups = _complete_facial_adjacency_groups(
            facial_adjacency_groups,
            element_id_dtype,
            face_id_dtype)
        facial_adjacency_groups = tuple(tuple(grps) for grps in facial_adjacency_groups)

    mesh = Mesh(
        groups=tuple(groups),
        vertices=vertices,
        is_conforming=is_conforming,
        vertex_id_dtype=vertex_id_dtype,
        element_id_dtype=element_id_dtype,
        face_id_dtype=face_id_dtype,
        _nodal_adjacency=nodal_adjacency,
        _facial_adjacency_groups=facial_adjacency_groups,
        factory_constructed=True
        )

    if force_positive_orientation:
        if mesh.dim == mesh.ambient_dim:
            import meshmode.mesh.processing as mproc
            mesh = mproc.perform_flips(
                    mesh,  mproc.find_volume_mesh_element_orientations(mesh) < 0)
        else:
            raise ValueError("cannot enforce positive element orientation "
                             "on non-volume meshes")

        # By default, element orientation will be tested again below.
        # As a matter of defense-in-depth, that's probably a good idea,
        # in order to help defend against potential bugs in element flipping.

    if __debug__ and not skip_tests:
        check_mesh_consistency(
            mesh,
            node_vertex_consistency_tolerance=node_vertex_consistency_tolerance,
            skip_element_orientation_test=skip_element_orientation_test)

    return mesh


# TODO: should be `init=True` once everything is ported to `make_mesh`
@dataclass(frozen=True, init=False, eq=False)
class Mesh:
    """
    .. autoproperty:: ambient_dim
    .. autoproperty:: dim
    .. autoproperty:: nvertices
    .. autoproperty:: nelements
    .. autoproperty:: base_element_nrs
    .. autoproperty:: base_node_nrs
    .. autoproperty:: vertex_dtype

    .. autoattribute :: groups
    .. autoattribute:: vertices
    .. autoattribute:: is_conforming

    .. autoattribute:: vertex_id_dtype
    .. autoattribute:: element_id_dtype
    .. autoattribute:: face_id_dtype

    .. autoproperty:: nodal_adjacency
    .. autoproperty:: facial_adjacency_groups

    .. autoattribute:: _nodal_adjacency
    .. autoattribute:: _facial_adjacency_groups

    .. automethod:: copy
    .. automethod:: __eq__
    """

    groups: tuple[MeshElementGroup, ...]
    """A tuple of :class:`MeshElementGroup` instances."""

    vertices: VertexArray | None
    """*None* or an array of vertex coordinates with shape
    *(ambient_dim, nvertices)*. If *None*, vertices are not known for this mesh
    and no adjacency information can be constructed.
    """

    is_conforming: bool | None
    """*True* if it is known that all element interfaces are conforming. *False*
    if it is known that some element interfaces are non-conforming. *None* if
    neither of the two is known.
    """

    vertex_id_dtype: np.dtype
    """The :class:`~numpy.dtype` used to index into the vertex array."""

    element_id_dtype: np.dtype
    """The :class:`~numpy.dtype` used to index into the element array (relative
    to each group).
    """

    face_id_dtype: np.dtype
    """The :class:`~numpy.dtype` used to index element faces (relative to each
    element).
    """

    # TODO: Once the @property(nodal_adjacency) is past its deprecation period
    # and removed, these can deprecated in favor of non-underscored variants.

    _nodal_adjacency: Literal[False] | NodalAdjacency | None
    """A description of the nodal adjacency of the mesh. This can be *False* if
    no adjacency is known or should be computed, *None* to compute the adjacency
    on demand or a given :class:`NodalAdjacency` instance.

    This attribute caches the values of :attr:`nodal_adjacency`.
    """

    _facial_adjacency_groups: \
        Literal[False] | tuple[tuple[FacialAdjacencyGroup, ...], ...] | None
    """A description of the facial adjacency of the mesh. This can be *False* if
    no adjacency is known or should be computed, *None* to compute the adjacency
    on demand or a list of :class:`FacialAdjacencyGroup` instances.

    This attribute caches the values of :attr:`facial_adjacency_groups`.
    """

    # TODO: remove once porting to `make_mesh` is complete.
    skip_tests: InitVar[bool] = False
    node_vertex_consistency_tolerance: InitVar[
        Literal[False] | float | None] = None
    skip_element_orientation_test: InitVar[bool] = False
    factory_constructed: InitVar[bool] = False

    def __init__(
            self,
            vertices: VertexArray | None,
            groups: Iterable[MeshElementGroup],
            is_conforming: bool | None = None,
            vertex_id_dtype: DTypeLike = np.dtype("int32"),  # noqa: B008
            element_id_dtype: DTypeLike = np.dtype("int32"),  # noqa: B008
            face_id_dtype: DTypeLike = np.dtype("int8"),  # noqa: B008
            # cached variables
            nodal_adjacency: NodalAdjacencyLike | None = None,
            facial_adjacency_groups: FacialAdjacencyLike | None = None,
            _nodal_adjacency: NodalAdjacencyLike | None = None,
            _facial_adjacency_groups: FacialAdjacencyLike | None = None,
            # init vars
            skip_tests: bool = False,
            node_vertex_consistency_tolerance: float | None = None,
            skip_element_orientation_test: bool = False,
            factory_constructed: bool = False,
            ) -> None:
        if _nodal_adjacency is None:
            if nodal_adjacency is not None:
                warn("Passing 'nodal_adjacency' is deprecated and will be removed "
                     "in 2025. Use the underscored '_nodal_adjacency' instead to "
                     "match the dataclass field.",
                     DeprecationWarning, stacklevel=2)

            actual_nodal_adjacency = nodal_adjacency
        else:
            if nodal_adjacency is not None:
                raise TypeError("passing both _nodal_adjacency and nodal adjacency "
                                "is not allowed")
            else:
                actual_nodal_adjacency = _nodal_adjacency

        if _facial_adjacency_groups is None:
            if facial_adjacency_groups is not None:
                warn("Passing 'facial_adjacency_groups' is deprecated and will be "
                     "removed in 2025. Use the underscored '_facial_adjacency_groups'"
                     " instead to match the dataclass field.",
                     DeprecationWarning, stacklevel=2)

            actual_facial_adjacency_groups = facial_adjacency_groups
        else:
            if facial_adjacency_groups is not None:
                raise TypeError("passing both _facial_adjacency_groups "
                                "and facial adjacency_groups is not allowed")
            else:
                actual_facial_adjacency_groups = _facial_adjacency_groups

        if not factory_constructed:
            warn(f"Calling '{type(self).__name__}(...)' constructor is deprecated. "
                 "Use the 'make_mesh(...)' factory function instead. The input "
                 "handling in the constructor will be removed in 2025.",
                 DeprecationWarning, stacklevel=2)

            vertex_id_dtype = np.dtype(vertex_id_dtype)
            element_id_dtype = np.dtype(element_id_dtype)
            face_id_dtype = np.dtype(face_id_dtype)

            if vertices is None:
                is_conforming = None

            if not is_conforming:
                if actual_nodal_adjacency is None:
                    actual_nodal_adjacency = False
                if actual_facial_adjacency_groups is None:
                    actual_facial_adjacency_groups = False

            if (
                    actual_nodal_adjacency is not False
                    and actual_nodal_adjacency is not None):
                if not isinstance(actual_nodal_adjacency, NodalAdjacency):
                    nb_starts, nbs = actual_nodal_adjacency
                    actual_nodal_adjacency = NodalAdjacency(
                            neighbors_starts=nb_starts,
                            neighbors=nbs)

                    del nb_starts
                    del nbs

            if (
                    actual_facial_adjacency_groups is not False
                    and actual_facial_adjacency_groups is not None):
                actual_facial_adjacency_groups = _complete_facial_adjacency_groups(
                    actual_facial_adjacency_groups,
                    element_id_dtype,
                    face_id_dtype)

        object.__setattr__(self, "groups", tuple(groups))
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "is_conforming", is_conforming)
        object.__setattr__(self, "vertex_id_dtype", vertex_id_dtype)
        object.__setattr__(self, "element_id_dtype", element_id_dtype)
        object.__setattr__(self, "face_id_dtype", face_id_dtype)
        object.__setattr__(self, "_nodal_adjacency", actual_nodal_adjacency)
        object.__setattr__(self, "_facial_adjacency_groups",
                           actual_facial_adjacency_groups)

        if __debug__ and not factory_constructed and not skip_tests:
            check_mesh_consistency(
                self,
                node_vertex_consistency_tolerance=node_vertex_consistency_tolerance,
                skip_element_orientation_test=skip_element_orientation_test)

    def copy(self, *,
             skip_tests: bool = False,
             node_vertex_consistency_tolerance:
                 Literal[False] | bool | None = None,
             skip_element_orientation_test: bool = False,
             # NOTE: this is set to *True* to avoid the meaningless warning in
             # `__init__` when calling `Mesh.copy`
             factory_constructed: bool = True,
             **kwargs: Any) -> Mesh:
        if "nodal_adjacency" in kwargs:
            kwargs["_nodal_adjacency"] = kwargs.pop("nodal_adjacency")

        if "facial_adjacency_groups" in kwargs:
            kwargs["_facial_adjacency_groups"] = (
                kwargs.pop("facial_adjacency_groups"))

        mesh = replace(self, factory_constructed=factory_constructed, **kwargs)
        if __debug__ and not skip_tests:
            check_mesh_consistency(
                mesh,
                node_vertex_consistency_tolerance=node_vertex_consistency_tolerance,
                skip_element_orientation_test=skip_element_orientation_test)

        return mesh

    @property
    def ambient_dim(self) -> int:
        """Ambient dimension in which the mesh is embedded."""
        from pytools import single_valued
        return single_valued(grp.nodes.shape[0] for grp in self.groups)

    @property
    def dim(self) -> int:
        """Dimension of the elements in the mesh."""
        from pytools import single_valued
        return single_valued(grp.dim for grp in self.groups)

    @property
    def nvertices(self) -> int:
        """Number of vertices in the mesh, if available."""
        if self.vertices is None:
            from meshmode import DataUnavailableError
            raise DataUnavailableError("vertices")

        return self.vertices.shape[-1]

    @property
    def nelements(self) -> int:
        """Number of elements in the mesh (sum over all the :attr:`~Mesh.groups`)."""
        return sum(grp.nelements for grp in self.groups)

    @property
    @memoize_method
    def base_element_nrs(self) -> IndexArray:
        """An array of size ``(len(groups),)`` of starting element indices for
        each group in the mesh.
        """
        return np.cumsum([0] + [grp.nelements for grp in self.groups[:-1]])

    @property
    @memoize_method
    def base_node_nrs(self) -> IndexArray:
        """An array of size ``(len(groups),)`` of starting node indices for
        each group in the mesh.
        """
        return np.cumsum([0] + [grp.nnodes for grp in self.groups[:-1]])

    @property
    def vertex_dtype(self) -> np.dtype:
        """The :class:`~numpy.dtype` of the :attr:`~Mesh.vertices` array, if any."""
        if self.vertices is None:
            from meshmode import DataUnavailableError
            raise DataUnavailableError("vertices")

        return self.vertices.dtype

    @property
    def nodal_adjacency(self) -> NodalAdjacency:
        """Nodal adjacency of the mesh, if available.

        This property gets the :attr:`Mesh._nodal_adjacency` of the mesh. If the
        attribute value is *None*, the adjacency is computed and cached.

        :raises DataUnavailableError: if the nodal adjacency cannot be obtained.
        """
        from meshmode import DataUnavailableError

        nodal_adjacency = self._nodal_adjacency
        if nodal_adjacency is False:
            raise DataUnavailableError("Nodal adjacency is not available")

        if nodal_adjacency is None:
            if not self.is_conforming:
                raise DataUnavailableError(
                    "Nodal adjacency can only be computed for conforming meshes"
                    )

            nodal_adjacency = _compute_nodal_adjacency_from_vertices(self)
            object.__setattr__(self, "_nodal_adjacency", nodal_adjacency)

        return nodal_adjacency

    @property
    def facial_adjacency_groups(
            self) -> Sequence[Sequence[FacialAdjacencyGroup]]:
        r"""Facial adjacency of the mesh, if available.

        This function gets the :attr:`Mesh._facial_adjacency_groups` of the mesh.
        If the attribute value is *None*, the adjacency is computed and cached.

        Each ``facial_adjacency_groups[igrp]`` gives the facial adjacency
        relations for group *igrp*, expressed as a list of
        :class:`FacialAdjacencyGroup` instances.

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

        Note that element groups are not necessarily geometrically contiguous
        like the figure may suggest.

        :raises DataUnavailableError: if the facial adjacency cannot be obtained.
        """
        from meshmode import DataUnavailableError

        fagrps_in = self._facial_adjacency_groups
        if fagrps_in is False:
            raise DataUnavailableError("Facial adjacency is not available")

        elif fagrps_in is None:
            if not self.is_conforming:
                raise DataUnavailableError(
                    "Facial adjacency can only be computed for conforming meshes"
                    )

            facial_adjacency_groups = _compute_facial_adjacency_from_vertices(
                self.groups, self.element_id_dtype, self.face_id_dtype)
            object.__setattr__(self, "_facial_adjacency_groups",
                               facial_adjacency_groups)
        else:
            facial_adjacency_groups = fagrps_in

        return facial_adjacency_groups

    @override
    def __eq__(self, other: object) -> bool:
        """Compare two meshes for equality.

        .. warning::

            This operation is very expensive, as it compares all the vertices and
            groups between the two meshes. If available, the nodal and facial
            adjacency information is compared as well.

        .. warning::

            Only the (uncached) :attr:`~Mesh._nodal_adjacency` and
            :attr:`~Mesh._facial_adjacency_groups` are compared. This can fail
            for two meshes if one called :meth:`~Mesh.nodal_adjacency`
            and the other one did not, even if they would be equal.
        """
        if type(self) is not type(other):
            return False
        assert isinstance(other, Mesh)

        return (
                optional_array_equal(self.vertices, other.vertices)
                and self.groups == other.groups
                and self.vertex_id_dtype == other.vertex_id_dtype
                and self.element_id_dtype == other.element_id_dtype
                and self._nodal_adjacency == other._nodal_adjacency
                and self._facial_adjacency_groups == other._facial_adjacency_groups
                and self.is_conforming == other.is_conforming)

    # Design experience: Try not to add too many global data structures to the
    # mesh. Let the element groups be responsible for that at the mesh level.
    #
    # There are more big, global structures on the discretization level.

# }}}


# {{{ node-vertex consistency test

def _mesh_group_node_vertex_error(mesh: Mesh, mgrp: MeshElementGroup) -> VertexArray:
    if isinstance(mgrp, ModepyElementGroup):
        basis = mp.basis_for_space(mgrp.space, mgrp.shape).functions
    else:
        raise TypeError(f"unsupported group type: {type(mgrp).__name__}")

    resampling_mat = mp.resampling_matrix(
            basis,
            mgrp.vertex_unit_coordinates().T,
            mgrp.unit_nodes)

    # dim, nelments, nvertices
    map_vertices = np.einsum(
            "ij,dej->dei", resampling_mat, mgrp.nodes)

    assert mesh.vertices is not None
    grp_vertices = mesh.vertices[:, mgrp.vertex_indices]

    return map_vertices - grp_vertices


def _test_group_node_vertex_consistency_resampling(
        mesh: Mesh, igrp: int, *, tol: float | None = None) -> None:
    if mesh.vertices is None:
        return

    mgrp = mesh.groups[igrp]

    if mgrp.nelements == 0:
        return

    from meshmode import InconsistentVerticesError

    per_vertex_errors = _mesh_group_node_vertex_error(mesh, mgrp)
    per_element_vertex_errors = np.max(
        np.max(np.abs(per_vertex_errors), axis=-1), axis=0)

    if tol is None:
        tol = float(1e3 * np.finfo(per_element_vertex_errors.dtype).eps)

    grp_vertices = mesh.vertices[:, mgrp.vertex_indices]

    coord_scales = np.max(np.max(np.abs(grp_vertices), axis=-1), axis=0)

    per_element_tols = tol + tol * coord_scales

    elements_above_tol, = np.where(per_element_vertex_errors >= per_element_tols)
    if len(elements_above_tol) > 0:
        i_grp_elem = elements_above_tol[0]
        ielem = i_grp_elem + mesh.base_element_nrs[igrp]

        raise InconsistentVerticesError(
            f"Vertex consistency check failed for element {ielem}; "
            f"{per_element_vertex_errors[i_grp_elem]} >= "
            f"{per_element_tols[i_grp_elem]}")


def _test_node_vertex_consistency(
        mesh: Mesh, *, tol: float | None = None) -> None:
    """Ensure that order of by-index vertices matches that of mapped unit vertices.

    :raises InconsistentVerticesError: if the vertices are not consistent.
    """
    for igrp, mgrp in enumerate(mesh.groups):
        if isinstance(mgrp, ModepyElementGroup):
            _test_group_node_vertex_consistency_resampling(mesh, igrp, tol=tol)
        else:
            warn("Not implemented: node-vertex consistency check for "
                 f"groups of type '{type(mgrp).__name__}'.",
                 stacklevel=3)

# }}}


# {{{ vertex-based nodal adjacency

def _compute_nodal_adjacency_from_vertices(mesh: Mesh) -> NodalAdjacency:
    # FIXME Native code would make this faster

    if mesh.vertices is None:
        raise ValueError("unable to compute nodal adjacency without vertices")

    _, nvertices = mesh.vertices.shape
    vertex_to_element: list[list[int]] = [[] for _ in range(nvertices)]

    for base_element_nr, grp in zip(mesh.base_element_nrs, mesh.groups, strict=True):
        if grp.vertex_indices is None:
            raise ValueError("unable to compute nodal adjacency without vertices")

        for iel_grp in range(grp.nelements):
            for ivertex in grp.vertex_indices[iel_grp]:
                vertex_to_element[ivertex].append(base_element_nr + iel_grp)

    element_to_element: list[set[int]] = [set() for _ in range(mesh.nelements)]
    for base_element_nr, grp in zip(mesh.base_element_nrs, mesh.groups, strict=True):
        assert grp.vertex_indices is not None

        for iel_grp in range(grp.nelements):
            for ivertex in grp.vertex_indices[iel_grp]:
                element_to_element[base_element_nr + iel_grp].update(
                        vertex_to_element[ivertex])

    for iel, neighbors in enumerate(element_to_element):
        neighbors.remove(iel)

    lengths = [len(el_list) for el_list in element_to_element]
    neighbors_starts = np.cumsum(
            np.array([0, *lengths], dtype=mesh.element_id_dtype))
    from pytools import flatten
    neighbors_ary = np.array(
            list(flatten(element_to_element)),
            dtype=mesh.element_id_dtype)

    assert neighbors_starts[-1] == len(neighbors_ary)

    return NodalAdjacency(
            neighbors_starts=neighbors_starts,
            neighbors=neighbors_ary)

# }}}


# {{{ vertex-based facial adjacency

@dataclass(frozen=True)
class _FaceIDs:
    """
    Data structure for storage of a list of face identifiers (group, element, face).
    Each attribute is a :class:`numpy.ndarray` of shape ``(nfaces,)``.

    .. autoattribute:: groups
    .. autoattribute:: elements
    .. autoattribute:: faces
    """

    groups: IndexArray
    """The index of the group containing the face."""
    elements: IndexArray
    """The group-relative index of the element containing the face."""
    faces: IndexArray
    """The element-relative index of face."""


def _concatenate_face_ids(face_ids_list: Sequence[_FaceIDs]) -> _FaceIDs:
    return _FaceIDs(
        groups=np.concatenate([ids.groups for ids in face_ids_list]),
        elements=np.concatenate([ids.elements for ids in face_ids_list]),
        faces=np.concatenate([ids.faces for ids in face_ids_list]))


T = TypeVar("T")


def _assert_not_none(val: T | None) -> T:
    assert val is not None
    return val


def _match_faces_by_vertices(
            groups: Sequence[MeshElementGroup],
            face_ids: _FaceIDs,
            vertex_index_map_func: Callable[[IndexArray], IndexArray] | None = None
        ) -> IndexArray:
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
        def default_vertex_index_map_func(vertices: IndexArray) -> IndexArray:
            return vertices

        vertex_index_map_func = default_vertex_index_map_func

    from pytools import single_valued
    vertex_id_dtype = single_valued(
        _assert_not_none(grp.vertex_indices).dtype for grp in groups)

    nfaces = len(face_ids.groups)

    max_face_vertices = max(len(ref_fvi) for grp in groups
        for ref_fvi in grp.face_vertex_indices())

    face_vertex_indices = np.empty((max_face_vertices, nfaces),
        dtype=vertex_id_dtype)
    face_vertex_indices[:, :] = -1

    for igrp, grp in enumerate(groups):
        assert grp.vertex_indices is not None

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
    no_diff_flags = ~np.any(diffs, axis=0)
    assert isinstance(no_diff_flags, np.ndarray)
    match_indices, = no_diff_flags.nonzero()

    return np.stack((order[match_indices], order[match_indices+1]))


def _compute_facial_adjacency_from_vertices(
        groups: Sequence[MeshElementGroup],
        element_id_dtype: np.dtype[np.integer],
        face_id_dtype: np.dtype[np.integer],
        face_vertex_indices_to_tags: Mapping[
            frozenset[int], Sequence[BoundaryTag]] | None = None,
        ) -> Sequence[Sequence[FacialAdjacencyGroup]]:
    if not groups:
        return []

    if face_vertex_indices_to_tags is not None:
        boundary_tags: set[BoundaryTag] = {
            tag
            for tags in face_vertex_indices_to_tags.values()
            for tag in tags
            if tags is not None}
    else:
        boundary_tags = set()

    boundary_tag_to_index = {tag: i for i, tag in enumerate(boundary_tags)}

    # Match up adjacent faces according to their vertex indices

    face_ids_per_group: list[_FaceIDs] = []
    for igrp, grp in enumerate(groups):
        indices = np.indices((grp.nfaces, grp.nelements), dtype=element_id_dtype)
        face_ids_per_group.append(_FaceIDs(
            groups=np.full(grp.nelements * grp.nfaces, igrp),
            elements=indices[1].flatten(),
            faces=indices[0].flatten().astype(face_id_dtype)))

    face_ids = _concatenate_face_ids(face_ids_per_group)

    face_index_pairs = _match_faces_by_vertices(groups, face_ids)

    del igrp  # pyright: ignore[reportPossiblyUnboundVariable]
    del grp   # pyright: ignore[reportPossiblyUnboundVariable]

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

    facial_adjacency_groups: list[list[FacialAdjacencyGroup]] = []
    for igrp, grp in enumerate(groups):
        assert grp.vertex_indices is not None

        grp_list: list[FacialAdjacencyGroup] = []

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
            belongs_to_bdry = np.full(
                (len(boundary_tags), len(bdry_elements)), False)

            if face_vertex_indices_to_tags is not None:
                for i in range(len(bdry_elements)):
                    ref_fvi = grp.face_vertex_indices()[bdry_element_faces[i]]
                    fvi = frozenset(grp.vertex_indices[bdry_elements[i], ref_fvi])
                    tags = face_vertex_indices_to_tags.get(fvi, None)
                    if tags is not None:
                        for tag in tags:
                            btag_idx = boundary_tag_to_index[tag]
                            belongs_to_bdry[btag_idx, i] = True

            for btag_idx, btag in enumerate(boundary_tags):
                indices, = np.where(belongs_to_bdry[btag_idx, :])
                if len(indices) > 0:
                    elements = bdry_elements[indices]
                    element_faces = bdry_element_faces[indices]
                    grp_list.append(
                        BoundaryAdjacencyGroup(
                            igroup=igrp,
                            boundary_tag=btag,
                            elements=elements,
                            element_faces=element_faces))

            is_untagged = ~np.any(belongs_to_bdry, axis=0)
            if np.any(is_untagged):
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
            igrp: int,
            bdry_grps: Sequence[BoundaryAdjacencyGroup],
            merged_btag: BoundaryTag,
            element_id_dtype: np.dtype,
            face_id_dtype: np.dtype,
        ) -> BoundaryAdjacencyGroup:
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

    max_ielem: int = max(
        np.max(grp.elements, initial=0)
        for grp in bdry_grps)
    max_iface: int = max(
        np.max(grp.element_faces, initial=0)
        for grp in bdry_grps)

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
            facial_adjacency_groups: Sequence[Sequence[FacialAdjacencyGroup]],
            element_id_dtype: np.dtype,
            face_id_dtype: np.dtype
        ) -> tuple[tuple[FacialAdjacencyGroup, ...], ...]:
    """
    Add :class:`~meshmode.mesh.BoundaryAdjacencyGroup` instances for
    :class:`~meshmode.mesh.BTAG_NONE`, :class:`~meshmode.mesh.BTAG_ALL`, and
    :class:`~meshmode.mesh.BTAG_REALLY_ALL` to a facial adjacency group list if
    they are not present.
    """

    completed_facial_adjacency_groups = [
        list(fagrps) for fagrps in facial_adjacency_groups
    ]

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

    return tuple(
        tuple(fagrps) for fagrps in completed_facial_adjacency_groups
    )

# }}}


# {{{ as_python

def _boundary_tag_as_python(boundary_tag: BoundaryTag) -> str:
    if isinstance(boundary_tag, type):
        return boundary_tag.__name__
    elif isinstance(boundary_tag, str):
        return boundary_tag
    else:
        return boundary_tag.as_python()


def _numpy_array_as_python(
        array: np.ndarray[tuple[int, ...], np.dtype[Any]] | None) -> str:
    if array is not None:
        return "np.array({}, dtype=np.{})".format(
                repr(array.tolist()),
                array.dtype.name)
    else:
        return "None"


def _affine_map_as_python(aff_map: AffineMap) -> str:
    return ("AffineMap("
        + _numpy_array_as_python(aff_map.matrix) + ", "
        + _numpy_array_as_python(aff_map.offset) + ")")


def as_python(mesh: Mesh, function_name: str = "make_mesh") -> str:
    """
    :returns: a snippet of Python code (as a string) that will recreate the
        mesh given as an input parameter.
    """

    from pytools.py_codegen import Indentation, PythonCodeGenerator
    cg = PythonCodeGenerator()
    cg("""
        # generated by meshmode.mesh.as_python

        import numpy as np
        from meshmode.mesh import (
            make_mesh as mm_make_mesh,
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

    cg(f"def {function_name}():")
    with Indentation(cg):
        cg("vertices = " + _numpy_array_as_python(mesh.vertices))
        cg("")
        cg("groups = []")
        cg("")
        for group in mesh.groups:
            cg(f"import {type(group).__module__}")
            cg("groups.append({}.{}.make_group(".format(
                type(group).__module__,
                type(group).__name__))
            cg(f"    order={group.order},")
            cg(f"    vertex_indices={_numpy_array_as_python(group.vertex_indices)},")
            cg(f"    nodes={_numpy_array_as_python(group.nodes)},")
            cg(f"    unit_nodes={_numpy_array_as_python(group.unit_nodes)}))")

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
            cg(f"facial_adjacency_groups = {mesh._facial_adjacency_groups!r}")

        # }}}

        cg("return mm_make_mesh(vertices, groups, skip_tests=True,")
        cg(f"    vertex_id_dtype=np.{mesh.vertex_id_dtype.name},")
        cg(f"    element_id_dtype=np.{mesh.element_id_dtype.name},")

        if isinstance(mesh._nodal_adjacency, NodalAdjacency):
            el_con_str = "({}, {})".format(
                    _numpy_array_as_python(
                        mesh._nodal_adjacency.neighbors_starts),
                    _numpy_array_as_python(
                        mesh._nodal_adjacency.neighbors),
                    )
        else:
            el_con_str = repr(mesh._nodal_adjacency)

        cg(f"    nodal_adjacency={el_con_str},")
        cg("    facial_adjacency_groups=facial_adjacency_groups,")
        cg(f"    is_conforming={mesh.is_conforming!r})")

        # FIXME: Handle facial adjacency, boundary tags

    return cg.get()

# }}}


# {{{ is_true_boundary

def is_true_boundary(boundary_tag: BoundaryTag) -> bool:
    if boundary_tag == BTAG_REALLY_ALL:
        return False
    elif isinstance(boundary_tag, type):
        return not issubclass(boundary_tag, BTAG_NO_BOUNDARY)
    else:
        return not isinstance(boundary_tag, BTAG_NO_BOUNDARY)

# }}}


# {{{ mesh_has_boundary

def mesh_has_boundary(mesh: Mesh, boundary_tag: BoundaryTag) -> bool:
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

def check_bc_coverage(
            mesh: Mesh,
            boundary_tags: Collection[BoundaryTag],
            incomplete_ok: bool = False,
            true_boundary_only: bool = True) -> None:
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
            all_btag: BoundaryTag = BTAG_ALL
        else:
            all_btag = BTAG_REALLY_ALL

        all_bdry_grp, = (
            fagrp for fagrp in fagrp_list
            if isinstance(fagrp, BoundaryAdjacencyGroup)
            and fagrp.boundary_tag == all_btag)

        matching_bdry_grps = [
            fagrp for fagrp in fagrp_list
            if isinstance(fagrp, BoundaryAdjacencyGroup)
            and fagrp.boundary_tag in boundary_tags]

        def get_bdry_counts(
                bdry_grp: BoundaryAdjacencyGroup
            ) -> np.ndarray[tuple[int, int], np.dtype[np.integer[Any]]]:
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

def is_boundary_tag_empty(mesh: Mesh, boundary_tag: BoundaryTag) -> bool:
    """Return *True* if the corresponding boundary tag does not occur as part of
    *mesh*.
    """
    if not mesh_has_boundary(mesh, boundary_tag):
        raise ValueError(f"invalid boundary tag {boundary_tag}.")

    for igrp in range(len(mesh.groups)):
        nfaces = sum(
            len(grp.elements) for grp in mesh.facial_adjacency_groups[igrp]
            if isinstance(grp, BoundaryAdjacencyGroup)
            and grp.boundary_tag == boundary_tag)
        if nfaces > 0:
            return False

    return True

# }}}


# {{{ is_affine_simplex_group

def is_affine_simplex_group(
            group: MeshElementGroup,
            abs_tol: float | None = None
        ) -> bool:
    if abs_tol is None:
        abs_tol = 1.0e-13

    if not isinstance(group, SimplexElementGroup):
        raise TypeError(f"expected a 'SimplexElementGroup': {type(group)}")

    if group.nelements == 0:
        # All zero of them are affine! :)
        return True

    # get matrices
    basis = mp.basis_for_space(group.space, group.shape)
    vinv = la.inv(mp.vandermonde(basis.functions, group.unit_nodes))
    diff = mp.differentiation_matrices(
            basis.functions, basis.gradients, group.unit_nodes)
    if not isinstance(diff, tuple):
        diff = (diff,)

    # construct all second derivative matrices (including cross terms)
    from itertools import product

    mats: list[np.ndarray[tuple[int, int], np.dtype[np.floating[Any]]]] = []
    for n in product(range(group.dim), repeat=2):
        if n[0] > n[1]:
            continue
        mats.append(vinv.dot(diff[n[0]].dot(diff[n[1]])))

    # check just the first element for a non-affine local-to-global mapping
    ddx_coeffs = np.einsum("aij,bj->abi", mats, group.nodes[:, 0, :])
    norm_inf: np.floating = np.max(np.abs(ddx_coeffs))
    if norm_inf > abs_tol:
        return False

    # check all elements for a non-affine local-to-global mapping
    ddx_coeffs = np.einsum("aij,bcj->abci", mats, group.nodes)
    norm_inf = np.max(np.abs(ddx_coeffs))
    return bool(norm_inf < abs_tol)

# }}}


__getattr__ = partial(module_getattr_for_deprecations, __name__, {
        "_ModepyElementGroup": ("ModepyElementGroup", ModepyElementGroup, 2026),
        })


# vim: foldmethod=marker
