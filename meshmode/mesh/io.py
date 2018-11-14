from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2014 Andreas Kloeckner"

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

import six
from six.moves import range, zip
import numpy as np

from gmsh_interop.reader import (  # noqa
        GmshMeshReceiverBase, ScriptSource, FileSource, LiteralSource,
        ScriptWithFilesSource,
        GmshSimplexElementBase,
        GmshTensorProductElementBase)


__doc__ = """

.. autoclass:: ScriptSource
.. autoclass:: FileSource
.. autoclass:: ScriptWithFilesSource

.. autofunction:: read_gmsh
.. autofunction:: generate_gmsh
.. autofunction:: from_meshpy
.. autofunction:: from_vertices_and_simplices
.. autofunction:: to_json

"""


# {{{ gmsh receiver

class GmshMeshReceiver(GmshMeshReceiverBase):
    def __init__(self, mesh_construction_kwargs=None):
        # Use data fields similar to meshpy.triangle.MeshInfo and
        # meshpy.tet.MeshInfo
        self.points = None
        self.elements = None
        self.element_types = None
        self.element_markers = None
        self.tags = []
        self.tag_to_my_index = {}  # map name of tag to index in attr:: self.tags
        self.gmsh_tag_index_to_mine = {}
        # list of sets, each node maps to a set of elements it is a member of
        self.node_to_containing_elements = None

        if mesh_construction_kwargs is None:
            mesh_construction_kwargs = {}

        self.mesh_construction_kwargs = mesh_construction_kwargs

    def set_up_nodes(self, count):
        # Preallocate array of nodes within list; treat None as sentinel value.
        # Preallocation not done for performance, but to assign values at indices
        # in random order.
        self.points = [None] * count
        self.node_to_containing_elements = [None] * count

    def add_node(self, node_nr, point):
        self.points[node_nr] = point
        self.node_to_containing_elements[node_nr] = set()

    def finalize_nodes(self):
        self.points = np.array(self.points, dtype=np.float64)

    def set_up_elements(self, count):
        # Preallocation of arrays for assignment elements in random order.
        self.element_vertices = [None] * count
        self.element_nodes = [None] * count
        self.element_types = [None] * count
        self.element_markers = [None] * count

    def add_element(self, element_nr, element_type, vertex_nrs,
            lexicographic_nodes, tag_numbers):
        self.element_vertices[element_nr] = vertex_nrs
        self.element_nodes[element_nr] = lexicographic_nodes
        self.element_types[element_nr] = element_type

        # only physical tags are supported
        if tag_numbers and len(tag_numbers) > 1:
            tag_numbers = [tag_numbers[0]]
        self.element_markers[element_nr] = tag_numbers

        # record this element in node to element map
        for node in lexicographic_nodes:
            self.node_to_containing_elements[node].add(element_nr)

    def finalize_elements(self):
        pass

    # May raise ValueError if try to add different tags with the same name
    def add_tag(self, name, index, dimension):
        # add tag if new
        if name not in self.tag_to_my_index:
            self.tag_to_my_index[name] = len(self.tags)
            self.gmsh_tag_index_to_mine[index] = len(self.tags)
            self.tags.append((name, index, dimension))
        else:
            # ensure not trying to add different tags with same name
            ndx = self.tag_to_my_index[name]
            _, prev_index, prev_dim = self.tags[ndx]
            if index != prev_index or dimension != prev_dim:
                raise ValueError("Distinct tags with the same name")

    def finalize_tags(self):
        pass

    def get_mesh(self):
        el_type_hist = {}
        for el_type in self.element_types:
            el_type_hist[el_type] = el_type_hist.get(el_type, 0) + 1

        if not el_type_hist:
            raise RuntimeError("empty mesh in gmsh input")

        groups = self.groups = []

        ambient_dim = self.points.shape[-1]

        mesh_bulk_dim = max(
                el_type.dimensions for el_type in six.iterkeys(el_type_hist))

        # {{{ build vertex numbering

        vertex_gmsh_index_to_mine = {}
        for el_vertices, el_type in zip(
                self.element_vertices, self.element_types):
            for gmsh_vertex_nr in el_vertices:
                if gmsh_vertex_nr not in vertex_gmsh_index_to_mine:
                    vertex_gmsh_index_to_mine[gmsh_vertex_nr] = \
                            len(vertex_gmsh_index_to_mine)

        # }}}

        # {{{ build vertex array

        gmsh_vertex_indices, my_vertex_indices = \
                list(zip(*six.iteritems(vertex_gmsh_index_to_mine)))
        vertices = np.empty(
                (ambient_dim, len(vertex_gmsh_index_to_mine)), dtype=np.float64)
        vertices[:, np.array(my_vertex_indices, np.intp)] = \
                self.points[np.array(gmsh_vertex_indices, np.intp)].T

        # }}}

        from meshmode.mesh import (Mesh,
                SimplexElementGroup, TensorProductElementGroup)

        bulk_el_types = set()

        def get_higher_dim_element(element_ndx):
            """Returns a set of the indices of elements with dimension
            mesh_bulk_dim which contain all the nodes of elt.

            :arg element_ndx: The index of an element
            """
            # Take the intersection of the sets of elements which
            # contain at least one of the nodes used by the
            # element at element_ndx
            higher_dim_elts = None
            for node in self.element_nodes[element_ndx]:
                if higher_dim_elts is None:
                    higher_dim_elts = set(self.node_to_containing_elements[node])
                else:
                    higher_dim_elts &= self.node_to_containing_elements[node]

            # only keep elements of dimension mesh_bulk_dim
            higher_dim_elts = {e for e in higher_dim_elts
                               if self.element_types[e].dimensions
                               == mesh_bulk_dim}

            # if no higher dimension elements, return empty set
            if higher_dim_elts is None:
                higher_dim_elts = set()

            return higher_dim_elts

        # populate tags from elements of small dimension to elements of
        # full dimension (mesh_bulk_dim)
        if self.tags:
            for elt_ndx in range(len(self.element_types)):
                if self.element_types[elt_ndx].dimensions == mesh_bulk_dim:
                    continue
                # if this element has no tags, continue
                if not self.element_markers[elt_ndx]:
                    continue

                higher_dim_elements = get_higher_dim_element(elt_ndx)
                for higher_dim_elt in higher_dim_elements:
                    for tag in self.element_markers[elt_ndx]:
                        if tag not in self.element_markers[higher_dim_elt]:
                            self.element_markers[higher_dim_elt].append(tag)

        # prepare bdy tags for Mesh
        bdy_tags = None
        if self.tags:
            bdy_tags = [t[0] for t in self.tags if t[-1] == mesh_bulk_dim - 1]

        # for each group, a list of non-empty boundary tags
        element_bdy_markers = None
        if self.tags:
            element_bdy_markers = []

        element_index_mine_to_gmsh = {}
        for group_el_type, ngroup_elements in six.iteritems(el_type_hist):
            if group_el_type.dimensions != mesh_bulk_dim:
                continue
            if self.tags:
                boundary_markers = []  # list of nonempty boundary tags in
                # preserving relative order of index of corresponding faces

            bulk_el_types.add(group_el_type)

            nodes = np.empty((ambient_dim, ngroup_elements, el_type.node_count()),
                    np.float64)
            el_vertex_count = group_el_type.vertex_count()
            vertex_indices = np.empty(
                    (ngroup_elements, el_vertex_count),
                    np.int32)
            i = 0

            for element, (el_vertices, el_nodes, el_type) in enumerate(zip(
                    self.element_vertices, self.element_nodes, self.element_types)):
                if el_type is not group_el_type:
                    continue

                nodes[:, i] = self.points[el_nodes].T
                vertex_indices[i] = [vertex_gmsh_index_to_mine[v_nr]
                        for v_nr in el_vertices]
                n_elements = len(element_index_mine_to_gmsh)
                element_index_mine_to_gmsh[n_elements] = element
                # record the tags associated to this element if any
                if self.tags and self.element_markers[element]:
                    element_tags = []
                    for t in self.element_markers[element]:
                        tag = self.tags[self.gmsh_tag_index_to_mine[t]]
                        if tag[-1] != mesh_bulk_dim - 1:
                            continue
                        element_tags.append(tag[0])
                    if element_tags:
                        boundary_markers.append(element_tags)

                i += 1

            if element_bdy_markers is not None:
                element_bdy_markers.append(boundary_markers)
            unit_nodes = (np.array(group_el_type.lexicographic_node_tuples(),
                    dtype=np.float64).T/group_el_type.order)*2 - 1

            if isinstance(group_el_type, GmshSimplexElementBase):
                group = SimplexElementGroup(
                    group_el_type.order,
                    vertex_indices,
                    nodes,
                    unit_nodes=unit_nodes
                    )

                if group.dim == 2:
                    from meshmode.mesh.processing import flip_simplex_element_group
                    group = flip_simplex_element_group(vertices, group,
                            np.ones(ngroup_elements, np.bool))

            elif isinstance(group_el_type, GmshTensorProductElementBase):
                gmsh_vertex_tuples = type(group_el_type)(order=1).gmsh_node_tuples()
                gmsh_vertex_tuples_loc_dict = dict(
                        (gvt, i)
                        for i, gvt in enumerate(gmsh_vertex_tuples))

                from pytools import (
                        generate_nonnegative_integer_tuples_below as gnitb)
                vertex_shuffle = np.array([
                    gmsh_vertex_tuples_loc_dict[vt]
                    for vt in gnitb(2, group_el_type.dimensions)])

                group = TensorProductElementGroup(
                    group_el_type.order,
                    vertex_indices[:, vertex_shuffle],
                    nodes,
                    unit_nodes=unit_nodes
                    )
            else:
                raise NotImplementedError("gmsh element type: %s"
                        % type(group_el_type).__name__)

            groups.append(group)

        # FIXME: This is heuristic.
        if len(bulk_el_types) == 1:
            is_conforming = True
        else:
            is_conforming = mesh_bulk_dim < 3

        return Mesh(
                vertices, groups,
                is_conforming=is_conforming,
                boundary_tags=bdy_tags,
                element_boundary_tags=element_bdy_markers,
                **self.mesh_construction_kwargs)

# }}}


# {{{ gmsh

def read_gmsh(filename, force_ambient_dim=None, mesh_construction_kwargs=None):
    """Read a gmsh mesh file from *filename* and return a
    :class:`meshmode.mesh.Mesh`.

    :arg force_ambient_dim: if not None, truncate point coordinates to
        this many dimensions.
    :arg mesh_construction_kwargs: *None* or a dictionary of keyword
        arguments passed to the :class:`meshmode.mesh.Mesh` constructor.
    """
    from gmsh_interop.reader import read_gmsh
    recv = GmshMeshReceiver(mesh_construction_kwargs=mesh_construction_kwargs)
    read_gmsh(recv, filename, force_dimension=force_ambient_dim)

    return recv.get_mesh()


def generate_gmsh(source, dimensions=None, order=None, other_options=[],
        extension="geo", gmsh_executable="gmsh", force_ambient_dim=None,
        output_file_name="output.msh", mesh_construction_kwargs=None):
    """Run :command:`gmsh` on the input given by *source*, and return a
    :class:`meshmode.mesh.Mesh` based on the result.

    :arg source: an instance of either :class:`FileSource` or
        :class:`LiteralSource`
    :arg force_ambient_dim: if not None, truncate point coordinates to
        this many dimensions.
    :arg mesh_construction_kwargs: *None* or a dictionary of keyword
        arguments passed to the :class:`meshmode.mesh.Mesh` constructor.
    """
    recv = GmshMeshReceiver(mesh_construction_kwargs=mesh_construction_kwargs)

    from gmsh_interop.runner import GmshRunner
    from gmsh_interop.reader import parse_gmsh
    with GmshRunner(source, dimensions, order=order,
            other_options=other_options, extension=extension,
            gmsh_executable=gmsh_executable) as runner:
        parse_gmsh(recv, runner.output_file,
                force_dimension=force_ambient_dim)

    mesh = recv.get_mesh()

    if force_ambient_dim is None:
        AXIS_NAMES = "xyz"  # noqa

        dim = mesh.vertices.shape[0]
        for idim in range(dim):
            if (mesh.vertices[idim] == 0).all():
                from warnings import warn
                warn("all vertices' %s coordinate is zero--perhaps you want to pass "
                        "force_ambient_dim=%d (pass any fixed value to "
                        "force_ambient_dim to silence this warning)" % (
                            AXIS_NAMES[idim], idim))
                break

    return mesh

# }}}


# {{{ meshpy

def from_meshpy(mesh_info, order=1):
    """Imports a mesh from a :mod:`meshpy` *mesh_info* data structure,
    which may be generated by either :mod:`meshpy.triangle` or
    :mod:`meshpy_tet`.
    """
    from meshmode.mesh import Mesh
    from meshmode.mesh.generation import make_group_from_vertices

    vertices = np.array(mesh_info.points).T
    elements = np.array(mesh_info.elements, np.int32)

    grp = make_group_from_vertices(vertices, elements, order)

    # FIXME: Should transfer boundary/volume markers

    return Mesh(
            vertices=vertices, groups=[grp],
            is_conforming=True)

# }}}


# {{{ from_vertices_and_simplices

def from_vertices_and_simplices(vertices, simplices, order=1, fix_orientation=False):
    """Imports a mesh from a numpy array of vertices and an array
    of simplices.

    :arg vertices:
        An array of vertex coordinates with shape
        *(ambient_dim, nvertices)*
    :arg simplices:
        An array *(nelements, nvertices)* of (mesh-wide)
        vertex indices.
    """
    from meshmode.mesh import Mesh
    from meshmode.mesh.generation import make_group_from_vertices

    grp = make_group_from_vertices(vertices, simplices, order)

    if fix_orientation:
        from meshmode.mesh.processing import (
                find_volume_mesh_element_group_orientation,
                flip_simplex_element_group)
        orient = find_volume_mesh_element_group_orientation(vertices, grp)
        grp = flip_simplex_element_group(vertices, grp, orient < 0)

    return Mesh(
            vertices=vertices, groups=[grp],
            is_conforming=True)

# }}}


# {{{ to_json

def to_json(mesh):
    """Return a JSON-able Python data structure for *mesh*. The structure directly
    reflects the :class:`Mesh` data structure."""

    def btag_to_json(btag):
        if isinstance(btag, str):
            return btag
        else:
            return btag.__name__

    def group_to_json(group):
        return {
            "type": type(group).__name__,
            "order": group.order,
            "vertex_indices": group.vertex_indices.tolist(),
            "nodes": group.nodes.tolist(),
            "unit_nodes": group.unit_nodes.tolist(),
            "element_nr_base": group.element_nr_base,
            "node_nr_base": group.node_nr_base,
            "dim": group.dim,
            }

    from meshmode import DataUnavailable

    def nodal_adjacency_to_json(mesh):
        try:
            na = mesh.nodal_adjacency
        except DataUnavailable:
            return None

        return {
            "neighbors_starts": na.neighbors_starts.tolist(),
            "neighbors": na.neighbors.tolist(),
            }

    return {
        # VERSION 0:
        # - initial version
        #
        # VERSION 1:
        # - added is_conforming

        "version": 1,
        "vertices": mesh.vertices.tolist(),
        "groups": [group_to_json(group) for group in mesh.groups],
        "nodal_adjacency": nodal_adjacency_to_json(mesh),
        # not yet implemented
        "facial_adjacency_groups": None,
        "boundary_tags": [btag_to_json(btag) for btag in mesh.boundary_tags],
        "btag_to_index": dict(
            (btag_to_json(btag), value)
            for btag, value in six.iteritems(mesh.btag_to_index)),
        "is_conforming": mesh.is_conforming,
        }

# }}}


# vim: foldmethod=marker
