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
        self.tags = None
        self.gmsh_tag_index_to_mine = None

        if mesh_construction_kwargs is None:
            mesh_construction_kwargs = {}

        self.mesh_construction_kwargs = mesh_construction_kwargs

    def set_up_nodes(self, count):
        # Preallocate array of nodes within list; treat None as sentinel value.
        # Preallocation not done for performance, but to assign values at indices
        # in random order.
        self.points = [None] * count

    def add_node(self, node_nr, point):
        self.points[node_nr] = point

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
        if tag_numbers:
            # only physical tags are supported
            physical_tag = tag_numbers[0]
            self.element_markers[element_nr] = [physical_tag]

    def finalize_elements(self):
        pass

    # May raise ValueError if try to add different tags with the same name
    def add_tag(self, name, index, dimension):
        if self.tags is None:
            self.tags = []
        if self.gmsh_tag_index_to_mine is None:
            self.gmsh_tag_index_to_mine = {}
        # add tag if new
        if index not in self.gmsh_tag_index_to_mine:
            self.gmsh_tag_index_to_mine[index] = len(self.tags)
            self.tags.append((name, dimension))
        else:
            # ensure not trying to add different tags with same index
            my_index = self.gmsh_tag_index_to_mine[index]
            recorded_name, recorded_dim = self.tags[my_index]
            if recorded_name != name or recorded_dim != dimension:
                raise ValueError("Distinct tags with the same tag id")

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
                el_type.dimensions for el_type in el_type_hist.keys())

        # {{{ build vertex numbering

        # map set of face vertex indices to list of tags associated to face
        face_vertex_indices_to_tags = {}
        vertex_gmsh_index_to_mine = {}
        for element, (el_vertices, el_type) in enumerate(zip(
                self.element_vertices, self.element_types)):
            for gmsh_vertex_nr in el_vertices:
                if gmsh_vertex_nr not in vertex_gmsh_index_to_mine:
                    vertex_gmsh_index_to_mine[gmsh_vertex_nr] = \
                            len(vertex_gmsh_index_to_mine)
            if self.tags:
                el_tag_indexes = [self.gmsh_tag_index_to_mine[t] for t in
                                  self.element_markers[element]]
                # record tags of boundary dimension
                el_tags = [self.tags[i][0] for i in el_tag_indexes if
                           self.tags[i][1] == mesh_bulk_dim - 1]
                el_grp_verts = {vertex_gmsh_index_to_mine[e] for e in el_vertices}
                face_vertex_indices = frozenset(el_grp_verts)
                if face_vertex_indices not in face_vertex_indices_to_tags:
                    face_vertex_indices_to_tags[face_vertex_indices] = []
                face_vertex_indices_to_tags[face_vertex_indices] += el_tags

        # }}}

        # {{{ build vertex array

        gmsh_vertex_indices, my_vertex_indices = \
                list(zip(*vertex_gmsh_index_to_mine.items()))
        vertices = np.empty(
                (ambient_dim, len(vertex_gmsh_index_to_mine)), dtype=np.float64)
        vertices[:, np.array(my_vertex_indices, np.intp)] = \
                self.points[np.array(gmsh_vertex_indices, np.intp)].T

        # }}}

        from meshmode.mesh import (Mesh,
                SimplexElementGroup, TensorProductElementGroup)

        bulk_el_types = set()

        for group_el_type, ngroup_elements in el_type_hist.items():
            if group_el_type.dimensions != mesh_bulk_dim:
                continue

            bulk_el_types.add(group_el_type)

            nodes = np.empty(
                    (ambient_dim, ngroup_elements, group_el_type.node_count()),
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
                vertex_indices[i] = [
                        vertex_gmsh_index_to_mine[v_nr] for v_nr in el_vertices
                        ]

                i += 1

            import modepy as mp
            if isinstance(group_el_type, GmshSimplexElementBase):
                shape = mp.Simplex(group_el_type.dimensions)
            elif isinstance(group_el_type, GmshTensorProductElementBase):
                shape = mp.Hypercube(group_el_type.dimensions)
            else:
                raise NotImplementedError(
                        f"gmsh element type: {type(group_el_type).__name__}")

            space = mp.space_for_shape(shape, group_el_type.order)
            unit_nodes = mp.equispaced_nodes_for_space(space, shape)

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
                vertex_shuffle = type(group_el_type)(
                        order=1).get_lexicographic_gmsh_node_indices()

                group = TensorProductElementGroup(
                    group_el_type.order,
                    vertex_indices[:, vertex_shuffle],
                    nodes,
                    unit_nodes=unit_nodes
                    )
            else:
                # NOTE: already checked above
                assert False

            groups.append(group)

        # FIXME: This is heuristic.
        if len(bulk_el_types) == 1:
            is_conforming = True
        else:
            is_conforming = mesh_bulk_dim < 3

        # construct boundary tags for mesh
        from meshmode.mesh import BTAG_ALL, BTAG_REALLY_ALL
        boundary_tags = [BTAG_ALL, BTAG_REALLY_ALL]
        if self.tags:
            boundary_tags += [tag for tag, dim in self.tags if
                              dim == mesh_bulk_dim-1]

        # compute facial adjacency for Mesh if there is tag information
        facial_adjacency_groups = None
        if is_conforming and self.tags:
            from meshmode.mesh import _compute_facial_adjacency_from_vertices
            facial_adjacency_groups = _compute_facial_adjacency_from_vertices(
                    groups, boundary_tags, np.int32, np.int8,
                    face_vertex_indices_to_tags)

        return Mesh(
                vertices, groups,
                is_conforming=is_conforming,
                facial_adjacency_groups=facial_adjacency_groups,
                boundary_tags=boundary_tags,
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
        output_file_name="output.msh", mesh_construction_kwargs=None,
        target_unit=None):
    """Run :command:`gmsh` on the input given by *source*, and return a
    :class:`meshmode.mesh.Mesh` based on the result.

    :arg source: an instance of either :class:`gmsh_interop.reader.FileSource` or
        :class:`gmsh_interop.reader.ScriptSource`
    :arg force_ambient_dim: if not *None*, truncate point coordinates to
        this many dimensions.
    :arg mesh_construction_kwargs: *None* or a dictionary of keyword
        arguments passed to the :class:`meshmode.mesh.Mesh` constructor.
    :arg target_unit: Value of the option *Geometry.OCCTargetUnit*.
        Supported values are the strings `'M'` or `'MM'`.
    """
    recv = GmshMeshReceiver(mesh_construction_kwargs=mesh_construction_kwargs)

    from gmsh_interop.runner import GmshRunner
    from gmsh_interop.reader import parse_gmsh

    if target_unit is None:
        target_unit = "MM"
        from warnings import warn
        warn(
                "Not specifying target_unit is deprecated. Set target_unit='MM' "
                "to retain prior behavior.", DeprecationWarning, stacklevel=2)

    with GmshRunner(source, dimensions, order=order,
            other_options=other_options, extension=extension,
            gmsh_executable=gmsh_executable,
            target_unit=target_unit) as runner:
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
    :mod:`meshpy.tet`.
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
        if grp.dim != vertices.shape[0]:
            raise ValueError("can only fix orientation of volume meshes")

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
    reflects the :class:`meshmode.mesh.Mesh` data structure."""

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
        "btag_to_index": {
            btag_to_json(btag): value

            for btag, value in mesh.btag_to_index.items()},
        "is_conforming": mesh.is_conforming,
        }

# }}}


# vim: foldmethod=marker
