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
        self.element_vertices = None
        self.element_nodes = None
        self.element_types = None
        self.element_markers = None
        self.tags = None
        self.groups = None
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

    def get_mesh(self, return_tag_to_elements_map=False):
        el_type_hist = {}
        for el_type in self.element_types:
            el_type_hist[el_type] = el_type_hist.get(el_type, 0) + 1

        if not el_type_hist:
            raise RuntimeError("empty mesh in gmsh input")

        groups = self.groups = []
        ambient_dim = self.points.shape[-1]

        mesh_bulk_dim = max(el_type.dimensions for el_type in el_type_hist)

        # {{{ build vertex numbering

        vertex_gmsh_index_to_mine = {}
        for el_vertices in self.element_vertices:
            for gmsh_vertex_nr in el_vertices:
                if gmsh_vertex_nr not in vertex_gmsh_index_to_mine:
                    vertex_gmsh_index_to_mine[gmsh_vertex_nr] = \
                            len(vertex_gmsh_index_to_mine)

        # }}}

        # {{{ build vertex array

        gmsh_vertex_indices, my_vertex_indices = \
                list(zip(*vertex_gmsh_index_to_mine.items()))
        vertices = np.empty(
                (ambient_dim, len(vertex_gmsh_index_to_mine)), dtype=np.float64)
        vertices[:, np.array(my_vertex_indices, np.intp)] = \
                self.points[np.array(gmsh_vertex_indices, np.intp)].T

        # }}}

        # {{{ build map from tags to arrays of face vertex indices

        tag_to_fvis = {}

        if self.tags:
            for element, el_vertices in enumerate(self.element_vertices):
                el_markers = self.element_markers[element]
                el_tag_indexes = (
                    [self.gmsh_tag_index_to_mine[t] for t in el_markers]
                    if el_markers is not None else [])
                # record tags of boundary dimension
                el_tags = [self.tags[i][0] for i in el_tag_indexes if
                           self.tags[i][1] == mesh_bulk_dim - 1]
                el_grp_verts = [vertex_gmsh_index_to_mine[e] for e in el_vertices]
                for tag in el_tags:
                    fvi_list = tag_to_fvis.setdefault(tag, [])
                    fvi_list.append(el_grp_verts)

            for tag in tag_to_fvis.keys():
                fvi_list = tag_to_fvis[tag]
                max_face_vertices = max(len(fvi) for fvi in fvi_list)
                fvis = np.full((len(fvi_list), max_face_vertices), -1)
                for i, fvi in enumerate(fvi_list):
                    fvis[i, :len(fvi)] = fvi
                tag_to_fvis[tag] = fvis

        # }}}

        from meshmode.mesh import (Mesh,
                SimplexElementGroup, TensorProductElementGroup)

        bulk_el_types = set()

        group_base_elem_nr = 0

        tag_to_meshwide_elements = {}
        tag_to_faces = []

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

            for el_vertices, el_nodes, el_type, el_markers in zip(
                    self.element_vertices, self.element_nodes, self.element_types,
                    self.element_markers):
                if el_type is not group_el_type:
                    continue

                nodes[:, i] = self.points[el_nodes].T
                vertex_indices[i] = [
                        vertex_gmsh_index_to_mine[v_nr] for v_nr in el_vertices
                        ]

                if el_markers is not None:
                    for t in el_markers:
                        tag = self.tags[self.gmsh_tag_index_to_mine[t]][0]
                        tagged_elements = tag_to_meshwide_elements.setdefault(
                            tag, [])
                        tagged_elements.append(group_base_elem_nr + i)

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
                group = SimplexElementGroup.make_group(
                    group_el_type.order,
                    vertex_indices,
                    nodes,
                    unit_nodes=unit_nodes
                    )

                if group.dim == 2:
                    from meshmode.mesh.processing import flip_simplex_element_group
                    group = flip_simplex_element_group(vertices, group,
                            np.ones(ngroup_elements, bool))

            elif isinstance(group_el_type, GmshTensorProductElementBase):
                vertex_shuffle = type(group_el_type)(
                        order=1).get_lexicographic_gmsh_node_indices()

                group = TensorProductElementGroup.make_group(
                    group_el_type.order,
                    vertex_indices[:, vertex_shuffle],
                    nodes,
                    unit_nodes=unit_nodes
                    )
            else:
                # NOTE: already checked above
                raise AssertionError()

            groups.append(group)

            group_base_elem_nr += group.nelements

            group_tag_to_faces = {}

            for tag, tagged_fvis in tag_to_fvis.items():
                for fid, ref_fvi in enumerate(group.face_vertex_indices()):
                    fvis = group.vertex_indices[:, ref_fvi]
                    # Combine known tagged fvis and current group fvis into a single
                    # array, lexicographically sort them, and find identical
                    # neighboring entries to get tags
                    if tagged_fvis.shape[1] < fvis.shape[1]:
                        continue
                    padded_fvis = np.full((fvis.shape[0], tagged_fvis.shape[1]), -1)
                    padded_fvis[:, :fvis.shape[1]] = fvis
                    extended_fvis = np.concatenate((tagged_fvis, padded_fvis))
                    # Make sure vertices are in the same order
                    extended_fvis = np.sort(extended_fvis, axis=1)
                    # Lexicographically sort the face vertex indices, then diff the
                    # result to find tagged faces with the same vertices
                    order = np.lexsort(extended_fvis.T)
                    diffs = np.diff(extended_fvis[order, :], axis=0)
                    match_indices, = (~np.any(diffs, axis=1)).nonzero()
                    # lexsort is stable, so the second entry in each match
                    # corresponds to the group face
                    face_element_indices = (
                        order[match_indices+1]
                        - len(tagged_fvis))
                    if len(face_element_indices) > 0:
                        elements_and_faces = np.stack(
                            (
                                face_element_indices,
                                np.full(face_element_indices.shape, fid)),
                            axis=-1)
                        if tag in group_tag_to_faces:
                            group_tag_to_faces[tag] = np.concatenate((
                                group_tag_to_faces[tag],
                                elements_and_faces))
                        else:
                            group_tag_to_faces[tag] = elements_and_faces

            tag_to_faces.append(group_tag_to_faces)

        for tag in tag_to_meshwide_elements.keys():
            tag_to_meshwide_elements[tag] = np.array(
                tag_to_meshwide_elements[tag], dtype=np.int32)

        # FIXME: This is heuristic.
        if len(bulk_el_types) == 1:
            is_conforming = True
        else:
            is_conforming = mesh_bulk_dim < 3

        # compute facial adjacency for Mesh if there is tag information
        facial_adjacency_groups = None
        if is_conforming and self.tags:
            from meshmode.mesh import _compute_facial_adjacency_from_vertices
            facial_adjacency_groups = _compute_facial_adjacency_from_vertices(
                    groups, np.int32, np.int8, tag_to_faces)

        mesh = Mesh(
                vertices, groups,
                is_conforming=is_conforming,
                facial_adjacency_groups=facial_adjacency_groups,
                **self.mesh_construction_kwargs)

        return (
            (mesh, tag_to_meshwide_elements)
            if return_tag_to_elements_map
            else mesh)

# }}}


# {{{ gmsh

def read_gmsh(
        filename, force_ambient_dim=None, mesh_construction_kwargs=None,
        return_tag_to_elements_map=False):
    """Read a gmsh mesh file from *filename* and return a
    :class:`meshmode.mesh.Mesh`.

    :arg force_ambient_dim: if not None, truncate point coordinates to
        this many dimensions.
    :arg mesh_construction_kwargs: *None* or a dictionary of keyword
        arguments passed to the :class:`meshmode.mesh.Mesh` constructor.
    :arg return_tag_to_elements_map: If *True*, return in addition to the mesh
        a :class:`dict` that maps each volume tag in the gmsh file to a
        :class:`numpy.ndarray` containing meshwide indices of the elements that
        belong to that volume.
    """
    from gmsh_interop.reader import read_gmsh
    recv = GmshMeshReceiver(mesh_construction_kwargs=mesh_construction_kwargs)
    read_gmsh(recv, filename, force_dimension=force_ambient_dim)

    return recv.get_mesh(return_tag_to_elements_map=return_tag_to_elements_map)


def generate_gmsh(source, dimensions=None, order=None, other_options=None,
        extension="geo", gmsh_executable="gmsh", force_ambient_dim=None,
        output_file_name="output.msh", mesh_construction_kwargs=None,
        target_unit=None, return_tag_to_elements_map=False):
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
    if other_options is None:
        other_options = []

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

    result = recv.get_mesh(return_tag_to_elements_map=return_tag_to_elements_map)

    if force_ambient_dim is None:
        if return_tag_to_elements_map:
            mesh = result[0]
        else:
            mesh = result

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

    return result

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

    def group_to_json(group):
        return {
            "type": type(group).__name__,
            "order": group.order,
            "vertex_indices": group.vertex_indices.tolist(),
            "nodes": group.nodes.tolist(),
            "unit_nodes": group.unit_nodes.tolist(),
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
        "is_conforming": mesh.is_conforming,
        }

# }}}


# vim: foldmethod=marker
