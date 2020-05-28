import numpy as np
import numpy.linalg as la
import six

from meshmode.interop import ExternalImporter
from meshmode.interop.fiat import FIATSimplexCellImporter


__doc__ = """
.. autoclass:: FinatLagrangeElementImporter
    :members:
"""


class FinatLagrangeElementImporter(ExternalImporter):
    """
    An importer for a FInAT element, usually instantiated from
    ``some_instantiated_firedrake_function_space.finat_element``

    This importer does not have an obvious meshmode counterpart,
    so is instead used for its method.
    In particular, methods :meth:`import_data`
    and :meth:`validate_to_data` are *NOT* implemented
    and there is no :attr:`to_data` to be computed.
    """
    def __init__(self, finat_element):
        """
        :param finat_element: A FInAT element of type
            :class:`finat.fiat_elements.Lagrange` or
            :class:`finat.fiat_elements.DiscontinuousLagrange`
            which uses affine mapping of the basis functions
            (i.e. ``finat_element.mapping`` must be
            ``"affine"``)

        :raises TypeError: If :param:`finat_element` is not of type
            :class:`finat.fiat_elements.Lagrange` or
            :class:`finat.fiat_elements.DiscontinuousLagrange`

        :raises ValueError: If :param:`finat_element` does not
            use affine mappings of the basis functions
        """
        # {{{ Check and store input

        # Check types
        from finat.fiat_elements import DiscontinuousLagrange, Lagrange
        if not isinstance(finat_element, (Lagrange, DiscontinuousLagrange)):
            raise TypeError(":param:`finat_element` must be of type"
                            " `finat.fiat_elements.Lagrange` or"
                            " `finat.fiat_elements.DiscontinuousLagrange`",
                            " not type `%s`" % type(finat_element))

        if finat_element.mapping != 'affine':
            raise ValueError("FInAT element must use affine mappings"
                             " of the bases")
        # }}}

        super(FinatLagrangeElementImporter, self).__init__(finat_element)

        self.cell_importer = FIATSimplexCellImporter(finat_element.cell)

        # computed and stored once :meth:`unit_nodes`, :meth:`unit_vertices`,
        # and :meth:`flip_matrix` are called
        self._unit_nodes = None
        self._unit_vertex_indices = None
        self._flip_matrix = None

    def _compute_unit_vertex_indices_and_nodes(self):
        """
        Compute the unit nodes, as well as the unit vertex indices,
        if they have not already been computed.
        """
        if self._unit_nodes is None or self._unit_vertex_indices is None:
            # {{{ Compute unit nodes
            node_nr_to_coords = {}
            unit_vertex_indices = []

            # Get unit nodes
            for dim, element_nrs in six.iteritems(
                    self.analog().entity_support_dofs()):
                for element_nr, node_list in six.iteritems(element_nrs):
                    # Get the nodes on the element (in meshmode reference coords)
                    pts_on_element = self.cell_importer.make_points(
                        dim, element_nr, self.analog().degree)
                    # Record any new nodes
                    i = 0
                    for node_nr in node_list:
                        if node_nr not in node_nr_to_coords:
                            node_nr_to_coords[node_nr] = pts_on_element[i]
                            i += 1
                            # If is a vertex, store the index
                            if dim == 0:
                                unit_vertex_indices.append(node_nr)

            # store vertex indices
            self._unit_vertex_indices = np.array(sorted(unit_vertex_indices))

            # Convert unit_nodes to array, then change to (dim, nunit_nodes)
            # from (nunit_nodes, dim)
            unit_nodes = np.array([node_nr_to_coords[i] for i in
                                   range(len(node_nr_to_coords))])
            self._unit_nodes = unit_nodes.T.copy()

            # }}}

    def dim(self):
        """
        :return: The dimension of the FInAT element's cell
        """
        return self.cell_importer.from_data.get_dimension()

    def unit_vertex_indices(self):
        """
        :return: An array of shape *(dim+1,)* of indices
                 so that *self.unit_nodes()[self.unit_vertex_indices()]*
                 are the vertices of the reference element.
        """
        self._compute_unit_vertex_indices_and_nodes()
        return self._unit_vertex_indices

    def unit_nodes(self):
        """
        :return: The unit nodes used by the FInAT element mapped
                 onto the appropriate :mod:`modepy` `reference
                 element <https://documen.tician.de/modepy/nodes.html>`_
                 as an array of shape *(dim, nunit_nodes)*.
        """
        self._compute_unit_vertex_indices_and_nodes()
        return self._unit_nodes

    def nunit_nodes(self):
        """
        :return: The number of unit nodes.
        """
        return self.unit_nodes().shape[1]

    def flip_matrix(self):
        """
        :return: The matrix which should be applied to the
                 *(dim, nunitnodes)*-shaped array of nodes corresponding to
                 an element in order to change orientation - <-> +.

                 The matrix will be *(dim, dim)* and orthogonal with
                 *np.float64* type entries.
        """
        if self._flip_matrix is None:
            # This is very similar to :mod:`meshmode` in processing.py
            # the function :function:`from_simplex_element_group`, but
            # we needed to use firedrake nodes

            from modepy.tools import barycentric_to_unit, unit_to_barycentric

            # Generate a resampling matrix that corresponds to the
            # first two barycentric coordinates being swapped.

            bary_unit_nodes = unit_to_barycentric(self.unit_nodes())

            flipped_bary_unit_nodes = bary_unit_nodes.copy()
            flipped_bary_unit_nodes[0, :] = bary_unit_nodes[1, :]
            flipped_bary_unit_nodes[1, :] = bary_unit_nodes[0, :]
            flipped_unit_nodes = barycentric_to_unit(flipped_bary_unit_nodes)

            from modepy import resampling_matrix, simplex_best_available_basis

            flip_matrix = resampling_matrix(
                simplex_best_available_basis(self.dim(), self.analog().degree),
                flipped_unit_nodes, self.unit_nodes())

            flip_matrix[np.abs(flip_matrix) < 1e-15] = 0

            # Flipping twice should be the identity
            assert la.norm(
                np.dot(flip_matrix, flip_matrix)
                - np.eye(len(flip_matrix))) < 1e-13

            self._flip_matrix = flip_matrix

        return self._flip_matrix

    def make_resampling_matrix(self, element_grp):
        """
            :param element_grp: A
                :class:`meshmode.discretization.InterpolatoryElementGroupBase` whose
                basis functions span the same space as the FInAT element.
            :return: A matrix which resamples a function sampled at
                     the firedrake unit nodes to a function sampled at
                     *element_grp.unit_nodes()* (by matrix multiplication)
        """
        from meshmode.discretization import InterpolatoryElementGroupBase
        assert isinstance(element_grp, InterpolatoryElementGroupBase), \
            "element group must be an interpolatory element group so that" \
            " can redistribute onto its nodes"

        from modepy import resampling_matrix
        return resampling_matrix(element_grp.basis(),
                                 new_nodes=element_grp.unit_nodes,
                                 old_nodes=self.unit_nodes())
