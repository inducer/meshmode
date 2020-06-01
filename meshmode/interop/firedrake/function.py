import numpy as np

from firedrake import Function

from meshmode.interop import ExternalImportHandler
from meshmode.interop.firedrake.function_space import FiredrakeWithGeometryImporter
from meshmode.interop.firedrake.function_space_coordless import \
    FiredrakeCoordinatelessFunctionImporter


class FiredrakeFunctionImporter(ExternalImportHandler):
    """
    An import handler for :mod:`firedrake` functions.

    .. attribute:: data

        A :mod:`firedrake` function
    """
    def __init__(self, function, function_space_importer):
        """
        :param function: A firedrake :class:`Function` or a
                       :class:`FiredrakeCoordinatelessFunctionImporter`

        :param function_space_importer: An instance of
                       :class:`FiredrakeWithGeometryImporter`
                       compatible with the function
                       space this function is built on
        """
        # {{{ Check types
        if not isinstance(function_space_importer,
                          FiredrakeWithGeometryImporter):
            raise TypeError(":param:`function_space_importer` must be of "
                            "type FiredrakeWithGeometryImporter")

        if not isinstance(function, (Function,
                                     FiredrakeCoordinatelessFunctionImporter)):
            raise TypeError(":param:`function` must be one of "
                            "(:class:`firedrake.function.Function`,"
                            " FiredrakeCoordinatelessFunctionImporter)")
        # }}}

        super(FiredrakeFunctionImporter, self).__init__(function)

        self._function_space_importer = function_space_importer
        if isinstance(function, Function):
            self._topological_importer = \
                FiredrakeCoordinatelessFunctionImporter(function,
                                                        function_space_importer)
        else:
            self._topological_importer = function

    def function_space_importer(self):
        """
        :returns: the :class:`FiredrakeWithGeometryImporter` instance
            used for the function space this object's :attr:`data` lives on
        """
        return self._function_space_importer

    @property
    def topological_importer(self):
        """
        :returns: The topological version of this object, i.e. the underlying
            coordinateless importer (an instance of
            :class:`FiredrakeCoordinatelessFunctionImporter`)
        """
        return self._topological_importer

    def as_field(self):
        """
        :returns: A numpy array holding the data of this function as a
              field on the corresponding meshmode mesh
        """
        return self.function_space_importer().convert_function(self)

    def set_from_field(self, field):
        """
        Set function :attr:`data`'s data
        from a field on the corresponding meshmode mesh.

        :param field: A numpy array representing a field on the corresponding
                      meshmode mesh
        """
        # Handle 1-D case
        if len(self.data.dat.data.shape) == 1 and len(field.shape) > 1:
            field = field.reshape(field.shape[1])

        # resample from nodes
        group = self.function_space_importer().discretization().groups[0]
        resampled = np.copy(field)
        resampled_view = group.view(resampled)
        resampling_mat = self.function_space_importer().resampling_mat(False)
        np.matmul(resampled_view, resampling_mat.T, out=resampled_view)

        # reorder data
        self.data.dat.data[:] = self.function_space_importer().reorder_nodes(
            resampled, firedrake_to_meshmode=False)[:]
