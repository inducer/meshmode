__copyright__ = "Copyright (C) 2020 Benjamin Sepanski"

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

from warnings import warn  # noqa

from meshmode.interop import ExternalImportHandler
from meshmode.interop.firedrake.mesh_topology import \
    FiredrakeMeshTopologyImporter
from meshmode.interop.FInAT import FinatLagrangeElementImporter


# {{{ Function space for coordinateless functions to live on


class FiredrakeFunctionSpaceImporter(ExternalImportHandler):
    """
    This is a counterpart of :class:`firedrake.functionspaceimpl.FunctionSpace`,

    This is not what usually results from a call to
    :func:`firedrake.functionspace.FunctionSpace`.
    When someone calls :func:`firedrake.functionspace.FunctionSpace`
    from the user side, they are usually talking about a
    :class:`firedrake.functionspaceimpl.WithGeometry`.

    Generally, this class is here to match Firedrake's design
    principles, i.e. so that we have something to put CoordinatelessFunction
    import handlers on.

    Just like we have "topological" and "geometric"
    meshes, one can think of think of this as a "topological" function space
    whose counterpart is a "WithGeometry"

    .. attribute:: data

        An instance of :class:`firedrake.functionspaceimpl.FunctionSpace`, i.e.
        a function space built to hold coordinateless function (functions
        whose domain is just a set of points with some topology, but no
        coordinates)

    .. attribute:: finat_element_importer

        An instance of
        :class:`meshmode.interop.FInAT.FinatLagrangeElementImporter` which
        is an importer for the *finat_element* of :attr:`data`.
    """
    def __init__(self, function_space, mesh_importer, finat_element_importer):
        """
        :param function_space: A :mod:`firedrake`
            :class:`firedrake.functionspaceimpl.FunctionSpace` or
            :class:`firedrake.functionspaceimpl.WithGeometry`. In the
            latter case, the underlying ``FunctionSpace`` is extracted
            from the ``WithGeometry``.
        :param mesh_importer: An instance
            of :class:`FiredrakeMeshTopology` created from the topological
            mesh of :param:`function_space`
        :param finat_element_importer: An instance
            of :class:`FinatLagrangeElementIMporter` created from the
            finat_element of :param:`function_space`

        :raises TypeError: If any of the arguments are the wrong type
        :raises ValueError: If :param:`mesh_importer` or
                           :param:`finat_element_importer` are importing
                           a different mesh or finat element than the provided
                           :param:`function_space` is built on.
        """
        # {{{ Some type-checking
        from firedrake.functionspaceimpl import FunctionSpace, WithGeometry

        if not isinstance(function_space, (FunctionSpace, WithGeometry)):
            raise TypeError(":param:`function_space` must be of type "
                            "``firedrake.functionspaceimpl.FunctionSpace`` "
                            "or ``firedrake.functionspaceimpl.WithGeometry`` ",
                            "not %s" % type(function_space))

        if not isinstance(mesh_importer, FiredrakeMeshTopologyImporter):
            raise TypeError(":param:`mesh_importer` must be either *None* "
                            "or of type :class:`meshmode.interop.firedrake."
                            "FiredrakeMeshTopologyImporter`")
        if not function_space.mesh() == mesh_importer.data:
            raise ValueError(":param:`mesh_importer`'s *data* attribute "
                             "must be the same mesh as returned by "
                             ":param:`function_space`'s *mesh()* method.")

        if not isinstance(finat_element_importer, FinatLagrangeElementImporter):
            raise TypeError(":param:`finat_element_importer` must be either "
                            "*None* "
                            "or of type :class:`meshmode.interop.FInAT."
                            "FinatLagragneElementImporter`")
        if not function_space.finat_element == finat_element_importer.data:
            raise ValueError(":param:`finat_element_importer`'s *data* "
                             "attribute "
                             "must be the same finat element as "
                             ":param:`function_space`'s *finat_element*"
                             " attribute.")

        # }}}

        # We want to ignore any geometry and then finish initialization
        function_space = function_space.topological
        super(FiredrakeFunctionSpaceImporter, self).__init__(function_space)

        self._mesh_importer = mesh_importer
        self.finat_element_importer = finat_element_importer

    @property
    def topological_importer(self):
        """
        A reference to self for compatability with 'geometrical' function spaces
        """
        return self

    def mesh_importer(self):
        """
        Return this object's mesh importer
        """
        return self._mesh_importer

# }}}


# {{{ Container to hold the coordinateless functions themselves


class FiredrakeCoordinatelessFunctionImporter(ExternalImportHandler):
    """
    A coordinateless function, i.e. a function defined on a set of
    points which only have an associated topology, no associated
    geometry.

    .. attribute:: data

        An instance of :mod:`firedrake` class
        :class:`firedrake.function.CoordinatelessFunction`.
        Note that a coordinateless function object in firedrake
        references concrete, but mutable data.
    """
    def __init__(self, function, function_space_importer):
        """
        :param function: The instance of
                :class:`firedrake.function.CoordinatelessFunction`
                which this object is importing. Becomes the
                :attr:`data` attribute.

        :param function_space_importer: An instance of
                :class:`FiredrakeFunctionSpaceImporter`
                which is importing the topological function space that
                :param:`function` is built on.

        :raises TypeError: If either parameter is the wrong type
        :raises ValueError: If :param:`function_space_importer` is an
                importer for a firedrake function space which is not
                identical to ``function.topological.function_space()``
        """
        # {{{ Some type-checking

        from firedrake.function import Function, CoordinatelessFunction
        if not isinstance(function, (CoordinatelessFunction, Function)):
            raise TypeError(":param:`function` must be of type "
                            "`firedrake.function.CoordinatelessFunction` "
                            " or `firedrdake.function.Function`")

        if not isinstance(function_space_importer,
                          FiredrakeFunctionSpaceImporter):
            raise TypeError(":param:`function_space_importer` must be of type "
                            "`meshmode.interop.firedrake."
                            "FiredrakeFunctionSpaceImporter`.")

        if not function_space_importer.data == function.function_space():
            raise ValueError(":param:`function_space_importer`'s *data* "
                             "attribute and ``function.function_space()`` "
                             "must be identical.")

        function = function.topological
        function_space_importer = function_space_importer.topological_importer

        super(FiredrakeCoordinatelessFunctionImporter, self).__init__(function)

        self._function_space_importer = function_space_importer

    def function_space_importer(self):
        """
        Return the
        :class:`meshmode.interop.firedrake.FiredrakeFunctionSpaceImporter`
        instance being used to import the function space
        of the underlying firedrake function this object represents.
        """
        return self._function_space_importer

    @property
    def topological_importer(self):
        """
        A reference to self for compatability with functions that have
        coordinates.
        """
        return self


# }}}
