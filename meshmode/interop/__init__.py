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

from abc import ABC

__doc__ = """
Development Interface
---------------------
.. autoclass:: ExternalTransporter
.. autoclass:: ExternalImporter
.. autoclass:: ExternalExporter
"""


# {{{ Generic, most abstract class for transporting meshmode <-> external

class ExternalTransporter(ABC):
    """
    .. attribute:: from_data

        The object which needs to be transported either to meshmode or
        from meshmode

    .. attribute:: to_data

        The "transported" object, i.e. the

        This attribute does not exist at instantiation time.
        If exporting (resp. importing) from meshmode
        then we are using an :class:`ExternalExporter`
        (resp. :class:`ExternalImporter`) instance. :attr:`to_data` is
        computed with a call to :fun:`ExternalExporter.export_data`
        (resp. :fun:`ExternalImporter.import_data`).

        :raises ValueError: if :attr:`to_data` is accessed before creation.
        :raises NotImplementedError: if :meth:`validate_to_data` is called
                                     without an implementation.
    """
    def __init__(self, from_data):
        self.from_data = from_data

    def validate_to_data(self):
        """
        Validate :attr:`to_data`

        :return: *True* if :attr:`to_data` has been computed and is valid
                 and *False* otherwise
        """
        raise NotImplementedError("*validate_to_data* method not implemented "
                                  "for object of type %s" % type(self))

    def __hash__(self):
        return hash((type(self), self.from_data))

    def __eq__(self, other):
        return isinstance(other, type(self)) and \
               isinstance(self, type(other)) and \
               self.from_data == other.from_data

    def __neq__(self, other):
        return not self.__eq__(other)

    def __getattr__(self, attr):
        if attr != 'to_data':
            return super(ExternalTransporter, self).__getattr__(attr)
        raise ValueError("Attribute *to_data* has not yet been computed. "
                         "An object of class *ExternalExporter* (resp. "
                         "*ExternalImporter*) must call *export_data()* "
                         "(resp. *import_data()*) to compute attribute *to_data*")

# }}}


# {{{ Define specific classes for meshmode -> external and meshmode <- external

class ExternalExporter(ExternalTransporter):
    """
    Subclass of :class:`ExternalTransporter` for meshmode -> external
    data transfer
    """
    def export_data(self):
        """
        Compute :attr:`to_data` from :attr:`from_data`
        """
        raise NotImplementedError("*export_data* method not implemented "
                                  "for type %s" % type(self))


class ExternalImporter(ExternalTransporter):
    """
    Subclass of :class:`ExternalTransporter` for external -> meshmode
    data transfer
    """
    def import_data(self):
        """
        Compute :attr:`to_data` from :attr:`from_data`
        """
        raise NotImplementedError("*import_data* method not implemented "
                                  "for type %s" % type(self))

# }}}


# vim: fdm=marker
