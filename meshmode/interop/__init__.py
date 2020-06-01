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

from abc import ABC

__doc__ = """
Development Interface
---------------------
.. autoclass:: ExternalDataHandler
.. autoclass:: ExternalExportHandler
.. autoclass:: ExternalImportHandler
"""


# {{{ Generic, most abstract class for transporting meshmode <-> external

class ExternalDataHandler(ABC):
    """
    A data handler takes data from meshmode and facilitates its use
    in another package or the reverse: takes data from another package
    and facilitates its use in meshmode.

    .. attribute:: data

        The object which needs to be interfaced either into meshmode or
        out of meshmode.
        Should not be modified after creation.
    """
    def __init__(self, data):
        self.data = data

    def __hash__(self):
        return hash((type(self), self.data))

    def __eq__(self, other):
        return isinstance(other, type(self)) and \
               isinstance(self, type(other)) and \
               self.data == other.data

    def __neq__(self, other):
        return not self.__eq__(other)

# }}}


# {{{ Define specific classes for meshmode -> external and meshmode <- external

class ExternalExportHandler(ExternalDataHandler):
    """
    Subclass of :class:`ExternalDataHandler` for meshmode -> external
    data transfer
    """
    pass


class ExternalImportHandler(ExternalDataHandler):
    """
    Subclass of :class:`ExternalDataHandler` for external -> meshmode
    data transfer
    """
    pass

# }}}


# vim: fdm=marker
