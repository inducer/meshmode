from __future__ import division, print_function, absolute_import

__copyright__ = """
Copyright (C) 2014 Andreas Kloeckner
Copyright (C) 2018 Alexandru Fikl
"""

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


# {{{ chained connection

class ChainedDiscretizationConnection(DiscretizationConnection):
    """Aggregates multiple :class:`DiscretizationConnection` instances
    into a single one.

    .. attribute:: connections
    """

    def __init__(self, connections, from_discr=None):
        if connections:
            if from_discr is not None:
                assert from_discr is connections[0].from_discr
            else:
                from_discr = connections[0].from_discr
            is_surjective = all(cnx.is_surjective for cnx in connections)
            to_discr = connections[-1].to_discr
        else:
            if from_discr is None:
                raise ValueError("connections may not be empty if from_discr "
                        "is not specified")

            to_discr = from_discr

            # It's an identity
            is_surjective = True

        super(ChainedDiscretizationConnection, self).__init__(
                from_discr, to_discr, is_surjective=is_surjective)

        self.connections = connections

    def __call__(self, queue, vec):
        for cnx in self.connections:
            vec = cnx(queue, vec)

        return vec

# }}}


# {{{ flatten_chained_connection

def _flatten_two_direct(first_c, second_c):
    batches = []


def flatten_chained_connection(conn):
    if not isinstance(conn, ChainedDiscretizationConnection):
        raise TypeError("conn must be a ChainedDiscretizationConnection")

    connections = conn.connections[:]

    # TODO: Insert any nested chained connections directly into connections

    from meshm
    for cn in connections:
        if not isinstance(conn, ChainedDiscretizationConnection):
            raise TypeError("conn must be a ChainedDiscretizationConnection")





    

# }}}

# vim: foldmethod=marker
