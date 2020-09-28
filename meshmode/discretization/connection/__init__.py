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

from meshmode.discretization.connection.direct import (
        InterpolationBatch,
        DiscretizationConnectionElementGroup,
        DiscretizationConnection,
        DirectDiscretizationConnection)
from meshmode.discretization.connection.chained import \
        ChainedDiscretizationConnection
from meshmode.discretization.connection.projection import \
        L2ProjectionInverseDiscretizationConnection

from meshmode.array_context import ArrayContext
from meshmode.discretization.connection.same_mesh import \
        make_same_mesh_connection
from meshmode.discretization.connection.face import (
        FACE_RESTR_INTERIOR, FACE_RESTR_ALL,
        make_face_restriction,
        make_face_to_all_faces_embedding)
from meshmode.discretization.connection.opposite_face import \
        make_opposite_face_connection, make_partition_connection
from meshmode.discretization.connection.refinement import \
        make_refinement_connection
from meshmode.discretization.connection.chained import \
        flatten_chained_connection

import logging
logger = logging.getLogger(__name__)


__all__ = [
        "DiscretizationConnection",
        "DirectDiscretizationConnection",
        "ChainedDiscretizationConnection",
        "L2ProjectionInverseDiscretizationConnection",

        "make_same_mesh_connection",
        "FACE_RESTR_INTERIOR", "FACE_RESTR_ALL",
        "make_face_restriction",
        "make_face_to_all_faces_embedding",
        "make_opposite_face_connection",
        "make_partition_connection",
        "make_refinement_connection",
        "flatten_chained_connection",

        "InterpolationBatch",
        "DiscretizationConnectionElementGroup",
        ]

__doc__ = """
Base classes
------------
.. autoclass:: DiscretizationConnection
.. autoclass:: ChainedDiscretizationConnection
.. autoclass:: L2ProjectionInverseDiscretizationConnection
.. autoclass:: DirectDiscretizationConnection


Same-mesh connections
---------------------
.. autofunction:: make_same_mesh_connection

Restriction to faces
--------------------
.. autodata:: FACE_RESTR_INTERIOR
.. autodata:: FACE_RESTR_ALL

.. autofunction:: make_face_restriction
.. autofunction:: make_face_to_all_faces_embedding

.. autofunction:: make_opposite_face_connection

Refinement
----------
.. autofunction:: make_refinement_connection

Flattening a :class:`ChainedDiscretizationConnection`
-----------------------------------------------------
.. autofunction:: flatten_chained_connection

Implementation details
----------------------

.. autoclass:: InterpolationBatch

.. autoclass:: DiscretizationConnectionElementGroup
"""


# {{{ check connection

def check_connection(actx: ArrayContext, connection: DirectDiscretizationConnection):
    from_discr = connection.from_discr
    to_discr = connection.to_discr

    assert len(connection.groups) == len(to_discr.groups)

    for cgrp, tgrp in zip(connection.groups, to_discr.groups):
        for batch in cgrp.batches:
            fgrp = from_discr.groups[batch.from_group_index]

            from_element_indices = actx.to_numpy(
                    actx.thaw(batch.from_element_indices))
            to_element_indices = actx.to_numpy(actx.thaw(batch.to_element_indices))

            assert (0 <= from_element_indices).all()
            assert (0 <= to_element_indices).all()
            assert (from_element_indices < fgrp.nelements).all()
            assert (to_element_indices < tgrp.nelements).all()
            if batch.to_element_face is not None:
                assert 0 <= batch.to_element_face < fgrp.mesh_el_group.nfaces

# }}}

# vim: foldmethod=marker
