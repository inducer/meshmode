from __future__ import annotations


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

import logging
from typing import TYPE_CHECKING

from meshmode.discretization.connection.chained import (
    ChainedDiscretizationConnection,
    flatten_chained_connection,
)
from meshmode.discretization.connection.direct import (
    DirectDiscretizationConnection,
    DiscretizationConnection,
    DiscretizationConnectionElementGroup,
    IdentityDiscretizationConnection,
    InterpolationBatch,
)
from meshmode.discretization.connection.face import (
    FACE_RESTR_ALL,
    FACE_RESTR_INTERIOR,
    make_face_restriction,
    make_face_to_all_faces_embedding,
)
from meshmode.discretization.connection.modal import (
    ModalToNodalDiscretizationConnection,
    NodalToModalDiscretizationConnection,
)
from meshmode.discretization.connection.opposite_face import (
    make_opposite_face_connection,
    make_partition_connection,
)
from meshmode.discretization.connection.projection import (
    L2ProjectionInverseDiscretizationConnection,
)
from meshmode.discretization.connection.refinement import make_refinement_connection
from meshmode.discretization.connection.same_mesh import make_same_mesh_connection


if TYPE_CHECKING:
    from arraycontext import ArrayContext


logger = logging.getLogger(__name__)


__all__ = [
    "FACE_RESTR_ALL",
    "FACE_RESTR_INTERIOR",
    "ChainedDiscretizationConnection",
    "DirectDiscretizationConnection",
    "DiscretizationConnection",
    "DiscretizationConnectionElementGroup",
    "IdentityDiscretizationConnection",
    "InterpolationBatch",
    "L2ProjectionInverseDiscretizationConnection",
    "ModalToNodalDiscretizationConnection",
    "NodalToModalDiscretizationConnection",
    "flatten_chained_connection",
    "make_face_restriction",
    "make_face_to_all_faces_embedding",
    "make_opposite_face_connection",
    "make_partition_connection",
    "make_refinement_connection",
    "make_same_mesh_connection",
]

__doc__ = """
Base classes
------------
.. autoclass:: DiscretizationConnection
.. autoclass:: IdentityDiscretizationConnection
.. autoclass:: ChainedDiscretizationConnection
.. autoclass:: L2ProjectionInverseDiscretizationConnection
.. autoclass:: DirectDiscretizationConnection

Mapping between modal and nodal representations
-----------------------------------------------

.. autoclass:: NodalToModalDiscretizationConnection
.. autoclass:: ModalToNodalDiscretizationConnection

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

References
----------

.. currentmodule:: arraycontext.typing

.. class:: ArrayT

    See :class:`arraycontext.ArrayT`.

.. class:: ArrayOrContainerT

    See :class:`arraycontext.ArrayOrContainerT`.

.. class:: ArrayOrContainerOrScalar

    See :attr:`arraycontext.ArrayOrContainerOrScalar`.

.. class:: ArrayOrContainerOrScalarT

    See :class:`arraycontext.ArrayOrContainerOrScalarT`.
"""


# {{{ check connection

def check_connection(actx: ArrayContext, connection: DirectDiscretizationConnection):
    from_discr = connection.from_discr
    to_discr = connection.to_discr

    assert len(connection.groups) == len(to_discr.groups)

    for cgrp, tgrp in zip(connection.groups, to_discr.groups, strict=True):
        for batch in cgrp.batches:
            fgrp = from_discr.groups[batch.from_group_index]

            from_element_indices = actx.to_numpy(
                    actx.thaw(batch.from_element_indices))
            to_element_indices = actx.to_numpy(actx.thaw(batch.to_element_indices))

            assert (from_element_indices >= 0).all()
            assert (to_element_indices >= 0).all()
            assert (from_element_indices < fgrp.nelements).all()
            assert (to_element_indices < tgrp.nelements).all()
            if batch.to_element_face is not None:
                assert 0 <= batch.to_element_face < fgrp.mesh_el_group.nfaces

# }}}

# vim: foldmethod=marker
