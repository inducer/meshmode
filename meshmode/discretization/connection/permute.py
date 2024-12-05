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

from arraycontext.metadata import NameHint

from meshmode.discretization.connection.direct import (
    DirectDiscretizationConnection,
    DiscretizationConnectionElementGroup,
    IdentityDiscretizationConnection,
    InterpolationBatch,
)
from meshmode.transform_metadata import (
    DiscretizationElementAxisTag,
)


# {{{ permute mesh elements

def make_element_permutation_connection(actx, from_discr):
    """Build discretization connection containing a permuted mesh.

    .. note::

        This function assumes a flattened DOF array, as produced by
        :class:`~arraycontext.flatten`.

    :arg actx: an :class:`~arraycontext.ArrayContext`.
    :arg from_discr: a :class:`Discretization`.
    """

    # Reorder the local mesh using Metis
    from meshmode.distributed import get_reordering_by_pymetis
    perm, iperm = get_reordering_by_pymetis(from_discr.mesh)

    # Create target discretization
    to_discr = from_discr.copy()

    groups = []
    for igrp, (fgrp, tgrp) in enumerate(
        zip(from_discr.groups, to_discr.groups, strict=True)):
        from arraycontext.metadata import NameHint
        all_elements = actx.tag(NameHint(f"all_el_ind_grp{igrp}"),
                                actx.tag_axis(0,
                                              DiscretizationElementAxisTag(),
                                              actx.from_numpy(
                                                  np.arange(
                                                      fgrp.nelements,
                                                      dtype=np.intp))))
        all_elements = actx.freeze(all_elements)
        # Permuted elements
        all_elements_perm = actx.tag(NameHint(f"all_el_ind_perm_grp{igrp}"),
                                     actx.tag_axis(0,
                                                   DiscretizationElementAxisTag(),
                                                   actx.from_numpy(perm)))
        all_elements_perm = actx.freeze(all_elements_perm)
        ibatch = InterpolationBatch(
                from_group_index=igrp,
                from_element_indices=all_elements,
                to_element_indices=all_elements_perm,  ## CHK??
                result_unit_nodes=tgrp.unit_nodes,
                to_element_face=None)

        groups.append(
                DiscretizationConnectionElementGroup([ibatch]))

    return DirectDiscretizationConnection(
        from_discr,
        to_discr,
        groups,
        is_surjective=True)

# }}}

# vim: foldmethod=marker
