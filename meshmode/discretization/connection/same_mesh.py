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


# {{{ same-mesh constructor

def make_same_mesh_connection(actx, to_discr, from_discr):
    from meshmode.discretization.connection.direct import (
            InterpolationBatch,
            DiscretizationConnectionElementGroup,
            DirectDiscretizationConnection)

    if from_discr.mesh is not to_discr.mesh:
        raise ValueError("from_discr and to_discr must be based on "
                "the same mesh")

    groups = []
    for igrp, (fgrp, tgrp) in enumerate(zip(from_discr.groups, to_discr.groups)):
        all_elements = actx.freeze(
                actx.from_numpy(
                    np.arange(
                        fgrp.nelements,
                        dtype=np.intp)))
        ibatch = InterpolationBatch(
                from_group_index=igrp,
                from_element_indices=all_elements,
                to_element_indices=all_elements,
                result_unit_nodes=tgrp.unit_nodes,
                to_element_face=None)

        groups.append(
                DiscretizationConnectionElementGroup([ibatch]))

    return DirectDiscretizationConnection(
            from_discr, to_discr, groups,
            is_surjective=True)

# }}}

# vim: foldmethod=marker
