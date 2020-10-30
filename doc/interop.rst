Interoperability with Other Discretization Packages
===================================================

Functionality in this subpackage helps import and export data to/from other
pieces of software, typically PDE solvers.

Nodal DG
--------

.. automodule:: meshmode.interop.nodal_dg

Firedrake
---------

Function Spaces/Discretizations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Users wishing to interact with :mod:`meshmode` from :mod:`firedrake`
will create a
:class:`~meshmode.interop.firedrake.connection.FiredrakeConnection`
using :func:`~meshmode.interop.firedrake.connection.build_connection_from_firedrake`,
while users wishing
to interact with :mod:`firedrake` from :mod:`meshmode` will use
will create a
:class:`~meshmode.interop.firedrake.connection.FiredrakeConnection`
using :func:`~meshmode.interop.firedrake.connection.build_connection_to_firedrake`.
It is not recommended to create a
:class:`~meshmode.interop.firedrake.connection.FiredrakeConnection` directly.

.. automodule:: meshmode.interop.firedrake.connection

Meshes
^^^^^^

.. automodule:: meshmode.interop.firedrake.mesh

Reference Cells
^^^^^^^^^^^^^^^

.. automodule:: meshmode.interop.firedrake.reference_cell


Implementation Details
^^^^^^^^^^^^^^^^^^^^^^

Converting between :mod:`firedrake` and :mod:`meshmode` is in general
straightforward. Some language is different:

* In a mesh, a :mod:`meshmode` "element" is a :mod:`firedrake` "cell"
* A :class:`~meshmode.discretization.Discretization` is a :mod:`firedrake`
  :class:`~firedrake.functionspaceimpl.WithGeometry`, usually
  created by calling the function :func:`~firedrake.functionspace.FunctionSpace`
  and referred to as a "function space"
* In a mesh, any vertices, faces, cells, etc. are :mod:`firedrake`
  "entities" (see `the PETSc documentation on dmplex <https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/DMPLEX/index.html>`__
  for more info on how topological mesh information is stored
  in :mod:`firedrake`).

Other than carefully tabulating how and which vertices/faces
correspond to other vertices/faces/cells, there are two main difficulties.

1. :mod:`meshmode` has discontinuous polynomial function spaces
   which may use different unit nodes than :mod:`firedrake`.
2. :mod:`meshmode` requires that all mesh elements be positively oriented,
   :mod:`firedrake` does not. Meanwhile, when :mod:`firedrake` creates
   a mesh, it changes the element ordering and the local vertex ordering.

(1.) is easily handled by insisting that the :mod:`firedrake`
:class:`~firedrake.functionspaceimpl.WithGeometry` uses polynomial elements
and that the group of the :class:`~meshmode.discretization.Discretization`
being converted is a
:class:`~meshmode.discretization.poly_element.InterpolatoryQuadratureSimplexElementGroup`
of the same order. Then, on each element, the function space being
represented is the same in :mod:`firedrake` and :mod:`meshmode`.
We may simply resample to one system or another's unit nodes.

To handle (2.),
once we associate a :mod:`meshmode`
element to the correct :mod:`firedrake` cell, we have something
like this picture:

.. graphviz::

    digraph{
        // created with graphviz2.38 dot
        // NODES

        mmNodes [label="Meshmode\nnodes"];
        mmRef   [label="Meshmode\nunit nodes"];
        fdRef   [label="Firedrake\nunit nodes"];
        fdNodes [label="Firedrake\nnodes"];

        // EDGES

        mmRef -> mmNodes [label=" f "];
        fdRef -> fdNodes [label=" g "];
    }

(Assume we have already
ensured that :mod:`meshmode` and :mod:`firedrake` use the
same reference element by mapping :mod:`firedrake`'s reference
element onto :mod:`meshmode`'s).
If :math:`f=g`, then we can resample function values from
one node set to the other. However, if :mod:`firedrake`
has reordered the vertices or if we flipped their order to
ensure :mod:`meshmode` has positively-oriented elements,
there is some map :math:`A` applied to the reference element
which implements this permutation of barycentric coordinates.
In this case, :math:`f=g\circ A`. Now, we have a connected diagram:

.. graphviz::

    digraph{
        // created with graphviz2.38 dot
        // NODES

        mmNodes [label="Meshmode\nnodes"];
        mmRef   [label="Meshmode\nunit nodes"];
        fdRef   [label="Firedrake\nunit nodes"];
        fdRef2  [label="Firedrake\nunit nodes"];
        fdNodes [label="Firedrake\nnodes"];

        // EDGES

        {rank=same; mmRef; fdRef;}
        {rank=same; mmNodes; fdNodes;}
        mmRef -> fdRef [label="Resampling", dir="both"];
        mmRef -> mmNodes [label=" f "];
        fdRef -> fdRef2 [label=" A "];
        fdRef2 -> fdNodes [label=" g "];
    }

In short, once we reorder the :mod:`firedrake` nodes so
that the mapping from the :mod:`meshmode` and :mod:`firedrake`
reference elements are the same, we can resample function values
at nodes from one set of unit nodes to another (and then undo
the reordering if converting function values
from :mod:`meshmode` to :mod:`firedrake`). The
information for this whole reordering process is
stored in
:attr:`~meshmode.interop.firedrake.connection.FiredrakeConnection.mm2fd_node_mapping`,
an array which associates each :mod:`meshmode` node
to the :mod:`firedrake` node found by tracing the
above diagram (i.e. it stores
:math:`g\circ A\circ \text{Resampling} \circ f^{-1}`).

For Developers: Firedrake Function Space Design Crash Course
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In Firedrake, meshes and function spaces have a close relationship.
In particular, this is  due to some structure described in this
`Firedrake pull request <https://github.com/firedrakeproject/firedrake/pull/627>`_.
If you wish to develop on / add to the implementation of conversion
between :mod:`meshmode` and :mod:`firedrake`, you will need
to understand their design style. Below is a crash course.

In short, it is the idea
that every function space should have a mesh, and the coordinates of the mesh
should be representable as a function on that same mesh, which must live
on some function space on the mesh... etc.
Under the hood, we divide between topological and geometric objects,
roughly as so:

(1) A reference element defined using :mod:`finat` and :mod:`FIAT`
    is used to define what meshmode calls the unit nodes and unit
    vertices. It is worth noting that :mod:`firedrake` does
    not require a positive orientation of elements and that its
    reference traingle is different than specified in :mod:`modepy`.

(2) A `~firedrake.mesh.MeshTopology`
    which holds information about connectivity
    and other topological properties, but nothing about geometry/coordinates
    etc.

(3) A class :class:`~firedrake.functionspaceimpl.FunctionSpace`
    created from a :mod:`finat` element and a
    `~firedrake.mesh.MeshTopology` which allows us to
    define functions mapping the nodes (defined by the
    :mod:`finat` element) of each element in the
    `~firedrake.mesh.MeshTopology` to some values.
    Note that the function :func:`~firedrake.functionspace.FunctionSpace`
    in the firedrake API is used to create objects of class
    :class:`~firedrake.functionspaceimpl.FunctionSpace`
    and :class:`~firedrake.functionspaceimpl.WithGeometry` (see
    (6)).

(4) A `~firedrake.function.CoordinatelessFunction`
    (in the sense that its *domain* has no coordinates)
    which is a function in a
    :class:`~firedrake.functionspaceimpl.FunctionSpace`.

(5) A `~firedrake.mesh.MeshGeometry` created from a
    :class:`~firedrake.functionspaceimpl.FunctionSpace`
    and a `~firedrake.function.CoordinatelessFunction`
    in that :class:`~firedrake.functionspaceimpl.FunctionSpace`
    which maps each dof to its geometric coordinates.

(6) A :class:`~firedrake.functionspaceimpl.WithGeometry` which is a
    :class:`~firedrake.functionspaceimpl.FunctionSpace` together
    with a `~firedrake.mesh.MeshGeometry`.
    This is the object returned
    usually returned to the user by a call
    to the :mod:`firedrake` function
    :func:`~firedrake.functionspace.FunctionSpace`.

(7) A :class:`~firedrake.function.Function` is defined on a
    :class:`~firedrake.functionspaceimpl.WithGeometry`.

Thus, by the coordinates of a mesh geometry we mean

(a) On the hidden back-end: a `~firedrake.function.CoordinatelessFunction`
    *f* on some function space defined only on the mesh topology.
(b) On the front-end: A :class:`~firedrake.function.Function`
    with the values of *f* but defined
    on a :class:`~firedrake.functionspaceimpl.WithGeometry`
    created from the :class:`~firedrake.functionspaceimpl.FunctionSpace`
    *f* lives in and the `~firedrake.mesh.MeshGeometry` *f* defines.

Basically, it's this picture (where :math:`a\to b` if :math:`b` depends on :math:`a`)

.. warning::

    In general, the :class:`~firedrake.functionspaceimpl.FunctionSpace`
    of the coordinates function
    of a :class:`~firedrake.functionspaceimpl.WithGeometry` may not be the same
    :class:`~firedrake.functionspaceimpl.FunctionSpace`
    as for functions which live in the
    :class:`~firedrake.functionspaceimpl.WithGeometry`.
    This picture
    only shows how the class definitions depend on each other.


.. graphviz::

    digraph{
        // created with graphviz2.38 dot
        // NODES

        top [label="Topological\nMesh"];
        ref [label="Reference\nElement"];
        fspace [label="Function Space"];
        coordless [label="Coordinateless\nFunction"];
        geo [label="Geometric\nMesh"];
        withgeo [label="With\nGeometry"];

        // EDGES

        top -> fspace;
        ref -> fspace;

        fspace -> coordless;

        top -> geo;
        coordless -> geo [label="Mesh\nCoordinates"];

        fspace -> withgeo;
        geo -> withgeo;
    }
