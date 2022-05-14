Common infrastructure
=====================

.. automodule:: meshmode
.. automodule:: meshmode.mesh.tools

Mesh management
===============

.. currentmodule:: meshmode.mesh

Design of the Data Structure
----------------------------

Why does a :class:`Mesh` need to be broken into :class:`MeshElementGroup` instances?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Elements can be of different types (e.g. triangle, quadrilateral,
tetrahedron, what have you). In addition, elements may vary in the
polynomial degree used to represent them (see also below).

All these bits of information could in principle be stored by element,
but having large, internally homogeneous groups is a good thing from an
efficiency standpoint. (So that you can, e.g., launch one GPU kernel to
deal with all order-3 triangles, instead of maybe having to dispatch
based on type and size inside the kernel)

What is the difference between 'vertices' and 'nodes'?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Nodes exist mainly to represent the (potentially non-affine) deformation of
each element, by a one-to-one correspondence with
:attr:`MeshElementGroup.unit_nodes`.  They are unique to each element. Vertices
on the other hand exist to clarify whether or not a point shared by two
elements is actually identical (or just happens to be "close"). This is done by
assigning (single, globally shared) vertex numbers and having elements refer to
them.

Consider the following picture:

.. tikz:: Mesh nodes and vertices
    :libs: decorations.markings
    :align: center
    :xscale: 60

    \tikzset{-node-/.style={decoration={
        markings,
        mark=at position #1 with {
            \draw[fill=green!60!black, line width=0.4] circle [radius=0.1];}},
        postaction={decorate}}}

    \draw [thick, -node-=.33,-node-=0.66] (0, 0) to (3, 0);
    \draw [thick, -node-=.33,-node-=0.66] (3, 0) to (0, 3);
    \draw [thick, -node-=.33,-node-=0.66] (0, 3) to (0, 0);
    \draw [fill=green!60!black] (0, 0) circle [radius=0.1];
    \draw [fill=green!60!black] (3, 0) circle [radius=0.1];
    \draw [fill=green!60!black] (0, 3) circle [radius=0.1];
    \draw [fill=green!60!black] (1, 1) circle [radius=0.1];

    \draw [ultra thick,bend left,->] (1.5, 2.5) to (3.5, 2.5);

    \begin{scope}[shift={(4, 2)},rotate=-47]
    \draw [thick, -node-=.33,-node-=0.66] (0, 0) to [bend left] (3, 0);
    \draw [thick, -node-=.33,-node-=0.66] (3, 0) to [bend right] (0, 3);
    \draw [thick, -node-=.33,-node-=0.66] (0, 3) to [bend right] (0, 0);
    \draw [dashed] (3, 0) to [bend left] (5, 3);
    \draw [dashed] (5, 3) to [bend right] (0, 3);

    \draw [fill=magenta!60] (0, 0) circle [radius=0.15];
    \draw [fill=magenta!60] (0, 3) circle [radius=0.15];
    \draw [fill=magenta!60] (3, 0) circle [radius=0.15];
    \draw [fill=green!60!black] (0.75, 1.5) circle [radius=0.1];
    \end{scope}

    \node at (1.5, -0.5) [below] {Reference Element};
    \draw [fill=green!60!black] (0.25, -1.5) circle [radius=0.1]
        node [right] {~ unit nodes};
    \node at (6.5, -0.5) [below] {Mesh Element};
    \draw [fill=green!60!black] (5.5, -1.5) circle [radius=0.1]
        node [right] {~ nodes (unique)};
    \draw [fill=magenta!60] (5.5, -2) circle [radius=0.15]
        node [right] {~ vertices (shared)};

Mesh Data Structure
-------------------

.. automodule:: meshmode.mesh

Mesh generation
---------------

.. automodule:: meshmode.mesh.generation

Mesh input/output
-----------------

.. automodule:: meshmode.mesh.io

Mesh processing
---------------

.. automodule:: meshmode.mesh.processing

Mesh refinement
---------------

.. automodule:: meshmode.mesh.refinement

Mesh visualization
------------------

.. automodule:: meshmode.mesh.visualization

.. vim: sw=4
