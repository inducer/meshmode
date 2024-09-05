meshmode: High-Order Meshes and Discontinuous Function Spaces
=============================================================

.. image:: https://gitlab.tiker.net/inducer/meshmode/badges/main/pipeline.svg
    :alt: Gitlab Build Status
    :target: https://gitlab.tiker.net/inducer/meshmode/commits/main
.. image:: https://github.com/inducer/meshmode/workflows/CI/badge.svg
    :alt: Github Build Status
    :target: https://github.com/inducer/meshmode/actions?query=branch%3Amain+workflow%3ACI
.. image:: https://badge.fury.io/py/meshmode.png
    :alt: Python Package Index Release Page
    :target: https://pypi.org/project/meshmode/

Meshmode provides the "boring bits" of high-order unstructured discretization,
for simplices (triangles, tetrahedra) and tensor products (quads, hexahedra).
Features:

- 1/2/3D, line/surface/volume discretizations in each, curvilinear supported.
- "Everything is a (separate) discretization." (mesh boundaries are, element surfaces are,
  refined versions of the same mesh are) "Connections" transfer information
  between discretizations.
- Periodic connectivity.
- Mesh partitioning (not just) for distributed execution (e.g. via MPI).
- Interpolatory, quadrature (overintegration), and modal element-local discretizations.
- Independent of execution environment (GPU/CPU, numpy, ...)
  via `array contexts <https://github.com/inducer/arraycontext/>`__.
- Simple mesh refinement (via bisection). Adjacency currently only
  maintained if uniform.
- Input from Gmsh, Visualization to Vtk (both high-order curvilinear).
- Easy data exchange with `Firedrake <https://www.firedrakeproject.org/>`__.

Meshmode emerged as the shared discretization layer for `pytential
<https://github.com/inducer/pytential/>`__ (layer potentials) and `grudge
<https://github.com/inducer/grudge>`__ (discontinuous Galerkin).

Places on the web related to meshmode:

* `Source code on Github <https://github.com/inducer/meshmode>`_
* `Documentation <https://documen.tician.de/meshmode>`_
