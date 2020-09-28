import numpy as np  # noqa
import sys


order = 4


def main():
    from meshmode.mesh.generation import (  # noqa
            make_curve_mesh, starfish)
    mesh1 = make_curve_mesh(starfish, np.linspace(0, 1, 20), 4)

    from meshmode.mesh.processing import affine_map, merge_disjoint_meshes
    mesh2 = affine_map(mesh1, b=np.array([2, 3]))

    mesh = merge_disjoint_meshes((mesh1, mesh2))

    from meshmode.mesh.visualization import draw_2d_mesh
    draw_2d_mesh(mesh, set_bounding_box=True)

    import matplotlib.pyplot as pt
    if sys.stdin.isatty():
        pt.show()
    else:
        pt.savefig("plot.pdf")


if __name__ == "__main__":
    main()
