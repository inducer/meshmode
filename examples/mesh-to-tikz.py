from meshmode.mesh.io import FileSource, generate_gmsh
from meshmode.mesh.visualization import mesh_to_tikz


h = 0.3
order = 1

mesh = generate_gmsh(
        FileSource("../test/blob-2d.step"), 2, order=order,
        force_ambient_dim=2,
        other_options=[
            "-string", f"Mesh.CharacteristicLengthMax = {h};"]
        )

print(mesh_to_tikz(mesh))
