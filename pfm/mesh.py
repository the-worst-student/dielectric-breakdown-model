from pathlib import Path

import gmsh

WIDTH = 1.0
HEIGHT = 1.0

MESH_SIZE = 0.01

DOMAIN = 1
BOTTOM = 11
RIGHT = 12
TOP = 13
LEFT = 14
MESH_DIR = Path(__file__).resolve().parent / "meshes"

def generate_mesh():
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    mesh_path = MESH_DIR / "dielectric.msh"

    gmsh.initialize()

    try:
        gmsh.model.add("dielectric")
        p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, MESH_SIZE)
        p1 = gmsh.model.geo.addPoint(WIDTH, 0.0, 0.0, MESH_SIZE)
        p2 = gmsh.model.geo.addPoint(WIDTH, HEIGHT, 0.0, MESH_SIZE)
        p3 = gmsh.model.geo.addPoint(0.0, HEIGHT, 0.0, MESH_SIZE)

        bottom = gmsh.model.geo.addLine(p0, p1)
        right = gmsh.model.geo.addLine(p1, p2)
        top = gmsh.model.geo.addLine(p2, p3)
        left = gmsh.model.geo.addLine(p3, p0)

        boundary = gmsh.model.geo.addCurveLoop(
            [bottom, right, top, left]
        )
        surface = gmsh.model.geo.addPlaneSurface([boundary])

        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(2, [surface], DOMAIN)
        gmsh.model.setPhysicalName(2, DOMAIN, "dielectric")
        boundaries = [
            (bottom, BOTTOM, "bottom"),
            (right, RIGHT, "right"),
            (top, TOP, "top"),
            (left, LEFT, "left"),
        ]

        for curve, tag, name in boundaries:
            gmsh.model.addPhysicalGroup(1, [curve], tag)
            gmsh.model.setPhysicalName(1, tag, name)

        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)

        gmsh.model.mesh.generate(2)
        gmsh.write(str(mesh_path))

        print(f"Сетка сохранена: {mesh_path}")

    finally:
        gmsh.finalize()


if __name__ == "__main__":
    generate_mesh()