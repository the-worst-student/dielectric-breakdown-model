from pathlib import Path

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, io
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmsh as gmshio

from mesh import BOTTOM, TOP, HEIGHT


ROOT = Path(__file__).resolve().parent

MESH_PATH = ROOT / "meshes" / "dielectric.msh"
RESULTS_DIR = ROOT / "results" / "electrostatics"

VOLTAGE = 1.0
EPSILON_R = 1.0

comm = MPI.COMM_WORLD

if comm.size != 1:
    raise RuntimeError("Первую проверку запускаем одним процессом.")

if not MESH_PATH.is_file():
    raise FileNotFoundError(
        f"Нет сетки {MESH_PATH}. Сначала запусти python pfm/mesh.py"
    )

mesh_data = gmshio.read_from_msh(
    str(MESH_PATH),
    comm,
    rank=0,
    gdim=2,
)

domain = mesh_data.mesh
facet_tags = mesh_data.facet_tags

if facet_tags is None:
    raise RuntimeError("В сетке отсутствуют метки границ.")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
V_phi = fem.functionspace(domain, ("Lagrange", 1))

fdim = domain.topology.dim - 1
bottom_facets = facet_tags.find(BOTTOM)
top_facets = facet_tags.find(TOP)
if bottom_facets.size == 0 or top_facets.size == 0:
    raise RuntimeError("Не найдены нижний или верхний электроды.")
bottom_dofs = fem.locate_dofs_topological(
    V_phi, fdim, bottom_facets
)
top_dofs = fem.locate_dofs_topological(
    V_phi, fdim, top_facets
)
bc_bottom = fem.dirichletbc(
    PETSc.ScalarType(0.0),
    bottom_dofs,
    V_phi,
)
bc_top = fem.dirichletbc(
    PETSc.ScalarType(VOLTAGE),
    top_dofs,
    V_phi,
)
bcs = [bc_bottom, bc_top]

phi_trial = ufl.TrialFunction(V_phi)
v = ufl.TestFunction(V_phi)

epsilon_r = fem.Constant(
    domain, PETSc.ScalarType(EPSILON_R)
)
source = fem.Constant(
    domain, PETSc.ScalarType(0.0)
)

dx = ufl.Measure("dx", domain=domain)

a = (
    epsilon_r
    * ufl.inner(ufl.grad(phi_trial), ufl.grad(v))
    * dx
)

L = source * ufl.conj(v) * dx

problem = LinearProblem(
    a,
    L,
    bcs=bcs,
    petsc_options_prefix="electrostatics_",
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_error_if_not_converged": True,
    },
)

phi = problem.solve()
phi.name = "potential"
phi.x.scatter_forward()

x = ufl.SpatialCoordinate(domain)

phi_exact = VOLTAGE * x[1] / HEIGHT
difference = phi - phi_exact

error_form = fem.form(
    ufl.inner(difference, difference) * dx
)

error_squared = float(
    np.real(fem.assemble_scalar(error_form))
)
error_L2 = np.sqrt(max(error_squared, 0.0))

print(f"L2 error of potential: {error_L2:.6e}")
print(f"Expected electric field: (0, {-VOLTAGE / HEIGHT})")

if not np.isfinite(error_L2) or error_L2 > 1e-8:
    raise RuntimeError(
        "Проверка электростатики не пройдена: "
        "проверь геометрию, метки и граничные условия."
    )

potential_path = RESULTS_DIR / "potential.pvd"
field_path = RESULTS_DIR / "electric_field.pvd"
gdim = domain.geometry.dim

V_E = fem.functionspace(
    domain,
    ("DG", 0, (gdim,)),
)

electric_field = fem.Function(V_E)
electric_field.name = "electric_field"

E_expression = fem.Expression(
    -ufl.grad(phi),
    V_E.element.interpolation_points,
)

electric_field.interpolate(E_expression)
electric_field.x.scatter_forward()
with io.VTKFile(comm, str(potential_path), "w") as vtk:
    vtk.write_function(phi, 0.0)

with io.VTKFile(comm, str(field_path), "w") as vtk:
    vtk.write_function(electric_field, 0.0)

print(f"Potential: {potential_path}")
print(f"Electric field: {field_path}")