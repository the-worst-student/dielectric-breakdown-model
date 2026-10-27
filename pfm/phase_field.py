import argparse
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import ufl
from dolfinx import fem, io
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI
from petsc4py import PETSc
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from charge_electrostatics import ChargeElectrostatics
from damage import DamageSolver
from material import MaterialField
from mesh import BOTTOM, TOP, HEIGHT, WIDTH


ROOT = Path(__file__).resolve().parent
MESH_PATH = ROOT / "meshes" / "dielectric.msh"
LENGTH_SCALE = min(WIDTH, HEIGHT)

EPSILON_R = 1.0
W_C = 1.0
ELL_PF = 0.02 * LENGTH_SCALE
GAMMA = W_C * ELL_PF**2
MOBILITY = 1.0
KAPPA = 1e-2

MATERIAL_LC = 0.20 * LENGTH_SCALE
MATERIAL_K = 4.0
MATERIAL_MODES = 1024
MATERIAL_BASIS_SEED = 0

SEED_LENGTH = 0.10 * HEIGHT
SEED_RADIUS = 0.015 * LENGTH_SCALE
D_THRESHOLD = 0.80

DT_MIN = 1e-6
DELTA_D_TARGET = 0.01
DELTA_D_MAX = 0.02
BOUNDS_TOL = 1e-8
SAVE_INTERVAL = 0.01
MAX_RETRIES = 10
MAX_STEPS = 1500
MAX_ATTEMPTS = 2500


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--distribution", choices=("weibull", "constant"), default="weibull"
    )
    parser.add_argument(
        "--gradient-model", choices=("constant", "local"), default="constant"
    )
    parser.add_argument("--dt-max", type=float, default=1e-3)
    parser.add_argument("--load-time", type=float, default=0.25)
    parser.add_argument("--hold-time", type=float, default=0.10)
    parser.add_argument("--reference-voltage", type=float, default=1.5)
    parser.add_argument("--max-seconds", type=float, default=300.0)
    args = parser.parse_args()
    positive = (
        args.dt_max, args.load_time,
        args.reference_voltage, args.max_seconds,
    )
    if not all(np.isfinite(v) and v > 0.0 for v in positive):
        parser.error("Шаг, время зарядки, напряжение и бюджет должны быть > 0.")
    if args.dt_max < DT_MIN:
        parser.error("--dt-max должно быть не меньше DT_MIN.")
    if not np.isfinite(args.hold_time) or args.hold_time < 0.0:
        parser.error("--hold-time должно быть конечным и >= 0.")
    return args


def initial_damage(x):
    distance_squared = (
        (x[0] - 0.5 * WIDTH)**2
        + np.maximum(x[1] - SEED_LENGTH, 0.0)**2
    )
    return np.exp(
        -distance_squared / SEED_RADIUS**2
    ).astype(PETSc.ScalarType)


def prescribed_charge(time, charge_max, load_time):
    return charge_max * float(np.clip(time / load_time, 0.0, 1.0))


def next_step_size(dt, increment, retried, dt_max):
    factor = float(np.clip(
        0.9 * DELTA_D_TARGET / max(increment, 1e-12), 0.5, 1.3
    ))
    if retried:
        factor = min(factor, 1.0)
    return float(np.clip(dt * factor, DT_MIN, dt_max))


class Diagnostics:
    def __init__(self, V, d, phi, epsilon_r, w_c, dx, *, gradient_coefficient):
        self.d = d
        self.comm = V.mesh.comm
        self.coordinates = V.tabulate_dof_coordinates()
        n_cells = V.mesh.topology.index_map(V.mesh.topology.dim).size_local
        cells = np.array([
            V.dofmap.cell_dofs(i) for i in range(n_cells)
        ], dtype=np.int32)
        if cells.shape[1] != 3:
            raise RuntimeError("Эта диагностика рассчитана на треугольники P1.")
        rows = cells[:, [0, 1, 1, 2, 2, 0]].ravel()
        cols = cells[:, [1, 0, 2, 1, 0, 2]].ravel()
        n_dofs = d.x.array.size
        self.adjacency = coo_matrix(
            (np.ones(rows.size), (rows, cols)),
            shape=(n_dofs, n_dofs),
        ).tocsr()
        self.roots = np.flatnonzero(d.x.array >= D_THRESHOLD)
        if self.roots.size == 0:
            raise RuntimeError(
                "Сетка не разрешает ядро зародыша при выбранном D_THRESHOLD."
            )
        points = self.coordinates[cells, :2]
        edge_lengths = np.linalg.norm(
            points[:, [1, 2, 0]] - points[:, [0, 1, 2]], axis=2
        )
        self.h_max = float(edge_lengths.max())
        self.area = self.integral(fem.form(1.0 * dx))
        s = 1.0 - d
        w = 1.0 - (4.0 * s**3 - 3.0 * s**4)
        self.forms = {
            "mean_damage": fem.form(d * dx),
            "damaged_area_fraction": fem.form(
                ufl.conditional(ufl.ge(d, D_THRESHOLD), 1.0, 0.0) * dx
            ),
            "damage_energy": fem.form(w_c * w * dx),
            "interface_energy": fem.form(
                0.25 * gradient_coefficient
                * ufl.inner(ufl.grad(d), ufl.grad(d)) * dx
            ),
            "electric_energy": fem.form(
                0.5 * epsilon_r
                * ufl.inner(ufl.grad(phi), ufl.grad(phi)) * dx
            ),
        }

    def integral(self, form):
        value = self.comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM)
        return float(np.real(value))

    def read(self):
        values = {name: self.integral(form) for name, form in self.forms.items()}
        values["mean_damage"] /= self.area
        values["damaged_area_fraction"] /= self.area
        active = np.flatnonzero(self.d.x.array >= D_THRESHOLD)
        graph = self.adjacency[active][:, active]
        _, labels = connected_components(graph, directed=False)
        root_labels = np.unique(labels[np.isin(active, self.roots)])
        connected = active[np.isin(labels, root_labels)]
        if connected.size == 0:
            raise RuntimeError("Потеряно ядро исходного повреждения.")
        points = self.coordinates[connected]
        values["connected_height_fraction"] = float(points[:, 1].max() / HEIGHT)
        values["connected_width_fraction"] = float(np.ptp(points[:, 0]) / WIDTH)
        values["total_energy"] = sum(values[name] for name in (
            "damage_energy", "interface_energy", "electric_energy"
        ))
        return values


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    if comm.size != 1:
        raise RuntimeError("Эта версия запускается одним MPI-процессом.")
    if np.issubdtype(np.dtype(PETSc.ScalarType), np.complexfloating):
        raise RuntimeError("Нужна вещественная сборка PETSc.")
    if not MESH_PATH.is_file():
        raise FileNotFoundError(f"Нет сетки: {MESH_PATH}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = ROOT / "results" / (
        f"pfm_gamma_{args.gradient_model}_{args.distribution}_seed{args.seed}_{stamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    mesh_data = gmshio.read_from_msh(str(MESH_PATH), comm, rank=0, gdim=2)
    domain = mesh_data.mesh
    if mesh_data.facet_tags is None:
        raise RuntimeError("В сетке отсутствуют метки электродов.")

    V = fem.functionspace(domain, ("Lagrange", 1))
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 6})
    material = MaterialField(
        mean=W_C, correlation_length=MATERIAL_LC,
        distribution=args.distribution, weibull_shape=MATERIAL_K,
        seed=args.seed, n_modes=MATERIAL_MODES,
        basis_seed=MATERIAL_BASIS_SEED,
    )
    w_c = fem.Function(V)
    w_c.name = "critical_energy"
    w_c.interpolate(material)
    w_c.x.scatter_forward()
    gradient_coefficient = (
        GAMMA if args.gradient_model == "constant" else w_c * ELL_PF**2
    )

    d = fem.Function(V)
    d.name = "damage"
    d.interpolate(initial_damage)
    d.x.scatter_forward()
    d_old = fem.Function(V)
    d_old.x.array[:] = d.x.array
    d_old.x.scatter_forward()

    s = 1.0 - d
    epsilon_r = EPSILON_R * (1.0 + KAPPA) / (
        4.0 * s**3 - 3.0 * s**4 + KAPPA
    )
    electrostatics = ChargeElectrostatics(
        V, mesh_data.facet_tags, epsilon_r, dx,
        bottom_tag=BOTTOM, top_tag=TOP,
    )
    phi = electrostatics.initialize(1.0)
    initial_capacitance = electrostatics.capacitance
    charge_max = initial_capacitance * args.reference_voltage
    electrostatics.solve(0.0)

    V_E = fem.functionspace(domain, ("DG", 0, (domain.geometry.dim,)))
    electric_field = fem.Function(V_E)
    electric_field.name = "electric_field"
    E_expression = fem.Expression(
        -ufl.grad(phi), V_E.element.interpolation_points
    )
    epsilon_visual = fem.Function(V)
    epsilon_visual.name = "relative_permittivity"
    epsilon_expression = fem.Expression(epsilon_r, V.element.interpolation_points)

    diagnostics = Diagnostics(
        V, d, phi, epsilon_r, w_c, dx,
        gradient_coefficient=gradient_coefficient,
    )
    metrics = diagnostics.read()
    initial_height = metrics["connected_height_fraction"]

    print(f"Результаты: {run_dir}", flush=True)
    print(
        f"Зарядка 0 -> {charge_max:.6f} за t={args.load_time:.3f}; "
        f"выдержка {args.hold_time:.3f}; ell_ref/h_max={ELL_PF / diagnostics.h_max:.2f}",
        flush=True,
    )
    print(
        f"Градиентная модель: {args.gradient_model}; Gamma_ref={GAMMA:.6e}; "
        f"w_c=[{w_c.x.array.min():.4f}, {w_c.x.array.max():.4f}]",
        flush=True,
    )

    damage_solver = DamageSolver(
        d, d_old, phi, epsilon_r, dx,
        w_c=w_c, ell=ELL_PF, mobility=MOBILITY,
        dt=min(0.5 * args.dt_max, 1e-3),
        gamma=gradient_coefficient,
    )
    time = 0.0
    step = 0
    attempts = 0
    rejected = 0
    current_dt = min(0.5 * args.dt_max, 1e-3)
    t_end = args.load_time + args.hold_time
    time_tol = 1e-12 * max(1.0, t_end)
    next_save = SAVE_INTERVAL
    last_saved = 0.0
    stop_reason = "running"
    onset_time = None
    started = perf_counter()

    def budget_reason():
        if perf_counter() - started >= args.max_seconds:
            return "wall_time_budget"
        if step >= MAX_STEPS:
            return "step_budget"
        if attempts >= MAX_ATTEMPTS:
            return "attempt_budget"
        return None

    def save_state(state_vtk, field_vtk):
        electric_field.interpolate(E_expression)
        electric_field.x.scatter_forward()
        epsilon_visual.interpolate(epsilon_expression)
        epsilon_visual.x.scatter_forward()
        state_vtk.write_function([phi, d, epsilon_visual, w_c], time)
        field_vtk.write_function(electric_field, time)

    try:
        with (
            io.VTKFile(comm, str(run_dir / "evolution.pvd"), "w") as state_vtk,
            io.VTKFile(comm, str(run_dir / "electric_field.pvd"), "w") as field_vtk,
        ):
            save_state(state_vtk, field_vtk)

            while t_end - time > time_tol:
                stop_reason = budget_reason() or "running"
                if stop_reason != "running":
                    break
                d_old.x.array[:] = d.x.array
                d_old.x.scatter_forward()
                boundary = args.load_time if time < args.load_time - time_tol else t_end
                trial_dt = min(current_dt, boundary - time)
                accepted = False

                for retry in range(MAX_RETRIES + 1):
                    limit = budget_reason()
                    if limit is not None:
                        stop_reason = limit
                        break
                    d.x.array[:] = d_old.x.array
                    d.x.scatter_forward()
                    trial_time = time + trial_dt
                    if boundary - trial_time <= time_tol:
                        trial_time = boundary
                    attempts += 1
                    electrostatics.solve(prescribed_charge(
                        trial_time, charge_max, args.load_time
                    ))
                    cause = None
                    try:
                        damage_solver.solve(trial_dt)
                    except PETSc.Error:
                        reason = int(damage_solver.problem.solver.getConvergedReason())
                        if reason not in (-5, -6):
                            raise
                        cause = f"snes_{reason}"

                    if cause is None:
                        values = d.x.array
                        increment_values = values - d_old.x.array
                        if (
                            not np.all(np.isfinite(values))
                            or values.min() < -BOUNDS_TOL
                            or values.max() > 1.0 + BOUNDS_TOL
                            or increment_values.min() < -BOUNDS_TOL
                        ):
                            raise RuntimeError("Нарушены границы или необратимость d.")
                        increment = float(increment_values.max())
                        if increment > DELTA_D_MAX:
                            cause = f"damage_increment_{increment:.6e}"

                    if cause is None:
                        accepted = True
                        iterations = damage_solver.problem.solver.getIterationNumber()
                        break

                    rejected += 1
                    print(
                        f"Повтор при t={time:.6f}, dt={trial_dt:.2e}: {cause}",
                        flush=True,
                    )
                    d.x.array[:] = d_old.x.array
                    d.x.scatter_forward()
                    if trial_dt * 0.5 < DT_MIN or retry == MAX_RETRIES:
                        stop_reason = "step_not_accepted"
                        break
                    trial_dt *= 0.5

                if not accepted:
                    if stop_reason == "running":
                        stop_reason = "step_not_accepted"
                    d.x.array[:] = d_old.x.array
                    d.x.scatter_forward()
                    electrostatics.solve(prescribed_charge(
                        time, charge_max, args.load_time
                    ))
                    break

                electrostatics.solve(prescribed_charge(
                    trial_time, charge_max, args.load_time
                ))
                time = trial_time
                step += 1
                metrics = diagnostics.read()

                if (
                    onset_time is None
                    and metrics["connected_height_fraction"] >= initial_height + 0.02
                ):
                    onset_time = time

                current_dt = next_step_size(
                    trial_dt, increment, retry > 0, args.dt_max
                )

                if metrics["connected_height_fraction"] >= 0.95:
                    stop_reason = "channel_reached_95_percent_height"
                if time >= next_save - time_tol or stop_reason != "running":
                    save_state(state_vtk, field_vtk)
                    last_saved = time
                    next_save = (np.floor(time / SAVE_INTERVAL) + 1) * SAVE_INTERVAL
                    print(
                        f"t={time:.5f}/{t_end:.3f}, step={step}, dt={trial_dt:.2e}, "
                        f"V={electrostatics.voltage:.4f}, "
                        f"C={electrostatics.capacitance:.4f}, "
                        f"height={metrics['connected_height_fraction']:.3f}, "
                        f"area={metrics['damaged_area_fraction']:.4f}, "
                        f"delta_d={increment:.3e}, SNES={iterations}, "
                        f"rejects={rejected}", flush=True,
                    )
                if stop_reason != "running":
                    break

            if stop_reason == "running":
                stop_reason = "completed"
            if time > last_saved + time_tol:
                save_state(state_vtk, field_vtk)

    except BaseException:
        stop_reason = "exception"
        raise
    finally:
        damage_solver.close()
        print(
            f"Статус: {stop_reason}; t={time:.6f}/{t_end:.6f}; "
            f"шагов={step}; попыток={attempts}; повторов={rejected}; "
            f"затрачено={perf_counter() - started:.1f} с",
            flush=True,
        )

    if onset_time is not None:
        print(f"Рост высоты канала на 0.02H достигнут при t={onset_time:.6f}")
    print(f"Визуализация: {run_dir / 'evolution.pvd'}", flush=True)


if __name__ == "__main__":
    main()
