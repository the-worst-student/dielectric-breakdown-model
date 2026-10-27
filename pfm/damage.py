import numpy as np
import ufl

from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import NonlinearProblem, assign


class DamageSolver:
    def __init__(self, d, d_old, phi, epsilon_r, dx, *, w_c, ell, mobility, dt, gamma=None):
        if np.issubdtype(np.dtype(PETSc.ScalarType), np.complexfloating):
            raise RuntimeError("Нужна вещественная сборка PETSc.")

        for name, value in (("ell", ell), ("mobility", mobility), ("dt", dt)):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} должно быть конечным и положительным.")

        self.d = d
        self.d_old = d_old
        self.dt = fem.Constant(d.function_space.mesh, PETSc.ScalarType(dt))

        if gamma is None:
            gamma = w_c * ell**2

        q = ufl.TestFunction(d.function_space)
        s = 1.0 - d
        w = 1.0 - (4.0 * s**3 - 3.0 * s**4)

        driving_functional = (
            -0.5 * epsilon_r * ufl.inner(ufl.grad(phi), ufl.grad(phi))
            + w_c * w
            + 0.25 * gamma * ufl.inner(ufl.grad(d), ufl.grad(d))
        ) * dx

        residual = (
            (d - d_old) * q / (mobility * self.dt) * dx
            + ufl.derivative(driving_functional, d, q)
        )

        self.problem = NonlinearProblem(
            residual,
            d,
            petsc_options_prefix="pfm_damage_",
            petsc_options={
                "snes_type": "vinewtonrsls",
                "snes_linesearch_type": "bt",
                "snes_rtol": 1e-8,
                "snes_atol": 1e-10,
                "snes_max_it": 50,
                "snes_error_if_not_converged": True,
                "ksp_type": "preonly",
                "pc_type": "lu",
                "ksp_error_if_not_converged": True,
            },
        )

        self.lower = self.problem.x.duplicate()
        self.upper = self.problem.x.duplicate()
        self.upper.set(1.0)

    def solve(self, dt):
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt должно быть конечным и положительным.")

        self.dt.value = PETSc.ScalarType(dt)
        self.d.x.array[:] = self.d_old.x.array
        self.d.x.scatter_forward()

        assign(self.d_old, self.lower)
        self.problem.solver.setVariableBounds(self.lower, self.upper)

        try:
            self.problem.solve()
        except PETSc.Error:
            self.d.x.array[:] = self.d_old.x.array
            self.d.x.scatter_forward()
            raise

        self.d.x.scatter_forward()

    def close(self):
        self.lower.destroy()
        self.upper.destroy()
