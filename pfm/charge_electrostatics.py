import numpy as np
import ufl

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem


class ChargeElectrostatics:
    def __init__(self, V, facet_tags, epsilon_r, dx, *, bottom_tag, top_tag):
        self.comm = V.mesh.comm

        self.unit_potential = fem.Function(V)
        self.unit_potential.name = "unit_potential"

        self.phi = fem.Function(V)
        self.phi.name = "potential"

        self.capacitance = np.nan
        self.voltage = np.nan
        self.charge = np.nan

        fdim = V.mesh.topology.dim - 1

        bottom_facets = facet_tags.find(bottom_tag)
        top_facets = facet_tags.find(top_tag)

        if bottom_facets.size == 0 or top_facets.size == 0:
            raise RuntimeError("Не найдены электроды.")

        bottom_dofs = fem.locate_dofs_topological(
            V, fdim, bottom_facets
        )

        top_dofs = fem.locate_dofs_topological(
            V, fdim, top_facets
        )

        bc_bottom = fem.dirichletbc(
            PETSc.ScalarType(0.0),
            bottom_dofs,
            V,
        )

        bc_top = fem.dirichletbc(
            PETSc.ScalarType(1.0),
            top_dofs,
            V,
        )

        trial = ufl.TrialFunction(V)
        test = ufl.TestFunction(V)

        source = fem.Constant(
            V.mesh,
            PETSc.ScalarType(0.0),
        )

        a = (
            epsilon_r
            * ufl.inner(ufl.grad(trial), ufl.grad(test))
            * dx
        )

        L = source * ufl.conj(test) * dx

        self.problem = LinearProblem(
            a,
            L,
            u=self.unit_potential,
            bcs=[bc_bottom, bc_top],
            petsc_options_prefix="pfm_charge_electrostatics_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "ksp_error_if_not_converged": True,
            },
        )

        self.capacitance_form = fem.form(
            epsilon_r
            * ufl.inner(
                ufl.grad(self.unit_potential),
                ufl.grad(self.unit_potential),
            )
            * dx
        )

    def _solve_unit(self):
        self.problem.solve()
        self.unit_potential.x.scatter_forward()

        local_value = fem.assemble_scalar(
            self.capacitance_form
        )

        global_value = self.comm.allreduce(
            local_value,
            op=MPI.SUM,
        )

        capacitance = float(np.real(global_value))

        if not np.isfinite(capacitance) or capacitance <= 0.0:
            raise RuntimeError(
                f"Некорректная ёмкость: {capacitance}"
            )

        self.capacitance = capacitance

    def _set_charge(self, charge):
        charge = float(charge)

        if not np.isfinite(charge):
            raise ValueError("Заряд должен быть конечным.")

        self.charge = charge
        self.voltage = charge / self.capacitance

        self.phi.x.array[:] = (
            self.voltage
            * self.unit_potential.x.array
        )

        self.phi.x.scatter_forward()

        return self.phi

    def initialize(self, voltage):
        voltage = float(voltage)

        if not np.isfinite(voltage):
            raise ValueError("Напряжение должно быть конечным.")

        self._solve_unit()

        return self._set_charge(
            self.capacitance * voltage
        )

    def solve(self, charge):
        self._solve_unit()

        return self._set_charge(charge)