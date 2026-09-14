import numpy as np

from .config import SimulationConfig

class ElectricField:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.phi = np.zeros(
            (config.ny, config.nx),
            dtype=float
        )
        y = np.linspace(
            0.0,
            config.voltage,
            config.ny
        )
        # Potential of capacitorr:
        # phi(y) = V0 * y / Ly
        self.phi[:] = y[:, None]

        self.channel = np.zeros(
            (config.ny, config.nx),
            dtype=bool
        )

        seed_x = config.nx // 2
        seed_y = 0

        self.channel[seed_y, seed_x] = True

        self.apply_boundary_conditions()
    def apply_boundary_conditions(self):
        self.phi[0, :] = 0.0
        self.phi[-1, :] = self.config.voltage
        self.phi[self.channel] = 0.0
        self.phi[:, 0] = self.phi[:, 1]
        self.phi[:, -1] = self.phi[:, -2]

        self.phi[0, :] = 0.0
        self.phi[-1, :] = self.config.voltage
        self.phi[self.channel] = 0.0

    def solve_laplace(self):
        error = np.inf

        for iteration in range(self.config.max_iterations):
            old_phi = self.phi.copy()

            self.phi[1:-1, 1:-1] = 0.25 * (
                    old_phi[2:, 1:-1]
                    + old_phi[:-2, 1:-1]
                    + old_phi[1:-1, 2:]
                    + old_phi[1:-1, :-2]
            )

            self.apply_boundary_conditions()

            error = np.max(
                np.abs(self.phi - old_phi)
            )

            if error < self.config.tolerance:
                return iteration + 1, error

        return self.config.max_iterations, error

    def electric_field(self):
        dphi_dy, dphi_dx = np.gradient(self.phi)
        Ex = -dphi_dx
        Ey = -dphi_dy
        E = np.sqrt(
            Ex ** 2 + Ey ** 2
        )
        return Ex, Ey, E