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

    def solve_laplace_jacobi_residual(self):
        check_interval = self.config.residual_check_interval

        for iteration in range(self.config.max_iterations):
            old_phi = self.phi.copy()

            self.phi[1:-1, 1:-1] = 0.25 * (
                    old_phi[2:, 1:-1]
                    + old_phi[:-2, 1:-1]
                    + old_phi[1:-1, 2:]
                    + old_phi[1:-1, :-2]
            )

            self.apply_boundary_conditions()

            if (iteration + 1) % check_interval == 0:
                residual = self.laplace_residual()

                if residual < self.config.tolerance:
                    return iteration + 1, residual

        residual = self.laplace_residual()

        return self.config.max_iterations, residual

    def electric_field(self):
        dphi_dy, dphi_dx = np.gradient(self.phi)
        Ex = -dphi_dx
        Ey = -dphi_dy
        E = np.sqrt(
            Ex ** 2 + Ey ** 2
        )
        return Ex, Ey, E

    def laplace_residual(self):
        center = self.phi[1:-1, 1:-1]

        neighbor_sum = (
                self.phi[2:, 1:-1]
                + self.phi[:-2, 1:-1]
                + self.phi[1:-1, 2:]
                + self.phi[1:-1, :-2]
        )

        residual = (
                neighbor_sum
                - 4.0 * center
        )

        free = ~self.channel[1:-1, 1:-1]

        if not np.any(free):
            return 0.0

        return np.max(
            np.abs(residual[free])
        )

    def solve_laplace_sor(self):
        omega = self.config.omega
        tolerance = self.config.tolerance
        max_iterations = self.config.max_iterations
        check_interval = self.config.residual_check_interval

        ny = self.config.ny
        nx = self.config.nx

        self.apply_boundary_conditions()
        yy, xx = np.indices((ny, nx))

        red = ((xx + yy) % 2 == 0)[1:-1, 1:-1]
        black = ~red

        for iteration in range(max_iterations):
            free = ~self.channel[1:-1, 1:-1]
            center = self.phi[1:-1, 1:-1]

            neighbor_avg = 0.25 * (
                    self.phi[2:, 1:-1]
                    + self.phi[:-2, 1:-1]
                    + self.phi[1:-1, 2:]
                    + self.phi[1:-1, :-2]
            )

            red_free = red & free

            center[red_free] = (
                    center[red_free]
                    + omega * (
                            neighbor_avg[red_free]
                            - center[red_free]
                    )
            )

            self.apply_boundary_conditions()
            center = self.phi[1:-1, 1:-1]

            neighbor_avg = 0.25 * (
                    self.phi[2:, 1:-1]
                    + self.phi[:-2, 1:-1]
                    + self.phi[1:-1, 2:]
                    + self.phi[1:-1, :-2]
            )

            black_free = black & free

            center[black_free] = (
                    center[black_free]
                    + omega * (
                            neighbor_avg[black_free]
                            - center[black_free]
                    )
            )

            self.apply_boundary_conditions()

            if (iteration + 1) % check_interval == 0:
                residual = self.laplace_residual()

                if residual < tolerance:
                    return iteration + 1, residual

        residual = self.laplace_residual()

        return max_iterations, residual