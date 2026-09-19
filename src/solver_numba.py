import numpy as np

from numba import njit


@njit(cache=True)
def apply_boundary_conditions_numba(phi, conductor, voltage):
    ny, nx = phi.shape

    for y in range(1, ny - 1):
        phi[y, 0] = phi[y, 1]
        phi[y, nx - 1] = phi[y, nx - 2]

    for x in range(nx):
        phi[0, x] = 0.0
        phi[ny - 1, x] = voltage

    for y in range(ny):
        for x in range(nx):
            if conductor[y, x]:
                phi[y, x] = 0.0


@njit(cache=True)
def laplace_residual_numba(phi, conductor):
    ny, nx = phi.shape
    residual = 0.0

    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            if conductor[y, x]:
                continue

            current_residual = abs(
                4.0 * phi[y, x]
                - phi[y - 1, x]
                - phi[y + 1, x]
                - phi[y, x - 1]
                - phi[y, x + 1]
            )

            if current_residual > residual:
                residual = current_residual

    return residual


@njit(cache=True)
def solve_laplace_sor_numba(phi, conductor, voltage, omega, tolerance, max_iterations, check_interval):
    ny, nx = phi.shape

    apply_boundary_conditions_numba(phi, conductor, voltage)

    for iteration in range(max_iterations):
        for parity in range(2):
            for y in range(1, ny - 1):
                if ((1 + y) & 1) == parity:
                    x_start = 1
                else:
                    x_start = 2

                for x in range(x_start, nx - 1, 2):
                    if conductor[y, x]:
                        continue

                    neighbor_avg = 0.25 * (phi[y - 1, x] + phi[y + 1, x] + phi[y, x - 1] + phi[y, x + 1])
                    phi[y, x] += omega * (neighbor_avg - phi[y, x])

            update_side_boundaries_numba(phi)

        if (iteration + 1) % check_interval == 0:
            residual = laplace_residual_numba(phi, conductor)

            if residual < tolerance:
                return iteration + 1, residual

    return max_iterations, laplace_residual_numba(phi, conductor)

@njit(cache=True)
def update_side_boundaries_numba(phi):
    ny, nx = phi.shape

    for y in range(1, ny - 1):
        phi[y, 0] = phi[y, 1]
        phi[y, nx - 1] = phi[y, nx - 2]


