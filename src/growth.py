import numpy as np

def get_front(channel):
    front = np.zeros_like(channel, dtype=bool)

    front[1:, :] |= channel[:-1, :]
    front[:-1, :] |= channel[1:, :]
    front[:, 1:] |= channel[:, :-1]
    front[:, :-1] |= channel[:, 1:]
    front[channel] = False

    front[0, :] = False

    return front

def get_growth_probabilities(
    E,
    strength,
    front,
    eta,
    mean_breakdown_field,
):
    field_values = E[front]
    strength_values = strength[front]

    threshold_values = (
        mean_breakdown_field
        * strength_values
    )

    driving = np.maximum(
        field_values / threshold_values - 1.0,
        0.0,
    )

    active = driving > 0.0

    if not np.any(active):
        return None

    weights = np.zeros_like(driving)
    weights[active] = driving[active] ** eta
    probabilities = weights / np.sum(weights)

    return probabilities

def choose_growth_cell(candidates, probabilities, rng):
    index = rng.choice(len(candidates), p=probabilities)

    return candidates[index]

def growth_step(field, strength, eta, mean_breakdown_field, rng):
    field.apply_boundary_conditions()
    field.solve_laplace_sor_numba()
    Ex, Ey, E = field.electric_field()
    conductor = field.needle | field.channel
    front = get_front(conductor)
    candidates = np.argwhere(front)
    if len(candidates) == 0:
        return None
    probabilities = get_growth_probabilities(
        E,
        strength,
        front,
        eta,
        mean_breakdown_field
    )

    if probabilities is None:
        return None

    next_cell = choose_growth_cell(
        candidates,
        probabilities,
        rng
    )
    y, x = next_cell
    field.channel[y, x] = True
    return next_cell

def reached_top_electrode(channel):
    return np.any(channel[-1, :])

