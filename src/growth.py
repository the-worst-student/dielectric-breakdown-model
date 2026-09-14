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

def get_growth_probabilities(E, front, eta):
    field_values = E[front]
    weights = field_values ** eta
    probabilities = weights / np.sum(weights)

    return probabilities

def choose_growth_cell(candidates, probabilities, rng):
    index = rng.choice(len(candidates), p=probabilities)

    return candidates[index]

def growth_step(field, eta, rng):
    field.apply_boundary_conditions()
    field.solve_laplace()
    Ex, Ey, E = field.electric_field()
    front = get_front(field.channel)
    candidates = np.argwhere(front)
    if len(candidates) == 0:
        return None
    probabilities = get_growth_probabilities(
        E,
        front,
        eta
    )
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

