import numpy as np

from .field import ElectricField
from .growth import growth_step, reached_top_electrode
from .disorder import generate_breakdown_strength
from .defects import build_weak_path

def run_simulation(config):
    field = ElectricField(config)

    random_strength = generate_breakdown_strength(
        ny=config.ny,
        nx=config.nx,
        correlation_length=config.correlation_length,
        weibull_shape=config.weibull_shape,
        seed=config.disorder_seed,
    )

    weak_path_factor, weak_path_mask = build_weak_path(
        config
    )

    strength = (
            random_strength
            * weak_path_factor
    )

    rng = np.random.default_rng(config.random_seed)
    history = []

    for step in range(config.max_growth_steps):
        cell = growth_step(
            field,
            strength,
            config.eta,
            config.mean_breakdown_field,
            rng,
        )

        if cell is None:
            return field, history, False

        history.append(tuple(map(int, cell)))

        if reached_top_electrode(field.channel):
            return field, history, True

    return field, history, False
