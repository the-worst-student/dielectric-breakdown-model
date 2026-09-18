import numpy as np

from .field import ElectricField
from .growth import growth_step, reached_top_electrode
from .disorder import generate_breakdown_strength


def run_simulation(config):
    field = ElectricField(config)

    strength = generate_breakdown_strength(
        ny=config.ny,
        nx=config.nx,
        correlation_length=config.correlation_length,
        weibull_shape=config.weibull_shape,
        seed=config.disorder_seed,
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
            return {
                "status": "arrested",
                "steps": len(history),
                "max_y": max([p[0] for p in history], default=None),
            }

        history.append(tuple(map(int, cell)))

        if reached_top_electrode(field.channel):
            return {
                "status": "breakdown",
                "steps": len(history),
                "max_y": max([p[0] for p in history], default=None),
            }

    return {
        "status": "max_steps",
        "steps": len(history),
        "max_y": max([p[0] for p in history], default=None),
    }
