import numpy as np
from IPython.core import history

from .field import ElectricField
from .growth import growth_step, reached_top_electrode

def run_simulation(config):
    field = ElectricField(config)
    rng = np.random.default_rng(seed=config.random_seed)
    history = []
    for step in range(config.max_iterations):
        cell = growth_step(
            field,
            config.eta,
            rng
        )
        if cell is None:
            return history, field, False
        history.append(
            tuple(map(int, cell))
        )
        if reached_top_electrode(field.channel):
            return field, history, True
    return field, history, False