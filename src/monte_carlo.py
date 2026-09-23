import numpy as np
import pandas as pd

from dataclasses import replace
from joblib import Parallel, delayed

from src.config import SimulationConfig
from src.simulator import run_simulation


def make_random_seed(master_seed: int, threshold: float, replicate: int) -> int:
    threshold_key = int(round(float(threshold) * 1_000_000))
    seed_sequence = np.random.SeedSequence([master_seed, threshold_key, replicate])
    return int(seed_sequence.generate_state(1, dtype=np.uint64)[0])


def run_one(threshold: float, threshold_index: int, replicate: int, master_seed: int, base_config=None):
    random_seed = make_random_seed(master_seed, threshold, replicate)

    config = SimulationConfig() if base_config is None else replace(base_config)
    config = replace(config, mean_breakdown_field=threshold, random_seed=random_seed)

    field, history, breakdown = run_simulation(config)

    if breakdown:
        status = "breakdown"
    elif len(history) == 0:
        status = "no_start"
    elif len(history) >= config.max_growth_steps:
        status = "max_steps"
    else:
        status = "arrested"

    max_y = int(np.max(np.where(field.channel)[0])) if np.any(field.channel) else 0

    return {
        "status": status,
        "steps": len(history),
        "max_y": max_y,
        "threshold": threshold,
        "threshold_index": threshold_index,
        "replicate": replicate,
        "random_seed": random_seed,
    }


def run_monte_carlo(thresholds, n_replicates, master_seed, n_jobs=1, base_config=None):
    tasks = [(threshold, i, r) for i, threshold in enumerate(thresholds) for r in range(n_replicates)]

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(run_one)(threshold, i, r, master_seed, base_config)
        for threshold, i, r in tasks
    )

    return pd.DataFrame(results)
