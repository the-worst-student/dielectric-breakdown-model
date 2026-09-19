import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from src.config import SimulationConfig
from src.simulator import run_simulation


def make_random_seed(master_seed: int, threshold_index: int, replicate: int) -> int:
    seed_sequence = np.random.SeedSequence([master_seed, threshold_index, replicate])
    return int(seed_sequence.generate_state(1, dtype=np.uint64)[0])


def run_one(threshold: float, threshold_index: int, replicate: int, master_seed: int, base_config=None):
    random_seed = make_random_seed(master_seed, threshold_index, replicate)

    config = SimulationConfig() if base_config is None else base_config
    config.mean_breakdown_field = threshold
    config.random_seed = random_seed

    result = run_simulation(config)
    result.update({
        "threshold": threshold,
        "threshold_index": threshold_index,
        "replicate": replicate,
        "random_seed": random_seed,
    })
    return result


def run_monte_carlo(thresholds, n_replicates, master_seed, n_jobs=1, base_config=None):
    tasks = [(threshold, i, r) for i, threshold in enumerate(thresholds) for r in range(n_replicates)]

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(run_one)(threshold, i, r, master_seed, base_config)
        for threshold, i, r in tasks
    )

    return pd.DataFrame(results)
