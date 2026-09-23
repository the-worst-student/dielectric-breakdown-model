from dataclasses import replace

from .defects import build_weak_path
from .morphology import compute_basic_morphology
from .simulator import run_simulation


def run_controlled_case(
    base_config,
    factor,
    disorder_seed,
    random_seed,
    capture_threshold=0.7,
):

    weak_path = replace(
        base_config.weak_path,
        kind="straight",
        factor=float(factor),
    )

    config = replace(
        base_config,
        weak_path=weak_path,
        disorder_seed=int(disorder_seed),
        random_seed=int(random_seed),
    )

    field, history, breakdown = run_simulation(
        config
    )

    _, weak_path_mask = build_weak_path(
        config
    )

    metrics = compute_basic_morphology(
        channel=field.channel,
        history=history,
        weak_path_mask=weak_path_mask,
    )

    if breakdown:
        status = "breakdown"

    elif len(history) == 0:
        status = "no_start"

    elif len(history) >= config.max_growth_steps:
        status = "max_steps"

    else:
        status = "arrested"

    captured = bool(
        metrics["area"] > 0
        and metrics["weak_path_fraction"]
        >= capture_threshold
    )

    return {
        "factor": float(factor),
        "disorder_seed": int(disorder_seed),
        "random_seed": int(random_seed),
        "status": status,
        "started": len(history) > 0,
        "breakdown": breakdown,
        "captured": captured,
        **metrics,
    }