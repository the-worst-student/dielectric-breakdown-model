import numpy as np
import pandas as pd

from dataclasses import replace

from src.simulator import run_simulation


def run_parameter_sweep(
    base_config,
    parameter_name,
    parameter_values,
    disorder_seeds,
    random_seeds,
):
    records = []

    for value in parameter_values:
        for disorder_seed in disorder_seeds:
            for random_seed in random_seeds:

                config = replace(
                    base_config,
                    **{parameter_name: value},
                    disorder_seed=disorder_seed,
                    random_seed=random_seed,
                )

                field, history, breakdown = run_simulation(config)

                if breakdown:
                    status = "breakdown"
                elif len(history) == 0:
                    status = "no_start"
                elif len(history) >= config.max_growth_steps:
                    status = "max_steps"
                else:
                    status = "arrested"

                if np.any(field.channel):
                    max_y = int(
                        np.max(np.where(field.channel)[0])
                    )
                else:
                    max_y = 0

                height_fraction = max_y / (config.ny - 1)

                records.append(
                    {
                        parameter_name: value,
                        "disorder_seed": disorder_seed,
                        "random_seed": random_seed,
                        "status": status,
                        "started": status != "no_start",
                        "breakdown": breakdown,
                        "steps": len(history),
                        "max_y": max_y,
                        "height_fraction": height_fraction,
                    }
                )

    return pd.DataFrame(records)


def build_material_summary(
    results_df,
    parameter_name,
):
    material_summary = (
        results_df
        .groupby([parameter_name, "disorder_seed"])
        .agg(
            n_started=("started", "sum"),
            n_breakdown=("breakdown", "sum"),
            p_start=("started", "mean"),
            p_breakdown=("breakdown", "mean"),
            mean_steps=("steps", "mean"),
            mean_height=("height_fraction", "mean"),
        )
        .reset_index()
    )

    material_summary["p_breakdown_given_start"] = (
        material_summary["n_breakdown"]
        / material_summary["n_started"]
    )

    material_summary.loc[
        material_summary["n_started"] == 0,
        "p_breakdown_given_start",
    ] = np.nan

    return material_summary


def bootstrap_probabilities(
    group,
    n_random,
    n_bootstrap=5000,
    random_state=42,
):
    rng = np.random.default_rng(random_state)

    estimates = []

    for _ in range(n_bootstrap):
        indices = rng.choice(
            len(group),
            size=len(group),
            replace=True,
        )

        sample = group.iloc[indices]

        n_started = sample["n_started"].sum()
        n_breakdown = sample["n_breakdown"].sum()
        n_total = len(sample) * n_random

        p_start = n_started / n_total
        p_breakdown = n_breakdown / n_total

        if n_started > 0:
            p_breakdown_given_start = (
                n_breakdown / n_started
            )
        else:
            p_breakdown_given_start = np.nan

        estimates.append(
            (
                p_start,
                p_breakdown,
                p_breakdown_given_start,
            )
        )

    estimates = np.asarray(estimates)

    return {
        "p_start": np.mean(estimates[:, 0]),
        "p_start_low": np.quantile(estimates[:, 0], 0.025),
        "p_start_high": np.quantile(estimates[:, 0], 0.975),

        "p_breakdown": np.mean(estimates[:, 1]),
        "p_breakdown_low": np.quantile(estimates[:, 1], 0.025),
        "p_breakdown_high": np.quantile(estimates[:, 1], 0.975),

        "p_breakdown_given_start": np.nanmean(estimates[:, 2]),
        "conditional_low": np.nanquantile(estimates[:, 2], 0.025),
        "conditional_high": np.nanquantile(estimates[:, 2], 0.975),
    }


def build_bootstrap_summary(
    material_summary,
    parameter_name,
    parameter_values,
    n_random,
    n_bootstrap=5000,
):
    records = []

    for i, value in enumerate(parameter_values):
        group = material_summary[
            material_summary[parameter_name] == value
        ]

        result = bootstrap_probabilities(
            group,
            n_random=n_random,
            n_bootstrap=n_bootstrap,
            random_state=42 + i,
        )

        result[parameter_name] = value

        records.append(result)

    return pd.DataFrame(records)