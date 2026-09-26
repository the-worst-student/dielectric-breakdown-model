import numpy as np

def depth_trajectory(history) -> np.ndarray:
    if len(history) == 0:
        return np.empty(
            0,
            dtype=int,
        )
    cells = np.asarray(history, dtype=int)
    y_coordinates = cells[:, 0]
    trajectory = np.maximum.accumulate(
        y_coordinates
    )

    return trajectory

def compute_basic_morphology(channel, history, weak_path_mask) -> dict:
    channel = np.asarray(channel, dtype=bool)
    weak_path_mask = np.asarray(weak_path_mask, dtype=bool)
    if channel.shape != weak_path_mask.shape:
        raise ValueError(
            "Дерево и маска канала должны иметь одинаковый размер"
        )
    ny, nx = channel.shape
    area = np.count_nonzero(channel)
    if area == 0:
        return {
            "steps": 0,
            "area": 0,
            "max_y": 0,
            "height_fraction": 0.0,
            "x_std_pixels": np.nan,
            "x_std_fraction": np.nan,
            "horizontal_span_pixels": 0,
            "horizontal_span_fraction": 0.0,
            "weak_path_cells": 0,
            "weak_path_fraction": np.nan,
            "mean_vertical_speed": np.nan,
        }
    y_coordinates, x_coordinates = np.where(channel)
    max_y = np.max(y_coordinates)
    height_fraction = max_y / (ny - 1)
    x_std_pixels = np.std(x_coordinates)
    x_std_fraction = x_std_pixels / (nx - 1)
    horizontal_span_pixels = (
            np.max(x_coordinates)
            - np.min(x_coordinates)
    )
    horizontal_span_fraction = (
        horizontal_span_pixels
        / (nx - 1)
    )
    weak_path_cells = int(
        np.count_nonzero(
            channel & weak_path_mask
        )
    )
    weak_path_fraction = (
        weak_path_cells / area
    )
    trajectory = depth_trajectory(
        history
    )
    if len(trajectory) < 2:
        mean_vertical_speed = 0.0
    else:
        mean_vertical_speed = float(
            (
                trajectory[-1]
                - trajectory[0]
            )
            / (len(trajectory) - 1)
        )

    return {
        "steps": len(history),
        "area": area,
        "max_y": max_y,
        "height_fraction": height_fraction,
        "x_std_pixels": x_std_pixels,
        "x_std_fraction": x_std_fraction,
        "horizontal_span_pixels": horizontal_span_pixels,
        "horizontal_span_fraction": horizontal_span_fraction,
        "weak_path_cells": weak_path_cells,
        "weak_path_fraction": weak_path_fraction,
        "mean_vertical_speed": mean_vertical_speed,
    }

def compute_y_metrics(
    channel,
    y_masks,
):
    tree_mask = np.asarray(
        channel,
        dtype=bool,
    )

    if tree_mask.ndim != 2:
        raise ValueError(
            "channel должен быть двумерным массивом"
        )

    required_masks = (
        "trunk",
        "left",
        "right",
    )

    masks = {}

    for name in required_masks:
        if name not in y_masks:
            raise ValueError(
                f"В y_masks отсутствует маска {name!r}"
            )

        mask = np.asarray(
            y_masks[name],
            dtype=bool,
        )

        if mask.shape != tree_mask.shape:
            raise ValueError(
                f"Размер маски {name!r} "
                "не совпадает с размером дерева"
            )

        masks[name] = mask

    trunk_mask = masks["trunk"]
    left_mask = masks["left"]
    right_mask = masks["right"]

    branch_overlap = (
        left_mask
        & right_mask
    )

    trunk_and_junction = (
        trunk_mask
        | branch_overlap
    )

    left_only = (
        left_mask
        & ~trunk_and_junction
    )

    right_only = (
        right_mask
        & ~trunk_and_junction
    )

    full_y_mask = (
        trunk_and_junction
        | left_only
        | right_only
    )

    tree_cells = int(
        np.count_nonzero(tree_mask)
    )

    trunk_cells = int(
        np.count_nonzero(
            tree_mask
            & trunk_and_junction
        )
    )

    left_cells = int(
        np.count_nonzero(
            tree_mask
            & left_only
        )
    )

    right_cells = int(
        np.count_nonzero(
            tree_mask
            & right_only
        )
    )

    y_cells = int(
        np.count_nonzero(
            tree_mask
            & full_y_mask
        )
    )

    outside_y_cells = (
        tree_cells
        - y_cells
    )

    branch_cells = (
        left_cells
        + right_cells
    )

    if tree_cells > 0:
        y_fraction = (
            y_cells
            / tree_cells
        )
    else:
        y_fraction = np.nan

    if branch_cells > 0:
        left_branch_fraction = (
            left_cells
            / branch_cells
        )

        right_branch_fraction = (
            right_cells
            / branch_cells
        )

        branch_preference = (
            left_cells
            - right_cells
        ) / branch_cells
    else:
        left_branch_fraction = np.nan
        right_branch_fraction = np.nan
        branch_preference = np.nan

    return {
        "tree_cells": tree_cells,
        "trunk_cells": trunk_cells,
        "left_cells": left_cells,
        "right_cells": right_cells,
        "branch_cells": branch_cells,
        "y_cells": y_cells,
        "outside_y_cells": outside_y_cells,
        "y_fraction": y_fraction,
        "left_branch_fraction": left_branch_fraction,
        "right_branch_fraction": right_branch_fraction,
        "branch_preference": branch_preference,
        "reached_fork": branch_cells > 0,
    }
