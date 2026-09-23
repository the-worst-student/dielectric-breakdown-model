import numpy as np

from .config import SimulationConfig

def make_segment_mask(
    config: SimulationConfig,
    start: tuple[float, float],
    end: tuple[float, float],
    width: float,
) -> np.ndarray:
    x = np.linspace(0.0, config.lx, config.nx)
    y = np.linspace(0.0, config.ly, config.ny)
    xx, yy = np.meshgrid(x, y)

    x0 = start[0] * config.lx
    y0 = start[1] * config.ly
    x1 = end[0] * config.lx
    y1 = end[1] * config.ly
    vx = x1 - x0
    vy = y1 - y0
    length_squared = vx ** 2 + vy ** 2
    if length_squared == 0.0:
        raise ValueError(
            "Начало и конец канала должны различаться"
        )
    projection = (
                 (xx - x0) * vx
                 + (yy - y0) * vy
                 ) / length_squared
    projection = np.clip(
        projection,
        0.0,
        1.0,
    )
    closest_x = x0 + projection * vx
    closest_y = y0 + projection * vy
    distance = np.sqrt(
        (xx - closest_x) ** 2
        + (yy - closest_y) ** 2
    )
    physical_width = (
            width * min(config.lx, config.ly)
    )
    mask = distance <= physical_width / 2.0
    return mask

def make_y_path_masks(config):
    weak_path = config.weak_path
    trunk_mask = make_segment_mask(
        config=config,
        start=weak_path.start,
        end=weak_path.branch_point,
        width=weak_path.width,
    )
    left_mask = make_segment_mask(
        config=config,
        start=weak_path.branch_point,
        end=weak_path.left_end,
        width=weak_path.width,
    )
    right_mask = make_segment_mask(
        config=config,
        start=weak_path.branch_point,
        end=weak_path.right_end,
        width=weak_path.width,
    )
    full_mask = trunk_mask | left_mask | right_mask
    return {
        "trunk": trunk_mask,
        "left": left_mask,
        "right": right_mask,
        "all": full_mask,
    }

def build_weak_path(config):
    shape = (config.ny, config.nx)

    factor_map = np.ones(
        shape,
        dtype=float,
    )

    empty_mask = np.zeros(
        shape,
        dtype=bool,
    )

    weak_path = config.weak_path
    kind = weak_path.kind.lower()

    if kind == "none":
        return factor_map, empty_mask

    if kind == "straight":
        if not 0.0 < weak_path.factor <= 1.0:
            raise ValueError(
                "factor должен удовлетворять условию "
                "0 < factor <= 1"
            )

        straight_mask = make_segment_mask(
            config=config,
            start=weak_path.start,
            end=weak_path.end,
            width=weak_path.width,
        )

        factor_map[straight_mask] = weak_path.factor

        return factor_map, straight_mask

    if kind in {"y", "y_branch"}:
        region_factors = {
            "trunk": weak_path.factor,
            "left": weak_path.left_factor,
            "right": weak_path.right_factor,
        }

        for region_name, factor in region_factors.items():
            if not 0.0 < factor <= 1.0:
                raise ValueError(
                    f"Коэффициент области {region_name} "
                    "должен удовлетворять условию "
                    "0 < factor <= 1"
                )

        y_masks = make_y_path_masks(config)

        trunk_mask = y_masks["trunk"]
        left_mask = y_masks["left"]
        right_mask = y_masks["right"]

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

        factor_map[trunk_and_junction] = (
            weak_path.factor
        )

        factor_map[left_only] = (
            weak_path.left_factor
        )

        factor_map[right_only] = (
            weak_path.right_factor
        )

        return factor_map, full_y_mask

    raise ValueError(
        "Неизвестный тип слабого канала: "
        f"{weak_path.kind!r}"
    )