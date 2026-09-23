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

def build_weak_path(
        config: SimulationConfig
) -> tuple[np.ndarray, np.ndarray]:
    path = config.weak_path
    factor_map = np.ones(
        (config.ny, config.nx),
        dtype=float,
    )
    empty_mask = np.zeros(
        (config.ny, config.nx),
        dtype=bool,
    )
    if path.kind == "none":
        return factor_map, empty_mask
    if not 0.0 < path.factor <= 1.0:
        raise ValueError(
            "factor должен находиться в интервале (0, 1]"
        )
    if path.kind == "straight":
        mask = make_segment_mask(
            config=config,
            start=path.start,
            end=path.end,
            width=path.width,
        )

        factor_map[mask] = path.factor

        return factor_map, mask

    raise ValueError(
        f"Неизвестный тип слабого пути: {path.kind}"
    )