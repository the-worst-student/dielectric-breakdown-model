from collections import deque

import numpy as np
from scipy.ndimage import label


FOUR_CONNECTIVITY = np.array(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ],
    dtype=np.uint8,
)

MORPHOLOGY_FEATURES = (
    "area",
    "area_fraction",
    "max_y",
    "height_fraction",
    "depth_fraction",
    "width_fraction",
    "x_std_fraction",
    "aspect_ratio",
    "centroid_offset_fraction",
    "anisotropy",
    "endpoint_count",
    "branching_points",
    "branching_density",
    "mean_branch_length",
    "max_branch_length",
    "main_path_length",
    "sinuosity",
    "side_branch_fraction",
    "fractal_dimension",
    "lacunarity_4",
    "lacunarity_8",
    "lacunarity_16",
    "lacunarity_32",
    "radius_25",
    "radius_50",
    "radius_75",
    "radius_90",
)

def _as_mask(channel):
    mask = np.asarray(channel, dtype=bool)

    if mask.ndim != 2:
        raise ValueError("channel must be a two-dimensional array")

    return mask

def _neighbor_degree(mask):
    degree = np.zeros(mask.shape, dtype=np.uint8)
    degree[1:, :] += mask[:-1, :]
    degree[:-1, :] += mask[1:, :]
    degree[:, 1:] += mask[:, :-1]
    degree[:, :-1] += mask[:, 1:]

    return degree

def _box_masses(mask, box_size):
    ny, nx = mask.shape
    pad_y = (-ny) % box_size
    pad_x = (-nx) % box_size
    padded = np.pad(mask, ((0, pad_y), (0, pad_x)))

    return padded.reshape(
        padded.shape[0] // box_size,
        box_size,
        padded.shape[1] // box_size,
        box_size,
    ).sum(axis=(1, 3))

def box_counting_dimension(channel, box_sizes=(2, 4, 8, 16, 32)):
    mask = _as_mask(channel)

    if np.count_nonzero(mask) < 2:
        return np.nan

    scales = []
    counts = []

    for box_size in box_sizes:
        if box_size <= 0 or box_size > max(mask.shape):
            continue

        count = np.count_nonzero(_box_masses(mask, box_size))

        if count > 1:
            scales.append(1.0 / box_size)
            counts.append(count)

    if len(counts) < 2:
        return np.nan

    return float(
        np.polyfit(
            np.log(scales),
            np.log(counts),
            deg=1,
        )[0]
    )

def lacunarity(channel, box_size):
    mask = _as_mask(channel)
    masses = _box_masses(mask, box_size).astype(float)
    mean_mass = masses.mean()

    if mean_mass == 0.0:
        return np.nan

    return float(np.mean(masses ** 2) / mean_mass ** 2)

def _main_path_metrics(mask):
    component_map, component_count = label(
        mask,
        structure=FOUR_CONNECTIVITY,
    )
    best_length = 0
    best_direct = np.nan

    for component_id in range(1, component_count + 1):
        component = component_map == component_id
        coordinates = np.argwhere(component)
        root_y = coordinates[:, 0].min()
        target_y = coordinates[:, 0].max()
        roots = coordinates[coordinates[:, 0] == root_y]
        targets = coordinates[coordinates[:, 0] == target_y]
        distance = np.full(mask.shape, -1, dtype=np.int32)
        origin_y = np.full(mask.shape, -1, dtype=np.int32)
        origin_x = np.full(mask.shape, -1, dtype=np.int32)
        queue = deque()

        for y, x in roots:
            distance[y, x] = 0
            origin_y[y, x] = y
            origin_x[y, x] = x
            queue.append((int(y), int(x)))

        while queue:
            y, x = queue.popleft()

            for next_y, next_x in (
                (y - 1, x),
                (y + 1, x),
                (y, x - 1),
                (y, x + 1),
            ):
                if not (0 <= next_y < mask.shape[0]):
                    continue
                if not (0 <= next_x < mask.shape[1]):
                    continue
                if not component[next_y, next_x]:
                    continue
                if distance[next_y, next_x] >= 0:
                    continue

                distance[next_y, next_x] = distance[y, x] + 1
                origin_y[next_y, next_x] = origin_y[y, x]
                origin_x[next_y, next_x] = origin_x[y, x]
                queue.append((next_y, next_x))

        for target_y_value, target_x_value in targets:
            path_length = int(distance[target_y_value, target_x_value])

            if path_length < best_length:
                continue

            source_y = origin_y[target_y_value, target_x_value]
            source_x = origin_x[target_y_value, target_x_value]
            direct = np.hypot(
                target_y_value - source_y,
                target_x_value - source_x,
            )

            best_length = path_length
            best_direct = float(direct)

    if best_length == 0 or not np.isfinite(best_direct) or best_direct == 0.0:
        sinuosity = np.nan
    else:
        sinuosity = float(best_length / best_direct)

    return best_length, sinuosity

def _topology_features(mask):
    area = int(np.count_nonzero(mask))
    degree = _neighbor_degree(mask)
    component_map, component_count = label(
        mask,
        structure=FOUR_CONNECTIVITY,
    )
    junction_mask = mask & (degree >= 3)
    _, branching_points = label(
        junction_mask,
        structure=FOUR_CONNECTIVITY,
    )
    all_endpoint_mask = mask & (degree == 1)
    endpoint_mask = all_endpoint_mask.copy()
    for component_id in range(1, component_count + 1):
        component = component_map == component_id
        component_y = np.flatnonzero(component.any(axis=1))

        if component_y.size:
            root_y = component_y[0]
            endpoint_mask[root_y, component[root_y, :]] = False

    endpoint_count = int(np.count_nonzero(endpoint_mask))
    critical_mask = all_endpoint_mask | junction_mask
    segment_mask = mask & ~critical_mask
    segment_map, segment_count = label(
        segment_mask,
        structure=FOUR_CONNECTIVITY,
    )
    segment_sizes = np.bincount(segment_map.ravel())[1:]

    if segment_count:
        segment_lengths = segment_sizes.astype(float) + 1.0
        mean_branch_length = float(segment_lengths.mean())
        max_branch_length = float(segment_lengths.max())
    else:
        mean_branch_length = np.nan
        max_branch_length = np.nan

    main_path_length, sinuosity = _main_path_metrics(mask)
    main_path_cells = min(main_path_length + 1, area)

    return {
        "endpoint_count": endpoint_count,
        "branching_points": int(branching_points),
        "branching_density": float(branching_points / area),
        "mean_branch_length": mean_branch_length,
        "max_branch_length": max_branch_length,
        "main_path_length": int(main_path_length),
        "sinuosity": sinuosity,
        "side_branch_fraction": float(1.0 - main_path_cells / area),
    }

def _geometry_features(mask):
    ny, nx = mask.shape
    y, x = np.where(mask)
    area = len(y)
    min_y = int(y.min())
    max_y = int(y.max())
    min_x = int(x.min())
    max_x = int(x.max())
    depth = max_y - min_y
    width = max_x - min_x
    center_x = (nx - 1) / 2.0
    normalized = np.column_stack(
        (
            y / max(ny - 1, 1),
            x / max(nx - 1, 1),
        )
    )

    if area > 1:
        covariance = np.cov(normalized, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(covariance)
        denominator = eigenvalues.sum()
        anisotropy = (
            float((eigenvalues[-1] - eigenvalues[0]) / denominator)
            if denominator > 0.0 else np.nan
        )
    else:
        anisotropy = np.nan

    return {
        "area": int(area),
        "area_fraction": float(area / (ny * nx)),
        "max_y": max_y,
        "height_fraction": float(max_y / max(ny - 1, 1)),
        "depth_fraction": float(depth / max(ny - 1, 1)),
        "width_fraction": float(width / max(nx - 1, 1)),
        "x_std_fraction": float(np.std(x) / max(nx - 1, 1)),
        "aspect_ratio": float(width / max(depth, 1)),
        "centroid_offset_fraction": float(
            abs(np.mean(x) - center_x) / max(nx - 1, 1)
        ),
        "anisotropy": anisotropy,
    }

def _radial_features(mask, quantiles=(0.25, 0.50, 0.75, 0.90)):
    ny, nx = mask.shape
    y, x = np.where(mask)
    root_y = y.min()
    root_x = np.median(x[y == root_y])
    distances = np.hypot(y - root_y, x - root_x)
    diagonal = np.hypot(ny - 1, nx - 1)
    features = {}

    for quantile in quantiles:
        name = f"radius_{int(round(100 * quantile))}"
        features[name] = float(np.quantile(distances, quantile) / diagonal)

    return features

def compute_morphology_features(
    channel,
    box_sizes=(2, 4, 8, 16, 32),
    lacunarity_sizes=(4, 8, 16, 32),
):
    mask = _as_mask(channel)

    if not np.any(mask):
        raise ValueError("morphology is undefined for an empty channel")

    features = _geometry_features(mask)
    features.update(_topology_features(mask))
    features["fractal_dimension"] = box_counting_dimension(
        mask,
        box_sizes=box_sizes,
    )

    for box_size in lacunarity_sizes:
        features[f"lacunarity_{box_size}"] = lacunarity(
            mask,
            box_size=box_size,
        )

    features.update(_radial_features(mask))

    if tuple(features) != MORPHOLOGY_FEATURES:
        raise RuntimeError("unexpected morphology feature schema")

    return features

