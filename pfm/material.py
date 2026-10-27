from math import gamma

import numpy as np
from scipy.special import log_ndtr


class MaterialField:
    def __init__(self, *, mean, correlation_length, distribution="weibull", weibull_shape=4.0, seed=42, n_modes=1024, basis_seed=0):
        parameters = np.array(
            [mean, correlation_length, weibull_shape],
            dtype=float,
        )

        if (
            not np.all(np.isfinite(parameters))
            or np.any(parameters <= 0.0)
        ):
            raise ValueError(
                "mean, correlation_length и weibull_shape "
                "должны быть конечными и положительными."
            )

        if distribution not in {"constant", "weibull"}:
            raise ValueError(
                "distribution: 'constant' или 'weibull'."
            )

        if (
            not isinstance(n_modes, (int, np.integer))
            or n_modes < 1
        ):
            raise ValueError(
                "n_modes должен быть положительным целым числом."
            )

        self.mean = float(mean)
        self.correlation_length = float(correlation_length)
        self.distribution = distribution
        self.weibull_shape = float(weibull_shape)
        self.n_modes = int(n_modes)

        self.scale = self.mean / gamma(
            1.0 + 1.0 / self.weibull_shape
        )

        basis_rng = np.random.default_rng(basis_seed)

        self.wave_vectors = basis_rng.normal(
            size=(self.n_modes, 2)
        ) * (
            np.sqrt(2.0) / self.correlation_length
        )

        rng = np.random.default_rng(seed)

        self.cos_weights = rng.standard_normal(self.n_modes)
        self.sin_weights = rng.standard_normal(self.n_modes)

    def gaussian(self, x):
        x = np.asarray(x, dtype=np.float64)

        if x.ndim != 2 or x.shape[0] < 2:
            raise ValueError(
                "Ожидаются координаты формы (2, N) или (3, N)."
            )

        values = np.empty(
            x.shape[1],
            dtype=np.float64,
        )

        for start in range(0, x.shape[1], 512):
            stop = min(start + 512, x.shape[1])

            phases = (
                self.wave_vectors
                @ x[:2, start:stop]
            )

            values[start:stop] = (
                self.cos_weights @ np.cos(phases)
                + self.sin_weights @ np.sin(phases)
            ) / np.sqrt(self.n_modes)

        return values

    def __call__(self, x):
        if self.distribution == "constant":
            return np.full(
                x.shape[1],
                self.mean,
                dtype=np.float64,
            )

        z = self.gaussian(x)

        exponential = -log_ndtr(-z)

        values = (
            self.scale
            * exponential**(1.0 / self.weibull_shape)
        )

        if (
            not np.all(np.isfinite(values))
            or np.any(values <= 0.0)
        ):
            raise FloatingPointError(
                "Получены неположительные или "
                "неконечные значения материала."
            )

        return values