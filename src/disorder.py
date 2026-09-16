import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import ndtr, gamma

def generate_breakdown_strength(
    ny,
    nx,
    correlation_length,
    weibull_shape,
    seed,
):
    rng = np.random.default_rng(seed)
    noise = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(ny, nx),
    )
    filter_sigma = correlation_length / np.sqrt(2.0)
    z = gaussian_filter(
        noise,
        sigma=filter_sigma,
        mode="reflect",
    )
    z = (z - np.mean(z)) / np.std(z)
    u = ndtr(z)
    eps = np.finfo(float).eps
    u = np.clip(u, eps, 1.0 - eps)
    scale = 1.0 / gamma(
        1.0 + 1.0 / weibull_shape
    )
    strength = scale * (
        -np.log1p(-u)
    ) ** (1.0 / weibull_shape)

    return strength