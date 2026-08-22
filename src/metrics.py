"""Reusable metrics for spatial network activity."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, sobel


def spatial_coherence(image: np.ndarray, smoothing: float = 1.0) -> float:
    """Return the mean structure-tensor coherence of one image."""
    gradient_x = sobel(image, axis=1)
    gradient_y = sobel(image, axis=0)
    tensor_xx = gaussian_filter(gradient_x * gradient_x, smoothing)
    tensor_yy = gaussian_filter(gradient_y * gradient_y, smoothing)
    tensor_xy = gaussian_filter(gradient_x * gradient_y, smoothing)
    difference = np.sqrt((tensor_xx - tensor_yy) ** 2 + 4 * tensor_xy**2)
    eigenvalue_1 = 0.5 * (tensor_xx + tensor_yy + difference)
    eigenvalue_2 = 0.5 * (tensor_xx + tensor_yy - difference)
    coherence = (eigenvalue_1 - eigenvalue_2) / (
        eigenvalue_1 + eigenvalue_2 + 1e-12
    )
    return float(np.nanmean(coherence))


def latent_coherence(
    activity: np.ndarray,
    pattern: np.ndarray,
    epsilon: float = 1e-12,
) -> float:
    """Return activity power in one spatial pattern."""
    activity = np.asarray(activity, dtype=float)
    pattern = np.asarray(pattern, dtype=float).reshape(-1)
    if activity.ndim != 2 or activity.shape[0] != pattern.size:
        raise ValueError("Activity must have shape (space, time) and match pattern.")
    latent = pattern @ activity / pattern.size
    latent_power = np.mean(latent**2)
    total_power = np.mean(activity**2)
    return float(np.sqrt(latent_power / (total_power + epsilon)))


def linear_dimension(activity: np.ndarray, variance_threshold: float = 0.9) -> int:
    """Return the PCA dimension that reaches the variance threshold."""
    if not 0 < variance_threshold <= 1:
        raise ValueError("variance_threshold must be in (0, 1].")
    matrix = np.asarray(activity, dtype=float).reshape(-1, activity.shape[-1])
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
    variance = singular_values**2
    total = variance.sum()
    if total == 0:
        return 0
    cumulative = np.cumsum(variance / total)
    return int(np.searchsorted(cumulative, variance_threshold) + 1)


def second_acf_peak(series: np.ndarray) -> tuple[float, np.ndarray]:
    """Return the first positive-lag ACF maximum and the normalized ACF."""
    series = np.asarray(series, dtype=float)
    if series.ndim != 1:
        raise ValueError("series must be one-dimensional.")
    centered = series - series.mean()
    correlation = np.correlate(centered, centered, mode="full")
    acf = correlation[correlation.size // 2 :]
    if acf.size == 0 or acf[0] == 0:
        return float("nan"), acf
    acf = acf / acf[0]
    for index in range(1, len(acf) - 1):
        if acf[index] >= acf[index - 1] and acf[index] >= acf[index + 1]:
            return float(acf[index]), acf
    return float("nan"), acf


def mean_second_acf_peak(
    activity: np.ndarray,
    sample_count: int,
    seed: int | None = None,
) -> tuple[float, np.ndarray]:
    """Return the mean second ACF peak across sampled spatial locations."""
    activity = np.asarray(activity)
    if activity.ndim != 3:
        raise ValueError("activity must have shape (height, width, time).")
    rng = np.random.default_rng(seed)
    flat = activity.reshape(-1, activity.shape[-1])
    count = min(sample_count, flat.shape[0])
    indices = rng.choice(flat.shape[0], size=count, replace=False)
    peaks = [second_acf_peak(flat[index])[0] for index in indices]
    valid_peaks = np.asarray([peak for peak in peaks if not np.isnan(peak)])
    mean_peak = float(np.mean(valid_peaks)) if valid_peaks.size else float("nan")
    return mean_peak, valid_peaks


# These names keep old scripts operational during migration.
def coherence_metric(image: np.ndarray, sigma: float = 1.0) -> float:
    return spatial_coherence(image, smoothing=sigma)


def coherence_chi(
    activity: np.ndarray,
    pattern: np.ndarray,
    eps: float = 1e-12,
) -> float:
    return latent_coherence(activity, pattern, epsilon=eps)


def linear_dimention(activity: np.ndarray, variance_threshold: float = 0.9) -> int:
    return linear_dimension(activity, variance_threshold)


def avg_second_acf_peak(
    data: np.ndarray,
    p: int,
    rng: int | np.random.Generator | None = None,
) -> tuple[float, np.ndarray]:
    seed = rng if isinstance(rng, int) else None
    return mean_second_acf_peak(data, sample_count=p, seed=seed)


def second_acf_peak_latent(series: np.ndarray) -> tuple[float, np.ndarray]:
    return second_acf_peak(series)
