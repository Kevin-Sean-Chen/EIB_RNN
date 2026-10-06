"""Analyze spontaneous activity for Figure 1."""

from __future__ import annotations

import numpy as np

from src.local import LocalConfig


def pca_variance_curve(activity: np.ndarray) -> np.ndarray:
    """Return normalized PCA variance in descending order."""
    matrix = np.asarray(activity, dtype=float).reshape(-1, activity.shape[-1])
    matrix = matrix - matrix.mean(axis=1, keepdims=True)
    singular_values = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
    variance = singular_values**2
    total = variance.sum()
    if total <= np.finfo(float).eps:
        return np.zeros_like(variance)
    return variance / total


def participation_dimension(activity: np.ndarray) -> float:
    """Return the PCA participation dimension of time-centered activity."""
    variance = pca_variance_curve(activity)
    denominator = np.sum(variance**2)
    if denominator <= np.finfo(float).eps:
        return 0.0
    return float(1.0 / denominator)


def _radial_shells(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return periodic radius values and integer radial-shell labels."""
    axis = np.minimum(np.arange(size), size - np.arange(size))
    radius = np.sqrt(axis[:, None] ** 2 + axis[None, :] ** 2)
    labels = np.floor(radius + 0.5).astype(int)
    return radius, labels


def radial_spatial_correlation(activity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the radial spatial autocorrelation averaged across time."""
    values = np.asarray(activity, dtype=float)
    if values.ndim != 3 or values.shape[0] != values.shape[1]:
        raise ValueError("Activity must have shape (size, size, time).")
    centered = values - values.mean(axis=(0, 1), keepdims=True)
    power = np.abs(np.fft.fft2(centered, axes=(0, 1))) ** 2
    correlation = np.fft.ifft2(power.mean(axis=2)).real
    if correlation[0, 0] <= np.finfo(float).eps:
        count = values.shape[0] // 2 + 1
        return np.arange(count, dtype=float), np.full(count, np.nan)
    correlation /= correlation[0, 0]
    _, labels = _radial_shells(values.shape[0])
    shell_count = values.shape[0] // 2 + 1
    curve = np.array(
        [correlation[labels == index].mean() for index in range(shell_count)]
    )
    distance = np.arange(shell_count, dtype=float) / values.shape[0]
    return distance, curve


def temporal_autocorrelation(
    activity: np.ndarray,
    sample_interval: float,
    max_lag_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean normalized sitewise temporal autocorrelation."""
    values = np.asarray(activity, dtype=float).reshape(-1, activity.shape[-1])
    values = values - values.mean(axis=1, keepdims=True)
    sample_count = values.shape[1]
    if max_lag_samples is None:
        max_lag_samples = sample_count // 4
    max_lag_samples = min(max_lag_samples, sample_count - 1)
    transform = np.fft.rfft(values, n=2 * sample_count, axis=1)
    covariance = np.fft.irfft(transform * transform.conj(), axis=1)[:, :sample_count]
    covariance /= np.arange(sample_count, 0, -1)[None, :]
    variance = covariance[:, 0]
    valid = variance > np.finfo(float).eps
    lag = np.arange(max_lag_samples + 1) * sample_interval
    if not np.any(valid):
        return lag, np.full(lag.shape, np.nan)
    normalized = covariance[valid] / variance[valid, None]
    return lag, normalized[:, : max_lag_samples + 1].mean(axis=0)


def radial_spatial_spectrum(activity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized radial spatial power outside the uniform mode."""
    values = np.asarray(activity, dtype=float)
    if values.ndim != 3 or values.shape[0] != values.shape[1]:
        raise ValueError("Activity must have shape (size, size, time).")
    centered = values - values.mean(axis=(0, 1), keepdims=True)
    mean_power = np.mean(np.abs(np.fft.fft2(centered, axes=(0, 1))) ** 2, axis=2)
    frequency = np.fft.fftfreq(values.shape[0]) * values.shape[0]
    radius = np.sqrt(frequency[:, None] ** 2 + frequency[None, :] ** 2)
    labels = np.floor(radius + 0.5).astype(int)
    wave_number = np.arange(1, labels.max() + 1)
    power = np.array([mean_power[labels == index].sum() for index in wave_number])
    total = power.sum()
    if total > np.finfo(float).eps:
        power /= total
    else:
        power[:] = np.nan
    return wave_number.astype(float), power


def temporal_spectrum(
    activity: np.ndarray,
    sample_interval: float,
    window_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized site-averaged temporal power across time windows."""
    values = np.asarray(activity, dtype=float).reshape(-1, activity.shape[-1])
    if window_samples <= 1 or window_samples > values.shape[1]:
        raise ValueError("window_samples must be between 2 and the record length.")
    window_count = values.shape[1] // window_samples
    trimmed = values[:, : window_count * window_samples]
    windows = trimmed.reshape(values.shape[0], window_count, window_samples)
    windows = windows - windows.mean(axis=2, keepdims=True)
    taper = np.hanning(window_samples)
    transform = np.fft.rfft(windows * taper[None, None, :], axis=2)
    power = np.mean(np.abs(transform) ** 2, axis=(0, 1))[1:]
    frequency = np.fft.rfftfreq(window_samples, d=sample_interval)[1:]
    total = power.sum()
    if total > np.finfo(float).eps:
        power /= total
    else:
        power[:] = np.nan
    return frequency, power


def _periodic_kernel(size: int, width: float) -> np.ndarray:
    """Return the periodic Gaussian kernel used by the local model."""
    dx = 1.0 / size
    wraps = np.arange(-int(np.ceil(10 * width)), int(np.ceil(10 * width)) + 1)
    axis = np.arange(-(size - 1) // 2, (size - 1) // 2 + 1)
    kernel_1d = np.sum(
        dx
        * (2 * np.pi * width**2) ** -0.5
        * np.exp(-0.5 * (dx * (axis[:, None] + wraps)) ** 2 / width**2),
        axis=1,
    )
    return np.outer(kernel_1d, kernel_1d)


def _periodic_convolution(activity: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply one periodic spatial convolution to all recorded frames."""
    kernel_transform = np.fft.fft2(np.fft.ifftshift(kernel))[:, :, None]
    activity_transform = np.fft.fft2(activity, axes=(0, 1))
    return np.fft.ifft2(activity_transform * kernel_transform, axes=(0, 1)).real


def current_balance(
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    config: LocalConfig,
) -> dict[str, float]:
    """Return mean E-cell current components and their local cancellation."""
    excitatory_local = _periodic_convolution(
        np.asarray(excitatory, dtype=float),
        _periodic_kernel(config.N, config.sigma_e),
    )
    inhibitory_local = _periodic_convolution(
        np.asarray(inhibitory, dtype=float),
        _periodic_kernel(config.N, config.sigma_i),
    )
    external = np.full_like(excitatory_local, config.u_e)
    excitatory_current = config.J_ee * excitatory_local
    inhibitory_current = config.J_ei * inhibitory_local
    net = external + excitatory_current + inhibitory_current
    denominator = (
        np.abs(external) + np.abs(excitatory_current) + np.abs(inhibitory_current)
    )
    cancellation = np.divide(
        np.abs(net),
        denominator,
        out=np.full_like(net, np.nan),
        where=denominator > 0.0,
    )
    active = net > 0.0
    active_values = cancellation[active & np.isfinite(cancellation)]
    inactive_values = cancellation[(~active) & np.isfinite(cancellation)]
    return {
        "external": float(external.mean()),
        "excitatory": float(excitatory_current.mean()),
        "inhibitory": float(inhibitory_current.mean()),
        "net": float(net.mean()),
        "cancellation": float(np.nanmean(cancellation)),
        "active_cancellation": (
            float(active_values.mean()) if active_values.size else float("nan")
        ),
        "inactive_cancellation": (
            float(inactive_values.mean()) if inactive_values.size else float("nan")
        ),
    }
