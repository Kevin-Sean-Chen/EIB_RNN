"""Measure moving-dot tracking in a driven spatial E/I network."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.signal import correlate
import torch

from src.driven import DrivenDotConfig, initial_rates, simulate_driven
from src.stimuli import moving_dot


@dataclass
class DrivenDotScanResult:
    """Store all outputs from one driven-dot scan."""

    K_values: np.ndarray
    input_com: np.ndarray
    lags: np.ndarray
    response_com: np.ndarray
    cross_correlation: np.ndarray
    peak_lags: np.ndarray
    peak_heights: np.ndarray
    example_activity: np.ndarray
    stimulus: np.ndarray


def normalized_center_of_mass(activity: np.ndarray | torch.Tensor) -> np.ndarray:
    """Return the normalized center of mass along the tracked grid axis."""
    tensor = torch.as_tensor(activity, dtype=torch.float32)
    coordinates = torch.linspace(0, 1, tensor.shape[1], dtype=tensor.dtype).view(
        1, tensor.shape[1], 1
    )
    center = (tensor * coordinates).sum(dim=(0, 1)) / (
        tensor.sum(dim=(0, 1)) + 1e-8
    )
    center = 2 * (center - center.min()) / (center.max() - center.min() + 1e-8) - 1
    return center.numpy().astype(float)


def overlap_cross_correlation(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return overlap-normalized cross-correlation and integer lags."""
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    raw = correlate(first_centered, second_centered, mode="full")
    lags = np.arange(-len(first) + 1, len(first))
    return raw / (len(first) - np.abs(lags)), lags


def run_tracking_scan(
    config: DrivenDotConfig,
    K_values: list[float],
    repetitions: int,
    max_lag: int,
) -> DrivenDotScanResult:
    """Run moving-dot tracking across network strengths."""
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")
    if not K_values or any(value <= 0 for value in K_values):
        raise ValueError("K_values must contain positive values.")
    if max_lag < 0:
        raise ValueError("max_lag must not be negative.")

    record_count = config.record_steps // config.steps_per_record
    stimulus = config.input_amplitude * moving_dot(
        config.N,
        record_count,
        config.dot_size,
        config.drift_rate,
        config.angle,
        config.device,
    )
    input_com = normalized_center_of_mass(stimulus.cpu())
    _, lags = overlap_cross_correlation(input_com, input_com)
    lag_window = np.flatnonzero((lags >= -max_lag) & (lags <= max_lag))
    rng = np.random.RandomState(config.seed)

    peak_lags = np.zeros((len(K_values), repetitions))
    peak_heights = np.zeros_like(peak_lags)
    first_response = []
    first_correlation = []
    first_activity = []

    for K_index, K in enumerate(K_values):
        run_config = replace(config, K=float(K))
        for repetition in range(repetitions):
            re0, ri0 = initial_rates(run_config, rng)
            re_all, _ = simulate_driven(run_config, stimulus, re0, ri0)
            response_com = normalized_center_of_mass(re_all)
            correlation, _ = overlap_cross_correlation(input_com, response_com)
            peak_index = lag_window[np.argmax(correlation[lag_window])]
            peak_lags[K_index, repetition] = lags[peak_index]
            peak_heights[K_index, repetition] = correlation[peak_index]
            if repetition == 0:
                first_response.append(response_com)
                first_correlation.append(correlation)
                first_activity.append(re_all)

    return DrivenDotScanResult(
        K_values=np.asarray(K_values, dtype=float),
        input_com=input_com,
        lags=lags,
        response_com=np.asarray(first_response),
        cross_correlation=np.asarray(first_correlation),
        peak_lags=peak_lags,
        peak_heights=peak_heights,
        example_activity=np.asarray(first_activity),
        stimulus=stimulus.cpu().numpy(),
    )


def scan_result_arrays(result: DrivenDotScanResult) -> dict[str, np.ndarray]:
    """Return driven-dot scan results as named arrays."""
    return {
        "K_values": result.K_values,
        "input_com": result.input_com,
        "lags": result.lags,
        "response_com": result.response_com,
        "cross_correlation": result.cross_correlation,
        "peak_lags": result.peak_lags,
        "peak_heights": result.peak_heights,
        "example_activity": result.example_activity,
        "stimulus": result.stimulus,
    }


def scan_metric_rows(result: DrivenDotScanResult) -> list[dict[str, float]]:
    """Return one summary row for each K value."""
    rows = []
    for index, K in enumerate(result.K_values):
        rows.append(
            {
                "K": K,
                "peak_lag_mean": result.peak_lags[index].mean(),
                "peak_lag_std": result.peak_lags[index].std(),
                "peak_height_mean": result.peak_heights[index].mean(),
                "peak_height_std": result.peak_heights[index].std(),
            }
        )
    return rows
