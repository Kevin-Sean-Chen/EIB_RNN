"""Measure directed waves on a periodic two-dimensional sheet."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DirectedWaveMetrics:
    """Store signed speed and directional coherence."""

    signed_speed: float
    directional_coherence: float


def phase_correlation_shift(first: np.ndarray, second: np.ndarray) -> tuple[int, int]:
    """Return the periodic integer shift from the first frame to the second."""
    first_fft = np.fft.fft2(first)
    second_fft = np.fft.fft2(second)
    cross_power = second_fft * np.conj(first_fft)
    magnitude = np.abs(cross_power)
    normalized = np.divide(
        cross_power,
        magnitude,
        out=np.zeros_like(cross_power),
        where=magnitude > np.finfo(float).eps,
    )
    correlation = np.abs(np.fft.ifft2(normalized))
    row, column = np.unravel_index(np.argmax(correlation), correlation.shape)
    height, width = first.shape
    row_shift = int(row if row < height // 2 else row - height)
    column_shift = int(column if column < width // 2 else column - width)
    return row_shift, column_shift


def directed_wave_metrics(
    activity: np.ndarray,
    sample_interval: float,
    lag_steps: int,
    expected_direction: tuple[float, float],
) -> DirectedWaveMetrics:
    """Measure propagation along one direction and consistency of motion."""
    activity = np.asarray(activity, dtype=float)
    if activity.ndim != 3 or activity.shape[2] <= lag_steps:
        raise ValueError("Activity must contain enough two-dimensional frames.")
    if sample_interval <= 0 or lag_steps <= 0:
        raise ValueError("Sampling values must be positive.")
    direction = np.asarray(expected_direction, dtype=float)
    direction_norm = np.linalg.norm(direction)
    if direction.shape != (2,) or direction_norm <= np.finfo(float).eps:
        raise ValueError("Expected direction must be a nonzero two-dimensional vector.")
    direction /= direction_norm

    size_y, size_x, frame_count = activity.shape
    velocities = []
    elapsed = sample_interval * lag_steps
    for first_index in range(frame_count - lag_steps):
        row_shift, column_shift = phase_correlation_shift(
            activity[:, :, first_index],
            activity[:, :, first_index + lag_steps],
        )
        velocities.append(
            np.array([column_shift / size_x, row_shift / size_y]) / elapsed
        )
    velocity = np.asarray(velocities)
    speed = np.linalg.norm(velocity, axis=1)
    mean_speed = float(np.mean(speed))
    if mean_speed <= np.finfo(float).eps:
        return DirectedWaveMetrics(0.0, 0.0)
    mean_velocity = velocity.mean(axis=0)
    return DirectedWaveMetrics(
        signed_speed=float(mean_velocity @ direction),
        directional_coherence=float(np.clip(np.linalg.norm(mean_velocity) / mean_speed, 0.0, 1.0)),
    )
