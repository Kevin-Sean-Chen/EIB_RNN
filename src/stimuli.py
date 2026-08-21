"""Create reusable spatial stimuli."""

from __future__ import annotations

import numpy as np
import torch


def moving_dot(
    N: int,
    steps: int,
    dot_size: float,
    drift_rate: float,
    angle: float = 0.0,
    device: str = "cpu",
) -> torch.Tensor:
    """Return one periodic movie of a moving Gaussian dot."""
    x = torch.linspace(0, 1, N, device=device)
    y = torch.linspace(0, 1, N, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    base_dot = torch.exp(
        -0.5 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2) / dot_size**2
    )
    cos_angle = float(np.cos(angle))
    sin_angle = float(np.sin(angle))
    stimulus = torch.zeros((N, N, steps), device=device)

    for step in range(steps):
        shift_x = int(round(drift_rate * step * cos_angle))
        shift_y = int(round(drift_rate * step * sin_angle))
        stimulus[:, :, step] = torch.roll(
            base_dot,
            shifts=(shift_y, shift_x),
            dims=(0, 1),
        )
    return stimulus


def two_moving_dots(
    N: int,
    steps: int,
    dot_size: float,
    drift_rate: float,
    angle: float = 0.0,
    separation: float = 0.1,
    device: str = "cpu",
) -> torch.Tensor:
    """Return one periodic movie of two moving Gaussian dots."""
    x = torch.linspace(0, 1, N, device=device)
    y = torch.linspace(0, 1, N, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    half_separation = separation / 2.0
    first = torch.exp(
        -0.5
        * ((X - (0.5 - half_separation)) ** 2 + (Y - 0.5) ** 2)
        / dot_size**2
    )
    second = torch.exp(
        -0.5
        * ((X - (0.5 + half_separation)) ** 2 + (Y - 0.5) ** 2)
        / dot_size**2
    )
    base = first + second
    cos_angle = float(np.cos(angle))
    sin_angle = float(np.sin(angle))
    stimulus = torch.zeros((N, N, steps), device=device)

    for step in range(steps):
        shift_x = int(round(drift_rate * step * cos_angle))
        shift_y = int(round(drift_rate * step * sin_angle))
        stimulus[:, :, step] = torch.roll(
            base,
            shifts=(shift_y, shift_x),
            dims=(0, 1),
        )
    return stimulus


def drifting_sine(
    N: int,
    steps: int,
    temporal_frequency: float,
    spatial_frequency: float,
    drift_rate: float,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a sine movie with a direction that changes with time."""
    x = torch.linspace(0, spatial_frequency * 2 * np.pi, N, device=device)
    y = torch.linspace(0, spatial_frequency * 2 * np.pi, N, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    stimulus = torch.zeros((N, N, steps), device=device)
    drift_series = torch.zeros(steps, device=device)

    for step in range(steps):
        drift_angle = torch.cos(torch.tensor(drift_rate * step, device=device))
        drift_series[step] = drift_angle
        direction_x = torch.cos(drift_angle)
        direction_y = torch.sin(drift_angle)
        frame = torch.sin(
            direction_x * X
            + direction_y * Y
            + step * temporal_frequency * (2 * np.pi / N)
        )
        stimulus[:, :, step] = (
            (frame - frame.min()) / (frame.max() - frame.min()) * 2 - 1
        )
    return stimulus, drift_series
