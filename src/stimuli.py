"""Create reusable spatial stimuli."""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import shift
from scipy.signal import convolve2d


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


def rigid_shift_movie(
    N: int,
    steps: int,
    dt: float,
    smoothing_width: float,
    distance: float,
    seed: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the legacy rigid-shift movie and its angle target."""
    rng = np.random.default_rng(seed)
    source = rng.standard_normal((N, N))
    sigma = smoothing_width * N
    coordinates = np.arange(N) - (N - 1) / 2
    X, Y = np.meshgrid(coordinates, coordinates, indexing="ij")
    kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    pattern = convolve2d(source, kernel, mode="same", boundary="wrap")
    pattern -= pattern.mean()
    pattern /= np.max(np.abs(pattern))

    time = np.arange(steps) * dt
    angles = np.sin(time / dt / np.pi / 2) + np.sin(time / dt / np.pi / 4)
    angles = angles / np.max(np.abs(angles)) * np.pi
    movie = np.empty((N, N, steps), dtype=np.float32)
    for index, angle in enumerate(angles):
        shift_x = distance * np.cos(angle)
        shift_y = distance * np.sin(angle)
        movie[:, :, index] = shift(pattern, [shift_y, shift_x], mode="wrap")
    return (
        torch.tensor(movie, dtype=torch.float32, device=device),
        torch.tensor(angles, dtype=torch.float32, device=device),
    )


def fixed_spatial_permutation(movie: torch.Tensor, seed: int) -> torch.Tensor:
    """Apply one fixed pixel permutation to all movie frames."""
    if movie.ndim != 3 or movie.shape[0] != movie.shape[1]:
        raise ValueError("The movie must have shape N by N by time.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pixel_count = movie.shape[0] * movie.shape[1]
    permutation = torch.randperm(pixel_count, generator=generator).to(movie.device)
    flat_movie = movie.reshape(pixel_count, movie.shape[-1])
    return flat_movie[permutation].reshape_as(movie)
