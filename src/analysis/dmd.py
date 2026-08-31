"""Analyze spatial activity with dynamic mode decomposition."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


DMD_SECTIONS = {
    "analysis": (
        "rank", "shown_modes", "subtract_mean", "lag", "dx", "dy",
    ),
}


@dataclass
class DMDConfig:
    """Define one spatial DMD analysis."""

    rank: int = 50
    shown_modes: int = 6
    subtract_mean: bool = True
    lag: int = 1
    dx: float = 1.0
    dy: float = 1.0

    def __post_init__(self) -> None:
        if self.rank <= 0 or self.shown_modes <= 0:
            raise ValueError("DMD ranks must be positive.")
        if self.lag <= 0:
            raise ValueError("DMD lag must be positive.")
        if self.dx <= 0 or self.dy <= 0:
            raise ValueError("Spatial grid steps must be positive.")


@dataclass
class DMDResult:
    """Store one spatial DMD result."""

    eigenvalues: np.ndarray
    modes: np.ndarray
    growth_rates: np.ndarray
    angular_frequencies: np.ndarray
    dominant_wavenumbers: np.ndarray
    singular_values: np.ndarray
    rank_values: np.ndarray
    rank_errors: np.ndarray


def spatial_dmd(activity: np.ndarray, sample_dt: float, config: DMDConfig) -> DMDResult:
    """Return truncated DMD modes for activity with shape N by N by time."""
    values = np.asarray(activity, dtype=float)
    if values.ndim != 3 or values.shape[0] != values.shape[1]:
        raise ValueError("Activity must have shape N by N by time.")
    if not np.isfinite(values).all():
        raise ValueError("Activity must contain finite values.")
    if sample_dt <= 0 or values.shape[-1] <= config.lag:
        raise ValueError("DMD sampling values are invalid.")

    if config.subtract_mean:
        values = values - values.mean(axis=-1, keepdims=True)
    size = values.shape[0]
    snapshots = values.reshape(size**2, values.shape[-1])
    first = snapshots[:, :-config.lag]
    second = snapshots[:, config.lag:]
    left, singular, right_h = np.linalg.svd(first, full_matrices=False)
    tolerance = np.finfo(float).eps * max(first.shape) * singular[0]
    available = int(np.sum(singular > tolerance))
    rank = min(config.rank, available)
    if rank == 0:
        raise ValueError("Activity has no nonzero dynamic modes.")
    left = left[:, :rank]
    singular = singular[:rank]
    right = right_h.conj().T[:, :rank]

    reduced = left.conj().T @ second @ right @ np.diag(1.0 / singular)
    eigenvalues, eigenvectors = np.linalg.eig(reduced)
    modes = second @ right @ np.diag(1.0 / singular) @ eigenvectors
    interval = config.lag * sample_dt
    continuous = np.log(eigenvalues.astype(complex)) / interval

    kx = 2 * np.pi * np.fft.fftfreq(size, d=config.dx)
    ky = 2 * np.pi * np.fft.fftfreq(size, d=config.dy)
    grid_x, grid_y = np.meshgrid(kx, ky, indexing="ij")
    magnitude = np.sqrt(grid_x**2 + grid_y**2)
    mode_images = modes.T.reshape(rank, size, size)
    dominant = np.zeros(rank)
    for index, mode in enumerate(mode_images):
        power = np.abs(np.fft.fft2(mode)) ** 2
        dominant[index] = np.sum(magnitude * power) / (np.sum(power) + 1e-12)

    order = np.argsort(np.abs(continuous.imag))[::-1]
    total_norm_squared = np.linalg.norm(second) ** 2
    rank_values = np.arange(1, rank + 1)
    rank_errors = np.zeros(rank)
    for index, test_rank in enumerate(rank_values):
        projected_norm_squared = np.linalg.norm(second @ right[:, :test_rank]) ** 2
        residual_squared = max(total_norm_squared - projected_norm_squared, 0.0)
        rank_errors[index] = np.sqrt(residual_squared / (total_norm_squared + 1e-24))

    return DMDResult(
        eigenvalues=eigenvalues[order],
        modes=mode_images[order],
        growth_rates=continuous.real[order],
        angular_frequencies=continuous.imag[order],
        dominant_wavenumbers=dominant[order],
        singular_values=singular,
        rank_values=rank_values,
        rank_errors=rank_errors,
    )
