"""Analyze spectra of local and low-rank spatial connectivity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import eigs

from src.disorder import gabor_pattern


SPECTRAL_SECTIONS = {
    "network": ("N", "sigma_e", "sigma_i"),
    "structure": (
        "strength", "frequency", "angle", "aspect_ratio", "row_balance",
    ),
    "scan": ("K_values", "phase_values"),
    "analysis": ("leading_eigenvalues",),
}


@dataclass
class SpectralConfig:
    """Define one connectivity-spectrum scan."""

    N: int = 31
    sigma_e: float = 0.05
    sigma_i: float = 0.05 * np.sqrt(2)
    strength: float = 0.5
    frequency: float = 5.0
    angle: float = 30.0
    aspect_ratio: float = 0.1
    row_balance: float = 0.0
    K_values: list[float] | None = None
    phase_values: list[float] | None = None
    leading_eigenvalues: int = 500

    def __post_init__(self) -> None:
        if self.N <= 0 or self.N % 2 != 1:
            raise ValueError("N must be a positive odd integer.")
        if self.sigma_e <= 0 or self.sigma_i <= 0:
            raise ValueError("Connection widths must be positive.")
        if self.K_values is None:
            self.K_values = [10, 100, 1000, 10000, 100000]
        if self.phase_values is None:
            self.phase_values = [0, 0.2 * np.pi, 0.4 * np.pi, 0.6 * np.pi, 0.8 * np.pi]
        if not self.K_values or any(value <= 0 for value in self.K_values):
            raise ValueError("K_values must contain positive values.")
        if not self.phase_values:
            raise ValueError("phase_values must not be empty.")
        dimension = 2 * self.N**2
        if self.leading_eigenvalues <= 0 or self.leading_eigenvalues >= dimension - 1:
            raise ValueError("leading_eigenvalues must be smaller than matrix dimension minus one.")


@dataclass
class SpectralResult:
    """Store one spectral scan."""

    K_values: np.ndarray
    phase_values: np.ndarray
    eigenvalues: np.ndarray
    leading_real: np.ndarray
    spectral_abscissa: np.ndarray
    stable: np.ndarray
    left_patterns: np.ndarray
    right_patterns: np.ndarray


def legacy_gaussian_kernel(N: int, sigma: float) -> np.ndarray:
    """Return the periodic Gaussian kernel used by the legacy scripts."""
    dx = 1.0 / N
    wraps = np.arange(-int(np.ceil(10 * sigma)), int(np.ceil(10 * sigma)) + 1)
    axis = np.arange(-(N - 1) // 2, (N - 1) // 2 + 1)
    one_d = np.sum(
        dx * (2 * np.pi * sigma**2) ** -0.5
        * np.exp(-0.5 * (dx * (axis[:, None] + wraps)) ** 2 / sigma**2),
        axis=1,
    )
    return np.outer(one_d, one_d)


def circular_convolution_matrix(kernel: np.ndarray) -> np.ndarray:
    """Return the dense operator for one centered periodic kernel."""
    N = kernel.shape[0]
    if kernel.shape != (N, N) or N % 2 != 1:
        raise ValueError("kernel must be an odd square matrix.")
    center = N // 2
    operator = np.empty((N**2, N**2), dtype=float)
    for source in range(N**2):
        row, column = divmod(source, N)
        response = np.roll(kernel, (row - center, column - center), axis=(0, 1))
        operator[:, source] = response.reshape(-1)
    return operator


def gabor_rank_one(config: SpectralConfig, phase: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the left and right Gabor vectors for one phase."""
    left = gabor_pattern(
        config.N,
        config.frequency,
        config.angle,
        0.0,
        config.aspect_ratio,
    ).reshape(-1)
    right = gabor_pattern(
        config.N,
        config.frequency,
        config.angle,
        phase,
        config.aspect_ratio,
    ).reshape(-1)
    return left, right


def legacy_block_operator(
    local_e: np.ndarray,
    local_i: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    K: float,
    strength: float,
    row_balance: float = 0.0,
) -> np.ndarray:
    """Return the exact two-population operator used by scan_spectral.py."""
    N = int(round(np.sqrt(left.size)))
    disorder = (
        strength * np.outer(left, right) - row_balance * np.outer(left, left)
    ) / N
    excitatory_path = local_e + disorder
    block = np.block(
        [[excitatory_path, local_i], [excitatory_path, local_i]]
    )
    return np.sqrt(K) * block - np.eye(block.shape[0])


def leading_spectrum(operator: np.ndarray, count: int) -> np.ndarray:
    """Return eigenvalues with the largest magnitude."""
    values = eigs(operator, k=count, which="LM", return_eigenvectors=False)
    order = np.argsort(np.abs(values))[::-1]
    return values[order]


def run_spectral_scan(config: SpectralConfig) -> SpectralResult:
    """Scan the legacy connectivity spectrum across K and phase."""
    local_e = circular_convolution_matrix(legacy_gaussian_kernel(config.N, config.sigma_e))
    local_i = circular_convolution_matrix(legacy_gaussian_kernel(config.N, config.sigma_i))
    shape = (len(config.K_values), len(config.phase_values), config.leading_eigenvalues)
    eigenvalues = np.empty(shape, dtype=complex)
    leading_real = np.empty(shape[:2])
    spectral_abscissa = np.empty(shape[:2])
    patterns_left = np.empty((len(config.phase_values), config.N**2))
    patterns_right = np.empty_like(patterns_left)

    for phase_index, phase in enumerate(config.phase_values):
        left, right = gabor_rank_one(config, phase)
        patterns_left[phase_index] = left
        patterns_right[phase_index] = right
        for K_index, K in enumerate(config.K_values):
            print(f"Analyze K={K:g}, phase={phase / np.pi:.2f} pi")
            operator = legacy_block_operator(
                local_e,
                local_i,
                left,
                right,
                K,
                config.strength,
                config.row_balance,
            )
            values = leading_spectrum(operator, config.leading_eigenvalues)
            eigenvalues[K_index, phase_index] = values
            leading_real[K_index, phase_index] = values[0].real
            spectral_abscissa[K_index, phase_index] = np.max(values.real)

    return SpectralResult(
        K_values=np.asarray(config.K_values),
        phase_values=np.asarray(config.phase_values),
        eigenvalues=eigenvalues,
        leading_real=leading_real,
        spectral_abscissa=spectral_abscissa,
        stable=spectral_abscissa < 0,
        left_patterns=patterns_left,
        right_patterns=patterns_right,
    )
