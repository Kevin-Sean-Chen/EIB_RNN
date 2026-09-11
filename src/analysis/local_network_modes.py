"""Analyze the transition from local spatial modes to non-local network modes.

The model is a two-population E/I rate network on a periodic 2D grid. Local
connections use Gaussian convolution. A random low-rank matrix adds non-local
connections to the excitatory pathway. The scan compares how well periodic
Laplacian geometric modes and full-network modes reconstruct spontaneous
activity.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


MODE_SCAN_SECTIONS = {
    "network": ("N", "K", "rank", "sigma_e", "sigma_i", "u_e", "u_i"),
    "simulation": (
        "seed",
        "n_seeds",
        "dt",
        "tau_e",
        "tau_i",
        "init_steps",
        "record_steps",
        "sample_every",
        "max_rate",
    ),
    "scan": ("strengths", "rho_f_values"),
    "analysis": ("auc_modes", "plot_modes"),
}


@dataclass
class ModeScanConfig:
    """Define one local-to-network mode scan."""

    N: int = 15
    K: float = 100.0
    strengths: list[float] | None = None
    rho_f_values: list[float] | None = None
    rank: int = 4
    seed: int = 7
    n_seeds: int = 3
    dt: float = 0.0001
    tau_e: float = 0.01
    tau_i: float = 0.01
    u_e: float = 10.0
    u_i: float = 0.0
    sigma_e: float = 0.05
    sigma_i: float = 0.05 * np.sqrt(2)
    init_steps: int = 1000
    record_steps: int = 2000
    sample_every: int = 5
    auc_modes: int = 40
    plot_modes: int = 80
    max_rate: float = 1e6

    def __post_init__(self) -> None:
        if self.rho_f_values is None:
            self.rho_f_values = [0, 0.05, 0.1, 0.2, 0.5, 1, 2, 4, 8]
        if self.N <= 0 or self.N % 2 != 1:
            raise ValueError("N must be a positive odd integer.")
        if self.K <= 0:
            raise ValueError("K must be positive.")
        if self.rank <= 0 or self.rank > self.N**2:
            raise ValueError("rank must be between 1 and N squared.")
        if self.n_seeds <= 0:
            raise ValueError("n_seeds must be positive.")
        if self.dt <= 0 or self.tau_e <= 0 or self.tau_i <= 0:
            raise ValueError("Time values must be positive.")
        if self.init_steps < 0 or self.record_steps <= 0 or self.sample_every <= 0:
            raise ValueError("Simulation step counts are invalid.")


@dataclass
class ScanResult:
    strengths: np.ndarray
    relative_strengths: np.ndarray
    relative_strength_std: np.ndarray
    spectral_strengths: np.ndarray
    spectral_strength_std: np.ndarray
    local_curves: np.ndarray
    network_curves: np.ndarray
    pca_curves: np.ndarray
    transition_index: np.ndarray
    transition_std: np.ndarray
    dimensions: np.ndarray
    dimension_std: np.ndarray
    neighbor_correlation: np.ndarray
    neighbor_correlation_std: np.ndarray
    correlation_length: np.ndarray
    correlation_length_std: np.ndarray
    low_k_fraction: np.ndarray
    low_k_fraction_std: np.ndarray
    nonlocal_fraction: np.ndarray
    nonlocal_fraction_std: np.ndarray
    null_fraction: np.ndarray
    null_fraction_std: np.ndarray
    full_nonlocal_overlap: np.ndarray
    full_nonlocal_overlap_std: np.ndarray
    lowrank_power: np.ndarray
    lowrank_power_std: np.ndarray
    total_variance: np.ndarray
    total_variance_std: np.ndarray
    lowrank_input_variance: np.ndarray
    lowrank_input_variance_std: np.ndarray
    lowrank_output_variance: np.ndarray
    lowrank_output_variance_std: np.ndarray
    excitatory_active_fraction: np.ndarray
    excitatory_active_fraction_std: np.ndarray
    inhibitory_active_fraction: np.ndarray
    inhibitory_active_fraction_std: np.ndarray
    excitatory_current_power: np.ndarray
    excitatory_current_power_std: np.ndarray
    inhibitory_current_power: np.ndarray
    inhibitory_current_power_std: np.ndarray
    net_current_power: np.ndarray
    net_current_power_std: np.ndarray
    ei_cancellation_ratio: np.ndarray
    ei_cancellation_ratio_std: np.ndarray
    ei_current_correlation: np.ndarray
    ei_current_correlation_std: np.ndarray
    mean_balance_e: np.ndarray
    mean_balance_e_std: np.ndarray
    mean_balance_i: np.ndarray
    mean_balance_i_std: np.ndarray
    active_local_balance_e: np.ndarray
    active_local_balance_e_std: np.ndarray
    active_local_balance_i: np.ndarray
    active_local_balance_i_std: np.ndarray
    inactive_local_balance_e: np.ndarray
    inactive_local_balance_e_std: np.ndarray
    inactive_local_balance_i: np.ndarray
    inactive_local_balance_i_std: np.ndarray
    mean_external_current_e: np.ndarray
    mean_external_current_e_std: np.ndarray
    mean_excitatory_current_e: np.ndarray
    mean_excitatory_current_e_std: np.ndarray
    mean_inhibitory_current_e: np.ndarray
    mean_inhibitory_current_e_std: np.ndarray
    mean_net_current_e: np.ndarray
    mean_net_current_e_std: np.ndarray
    mean_external_current_i: np.ndarray
    mean_external_current_i_std: np.ndarray
    mean_excitatory_current_i: np.ndarray
    mean_excitatory_current_i_std: np.ndarray
    mean_inhibitory_current_i: np.ndarray
    mean_inhibitory_current_i_std: np.ndarray
    mean_net_current_i: np.ndarray
    mean_net_current_i_std: np.ndarray
    example_rates: list[np.ndarray]
    stable: np.ndarray
    geometric_shell_counts: np.ndarray


def scan_result_arrays(result: ScanResult) -> dict[str, np.ndarray]:
    """Return all scan results as named arrays."""
    return {
        "strengths": result.strengths,
        "relative_strengths": result.relative_strengths,
        "relative_strength_std": result.relative_strength_std,
        "spectral_strengths": result.spectral_strengths,
        "spectral_strength_std": result.spectral_strength_std,
        "local_curves": result.local_curves,
        "network_curves": result.network_curves,
        "pca_curves": result.pca_curves,
        "transition_index": result.transition_index,
        "transition_std": result.transition_std,
        "dimensions": result.dimensions,
        "dimension_std": result.dimension_std,
        "neighbor_correlation": result.neighbor_correlation,
        "neighbor_correlation_std": result.neighbor_correlation_std,
        "correlation_length": result.correlation_length,
        "correlation_length_std": result.correlation_length_std,
        "low_k_fraction": result.low_k_fraction,
        "low_k_fraction_std": result.low_k_fraction_std,
        "nonlocal_fraction": result.nonlocal_fraction,
        "nonlocal_fraction_std": result.nonlocal_fraction_std,
        "null_fraction": result.null_fraction,
        "null_fraction_std": result.null_fraction_std,
        "full_nonlocal_overlap": result.full_nonlocal_overlap,
        "full_nonlocal_overlap_std": result.full_nonlocal_overlap_std,
        "lowrank_power": result.lowrank_power,
        "lowrank_power_std": result.lowrank_power_std,
        "total_variance": result.total_variance,
        "total_variance_std": result.total_variance_std,
        "lowrank_input_variance": result.lowrank_input_variance,
        "lowrank_input_variance_std": result.lowrank_input_variance_std,
        "lowrank_output_variance": result.lowrank_output_variance,
        "lowrank_output_variance_std": result.lowrank_output_variance_std,
        "excitatory_active_fraction": result.excitatory_active_fraction,
        "excitatory_active_fraction_std": result.excitatory_active_fraction_std,
        "inhibitory_active_fraction": result.inhibitory_active_fraction,
        "inhibitory_active_fraction_std": result.inhibitory_active_fraction_std,
        "excitatory_current_power": result.excitatory_current_power,
        "excitatory_current_power_std": result.excitatory_current_power_std,
        "inhibitory_current_power": result.inhibitory_current_power,
        "inhibitory_current_power_std": result.inhibitory_current_power_std,
        "net_current_power": result.net_current_power,
        "net_current_power_std": result.net_current_power_std,
        "ei_cancellation_ratio": result.ei_cancellation_ratio,
        "ei_cancellation_ratio_std": result.ei_cancellation_ratio_std,
        "ei_current_correlation": result.ei_current_correlation,
        "ei_current_correlation_std": result.ei_current_correlation_std,
        "mean_balance_e": result.mean_balance_e,
        "mean_balance_e_std": result.mean_balance_e_std,
        "mean_balance_i": result.mean_balance_i,
        "mean_balance_i_std": result.mean_balance_i_std,
        "active_local_balance_e": result.active_local_balance_e,
        "active_local_balance_e_std": result.active_local_balance_e_std,
        "active_local_balance_i": result.active_local_balance_i,
        "active_local_balance_i_std": result.active_local_balance_i_std,
        "inactive_local_balance_e": result.inactive_local_balance_e,
        "inactive_local_balance_e_std": result.inactive_local_balance_e_std,
        "inactive_local_balance_i": result.inactive_local_balance_i,
        "inactive_local_balance_i_std": result.inactive_local_balance_i_std,
        "mean_external_current_e": result.mean_external_current_e,
        "mean_external_current_e_std": result.mean_external_current_e_std,
        "mean_excitatory_current_e": result.mean_excitatory_current_e,
        "mean_excitatory_current_e_std": result.mean_excitatory_current_e_std,
        "mean_inhibitory_current_e": result.mean_inhibitory_current_e,
        "mean_inhibitory_current_e_std": result.mean_inhibitory_current_e_std,
        "mean_net_current_e": result.mean_net_current_e,
        "mean_net_current_e_std": result.mean_net_current_e_std,
        "mean_external_current_i": result.mean_external_current_i,
        "mean_external_current_i_std": result.mean_external_current_i_std,
        "mean_excitatory_current_i": result.mean_excitatory_current_i,
        "mean_excitatory_current_i_std": result.mean_excitatory_current_i_std,
        "mean_inhibitory_current_i": result.mean_inhibitory_current_i,
        "mean_inhibitory_current_i_std": result.mean_inhibitory_current_i_std,
        "mean_net_current_i": result.mean_net_current_i,
        "mean_net_current_i_std": result.mean_net_current_i_std,
        "example_rates": np.stack(result.example_rates),
        "stable": result.stable,
        "geometric_shell_counts": result.geometric_shell_counts,
    }


def scan_metric_rows(result: ScanResult) -> list[dict[str, float | bool]]:
    """Return one summary row for each scanned strength."""
    rows = []
    for index in range(result.strengths.size):
        rows.append(
            {
                "strength": result.strengths[index],
                "relative_strength": result.relative_strengths[index],
                "spectral_strength": result.spectral_strengths[index],
                "network_advantage": result.transition_index[index],
                "pca_dimension": result.dimensions[index],
                "neighbor_correlation": result.neighbor_correlation[index],
                "correlation_length": result.correlation_length[index],
                "low_k_fraction": result.low_k_fraction[index],
                "nonlocal_fraction": result.nonlocal_fraction[index],
                "null_fraction": result.null_fraction[index],
                "full_nonlocal_overlap": result.full_nonlocal_overlap[index],
                "lowrank_power": result.lowrank_power[index],
                "lowrank_power_std": result.lowrank_power_std[index],
                "total_variance": result.total_variance[index],
                "total_variance_std": result.total_variance_std[index],
                "lowrank_input_variance": result.lowrank_input_variance[index],
                "lowrank_input_variance_std": result.lowrank_input_variance_std[index],
                "lowrank_output_variance": result.lowrank_output_variance[index],
                "lowrank_output_variance_std": result.lowrank_output_variance_std[index],
                "excitatory_active_fraction": result.excitatory_active_fraction[index],
                "excitatory_active_fraction_std": result.excitatory_active_fraction_std[index],
                "inhibitory_active_fraction": result.inhibitory_active_fraction[index],
                "inhibitory_active_fraction_std": result.inhibitory_active_fraction_std[index],
                "excitatory_current_power": result.excitatory_current_power[index],
                "excitatory_current_power_std": result.excitatory_current_power_std[index],
                "inhibitory_current_power": result.inhibitory_current_power[index],
                "inhibitory_current_power_std": result.inhibitory_current_power_std[index],
                "net_current_power": result.net_current_power[index],
                "net_current_power_std": result.net_current_power_std[index],
                "ei_cancellation_ratio": result.ei_cancellation_ratio[index],
                "ei_cancellation_ratio_std": result.ei_cancellation_ratio_std[index],
                "ei_current_correlation": result.ei_current_correlation[index],
                "ei_current_correlation_std": result.ei_current_correlation_std[index],
                "mean_balance_e": result.mean_balance_e[index],
                "mean_balance_e_std": result.mean_balance_e_std[index],
                "mean_balance_i": result.mean_balance_i[index],
                "mean_balance_i_std": result.mean_balance_i_std[index],
                "active_local_balance_e": result.active_local_balance_e[index],
                "active_local_balance_e_std": result.active_local_balance_e_std[index],
                "active_local_balance_i": result.active_local_balance_i[index],
                "active_local_balance_i_std": result.active_local_balance_i_std[index],
                "inactive_local_balance_e": result.inactive_local_balance_e[index],
                "inactive_local_balance_e_std": result.inactive_local_balance_e_std[index],
                "inactive_local_balance_i": result.inactive_local_balance_i[index],
                "inactive_local_balance_i_std": result.inactive_local_balance_i_std[index],
                "mean_external_current_e": result.mean_external_current_e[index],
                "mean_external_current_e_std": result.mean_external_current_e_std[index],
                "mean_excitatory_current_e": result.mean_excitatory_current_e[index],
                "mean_excitatory_current_e_std": result.mean_excitatory_current_e_std[index],
                "mean_inhibitory_current_e": result.mean_inhibitory_current_e[index],
                "mean_inhibitory_current_e_std": result.mean_inhibitory_current_e_std[index],
                "mean_net_current_e": result.mean_net_current_e[index],
                "mean_net_current_e_std": result.mean_net_current_e_std[index],
                "mean_external_current_i": result.mean_external_current_i[index],
                "mean_external_current_i_std": result.mean_external_current_i_std[index],
                "mean_excitatory_current_i": result.mean_excitatory_current_i[index],
                "mean_excitatory_current_i_std": result.mean_excitatory_current_i_std[index],
                "mean_inhibitory_current_i": result.mean_inhibitory_current_i[index],
                "mean_inhibitory_current_i_std": result.mean_inhibitory_current_i_std[index],
                "mean_net_current_i": result.mean_net_current_i[index],
                "mean_net_current_i_std": result.mean_net_current_i_std[index],
                "stable": bool(result.stable[index]),
            }
        )
    return rows


def periodic_gaussian_kernel(N: int, sigma: float) -> np.ndarray:
    """Return a normalized Gaussian kernel on a periodic square grid."""
    axis = np.arange(N)
    distance = np.minimum(axis, N - axis) / N
    one_d = np.exp(-0.5 * (distance / sigma) ** 2)
    one_d /= one_d.sum()
    return np.outer(one_d, one_d)


def convolution_matrix(kernel: np.ndarray) -> np.ndarray:
    """Return the dense operator for periodic convolution."""
    N = kernel.shape[0]
    operator = np.empty((N * N, N * N), dtype=np.float64)
    for source in range(N * N):
        row, col = divmod(source, N)
        response = np.roll(np.roll(kernel, row, axis=0), col, axis=1)
        operator[:, source] = response.reshape(-1)
    return 0.5 * (operator + operator.T)


def periodic_laplacian(N: int) -> np.ndarray:
    """Return the positive graph Laplacian of a periodic square grid."""
    D = N * N
    laplacian = np.zeros((D, D), dtype=np.float64)
    for row in range(N):
        for col in range(N):
            index = row * N + col
            laplacian[index, index] = 4.0
            for next_row, next_col in (
                ((row - 1) % N, col),
                ((row + 1) % N, col),
                (row, (col - 1) % N),
                (row, (col + 1) % N),
            ):
                laplacian[index, next_row * N + next_col] = -1.0
    return laplacian


def geometric_modes(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Return periodic Laplacian modes and complete eigenvalue-shell counts."""
    eigenvalues, modes = np.linalg.eigh(periodic_laplacian(N))
    shell_ends = np.flatnonzero(~np.isclose(eigenvalues[:-1], eigenvalues[1:])) + 1
    shell_counts = np.concatenate([shell_ends, [eigenvalues.size]])
    return modes, shell_counts


def make_nonlocal_operator(
    N: int, rank: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return chi = M N.T / N, as used in the earlier disorder scripts."""
    D = N * N
    m_vectors = rng.standard_normal((D, rank))
    n_vectors = rng.standard_normal((D, rank))
    m_vectors -= m_vectors.mean(axis=0, keepdims=True)
    n_vectors -= n_vectors.mean(axis=0, keepdims=True)
    operator = (m_vectors @ n_vectors.T) / N
    return operator, m_vectors, n_vectors


def ordered_modes(operator: np.ndarray) -> np.ndarray:
    """Return orthonormal response modes ordered by singular value."""
    modes, _, _ = np.linalg.svd(operator, full_matrices=True)
    return modes


def subspace_variance_fraction(activity: np.ndarray, modes: np.ndarray) -> float:
    """Return the activity variance captured by an orthonormal subspace."""
    if not np.isfinite(activity).all():
        return np.nan
    centered = activity - activity.mean(axis=1, keepdims=True)
    total = np.sum(centered**2)
    if total <= np.finfo(float).eps:
        return 0.0
    return float(np.sum((modes.T @ centered) ** 2) / total)


@dataclass
class ActivityDiagnostics:
    """Store fluctuation power and low-rank input-output variance."""

    lowrank_power: float
    total_variance: float
    lowrank_fraction: float
    lowrank_input_variance: float
    lowrank_output_variance: float


@dataclass
class CurrentDiagnostics:
    """Store centered E/I current power and cancellation metrics."""

    excitatory_power: float
    inhibitory_power: float
    net_power: float
    cancellation_ratio: float
    ei_correlation: float


@dataclass
class PopulationBalanceDiagnostics:
    """Store mean and local balance for one population."""

    mean_balance: float
    active_local_balance: float
    inactive_local_balance: float
    mean_external_current: float
    mean_excitatory_current: float
    mean_inhibitory_current: float
    mean_net_current: float


def population_balance_diagnostics(
    external_current: np.ndarray,
    excitatory_current: np.ndarray,
    inhibitory_current: np.ndarray,
    active: np.ndarray,
) -> PopulationBalanceDiagnostics:
    """Return uncentered mean and sitewise current balance."""
    shape = external_current.shape
    if (
        external_current.ndim != 2
        or excitatory_current.shape != shape
        or inhibitory_current.shape != shape
        or active.shape != shape
    ):
        raise ValueError("Balance inputs must have matching (space, time) shapes.")
    if shape[1] == 0:
        raise ValueError("Balance inputs must contain at least one time sample.")
    arrays = (external_current, excitatory_current, inhibitory_current)
    if not all(np.isfinite(values).all() for values in arrays):
        return PopulationBalanceDiagnostics(*(np.nan for _ in range(7)))

    net_current = external_current + excitatory_current + inhibitory_current
    mean_external = float(np.mean(external_current))
    mean_excitatory = float(np.mean(excitatory_current))
    mean_inhibitory = float(np.mean(inhibitory_current))
    mean_net = float(np.mean(net_current))
    mean_denominator = (
        abs(mean_external) + abs(mean_excitatory) + abs(mean_inhibitory)
    )
    mean_balance = (
        abs(mean_net) / mean_denominator if mean_denominator > 0.0 else np.nan
    )
    local_denominator = (
        np.abs(external_current)
        + np.abs(excitatory_current)
        + np.abs(inhibitory_current)
    )
    local_ratio = np.divide(
        np.abs(net_current),
        local_denominator,
        out=np.full(shape, np.nan, dtype=float),
        where=local_denominator > 0.0,
    )
    active_values = local_ratio[active & np.isfinite(local_ratio)]
    inactive_values = local_ratio[(~active) & np.isfinite(local_ratio)]
    return PopulationBalanceDiagnostics(
        mean_balance=mean_balance,
        active_local_balance=(
            float(np.mean(active_values)) if active_values.size else np.nan
        ),
        inactive_local_balance=(
            float(np.mean(inactive_values)) if inactive_values.size else np.nan
        ),
        mean_external_current=mean_external,
        mean_excitatory_current=mean_excitatory,
        mean_inhibitory_current=mean_inhibitory,
        mean_net_current=mean_net,
    )


def current_diagnostics(
    excitatory_current: np.ndarray,
    inhibitory_current: np.ndarray,
) -> CurrentDiagnostics:
    """Return temporal E/I current cancellation diagnostics."""
    if excitatory_current.ndim != 2 or inhibitory_current.shape != excitatory_current.shape:
        raise ValueError("E and I currents must have matching (space, time) shapes.")
    if excitatory_current.shape[1] == 0:
        raise ValueError("Current arrays must contain at least one time sample.")
    if not np.isfinite(excitatory_current).all() or not np.isfinite(inhibitory_current).all():
        return CurrentDiagnostics(*(np.nan for _ in range(5)))

    centered_e = excitatory_current - excitatory_current.mean(axis=1, keepdims=True)
    centered_i = inhibitory_current - inhibitory_current.mean(axis=1, keepdims=True)
    time_count = excitatory_current.shape[1]
    power_e = float(np.sum(centered_e**2) / time_count)
    power_i = float(np.sum(centered_i**2) / time_count)
    covariance = float(np.sum(centered_e * centered_i) / time_count)
    centered_net = centered_e + centered_i
    net_power = float(np.sum(centered_net**2) / time_count)
    denominator = power_e + power_i
    cancellation_ratio = net_power / denominator if denominator > 0.0 else np.nan
    correlation_denominator = np.sqrt(power_e * power_i)
    correlation = (
        covariance / correlation_denominator
        if correlation_denominator > 0.0
        else np.nan
    )
    return CurrentDiagnostics(
        excitatory_power=power_e,
        inhibitory_power=power_i,
        net_power=net_power,
        cancellation_ratio=cancellation_ratio,
        ei_correlation=float(np.clip(correlation, -1.0, 1.0)),
    )


def activity_diagnostics(
    activity: np.ndarray,
    left_vectors: np.ndarray,
    right_vectors: np.ndarray,
    strength: float,
    spatial_width: int,
) -> ActivityDiagnostics:
    """Return diagnostics for time-centered excitatory activity."""
    if activity.ndim != 2:
        raise ValueError("Activity must have shape (space, time).")
    if left_vectors.ndim != 2 or right_vectors.shape != left_vectors.shape:
        raise ValueError("Low-rank vectors must have matching matrix shapes.")
    if activity.shape[0] != left_vectors.shape[0]:
        raise ValueError("Activity and low-rank vectors must use the same space.")
    if spatial_width <= 0:
        raise ValueError("Spatial width must be positive.")
    if not np.isfinite(activity).all():
        return ActivityDiagnostics(*(np.nan for _ in range(5)))

    centered = activity - activity.mean(axis=1, keepdims=True)
    time_count = activity.shape[1]
    total_variance = float(np.sum(centered**2) / time_count)
    left_modes, singular_values, _ = np.linalg.svd(left_vectors, full_matrices=False)
    tolerance = (
        max(left_vectors.shape) * np.finfo(float).eps * singular_values[0]
        if singular_values.size
        else 0.0
    )
    left_modes = left_modes[:, singular_values > tolerance]
    lowrank_power = float(np.sum((left_modes.T @ centered) ** 2) / time_count)
    lowrank_fraction = (
        lowrank_power / total_variance
        if total_variance > np.finfo(float).eps
        else 0.0
    )
    lowrank_input = right_vectors.T @ centered / spatial_width
    lowrank_output = strength * left_vectors @ lowrank_input
    return ActivityDiagnostics(
        lowrank_power=lowrank_power,
        total_variance=total_variance,
        lowrank_fraction=lowrank_fraction,
        lowrank_input_variance=float(np.sum(lowrank_input**2) / time_count),
        lowrank_output_variance=float(np.sum(lowrank_output**2) / time_count),
    )


def subspace_overlap(first: np.ndarray, second: np.ndarray) -> float:
    """Return the mean squared overlap between equal-size subspaces."""
    if first.shape[1] != second.shape[1]:
        raise ValueError("Subspaces must have the same dimension.")
    return float(np.linalg.norm(first.T @ second, ord="fro") ** 2 / first.shape[1])


def simulate(
    W_e: np.ndarray,
    W_i: np.ndarray,
    W_nonlocal: np.ndarray,
    g: float,
    K: float,
    dt: float,
    tau: np.ndarray,
    u: np.ndarray,
    J0: np.ndarray,
    init_steps: int,
    record_steps: int,
    sample_every: int,
    re0: np.ndarray,
    ri0: np.ndarray,
    max_rate: float,
) -> tuple[
    np.ndarray,
    bool,
    float,
    float,
    CurrentDiagnostics,
    PopulationBalanceDiagnostics,
    PopulationBalanceDiagnostics,
]:
    """Return rates, activity, current, and population balance metrics."""
    re = re0.copy()
    ri = ri0.copy()
    full_e = W_e + g * W_nonlocal
    sqrt_K = np.sqrt(K)
    records = []
    excitatory_active = []
    inhibitory_active = []
    excitatory_currents = []
    inhibitory_currents = []
    inhibitory_population_e_currents = []
    inhibitory_population_i_currents = []
    excitatory_active_masks = []
    inhibitory_active_masks = []

    for step in range(init_steps + record_steps):
        with np.errstate(over="ignore", invalid="ignore"):
            conv_e = full_e @ re
            conv_i = W_i @ ri
            mu_e = sqrt_K * (u[0] + J0[0, 0] * conv_e + J0[0, 1] * conv_i)
            mu_i = sqrt_K * (u[1] + J0[1, 0] * conv_e + J0[1, 1] * conv_i)
            re += (dt / tau[0]) * (-re + np.maximum(mu_e, 0.0))
            ri += (dt / tau[1]) * (-ri + np.maximum(mu_i, 0.0))

        finite = np.isfinite(re).all() and np.isfinite(ri).all()
        bounded = np.max(np.abs(re)) <= max_rate and np.max(np.abs(ri)) <= max_rate
        if not finite or not bounded:
            sample_count = len(range(0, record_steps, sample_every))
            missing = CurrentDiagnostics(*(np.nan for _ in range(5)))
            missing_balance = PopulationBalanceDiagnostics(
                *(np.nan for _ in range(7))
            )
            return (
                np.full((re.size, sample_count), np.nan),
                False,
                np.nan,
                np.nan,
                missing,
                missing_balance,
                missing_balance,
            )
        if step >= init_steps and (step - init_steps) % sample_every == 0:
            records.append(re.copy())
            excitatory_active.append(np.mean(mu_e > 0.0))
            inhibitory_active.append(np.mean(mu_i > 0.0))
            excitatory_currents.append(sqrt_K * J0[0, 0] * conv_e)
            inhibitory_currents.append(sqrt_K * J0[0, 1] * conv_i)
            inhibitory_population_e_currents.append(sqrt_K * J0[1, 0] * conv_e)
            inhibitory_population_i_currents.append(sqrt_K * J0[1, 1] * conv_i)
            excitatory_active_masks.append(mu_e > 0.0)
            inhibitory_active_masks.append(mu_i > 0.0)

    excitatory_current_array = np.stack(excitatory_currents, axis=1)
    inhibitory_current_array = np.stack(inhibitory_currents, axis=1)
    inhibitory_population_e_array = np.stack(
        inhibitory_population_e_currents, axis=1
    )
    inhibitory_population_i_array = np.stack(
        inhibitory_population_i_currents, axis=1
    )
    currents = current_diagnostics(
        excitatory_current_array,
        inhibitory_current_array,
    )
    excitatory_balance = population_balance_diagnostics(
        np.full_like(excitatory_current_array, sqrt_K * u[0]),
        excitatory_current_array,
        inhibitory_current_array,
        np.stack(excitatory_active_masks, axis=1),
    )
    inhibitory_balance = population_balance_diagnostics(
        np.full_like(inhibitory_population_e_array, sqrt_K * u[1]),
        inhibitory_population_e_array,
        inhibitory_population_i_array,
        np.stack(inhibitory_active_masks, axis=1),
    )
    return (
        np.stack(records, axis=1),
        True,
        float(np.mean(excitatory_active)),
        float(np.mean(inhibitory_active)),
        currents,
        excitatory_balance,
        inhibitory_balance,
    )


def reconstruction_curve(activity: np.ndarray, modes: np.ndarray) -> np.ndarray:
    """Return cumulative variance captured by an ordered orthonormal basis."""
    if not np.isfinite(activity).all():
        return np.full(modes.shape[1], np.nan)
    centered = activity - activity.mean(axis=1, keepdims=True)
    total = np.sum(centered**2)
    if total <= np.finfo(float).eps:
        return np.zeros(modes.shape[1])
    coefficients = modes.T @ centered
    power = np.sum(coefficients**2, axis=1)
    return np.cumsum(power) / total


def effective_dimension(activity: np.ndarray, threshold: float = 0.9) -> float:
    """Return the PCA dimension that captures the requested variance."""
    if not np.isfinite(activity).all():
        return np.nan
    centered = activity - activity.mean(axis=1, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    power = singular_values**2
    if power.sum() <= np.finfo(float).eps:
        return 0
    return int(np.searchsorted(np.cumsum(power) / power.sum(), threshold) + 1)


def pca_reconstruction_curve(activity: np.ndarray, mode_count: int) -> np.ndarray:
    """Return the optimal cumulative reconstruction curve from PCA."""
    if not np.isfinite(activity).all():
        return np.full(mode_count, np.nan)
    centered = activity - activity.mean(axis=1, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    power = singular_values**2
    if power.sum() <= np.finfo(float).eps:
        return np.zeros(mode_count)
    curve = np.cumsum(power) / power.sum()
    result = np.ones(mode_count)
    result[: curve.size] = curve
    return result


def finite_mean_std(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return finite-value mean and standard deviation without empty warnings."""
    finite = np.isfinite(values)
    count = finite.sum(axis=axis)
    total = np.where(finite, values, 0.0).sum(axis=axis)
    mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    expanded_mean = np.expand_dims(mean, axis=axis)
    squared = np.where(finite, (values - expanded_mean) ** 2, 0.0).sum(axis=axis)
    std = np.sqrt(np.divide(squared, count, out=np.full_like(total, np.nan), where=count > 0))
    return mean, std


def spatial_statistics(activity: np.ndarray, N: int) -> tuple[float, float, float]:
    """Return neighbor correlation, correlation length, and low-k power."""
    frames = activity.T.reshape(-1, N, N)
    centered = frames - frames.mean(axis=(1, 2), keepdims=True)
    variance = np.mean(centered**2)
    if not np.isfinite(variance) or variance <= np.finfo(float).eps:
        return np.nan, np.nan, np.nan

    neighbor = 0.5 * (
        np.mean(centered * np.roll(centered, 1, axis=1))
        + np.mean(centered * np.roll(centered, 1, axis=2))
    ) / variance

    power = np.abs(np.fft.fft2(centered, axes=(1, 2))) ** 2
    autocorrelation = np.fft.ifft2(power.mean(axis=0)).real
    autocorrelation /= autocorrelation[0, 0]
    axis = np.minimum(np.arange(N), N - np.arange(N))
    radius = np.sqrt(axis[:, None] ** 2 + axis[None, :] ** 2)
    radial_distance = np.arange(int(np.floor(radius.max())) + 1)
    radial_curve = np.array(
        [autocorrelation[(radius >= d - 0.5) & (radius < d + 0.5)].mean()
         for d in radial_distance]
    )
    below = np.flatnonzero(radial_curve[1:] <= np.exp(-1.0))
    if below.size == 0:
        length = float(radial_distance[-1])
    else:
        upper = int(below[0] + 1)
        lower = upper - 1
        denominator = radial_curve[lower] - radial_curve[upper]
        fraction = 0.0 if denominator <= 0 else (
            radial_curve[lower] - np.exp(-1.0)
        ) / denominator
        length = lower + float(np.clip(fraction, 0.0, 1.0))

    frequency = np.fft.fftfreq(N) * N
    wave_number = np.sqrt(frequency[:, None] ** 2 + frequency[None, :] ** 2)
    mean_power = power.mean(axis=0)
    nonzero = wave_number > 0
    low_k = nonzero & (wave_number <= 2.0)
    low_k_fraction = mean_power[low_k].sum() / mean_power[nonzero].sum()
    return float(neighbor), length, float(low_k_fraction)


def run_scan(args: ModeScanConfig) -> ScanResult:
    """Run the non-local strength scan."""
    if args.N % 2 != 1:
        raise ValueError("N must be an odd integer.")

    sigma = np.array([args.sigma_e, args.sigma_i])
    W_e = convolution_matrix(periodic_gaussian_kernel(args.N, sigma[0]))
    W_i = convolution_matrix(periodic_gaussian_kernel(args.N, sigma[1]))
    _, geometric_shell_counts = geometric_modes(args.N)
    local_modes = ordered_modes(W_e)
    if args.strengths is None:
        scan_values = np.asarray(args.rho_f_values, dtype=float)
        scan_rho_f = True
    else:
        scan_values = np.asarray(args.strengths, dtype=float)
        scan_rho_f = False

    local_curves_all = []
    network_curves_all = []
    pca_curves_all = []
    dimensions_all = []
    neighbor_correlation_all = []
    correlation_length_all = []
    low_k_fraction_all = []
    nonlocal_fraction_all = []
    null_fraction_all = []
    full_nonlocal_overlap_all = []
    lowrank_power_all = []
    total_variance_all = []
    lowrank_input_variance_all = []
    lowrank_output_variance_all = []
    excitatory_active_fraction_all = []
    inhibitory_active_fraction_all = []
    excitatory_current_power_all = []
    inhibitory_current_power_all = []
    net_current_power_all = []
    ei_cancellation_ratio_all = []
    ei_current_correlation_all = []
    balance_sources = {
        "mean_balance_e": ("e", "mean_balance"),
        "mean_balance_i": ("i", "mean_balance"),
        "active_local_balance_e": ("e", "active_local_balance"),
        "active_local_balance_i": ("i", "active_local_balance"),
        "inactive_local_balance_e": ("e", "inactive_local_balance"),
        "inactive_local_balance_i": ("i", "inactive_local_balance"),
        "mean_external_current_e": ("e", "mean_external_current"),
        "mean_excitatory_current_e": ("e", "mean_excitatory_current"),
        "mean_inhibitory_current_e": ("e", "mean_inhibitory_current"),
        "mean_net_current_e": ("e", "mean_net_current"),
        "mean_external_current_i": ("i", "mean_external_current"),
        "mean_excitatory_current_i": ("i", "mean_excitatory_current"),
        "mean_inhibitory_current_i": ("i", "mean_inhibitory_current"),
        "mean_net_current_i": ("i", "mean_net_current"),
    }
    balance_values_all = {name: [] for name in balance_sources}
    relative_strengths_all = []
    spectral_strengths_all = []
    strengths_all = []
    stable_all = []
    example_rates = [None] * len(scan_values)

    tau = np.array([args.tau_e, args.tau_i])
    u = np.array([args.u_e, args.u_i])
    J0 = np.array([[1.0, -4.0], [2.0, -2.0]])
    local_spectral_norm = np.linalg.norm(W_e, ord=2)
    local_frobenius_norm = np.linalg.norm(W_e, ord="fro")
    for seed_offset in range(args.n_seeds):
        rng = np.random.default_rng(args.seed + seed_offset)
        W_nonlocal, left_vectors, right_vectors = make_nonlocal_operator(
            args.N, args.rank, rng
        )
        nonlocal_modes = ordered_modes(W_nonlocal)[:, : args.rank]
        null_rng = np.random.default_rng(args.seed + seed_offset + 1_000_000)
        W_null, _, _ = make_nonlocal_operator(args.N, args.rank, null_rng)
        null_modes = ordered_modes(W_null)[:, : args.rank]
        fixed_rate = -np.linalg.solve(J0, u)
        re0 = fixed_rate[0] + 0.05 * rng.random(args.N * args.N)
        ri0 = fixed_rate[1] + 0.08 * rng.random(args.N * args.N)
        seed_local = []
        seed_network = []
        seed_pca = []
        seed_dimensions = []
        seed_neighbor_correlation = []
        seed_correlation_length = []
        seed_low_k_fraction = []
        seed_nonlocal_fraction = []
        seed_null_fraction = []
        seed_full_nonlocal_overlap = []
        seed_lowrank_power = []
        seed_total_variance = []
        seed_lowrank_input_variance = []
        seed_lowrank_output_variance = []
        seed_excitatory_active_fraction = []
        seed_inhibitory_active_fraction = []
        seed_excitatory_current_power = []
        seed_inhibitory_current_power = []
        seed_net_current_power = []
        seed_ei_cancellation_ratio = []
        seed_ei_current_correlation = []
        seed_balance_values = {name: [] for name in balance_sources}
        seed_relative = []
        seed_spectral = []
        seed_strengths = []
        seed_stable = []

        nonlocal_spectral_norm = np.linalg.norm(W_nonlocal, ord=2)
        nonlocal_frobenius_norm = np.linalg.norm(W_nonlocal, ord="fro")
        for strength_index, scan_value in enumerate(scan_values):
            if scan_rho_f:
                strength = (
                    scan_value * local_frobenius_norm / nonlocal_frobenius_norm
                )
            else:
                strength = scan_value
            relative = (
                strength * nonlocal_frobenius_norm / local_frobenius_norm
            )
            spectral = strength * nonlocal_spectral_norm / local_spectral_norm
            print(
                f"Run seed {args.seed + seed_offset}, g={strength:g}, "
                f"rho_F={relative:.3g}, rho_2={spectral:.3g}"
            )
            (
                activity,
                is_stable,
                active_e,
                active_i,
                currents,
                balance_e,
                balance_i,
            ) = simulate(
                W_e,
                W_i,
                W_nonlocal,
                strength,
                args.K,
                args.dt,
                tau,
                u,
                J0,
                args.init_steps,
                args.record_steps,
                args.sample_every,
                re0,
                ri0,
                args.max_rate,
            )
            if not is_stable:
                print(
                    f"Unstable dynamics at g={strength:g}. "
                    "Record NaN metrics and continue the scan."
                )
            network_modes = ordered_modes(W_e + strength * W_nonlocal)
            seed_local.append(reconstruction_curve(activity, local_modes))
            seed_network.append(reconstruction_curve(activity, network_modes))
            seed_pca.append(pca_reconstruction_curve(activity, local_modes.shape[1]))
            seed_dimensions.append(effective_dimension(activity))
            neighbor, length, low_k = spatial_statistics(activity, args.N)
            seed_neighbor_correlation.append(neighbor)
            seed_correlation_length.append(length)
            seed_low_k_fraction.append(low_k)
            seed_nonlocal_fraction.append(
                subspace_variance_fraction(activity, nonlocal_modes)
            )
            seed_null_fraction.append(
                subspace_variance_fraction(activity, null_modes)
            )
            seed_full_nonlocal_overlap.append(
                subspace_overlap(network_modes[:, : args.rank], nonlocal_modes)
            )
            diagnostics = activity_diagnostics(
                activity,
                left_vectors,
                right_vectors,
                strength,
                args.N,
            )
            seed_lowrank_power.append(diagnostics.lowrank_power)
            seed_total_variance.append(diagnostics.total_variance)
            seed_lowrank_input_variance.append(
                diagnostics.lowrank_input_variance
            )
            seed_lowrank_output_variance.append(
                diagnostics.lowrank_output_variance
            )
            seed_excitatory_active_fraction.append(active_e)
            seed_inhibitory_active_fraction.append(active_i)
            seed_excitatory_current_power.append(currents.excitatory_power)
            seed_inhibitory_current_power.append(currents.inhibitory_power)
            seed_net_current_power.append(currents.net_power)
            seed_ei_cancellation_ratio.append(currents.cancellation_ratio)
            seed_ei_current_correlation.append(currents.ei_correlation)
            balances = {"e": balance_e, "i": balance_i}
            for name, (population, attribute) in balance_sources.items():
                seed_balance_values[name].append(
                    getattr(balances[population], attribute)
                )
            seed_relative.append(relative)
            seed_spectral.append(spectral)
            seed_strengths.append(strength)
            seed_stable.append(is_stable)
            if seed_offset == 0:
                example_rates[strength_index] = activity

        local_curves_all.append(seed_local)
        network_curves_all.append(seed_network)
        pca_curves_all.append(seed_pca)
        dimensions_all.append(seed_dimensions)
        neighbor_correlation_all.append(seed_neighbor_correlation)
        correlation_length_all.append(seed_correlation_length)
        low_k_fraction_all.append(seed_low_k_fraction)
        nonlocal_fraction_all.append(seed_nonlocal_fraction)
        null_fraction_all.append(seed_null_fraction)
        full_nonlocal_overlap_all.append(seed_full_nonlocal_overlap)
        lowrank_power_all.append(seed_lowrank_power)
        total_variance_all.append(seed_total_variance)
        lowrank_input_variance_all.append(seed_lowrank_input_variance)
        lowrank_output_variance_all.append(seed_lowrank_output_variance)
        excitatory_active_fraction_all.append(seed_excitatory_active_fraction)
        inhibitory_active_fraction_all.append(seed_inhibitory_active_fraction)
        excitatory_current_power_all.append(seed_excitatory_current_power)
        inhibitory_current_power_all.append(seed_inhibitory_current_power)
        net_current_power_all.append(seed_net_current_power)
        ei_cancellation_ratio_all.append(seed_ei_cancellation_ratio)
        ei_current_correlation_all.append(seed_ei_current_correlation)
        for name in balance_sources:
            balance_values_all[name].append(seed_balance_values[name])
        relative_strengths_all.append(seed_relative)
        spectral_strengths_all.append(seed_spectral)
        strengths_all.append(seed_strengths)
        stable_all.append(seed_stable)

    local_curves_all = np.asarray(local_curves_all)
    network_curves_all = np.asarray(network_curves_all)
    pca_curves_all = np.asarray(pca_curves_all)
    local_curves, _ = finite_mean_std(local_curves_all)
    network_curves, _ = finite_mean_std(network_curves_all)
    pca_curves, _ = finite_mean_std(pca_curves_all)
    shell_counts = geometric_shell_counts[geometric_shell_counts <= args.auc_modes]
    if shell_counts.size == 0:
        shell_counts = geometric_shell_counts[:1]
    shell_indices = shell_counts - 1
    transition_all = np.mean(
        network_curves_all[:, :, shell_indices] - local_curves_all[:, :, shell_indices],
        axis=2,
    )
    transition_index, transition_std = finite_mean_std(transition_all)
    dimensions, dimension_std = finite_mean_std(np.asarray(dimensions_all))
    neighbor_correlation, neighbor_correlation_std = finite_mean_std(
        np.asarray(neighbor_correlation_all)
    )
    correlation_length, correlation_length_std = finite_mean_std(
        np.asarray(correlation_length_all)
    )
    low_k_fraction, low_k_fraction_std = finite_mean_std(
        np.asarray(low_k_fraction_all)
    )
    nonlocal_fraction, nonlocal_fraction_std = finite_mean_std(
        np.asarray(nonlocal_fraction_all)
    )
    null_fraction, null_fraction_std = finite_mean_std(
        np.asarray(null_fraction_all)
    )
    full_nonlocal_overlap, full_nonlocal_overlap_std = finite_mean_std(
        np.asarray(full_nonlocal_overlap_all)
    )
    lowrank_power, lowrank_power_std = finite_mean_std(
        np.asarray(lowrank_power_all)
    )
    total_variance, total_variance_std = finite_mean_std(
        np.asarray(total_variance_all)
    )
    lowrank_input_variance, lowrank_input_variance_std = finite_mean_std(
        np.asarray(lowrank_input_variance_all)
    )
    lowrank_output_variance, lowrank_output_variance_std = finite_mean_std(
        np.asarray(lowrank_output_variance_all)
    )
    excitatory_active_fraction, excitatory_active_fraction_std = finite_mean_std(
        np.asarray(excitatory_active_fraction_all)
    )
    inhibitory_active_fraction, inhibitory_active_fraction_std = finite_mean_std(
        np.asarray(inhibitory_active_fraction_all)
    )
    excitatory_current_power, excitatory_current_power_std = finite_mean_std(
        np.asarray(excitatory_current_power_all)
    )
    inhibitory_current_power, inhibitory_current_power_std = finite_mean_std(
        np.asarray(inhibitory_current_power_all)
    )
    net_current_power, net_current_power_std = finite_mean_std(
        np.asarray(net_current_power_all)
    )
    ei_cancellation_ratio, ei_cancellation_ratio_std = finite_mean_std(
        np.asarray(ei_cancellation_ratio_all)
    )
    ei_current_correlation, ei_current_correlation_std = finite_mean_std(
        np.asarray(ei_current_correlation_all)
    )
    balance_arrays = {}
    for name, values in balance_values_all.items():
        mean, std = finite_mean_std(np.asarray(values))
        balance_arrays[name] = mean
        balance_arrays[name + "_std"] = std
    relative_strengths, relative_strength_std = finite_mean_std(
        np.asarray(relative_strengths_all)
    )
    spectral_strengths, spectral_strength_std = finite_mean_std(
        np.asarray(spectral_strengths_all)
    )
    strengths, _ = finite_mean_std(np.asarray(strengths_all))
    stable = np.any(np.asarray(stable_all), axis=0)
    return ScanResult(
        strengths=strengths,
        relative_strengths=relative_strengths,
        relative_strength_std=relative_strength_std,
        spectral_strengths=spectral_strengths,
        spectral_strength_std=spectral_strength_std,
        local_curves=local_curves,
        network_curves=network_curves,
        pca_curves=pca_curves,
        transition_index=transition_index,
        transition_std=transition_std,
        dimensions=np.asarray(dimensions),
        dimension_std=dimension_std,
        neighbor_correlation=neighbor_correlation,
        neighbor_correlation_std=neighbor_correlation_std,
        correlation_length=correlation_length,
        correlation_length_std=correlation_length_std,
        low_k_fraction=low_k_fraction,
        low_k_fraction_std=low_k_fraction_std,
        nonlocal_fraction=nonlocal_fraction,
        nonlocal_fraction_std=nonlocal_fraction_std,
        null_fraction=null_fraction,
        null_fraction_std=null_fraction_std,
        full_nonlocal_overlap=full_nonlocal_overlap,
        full_nonlocal_overlap_std=full_nonlocal_overlap_std,
        lowrank_power=lowrank_power,
        lowrank_power_std=lowrank_power_std,
        total_variance=total_variance,
        total_variance_std=total_variance_std,
        lowrank_input_variance=lowrank_input_variance,
        lowrank_input_variance_std=lowrank_input_variance_std,
        lowrank_output_variance=lowrank_output_variance,
        lowrank_output_variance_std=lowrank_output_variance_std,
        excitatory_active_fraction=excitatory_active_fraction,
        excitatory_active_fraction_std=excitatory_active_fraction_std,
        inhibitory_active_fraction=inhibitory_active_fraction,
        inhibitory_active_fraction_std=inhibitory_active_fraction_std,
        excitatory_current_power=excitatory_current_power,
        excitatory_current_power_std=excitatory_current_power_std,
        inhibitory_current_power=inhibitory_current_power,
        inhibitory_current_power_std=inhibitory_current_power_std,
        net_current_power=net_current_power,
        net_current_power_std=net_current_power_std,
        ei_cancellation_ratio=ei_cancellation_ratio,
        ei_cancellation_ratio_std=ei_cancellation_ratio_std,
        ei_current_correlation=ei_current_correlation,
        ei_current_correlation_std=ei_current_correlation_std,
        example_rates=example_rates,
        stable=stable,
        geometric_shell_counts=geometric_shell_counts,
        **balance_arrays,
    )


def plot_result(result: ScanResult, args: ModeScanConfig) -> plt.Figure:
    """Plot reconstruction curves, transition score, and example activity."""
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)

    ax_curve = fig.add_subplot(grid[0, 0])
    stable_indices = np.flatnonzero(result.stable)
    if stable_indices.size == 0:
        selected = [0]
    else:
        intermediate = stable_indices[
            np.argmin(np.abs(result.relative_strengths[stable_indices] - 1.0))
        ]
        selected = sorted(set([stable_indices[0], intermediate, stable_indices[-1]]))
    for index in selected:
        count = min(args.plot_modes, result.local_curves.shape[1])
        modes = result.geometric_shell_counts[result.geometric_shell_counts <= count]
        mode_indices = modes - 1
        label = (
            f"g={result.strengths[index]:g}, "
            f"rho_F={result.relative_strengths[index]:.2g}"
        )
        ax_curve.plot(
            modes,
            result.local_curves[index, mode_indices],
            "--",
            label=f"Local, {label}",
        )
        ax_curve.plot(
            modes,
            result.network_curves[index, mode_indices],
            label=f"Network, {label}",
        )
    reference = selected[len(selected) // 2]
    count = min(args.plot_modes, result.pca_curves.shape[1])
    modes = result.geometric_shell_counts[result.geometric_shell_counts <= count]
    ax_curve.plot(
        modes,
        result.pca_curves[reference, modes - 1],
        color="black",
        linestyle=":",
        linewidth=2,
        label="PCA upper bound",
    )
    ax_curve.set(
        xlabel="Number of modes in complete geometric shells",
        ylabel="Captured variance",
        ylim=(0, 1.02),
    )
    ax_curve.legend(fontsize=8)
    ax_curve.set_title("Mode reconstruction")

    ax_transition = fig.add_subplot(grid[0, 1])
    ax_transition.axhline(0.0, color="black", linewidth=1)
    ax_transition.plot(result.relative_strengths, result.transition_index, "o-")
    ax_transition.fill_between(
        result.relative_strengths,
        result.transition_index - result.transition_std,
        result.transition_index + result.transition_std,
        alpha=0.2,
    )
    ax_transition.set(
        xlabel="Relative total strength, rho_F",
        ylabel="Full minus local variance",
        title="Full-mode advantage",
    )

    ax_dimension = fig.add_subplot(grid[0, 2])
    ax_dimension.plot(result.relative_strengths, result.dimensions, "o-")
    ax_dimension.fill_between(
        result.relative_strengths,
        result.dimensions - result.dimension_std,
        result.dimensions + result.dimension_std,
        alpha=0.2,
    )
    ax_dimension.set(
        xlabel="Relative total strength, rho_F",
        ylabel="PCA dimension",
        title="Dimension at 90% variance",
    )

    for panel, index in enumerate(selected):
        ax = fig.add_subplot(grid[1, panel])
        rates = result.example_rates[index]
        title = (
            f"Activity, g={result.strengths[index]:g}, "
            f"rho_F={result.relative_strengths[index]:.2g}, "
            f"rho_2={result.spectral_strengths[index]:.2g}"
        )
        if not result.stable[index]:
            ax.text(0.5, 0.5, "Unstable dynamics", ha="center", va="center")
            ax.set_title(title)
            ax.set(xticks=[], yticks=[])
            continue
        frame = rates[:, -1].reshape(args.N, args.N)
        image = ax.imshow(frame, origin="lower", cmap="viridis")
        ax.set_title(title)
        ax.set(xticks=[], yticks=[])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Local and network modes in a 2D E/I network, K={args.K:g}")
    return fig
