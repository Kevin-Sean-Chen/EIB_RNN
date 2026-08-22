"""Analyze activity produced by rank-one Gabor disorder."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.sparse.linalg import eigs

from src.disorder import DisorderConfig, simulate_disorder
from src.metrics import latent_coherence, mean_second_acf_peak, second_acf_peak
from src.analysis.spectral import (
    circular_convolution_matrix,
    legacy_block_operator,
    legacy_gaussian_kernel,
)


@dataclass
class RankOneResult:
    """Store one rank-one activity and alignment scan."""

    K_values: np.ndarray
    phase_values: np.ndarray
    latent_coherence: np.ndarray
    active_fraction: np.ndarray
    balance_error: np.ndarray
    latent_acf_peak: np.ndarray
    activity_acf_peak: np.ndarray
    eigenvector_alignment: np.ndarray
    spectral_abscissa: np.ndarray
    kappa: np.ndarray
    example_activity: np.ndarray
    left_patterns: np.ndarray
    right_patterns: np.ndarray


def eigenvector_alignment(
    operator: np.ndarray,
    right_pattern: np.ndarray,
    spatial_size: int,
    count: int = 10,
) -> tuple[float, float]:
    """Return right-pattern alignment and the spectral abscissa."""
    count = min(count, operator.shape[0] - 2)
    values, vectors = eigs(operator, k=count, which="LM")
    leading_index = int(np.argmax(values.real))
    population_vector = vectors[:spatial_size, leading_index]
    denominator = np.linalg.norm(right_pattern) * np.linalg.norm(population_vector)
    alignment = np.abs(right_pattern @ population_vector) / denominator
    return float(alignment), float(np.max(values.real))


def run_rank_one_scan(
    base_config: DisorderConfig,
    K_values: list[float],
    phase_values: list[float],
    acf_sample_count: int,
) -> RankOneResult:
    """Scan rank-one Gabor disorder across K and phase."""
    shape = (len(K_values), len(phase_values))
    metrics = [np.empty(shape) for _ in range(7)]
    (
        coherence,
        active_fraction,
        balance_error,
        latent_peak,
        activity_peak,
        alignment,
        spectral_abscissa,
    ) = metrics
    record_count = base_config.record_steps // base_config.steps_per_record
    kappa = np.empty(shape + (record_count,))
    left_patterns = np.empty((len(phase_values), base_config.N**2))
    right_patterns = np.empty_like(left_patterns)
    example_activity = None
    local_e = circular_convolution_matrix(
        legacy_gaussian_kernel(base_config.N, base_config.sigma_e)
    )
    local_i = circular_convolution_matrix(
        legacy_gaussian_kernel(base_config.N, base_config.sigma_i)
    )

    for K_index, K in enumerate(K_values):
        for phase_index, phase in enumerate(phase_values):
            print(f"Run K={K:g}, phase={phase / np.pi:.2f} pi")
            config = replace(
                base_config,
                K=K,
                rank=1,
                pattern_type="gabor",
                phase_offset=phase,
            )
            result = simulate_disorder(config)
            activity = result.excitatory.reshape(config.N**2, -1)
            left = result.left_patterns[:, 0]
            right = result.right_patterns[:, 0] / config.strength
            left_patterns[phase_index] = left
            right_patterns[phase_index] = right
            coherence[K_index, phase_index] = latent_coherence(activity, left)
            active_fraction[K_index, phase_index] = np.mean(result.excitatory_field > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                relative_balance = np.abs(
                    result.excitatory_field - result.inhibitory_field
                ) / np.abs(result.excitatory_field)
            finite_balance = relative_balance[np.isfinite(relative_balance)]
            balance_error[K_index, phase_index] = (
                np.mean(finite_balance) if finite_balance.size else np.nan
            )
            latent_trace = left @ activity / config.N
            kappa[K_index, phase_index] = latent_trace
            latent_peak[K_index, phase_index] = second_acf_peak(latent_trace)[0]
            activity_peak[K_index, phase_index] = mean_second_acf_peak(
                result.excitatory_field,
                sample_count=acf_sample_count,
                seed=config.seed,
            )[0]
            operator = legacy_block_operator(
                local_e,
                local_i,
                left,
                right,
                K,
                config.strength,
            )
            (
                alignment[K_index, phase_index],
                spectral_abscissa[K_index, phase_index],
            ) = eigenvector_alignment(operator, right, config.N**2)
            if K_index == len(K_values) - 1 and phase_index == len(phase_values) - 1:
                example_activity = result.excitatory

    return RankOneResult(
        K_values=np.asarray(K_values),
        phase_values=np.asarray(phase_values),
        latent_coherence=coherence,
        active_fraction=active_fraction,
        balance_error=balance_error,
        latent_acf_peak=latent_peak,
        activity_acf_peak=activity_peak,
        eigenvector_alignment=alignment,
        spectral_abscissa=spectral_abscissa,
        kappa=kappa,
        example_activity=example_activity,
        left_patterns=left_patterns,
        right_patterns=right_patterns,
    )
