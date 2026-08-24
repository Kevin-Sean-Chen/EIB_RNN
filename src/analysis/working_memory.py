"""Scan spatial working-memory performance across network scale."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.disorder import gabor_pattern
from src.learning.force import train_force
from src.learning.ridge import train_and_evaluate_ridge
from src.models.working_memory import SpatialWorkingMemoryReservoir
from src.tasks.working_memory import WorkingMemoryConfig, make_working_memory_trial


@dataclass
class WorkingMemoryScanResult:
    """Store one RLS working-memory scan."""

    K_values: np.ndarray
    output_mse: np.ndarray
    memory_mse: np.ndarray
    output_r2: np.ndarray
    memory_r2: np.ndarray
    final_training_output_mse: np.ndarray
    final_training_memory_mse: np.ndarray


def make_patterns(config: WorkingMemoryConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed trigger patterns and one shared go cue."""
    first = gabor_pattern(config.N, config.trigger_one_frequency, config.trigger_one_angle, 0.5, 0.1)
    second = gabor_pattern(config.N, config.trigger_two_frequency, config.trigger_two_angle, 0.5, 0.1)
    if config.go_pattern_type == "gabor":
        go_cue = gabor_pattern(config.N, config.go_frequency, config.go_angle, 0.5, 0.1)
    else:
        go_cue = np.random.default_rng(config.seed).standard_normal((config.N, config.N))
    return tuple(np.asarray(value * config.pattern_scale, dtype=np.float32) for value in (first, second, go_cue))


def make_spatial_model(config: WorkingMemoryConfig, K: float) -> SpatialWorkingMemoryReservoir:
    """Return one fixed spatial reservoir."""
    return SpatialWorkingMemoryReservoir(
        N=config.N, dt=config.dt, tau_e=config.tau_e, tau_i=config.tau_i,
        K=K, coupling=config.coupling, drive=config.drive, sigma=config.sigma,
        stimulus_gain=config.stimulus_gain, feedback_gain=config.feedback_gain,
        feedback_scale=config.feedback_scale,
        use_feedback_output=config.use_feedback_output,
        use_feedback_memory=config.use_feedback_memory,
        init_scale=config.init_scale, microsteps=config.microsteps,
        field_clip=config.field_clip, seed=config.seed,
        normalize_kernel=config.normalize_kernel, nonlinearity=config.nonlinearity,
        device=config.device,
    )


def run_working_memory_K_scan(
    config: WorkingMemoryConfig,
    K_values: list[float],
) -> WorkingMemoryScanResult:
    """Train and evaluate one spatial RLS model for each K value."""
    if config.model_type != "spatial":
        raise ValueError("The K scan requires model_type='spatial'.")
    if not K_values or any(value <= 0 for value in K_values):
        raise ValueError("K_values must contain positive values.")
    patterns = make_patterns(config)
    metrics = {name: [] for name in (
        "output_mse", "memory_mse", "output_r2", "memory_r2",
        "final_training_output_mse", "final_training_memory_mse",
    )}
    for K in K_values:
        model = make_spatial_model(config, K)

        def trial_factory(choice: int | None):
            return make_working_memory_trial(
                config.N, config.steps, config.delay_steps, config.cue_steps,
                patterns, choice=choice, ramp_memory=config.ramp_memory,
            )

        if config.learning_method == "ridge":
            result = train_and_evaluate_ridge(
                model, trial_factory, config.training_trials, config.evaluation_trials,
                config.cue_steps + config.delay_steps, config.cue_steps,
                config.ridge_lambda, config.seed,
            )
            metrics["output_mse"].append(result.output_mse)
            metrics["memory_mse"].append(result.memory_mse)
            metrics["output_r2"].append(result.output_r2)
            metrics["memory_r2"].append(result.memory_r2)
            metrics["final_training_output_mse"].append(np.nan)
            metrics["final_training_memory_mse"].append(np.nan)
        else:
            result = train_force(
                model, trial_factory, "spatial", config.training_trials,
                config.evaluation_trials, config.delta, config.forgetting_factor,
                config.train_output, config.train_memory, config.seed,
                evaluation_start_step=config.cue_steps + config.delay_steps,
            )
            metrics["output_mse"].append(result.evaluation_output_mse)
            metrics["memory_mse"].append(result.evaluation_memory_mse)
            metrics["output_r2"].append(result.evaluation_output_r2)
            metrics["memory_r2"].append(result.evaluation_memory_r2)
            metrics["final_training_output_mse"].append(result.output_mse_history[-1])
            metrics["final_training_memory_mse"].append(result.memory_mse_history[-1])
    return WorkingMemoryScanResult(
        K_values=np.asarray(K_values, dtype=float),
        **{name: np.asarray(values, dtype=float) for name, values in metrics.items()},
    )
