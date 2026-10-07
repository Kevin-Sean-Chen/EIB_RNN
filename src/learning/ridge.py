"""Fit fixed-reservoir readouts with ridge regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from src.learning.force import Trial, run_trial


@dataclass
class RidgeResult:
    """Store ridge evaluation metrics."""

    output_mse: float
    memory_mse: float
    output_r2: float
    training_output_r2: float
    memory_r2: float
    final_output_accuracy: float
    example_output: np.ndarray
    example_memory: np.ndarray
    example_output_target: np.ndarray
    example_memory_target: np.ndarray


def solve_ridge(features: np.ndarray, targets: np.ndarray, penalty: float) -> np.ndarray:
    """Return the closed-form ridge weights."""
    width = features.shape[1]
    system = features.T @ features + penalty * np.eye(width, dtype=np.float64)
    return np.linalg.solve(system, features.T @ targets)


@torch.no_grad()
def collect_readout_data(
    model,
    trial_factory: Callable[[int | None], Trial],
    trial_count: int,
    start_step: int,
    target_index: int,
    seed: int,
    init_steps: int = 0,
    stop_step: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect bias-augmented features and one target after a start step."""
    features = []
    targets = []
    for trial_index in range(trial_count):
        trial = trial_factory(trial_index % 2)
        run = run_trial(
            model, trial, "spatial", seed + trial_index, init_steps,
        )
        activity = run[2][:, start_step:stop_step].T.cpu().numpy().astype(np.float64)
        target = run[target_index][start_step:stop_step].reshape(-1, 1).cpu().numpy().astype(np.float64)
        features.append(activity)
        targets.append(target)
    matrix = np.concatenate(features)
    matrix = np.concatenate([matrix, np.ones((matrix.shape[0], 1))], axis=1)
    return matrix, np.concatenate(targets)


@torch.no_grad()
def train_and_evaluate_ridge(
    model,
    trial_factory: Callable[[int | None], Trial],
    training_trials: int,
    evaluation_trials: int,
    output_start_step: int,
    memory_start_step: int,
    penalty: float,
    seed: int,
    init_steps: int = 0,
    output_stop_step: int | None = None,
) -> RidgeResult:
    """Fit the legacy readout windows and return post-go scores."""
    memory_x, memory_y = collect_readout_data(
        model, trial_factory, training_trials, memory_start_step, 4, seed,
        init_steps,
    )
    output_x, output_y = collect_readout_data(
        model, trial_factory, training_trials, output_start_step, 3,
        seed + training_trials, init_steps, output_stop_step,
    )
    output_weights = solve_ridge(output_x, output_y, penalty)
    memory_weights = solve_ridge(memory_x, memory_y, penalty)
    model.output_weights.copy_(torch.tensor(output_weights, dtype=torch.float32, device=model.output_weights.device))
    model.memory_weights.copy_(torch.tensor(memory_weights, dtype=torch.float32, device=model.memory_weights.device))

    predictions = {name: [] for name in ("output", "memory", "output_target", "memory_target")}
    example = None
    final_correct = []
    for trial_index in range(evaluation_trials):
        run = run_trial(
            model, trial_factory(trial_index % 2), "spatial",
            seed + 2 * training_trials + trial_index,
            init_steps,
        )
        for name, value in zip(predictions, (run[0], run[1], run[3], run[4])):
            predictions[name].append(value[output_start_step:output_stop_step].cpu())
        if example is None:
            example = run
        final_correct.append(float(run[0][-1] * run[3][-1] > 0))
    values = {name: torch.cat(items) for name, items in predictions.items()}

    def mse(name: str) -> float:
        return float(torch.mean((values[name] - values[f"{name}_target"]) ** 2))

    def r2(name: str) -> float:
        target = values[f"{name}_target"]
        residual = torch.sum((values[name] - target) ** 2)
        total = torch.sum((target - torch.mean(target)) ** 2)
        return float(1.0 - residual / (total + 1e-12))

    training_prediction = output_x @ output_weights
    training_residual = np.sum((training_prediction - output_y) ** 2)
    training_total = np.sum((output_y - output_y.mean()) ** 2)
    return RidgeResult(
        output_mse=mse("output"),
        memory_mse=mse("memory"),
        output_r2=r2("output"),
        training_output_r2=float(1.0 - training_residual / (training_total + 1e-12)),
        memory_r2=r2("memory"),
        final_output_accuracy=float(np.mean(final_correct)),
        example_output=example[0].cpu().numpy(),
        example_memory=example[1].cpu().numpy(),
        example_output_target=example[3].cpu().numpy(),
        example_memory_target=example[4].cpu().numpy(),
    )
