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
    memory_r2: float


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
) -> tuple[np.ndarray, np.ndarray]:
    """Collect bias-augmented features and one target after a start step."""
    features = []
    targets = []
    for trial_index in range(trial_count):
        trial = trial_factory(trial_index % 2)
        run = run_trial(model, trial, "spatial", seed + trial_index)
        activity = run[2][:, start_step:].T.cpu().numpy().astype(np.float64)
        target = run[target_index][start_step:].reshape(-1, 1).cpu().numpy().astype(np.float64)
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
) -> RidgeResult:
    """Fit the legacy readout windows and return post-go scores."""
    memory_x, memory_y = collect_readout_data(
        model, trial_factory, training_trials, memory_start_step, 4, seed,
    )
    output_x, output_y = collect_readout_data(
        model, trial_factory, training_trials, output_start_step, 3,
        seed + training_trials,
    )
    output_weights = solve_ridge(output_x, output_y, penalty)
    memory_weights = solve_ridge(memory_x, memory_y, penalty)
    model.output_weights.copy_(torch.tensor(output_weights, dtype=torch.float32, device=model.output_weights.device))
    model.memory_weights.copy_(torch.tensor(memory_weights, dtype=torch.float32, device=model.memory_weights.device))

    predictions = {name: [] for name in ("output", "memory", "output_target", "memory_target")}
    for trial_index in range(evaluation_trials):
        run = run_trial(
            model, trial_factory(trial_index % 2), "spatial",
            seed + 2 * training_trials + trial_index,
        )
        for name, value in zip(predictions, (run[0], run[1], run[3], run[4])):
            predictions[name].append(value[output_start_step:].cpu())
    values = {name: torch.cat(items) for name, items in predictions.items()}

    def mse(name: str) -> float:
        return float(torch.mean((values[name] - values[f"{name}_target"]) ** 2))

    def r2(name: str) -> float:
        target = values[f"{name}_target"]
        residual = torch.sum((values[name] - target) ** 2)
        total = torch.sum((target - torch.mean(target)) ** 2)
        return float(1.0 - residual / (total + 1e-12))

    return RidgeResult(mse("output"), mse("memory"), r2("output"), r2("memory"))
