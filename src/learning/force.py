"""Train fixed reservoirs with online FORCE/RLS updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from src.learning.rls import initialize_inverse_correlation, update_rls


Trial = tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]


@dataclass
class ForceResult:
    """Store RLS training and evaluation results."""

    output_mse_history: np.ndarray
    memory_mse_history: np.ndarray
    evaluation_output_mse: float
    evaluation_memory_mse: float
    evaluation_output_r2: float
    evaluation_memory_r2: float
    output_weights: np.ndarray
    memory_weights: np.ndarray
    example_output: np.ndarray
    example_memory: np.ndarray
    example_output_target: np.ndarray
    example_memory_target: np.ndarray
    example_activity: np.ndarray


def _input_at(stimulus: torch.Tensor, step: int, model_type: str) -> torch.Tensor:
    frame = stimulus[:, :, step]
    return frame if model_type == "spatial" else frame.reshape(-1)


@torch.no_grad()
def run_trial(model, trial: Trial, model_type: str, state_seed: int):
    """Run one trial without readout updates."""
    output_target, memory_target, stimulus, choice = trial
    state = model.initial_state(state_seed)
    outputs = []
    memories = []
    activities = []
    for step in range(stimulus.shape[-1]):
        state = model.step(state, _input_at(stimulus, step, model_type))
        features = model.features(state)
        output, memory = model.predict(features)
        outputs.append(output.squeeze())
        memories.append(memory.squeeze())
        activities.append(features)
    return (
        torch.stack(outputs),
        torch.stack(memories),
        torch.stack(activities, dim=-1),
        output_target,
        memory_target,
        choice,
    )


@torch.no_grad()
def train_force(
    model,
    trial_factory: Callable[[int | None], Trial],
    model_type: str,
    training_trials: int,
    evaluation_trials: int,
    delta: float,
    forgetting_factor: float,
    train_output: bool,
    train_memory: bool,
    seed: int,
    evaluation_start_step: int = 0,
) -> ForceResult:
    """Train output and memory readouts with RLS only."""
    feature_count = model.feature_count + 1
    output_inverse = initialize_inverse_correlation(feature_count, delta, model.output_weights.device)
    memory_inverse = initialize_inverse_correlation(feature_count, delta, model.memory_weights.device)
    output_history = []
    memory_history = []

    for trial_index in range(training_trials):
        # Alternate choices so each short training run includes both targets.
        trial = trial_factory(trial_index % 2)
        output_target, memory_target, stimulus, _ = trial
        state = model.initial_state(seed + trial_index)
        output_squared_error = 0.0
        memory_squared_error = 0.0
        for step in range(stimulus.shape[-1]):
            state = model.step(state, _input_at(stimulus, step, model_type))
            features = model.features(state)
            augmented = torch.cat(
                [features, torch.ones(1, device=features.device)]
            )
            output = augmented @ model.output_weights
            memory = augmented @ model.memory_weights
            output_error = output.reshape(-1) - output_target[step].to(output.device).reshape(-1)
            memory_error = memory.reshape(-1) - memory_target[step].to(memory.device).reshape(-1)
            output_squared_error += float(torch.mean(output_error**2))
            memory_squared_error += float(torch.mean(memory_error**2))
            if train_output:
                model.output_weights[:], output_inverse = update_rls(
                    model.output_weights,
                    output_inverse,
                    augmented,
                    output_error,
                    forgetting_factor,
                )
            if train_memory:
                model.memory_weights[:], memory_inverse = update_rls(
                    model.memory_weights,
                    memory_inverse,
                    augmented,
                    memory_error,
                    forgetting_factor,
                )
        output_history.append(output_squared_error / stimulus.shape[-1])
        memory_history.append(memory_squared_error / stimulus.shape[-1])

    evaluation_output = []
    evaluation_memory = []
    pooled_output = []
    pooled_output_target = []
    pooled_memory = []
    pooled_memory_target = []
    example = None
    for trial_index in range(evaluation_trials):
        trial = trial_factory(trial_index % 2)
        run = run_trial(model, trial, model_type, seed + training_trials + trial_index)
        output, memory, activity, output_target, memory_target, _ = run
        output_slice = output[evaluation_start_step:]
        output_target_slice = output_target[evaluation_start_step:]
        memory_slice = memory[evaluation_start_step:]
        memory_target_slice = memory_target[evaluation_start_step:]
        evaluation_output.append(float(torch.mean((output_slice - output_target_slice) ** 2)))
        evaluation_memory.append(float(torch.mean((memory_slice - memory_target_slice) ** 2)))
        pooled_output.append(output_slice)
        pooled_output_target.append(output_target_slice)
        pooled_memory.append(memory_slice)
        pooled_memory_target.append(memory_target_slice)
        if example is None:
            example = run

    output, memory, activity, output_target, memory_target, _ = example
    def pooled_r2(predictions, targets) -> float:
        prediction = torch.cat(predictions)
        target = torch.cat(targets)
        residual = torch.sum((prediction - target) ** 2)
        total = torch.sum((target - torch.mean(target)) ** 2)
        return float(1.0 - residual / (total + 1e-12))

    return ForceResult(
        output_mse_history=np.asarray(output_history),
        memory_mse_history=np.asarray(memory_history),
        evaluation_output_mse=float(np.mean(evaluation_output)),
        evaluation_memory_mse=float(np.mean(evaluation_memory)),
        evaluation_output_r2=pooled_r2(pooled_output, pooled_output_target),
        evaluation_memory_r2=pooled_r2(pooled_memory, pooled_memory_target),
        output_weights=model.output_weights.cpu().numpy().copy(),
        memory_weights=model.memory_weights.cpu().numpy().copy(),
        example_output=output.cpu().numpy(),
        example_memory=memory.cpu().numpy(),
        example_output_target=output_target.cpu().numpy(),
        example_memory_target=memory_target.cpu().numpy(),
        example_activity=activity.cpu().numpy(),
    )
