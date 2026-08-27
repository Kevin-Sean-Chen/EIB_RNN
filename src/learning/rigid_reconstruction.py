"""Train rigid-shift reconstruction with RLS."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

from src.learning.rls import initialize_inverse_correlation, update_rls


@dataclass
class RigidReconstructionResult:
    """Store reconstruction training and evaluation data."""

    training_mse: np.ndarray
    evaluation_mse: float
    evaluation_r2: float
    target: np.ndarray
    prediction: np.ndarray
    activity: np.ndarray
    readout_weights: np.ndarray


@torch.no_grad()
def settled_state(model, seed: int):
    """Return a state settled under the configured baseline input."""
    state = model.initial_state(seed)
    zero_input = torch.zeros(
        model.config.N, model.config.N, device=model.readout.device,
    )
    for _ in range(model.config.init_steps):
        state = model.step(state, zero_input)
    return state


@torch.no_grad()
def run_reconstruction_trial(
    model, stimulus, target, seed: int, update: bool, inverse=None,
    initial_state=None,
):
    """Run one reconstruction trial and optionally update its readout."""
    state = settled_state(model, seed) if initial_state is None else initial_state
    predictions = []
    activities = []
    squared_error = 0.0
    for step in range(stimulus.shape[-1]):
        state = model.step(state, stimulus[:, :, step])
        features = model.features(state)
        prediction = model.predict(features).reshape(-1)
        error = prediction - target[step].reshape(-1)
        squared_error += float(torch.mean(error**2))
        if update:
            model.readout[:], inverse = update_rls(
                model.readout, inverse, features, error,
                model.config.forgetting_factor,
            )
        predictions.append(prediction.squeeze())
        activities.append(features[:-1])
    return (
        torch.stack(predictions), torch.stack(activities, dim=-1),
        squared_error / stimulus.shape[-1], inverse,
    )


def train_rigid_reconstruction(model, stimulus, target) -> RigidReconstructionResult:
    """Train the fixed-reservoir linear readout."""
    config = model.config
    history = []
    training_state = settled_state(model, config.seed)
    if config.learning_method == "adam":
        weights = nn.Parameter(model.readout.detach().clone())
        optimizer = torch.optim.Adam([weights], lr=config.learning_rate)
        for epoch in range(config.epochs):
            with torch.no_grad():
                state = settled_state(model, config.seed + epoch)
                features = []
                for step in range(stimulus.shape[-1]):
                    state = model.step(state, stimulus[:, :, step])
                    features.append(model.features(state))
                feature_matrix = torch.stack(features)
            optimizer.zero_grad()
            prediction = (feature_matrix @ weights).squeeze()
            loss = torch.mean((prediction - target) ** 2)
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach()))
        model.readout.copy_(weights.detach())
    else:
        inverse = initialize_inverse_correlation(model.feature_count, config.delta, model.readout.device)
        for trial in range(config.training_trials):
            _, _, mse, inverse = run_reconstruction_trial(
                model, stimulus, target, config.seed + trial, True, inverse,
                initial_state=training_state if trial == 0 else None,
            )
            history.append(mse)
    evaluation_predictions = []
    example = None
    for trial in range(config.evaluation_trials):
        generator = torch.Generator(device="cpu").manual_seed(config.seed + 1000 + trial)
        evaluation_state = tuple(
            value + config.evaluation_perturbation
            * torch.randn(value.shape, generator=generator).to(value.device)
            for value in training_state
        )
        run = run_reconstruction_trial(
            model, stimulus, target, config.seed + config.training_trials + trial,
            False, initial_state=evaluation_state,
        )
        evaluation_predictions.append(run[0])
        if example is None:
            example = run
    pooled_prediction = torch.cat(evaluation_predictions)
    pooled_target = target.repeat(config.evaluation_trials)
    residual = torch.sum((pooled_prediction - pooled_target) ** 2)
    total = torch.sum((pooled_target - torch.mean(pooled_target)) ** 2)
    return RigidReconstructionResult(
        training_mse=np.asarray(history),
        evaluation_mse=float(torch.mean((pooled_prediction - pooled_target) ** 2)),
        evaluation_r2=float(1.0 - residual / (total + 1e-12)),
        target=target.cpu().numpy(),
        prediction=example[0].cpu().numpy(),
        activity=example[1].cpu().numpy(),
        readout_weights=model.readout.cpu().numpy().copy(),
    )
