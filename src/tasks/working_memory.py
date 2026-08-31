"""Create trials for the two-choice working-memory task."""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass


WORKING_MEMORY_SECTIONS = {
    "model": ("model_type", "N", "device", "nonlinearity", "seed"),
    "network": (
        "dt", "tau_e", "tau_i", "K", "J_ee", "J_ei", "J_ie", "J_ii",
        "sigma_e", "sigma_i", "u_e", "u_i", "recurrent_gain",
        "normalize_kernel", "balance_random_rows", "init_scale", "microsteps",
        "field_clip",
    ),
    "task": (
        "steps", "delay_steps", "cue_steps", "stimulus_gain", "ramp_memory",
        "trigger_one_frequency", "trigger_one_angle", "trigger_two_frequency",
        "trigger_two_angle", "go_frequency", "go_angle", "pattern_scale",
        "go_pattern_type",
    ),
    "learning": (
        "training_trials", "evaluation_trials", "delta", "forgetting_factor",
        "feedback_gain", "feedback_scale", "use_feedback_output",
        "use_feedback_memory", "train_output", "train_memory",
        "learning_method", "ridge_lambda",
    ),
}


@dataclass
class WorkingMemoryConfig:
    """Define one RLS working-memory experiment."""

    model_type: str = "spatial"
    N: int = 23
    device: str = "cpu"
    nonlinearity: str = "relu"
    seed: int = 7
    dt: float = 0.001
    tau_e: float = 0.01
    tau_i: float = 0.01
    K: float = 20.0
    J_ee: float = 1.0
    J_ei: float = -4.0
    J_ie: float = 2.0
    J_ii: float = -2.0
    sigma_e: float = 0.05
    sigma_i: float = 0.05 * np.sqrt(2)
    u_e: float = 10.0
    u_i: float = 0.0
    recurrent_gain: float = 1.5
    normalize_kernel: bool = True
    balance_random_rows: bool = False
    init_scale: float = 0.1
    microsteps: int = 1
    field_clip: float | None = 100.0
    steps: int = 500
    delay_steps: int = 250
    cue_steps: int = 20
    stimulus_gain: float = 10.0
    ramp_memory: bool = False
    trigger_one_frequency: float = 2.5
    trigger_one_angle: float = 30.0
    trigger_two_frequency: float = 0.5
    trigger_two_angle: float = 60.0
    go_frequency: float = 1.5
    go_angle: float = 90.0
    pattern_scale: float = 0.1
    go_pattern_type: str = "random"
    training_trials: int = 10
    evaluation_trials: int = 10
    delta: float = 0.1
    forgetting_factor: float = 1.0
    feedback_gain: float = 0.001
    feedback_scale: float = 0.01
    use_feedback_output: bool = True
    use_feedback_memory: bool = True
    train_output: bool = True
    train_memory: bool = True
    learning_method: str = "rls"
    ridge_lambda: float = 0.01

    def __post_init__(self) -> None:
        if self.model_type not in ("spatial", "non_spatial"):
            raise ValueError("model_type must be 'spatial' or 'non_spatial'.")
        if self.N <= 0 or self.N % 2 != 1:
            raise ValueError("N must be a positive odd integer.")
        if self.training_trials <= 0 or self.evaluation_trials <= 0:
            raise ValueError("Trial counts must be positive.")
        if self.microsteps <= 0:
            raise ValueError("microsteps must be positive.")
        if self.field_clip is not None and self.field_clip <= 0:
            raise ValueError("field_clip must be positive or null.")
        if self.learning_method not in ("rls", "ridge"):
            raise ValueError("learning_method must be 'rls' or 'ridge'.")
        if self.ridge_lambda <= 0:
            raise ValueError("ridge_lambda must be positive.")
        if self.delta <= 0 or not 0 < self.forgetting_factor <= 1:
            raise ValueError("RLS parameters are invalid.")
        if self.go_pattern_type not in ("random", "gabor"):
            raise ValueError("go_pattern_type must be 'random' or 'gabor'.")

    @property
    def coupling(self) -> np.ndarray:
        return np.array([[self.J_ee, self.J_ei], [self.J_ie, self.J_ii]])

    @property
    def drive(self) -> np.ndarray:
        return np.array([self.u_e, self.u_i])

    @property
    def sigma(self) -> np.ndarray:
        return np.array([self.sigma_e, self.sigma_i])


def make_working_memory_trial(
    N: int,
    steps: int,
    delay_steps: int,
    cue_steps: int,
    input_patterns: tuple[np.ndarray, np.ndarray, np.ndarray],
    choice: int | None = None,
    ramp_memory: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Return output target, memory target, stimulus, and binary choice."""
    if N <= 0 or steps <= 0 or delay_steps < 0 or cue_steps <= 0:
        raise ValueError("Trial sizes must be valid.")
    if steps < delay_steps + 2 * cue_steps:
        raise ValueError("steps must include trigger, delay, and go-cue periods.")
    if choice is None:
        generator = rng if rng is not None else np.random.default_rng()
        choice = int(generator.integers(0, 2))
    if choice not in (0, 1):
        raise ValueError("choice must be 0 or 1.")
    trigger_left, trigger_right, go_cue = input_patterns
    for pattern in input_patterns:
        if np.asarray(pattern).shape != (N, N):
            raise ValueError("Each input pattern must have shape (N, N).")

    stimulus = np.zeros((N, N, steps), dtype=np.float32)
    output_target = np.zeros(steps, dtype=np.float32)
    memory_target = np.zeros(steps, dtype=np.float32)
    sign = 1.0 if choice == 0 else -1.0
    trigger = trigger_left if choice == 0 else trigger_right
    delay_end = cue_steps + delay_steps
    go_end = delay_end + cue_steps

    for step in range(steps):
        if step < cue_steps:
            stimulus[:, :, step] = trigger
            memory_target[step] = sign
        elif step < delay_end:
            if ramp_memory:
                progress = (step - cue_steps + 1) / float(delay_steps)
                memory_target[step] = np.clip(progress, 0.0, 1.0) * sign
            else:
                memory_target[step] = sign
        elif step < go_end:
            stimulus[:, :, step] = go_cue
            progress = (step - delay_end + 1) / float(cue_steps)
            output_target[step] = np.clip(progress, 0.0, 1.0) * sign
            memory_target[step] = sign
        else:
            output_target[step] = sign
            memory_target[step] = sign

    return (
        torch.tensor(output_target),
        torch.tensor(memory_target),
        torch.tensor(stimulus),
        choice,
    )
