"""Define same-time rigid-shift reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


RIGID_RECONSTRUCTION_SECTIONS = {
    "model": ("N", "device", "seed"),
    "network": (
        "dt", "tau_e", "tau_i", "K", "J_ee", "J_ei", "J_ie", "J_ii",
        "sigma_e", "sigma_i", "u_e", "u_i", "stimulus_gain", "init_scale",
        "baseline_mode", "field_clip",
    ),
    "task": ("steps", "smoothing_width", "shift_distance"),
    "learning": (
        "learning_method", "training_trials", "evaluation_trials", "epochs",
        "learning_rate", "delta", "forgetting_factor",
    ),
}


@dataclass
class RigidReconstructionConfig:
    """Define one rigid-shift reconstruction experiment."""

    N: int = 31
    device: str = "cpu"
    seed: int = 7
    dt: float = 0.001
    tau_e: float = 0.01
    tau_i: float = 0.01
    K: float = 10.0
    J_ee: float = 1.0
    J_ei: float = -4.0
    J_ie: float = 2.0
    J_ii: float = -2.0
    sigma_e: float = 0.05
    sigma_i: float = 0.05 * np.sqrt(2)
    u_e: float = 10.0
    u_i: float = 0.0
    stimulus_gain: float = 1.0
    init_scale: float = 1.0
    baseline_mode: str = "replace"
    field_clip: float | None = None
    steps: int = 500
    smoothing_width: float = 1.0 / 31.0
    shift_distance: float = 10.0
    training_trials: int = 20
    evaluation_trials: int = 10
    learning_method: str = "adam"
    epochs: int = 60
    learning_rate: float = 0.01
    delta: float = 0.1
    forgetting_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.N <= 0 or self.N % 2 != 1:
            raise ValueError("N must be a positive odd integer.")
        if self.baseline_mode not in ("replace", "add"):
            raise ValueError("baseline_mode must be 'replace' or 'add'.")
        if min(self.steps, self.training_trials, self.evaluation_trials) <= 0:
            raise ValueError("Step and trial counts must be positive.")
        if self.learning_method not in ("adam", "rls"):
            raise ValueError("learning_method must be 'adam' or 'rls'.")
        if self.epochs <= 0 or self.learning_rate <= 0:
            raise ValueError("Adam settings must be positive.")

    @property
    def coupling(self) -> np.ndarray:
        return np.array([[self.J_ee, self.J_ei], [self.J_ie, self.J_ii]])
