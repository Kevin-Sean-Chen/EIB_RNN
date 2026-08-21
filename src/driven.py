"""Simulate a driven two-population spatial E/I network."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


DRIVEN_DOT_SECTIONS = {
    "network": (
        "N",
        "K",
        "J_ee",
        "J_ei",
        "J_ie",
        "J_ii",
        "sigma_e",
        "sigma_i",
        "u_e",
        "u_i",
    ),
    "simulation": (
        "seed",
        "device",
        "dt",
        "init_steps",
        "record_steps",
        "steps_per_record",
        "tau_e",
        "tau_i",
    ),
    "stimulus": ("dot_size", "drift_rate", "angle", "input_amplitude"),
}


@dataclass
class DrivenDotConfig:
    """Define one driven-dot simulation."""

    N: int = 15
    K: float = 100.0
    J_ee: float = 1.0
    J_ei: float = -4.0
    J_ie: float = 2.0
    J_ii: float = -2.0
    sigma_e: float = 0.05
    sigma_i: float = 0.05 * np.sqrt(2)
    u_e: float = 10.0
    u_i: float = 0.0
    seed: int = 7
    device: str = "cpu"
    dt: float = 0.0001
    init_steps: int = 333
    record_steps: int = 333
    steps_per_record: int = 1
    tau_e: float = 0.01
    tau_i: float = 0.01
    dot_size: float = 0.2
    drift_rate: float = 0.7
    angle: float = 0.0
    input_amplitude: float = 10.0

    def __post_init__(self) -> None:
        if self.N <= 0 or self.N % 2 != 1:
            raise ValueError("N must be a positive odd integer.")
        if self.K <= 0:
            raise ValueError("K must be positive.")
        if self.dt <= 0 or self.tau_e <= 0 or self.tau_i <= 0:
            raise ValueError("Time values must be positive.")
        if self.init_steps < 0 or self.record_steps <= 0:
            raise ValueError("Simulation step counts are invalid.")
        if self.steps_per_record <= 0:
            raise ValueError("steps_per_record must be positive.")
        if self.record_steps % self.steps_per_record != 0:
            raise ValueError("record_steps must be divisible by steps_per_record.")
        if self.dot_size <= 0:
            raise ValueError("dot_size must be positive.")

    @property
    def J0(self) -> np.ndarray:
        """Return the two-population coupling matrix."""
        return np.array([[self.J_ee, self.J_ei], [self.J_ie, self.J_ii]])

    @property
    def tau(self) -> np.ndarray:
        """Return population time constants."""
        return np.array([self.tau_e, self.tau_i])

    @property
    def drive(self) -> np.ndarray:
        """Return population background inputs."""
        return np.array([self.u_e, self.u_i])

    @property
    def sigma(self) -> np.ndarray:
        """Return population connection widths."""
        return np.array([self.sigma_e, self.sigma_i])


def driven_step(
    re: np.ndarray,
    ri: np.ndarray,
    config: DrivenDotConfig,
    input_pattern: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance the driven network by one recorded step."""
    device = torch.device(config.device)
    re_tensor = torch.tensor(re, dtype=torch.float32, device=device)[None, None]
    ri_tensor = torch.tensor(ri, dtype=torch.float32, device=device)[None, None]
    input_pattern = input_pattern.to(device)
    dx = 1 / config.N
    sigma = config.sigma
    wraps = np.arange(
        -int(np.ceil(10 * np.max(sigma))),
        int(np.ceil(10 * np.max(sigma))) + 1,
    )
    axis = np.arange(-(config.N - 1) // 2, (config.N - 1) // 2 + 1)
    kernel_e = np.sum(
        dx
        * (2 * np.pi * sigma[0] ** 2) ** -0.5
        * np.exp(-0.5 * (dx * (axis[:, None] + wraps)) ** 2 / sigma[0] ** 2),
        axis=1,
    )
    kernel_i = np.sum(
        dx
        * (2 * np.pi * sigma[1] ** 2) ** -0.5
        * np.exp(-0.5 * (dx * (axis[:, None] + wraps)) ** 2 / sigma[1] ** 2),
        axis=1,
    )
    weight_e = torch.tensor(
        np.outer(kernel_e, kernel_e), dtype=torch.float32, device=device
    )[None, None]
    weight_i = torch.tensor(
        np.outer(kernel_i, kernel_i), dtype=torch.float32, device=device
    )[None, None]
    pad = (weight_e.shape[-1] // 2, weight_e.shape[-2] // 2)
    J0 = config.J0
    drive = config.drive

    for _ in range(config.steps_per_record):
        padded_e = F.pad(re_tensor, (pad[0], pad[0], pad[1], pad[1]), mode="circular")
        padded_i = F.pad(ri_tensor, (pad[0], pad[0], pad[1], pad[1]), mode="circular")
        conv_e = F.conv2d(padded_e, weight_e)
        conv_i = F.conv2d(padded_i, weight_i)
        mu_e = config.K**0.5 * (
            drive[0] + J0[0, 0] * conv_e + J0[0, 1] * conv_i + input_pattern
        )
        mu_i = config.K**0.5 * (
            drive[1] + J0[1, 0] * conv_e + J0[1, 1] * conv_i
        )
        re_tensor += (config.dt / config.tau_e) * (-re_tensor + torch.relu(mu_e))
        ri_tensor += (config.dt / config.tau_i) * (-ri_tensor + torch.relu(mu_i))

    return re_tensor.squeeze().cpu().numpy(), ri_tensor.squeeze().cpu().numpy()


def initial_rates(
    config: DrivenDotConfig,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one random initial state near the balanced fixed rate."""
    fixed_rate = -np.linalg.inv(config.J0) @ config.drive
    re = fixed_rate[0] + 0.05 * rng.rand(config.N, config.N)
    ri = fixed_rate[1] + 0.08 * rng.rand(config.N, config.N)
    return re, ri


def simulate_driven(
    config: DrivenDotConfig,
    stimulus: torch.Tensor,
    re0: np.ndarray,
    ri0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return recorded excitatory and inhibitory activity."""
    record_count = config.record_steps // config.steps_per_record
    if stimulus.shape != (config.N, config.N, record_count):
        raise ValueError("Stimulus shape does not match the simulation configuration.")
    re = re0.copy()
    ri = ri0.copy()
    zero_input = torch.zeros((config.N, config.N), device=stimulus.device)
    init_count = config.init_steps // config.steps_per_record
    for _ in range(init_count):
        re, ri = driven_step(re, ri, config, zero_input)

    re_all = np.full((config.N, config.N, record_count), np.nan)
    ri_all = np.full((config.N, config.N, record_count), np.nan)
    for index in range(record_count):
        re, ri = driven_step(re, ri, config, stimulus[:, :, index])
        re_all[:, :, index] = re
        ri_all[:, :, index] = ri
    return re_all, ri_all
