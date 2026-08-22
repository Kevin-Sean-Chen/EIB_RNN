"""Simulate the baseline local spatial E/I network."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


LOCAL_SECTIONS = {
    "network": (
        "N", "K", "J_ee", "J_ei", "J_ie", "J_ii",
        "sigma_e", "sigma_i", "u_e", "u_i",
    ),
    "simulation": (
        "seed", "device", "dt", "init_steps", "record_steps",
        "steps_per_record", "tau_e", "tau_i",
    ),
}


@dataclass
class LocalConfig:
    """Define one baseline local-network simulation."""

    N: int = 31
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
    init_steps: int = 1000
    record_steps: int = 2000
    steps_per_record: int = 5
    tau_e: float = 0.01
    tau_i: float = 0.01

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
        if self.init_steps % self.steps_per_record != 0:
            raise ValueError("init_steps must be divisible by steps_per_record.")
        if self.record_steps % self.steps_per_record != 0:
            raise ValueError("record_steps must be divisible by steps_per_record.")

    @property
    def coupling(self) -> np.ndarray:
        return np.array([[self.J_ee, self.J_ei], [self.J_ie, self.J_ii]])

    @property
    def drive(self) -> np.ndarray:
        return np.array([self.u_e, self.u_i])

    @property
    def sigma(self) -> np.ndarray:
        return np.array([self.sigma_e, self.sigma_i])


@dataclass
class LocalResult:
    """Store one baseline simulation."""

    excitatory: np.ndarray
    inhibitory: np.ndarray
    time: np.ndarray


def _kernels(config: LocalConfig) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    device = torch.device(config.device)
    dx = 1 / config.N
    wraps = np.arange(
        -int(np.ceil(10 * np.max(config.sigma))),
        int(np.ceil(10 * np.max(config.sigma))) + 1,
    )
    axis = np.arange(-(config.N - 1) // 2, (config.N - 1) // 2 + 1)
    kernels = []
    for width in config.sigma:
        kernel_1d = np.sum(
            dx * (2 * np.pi * width**2) ** -0.5
            * np.exp(-0.5 * (dx * (axis[:, None] + wraps)) ** 2 / width**2),
            axis=1,
        )
        kernel = torch.tensor(
            np.outer(kernel_1d, kernel_1d), dtype=torch.float32, device=device
        )[None, None]
        kernels.append(kernel)
    pad = (kernels[0].shape[-1] // 2, kernels[0].shape[-2] // 2)
    return kernels[0], kernels[1], pad


def initial_rates(
    config: LocalConfig,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """Return initial rates near the balanced fixed rate."""
    fixed_rate = -np.linalg.inv(config.coupling) @ config.drive
    excitatory = fixed_rate[0] + 0.05 * rng.rand(config.N, config.N)
    inhibitory = fixed_rate[1] + 0.08 * rng.rand(config.N, config.N)
    return excitatory, inhibitory


def simulate_local(config: LocalConfig) -> LocalResult:
    """Run one baseline local-network simulation."""
    device = torch.device(config.device)
    rng = np.random.RandomState(config.seed)
    excitatory, inhibitory = initial_rates(config, rng)
    re = torch.tensor(excitatory, dtype=torch.float32, device=device)[None, None]
    ri = torch.tensor(inhibitory, dtype=torch.float32, device=device)[None, None]
    kernel_e, kernel_i, pad = _kernels(config)
    coupling = config.coupling
    drive = config.drive
    init_records = config.init_steps // config.steps_per_record
    record_count = config.record_steps // config.steps_per_record
    re_all = np.empty((config.N, config.N, record_count))
    ri_all = np.empty((config.N, config.N, record_count))

    for record_index in range(init_records + record_count):
        for _ in range(config.steps_per_record):
            padded_e = F.pad(re, (pad[0], pad[0], pad[1], pad[1]), mode="circular")
            padded_i = F.pad(ri, (pad[0], pad[0], pad[1], pad[1]), mode="circular")
            local_e = F.conv2d(padded_e, kernel_e)
            local_i = F.conv2d(padded_i, kernel_i)
            field_e = config.K**0.5 * (
                drive[0] + coupling[0, 0] * local_e + coupling[0, 1] * local_i
            )
            field_i = config.K**0.5 * (
                drive[1] + coupling[1, 0] * local_e + coupling[1, 1] * local_i
            )
            re += (config.dt / config.tau_e) * (-re + torch.relu(field_e))
            ri += (config.dt / config.tau_i) * (-ri + torch.relu(field_i))
        output_index = record_index - init_records
        if output_index >= 0:
            re_all[:, :, output_index] = re.squeeze().cpu().numpy()
            ri_all[:, :, output_index] = ri.squeeze().cpu().numpy()

    time = config.dt * config.steps_per_record * np.arange(1, record_count + 1)
    return LocalResult(re_all, ri_all, time)
