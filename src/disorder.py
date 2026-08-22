"""Simulate a spatial E/I network with low-rank disorder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


DISORDER_SECTIONS = {
    "network": (
        "N", "K", "J_ee", "J_ei", "J_ie", "J_ii",
        "sigma_e", "sigma_i", "u_e", "u_i",
    ),
    "simulation": (
        "seed", "device", "dt", "init_steps", "record_steps",
        "steps_per_record", "tau_e", "tau_i",
    ),
    "disorder": (
        "strength", "rank", "pattern_type", "frequency", "angle",
        "phase_offset", "aspect_ratio", "legacy_dense_modulation",
    ),
}


@dataclass
class DisorderConfig:
    """Define one low-rank disorder simulation."""

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
    strength: float = 0.5
    rank: int = 2
    pattern_type: str = "random"
    frequency: float = 5.0
    angle: float = 30.0
    phase_offset: float = 0.5
    aspect_ratio: float = 0.1
    legacy_dense_modulation: bool = False

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
        if self.rank not in (1, 2):
            raise ValueError("rank must be 1 or 2.")
        if self.pattern_type not in ("random", "gabor"):
            raise ValueError("pattern_type must be 'random' or 'gabor'.")

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
class DisorderResult:
    """Store one disorder simulation."""

    excitatory: np.ndarray
    inhibitory: np.ndarray
    excitatory_field: np.ndarray
    inhibitory_field: np.ndarray
    left_patterns: np.ndarray
    right_patterns: np.ndarray


def gabor_pattern(
    N: int,
    frequency: float,
    angle_degrees: float,
    phase: float,
    aspect_ratio: float,
) -> np.ndarray:
    """Return one normalized Gabor pattern."""
    axis = np.arange(N) - (N - 1) / 2
    y, x = np.meshgrid(axis, axis, indexing="ij")
    angle = np.deg2rad(angle_degrees)
    rotated_x = x * np.cos(angle) + y * np.sin(angle)
    rotated_y = -x * np.sin(angle) + y * np.cos(angle)
    sigma_x = 0.2 * N
    sigma_y = sigma_x / aspect_ratio
    envelope = np.exp(
        -0.5 * ((rotated_x / sigma_x) ** 2 + (rotated_y / sigma_y) ** 2)
    )
    carrier = np.cos(2 * np.pi * frequency * rotated_x / N + phase)
    pattern = envelope * carrier
    maximum = np.max(np.abs(pattern))
    return pattern / maximum if maximum > 0 else pattern


def make_disorder_patterns(config: DisorderConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Return left and right low-rank patterns with shape (N squared, rank)."""
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    size = config.N**2
    if config.pattern_type == "random":
        first_left = torch.randint(0, 2, (size, 1), generator=generator).float() * 2 - 1
        left_columns = [first_left]
        if config.rank == 2:
            candidate = torch.randn(size, 1, generator=generator)
            candidate -= (first_left.T @ candidate) / (first_left.T @ first_left) * first_left
            second_left = torch.sign(candidate)
            second_left[second_left == 0] = 1
            left_columns.append(second_left)
        left = torch.cat(left_columns, dim=1)
        right_raw = torch.randn(size, config.rank, generator=generator)
        right, _ = torch.linalg.qr(right_raw, mode="reduced")
    else:
        phases = [0.0, config.phase_offset]
        left_columns = []
        right_columns = []
        for index in range(config.rank):
            left_phase = phases[index]
            right_phase = left_phase + config.phase_offset
            left_columns.append(
                gabor_pattern(
                    config.N, config.frequency, config.angle, left_phase,
                    config.aspect_ratio,
                ).reshape(-1)
            )
            right_columns.append(
                gabor_pattern(
                    config.N, config.frequency, config.angle, right_phase,
                    config.aspect_ratio,
                ).reshape(-1)
            )
        left = torch.tensor(np.column_stack(left_columns), dtype=torch.float32)
        right = torch.tensor(np.column_stack(right_columns), dtype=torch.float32)
    return left, right * config.strength


def _kernels(config: DisorderConfig) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
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
        kernels.append(
            torch.tensor(np.outer(kernel_1d, kernel_1d), dtype=torch.float32, device=device)[None, None]
        )
    pad = (kernels[0].shape[-1] // 2, kernels[0].shape[-2] // 2)
    return kernels[0], kernels[1], pad


def initial_rates(
    config: DisorderConfig,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """Return initial rates near the balanced fixed rate."""
    fixed_rate = -np.linalg.inv(config.coupling) @ config.drive
    excitatory = fixed_rate[0] + 0.05 * rng.rand(config.N, config.N)
    inhibitory = fixed_rate[1] + 0.08 * rng.rand(config.N, config.N)
    return excitatory, inhibitory


def simulate_disorder(config: DisorderConfig) -> DisorderResult:
    """Run one low-rank disorder simulation."""
    device = torch.device(config.device)
    rng = np.random.RandomState(config.seed)
    excitatory, inhibitory = initial_rates(config, rng)
    re = torch.tensor(excitatory, dtype=torch.float32, device=device)[None, None]
    ri = torch.tensor(inhibitory, dtype=torch.float32, device=device)[None, None]
    left, right = make_disorder_patterns(config)
    left = left.to(device)
    right = right.to(device)
    kernel_e, kernel_i, pad = _kernels(config)
    dense_e = None
    if config.legacy_dense_modulation:
        eye = torch.eye(config.N**2, dtype=torch.float32, device=device)
        basis = eye.reshape(config.N**2, 1, config.N, config.N)
        responses = F.conv2d(
            F.pad(basis, (pad[0], pad[0], pad[1], pad[1]), mode="circular"),
            kernel_e,
        )
        dense_e = responses.reshape(config.N**2, config.N**2).T
        first_left = left[:, 0]
        row_modulation = torch.outer(first_left, first_left) / config.N
        dense_e = dense_e - dense_e * row_modulation
    coupling = config.coupling
    drive = config.drive
    record_count = config.record_steps // config.steps_per_record
    total_records = config.init_steps // config.steps_per_record + record_count
    arrays = [np.empty((config.N, config.N, record_count)) for _ in range(4)]

    for record_index in range(total_records):
        for _ in range(config.steps_per_record):
            padded_e = F.pad(re, (pad[0], pad[0], pad[1], pad[1]), mode="circular")
            padded_i = F.pad(ri, (pad[0], pad[0], pad[1], pad[1]), mode="circular")
            if dense_e is None:
                local_e = F.conv2d(padded_e, kernel_e)
            else:
                local_e = (dense_e @ re.flatten()).reshape(config.N, config.N)
            local_i = F.conv2d(padded_i, kernel_i)
            disorder = left @ (right.T @ re.flatten()) / config.N
            effective_e = local_e + disorder.reshape(config.N, config.N)
            field_e = config.K**0.5 * (
                drive[0] + coupling[0, 0] * effective_e + coupling[0, 1] * local_i
            )
            field_i = config.K**0.5 * (
                drive[1] + coupling[1, 0] * effective_e + coupling[1, 1] * local_i
            )
            re += (config.dt / config.tau_e) * (-re + torch.relu(field_e))
            ri += (config.dt / config.tau_i) * (-ri + torch.relu(field_i))
        output_index = record_index - config.init_steps // config.steps_per_record
        if output_index >= 0:
            for target, value in zip(arrays, (re, ri, field_e, field_i)):
                target[:, :, output_index] = value.squeeze().detach().cpu().numpy()

    return DisorderResult(
        excitatory=arrays[0],
        inhibitory=arrays[1],
        excitatory_field=arrays[2],
        inhibitory_field=arrays[3],
        left_patterns=left.detach().cpu().numpy(),
        right_patterns=right.detach().cpu().numpy(),
    )
