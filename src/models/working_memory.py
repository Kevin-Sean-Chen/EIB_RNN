"""Fixed spatial and non-spatial reservoirs for working memory."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def periodic_gaussian_kernel(
    N: int,
    sigma: float,
    normalize: bool,
    device: torch.device,
) -> torch.Tensor:
    """Return one separable periodic Gaussian kernel."""
    axis = torch.arange(-(N - 1) // 2, (N - 1) // 2 + 1, device=device)
    dx = 1.0 / N
    wrap = int(math.ceil(10.0 * sigma))
    shifts = torch.arange(-wrap, wrap + 1, device=device)
    distance = dx * (axis[:, None] + shifts[None, :])
    weights = (
        dx
        * torch.exp(-0.5 * (distance / sigma) ** 2)
        / (math.sqrt(2 * math.pi) * sigma)
    ).sum(dim=1)
    if normalize:
        weights = weights / (weights.sum() + 1e-12)
    return torch.outer(weights, weights)[None, None]


class SpatialWorkingMemoryReservoir(nn.Module):
    """Fixed spatial E/I dynamics with RLS-controlled readouts."""

    def __init__(
        self,
        N: int,
        dt: float,
        tau_e: float,
        tau_i: float,
        K: float,
        coupling: np.ndarray,
        drive: np.ndarray,
        sigma: np.ndarray,
        stimulus_gain: float,
        feedback_gain: float,
        feedback_scale: float,
        use_feedback_output: bool,
        use_feedback_memory: bool,
        init_scale: float,
        microsteps: int,
        field_clip: float | None,
        seed: int,
        normalize_kernel: bool = True,
        nonlinearity: str = "relu",
        device: str = "cpu",
    ) -> None:
        super().__init__()
        if nonlinearity not in ("relu", "tanh"):
            raise ValueError("nonlinearity must be 'relu' or 'tanh'.")
        self.N = N
        self.feature_count = N**2
        self.dt = dt
        self.tau_e = tau_e
        self.tau_i = tau_i
        self.sqrt_K = K**0.5
        self.stimulus_gain = stimulus_gain
        self.feedback_gain = feedback_gain
        self.use_feedback_output = use_feedback_output
        self.use_feedback_memory = use_feedback_memory
        self.init_scale = init_scale
        self.microsteps = microsteps
        self.field_clip = field_clip
        self.nonlinearity = nonlinearity
        self.device_name = device
        target_device = torch.device(device)
        self.register_buffer("coupling", torch.tensor(coupling, dtype=torch.float32, device=target_device))
        self.register_buffer("drive", torch.tensor(drive, dtype=torch.float32, device=target_device))
        self.register_buffer(
            "kernel_e",
            periodic_gaussian_kernel(N, float(sigma[0]), normalize_kernel, target_device),
        )
        self.register_buffer(
            "kernel_i",
            periodic_gaussian_kernel(N, float(sigma[1]), normalize_kernel, target_device),
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.register_buffer(
            "feedback_output",
            torch.randn(N, N, generator=generator).to(target_device) * feedback_scale,
        )
        self.register_buffer(
            "feedback_memory",
            torch.randn(N, N, generator=generator).to(target_device) * feedback_scale,
        )
        self.register_buffer("output_weights", torch.zeros(self.feature_count + 1, 1, device=target_device))
        self.register_buffer("memory_weights", torch.zeros(self.feature_count + 1, 1, device=target_device))

    def activation(self, value: torch.Tensor) -> torch.Tensor:
        return F.relu(value) if self.nonlinearity == "relu" else torch.tanh(value)

    def initial_state(self, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        re = torch.randn(self.N, self.N, generator=generator).to(self.coupling.device) * self.init_scale
        ri = torch.randn(self.N, self.N, generator=generator).to(self.coupling.device) * self.init_scale
        return re, ri

    def predict(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        augmented = torch.cat([features, torch.ones(1, device=features.device)])
        return augmented @ self.output_weights, augmented @ self.memory_weights

    @torch.no_grad()
    def step(
        self,
        state: tuple[torch.Tensor, torch.Tensor],
        stimulus: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        re, ri = state
        pad = self.kernel_e.shape[-1] // 2
        for _ in range(self.microsteps):
            re_image = re[None, None]
            ri_image = ri[None, None]
            local_e = F.conv2d(F.pad(re_image, (pad, pad, pad, pad), mode="circular"), self.kernel_e).squeeze()
            local_i = F.conv2d(F.pad(ri_image, (pad, pad, pad, pad), mode="circular"), self.kernel_i).squeeze()
            features = self.activation(re).reshape(-1)
            output, memory = self.predict(features)
            feedback = torch.zeros_like(re)
            if self.use_feedback_output:
                feedback = feedback + self.feedback_output * output.squeeze()
            if self.use_feedback_memory:
                feedback = feedback + self.feedback_memory * memory.squeeze()
            field_e = self.sqrt_K * (
                self.drive[0]
                + self.coupling[0, 0] * local_e
                + self.coupling[0, 1] * local_i
                + self.stimulus_gain * stimulus
            )
            field_i = self.sqrt_K * (
                self.drive[1]
                + self.coupling[1, 0] * local_e
                + self.coupling[1, 1] * local_i
            )
            # The legacy model applies feedback after the recurrent-field clamp.
            if self.field_clip is not None:
                field_e = torch.clamp(field_e, -self.field_clip, self.field_clip)
                field_i = torch.clamp(field_i, -self.field_clip, self.field_clip)
            field_e = field_e + self.feedback_gain * feedback
            re = re + (self.dt / self.tau_e) * (-re + self.activation(field_e))
            ri = ri + (self.dt / self.tau_i) * (-ri + self.activation(field_i))
        return re, ri

    def features(self, state: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.activation(state[0]).reshape(-1)


class NonSpatialWorkingMemoryReservoir(nn.Module):
    """Fixed random recurrent dynamics with RLS-controlled readouts."""

    def __init__(
        self,
        unit_count: int,
        dt: float,
        tau: float,
        recurrent_gain: float,
        stimulus_gain: float,
        feedback_gain: float,
        feedback_scale: float,
        init_scale: float,
        seed: int,
        nonlinearity: str = "relu",
        device: str = "cpu",
    ) -> None:
        super().__init__()
        if nonlinearity not in ("relu", "tanh"):
            raise ValueError("nonlinearity must be 'relu' or 'tanh'.")
        self.feature_count = unit_count
        self.dt = dt
        self.tau = tau
        self.feedback_gain = feedback_gain
        self.stimulus_gain = stimulus_gain
        self.init_scale = init_scale
        self.nonlinearity = nonlinearity
        target_device = torch.device(device)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        recurrent = torch.randn(unit_count, unit_count, generator=generator)
        recurrent *= recurrent_gain / math.sqrt(unit_count)
        self.register_buffer("recurrent", recurrent.to(target_device))
        self.register_buffer(
            "feedback_output",
            torch.randn(unit_count, generator=generator).to(target_device) * feedback_scale,
        )
        self.register_buffer(
            "feedback_memory",
            torch.randn(unit_count, generator=generator).to(target_device) * feedback_scale,
        )
        self.register_buffer("output_weights", torch.zeros(unit_count + 1, 1, device=target_device))
        self.register_buffer("memory_weights", torch.zeros(unit_count + 1, 1, device=target_device))

    def activation(self, value: torch.Tensor) -> torch.Tensor:
        return F.relu(value) if self.nonlinearity == "relu" else torch.tanh(value)

    def initial_state(self, seed: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(self.feature_count, generator=generator).to(self.recurrent.device) * self.init_scale

    def predict(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        augmented = torch.cat([features, torch.ones(1, device=features.device)])
        return augmented @ self.output_weights, augmented @ self.memory_weights

    @torch.no_grad()
    def step(self, state: torch.Tensor, stimulus: torch.Tensor) -> torch.Tensor:
        features = self.activation(state)
        output, memory = self.predict(features)
        feedback = self.feedback_output * output.squeeze() + self.feedback_memory * memory.squeeze()
        field = self.recurrent @ state + self.stimulus_gain * stimulus + self.feedback_gain * feedback
        return state + (self.dt / self.tau) * (-state + self.activation(field))

    def features(self, state: torch.Tensor) -> torch.Tensor:
        return self.activation(state)
