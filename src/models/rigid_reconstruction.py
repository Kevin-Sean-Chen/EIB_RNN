"""Fixed spatial reservoir for rigid-shift reconstruction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.working_memory import periodic_gaussian_kernel
from src.tasks.rigid_reconstruction import RigidReconstructionConfig


class RigidReconstructionReservoir(nn.Module):
    """Run fixed E/I dynamics with one RLS readout."""

    def __init__(self, config: RigidReconstructionConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_count = config.N**2 + 1
        device = torch.device(config.device)
        self.register_buffer("coupling", torch.tensor(config.coupling, dtype=torch.float32, device=device))
        self.register_buffer("kernel_e", periodic_gaussian_kernel(config.N, config.sigma_e, False, device))
        self.register_buffer("kernel_i", periodic_gaussian_kernel(config.N, config.sigma_i, False, device))
        self.register_buffer("readout", torch.zeros(self.feature_count, 1, device=device))

    def initial_state(self, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        shape = (self.config.N, self.config.N)
        re = torch.randn(shape, generator=generator).to(self.readout.device) * self.config.init_scale
        ri = torch.randn(shape, generator=generator).to(self.readout.device) * self.config.init_scale
        return re, ri

    @torch.no_grad()
    def step(self, state, stimulus: torch.Tensor):
        re, ri = state
        pad = self.kernel_e.shape[-1] // 2
        local_e = F.conv2d(F.pad(re[None, None], (pad, pad, pad, pad), mode="circular"), self.kernel_e).squeeze()
        local_i = F.conv2d(F.pad(ri[None, None], (pad, pad, pad, pad), mode="circular"), self.kernel_i).squeeze()
        baseline_e = 0.0 if self.config.baseline_mode == "replace" else self.config.u_e
        field_e = self.config.K**0.5 * (
            baseline_e + self.coupling[0, 0] * local_e
            + self.coupling[0, 1] * local_i + self.config.stimulus_gain * stimulus
        )
        field_i = self.config.K**0.5 * (
            self.config.u_i + self.coupling[1, 0] * local_e + self.coupling[1, 1] * local_i
        )
        if self.config.field_clip is not None:
            field_e = torch.clamp(field_e, -self.config.field_clip, self.config.field_clip)
            field_i = torch.clamp(field_i, -self.config.field_clip, self.config.field_clip)
        re = re + self.config.dt / self.config.tau_e * (-re + F.relu(field_e))
        ri = ri + self.config.dt / self.config.tau_i * (-ri + F.relu(field_i))
        return re, ri

    def features(self, state) -> torch.Tensor:
        return torch.cat([F.relu(state[0]).reshape(-1), torch.ones(1, device=self.readout.device)])

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.readout
