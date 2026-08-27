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


class NonSpatialRigidReconstructionReservoir(nn.Module):
    """Run a same-size random ReLU reservoir with one RLS readout."""

    def __init__(self, config: RigidReconstructionConfig) -> None:
        super().__init__()
        self.config = config
        unit_count = config.N**2
        self.feature_count = unit_count + 1
        device = torch.device(config.device)
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        mask = torch.rand(unit_count, unit_count, generator=generator) < config.recurrent_sparsity
        recurrent = torch.randn(unit_count, unit_count, generator=generator)
        recurrent *= mask
        connected_count = mask.sum(dim=1, keepdim=True).clamp_min(1)
        connected_mean = recurrent.sum(dim=1, keepdim=True) / connected_count
        recurrent = (recurrent - connected_mean) * mask
        recurrent *= config.recurrent_gain / (config.recurrent_sparsity * unit_count) ** 0.5
        self.register_buffer("recurrent", recurrent.to(device))
        self.register_buffer("readout", torch.zeros(self.feature_count, 1, device=device))

    def initial_state(self, seed: int) -> tuple[torch.Tensor]:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        state = torch.randn(self.config.N**2, generator=generator).to(self.readout.device)
        return (state * self.config.init_scale,)

    @torch.no_grad()
    def step(self, state, stimulus: torch.Tensor):
        rate = state[0]
        baseline = 0.0 if self.config.baseline_mode == "replace" else self.config.u_e
        external_drive = self.config.K**0.5 * (
            baseline + self.config.stimulus_gain * stimulus.reshape(-1)
        )
        field = self.recurrent @ rate + external_drive
        if self.config.field_clip is not None:
            field = torch.clamp(field, -self.config.field_clip, self.config.field_clip)
        rate = rate + self.config.dt / self.config.tau_e * (-rate + F.relu(field))
        return (rate,)

    def features(self, state) -> torch.Tensor:
        return torch.cat([F.relu(state[0]), torch.ones(1, device=self.readout.device)])

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.readout


class RandomEIRigidReconstructionReservoir(nn.Module):
    """Run a strongly coupled random E/I ReLU reservoir."""

    def __init__(self, config: RigidReconstructionConfig) -> None:
        super().__init__()
        self.config = config
        unit_count = config.N**2
        degree = int(round(config.K))
        if degree <= 0 or degree > unit_count:
            raise ValueError("K must give a valid random E/I in-degree.")
        self.feature_count = unit_count + 1
        device = torch.device(config.device)
        for name, seed_offset in (("ee", 0), ("ei", 1), ("ie", 2), ("ii", 3)):
            generator = torch.Generator(device="cpu").manual_seed(config.seed + seed_offset)
            scores = torch.rand((unit_count, unit_count), generator=generator)
            indices = torch.topk(scores, degree, dim=1, sorted=False).indices
            connectivity = torch.zeros((unit_count, unit_count))
            connectivity.scatter_(1, indices, 1.0 / degree)
            self.register_buffer(f"connectivity_{name}", connectivity.to(device))
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
        flat_e = re.reshape(-1)
        flat_i = ri.reshape(-1)
        local_ee = self.connectivity_ee @ flat_e
        local_ei = self.connectivity_ei @ flat_i
        local_ie = self.connectivity_ie @ flat_e
        local_ii = self.connectivity_ii @ flat_i
        baseline_e = 0.0 if self.config.baseline_mode == "replace" else self.config.u_e
        field_e = self.config.K**0.5 * (
            baseline_e + self.config.J_ee * local_ee + self.config.J_ei * local_ei
            + self.config.stimulus_gain * stimulus.reshape(-1)
        )
        field_i = self.config.K**0.5 * (
            self.config.u_i + self.config.J_ie * local_ie + self.config.J_ii * local_ii
        )
        if self.config.field_clip is not None:
            field_e = torch.clamp(field_e, -self.config.field_clip, self.config.field_clip)
            field_i = torch.clamp(field_i, -self.config.field_clip, self.config.field_clip)
        flat_e = flat_e + self.config.dt / self.config.tau_e * (-flat_e + F.relu(field_e))
        flat_i = flat_i + self.config.dt / self.config.tau_i * (-flat_i + F.relu(field_i))
        return flat_e.reshape_as(re), flat_i.reshape_as(ri)

    def features(self, state) -> torch.Tensor:
        return torch.cat([F.relu(state[0]).reshape(-1), torch.ones(1, device=self.readout.device)])

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.readout
