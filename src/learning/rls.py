"""Recursive least-squares updates for readout learning."""

from __future__ import annotations

import torch


def initialize_inverse_correlation(
    feature_count: int,
    delta: float = 1e-2,
    device: str = "cpu",
) -> torch.Tensor:
    """Return the initial inverse feature-correlation estimate."""
    if feature_count <= 0:
        raise ValueError("feature_count must be positive.")
    if delta <= 0:
        raise ValueError("delta must be positive.")
    return torch.eye(feature_count, device=device) / delta


@torch.no_grad()
def update_rls(
    weights: torch.Tensor,
    inverse_correlation: torch.Tensor,
    features: torch.Tensor,
    error: torch.Tensor,
    forgetting_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one multi-output recursive least-squares update."""
    if not 0 < forgetting_factor <= 1:
        raise ValueError("forgetting_factor must be in (0, 1].")
    projected = inverse_correlation @ features
    denominator = (
        forgetting_factor + features @ projected
    ).clamp_min(1e-12)
    gain = projected / denominator
    weights = weights - torch.outer(gain, error)
    inverse_correlation = (
        inverse_correlation
        - torch.outer(gain, features @ inverse_correlation)
    ) / forgetting_factor
    return weights, inverse_correlation
