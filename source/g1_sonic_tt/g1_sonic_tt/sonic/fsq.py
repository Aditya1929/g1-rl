"""Token squashing / FSQ quantization shared by training, BC and deploy."""
from __future__ import annotations

import torch

from .constants import FSQ_MAX, FSQ_MIN, FSQ_STEP


def squash(a: torch.Tensor) -> torch.Tensor:
    """Absolute mode: policy output (unbounded) -> continuous token in (-1, 1) (quantize clips the top)."""
    return torch.tanh(a)


def unsquash(z: torch.Tensor, eps: float = 0.98) -> torch.Tensor:
    """Inverse of squash, clipped so tokens at -1 stay finite (-0.98 rounds back to -1)."""
    return torch.atanh(torch.clamp(z, -eps, eps))


def quantize(z: torch.Tensor) -> torch.Tensor:
    """Snap to SONIC's FSQ grid: 32 levels, step 1/16, in [-1, 15/16]."""
    return torch.clamp(torch.round(z / FSQ_STEP) * FSQ_STEP, FSQ_MIN, FSQ_MAX)
