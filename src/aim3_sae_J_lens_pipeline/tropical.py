"""Tropical (categorical) distances d_M(x,y) = -ln π(y|x). No cosine / Euclidean."""

from __future__ import annotations

import math

import torch

from .config import TROPICAL_EPS


def tropical_distance_from_logprob(logprob: float, *, eps: float = TROPICAL_EPS) -> float:
    """d_M = -ln π(y|x). ``logprob`` is already ln π; clamp for numerical safety."""
    lp = float(logprob)
    if not math.isfinite(lp):
        return float("inf")
    # ln π ≤ 0; clamp tiny π so distances stay finite.
    lp = max(lp, math.log(eps))
    return float(-lp)


def l1_normalize(v: torch.Tensor, *, eps: float = TROPICAL_EPS) -> torch.Tensor:
    v = torch.clamp(v.float(), min=0.0)
    z = float(v.sum())
    if z <= eps:
        return torch.zeros_like(v)
    return v / z


def sae_overlap_prob(v_x: torch.Tensor, v_y: torch.Tensor, *, eps: float = TROPICAL_EPS) -> float:
    """
    Treat L1-normalized SAE activations as distributions; inner product is the
    probability mass they jointly assign to the same features.
    """
    px = l1_normalize(v_x, eps=eps)
    py = l1_normalize(v_y, eps=eps)
    return float(torch.dot(px, py).clamp(min=0.0).item())


def sae_tropical_distance(
    v_x: torch.Tensor, v_y: torch.Tensor, *, eps: float = TROPICAL_EPS
) -> float:
    """d_M on SAE features: -ln ⟨π_x, π_y⟩ (not cosine)."""
    p = sae_overlap_prob(v_x, v_y, eps=eps)
    return float(-math.log(max(p, eps)))


def triangle_slack(d_xy: float, d_yz: float, d_xz: float) -> float:
    """
    slack = d(x,z) - d(x,y) - d(y,z). Positive ⇒ ordinary triangle inequality
    is violated (tropical / additive tree metrics require slack ≤ 0).
    """
    return float(d_xz - d_xy - d_yz)


def reconstruct_sae_vector(
    indices: torch.Tensor,
    values: torch.Tensor,
    d_sae: int,
) -> torch.Tensor:
    vec = torch.zeros(int(d_sae), dtype=torch.float32)
    if indices.numel() == 0:
        return vec
    vec[indices.long()] = values.float()
    return vec
