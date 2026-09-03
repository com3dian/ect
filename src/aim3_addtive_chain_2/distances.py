"""Tropical distances on a shared trajectory (SAE + logit-lens)."""

from __future__ import annotations

import math
from typing import Any

import torch

from .config import TROPICAL_EPS
from .model_io import logit_lens_logprob_for_text


def tropical_distance_from_logprob(logprob: float, *, eps: float = TROPICAL_EPS) -> float:
    lp = float(logprob)
    if not math.isfinite(lp):
        return float("inf")
    lp = max(lp, math.log(eps))
    return float(-lp)


def l1_normalize(v: torch.Tensor, *, eps: float = TROPICAL_EPS) -> torch.Tensor:
    v = torch.clamp(v.float(), min=0.0)
    z = float(v.sum())
    if z <= eps:
        return torch.zeros_like(v)
    return v / z


def sae_tropical_distance(
    v_x: torch.Tensor, v_y: torch.Tensor, *, eps: float = TROPICAL_EPS
) -> float:
    px = l1_normalize(v_x, eps=eps)
    py = l1_normalize(v_y, eps=eps)
    p = float(torch.dot(px, py).clamp(min=0.0).item())
    return float(-math.log(max(p, eps)))


def type_a_sae_distances(sae_vecs: list[torch.Tensor]) -> dict[str, float]:
    """Path hops + shortcut on SAE states at n1..n4 (same forward)."""
    assert len(sae_vecs) >= 4
    d12 = sae_tropical_distance(sae_vecs[0], sae_vecs[1])
    d23 = sae_tropical_distance(sae_vecs[1], sae_vecs[2])
    d34 = sae_tropical_distance(sae_vecs[2], sae_vecs[3])
    d14 = sae_tropical_distance(sae_vecs[0], sae_vecs[3])
    hops = d12 + d23 + d34
    return {
        "n1_n2": d12,
        "n2_n3": d23,
        "n3_n4": d34,
        "n1_n4": d14,
        "sum_hops": hops,
        "additive_gap": d14 - hops,
    }


def type_a_logit_lens_distances(
    model: Any,
    tokenizer: Any,
    residuals: list[torch.Tensor],
    labels: tuple[str, ...],
    *,
    device: torch.device,
) -> dict[str, float]:
    """
    From residual at node i, logit-lens score of label j (shared trajectory states).

    Hop i→i+1: state_i predicts label_{i+1}.
    Shortcut 1→4: state_1 predicts label_4.
    """
    assert len(residuals) >= 4 and len(labels) >= 4

    def d_state_to_label(i_state: int, j_label: int) -> float:
        lp = logit_lens_logprob_for_text(
            model,
            tokenizer,
            residuals[i_state],
            labels[j_label],
            device=device,
        )
        return tropical_distance_from_logprob(lp)

    d12 = d_state_to_label(0, 1)
    d23 = d_state_to_label(1, 2)
    d34 = d_state_to_label(2, 3)
    d14 = d_state_to_label(0, 3)
    hops = d12 + d23 + d34
    return {
        "n1_n2": d12,
        "n2_n3": d23,
        "n3_n4": d34,
        "n1_n4": d14,
        "sum_hops": hops,
        "additive_gap": d14 - hops,
    }


def type_b_sae_distances(sae_vecs: list[torch.Tensor]) -> dict[str, float]:
    assert len(sae_vecs) >= 3
    d_bs = sae_tropical_distance(sae_vecs[0], sae_vecs[1])
    d_sc = sae_tropical_distance(sae_vecs[1], sae_vecs[2])
    d_bc = sae_tropical_distance(sae_vecs[0], sae_vecs[2])
    return {
        "base_shift": d_bs,
        "shift_combined": d_sc,
        "base_combined": d_bc,
        "triangle_slack": d_bc - d_bs - d_sc,
    }


def type_c_sae_distances(sae_vecs: list[torch.Tensor]) -> dict[str, float]:
    assert len(sae_vecs) >= 3
    # entailed, superordinate, framed
    d_es = sae_tropical_distance(sae_vecs[0], sae_vecs[1])
    d_fe = sae_tropical_distance(sae_vecs[2], sae_vecs[0])
    d_fs = sae_tropical_distance(sae_vecs[2], sae_vecs[1])
    return {
        "entailed_superordinate": d_es,
        "framed_entailed": d_fe,
        "framed_superordinate": d_fs,
        "triangle_slack": d_es - d_fe - d_fs,
    }
