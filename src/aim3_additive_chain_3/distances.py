"""Tropical distances on J-lens features along a shared trajectory."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .config import TROPICAL_EPS


def l1_normalize(v: torch.Tensor, *, eps: float = TROPICAL_EPS) -> torch.Tensor:
    v = torch.clamp(v.float(), min=0.0)
    z = float(v.sum())
    if z <= eps:
        return torch.zeros_like(v)
    return v / z


def j_feature_distribution(z: torch.Tensor, *, eps: float = TROPICAL_EPS) -> torch.Tensor:
    """
    Map raw J-space coords to a non-negative distribution.

    softplus keeps causal signed effects while yielding a usable π for d_M.
    """
    return l1_normalize(F.softplus(z.float()), eps=eps)


def j_tropical_distance(
    z_x: torch.Tensor, z_y: torch.Tensor, *, eps: float = TROPICAL_EPS
) -> float:
    px = j_feature_distribution(z_x, eps=eps)
    py = j_feature_distribution(z_y, eps=eps)
    p = float(torch.dot(px, py).clamp(min=0.0).item())
    return float(-math.log(max(p, eps)))


def type_a_j_distances(j_vecs: list[torch.Tensor]) -> dict[str, float]:
    assert len(j_vecs) >= 4
    d12 = j_tropical_distance(j_vecs[0], j_vecs[1])
    d23 = j_tropical_distance(j_vecs[1], j_vecs[2])
    d34 = j_tropical_distance(j_vecs[2], j_vecs[3])
    d14 = j_tropical_distance(j_vecs[0], j_vecs[3])
    hops = d12 + d23 + d34
    return {
        "n1_n2": d12,
        "n2_n3": d23,
        "n3_n4": d34,
        "n1_n4": d14,
        "sum_hops": hops,
        "additive_gap": d14 - hops,
    }


def type_b_j_distances(j_vecs: list[torch.Tensor]) -> dict[str, float]:
    assert len(j_vecs) >= 3
    d_bs = j_tropical_distance(j_vecs[0], j_vecs[1])
    d_sc = j_tropical_distance(j_vecs[1], j_vecs[2])
    d_bc = j_tropical_distance(j_vecs[0], j_vecs[2])
    return {
        "base_shift": d_bs,
        "shift_combined": d_sc,
        "base_combined": d_bc,
        "triangle_slack": d_bc - d_bs - d_sc,
    }


def type_c_j_distances(j_vecs: list[torch.Tensor]) -> dict[str, float]:
    assert len(j_vecs) >= 3
    d_es = j_tropical_distance(j_vecs[0], j_vecs[1])
    d_fe = j_tropical_distance(j_vecs[2], j_vecs[0])
    d_fs = j_tropical_distance(j_vecs[2], j_vecs[1])
    return {
        "entailed_superordinate": d_es,
        "framed_entailed": d_fe,
        "framed_superordinate": d_fs,
        "triangle_slack": d_es - d_fe - d_fs,
    }
