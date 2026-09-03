"""
Path costs via Total Variation / half-L1 between consecutive fingerprints.

δ(a,b) = (1/2) Σ_i |π_a_i - π_b_i|

Path L = δ(n1,n2)+δ(n2,n3)+δ(n3,n4)
Direct D = δ(n1,n4)
Gap = D - L   (≤ 0 always for TVD; ≈ 0 ⇒ geodesic / additive path)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import TROPICAL_EPS


def l1_normalize(v: torch.Tensor, *, eps: float = TROPICAL_EPS) -> torch.Tensor:
    v = torch.clamp(v.float(), min=0.0)
    z = float(v.sum())
    if z <= eps:
        # uniform fallback so TVD stays well-defined
        return torch.full_like(v, 1.0 / max(1, v.numel()))
    return v / z


def to_fingerprint(v: torch.Tensor, *, kind: str = "sae") -> torch.Tensor:
    """Turn raw features into a probability-like fingerprint π."""
    if kind == "sae":
        return l1_normalize(v)
    # J-space: softplus then L1-normalize
    return l1_normalize(F.softplus(v.float()))


def tvd(pi: torch.Tensor, pj: torch.Tensor) -> float:
    """Total variation distance = half L1 between two fingerprints."""
    a = pi.float().reshape(-1)
    b = pj.float().reshape(-1)
    if a.numel() != b.numel():
        raise ValueError(f"fingerprint size mismatch {a.numel()} vs {b.numel()}")
    return float(0.5 * torch.sum(torch.abs(a - b)).item())


def type_a_tvd_path(fingerprints: list[torch.Tensor]) -> dict[str, float]:
    assert len(fingerprints) >= 4
    d12 = tvd(fingerprints[0], fingerprints[1])
    d23 = tvd(fingerprints[1], fingerprints[2])
    d34 = tvd(fingerprints[2], fingerprints[3])
    d14 = tvd(fingerprints[0], fingerprints[3])
    hops = d12 + d23 + d34
    return {
        "n1_n2": d12,
        "n2_n3": d23,
        "n3_n4": d34,
        "n1_n4": d14,
        "sum_hops": hops,
        "additive_gap": d14 - hops,
    }


def type_b_tvd_path(fingerprints: list[torch.Tensor]) -> dict[str, float]:
    assert len(fingerprints) >= 3
    d_bs = tvd(fingerprints[0], fingerprints[1])
    d_sc = tvd(fingerprints[1], fingerprints[2])
    d_bc = tvd(fingerprints[0], fingerprints[2])
    return {
        "base_shift": d_bs,
        "shift_combined": d_sc,
        "base_combined": d_bc,
        "triangle_slack": d_bc - d_bs - d_sc,
    }


def type_c_tvd_path(fingerprints: list[torch.Tensor]) -> dict[str, float]:
    assert len(fingerprints) >= 3
    d_es = tvd(fingerprints[0], fingerprints[1])
    d_fe = tvd(fingerprints[2], fingerprints[0])
    d_fs = tvd(fingerprints[2], fingerprints[1])
    return {
        "entailed_superordinate": d_es,
        "framed_entailed": d_fe,
        "framed_superordinate": d_fs,
        "triangle_slack": d_es - d_fe - d_fs,
    }
