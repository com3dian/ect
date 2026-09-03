"""
SAE composition metrics: TVD of empirical AND/OR vs min/mean/max/prod baselines,
plus Jaccard similarity over active feature indices of V_A and V_B.
"""

from __future__ import annotations

from typing import Any

import torch

from .config import JACCARD_THRESHOLD


def l1_normalize(v: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Normalize a non-negative vector to sum to 1."""
    v = torch.clamp(v.float(), min=0.0)
    total = float(v.sum().item())
    if total <= eps:
        return torch.zeros_like(v)
    return v / total


def sparse_topk(v: torch.Tensor, top_k: int) -> dict[str, list]:
    """
    L1-normalize, then keep the Top-K features by activation mass.

    Returns JSON-friendly {"feature_ids": [...], "values": [...]} of length ≤ top_k
    (trailing zeros are dropped). Values sum to ≤ 1.
    """
    p = l1_normalize(v)
    k = min(int(top_k), int(p.numel()))
    if k <= 0:
        return {"feature_ids": [], "values": []}
    vals, idxs = torch.topk(p, k)
    mask = vals > 0
    return {
        "feature_ids": [int(i) for i in idxs[mask].tolist()],
        "values": [float(x) for x in vals[mask].tolist()],
    }


def pointwise_min(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.minimum(a, b)


def pointwise_max(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.maximum(a, b)


def pointwise_mean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 0.5 * (a + b)


def pointwise_prod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a * b


def total_variation_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    """TVD = 0.5 * L1 for probability-like non-negative vectors."""
    return float(0.5 * torch.sum(torch.abs(p - q)).item())


def active_feature_set(v: torch.Tensor, *, threshold: float = JACCARD_THRESHOLD) -> set[int]:
    idxs = (v > threshold).nonzero(as_tuple=False).view(-1)
    return {int(i) for i in idxs.tolist()}


def jaccard_similarity(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return float(len(a & b) / len(union))


def compute_composition_metrics(
    v_a: torch.Tensor,
    v_b: torch.Tensor,
    v_and: torch.Tensor,
    v_or: torch.Tensor,
    *,
    jaccard_threshold: float = JACCARD_THRESHOLD,
) -> dict[str, Any]:
    """
    L1-normalize all vectors, form theoretical baselines, score TVD + Jaccard.

    Returns flat float metrics suitable for CSV export.
    """
    a = l1_normalize(v_a)
    b = l1_normalize(v_b)
    and_v = l1_normalize(v_and)
    or_v = l1_normalize(v_or)

    baselines = {
        "min": l1_normalize(pointwise_min(a, b)),
        "max": l1_normalize(pointwise_max(a, b)),
        "mean": l1_normalize(pointwise_mean(a, b)),
        "prod": l1_normalize(pointwise_prod(a, b)),
    }

    out: dict[str, Any] = {
        "n_active_a": float(len(active_feature_set(a, threshold=jaccard_threshold))),
        "n_active_b": float(len(active_feature_set(b, threshold=jaccard_threshold))),
        "n_active_and": float(len(active_feature_set(and_v, threshold=jaccard_threshold))),
        "n_active_or": float(len(active_feature_set(or_v, threshold=jaccard_threshold))),
        "jaccard_a_b": jaccard_similarity(
            active_feature_set(a, threshold=jaccard_threshold),
            active_feature_set(b, threshold=jaccard_threshold),
        ),
        "min_raw_mass": float(pointwise_min(a, b).sum().item()),
        "max_raw_mass": float(pointwise_max(a, b).sum().item()),
        "prod_raw_mass": float(pointwise_prod(a, b).sum().item()),
    }

    for name, theory in baselines.items():
        out[f"tvd_and_vs_{name}"] = total_variation_distance(and_v, theory)
        out[f"tvd_or_vs_{name}"] = total_variation_distance(or_v, theory)

    return out
