"""
Discrete SAE set metrics: Jaccard of active feature indices vs ∩ / ∪ theories.
"""

from __future__ import annotations

from typing import Any

import torch

from .config import ACTIVATION_THRESHOLD


def get_active_feature_set(
    activation_vector: torch.Tensor,
    threshold: float = ACTIVATION_THRESHOLD,
) -> set[int]:
    """
    Binarize SAE activations: indices with activation > threshold.

    README uses a strict absolute threshold (default theta = 1.0) to drop
    background / syntax noise and keep strongly active monosemantic features.
    """
    v = activation_vector.float().view(-1)
    idxs = (v > float(threshold)).nonzero(as_tuple=False).view(-1)
    return {int(i) for i in idxs.tolist()}


def jaccard_similarity(a: set[int], b: set[int]) -> float:
    """|A ∩ B| / |A ∪ B|; empty∪empty → 1.0; empty union otherwise → 0.0."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return float(len(a & b) / len(union))


def compute_jaccard_metrics(
    v_a: torch.Tensor,
    v_b: torch.Tensor,
    v_and: torch.Tensor,
    v_or: torch.Tensor,
    *,
    threshold: float = ACTIVATION_THRESHOLD,
) -> dict[str, Any]:
    """
    Build F_A, F_B, F_AND, F_OR and score:

      jaccard_and_vs_min  = J(F_AND, F_A ∩ F_B)   # categorical limit
      jaccard_or_vs_max   = J(F_OR,  F_A ∪ F_B)   # categorical colimit
      jaccard_and_vs_max  = J(F_AND, F_A ∪ F_B)   # control (union blend)
      jaccard_or_vs_min   = J(F_OR,  F_A ∩ F_B)   # optional contrast
    """
    f_a = get_active_feature_set(v_a, threshold=threshold)
    f_b = get_active_feature_set(v_b, threshold=threshold)
    f_and = get_active_feature_set(v_and, threshold=threshold)
    f_or = get_active_feature_set(v_or, threshold=threshold)

    f_min = f_a & f_b  # intersection / limit
    f_max = f_a | f_b  # union / colimit

    out: dict[str, Any] = {
        "threshold": float(threshold),
        "n_active_a": float(len(f_a)),
        "n_active_b": float(len(f_b)),
        "n_active_and": float(len(f_and)),
        "n_active_or": float(len(f_or)),
        "n_intersection": float(len(f_min)),
        "n_union": float(len(f_max)),
        "jaccard_a_b": jaccard_similarity(f_a, f_b),
        "jaccard_and_vs_min": jaccard_similarity(f_and, f_min),
        "jaccard_and_vs_max": jaccard_similarity(f_and, f_max),
        "jaccard_or_vs_min": jaccard_similarity(f_or, f_min),
        "jaccard_or_vs_max": jaccard_similarity(f_or, f_max),
    }

    # Pair-level "did limit beat union-control for AND?"
    out["and_prefers_min"] = (
        1.0 if out["jaccard_and_vs_min"] > out["jaccard_and_vs_max"] else 0.0
    )
    out["or_prefers_max"] = (
        1.0 if out["jaccard_or_vs_max"] > out["jaccard_or_vs_min"] else 0.0
    )
    return out
