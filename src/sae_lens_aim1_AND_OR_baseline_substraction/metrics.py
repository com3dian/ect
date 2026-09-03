"""
Differential SAE metrics aligned to the protocol:

  tilde = ReLU(V - V_base)
  hat   = tilde / (||tilde||_1 + eps)
  theo_* built from tilde_A / tilde_B (then L1-normalized)
  TVD(hat_actual, theo)
"""

from __future__ import annotations

from typing import Any

import torch

from .config import JACCARD_THRESHOLD

THEORY_NAMES = ("min", "mean", "max", "prod")
EPS = 1e-12


def l1_normalize(v: torch.Tensor, *, eps: float = EPS) -> torch.Tensor:
    """hat = tilde / (||tilde||_1 + eps), matching the protocol."""
    v = torch.clamp(v.float(), min=0.0)
    total = float(v.sum().item())
    return v / (total + eps)


def baseline_subtract(v: torch.Tensor, v_base: torch.Tensor) -> torch.Tensor:
    """ReLU(V − V_base)."""
    return torch.relu(v.float() - v_base.float())


def apply_differential(
    v_a: torch.Tensor,
    v_b: torch.Tensor,
    v_and: torch.Tensor,
    v_or: torch.Tensor,
    v_base_single: torch.Tensor,
    v_base_composed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        baseline_subtract(v_a, v_base_single),
        baseline_subtract(v_b, v_base_single),
        baseline_subtract(v_and, v_base_composed),
        baseline_subtract(v_or, v_base_composed),
    )


def sparse_topk(v: torch.Tensor, top_k: int) -> dict[str, list]:
    """L1-normalize, then keep Top-K features by mass (zeros dropped)."""
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


def pointwise_prod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a * b


def total_variation_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    return float(0.5 * torch.sum(torch.abs(p - q)).item())


def l0_count(v: torch.Tensor, *, threshold: float = JACCARD_THRESHOLD) -> int:
    return int((v.float() > threshold).sum().item())


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


def theoretical_baselines(
    tilde_a: torch.Tensor,
    tilde_b: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Protocol §5A — theories from denoised (unnormalized) tilde vectors:

      min  = Norm(min(tilde_A, tilde_B))
      max  = Norm(max(tilde_A, tilde_B))
      mean = Norm(tilde_A + tilde_B)
      prod = Norm(tilde_A ⊙ tilde_B)
    """
    return {
        "min": l1_normalize(pointwise_min(tilde_a, tilde_b)),
        "max": l1_normalize(pointwise_max(tilde_a, tilde_b)),
        "mean": l1_normalize(tilde_a.float() + tilde_b.float()),
        "prod": l1_normalize(pointwise_prod(tilde_a, tilde_b)),
    }


def compute_composition_metrics(
    tilde_a: torch.Tensor,
    tilde_b: torch.Tensor,
    tilde_and: torch.Tensor,
    tilde_or: torch.Tensor,
    *,
    raw_a: torch.Tensor | None = None,
    raw_b: torch.Tensor | None = None,
    raw_and: torch.Tensor | None = None,
    raw_or: torch.Tensor | None = None,
    jaccard_threshold: float = JACCARD_THRESHOLD,
) -> dict[str, Any]:
    """
    Score TVD(hat_AND/OR, theo_*) plus L0 before/after and operator win flags.
    """
    hat_a = l1_normalize(tilde_a)
    hat_b = l1_normalize(tilde_b)
    hat_and = l1_normalize(tilde_and)
    hat_or = l1_normalize(tilde_or)

    baselines = theoretical_baselines(tilde_a, tilde_b)

    tvd_and = {
        name: total_variation_distance(hat_and, theory) for name, theory in baselines.items()
    }
    tvd_or = {
        name: total_variation_distance(hat_or, theory) for name, theory in baselines.items()
    }
    winner_and = min(THEORY_NAMES, key=lambda n: tvd_and[n])
    winner_or = min(THEORY_NAMES, key=lambda n: tvd_or[n])

    out: dict[str, Any] = {
        "mass_a": float(tilde_a.float().sum().item()),
        "mass_b": float(tilde_b.float().sum().item()),
        "mass_and": float(tilde_and.float().sum().item()),
        "mass_or": float(tilde_or.float().sum().item()),
        # L0 after subtraction (on tilde)
        "l0_after_a": float(l0_count(tilde_a, threshold=jaccard_threshold)),
        "l0_after_b": float(l0_count(tilde_b, threshold=jaccard_threshold)),
        "l0_after_and": float(l0_count(tilde_and, threshold=jaccard_threshold)),
        "l0_after_or": float(l0_count(tilde_or, threshold=jaccard_threshold)),
        # aliases used by older plot code / summaries
        "n_active_a": float(l0_count(tilde_a, threshold=jaccard_threshold)),
        "n_active_b": float(l0_count(tilde_b, threshold=jaccard_threshold)),
        "n_active_and": float(l0_count(tilde_and, threshold=jaccard_threshold)),
        "n_active_or": float(l0_count(tilde_or, threshold=jaccard_threshold)),
        "jaccard_a_b": jaccard_similarity(
            active_feature_set(tilde_a, threshold=jaccard_threshold),
            active_feature_set(tilde_b, threshold=jaccard_threshold),
        ),
        "min_raw_mass": float(pointwise_min(tilde_a, tilde_b).sum().item()),
        "max_raw_mass": float(pointwise_max(tilde_a, tilde_b).sum().item()),
        "prod_raw_mass": float(pointwise_prod(tilde_a, tilde_b).sum().item()),
        "tvd_theory_min_vs_mean": total_variation_distance(
            baselines["min"], baselines["mean"]
        ),
        "tvd_theory_max_vs_mean": total_variation_distance(
            baselines["max"], baselines["mean"]
        ),
        "tvd_theory_min_vs_max": total_variation_distance(
            baselines["min"], baselines["max"]
        ),
        "winner_and": winner_and,
        "winner_or": winner_or,
    }

    if raw_a is not None:
        out["l0_before_a"] = float(l0_count(raw_a, threshold=jaccard_threshold))
    if raw_b is not None:
        out["l0_before_b"] = float(l0_count(raw_b, threshold=jaccard_threshold))
    if raw_and is not None:
        out["l0_before_and"] = float(l0_count(raw_and, threshold=jaccard_threshold))
    if raw_or is not None:
        out["l0_before_or"] = float(l0_count(raw_or, threshold=jaccard_threshold))

    for name in THEORY_NAMES:
        out[f"tvd_and_vs_{name}"] = tvd_and[name]
        out[f"tvd_or_vs_{name}"] = tvd_or[name]
        out[f"win_and_{name}"] = 1.0 if winner_and == name else 0.0
        out[f"win_or_{name}"] = 1.0 if winner_or == name else 0.0

    # unused but kept for type checkers / callers that expected hats
    _ = (hat_a, hat_b)
    return out
