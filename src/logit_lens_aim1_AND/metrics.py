"""
Composition metrics over aligned Top-K distributions.

For each vocab index v we form three ECT-style aggregations of p_A and p_B:
  min  → pointwise minimum  (categorical limit / Logical AND)
  mean → pointwise average
  max  → pointwise maximum  (categorical colimit / Logical OR)

Then we score how close each aggregation is to the composed distribution p_AB
using L1 distance and cosine similarity (easy to extend later).
"""

from __future__ import annotations

import math
from typing import Any, Mapping


def _as_map(dist: Any) -> dict[int, float]:
    """Accept TopKDistribution or a plain {token_id: prob} mapping."""
    if hasattr(dist, "as_dict"):
        return dist.as_dict()
    return {int(k): float(v) for k, v in dict(dist).items()}


def _align(
    dist_a: Mapping[int, float],
    dist_b: Mapping[int, float],
    dist_ab: Mapping[int, float],
) -> tuple[list[float], list[float], list[float]]:
    """
    Build dense vectors on the union of Top-K supports.
    Missing tokens are treated as probability 0 (truncated mass elsewhere).
    """
    vocab = sorted(set(dist_a) | set(dist_b) | set(dist_ab))
    a = [float(dist_a.get(t, 0.0)) for t in vocab]
    b = [float(dist_b.get(t, 0.0)) for t in vocab]
    ab = [float(dist_ab.get(t, 0.0)) for t in vocab]
    return a, b, ab


def _l1(x: list[float], y: list[float]) -> float:
    return float(sum(abs(xi - yi) for xi, yi in zip(x, y)))


def _dot(x: list[float], y: list[float]) -> float:
    return float(sum(xi * yi for xi, yi in zip(x, y)))


def _norm(x: list[float]) -> float:
    return math.sqrt(sum(xi * xi for xi in x))


def _cosine(x: list[float], y: list[float]) -> float:
    nx, ny = _norm(x), _norm(y)
    if nx == 0.0 or ny == 0.0:
        return 0.0
    return _dot(x, y) / (nx * ny)


def pointwise_min(a: list[float], b: list[float]) -> list[float]:
    return [min(ai, bi) for ai, bi in zip(a, b)]


def pointwise_mean(a: list[float], b: list[float]) -> list[float]:
    return [0.5 * (ai + bi) for ai, bi in zip(a, b)]


def pointwise_max(a: list[float], b: list[float]) -> list[float]:
    return [max(ai, bi) for ai, bi in zip(a, b)]


def compute_composition_metrics(
    dist_a: Any,
    dist_b: Any,
    dist_ab: Any,
) -> dict[str, float]:
    """
    Return min/mean/max comparison metrics vs p_AB.

    Keys are stable so we can append more metrics later without breaking CSV.
    """
    a_map = _as_map(dist_a)
    b_map = _as_map(dist_b)
    ab_map = _as_map(dist_ab)

    a, b, ab = _align(a_map, b_map, ab_map)
    agg = {
        "min": pointwise_min(a, b),
        "mean": pointwise_mean(a, b),
        "max": pointwise_max(a, b),
    }

    out: dict[str, float] = {}
    for name, vec in agg.items():
        out[f"{name}_l1_to_ab"] = _l1(vec, ab)
        out[f"{name}_cosine_to_ab"] = _cosine(vec, ab)
        # Mass of the aggregation (useful for AND: min mass collapses when exclusive).
        out[f"{name}_mass"] = float(sum(vec))
    return out
