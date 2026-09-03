"""
OR composition metrics: empirical P_{A∨B} vs normalized max(P_A, P_B).

The categorical colimit (logical OR) is the pointwise maximum of the cached
concept distributions, re-normalized to sum to 1 before divergence scoring.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


def _as_map(dist: Any) -> dict[int, float]:
    if hasattr(dist, "as_dict"):
        return dist.as_dict()
    if isinstance(dist, Mapping) and "token_ids" in dist and "probs" in dist:
        return {
            int(t): float(p)
            for t, p in zip(dist["token_ids"], dist["probs"])
        }
    return {int(k): float(v) for k, v in dict(dist).items()}


def _align_maps(*maps: Mapping[int, float]) -> tuple[list[int], list[list[float]]]:
    vocab = sorted(set().union(*(m.keys() for m in maps)))
    aligned = [[float(m.get(t, 0.0)) for t in vocab] for m in maps]
    return vocab, aligned


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


def compute_theoretical_or(
    dist_a: Any,
    dist_b: Any,
) -> dict[int, float]:
    """Pointwise max of P_A and P_B, re-normalized to a probability mass."""
    a_map = _as_map(dist_a)
    b_map = _as_map(dist_b)
    vocab = sorted(set(a_map) | set(b_map))
    raw = [max(a_map.get(t, 0.0), b_map.get(t, 0.0)) for t in vocab]
    total = float(sum(raw))
    if total <= 0.0:
        return {t: 0.0 for t in vocab}
    return {t: v / total for t, v in zip(vocab, raw)}


def _kl_divergence(p: list[float], q: list[float], *, eps: float = 1e-12) -> float:
    out = 0.0
    for pi, qi in zip(p, q):
        if pi <= 0.0:
            continue
        q_safe = max(qi, eps)
        out += pi * math.log(pi / q_safe)
    return float(out)


def _js_divergence(p: list[float], q: list[float], *, eps: float = 1e-12) -> float:
    m = [0.5 * (pi + qi) for pi, qi in zip(p, q)]
    return 0.5 * _kl_divergence(p, m, eps=eps) + 0.5 * _kl_divergence(q, m, eps=eps)


def compute_or_metrics(
    dist_a: Any,
    dist_b: Any,
    dist_or: Any,
) -> dict[str, float]:
    """
    Compare empirical P_{A∨B} to the normalized theoretical max(P_A, P_B).

    Returns stable metric keys for CSV / JSONL export.
    """
    theor_map = compute_theoretical_or(dist_a, dist_b)
    or_map = _as_map(dist_or)

    _, (theor_vec, or_vec) = _align_maps(theor_map, or_map)
    raw_max = [max(_as_map(dist_a).get(t, 0.0), _as_map(dist_b).get(t, 0.0))
               for t in sorted(set(_as_map(dist_a)) | set(_as_map(dist_b)))]

    return {
        "max_raw_mass": float(sum(raw_max)),
        "max_norm_l1_to_or": _l1(theor_vec, or_vec),
        "max_norm_cosine_to_or": _cosine(theor_vec, or_vec),
        "max_norm_kl_or_to_theoretical": _kl_divergence(or_vec, theor_vec),
        "max_norm_kl_theoretical_to_or": _kl_divergence(theor_vec, or_vec),
        "max_norm_js_to_or": _js_divergence(theor_vec, or_vec),
    }
