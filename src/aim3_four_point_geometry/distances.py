"""Four-point slack, triangle slack, and paired inference statistics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .config import TROPICAL_EPS
from .model_io import sequence_target_logprob
from .prompts import (
    ClozeQuery,
    PAIR_INDICES,
    build_type_a_six_pair_queries,
    build_type_b_queries,
    build_type_c_queries,
    pair_key,
    permute_nodes,
    type_a_nodes,
)


def tropical_distance_from_sum_logprob(
    sum_logprob: float, *, eps: float = TROPICAL_EPS
) -> float:
    lp = float(sum_logprob)
    if not math.isfinite(lp):
        return float("inf")
    lp = max(lp, math.log(eps))
    return float(-lp)


def score_query(
    tokenizer: Any,
    model: Any,
    query: ClozeQuery,
    *,
    device: Any,
    max_tokens: int,
) -> dict[str, float | int | str]:
    pack = sequence_target_logprob(
        tokenizer,
        model,
        query.prefix,
        query.target,
        device=device,
        max_tokens=max_tokens,
    )
    sum_lp = float(pack["sum_logprob"])
    return {
        "name": query.name,
        "d_M": tropical_distance_from_sum_logprob(sum_lp),
        "sum_logprob": sum_lp,
        "n_tokens": int(pack["n_tokens"]),
    }


def four_point_slack(distances: dict[str, float]) -> dict[str, float]:
    """
    Four-point slack on nodes 0,1,2,3 with d_ij = distances[pair_key(i,j)].

    S1 = d01+d23, S2 = d02+d13, S3 = d03+d12
    slack = max(S) - second_max(S)  (0 iff tree metric on four points).
    """
    d = distances
    s1 = float(d[pair_key(0, 1)] + d[pair_key(2, 3)])
    s2 = float(d[pair_key(0, 2)] + d[pair_key(1, 3)])
    s3 = float(d[pair_key(0, 3)] + d[pair_key(1, 2)])
    sums = sorted([s1, s2, s3], reverse=True)
    slack = float(sums[0] - sums[1])
    return {
        "s1_d01_d23": s1,
        "s2_d02_d13": s2,
        "s3_d03_d12": s3,
        "fp_slack": slack,
    }


def type_a_four_point_pack(
    tokenizer: Any,
    model: Any,
    row: dict[str, Any],
    *,
    device: Any,
    variant: str,
    max_tokens: int,
) -> dict[str, Any]:
    nodes = type_a_nodes(row)
    chain = permute_nodes(nodes, variant)
    queries = build_type_a_six_pair_queries(nodes, variant=variant)
    scored = {
        q.name: score_query(tokenizer, model, q, device=device, max_tokens=max_tokens)
        for q in queries
    }
    distances = {k: float(v["d_M"]) for k, v in scored.items()}
    fp = four_point_slack(distances)
    return {
        "variant": variant,
        "nodes_original": list(nodes),
        "nodes_chain": list(chain),
        "distances": distances,
        "four_point": fp,
        "queries": scored,
    }


def type_b_triangle(
    tokenizer: Any,
    model: Any,
    row: dict[str, Any],
    *,
    device: Any,
    max_tokens: int,
) -> dict[str, Any]:
    scored = {
        q.name: score_query(
            tokenizer, model, q, device=device, max_tokens=max_tokens
        )
        for q in build_type_b_queries(row)
    }
    d_bs = float(scored["base_shift"]["d_M"])
    d_sc = float(scored["shift_combined"]["d_M"])
    d_bc = float(scored["base_combined"]["d_M"])
    return {
        "distances": {
            "base_shift": d_bs,
            "shift_combined": d_sc,
            "base_combined": d_bc,
            "triangle_slack": d_bc - d_bs - d_sc,
        },
        "queries": scored,
    }


def type_c_triangle(
    tokenizer: Any,
    model: Any,
    row: dict[str, Any],
    *,
    device: Any,
    max_tokens: int,
) -> dict[str, Any]:
    scored = {
        q.name: score_query(
            tokenizer, model, q, device=device, max_tokens=max_tokens
        )
        for q in build_type_c_queries(row)
    }
    d_es = float(scored["entailed_superordinate"]["d_M"])
    d_fe = float(scored["framed_entailed"]["d_M"])
    d_fs = float(scored["framed_superordinate"]["d_M"])
    return {
        "distances": {
            "entailed_superordinate": d_es,
            "framed_entailed": d_fe,
            "framed_superordinate": d_fs,
            "triangle_slack": d_es - d_fe - d_fs,
        },
        "queries": scored,
    }


def paired_fp_slack_stats(
    natural: np.ndarray,
    other: np.ndarray,
) -> dict[str, Any]:
    """
    Paired comparison on the same four concepts, two orderings.

    - diff = fp_slack_other - fp_slack_natural  (positive ⇒ natural more tree-like)
    - Wilcoxon signed-rank on diff (non-parametric; skewed/non-negative slack OK)
    - Binomial / sign test: fraction with natural < other vs p=0.5
    """
    nat = np.asarray(natural, dtype=float)
    oth = np.asarray(other, dtype=float)
    m = min(nat.size, oth.size)
    if m == 0:
        return {"n": 0}

    nat, oth = nat[:m], oth[:m]
    finite = np.isfinite(nat) & np.isfinite(oth)
    nat, oth = nat[finite], oth[finite]
    n = int(nat.size)
    if n == 0:
        return {"n": 0}

    diff = oth - nat
    natural_better = nat < oth
    n_nat_better = int(np.sum(natural_better))
    frac_nat_better = float(n_nat_better / n)

    out: dict[str, Any] = {
        "n": n,
        "natural_fp_slack_mean": float(np.mean(nat)),
        "other_fp_slack_mean": float(np.mean(oth)),
        "diff_mean_other_minus_natural": float(np.mean(diff)),
        "diff_median_other_minus_natural": float(np.median(diff)),
        "paired_frac_natural_lower": frac_nat_better,
        "paired_n_natural_lower": n_nat_better,
    }

    try:
        from scipy import stats

        # Wilcoxon on other - natural; H1: median > 0 (natural more tree-like)
        if np.any(diff != 0):
            w = stats.wilcoxon(diff, alternative="greater", zero_method="wilcox")
            out["wilcoxon"] = {
                "statistic": float(w.statistic),
                "pvalue": float(w.pvalue),
                "alternative": "greater (other - natural > 0)",
            }
        else:
            out["wilcoxon"] = {"statistic": None, "pvalue": None, "note": "all diffs zero"}

        # Binomial test: count(natural < other) vs Binomial(n, 0.5)
        binom = stats.binomtest(n_nat_better, n, p=0.5, alternative="greater")
        out["binomial_sign"] = {
            "successes": n_nat_better,
            "n": n,
            "pvalue": float(binom.pvalue),
            "alternative": "greater than 0.5 (natural wins more often)",
        }
    except ImportError:
        out["note"] = "scipy not available; Wilcoxon/binomial skipped"

    return out


def paired_natural_vs_shuffled_stats(
    natural: np.ndarray,
    shuffled: np.ndarray,
) -> dict[str, Any]:
    """Alias for natural vs shuffled_middles (primary Type A comparison)."""
    return paired_fp_slack_stats(natural, shuffled)
