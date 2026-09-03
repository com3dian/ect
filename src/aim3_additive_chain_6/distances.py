"""
Markov factorization distances: d_M = -sum_token log π(target | cloze prefix).

Tests π(n4|n1) ?≈ π(n2|n1)·π(n3|n2)·π(n4|n3).
"""

from __future__ import annotations

import math
from typing import Any

from .config import TROPICAL_EPS
from .model_io import sequence_target_logprob
from .prompts import (
    ClozeQuery,
    build_type_a_markov_queries,
    permute_nodes,
    type_a_nodes,
)


def tropical_distance_from_sum_logprob(
    sum_logprob: float, *, eps: float = TROPICAL_EPS
) -> float:
    lp = float(sum_logprob)
    if not math.isfinite(lp):
        return float("inf")
    # clamp π >= eps  ⇒  sum_logprob >= log(eps) only as a floor on d_M
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
        "cond_concept": query.cond_concept,
        "target_concept": query.target_concept,
        "prefix": query.prefix,
        "target": query.target,
        "sum_logprob": sum_lp,
        "mean_logprob": float(pack["mean_logprob"]),
        "first_logprob": float(pack["first_logprob"]),
        "n_tokens": int(pack["n_tokens"]),
        "d_M": tropical_distance_from_sum_logprob(sum_lp),
    }


def type_a_markov_pack(
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
    queries = build_type_a_markov_queries(nodes, variant=variant)
    scored = {
        q.name: score_query(
            tokenizer, model, q, device=device, max_tokens=max_tokens
        )
        for q in queries
    }

    d12 = float(scored["n1_n2"]["d_M"])
    d23 = float(scored["n2_n3"]["d_M"])
    d34 = float(scored["n3_n4"]["d_M"])
    d14 = float(scored["n1_n4"]["d_M"])
    hops = d12 + d23 + d34
    gap = d14 - hops
    rel = gap / hops if hops > TROPICAL_EPS else float("nan")

    # Probability-domain view of the same claim:
    # product_hops = exp(sum hop logprobs), direct = exp(logprob n1→n4)
    lp12 = float(scored["n1_n2"]["sum_logprob"])
    lp23 = float(scored["n2_n3"]["sum_logprob"])
    lp34 = float(scored["n3_n4"]["sum_logprob"])
    lp14 = float(scored["n1_n4"]["sum_logprob"])
    log_product = lp12 + lp23 + lp34
    # log(π_direct / product) = lp14 - log_product = -gap
    log_ratio_direct_over_product = lp14 - log_product

    return {
        "variant": variant,
        "nodes_original": list(nodes),
        "nodes_chain": list(chain),
        "distances": {
            "n1_n2": d12,
            "n2_n3": d23,
            "n3_n4": d34,
            "n1_n4": d14,
            "sum_hops": hops,
            "additive_gap": gap,
            "relative_gap": rel,
        },
        "sum_logprobs": {
            "n1_n2": lp12,
            "n2_n3": lp23,
            "n3_n4": lp34,
            "n1_n4": lp14,
            "log_product_hops": log_product,
            "log_ratio_direct_over_product": log_ratio_direct_over_product,
        },
        "queries": scored,
    }


def type_b_triangle(
    tokenizer: Any,
    model: Any,
    queries: list[ClozeQuery],
    *,
    device: Any,
    max_tokens: int,
) -> dict[str, Any]:
    scored = {
        q.name: score_query(
            tokenizer, model, q, device=device, max_tokens=max_tokens
        )
        for q in queries
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
    queries: list[ClozeQuery],
    *,
    device: Any,
    max_tokens: int,
) -> dict[str, Any]:
    scored = {
        q.name: score_query(
            tokenizer, model, q, device=device, max_tokens=max_tokens
        )
        for q in queries
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
