"""Chain-rule tautology + short-token Markov distances (v7)."""

from __future__ import annotations

import math
from typing import Any

from .config import MAX_TARGET_TOKENS, TROPICAL_EPS
from .model_io import sequence_target_logprob
from .prompts import (
    ClozeQuery,
    build_type_a_markov_queries,
    chain_rule_line,
    chain_rule_stem,
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
        "cond_concept": query.cond_concept,
        "target_concept": query.target_concept,
        "sum_logprob": sum_lp,
        "n_tokens": int(pack["n_tokens"]),
        "d_M": tropical_distance_from_sum_logprob(sum_lp),
    }


def chain_rule_sanity(
    tokenizer: Any,
    model: Any,
    row: dict[str, Any],
    *,
    device: Any,
    max_tokens: int = MAX_TARGET_TOKENS,
) -> dict[str, Any]:
    """
    Tautology check on the SAME continuation string:

      log π(n2 n3 n4 | stem(n1))
        ?≈  log π(n2|stem) + log π(n3|stem+n2) + log π(n4|stem+n2+n3)

    Gap ≈ 0 ⇒ scoring/tokenization composition is consistent.
    This is NOT the Markov geodesic claim from v6.
    """
    n1, n2, n3, n4 = type_a_nodes(row)
    stem = chain_rule_stem(n1)
    line2 = chain_rule_line(n2)
    line3 = chain_rule_line(n3)
    line4 = chain_rule_line(n4)
    # Join with newlines so the joint string matches stepwise prefix growth.
    joint_text = f"{line2}\n{line3}\n{line4}"

    # Allow enough tokens for three short lines.
    joint_budget = max(int(max_tokens) * 3, 24)

    joint = sequence_target_logprob(
        tokenizer,
        model,
        stem,
        joint_text,
        device=device,
        max_tokens=joint_budget,
        target_as_raw=True,
    )

    lp2 = sequence_target_logprob(
        tokenizer,
        model,
        stem,
        line2,
        device=device,
        max_tokens=max_tokens,
        target_as_raw=True,
    )
    pref2 = stem + line2 + "\n"
    lp3 = sequence_target_logprob(
        tokenizer,
        model,
        pref2,
        line3,
        device=device,
        max_tokens=max_tokens,
        target_as_raw=True,
    )
    pref3 = pref2 + line3 + "\n"
    lp4 = sequence_target_logprob(
        tokenizer,
        model,
        pref3,
        line4,
        device=device,
        max_tokens=max_tokens,
        target_as_raw=True,
    )

    j = float(joint["sum_logprob"])
    f2 = float(lp2["sum_logprob"])
    f3 = float(lp3["sum_logprob"])
    f4 = float(lp4["sum_logprob"])
    fact = f2 + f3 + f4
    gap = j - fact  # ≈ 0 if consistent

    return {
        "nodes": [n1, n2, n3, n4],
        "stem": stem,
        "joint_text": joint_text,
        "joint_sum_logprob": j,
        "factor_sum_logprob": fact,
        "factor_parts": {"n2": f2, "n3": f3, "n4": f4},
        "chain_rule_gap": gap,
        "abs_gap": abs(gap) if math.isfinite(gap) else float("inf"),
        "n_tokens_joint": int(joint["n_tokens"]),
        "n_tokens_factors": [
            int(lp2["n_tokens"]),
            int(lp3["n_tokens"]),
            int(lp4["n_tokens"]),
        ],
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
        "queries": scored,
    }
