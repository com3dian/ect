"""True tropical distances d_M = -ln π(y|x) on shared-trajectory prefixes."""

from __future__ import annotations

import math
from typing import Any

import torch

from .config import TROPICAL_EPS
from .model_io import mean_target_logprob
from .prompts import TrajectoryPrompt, type_a_pairs


def tropical_distance_from_logprob(logprob: float, *, eps: float = TROPICAL_EPS) -> float:
    lp = float(logprob)
    if not math.isfinite(lp):
        return float("inf")
    lp = max(lp, math.log(eps))
    return float(-lp)


def score_pair(
    tokenizer: Any,
    model: Any,
    traj: TrajectoryPrompt,
    *,
    prefix_idx: int,
    target_idx: int,
    device: torch.device,
) -> tuple[float, float]:
    """Return (logprob, d_M) for target label given prefix through waypoint."""
    prefix = traj.prefix_through(prefix_idx)
    target = traj.labels[target_idx]
    logp = mean_target_logprob(
        tokenizer, model, prefix, target, device=device
    )
    return logp, tropical_distance_from_logprob(logp)


def type_a_ln_distances(
    tokenizer: Any,
    model: Any,
    traj: TrajectoryPrompt,
    *,
    device: torch.device,
) -> dict[str, float]:
    """
    Path hops + shortcut under true d_M=-ln π(label|prefix).

    Shortcut scores n4 given prefix through n1 (label jump).
    """
    out: dict[str, float] = {}
    logps: dict[str, float] = {}
    for pair in type_a_pairs(traj):
        name = pair["pair_name"]
        logp, dist = score_pair(
            tokenizer,
            model,
            traj,
            prefix_idx=int(pair["prefix_idx"]),
            target_idx=int(pair["target_idx"]),
            device=device,
        )
        out[name] = dist
        logps[name] = logp

    hops = out["n1_n2"] + out["n2_n3"] + out["n3_n4"]
    out["sum_hops"] = hops
    out["additive_gap"] = out["n1_n4"] - hops
    out_log = {f"logprob_{k}": v for k, v in logps.items()}
    out.update(out_log)
    return out


def type_b_ln_distances(
    tokenizer: Any,
    model: Any,
    traj: TrajectoryPrompt,
    *,
    device: torch.device,
) -> dict[str, float]:
    """
    base→shift, shift→combined label/text jumps, base→combined shortcut.

    For B, labels are the block texts themselves (base/shift/combined strings
    stored in traj.labels).
    """
    # prefix base → score shift text
    _, d_bs = score_pair(tokenizer, model, traj, prefix_idx=0, target_idx=1, device=device)
    _, d_sc = score_pair(tokenizer, model, traj, prefix_idx=1, target_idx=2, device=device)
    _, d_bc = score_pair(tokenizer, model, traj, prefix_idx=0, target_idx=2, device=device)
    return {
        "base_shift": d_bs,
        "shift_combined": d_sc,
        "base_combined": d_bc,
        "triangle_slack": d_bc - d_bs - d_sc,
    }


def type_c_ln_distances(
    tokenizer: Any,
    model: Any,
    traj: TrajectoryPrompt,
    *,
    device: torch.device,
) -> dict[str, float]:
    # entailed → superordinate; framed → entailed; framed → superordinate
    # waypoints: 0=entailed, 1=superordinate, 2=framed
    _, d_es = score_pair(tokenizer, model, traj, prefix_idx=0, target_idx=1, device=device)
    _, d_fe = score_pair(tokenizer, model, traj, prefix_idx=2, target_idx=0, device=device)
    _, d_fs = score_pair(tokenizer, model, traj, prefix_idx=2, target_idx=1, device=device)
    return {
        "entailed_superordinate": d_es,
        "framed_entailed": d_fe,
        "framed_superordinate": d_fs,
        "triangle_slack": d_es - d_fe - d_fs,
    }
