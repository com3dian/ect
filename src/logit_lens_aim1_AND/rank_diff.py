"""
Rank-aligned probability differences: min / mean / max vs ground-truth p_AB.

For each pair:
  1. Sort p_AB Top-K tokens by probability (high → low).
  2. At those tokens, evaluate pointwise min / mean / max of (p_A, p_B)
     (missing Top-K mass treated as 0).
  3. Record absolute difference |agg(token) − p_AB(token)| at each rank.

Across pairs in a category, aggregate at each rank:
  mean, 5th percentile, 95th percentile.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .metrics import pointwise_max, pointwise_mean, pointwise_min


AGG_NAMES = ("min", "mean", "max")


def _topk_maps(row: Mapping[str, Any]) -> tuple[dict[int, float], dict[int, float], list[int], np.ndarray]:
    """Return p_A map, p_B map, p_AB token ids (unsorted), p_AB probs (unsorted)."""
    a = {int(t): float(p) for t, p in zip(row["P_A"]["token_ids"], row["P_A"]["probs"])}
    b = {int(t): float(p) for t, p in zip(row["P_B"]["token_ids"], row["P_B"]["probs"])}
    ab_ids = [int(t) for t in row["P_AB"]["token_ids"]]
    ab_probs = np.asarray(row["P_AB"]["probs"], dtype=np.float64)
    return a, b, ab_ids, ab_probs


def pair_rank_differences(row: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """
    Sort ground-truth p_AB high→low, then return absolute diffs for min/mean/max.

    Each array has shape (K,) with K = len(p_AB Top-K).
    """
    a_map, b_map, ab_ids, ab_probs = _topk_maps(row)
    order = np.argsort(-ab_probs, kind="mergesort")
    tids = [ab_ids[i] for i in order]
    p_ab = ab_probs[order]

    a = [float(a_map.get(t, 0.0)) for t in tids]
    b = [float(b_map.get(t, 0.0)) for t in tids]
    aggs = {
        "min": np.asarray(pointwise_min(a, b), dtype=np.float64),
        "mean": np.asarray(pointwise_mean(a, b), dtype=np.float64),
        "max": np.asarray(pointwise_max(a, b), dtype=np.float64),
    }
    return {name: np.abs(vec - p_ab) for name, vec in aggs.items()}


def accumulate_category_diffs(
    distributions_jsonl: Path | str,
    *,
    categories: Iterable[int] | None = None,
    max_pairs: int | None = None,
) -> dict[int, dict[str, Any]]:
    """
    Stream distributions JSONL and collect per-category difference matrices.

    Returns:
      {
        category_id: {
          "category_name": str,
          "n_pairs": int,
          "diffs": {"min"|"mean"|"max": ndarray (n_pairs, K)},
        },
        ...
      }
    """
    path = Path(distributions_jsonl)
    wanted = set(categories) if categories is not None else None
    buckets: dict[int, dict[str, Any]] = {}
    n_seen = 0

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cid = int(row["category_id"])
            if wanted is not None and cid not in wanted:
                continue

            diffs = pair_rank_differences(row)
            if cid not in buckets:
                buckets[cid] = {
                    "category_name": str(row.get("category_name", cid)),
                    "lists": {name: [] for name in AGG_NAMES},
                }
            for name in AGG_NAMES:
                buckets[cid]["lists"][name].append(diffs[name])

            n_seen += 1
            if max_pairs is not None and n_seen >= max_pairs:
                break

    out: dict[int, dict[str, Any]] = {}
    for cid, blob in buckets.items():
        stacked = {name: np.stack(blob["lists"][name], axis=0) for name in AGG_NAMES}
        out[cid] = {
            "category_name": blob["category_name"],
            "n_pairs": int(stacked["min"].shape[0]),
            "top_k": int(stacked["min"].shape[1]),
            "diffs": stacked,
        }
    return out


def summarize_diffs(
    diffs: Mapping[str, np.ndarray],
    *,
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Per aggregation: mean / lower / upper quantile across pairs at each rank.

    diffs[name] has shape (n_pairs, K).
    """
    summary: dict[str, dict[str, np.ndarray]] = {}
    for name, mat in diffs.items():
        summary[name] = {
            "mean": np.mean(mat, axis=0),
            "q_low": np.quantile(mat, q_low, axis=0),
            "q_high": np.quantile(mat, q_high, axis=0),
        }
    return summary


def compute_rank_diff_summary(
    distributions_jsonl: Path | str,
    *,
    categories: Iterable[int] | None = None,
    max_pairs: int | None = None,
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> dict[int, dict[str, Any]]:
    """Accumulate diffs and return ready-to-plot summaries per category."""
    raw = accumulate_category_diffs(
        distributions_jsonl,
        categories=categories,
        max_pairs=max_pairs,
    )
    out: dict[int, dict[str, Any]] = {}
    for cid, blob in raw.items():
        out[cid] = {
            "category_name": blob["category_name"],
            "n_pairs": blob["n_pairs"],
            "top_k": blob["top_k"],
            "summary": summarize_diffs(blob["diffs"], q_low=q_low, q_high=q_high),
        }
    return out


def save_rank_diff_summary(summary: Mapping[int, Mapping[str, Any]], path: Path | str) -> Path:
    """Save compact NPZ with mean / q05 / q95 curves per category × aggregation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    meta_rows: list[dict[str, Any]] = []
    for cid, blob in sorted(summary.items()):
        meta_rows.append(
            {
                "category_id": int(cid),
                "category_name": blob["category_name"],
                "n_pairs": int(blob["n_pairs"]),
                "top_k": int(blob["top_k"]),
            }
        )
        for agg, curves in blob["summary"].items():
            prefix = f"c{cid}_{agg}_"
            payload[prefix + "mean"] = np.asarray(curves["mean"], dtype=np.float64)
            payload[prefix + "q_low"] = np.asarray(curves["q_low"], dtype=np.float64)
            payload[prefix + "q_high"] = np.asarray(curves["q_high"], dtype=np.float64)
    payload["meta_json"] = np.asarray(json.dumps(meta_rows))
    np.savez_compressed(path, **payload)
    return path


def load_rank_diff_summary(path: Path | str) -> dict[int, dict[str, Any]]:
    """Inverse of save_rank_diff_summary."""
    data = np.load(path, allow_pickle=False)
    meta_rows = json.loads(str(data["meta_json"]))
    out: dict[int, dict[str, Any]] = {}
    for row in meta_rows:
        cid = int(row["category_id"])
        summary: dict[str, dict[str, np.ndarray]] = {}
        for agg in AGG_NAMES:
            prefix = f"c{cid}_{agg}_"
            summary[agg] = {
                "mean": np.asarray(data[prefix + "mean"]),
                "q_low": np.asarray(data[prefix + "q_low"]),
                "q_high": np.asarray(data[prefix + "q_high"]),
            }
        out[cid] = {
            "category_name": row["category_name"],
            "n_pairs": int(row["n_pairs"]),
            "top_k": int(row["top_k"]),
            "summary": summary,
        }
    return out


def plot_category_rank_diffs(
    summary: Mapping[int, Mapping[str, Any]],
    *,
    out_dir: Path | str | None = None,
    show: bool = True,
    figsize: tuple[float, float] = (11, 3.6),
    log_x: bool = True,
) -> list[Path]:
    """
    One figure per category: three panels (min / mean / max).

    X = token rank by p_AB (0 = highest prob).
    Y = absolute difference |agg − p_AB|.
    Ribbon = 5th–95th percentile across pairs; line = mean.
    """
    import matplotlib.pyplot as plt

    out_paths: list[Path] = []
    out_dir_path = Path(out_dir) if out_dir is not None else None
    if out_dir_path is not None:
        out_dir_path.mkdir(parents=True, exist_ok=True)

    colors = {"min": "#1b9e77", "mean": "#d95f02", "max": "#7570b3"}

    for cid in sorted(summary):
        blob = summary[cid]
        k = int(blob["top_k"])
        # rank 0..K-1; use 1..K on a log axis so the probability head is visible
        x = np.arange(1, k + 1) if log_x else np.arange(k)
        fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
        fig.suptitle(
            f"Cat {cid}: {blob['category_name']}  (n={blob['n_pairs']} pairs)\n"
            "x = p_AB rank (high→low)   y = |agg − p_AB|   ribbon = 5%–95% quantile",
            fontsize=11,
        )
        for ax, agg in zip(axes, AGG_NAMES):
            curves = blob["summary"][agg]
            c = colors[agg]
            ax.fill_between(x, curves["q_low"], curves["q_high"], color=c, alpha=0.25, linewidth=0)
            ax.plot(x, curves["mean"], color=c, lw=1.5, label="mean")
            ax.set_title(agg)
            ax.set_xlabel("token rank (p_AB high → low)")
            if log_x:
                ax.set_xscale("log")
                ax.set_xlim(1, k)
            ax.set_ylim(bottom=0.0)
            ax.grid(True, alpha=0.25, which="both")
        axes[0].set_ylabel("|probability difference|")
        fig.tight_layout()

        if out_dir_path is not None:
            dest = out_dir_path / f"rank_diff_cat{cid}.png"
            fig.savefig(dest, dpi=150, bbox_inches="tight")
            out_paths.append(dest)
        if show:
            plt.show()
        else:
            plt.close(fig)

    return out_paths
