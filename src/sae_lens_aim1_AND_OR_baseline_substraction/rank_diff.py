"""
Rank-aligned differential SAE feature diffs: min / mean / max vs V'_AND / V'_OR.

Operates on baseline-subtracted Top-K features (ReLU(V − V_base)).

For each pair and operator op ∈ {and, or}:
  1. Sort empirical V'_op Top-K features by mass (high → low).
  2. At those feature ids, evaluate pointwise min / mean / max of (V'_A, V'_B)
     (missing Top-K mass treated as 0).
  3. Record |agg(feature) − V'_op(feature)| at each rank.

Across pairs in a category, aggregate at each rank:
  mean, 5th percentile, 95th percentile.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


AGG_NAMES = ("min", "mean", "max")
OP_KEYS = {"and": "V_AND", "or": "V_OR"}


def _pointwise_min(a: list[float], b: list[float]) -> np.ndarray:
    return np.minimum(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))


def _pointwise_max(a: list[float], b: list[float]) -> np.ndarray:
    return np.maximum(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))


def _pointwise_mean(a: list[float], b: list[float]) -> np.ndarray:
    return 0.5 * (np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64))


def _feature_map(blob: Mapping[str, Any]) -> dict[int, float]:
    return {
        int(fid): float(val)
        for fid, val in zip(blob["feature_ids"], blob["values"])
    }


def _pad_to_k(ids: list[int], vals: np.ndarray, k: int) -> tuple[list[int], np.ndarray]:
    """Truncate or zero-pad so length == k."""
    if len(ids) >= k:
        return ids[:k], vals[:k]
    pad_n = k - len(ids)
    ids = ids + ([-1] * pad_n)
    vals = np.concatenate([vals, np.zeros(pad_n, dtype=np.float64)])
    return ids, vals


def pair_rank_differences(
    row: Mapping[str, Any],
    *,
    op: str,
    top_k: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Sort empirical V_op high→low, then return absolute diffs for min/mean/max.

    Each array has shape (K,) with K = top_k or len(empirical features).
    """
    if op not in OP_KEYS:
        raise ValueError(f"op must be one of {tuple(OP_KEYS)}, got {op!r}")

    a_map = _feature_map(row["V_A"])
    b_map = _feature_map(row["V_B"])
    emp = row[OP_KEYS[op]]
    emp_ids = [int(t) for t in emp["feature_ids"]]
    emp_vals = np.asarray(emp["values"], dtype=np.float64)

    order = np.argsort(-emp_vals, kind="mergesort")
    emp_ids = [emp_ids[i] for i in order]
    emp_vals = emp_vals[order]

    k = int(top_k) if top_k is not None else int(row.get("top_k", len(emp_ids)))
    if k <= 0:
        k = max(len(emp_ids), 1)
    emp_ids, emp_vals = _pad_to_k(emp_ids, emp_vals, k)

    a = [float(a_map.get(t, 0.0)) if t >= 0 else 0.0 for t in emp_ids]
    b = [float(b_map.get(t, 0.0)) if t >= 0 else 0.0 for t in emp_ids]
    aggs = {
        "min": _pointwise_min(a, b),
        "mean": _pointwise_mean(a, b),
        "max": _pointwise_max(a, b),
    }
    return {name: np.abs(vec - emp_vals) for name, vec in aggs.items()}


def accumulate_category_diffs(
    distributions_jsonl: Path | str,
    *,
    op: str,
    categories: Iterable[int] | None = None,
    max_pairs: int | None = None,
    top_k: int | None = None,
) -> dict[int, dict[str, Any]]:
    path = Path(distributions_jsonl)
    wanted = set(categories) if categories is not None else None
    buckets: dict[int, dict[str, Any]] = {}
    n_seen = 0
    resolved_k: int | None = top_k

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cid = int(row["category_id"])
            if wanted is not None and cid not in wanted:
                continue

            if resolved_k is None:
                resolved_k = int(row.get("top_k") or len(row[OP_KEYS[op]]["feature_ids"]))

            diffs = pair_rank_differences(row, op=op, top_k=resolved_k)
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
            "op": op,
            "diffs": stacked,
        }
    return out


def summarize_diffs(
    diffs: Mapping[str, np.ndarray],
    *,
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> dict[str, dict[str, np.ndarray]]:
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
    op: str,
    categories: Iterable[int] | None = None,
    max_pairs: int | None = None,
    top_k: int | None = None,
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> dict[int, dict[str, Any]]:
    raw = accumulate_category_diffs(
        distributions_jsonl,
        op=op,
        categories=categories,
        max_pairs=max_pairs,
        top_k=top_k,
    )
    out: dict[int, dict[str, Any]] = {}
    for cid, blob in raw.items():
        out[cid] = {
            "category_name": blob["category_name"],
            "n_pairs": blob["n_pairs"],
            "top_k": blob["top_k"],
            "op": op,
            "summary": summarize_diffs(blob["diffs"], q_low=q_low, q_high=q_high),
        }
    return out


def save_rank_diff_summary(summary: Mapping[int, Mapping[str, Any]], path: Path | str) -> Path:
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
                "op": blob.get("op"),
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
            "op": row.get("op"),
            "summary": summary,
        }
    return out


def plot_category_rank_diffs(
    summary: Mapping[int, Mapping[str, Any]],
    *,
    op: str,
    out_dir: Path | str | None = None,
    show: bool = False,
    figsize: tuple[float, float] = (11, 3.6),
    log_x: bool = True,
    layer_index: int | None = None,
) -> list[Path]:
    """
    One figure per category: three panels (min / mean / max).

    X = SAE feature rank by V_op (0 = highest mass).
    Y = absolute difference |agg − V_op|.
    Ribbon = 5th–95th percentile across pairs; line = mean.
    """
    import matplotlib.pyplot as plt

    if op not in OP_KEYS:
        raise ValueError(f"op must be one of {tuple(OP_KEYS)}, got {op!r}")

    out_paths: list[Path] = []
    out_dir_path = Path(out_dir) if out_dir is not None else None
    if out_dir_path is not None:
        out_dir_path.mkdir(parents=True, exist_ok=True)

    colors = {"min": "#1b9e77", "mean": "#d95f02", "max": "#7570b3"}
    emp_label = "V'_AND" if op == "and" else "V'_OR"
    layer_tag = f"L={layer_index}  " if layer_index is not None else ""

    for cid in sorted(summary):
        blob = summary[cid]
        k = int(blob["top_k"])
        x = np.arange(1, k + 1) if log_x else np.arange(k)
        fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
        fig.suptitle(
            f"{layer_tag}[diff] Cat {cid}: {blob['category_name']}  "
            f"(n={blob['n_pairs']} pairs)\n"
            f"x = {emp_label} feature rank (high→low)   "
            f"y = |agg − {emp_label}|   ribbon = 5%–95% quantile   "
            f"(ReLU(V−V_base))",
            fontsize=11,
        )
        for ax, agg in zip(axes, AGG_NAMES):
            curves = blob["summary"][agg]
            c = colors[agg]
            ax.fill_between(x, curves["q_low"], curves["q_high"], color=c, alpha=0.25, linewidth=0)
            ax.plot(x, curves["mean"], color=c, lw=1.5, label="mean")
            ax.set_title(agg)
            ax.set_xlabel(f"feature rank ({emp_label} high → low)")
            if log_x:
                ax.set_xscale("log")
                ax.set_xlim(1, k)
            ax.set_ylim(bottom=0.0)
            ax.grid(True, alpha=0.25, which="both")
        axes[0].set_ylabel("|feature-mass difference|")
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
