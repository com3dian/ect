"""Type A plot: natural vs control Markov-cloze additive gaps (v6)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    DEFAULT_OUTPUT_DIR,
    LAYER_INDEX,
    distances_jsonl,
    figures_dir_for_layer,
)

COLORS = {
    "n1_n2": "#2a6f6f",
    "n2_n3": "#3d7c47",
    "n3_n4": "#8a6d3b",
    "n1_n4": "#a33b2b",
    "sum_hops": "#4a5568",
    "natural": "#1f4e5f",
    "shuffled_middles": "#8a4b2b",
    "reversed": "#5c4d7a",
}


def load_type_a(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type_key") == "A":
                rows.append(row)
    return rows


def _variant_distances(
    rows: list[dict[str, Any]], variant: str
) -> dict[str, np.ndarray]:
    keys = ("n1_n2", "n2_n3", "n3_n4", "n1_n4", "sum_hops", "additive_gap", "relative_gap")
    buckets: dict[str, list[float]] = {k: [] for k in keys}
    for row in rows:
        pack = (row.get("variants") or {}).get(variant)
        if not pack:
            continue
        d = pack.get("distances") or {}
        if not all(k in d for k in ("n1_n2", "n2_n3", "n3_n4", "n1_n4")):
            continue
        for k in ("n1_n2", "n2_n3", "n3_n4", "n1_n4"):
            buckets[k].append(float(d[k]))
        hops = float(d.get("sum_hops", d["n1_n2"] + d["n2_n3"] + d["n3_n4"]))
        gap = float(d.get("additive_gap", d["n1_n4"] - hops))
        rel = float(d.get("relative_gap", gap / hops if hops else float("nan")))
        buckets["sum_hops"].append(hops)
        buckets["additive_gap"].append(gap)
        buckets["relative_gap"].append(rel)
    return {k: np.asarray(v, dtype=np.float64) for k, v in buckets.items()}


def _padded_ylim(arrays: list[np.ndarray], *, floor: float | None = 0.0) -> tuple[float, float]:
    lo = float(min(a.min() for a in arrays if a.size))
    hi = float(max(a.max() for a in arrays if a.size))
    span = max(hi - lo, 1e-3)
    pad = 0.08 * span
    if floor is not None and lo >= 0:
        return floor, hi + pad
    return lo - pad, hi + pad


def plot_type_a(
    rows: list[dict[str, Any]],
    *,
    out_path: Path,
    layer_index: int,
    primary_variant: str = "natural",
    model_id: str = "google/gemma-2-9b",
) -> Path:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    variants = []
    for r in rows:
        variants.extend((r.get("variants") or {}).keys())
    variants = list(dict.fromkeys(variants))
    if primary_variant not in variants and variants:
        primary_variant = variants[0]

    nat = _variant_distances(rows, primary_variant)
    n = int(nat["additive_gap"].size)
    if n == 0:
        raise SystemExit(f"No distances for variant={primary_variant}")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.8, 5.4),
        gridspec_kw={"width_ratios": [1.25, 1.1], "wspace": 0.32},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("#f7f4ef")
    for ax in axes:
        ax.set_facecolor("#fbfaf7")

    # Left: natural path costs
    ax = axes[0]
    order = ["n1_n2", "n2_n3", "n3_n4", "sum_hops", "n1_n4"]
    labels = [
        r"$d(n_1\!\to\!n_2)$",
        r"$d(n_2\!\to\!n_3)$",
        r"$d(n_3\!\to\!n_4)$",
        r"$\sum$ hops",
        r"$d(n_1\!\to\!n_4)$",
    ]
    data = [nat[k] for k in order]
    pos = np.arange(len(order))
    bp = ax.boxplot(
        data,
        positions=pos,
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        flierprops={
            "marker": "o",
            "markersize": 2.5,
            "markerfacecolor": "#666",
            "markeredgecolor": "none",
            "alpha": 0.35,
        },
        medianprops={"color": "#1a1a1a", "linewidth": 1.7},
    )
    for box, key in zip(bp["boxes"], order):
        box.set_facecolor(COLORS.get(key, "#666"))
        box.set_alpha(0.78)
        box.set_edgecolor("#222")
    means = [float(np.mean(nat[k])) for k in order]
    y0, y1 = _padded_ylim(data, floor=0.0)
    ax.scatter(pos, means, color="#111", s=32, zorder=5)
    for x, m in zip(pos, means):
        ax.text(x, m + 0.025 * (y1 - y0), f"{m:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(r"$d_M=-\sum\log\pi($concept$\mid$cloze$)$")
    ax.set_title(f"Markov cloze path costs ({primary_variant})", fontsize=11)
    ax.set_ylim(y0, y1)
    ax.grid(True, axis="y", alpha=0.28)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Right: gap histograms — natural vs controls
    ax = axes[1]
    series = []
    for vname in variants:
        arr = _variant_distances(rows, vname)["additive_gap"]
        if arr.size:
            series.append((vname, arr))
    all_g = np.concatenate([a for _, a in series])
    lo, hi = float(all_g.min()), float(all_g.max())
    span = max(hi - lo, 1.0)
    bins = np.linspace(lo - 0.05 * span, hi + 0.05 * span, 36)
    for vname, arr in series:
        ax.hist(
            arr,
            bins=bins,
            color=COLORS.get(vname, "#555"),
            alpha=0.55,
            edgecolor="#333",
            linewidth=0.4,
            label=f"{vname} (mean {np.mean(arr):.2f})",
        )
    ax.axvline(0.0, color="#a33b2b", lw=1.8, ls="--", label="equality (gap=0)")

    g_nat = nat["additive_gap"]
    r_nat = nat["relative_gap"]
    r_nat = r_nat[np.isfinite(r_nat)]
    note = (
        f"n = {n}  ({primary_variant})\n"
        f"|gap|<1: {100 * np.mean(np.abs(g_nat) < 1):.0f}%\n"
        f"|gap|<2: {100 * np.mean(np.abs(g_nat) < 2):.0f}%\n"
        f"gap≤0: {100 * np.mean(g_nat <= 0):.0f}%"
    )
    if r_nat.size:
        note += f"\nrel gap mean: {np.mean(r_nat):.2f}"
    ax.text(
        0.98,
        0.97,
        note,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff", edgecolor="#ccc", alpha=0.95),
    )
    ax.set_xlabel(r"additive gap  $d(n_1,n_4)-\sum$ hops")
    ax.set_ylabel("count")
    ax.set_title(r"Factorization: $\pi(n_4|n_1)$ vs $\prod$ hop $\pi$", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.28)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    model_short = str(model_id).split("/")[-1]
    fig.suptitle(
        f"Type A v6 — Markov cloze sequence logπ   ·   {model_short} layer {layer_index}",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--layer-index", type=int, default=LAYER_INDEX)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--distances", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--primary-variant", default="natural")
    p.add_argument("--model-id", default=os.getenv("SAE_MODEL", "google/gemma-2-9b"))
    args = p.parse_args()

    dist = args.distances or distances_jsonl(args.layer_index, args.output_dir)
    out = args.out or (
        figures_dir_for_layer(args.layer_index, args.output_dir)
        / "type_a_markov_factorization.png"
    )
    rows = load_type_a(dist)
    if not rows:
        raise SystemExit(f"No Type A rows in {dist}")
    path = plot_type_a(
        rows,
        out_path=out,
        layer_index=args.layer_index,
        primary_variant=args.primary_variant,
        model_id=args.model_id,
    )
    print(f"n_type_A={len(rows)}")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
