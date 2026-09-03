"""
Type A plot for shared-trajectory v2 (SAE + logit-lens additive gaps).
"""

from __future__ import annotations

import argparse
import json
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
    "sae": "#1f4e5f",
    "lens": "#8a4b2b",
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


def _metric_arrays(
    rows: list[dict[str, Any]], key: str
) -> dict[str, np.ndarray]:
    """key in {sae_distances, logit_lens_distances}."""
    buckets: dict[str, list[float]] = {
        k: [] for k in ("n1_n2", "n2_n3", "n3_n4", "n1_n4", "sum_hops", "additive_gap")
    }
    for row in rows:
        d = row.get(key) or {}
        if not all(k in d for k in ("n1_n2", "n2_n3", "n3_n4", "n1_n4")):
            continue
        for k in ("n1_n2", "n2_n3", "n3_n4", "n1_n4"):
            buckets[k].append(float(d[k]))
        hops = float(d.get("sum_hops", d["n1_n2"] + d["n2_n3"] + d["n3_n4"]))
        gap = float(d.get("additive_gap", d["n1_n4"] - hops))
        buckets["sum_hops"].append(hops)
        buckets["additive_gap"].append(gap)
    return {k: np.asarray(v, dtype=np.float64) for k, v in buckets.items()}


def plot_type_a(
    sae: dict[str, np.ndarray],
    lens: dict[str, np.ndarray],
    *,
    out_path: Path,
    layer_index: int,
) -> Path:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    fig.patch.set_facecolor("#f7f4ef")
    for ax in axes:
        ax.set_facecolor("#fbfaf7")

    # Left: SAE chain profile
    ax = axes[0]
    order = ["n1_n2", "n2_n3", "n3_n4", "sum_hops", "n1_n4"]
    labels = [
        r"$d(n_1\!\to\!n_2)$",
        r"$d(n_2\!\to\!n_3)$",
        r"$d(n_3\!\to\!n_4)$",
        r"$\sum$ hops",
        r"$d(n_1\!\to\!n_4)$",
    ]
    data = [sae[k] for k in order]
    pos = np.arange(len(order))
    bp = ax.boxplot(
        data,
        positions=pos,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1a1a1a", "linewidth": 1.6},
    )
    for box, key in zip(bp["boxes"], order):
        box.set_facecolor(COLORS.get(key, "#666"))
        box.set_alpha(0.75)
    means = [float(np.mean(sae[k])) for k in order]
    ax.scatter(pos, means, color="#111", s=28, zorder=4)
    ax.annotate(
        "",
        xy=(4, means[4]),
        xytext=(3, means[3]),
        arrowprops=dict(arrowstyle="->", color="#a33b2b", lw=1.4),
    )
    ax.text(3.5, max(means[3], means[4]) + 0.15 * max(means), "direct vs Σ hops",
            ha="center", fontsize=8.5, color="#a33b2b")
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(r"SAE tropical $d_M=-\ln\langle\pi_i,\pi_j\rangle$")
    ax.set_title("Shared-trajectory SAE chain distances", fontsize=11)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, axis="y", alpha=0.28)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Right: additive gaps SAE vs logit-lens
    ax = axes[1]
    g_sae = sae["additive_gap"]
    g_ll = lens["additive_gap"]
    all_g = np.concatenate([g_sae, g_ll])
    lo, hi = float(all_g.min() - 0.3), float(all_g.max() + 0.3)
    # ensure 0 in range for reference
    lo, hi = min(lo, -0.2), max(hi, 0.2)
    bins = np.linspace(lo, hi, 30)
    ax.hist(g_sae, bins=bins, color=COLORS["sae"], alpha=0.7, edgecolor="#0d2b33",
            linewidth=0.5, label=f"SAE (mean {np.mean(g_sae):.2f})")
    ax.hist(g_ll, bins=bins, color=COLORS["lens"], alpha=0.55, edgecolor="#4a2814",
            linewidth=0.5, label=f"logit-lens (mean {np.mean(g_ll):.2f})")
    ax.axvline(0.0, color="#a33b2b", lw=1.8, ls="--", label="equality (gap=0)")
    n = len(g_sae)
    note = (
        f"n = {n}\n"
        f"SAE |gap|<1: {100 * np.mean(np.abs(g_sae) < 1):.0f}%\n"
        f"lens |gap|<1: {100 * np.mean(np.abs(g_ll) < 1):.0f}%\n"
        f"SAE gap≤0: {100 * np.mean(g_sae <= 0):.0f}%"
    )
    ax.text(0.04, 0.96, note, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff", edgecolor="#ccc", alpha=0.92))
    ax.set_xlabel(r"additive gap  $d(n_1,n_4)-\sum$ hops")
    ax.set_ylabel("count")
    ax.set_title("Shared-trajectory additivity", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.grid(True, axis="y", alpha=0.28)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.suptitle(
        f"Type A v2 — Shared trajectory   ·   Gemma-2-2B SAE layer {layer_index}",
        fontsize=12.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--layer-index", type=int, default=LAYER_INDEX)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--distances", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    dist = args.distances or distances_jsonl(args.layer_index, args.output_dir)
    out = args.out or (
        figures_dir_for_layer(args.layer_index, args.output_dir)
        / "type_a_shared_trajectory.png"
    )
    rows = load_type_a(dist)
    if not rows:
        raise SystemExit(f"No Type A rows in {dist}")
    sae = _metric_arrays(rows, "sae_distances")
    lens = _metric_arrays(rows, "logit_lens_distances")
    path = plot_type_a(sae, lens, out_path=out, layer_index=args.layer_index)
    print(f"n_type_A={len(sae['additive_gap'])}")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
