"""
Type A visualization: phylogenetic chain tropical distances.

Two-panel figure:
  (left)  hop / direct / Σ-hops distance distributions
  (right) additive gap  d(n1,n4) − [d(n1,n2)+d(n2,n3)+d(n3,n4)]
          (≤ 0 ⇒ tropical triangle / chain additivity holds)
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


CHAIN_KEYS = ("n1_n2", "n2_n3", "n3_n4", "n1_n4")
CHAIN_LABELS = {
    "n1_n2": r"$d(n_1\!\to\!n_2)$",
    "n2_n3": r"$d(n_2\!\to\!n_3)$",
    "n3_n4": r"$d(n_3\!\to\!n_4)$",
    "n1_n4": r"$d(n_1\!\to\!n_4)$",
    "sum_hops": r"$\sum$ hops",
}

# Calm scientific palette (avoid purple-on-white AI default).
COLORS = {
    "n1_n2": "#2a6f6f",
    "n2_n3": "#3d7c47",
    "n3_n4": "#8a6d3b",
    "n1_n4": "#a33b2b",
    "sum_hops": "#4a5568",
    "gap": "#1f4e5f",
}


def load_type_a_rows(
    path: Path | str,
) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("type_key") == "A":
                rows.append(row)
    return rows


def extract_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    hops: dict[str, list[float]] = {k: [] for k in CHAIN_KEYS}
    sum_hops: list[float] = []
    gaps: list[float] = []

    for row in rows:
        td = row.get("token_distances") or {}
        vals = []
        ok = True
        for k in CHAIN_KEYS:
            v = td.get(k)
            if v is None or not np.isfinite(float(v)):
                ok = False
                break
            vals.append(float(v))
        if not ok:
            continue
        for k, v in zip(CHAIN_KEYS, vals):
            hops[k].append(v)
        s = vals[0] + vals[1] + vals[2]
        sum_hops.append(s)
        gap = row.get("additive_gap_n1_n4")
        if gap is None:
            gap = vals[3] - s
        gaps.append(float(gap))

    out = {k: np.asarray(v, dtype=np.float64) for k, v in hops.items()}
    out["sum_hops"] = np.asarray(sum_hops, dtype=np.float64)
    out["gap"] = np.asarray(gaps, dtype=np.float64)
    return out


def plot_type_a(
    arrays: dict[str, np.ndarray],
    *,
    out_path: Path | str,
    layer_index: int,
    figsize: tuple[float, float] = (11.2, 4.4),
) -> Path:
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = int(arrays["gap"].size)

    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.25, 1.0]})
    fig.patch.set_facecolor("#f7f4ef")
    for ax in axes:
        ax.set_facecolor("#fbfaf7")

    # ----- Left: chain distance profile -----
    ax = axes[0]
    order = ["n1_n2", "n2_n3", "n3_n4", "sum_hops", "n1_n4"]
    data = [arrays[k] for k in order]
    positions = np.arange(len(order))
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1a1a1a", "linewidth": 1.6},
        whiskerprops={"color": "#444444"},
        capprops={"color": "#444444"},
        boxprops={"linewidth": 0.8, "edgecolor": "#333333"},
    )
    for box, key in zip(bp["boxes"], order):
        box.set_facecolor(COLORS[key])
        box.set_alpha(0.72)

    means = [float(np.mean(arrays[k])) for k in order]
    ax.scatter(positions, means, color="#111111", s=28, zorder=4, label="mean")

    # Connect hop means → sum → direct for the additive story.
    ax.plot(
        [0, 1, 2],
        [means[0], means[1], means[2]],
        color="#666666",
        lw=1.0,
        ls=":",
        alpha=0.7,
        zorder=2,
    )
    ax.annotate(
        "",
        xy=(4, means[4]),
        xytext=(3, means[3]),
        arrowprops=dict(arrowstyle="->", color="#a33b2b", lw=1.4),
    )
    ax.text(
        3.5,
        max(means[3], means[4]) + 1.2,
        "direct ≤ Σ hops",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#a33b2b",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels([CHAIN_LABELS[k] for k in order], fontsize=9)
    ax.set_ylabel(r"tropical distance $d_M=-\ln\pi(y\mid x)$")
    ax.set_title("Chain hop distances (token \(d_M\))", fontsize=11, pad=8)
    ax.grid(True, axis="y", alpha=0.28, color="#888888")
    ax.set_ylim(bottom=0.0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # ----- Right: additive gap -----
    ax = axes[1]
    gaps = arrays["gap"]
    bins = np.linspace(gaps.min() - 0.5, min(0.5, gaps.max() + 0.5), 28)
    ax.hist(
        gaps,
        bins=bins,
        color=COLORS["gap"],
        alpha=0.85,
        edgecolor="#0d2b33",
        linewidth=0.6,
    )
    ax.axvline(0.0, color="#a33b2b", lw=1.8, ls="--", label="equality boundary")
    ax.axvline(float(np.mean(gaps)), color="#111111", lw=1.4, label=f"mean = {np.mean(gaps):.1f}")

    frac = float(np.mean(gaps <= 0.0))
    note = (
        f"n = {n}\n"
        f"gap ≤ 0: {100 * frac:.0f}%\n"
        f"mean gap: {np.mean(gaps):.1f}"
    )
    ax.text(
        0.04,
        0.96,
        note,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.92),
    )

    ax.set_xlabel(r"additive gap  $d(n_1,n_4)-\sum$ hops")
    ax.set_ylabel("count")
    ax.set_title("Tropical chain additivity", fontsize=11, pad=8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(True, axis="y", alpha=0.28, color="#888888")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.suptitle(
        f"Type A — Strict Phylogenetic Chains   ·   Gemma-2-2B SAE layer {layer_index}",
        fontsize=12.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot Type A tropical chain geometry.")
    p.add_argument("--layer-index", type=int, default=LAYER_INDEX)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--distances", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    dist_path = args.distances or distances_jsonl(args.layer_index, args.output_dir)
    fig_dir = figures_dir_for_layer(args.layer_index, args.output_dir)
    out_path = args.out or (fig_dir / "type_a_chain_additivity.png")

    rows = load_type_a_rows(dist_path)
    if not rows:
        raise SystemExit(f"No Type A rows in {dist_path}")
    arrays = extract_arrays(rows)
    path = plot_type_a(arrays, out_path=out_path, layer_index=args.layer_index)
    print(f"n_type_A={len(arrays['gap'])}")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
