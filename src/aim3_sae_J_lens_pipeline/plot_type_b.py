"""
Type B visualization: sudden contextual shifts.

Two-panel figure:
  (left)  trajectory edge distances base→shift, shift→combined, base→combined
  (right) tropical triangle slack for base–shift–combined (token vs SAE)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    DEFAULT_OUTPUT_DIR,
    LAYER_INDEX,
    distances_jsonl,
    figures_dir_for_layer,
    triangle_csv,
)

EDGE_KEYS = ("base_shift", "shift_combined", "base_combined")
EDGE_LABELS = {
    "base_shift": r"$d($base$\to$shift$)$",
    "shift_combined": r"$d($shift$\to$comb$)$",
    "base_combined": r"$d($base$\to$comb$)$",
}
COLORS = {
    "base_shift": "#8a4b2b",
    "shift_combined": "#2f5d7c",
    "base_combined": "#3d6b4f",
    "token": "#8a4b2b",
    "sae": "#2f5d7c",
}


def load_type_b_distances(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("type_key") == "B":
                rows.append(row)
    return rows


def load_type_b_triangles(path: Path | str) -> dict[str, np.ndarray]:
    token: list[float] = []
    sae: list[float] = []
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("type_key") != "B":
                continue
            slack = float(row["slack"])
            if row.get("metric") == "token":
                token.append(slack)
            elif row.get("metric") == "sae":
                sae.append(slack)
    return {
        "token": np.asarray(token, dtype=np.float64),
        "sae": np.asarray(sae, dtype=np.float64),
    }


def extract_edge_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    buckets: dict[str, list[float]] = {k: [] for k in EDGE_KEYS}
    for row in rows:
        td = row.get("token_distances") or {}
        if not all(k in td and np.isfinite(float(td[k])) for k in EDGE_KEYS):
            continue
        for k in EDGE_KEYS:
            buckets[k].append(float(td[k]))
    return {k: np.asarray(v, dtype=np.float64) for k, v in buckets.items()}


def plot_type_b(
    edges: dict[str, np.ndarray],
    slacks: dict[str, np.ndarray],
    *,
    out_path: Path | str,
    layer_index: int,
    figsize: tuple[float, float] = (11.2, 4.4),
) -> Path:
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = int(edges["base_shift"].size)

    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.15, 1.05]})
    fig.patch.set_facecolor("#f7f4ef")
    for ax in axes:
        ax.set_facecolor("#fbfaf7")

    # Left: trajectory edges
    ax = axes[0]
    order = list(EDGE_KEYS)
    data = [edges[k] for k in order]
    positions = np.arange(len(order))
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1a1a1a", "linewidth": 1.6},
        whiskerprops={"color": "#444444"},
        capprops={"color": "#444444"},
        boxprops={"linewidth": 0.8, "edgecolor": "#333333"},
    )
    for box, key in zip(bp["boxes"], order):
        box.set_facecolor(COLORS[key])
        box.set_alpha(0.75)
    means = [float(np.mean(edges[k])) for k in order]
    ax.scatter(positions, means, color="#111111", s=28, zorder=4)

    # Annotate the triangle story: shortcut vs via-shift
    ax.annotate(
        "",
        xy=(2, means[2]),
        xytext=(0.5, max(means[0], means[1]) + 0.5),
        arrowprops=dict(arrowstyle="->", color="#8a4b2b", lw=1.2),
    )
    ax.text(
        1.2,
        max(means) + 1.0,
        r"shortcut $d($base$\to$comb$)$",
        fontsize=8.5,
        color="#8a4b2b",
        ha="center",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels([EDGE_LABELS[k] for k in order], fontsize=9)
    ax.set_ylabel(r"tropical distance $d_M=-\ln\pi(y\mid x)$")
    ax.set_title("Shift trajectory edges (token \(d_M\))", fontsize=11, pad=8)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, axis="y", alpha=0.28, color="#888888")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Right: slack distributions
    ax = axes[1]
    token = slacks["token"]
    sae = slacks["sae"]
    all_slack = np.concatenate([token, sae]) if sae.size else token
    lo = float(np.min(all_slack) - 0.5)
    hi = float(max(0.5, np.max(all_slack) + 0.5))
    bins = np.linspace(lo, hi, 30)

    ax.hist(
        token,
        bins=bins,
        color=COLORS["token"],
        alpha=0.65,
        edgecolor="#4a2814",
        linewidth=0.5,
        label=f"token (mean {np.mean(token):.1f})",
    )
    if sae.size:
        ax.hist(
            sae,
            bins=bins,
            color=COLORS["sae"],
            alpha=0.55,
            edgecolor="#1a3344",
            linewidth=0.5,
            label=f"SAE (mean {np.mean(sae):.1f})",
        )
    ax.axvline(0.0, color="#a33b2b", lw=1.8, ls="--", label="equality boundary")

    n_viol_tok = int(np.sum(token > 0))
    n_viol_sae = int(np.sum(sae > 0)) if sae.size else 0
    note = (
        f"n = {n}\n"
        f"token viol: {n_viol_tok}/{token.size}\n"
        f"SAE viol: {n_viol_sae}/{sae.size if sae.size else 0}"
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

    ax.set_xlabel(r"triangle slack  $d($base,comb$) - d($base,shift$) - d($shift,comb$)$")
    ax.set_ylabel("count")
    ax.set_title("Tropical triangle under disruption", fontsize=11, pad=8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(True, axis="y", alpha=0.28, color="#888888")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.suptitle(
        f"Type B — Sudden Contextual Shifts   ·   Gemma-2-2B SAE layer {layer_index}",
        fontsize=12.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot Type B tropical shift geometry.")
    p.add_argument("--layer-index", type=int, default=LAYER_INDEX)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--distances", type=Path, default=None)
    p.add_argument("--triangle", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    dist_path = args.distances or distances_jsonl(args.layer_index, args.output_dir)
    tri_path = args.triangle or triangle_csv(args.layer_index, args.output_dir)
    fig_dir = figures_dir_for_layer(args.layer_index, args.output_dir)
    out_path = args.out or (fig_dir / "type_b_contextual_shifts.png")

    rows = load_type_b_distances(dist_path)
    if not rows:
        raise SystemExit(f"No Type B rows in {dist_path}")
    edges = extract_edge_arrays(rows)
    slacks = load_type_b_triangles(tri_path)
    path = plot_type_b(edges, slacks, out_path=out_path, layer_index=args.layer_index)
    print(f"n_type_B={edges['base_shift'].size}")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
