"""
Type C visualization: adversarial / framed entailments.

Two-panel figure:
  (left)  scatter: bare entailment distance vs framed distances (rigidity)
  (right) triangle slack (token vs SAE), with token violators marked
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

COLORS = {
    "framed_entailed": "#2f5d7c",
    "framed_superordinate": "#8a4b2b",
    "token": "#8a4b2b",
    "sae": "#2f5d7c",
    "viol": "#a33b2b",
}


def load_type_c_distances(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("type_key") == "C":
                rows.append(row)
    return rows


def load_type_c_triangles(path: Path | str) -> dict[str, np.ndarray]:
    token: list[float] = []
    sae: list[float] = []
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("type_key") != "C":
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


def extract_scatter_arrays(
    rows: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    bare: list[float] = []
    framed_e: list[float] = []
    framed_s: list[float] = []
    for row in rows:
        td = row.get("token_distances") or {}
        need = ("entailed_superordinate", "framed_entailed", "framed_superordinate")
        if not all(k in td and np.isfinite(float(td[k])) for k in need):
            continue
        bare.append(float(td["entailed_superordinate"]))
        framed_e.append(float(td["framed_entailed"]))
        framed_s.append(float(td["framed_superordinate"]))
    return {
        "bare": np.asarray(bare, dtype=np.float64),
        "framed_entailed": np.asarray(framed_e, dtype=np.float64),
        "framed_superordinate": np.asarray(framed_s, dtype=np.float64),
    }


def plot_type_c(
    scatter: dict[str, np.ndarray],
    slacks: dict[str, np.ndarray],
    *,
    out_path: Path | str,
    layer_index: int,
    figsize: tuple[float, float] = (11.2, 4.4),
) -> Path:
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = int(scatter["bare"].size)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor("#f7f4ef")
    for ax in axes:
        ax.set_facecolor("#fbfaf7")

    # Left: rigidity scatter
    ax = axes[0]
    bare = scatter["bare"]
    fe = scatter["framed_entailed"]
    fs = scatter["framed_superordinate"]
    lim_hi = float(max(bare.max(), fe.max(), fs.max()) * 1.05)
    lim_lo = 0.0
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], ls="--", color="#888888", lw=1.2, label="y = x")
    ax.scatter(
        bare,
        fe,
        s=28,
        alpha=0.7,
        color=COLORS["framed_entailed"],
        edgecolors="#1a3344",
        linewidths=0.4,
        label=r"framed $\to$ entailed",
        zorder=3,
    )
    ax.scatter(
        bare,
        fs,
        s=28,
        alpha=0.7,
        color=COLORS["framed_superordinate"],
        edgecolors="#4a2814",
        linewidths=0.4,
        label=r"framed $\to$ superordinate",
        zorder=3,
    )
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"bare entailment $d($entailed$\to$superordinate$)$")
    ax.set_ylabel(r"framed distance $d_M$")
    ax.set_title("Does framing warp the entailment edge?", fontsize=11, pad=8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.28, color="#888888")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Right: slack hist with violators
    ax = axes[1]
    token = slacks["token"]
    sae = slacks["sae"]
    all_slack = np.concatenate([token, sae]) if sae.size else token
    lo = float(np.min(all_slack) - 0.5)
    hi = float(max(0.8, np.max(all_slack) + 0.5))
    bins = np.linspace(lo, hi, 32)

    ax.hist(
        token[token <= 0],
        bins=bins,
        color=COLORS["token"],
        alpha=0.65,
        edgecolor="#4a2814",
        linewidth=0.5,
        label=f"token ok (n={int(np.sum(token <= 0))})",
    )
    viol = token[token > 0]
    if viol.size:
        ax.hist(
            viol,
            bins=bins,
            color=COLORS["viol"],
            alpha=0.9,
            edgecolor="#5a1810",
            linewidth=0.6,
            label=f"token violators (n={viol.size})",
        )
    if sae.size:
        ax.hist(
            sae,
            bins=bins,
            color=COLORS["sae"],
            alpha=0.45,
            edgecolor="#1a3344",
            linewidth=0.5,
            label=f"SAE (mean {np.mean(sae):.1f})",
        )
    ax.axvline(0.0, color="#a33b2b", lw=1.8, ls="--", label="equality boundary")

    note = (
        f"n = {n}\n"
        f"token viol: {int(np.sum(token > 0))}/{token.size} "
        f"({100 * float(np.mean(token > 0)):.1f}%)\n"
        f"SAE viol: {int(np.sum(sae > 0))}/{sae.size if sae.size else 0}"
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

    ax.set_xlabel(
        r"triangle slack  $d($entailed,super$) - d($framed,entailed$) - d($framed,super$)$"
    )
    ax.set_ylabel("count")
    ax.set_title("Triangle inequality under framing", fontsize=11, pad=8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(True, axis="y", alpha=0.28, color="#888888")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.suptitle(
        f"Type C — Adversarial Functors   ·   Gemma-2-2B SAE layer {layer_index}",
        fontsize=12.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot Type C framed-entailment geometry.")
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
    out_path = args.out or (fig_dir / "type_c_adversarial_functors.png")

    rows = load_type_c_distances(dist_path)
    if not rows:
        raise SystemExit(f"No Type C rows in {dist_path}")
    scatter = extract_scatter_arrays(rows)
    slacks = load_type_c_triangles(tri_path)
    path = plot_type_c(scatter, slacks, out_path=out_path, layer_index=args.layer_index)
    print(f"n_type_C={scatter['bare'].size}")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
