"""Calibration plots for Aim-3 v7."""

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
    "chain": "#1f4e5f",
    "natural": "#2a6f6f",
    "shuffled_middles": "#8a4b2b",
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def plot_calibration(
    rows: list[dict[str, Any]],
    *,
    out_path: Path,
    layer_index: int,
    model_id: str = "google/gemma-2-9b",
) -> Path:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cr = np.asarray(
        [float(r["chain_rule"]["chain_rule_gap"]) for r in rows], dtype=float
    )
    cr = cr[np.isfinite(cr)]

    short = [r for r in rows if r.get("markov_variants")]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.6, 5.2),
        gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.32},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("#f7f4ef")
    for ax in axes:
        ax.set_facecolor("#fbfaf7")

    ax = axes[0]
    if cr.size:
        span = max(float(np.max(np.abs(cr))), 1e-3)
        # Zoom near zero if scoring is consistent; otherwise show full range.
        if float(np.mean(np.abs(cr))) < 0.05:
            lim = max(0.05, float(np.percentile(np.abs(cr), 99)) * 3, span)
            bins = np.linspace(-lim, lim, 41)
        else:
            lo, hi = float(cr.min()), float(cr.max())
            pad = 0.05 * max(hi - lo, 1e-3)
            bins = np.linspace(lo - pad, hi + pad, 41)
        ax.hist(cr, bins=bins, color=COLORS["chain"], alpha=0.75, edgecolor="#333", linewidth=0.4)
        ax.axvline(0.0, color="#a33b2b", lw=1.8, ls="--", label="exact tautology")
        note = (
            f"n = {cr.size}\n"
            f"mean = {np.mean(cr):.4g}\n"
            f"MAE = {np.mean(np.abs(cr)):.4g}\n"
            f"max|gap| = {np.max(np.abs(cr)):.4g}\n"
            f"|gap|<1e-2: {100 * np.mean(np.abs(cr) < 1e-2):.0f}%"
        )
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
    ax.set_xlabel(r"chain-rule gap  $\log\pi_{\mathrm{joint}}-\sum\log\pi_{\mathrm{factors}}$")
    ax.set_ylabel("count")
    ax.set_title("Sanity: joint vs factorized (should ≈ 0)", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.28)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    series: list[tuple[str, np.ndarray, str]] = []
    for vname, color in (
        ("natural", COLORS["natural"]),
        ("shuffled_middles", COLORS["shuffled_middles"]),
    ):
        vals = []
        for r in short:
            pack = (r.get("markov_variants") or {}).get(vname)
            if pack:
                vals.append(float(pack["distances"]["additive_gap"]))
        if vals:
            series.append((vname, np.asarray(vals, dtype=float), color))
    if series:
        all_g = np.concatenate([a for _, a, _ in series])
        lo, hi = float(all_g.min()), float(all_g.max())
        pad = 0.05 * max(hi - lo, 1.0)
        bins = np.linspace(lo - pad, hi + pad, 32)
        for lab, vals, color in series:
            ax.hist(
                vals,
                bins=bins,
                color=color,
                alpha=0.6,
                edgecolor="#333",
                linewidth=0.4,
                label=f"{lab} (mean {np.mean(vals):.2f})",
            )
        ax.axvline(0.0, color="#a33b2b", lw=1.8, ls="--", label="Markov equality")
        nat = series[0][1]
        note = (
            f"short-BPE n = {len(short)}\n"
            f"natural |gap|<2: {100 * np.mean(np.abs(nat) < 2):.0f}%\n"
            f"natural rel≈ see summary"
        )
        if len(series) >= 2:
            a, b = series[0][1], series[1][1]
            m = min(a.size, b.size)
            note += f"\nnat closer than shuf: {100 * np.mean(np.abs(a[:m]) < np.abs(b[:m])):.0f}%"
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
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    else:
        ax.text(0.5, 0.5, "No short-BPE Markov rows", ha="center", va="center")
    ax.set_xlabel(r"Markov additive gap  $d(n_1,n_4)-\sum$ hops")
    ax.set_ylabel("count")
    ax.set_title("Short-token Markov cloze (v6 claim)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.28)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    model_short = str(model_id).split("/")[-1]
    fig.suptitle(
        f"Type A v7 — Calibration   ·   {model_short} layer {layer_index}",
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
    p.add_argument("--model-id", default=os.getenv("SAE_MODEL", "google/gemma-2-9b"))
    args = p.parse_args()

    dist = args.distances or distances_jsonl(args.layer_index, args.output_dir)
    out = args.out or (
        figures_dir_for_layer(args.layer_index, args.output_dir) / "type_a_calibration.png"
    )
    rows = load_rows(dist)
    if not rows:
        raise SystemExit(f"No rows in {dist}")
    path = plot_calibration(
        rows,
        out_path=out,
        layer_index=args.layer_index,
        model_id=args.model_id,
    )
    print(f"n={len(rows)}")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
