"""Type A plot for TVD / half-L1 path costs on shared trajectory (v5)."""

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
    "j": "#8a4b2b",
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


def _metric_arrays(rows: list[dict[str, Any]], key: str) -> dict[str, np.ndarray]:
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


def _padded_ylim(arrays: list[np.ndarray], *, floor: float | None = 0.0) -> tuple[float, float]:
    lo = float(min(a.min() for a in arrays))
    hi = float(max(a.max() for a in arrays))
    span = max(hi - lo, 1e-3)
    pad = 0.08 * span
    bottom = lo - pad if floor is None else min(floor, lo - pad * 0.25)
    if floor is not None and lo >= 0:
        bottom = floor
    return bottom, hi + pad


def _draw_path_panel(
    ax,
    arrays: dict[str, np.ndarray],
    *,
    ylabel: str,
    title: str,
    mean_fmt: str = "{:.2f}",
    annotate_gap: bool = True,
) -> None:
    order = ["n1_n2", "n2_n3", "n3_n4", "sum_hops", "n1_n4"]
    labels = [
        r"$\delta(n_1,n_2)$",
        r"$\delta(n_2,n_3)$",
        r"$\delta(n_3,n_4)$",
        r"$\sum\delta$ (path)",
        r"$\delta(n_1,n_4)$",
    ]
    data = [arrays[k] for k in order]
    pos = np.arange(len(order))
    bp = ax.boxplot(
        data,
        positions=pos,
        widths=0.58,
        patch_artist=True,
        showfliers=True,
        flierprops={
            "marker": "o",
            "markersize": 2.5,
            "markerfacecolor": "#666",
            "markeredgecolor": "none",
            "alpha": 0.35,
        },
        whiskerprops={"color": "#333", "linewidth": 1.0},
        capprops={"color": "#333", "linewidth": 1.0},
        medianprops={"color": "#1a1a1a", "linewidth": 1.8},
    )
    for box, key in zip(bp["boxes"], order):
        box.set_facecolor(COLORS.get(key, "#666"))
        box.set_alpha(0.78)
        box.set_edgecolor("#222")
        box.set_linewidth(0.9)

    means = [float(np.mean(arrays[k])) for k in order]
    ax.scatter(pos, means, color="#111", s=36, zorder=5)
    y0, y1 = _padded_ylim(data, floor=0.0)
    for x, m in zip(pos, means):
        ax.text(
            x,
            m + 0.03 * (y1 - y0),
            mean_fmt.format(m),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#111",
        )

    if annotate_gap and means[3] > means[4] + 1e-6:
        ax.annotate(
            "path > direct",
            xy=(4, means[4]),
            xytext=(3.1, means[3] * 0.78),
            fontsize=8,
            color="#a33b2b",
            ha="center",
            arrowprops=dict(
                arrowstyle="->",
                color="#a33b2b",
                lw=1.3,
                connectionstyle="arc3,rad=0.15",
            ),
        )

    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(y0, y1)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.grid(True, axis="y", alpha=0.28)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _draw_gap_hist(
    ax,
    series: list[tuple[str, np.ndarray, str]],
    *,
    title: str,
    note: str,
    legend_loc: str = "upper left",
) -> None:
    all_g = np.concatenate([a for _, a, _ in series])
    lo = float(all_g.min())
    hi = float(all_g.max())
    span = max(hi - lo, 0.05)
    x_lo = lo - 0.06 * span
    x_hi = max(0.02, hi + 0.08 * span)
    bins = np.linspace(x_lo, x_hi, 36)

    for lab, vals, color in series:
        ax.hist(
            vals,
            bins=bins,
            color=color,
            alpha=0.65,
            edgecolor="#333",
            linewidth=0.45,
            label=f"{lab} (mean {np.mean(vals):.3f})",
        )
    ax.axvline(0.0, color="#a33b2b", lw=1.8, ls="--", label="equality (gap=0)")
    ax.text(
        0.98,
        0.97,
        note,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=8.5,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#fff",
            edgecolor="#ccc",
            alpha=0.95,
        ),
    )
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel(r"additive gap  $\delta(n_1,n_4)-\sum\delta$ hops")
    ax.set_ylabel("count")
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc=legend_loc)
    ax.grid(True, axis="y", alpha=0.28)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_type_a(
    sae: dict[str, np.ndarray],
    j_arr: dict[str, np.ndarray] | None,
    *,
    out_path: Path,
    layer_index: int,
) -> Path:
    """Combined SAE path costs + SAE/J gap overlay (original two-panel figure)."""
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = int(sae["additive_gap"].size)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.8, 5.4),
        gridspec_kw={"width_ratios": [1.35, 1.05], "wspace": 0.32},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("#f7f4ef")
    for ax in axes:
        ax.set_facecolor("#fbfaf7")

    _draw_path_panel(
        axes[0],
        sae,
        ylabel=r"TVD $=\frac{1}{2}\|\pi_a-\pi_b\|_1$  (SAE fingerprints)",
        title="Shared-trajectory path costs (SAE)",
        mean_fmt="{:.2f}",
    )

    g_sae = sae["additive_gap"]
    series: list[tuple[str, np.ndarray, str]] = [("SAE", g_sae, COLORS["sae"])]
    note = (
        f"n = {n}\n"
        f"SAE |gap|<0.05: {100 * np.mean(np.abs(g_sae) < 0.05):.0f}%\n"
        f"SAE |gap|<0.1: {100 * np.mean(np.abs(g_sae) < 0.1):.0f}%\n"
        f"SAE gap≤0: {100 * np.mean(g_sae <= 0):.0f}%"
    )
    if j_arr is not None and j_arr["additive_gap"].size:
        g_j = j_arr["additive_gap"]
        series.append(("J-lens", g_j, COLORS["j"]))
        note += (
            f"\nJ |gap|<0.05: {100 * np.mean(np.abs(g_j) < 0.05):.0f}%"
            f"\nJ |gap|<0.1: {100 * np.mean(np.abs(g_j) < 0.1):.0f}%"
        )

    _draw_gap_hist(
        axes[1],
        series,
        title="Path additivity (TVD / half-L1)",
        note=note,
        legend_loc="upper left",
    )

    fig.suptitle(
        f"Type A v5 — Shared traj + TVD path cost   ·   Gemma-2-2B SAE layer {layer_index}",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def plot_type_a_j_lens(
    j_arr: dict[str, np.ndarray],
    *,
    out_path: Path,
    layer_index: int,
) -> Path:
    """Dedicated J-lens figure with path-cost boxes + gap histogram (J scale)."""
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = int(j_arr["additive_gap"].size)
    g_j = j_arr["additive_gap"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.8, 5.4),
        gridspec_kw={"width_ratios": [1.35, 1.05], "wspace": 0.32},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("#f7f4ef")
    for ax in axes:
        ax.set_facecolor("#fbfaf7")

    _draw_path_panel(
        axes[0],
        j_arr,
        ylabel=r"TVD $=\frac{1}{2}\|\pi_a-\pi_b\|_1$  (J-lens fingerprints)",
        title="Shared-trajectory path costs (J-lens)",
        mean_fmt="{:.3f}",
    )

    note = (
        f"n = {n}\n"
        f"mean gap = {np.mean(g_j):.3f}\n"
        f"median gap = {np.median(g_j):.3f}\n"
        f"|gap|<0.05: {100 * np.mean(np.abs(g_j) < 0.05):.0f}%\n"
        f"|gap|<0.1: {100 * np.mean(np.abs(g_j) < 0.1):.0f}%\n"
        f"gap≤0: {100 * np.mean(g_j <= 0):.0f}%"
    )
    _draw_gap_hist(
        axes[1],
        [("J-lens", g_j, COLORS["j"])],
        title="J-lens path additivity (zoomed)",
        note=note,
        legend_loc="upper left",
    )

    fig.suptitle(
        f"Type A v5 — J-lens TVD path cost   ·   Gemma-2-2B layer {layer_index}",
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
    p.add_argument("--out-j", type=Path, default=None)
    p.add_argument("--skip-combined", action="store_true")
    args = p.parse_args()

    dist = args.distances or distances_jsonl(args.layer_index, args.output_dir)
    fig_dir = figures_dir_for_layer(args.layer_index, args.output_dir)
    out = args.out or (fig_dir / "type_a_tvd_path.png")
    out_j = args.out_j or (fig_dir / "type_a_tvd_path_j_lens.png")

    rows = load_type_a(dist)
    if not rows:
        raise SystemExit(f"No Type A rows in {dist}")
    sae = _metric_arrays(rows, "sae_tvd")
    j_arr = None
    if any("j_tvd" in r for r in rows):
        j_arr = _metric_arrays(rows, "j_tvd")

    if not args.skip_combined:
        path = plot_type_a(sae, j_arr, out_path=out, layer_index=args.layer_index)
        print(f"n_type_A={len(sae['additive_gap'])}")
        print(f"Wrote {path}")

    if j_arr is not None and j_arr["additive_gap"].size:
        path_j = plot_type_a_j_lens(
            j_arr, out_path=out_j, layer_index=args.layer_index
        )
        print(f"n_type_A_j={len(j_arr['additive_gap'])}")
        print(f"Wrote {path_j}")
    else:
        print("No j_tvd rows — skipped J-lens figure.")


if __name__ == "__main__":
    main()
