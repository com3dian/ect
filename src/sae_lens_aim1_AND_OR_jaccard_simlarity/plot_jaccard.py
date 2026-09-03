"""
Jaccard boxplots for SAE set-theory AND/OR experiment.

Primary comparison (per README):
  AND vs Min-Target (∩)  vs  AND vs Max-Target (∪ control)

Also writes OR panels and a cross-category summary bar chart.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_OUTPUT_DIR,
    LAYER_INDEX,
    figures_dir_for_layer,
    metrics_csv_for_layer,
)


# (column, panel title, color)
AND_SERIES = (
    ("jaccard_and_vs_min", "AND vs ∩ (limit)", "#1b9e77"),
    ("jaccard_and_vs_max", "AND vs ∪ (control)", "#d95f02"),
)
OR_SERIES = (
    ("jaccard_or_vs_max", "OR vs ∪ (colimit)", "#7570b3"),
    ("jaccard_or_vs_min", "OR vs ∩ (control)", "#e7298a"),
)


def load_metrics(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = ["category_id", "category_name"] + [c for c, _, _ in AND_SERIES + OR_SERIES]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Metrics CSV missing columns: {missing}")
    return df


def summarize_category(
    df: pd.DataFrame,
    *,
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    cols = [c for c, _, _ in AND_SERIES + OR_SERIES]
    for cid, g in df.groupby("category_id"):
        cid_i = int(cid)
        blob: dict[str, Any] = {
            "category_name": str(g["category_name"].iloc[0]),
            "n_pairs": int(len(g)),
            "series": {},
        }
        for col in cols:
            vals = g[col].astype(float).to_numpy()
            blob["series"][col] = {
                "values": vals,
                "mean": float(np.mean(vals)),
                "q_low": float(np.quantile(vals, q_low)),
                "q_high": float(np.quantile(vals, q_high)),
            }
        out[cid_i] = blob
    return out


def plot_category_jaccard(
    summary: dict[int, dict[str, Any]],
    *,
    out_dir: Path | str,
    layer_index: int | None = None,
    figsize: tuple[float, float] = (9.5, 5.2),
) -> list[Path]:
    """One figure per category: AND row (∩ vs ∪) and OR row (∪ vs ∩)."""
    import matplotlib.pyplot as plt

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    layer_tag = f"L={layer_index}  " if layer_index is not None else ""
    paths: list[Path] = []

    for cid in sorted(summary):
        blob = summary[cid]
        fig, axes = plt.subplots(2, 2, figsize=figsize, sharey=True)
        fig.suptitle(
            f"{layer_tag}Cat {cid}: {blob['category_name']}  (n={blob['n_pairs']} pairs)\n"
            "y = Jaccard(empirical set, theory set)   boxes = pair distribution",
            fontsize=11,
        )
        for row_i, series in enumerate((AND_SERIES, OR_SERIES)):
            for col_i, (col, title, color) in enumerate(series):
                ax = axes[row_i, col_i]
                vals = blob["series"][col]["values"]
                bp = ax.boxplot(
                    [vals],
                    positions=[0],
                    widths=0.55,
                    patch_artist=True,
                    showfliers=False,
                )
                for box in bp["boxes"]:
                    box.set(facecolor=color, alpha=0.45)
                for median in bp["medians"]:
                    median.set(color=color, linewidth=1.8)
                ax.scatter(
                    [0],
                    [blob["series"][col]["mean"]],
                    color=color,
                    s=36,
                    zorder=3,
                )
                ax.set_title(title)
                ax.set_xticks([])
                ax.set_xlim(-0.7, 0.7)
                ax.set_ylim(0.0, 1.05)
                ax.grid(True, axis="y", alpha=0.25)
            axes[row_i, 0].set_ylabel("Jaccard")
        fig.tight_layout()
        dest = out_dir_path / f"jaccard_cat{cid}.png"
        fig.savefig(dest, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(dest)

    return paths


def plot_and_min_vs_max_focus(
    summary: dict[int, dict[str, Any]],
    *,
    out_dir: Path | str,
    layer_index: int | None = None,
    figsize: tuple[float, float] = (10, 3.8),
) -> Path:
    """
    README focus plot: AND vs Min-Target vs AND vs Max-Target across categories.
    """
    import matplotlib.pyplot as plt

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    layer_tag = f"L={layer_index}  " if layer_index is not None else ""

    cids = sorted(summary)
    fig, axes = plt.subplots(1, len(cids), figsize=figsize, sharey=True)
    if len(cids) == 1:
        axes = [axes]
    fig.suptitle(
        f"{layer_tag}AND Jaccard: ∩ (limit) vs ∪ (control)\n"
        "If AND executes a categorical limit, green (∩) should exceed orange (∪)",
        fontsize=11,
    )

    for ax, cid in zip(axes, cids):
        blob = summary[cid]
        data = []
        colors = []
        labels = []
        for col, title, color in AND_SERIES:
            data.append(blob["series"][col]["values"])
            colors.append(color)
            labels.append("∩" if "min" in col else "∪")
        bp = ax.boxplot(data, positions=[0, 1], widths=0.55, patch_artist=True, showfliers=False)
        for box, color in zip(bp["boxes"], colors):
            box.set(facecolor=color, alpha=0.45)
        for median, color in zip(bp["medians"], colors):
            median.set(color=color, linewidth=1.8)
        for i, (col, _, color) in enumerate(AND_SERIES):
            ax.scatter([i], [blob["series"][col]["mean"]], color=color, s=28, zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_title(f"C{cid}\nn={blob['n_pairs']}", fontsize=9)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("Jaccard(F_AND, ·)")
    fig.tight_layout()
    dest = out_dir_path / "jaccard_and_min_vs_max.png"
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dest


def plot_summary_bars(
    summary: dict[int, dict[str, Any]],
    *,
    out_dir: Path | str,
    layer_index: int | None = None,
    figsize: tuple[float, float] = (11, 4.2),
) -> Path:
    import matplotlib.pyplot as plt

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    layer_tag = f"L={layer_index}  " if layer_index is not None else ""

    cids = sorted(summary)
    x = np.arange(len(cids))
    width = 0.2

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    fig.suptitle(
        f"{layer_tag}SAE Jaccard by category\n"
        "bars = mean across pairs; error = 5th–95th percentile",
        fontsize=11,
    )

    for ax, series, title in (
        (axes[0], AND_SERIES, "AND vs theory sets"),
        (axes[1], OR_SERIES, "OR vs theory sets"),
    ):
        for i, (col, label, color) in enumerate(series):
            means = [summary[c]["series"][col]["mean"] for c in cids]
            lows = [summary[c]["series"][col]["q_low"] for c in cids]
            highs = [summary[c]["series"][col]["q_high"] for c in cids]
            yerr = np.vstack(
                [
                    np.asarray(means) - np.asarray(lows),
                    np.asarray(highs) - np.asarray(means),
                ]
            )
            short = "∩" if "min" in col else "∪"
            ax.bar(
                x + (i - 0.5) * width,
                means,
                width=width,
                color=color,
                alpha=0.85,
                label=f"{short}: {label.split('(')[0].strip()}",
                yerr=yerr,
                capsize=3,
                error_kw={"elinewidth": 0.9},
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f"C{c}" for c in cids])
        ax.set_xlabel("category")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Jaccard")
    fig.tight_layout()
    dest = out_dir_path / "jaccard_summary.png"
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot SAE Jaccard AND/OR set-theory comparisons."
    )
    parser.add_argument("--layer-index", type=int, default=LAYER_INDEX)
    parser.add_argument("--metrics-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--categories", type=int, nargs="*", default=None)
    args = parser.parse_args()

    layer = args.layer_index
    metrics_csv = args.metrics_csv or metrics_csv_for_layer(layer, args.output_dir)
    out_dir = args.out_dir or figures_dir_for_layer(layer, args.output_dir)

    if not metrics_csv.exists():
        raise SystemExit(f"Metrics CSV not found: {metrics_csv}")

    df = load_metrics(metrics_csv)
    if args.categories is not None:
        df = df[df["category_id"].isin(args.categories)]
    if df.empty:
        raise SystemExit("No rows to plot after filtering.")

    summary = summarize_category(df)
    paths = plot_category_jaccard(summary, out_dir=out_dir, layer_index=layer)
    focus = plot_and_min_vs_max_focus(summary, out_dir=out_dir, layer_index=layer)
    summary_path = plot_summary_bars(summary, out_dir=out_dir, layer_index=layer)

    print(f"Loaded {len(df)} pairs from {metrics_csv}")
    for cid, blob in sorted(summary.items()):
        s = blob["series"]
        print(
            f"cat={cid} n={blob['n_pairs']} | {blob['category_name']} | "
            f"AND∩={s['jaccard_and_vs_min']['mean']:.3f} "
            f"AND∪={s['jaccard_and_vs_max']['mean']:.3f} "
            f"OR∪={s['jaccard_or_vs_max']['mean']:.3f}"
        )
    for p in paths:
        print("Saved plot →", p)
    print("Saved focus  →", focus)
    print("Saved summary →", summary_path)


if __name__ == "__main__":
    main()
