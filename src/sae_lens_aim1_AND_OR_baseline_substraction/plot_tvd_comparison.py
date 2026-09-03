"""
Plot differential SAE composition TVD: min / mean / max vs empirical AND and OR.

For each orthogonality category (after ReLU(V − V_base)):
  - Top row:  TVD(V'_AND, theory) for theory in {min, mean, max}
  - Bottom:   TVD(V'_OR,  theory) for theory in {min, mean, max}

Boxes = distribution across pairs; whiskers / ribbons use 5th–95th percentiles.
Also writes a cross-category summary bar chart.
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
    figures_tvd_dir_for_layer,
    metrics_csv_for_layer,
)


AGG_NAMES = ("min", "mean", "max")
COLORS = {"min": "#1b9e77", "mean": "#d95f02", "max": "#7570b3"}


def _tvd_columns(op: str) -> dict[str, str]:
    """Map aggregation name → metrics CSV column for AND or OR."""
    return {agg: f"tvd_{op}_vs_{agg}" for agg in AGG_NAMES}


def load_metrics(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = ["category_id", "category_name"]
    for op in ("and", "or"):
        needed.extend(_tvd_columns(op).values())
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
    for cid, g in df.groupby("category_id"):
        cid_i = int(cid)
        blob: dict[str, Any] = {
            "category_name": str(g["category_name"].iloc[0]),
            "n_pairs": int(len(g)),
            "and": {},
            "or": {},
        }
        for op in ("and", "or"):
            for agg, col in _tvd_columns(op).items():
                vals = g[col].astype(float).to_numpy()
                blob[op][agg] = {
                    "values": vals,
                    "mean": float(np.mean(vals)),
                    "q_low": float(np.quantile(vals, q_low)),
                    "q_high": float(np.quantile(vals, q_high)),
                }
        out[cid_i] = blob
    return out


def plot_category_tvd(
    summary: dict[int, dict[str, Any]],
    *,
    out_dir: Path | str,
    layer_index: int | None = None,
    figsize: tuple[float, float] = (10, 5.5),
) -> list[Path]:
    """One figure per category: AND / OR rows × min / mean / max box panels."""
    import matplotlib.pyplot as plt

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    layer_tag = f"L={layer_index}  " if layer_index is not None else ""
    paths: list[Path] = []

    for cid in sorted(summary):
        blob = summary[cid]
        fig, axes = plt.subplots(2, 3, figsize=figsize, sharey=True)
        fig.suptitle(
            f"{layer_tag}[diff] Cat {cid}: {blob['category_name']}  "
            f"(n={blob['n_pairs']} pairs)\n"
            "y = TVD(V', theory) after ReLU(V−V_base)   boxes = pair distribution",
            fontsize=11,
        )
        for row_i, op in enumerate(("and", "or")):
            for col_i, agg in enumerate(AGG_NAMES):
                ax = axes[row_i, col_i]
                vals = blob[op][agg]["values"]
                color = COLORS[agg]
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
                    [blob[op][agg]["mean"]],
                    color=color,
                    s=36,
                    zorder=3,
                    label="mean",
                )
                ax.set_title(f"{op.upper()} vs {agg}")
                ax.set_xticks([])
                ax.set_xlim(-0.7, 0.7)
                ax.set_ylim(bottom=0.0)
                ax.grid(True, axis="y", alpha=0.25)
            axes[row_i, 0].set_ylabel(f"TVD ({op.upper()})")
        fig.tight_layout()
        dest = out_dir_path / f"tvd_and_or_cat{cid}.png"
        fig.savefig(dest, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(dest)

    return paths


def plot_summary_bars(
    summary: dict[int, dict[str, Any]],
    *,
    out_dir: Path | str,
    layer_index: int | None = None,
    figsize: tuple[float, float] = (11, 4.2),
) -> Path:
    """Cross-category mean TVD bars with 5–95% error bars for AND and OR."""
    import matplotlib.pyplot as plt

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    layer_tag = f"L={layer_index}  " if layer_index is not None else ""

    cids = sorted(summary)
    x = np.arange(len(cids))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    fig.suptitle(
        f"{layer_tag}Differential SAE composition TVD by category "
        "(ReLU(V−V_base))\n"
        "bars = mean across pairs; error = 5th–95th percentile",
        fontsize=11,
    )

    for ax, op in zip(axes, ("and", "or")):
        for i, agg in enumerate(AGG_NAMES):
            means = [summary[c][op][agg]["mean"] for c in cids]
            lows = [summary[c][op][agg]["q_low"] for c in cids]
            highs = [summary[c][op][agg]["q_high"] for c in cids]
            yerr = np.vstack(
                [
                    np.asarray(means) - np.asarray(lows),
                    np.asarray(highs) - np.asarray(means),
                ]
            )
            ax.bar(
                x + (i - 1) * width,
                means,
                width=width,
                color=COLORS[agg],
                alpha=0.85,
                label=agg,
                yerr=yerr,
                capsize=3,
                error_kw={"elinewidth": 0.9},
            )
        ax.set_title(f"Empirical {op.upper()} vs theory")
        ax.set_xticks(x)
        ax.set_xticklabels([f"C{c}" for c in cids])
        ax.set_xlabel("category")
        ax.set_ylim(bottom=0.0)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False)
    axes[0].set_ylabel("TVD")
    fig.tight_layout()

    dest = out_dir_path / "tvd_and_or_summary.png"
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot differential AND/OR TVD vs min/mean/max from SAE metrics CSV."
        )
    )
    parser.add_argument("--layer-index", type=int, default=LAYER_INDEX)
    parser.add_argument("--metrics-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--categories", type=int, nargs="*", default=None)
    args = parser.parse_args()

    layer = args.layer_index
    metrics_csv = args.metrics_csv or metrics_csv_for_layer(layer, args.output_dir)
    out_dir = args.out_dir or figures_tvd_dir_for_layer(layer, args.output_dir)

    if not metrics_csv.exists():
        raise SystemExit(f"Metrics CSV not found: {metrics_csv}")

    df = load_metrics(metrics_csv)
    if args.categories is not None:
        df = df[df["category_id"].isin(args.categories)]
    if df.empty:
        raise SystemExit("No rows to plot after filtering.")

    summary = summarize_category(df)
    paths = plot_category_tvd(summary, out_dir=out_dir, layer_index=layer)
    summary_path = plot_summary_bars(summary, out_dir=out_dir, layer_index=layer)

    print(f"Loaded {len(df)} pairs from {metrics_csv}")
    for cid, blob in sorted(summary.items()):
        print(f"cat={cid} n={blob['n_pairs']} | {blob['category_name']}")
    for p in paths:
        print("Saved plot →", p)
    print("Saved summary →", summary_path)


if __name__ == "__main__":
    main()
