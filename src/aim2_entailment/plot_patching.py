"""
Figures for Aim-2 patching: clean vs corrupted vs patched, trad vs ECT.

CSV stores mean hypothesis-token *log*-probabilities. The plots show those
log-probs (higher = more likely) so differences stay visible; token
probability is P = exp(log p).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_OUTPUT_DIR, figures_dir, results_csv

CONDITION_COLS = {
    "clean": "baseline_logprob_clean",
    "corrupted": "baseline_logprob_corrupted",
    "patched": "patched_logprob",
}
CONDITION_LABELS = {
    "clean": "Clean baseline",
    "corrupted": "Corrupted baseline",
    "patched": "Patched",
}
CONDITION_COLORS = {
    "clean": "#2ca02c",
    "corrupted": "#7f7f7f",
    "patched": "#d62728",
}
CONDITION_STYLES = {
    "clean": "--",
    "corrupted": ":",
    "patched": "-",
}
METHOD_COLORS = {"trad": "#1f77b4", "ect": "#ff7f0e"}
METHOD_LABELS = {"trad": r"$v_{\mathrm{trad}}$", "ect": r"$v_{\mathrm{ECT}}$"}


def load_results(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = [
        "pair_id",
        "layer_index",
        "alpha",
        "vector_type",
        *CONDITION_COLS.values(),
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Results CSV missing columns: {missing}")
    df["layer_index"] = df["layer_index"].astype(int)
    df["alpha"] = df["alpha"].astype(float)
    df["vector_type"] = df["vector_type"].astype(str).str.lower()
    return df


def _mean_sem(values: pd.Series) -> tuple[float, float]:
    vals = values.astype(float).to_numpy()
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(vals))
    if n < 2:
        return mean, 0.0
    return mean, float(np.std(vals, ddof=1) / np.sqrt(n))


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    keys = ["vector_type", "alpha", "layer_index"]
    for (vtype, alpha, layer), g in df.groupby(keys, sort=True):
        rec: dict[str, float | int | str] = {
            "vector_type": str(vtype),
            "alpha": float(alpha),
            "layer_index": int(layer),
            "n_pairs": int(g["pair_id"].nunique()),
        }
        for name, col in CONDITION_COLS.items():
            mean, sem = _mean_sem(g[col])
            rec[f"{name}_mean"] = mean
            rec[f"{name}_sem"] = sem
        rows.append(rec)
    return pd.DataFrame(rows)


def _draw_condition_lines(ax, sub: pd.DataFrame, *, patched_color: str) -> None:
    layers = sub["layer_index"].to_numpy()
    for name in ("clean", "corrupted", "patched"):
        color = patched_color if name == "patched" else CONDITION_COLORS[name]
        ax.errorbar(
            layers,
            sub[f"{name}_mean"],
            yerr=sub[f"{name}_sem"],
            color=color,
            linestyle=CONDITION_STYLES[name],
            marker="o",
            markersize=4.5,
            capsize=2.5,
            linewidth=1.6,
            label=CONDITION_LABELS[name],
        )


def plot_conditions_by_layer(
    df: pd.DataFrame,
    *,
    out_dir: Path | str,
) -> Path:
    """
    2 rows (trad, ECT) × one column per α.

    Each panel: mean ± SEM across pairs of clean, corrupted, and patched
    hypothesis-token log-probability vs layer.
    """
    import matplotlib.pyplot as plt

    summary = _summarize(df)
    methods = [m for m in ("trad", "ect") if m in set(summary["vector_type"])]
    alphas = sorted(summary["alpha"].unique())
    if not methods or not alphas:
        raise ValueError("No trad/ect rows to plot.")

    n_rows, n_cols = len(methods), len(alphas)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(3.4 * n_cols, 6.5), 3.3 * n_rows + 0.6),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    n_pairs = int(df["pair_id"].nunique())
    src = df["source"].dropna().astype(str).unique().tolist() if "source" in df.columns else []
    src_tag = "SNLI " if src == ["snli"] else ""
    fig.suptitle(
        "Aim-2 activation patching: hypothesis-token log-probability\n"
        f"mean ± SEM across {n_pairs} {src_tag}entailment pair(s);  "
        r"$P = \exp(\mathrm{log}\ p)$",
        fontsize=11,
    )

    for row_i, method in enumerate(methods):
        for col_i, alpha in enumerate(alphas):
            ax = axes[row_i][col_i]
            sub = summary[
                (summary["vector_type"] == method) & (summary["alpha"] == alpha)
            ].sort_values("layer_index")
            if sub.empty:
                ax.set_visible(False)
                continue
            _draw_condition_lines(ax, sub, patched_color=METHOD_COLORS[method])
            ax.set_title(f"{METHOD_LABELS[method]}   α={alpha:g}", fontsize=10)
            ax.grid(True, alpha=0.28)
            if row_i == n_rows - 1:
                ax.set_xlabel("layer")
            if col_i == 0:
                ax.set_ylabel("mean log-prob")
            if row_i == 0 and col_i == 0:
                ax.legend(frameon=False, fontsize=8, loc="best")

    fig.tight_layout()
    dest = Path(out_dir) / "patching_logprob_trad_vs_ect.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return dest


def plot_trad_vs_ect_overlay(
    df: pd.DataFrame,
    *,
    out_dir: Path | str,
) -> Path:
    """
    One column per α: clean / corrupted as shared references, patched trad vs ECT.
    """
    import matplotlib.pyplot as plt

    summary = _summarize(df)
    alphas = sorted(summary["alpha"].unique())
    n_cols = len(alphas)
    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(max(3.6 * n_cols, 6.5), 4.2),
        sharey=True,
        squeeze=False,
    )
    n_pairs = int(df["pair_id"].nunique())
    fig.suptitle(
        "Patched trad vs ECT against clean and corrupted baselines\n"
        f"mean ± SEM across {n_pairs} pair(s)",
        fontsize=11,
    )

    for col_i, alpha in enumerate(alphas):
        ax = axes[0][col_i]
        sub_any = summary[summary["alpha"] == alpha]
        # Baselines do not depend on vector_type; use trad if present else first.
        base = sub_any[sub_any["vector_type"] == "trad"]
        if base.empty:
            base = sub_any.drop_duplicates("layer_index")
        base = base.sort_values("layer_index")
        ax.errorbar(
            base["layer_index"],
            base["clean_mean"],
            yerr=base["clean_sem"],
            color=CONDITION_COLORS["clean"],
            linestyle="--",
            marker="s",
            markersize=4,
            capsize=2,
            linewidth=1.4,
            label="Clean baseline",
        )
        ax.errorbar(
            base["layer_index"],
            base["corrupted_mean"],
            yerr=base["corrupted_sem"],
            color=CONDITION_COLORS["corrupted"],
            linestyle=":",
            marker="s",
            markersize=4,
            capsize=2,
            linewidth=1.4,
            label="Corrupted baseline",
        )
        for method in ("trad", "ect"):
            sub = sub_any[sub_any["vector_type"] == method].sort_values("layer_index")
            if sub.empty:
                continue
            ax.errorbar(
                sub["layer_index"],
                sub["patched_mean"],
                yerr=sub["patched_sem"],
                color=METHOD_COLORS[method],
                linestyle="-",
                marker="o",
                markersize=5,
                capsize=2.5,
                linewidth=1.8,
                label=f"Patched {METHOD_LABELS[method]}",
            )
        ax.set_title(f"α={alpha:g}", fontsize=10)
        ax.set_xlabel("layer")
        ax.grid(True, alpha=0.28)
        if col_i == 0:
            ax.set_ylabel("mean log-prob")
            ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    dest = Path(out_dir) / "patching_trad_vs_ect_overlay.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return dest


def plot_patching_results(
    csv_path: Path | str,
    output_dir: Path | str | None = None,
) -> list[Path]:
    csv_path = Path(csv_path)
    output_dir = Path(output_dir) if output_dir is not None else csv_path.parent
    out_dir = figures_dir(output_dir)
    df = load_results(csv_path)
    if df.empty:
        raise ValueError(f"No rows in {csv_path}")
    return [
        plot_conditions_by_layer(df, out_dir=out_dir),
        plot_trad_vs_ect_overlay(df, out_dir=out_dir),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Aim-2 patched vs clean/corrupted log-probs (trad vs ECT)."
    )
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    csv_path = args.csv or results_csv(args.output_dir)
    if not csv_path.exists():
        raise SystemExit(f"Results CSV not found: {csv_path}")
    paths = plot_patching_results(csv_path, args.output_dir)
    print(f"Loaded {csv_path}")
    for path in paths:
        print(f"Saved plot → {path}")


if __name__ == "__main__":
    main()
