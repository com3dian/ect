"""
Per-sample plots at a fixed layer: clean / corrupted baselines vs patched v_trad & v_ECT.

Reads patching_results.csv (hypothesis-token log-probs). One point per SNLI pair.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_OUTPUT_DIR, figures_dir, results_csv
from .plot_patching import CONDITION_COLORS, METHOD_COLORS, load_results

ECT_WINS_COLOR = "#1f77b4"   # blue: v_ECT > v_trad
TRAD_WINS_COLOR = "#d62728"  # red:  v_trad > v_ECT


def _resolve_alpha(sub: pd.DataFrame, alpha: float | None) -> float:
    if alpha is not None:
        return float(alpha)
    alphas = sorted(sub["alpha"].unique())
    return float(min(alphas, key=lambda a: abs(float(a) - 1.0)))


def _merge_trad_ect(
    df: pd.DataFrame,
    *,
    layer_index: int,
    alpha: float | None,
) -> pd.DataFrame:
    base = df[df["layer_index"] == layer_index].copy()
    if base.empty:
        raise ValueError(f"No rows for layer={layer_index}")

    alpha_used = _resolve_alpha(base, alpha)
    trad = base[(base["vector_type"] == "trad") & np.isclose(base["alpha"], alpha_used)]
    ect = base[(base["vector_type"] == "ect") & np.isclose(base["alpha"], alpha_used)]
    if trad.empty:
        raise ValueError(f"No trad rows for layer={layer_index}, alpha={alpha_used}")
    if ect.empty:
        raise ValueError(f"No ect rows for layer={layer_index}, alpha={alpha_used}")

    trad = trad.sort_values("pair_id").reset_index(drop=True)
    ect = ect.sort_values("pair_id").reset_index(drop=True)
    if set(trad["pair_id"]) != set(ect["pair_id"]):
        missing_trad = set(ect["pair_id"]) - set(trad["pair_id"])
        missing_ect = set(trad["pair_id"]) - set(ect["pair_id"])
        raise ValueError(
            f"pair_id mismatch at layer={layer_index}, alpha={alpha_used}: "
            f"missing trad={len(missing_trad)}, missing ect={len(missing_ect)}"
        )

    merged = trad[
        [
            "pair_id",
            "baseline_logprob_clean",
            "baseline_logprob_corrupted",
            "patched_logprob",
            "alpha",
        ]
    ].rename(columns={"patched_logprob": "patched_trad"})
    merged = merged.merge(
        ect[["pair_id", "patched_logprob"]].rename(columns={"patched_logprob": "patched_ect"}),
        on="pair_id",
        how="inner",
    )
    merged["alpha"] = alpha_used
    merged["diff_ect_minus_trad"] = merged["patched_ect"].astype(float) - merged["patched_trad"].astype(float)
    return merged.sort_values("pair_id").reset_index(drop=True)


def plot_layer_per_sample_all(
    merged: pd.DataFrame,
    *,
    layer_index: int,
    out_dir: Path | str,
) -> Path:
    """All samples on one figure: baselines + patched v_trad + patched v_ECT."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    alpha_used = float(merged["alpha"].iloc[0])
    n = len(merged)
    x = np.arange(n)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean = merged["baseline_logprob_clean"].astype(float).to_numpy()
    corrupt = merged["baseline_logprob_corrupted"].astype(float).to_numpy()
    patched_trad = merged["patched_trad"].astype(float).to_numpy()
    patched_ect = merged["patched_ect"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(max(14, n * 0.045), 6))
    ax.scatter(x, clean, s=6, c=CONDITION_COLORS["clean"], alpha=0.55, label="Clean baseline", zorder=2)
    ax.scatter(x, corrupt, s=6, c=CONDITION_COLORS["corrupted"], alpha=0.55, label="Corrupted baseline", zorder=2)
    ax.scatter(
        x,
        patched_trad,
        s=7,
        c=METHOD_COLORS["trad"],
        alpha=0.75,
        label=r"Patched $v_{\mathrm{trad}}$",
        zorder=3,
    )
    ax.scatter(
        x,
        patched_ect,
        s=7,
        c=METHOD_COLORS["ect"],
        alpha=0.75,
        label=r"Patched $v_{\mathrm{ECT}}$",
        zorder=3,
    )

    for i in range(n):
        ax.plot(
            [x[i], x[i], x[i], x[i]],
            [clean[i], corrupt[i], patched_trad[i], patched_ect[i]],
            color="#dddddd",
            linewidth=0.35,
            alpha=0.45,
            zorder=1,
        )

    ax.set_title(
        rf"Layer {layer_index}, $\alpha={alpha_used:g}$: per-sample log-prob ({n} pairs)",
        fontsize=11,
    )
    ax.set_xlabel("sample index (sorted by pair_id)")
    ax.set_ylabel("hypothesis-token log-prob")
    ax.grid(True, axis="y", alpha=0.28)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CONDITION_COLORS["clean"], markersize=6, label="Clean baseline"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CONDITION_COLORS["corrupted"], markersize=6, label="Corrupted baseline"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=METHOD_COLORS["trad"], markersize=6, label=r"Patched $v_{\mathrm{trad}}$"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=METHOD_COLORS["ect"], markersize=6, label=r"Patched $v_{\mathrm{ECT}}$"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="best", fontsize=9)

    fig.tight_layout()
    dest = out_dir / f"layer{layer_index}_per_sample_all_a{alpha_used:g}.png"
    fig.savefig(dest, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return dest


def plot_layer_ect_vs_trad_sorted(
    merged: pd.DataFrame,
    *,
    layer_index: int,
    out_dir: Path | str,
) -> Path:
    """
    v_ECT vs v_trad patched log-probs, sorted by (v_ECT - v_trad).

    Blue connector when v_ECT > v_trad; red when v_trad > v_ECT.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    alpha_used = float(merged["alpha"].iloc[0])
    sub = merged.sort_values("diff_ect_minus_trad", ascending=True).reset_index(drop=True)
    n = len(sub)
    x = np.arange(n)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patched_trad = sub["patched_trad"].astype(float).to_numpy()
    patched_ect = sub["patched_ect"].astype(float).to_numpy()
    clean = sub["baseline_logprob_clean"].astype(float).to_numpy()
    corrupt = sub["baseline_logprob_corrupted"].astype(float).to_numpy()
    diffs = sub["diff_ect_minus_trad"].astype(float).to_numpy()

    # Re-center: corrupted baseline at 0; others show delta vs corrupted.
    clean_rel = clean - corrupt
    corrupt_rel = np.zeros_like(corrupt)
    patched_trad_rel = patched_trad - corrupt
    patched_ect_rel = patched_ect - corrupt

    fig, ax = plt.subplots(figsize=(max(14, n * 0.045), 6))
    ax.axhline(0.0, color=CONDITION_COLORS["corrupted"], linewidth=0.8, linestyle=":", alpha=0.6, zorder=1)
    ax.scatter(
        x,
        clean_rel,
        s=6,
        c=CONDITION_COLORS["clean"],
        alpha=0.55,
        label="Clean baseline",
        zorder=2,
    )
    ax.scatter(
        x,
        corrupt_rel,
        s=6,
        c=CONDITION_COLORS["corrupted"],
        alpha=0.55,
        label="Corrupted baseline (0)",
        zorder=2,
    )
    ax.scatter(
        x,
        patched_trad_rel,
        s=8,
        c=METHOD_COLORS["trad"],
        alpha=0.8,
        label=r"Patched $v_{\mathrm{trad}}$",
        zorder=3,
    )
    ax.scatter(
        x,
        patched_ect_rel,
        s=8,
        c=METHOD_COLORS["ect"],
        alpha=0.8,
        label=r"Patched $v_{\mathrm{ECT}}$",
        zorder=3,
    )

    n_ect_wins = 0
    n_trad_wins = 0
    for i in range(n):
        y0, y1 = patched_trad_rel[i], patched_ect_rel[i]
        if diffs[i] > 0:
            color = ECT_WINS_COLOR
            n_ect_wins += 1
        elif diffs[i] < 0:
            color = TRAD_WINS_COLOR
            n_trad_wins += 1
        else:
            color = "#999999"
        ax.plot([x[i], x[i]], [y0, y1], color=color, linewidth=0.9, alpha=0.85, zorder=2)

    ax.set_title(
        rf"Layer {layer_index}, $\alpha={alpha_used:g}$: "
        rf"$v_{{\mathrm{{ECT}}}}$ vs $v_{{\mathrm{{trad}}}}$ "
        f"(sorted by $\Delta$log-prob; {n} pairs)\n"
        rf"$v_{{\mathrm{{ECT}}}}$ wins: {n_ect_wins} (blue)   "
        rf"$v_{{\mathrm{{trad}}}}$ wins: {n_trad_wins} (red)",
        fontsize=11,
    )
    ax.set_xlabel(r"sample index (sorted by $v_{\mathrm{ECT}} - v_{\mathrm{trad}}$ patched log-prob)")
    ax.set_ylabel(r"$\Delta$ log-prob vs corrupted baseline")
    ax.grid(True, axis="y", alpha=0.28)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CONDITION_COLORS["clean"], markersize=6, label="Clean baseline"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CONDITION_COLORS["corrupted"], markersize=6, label="Corrupted baseline (0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=METHOD_COLORS["trad"], markersize=6, label=r"Patched $v_{\mathrm{trad}}$"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=METHOD_COLORS["ect"], markersize=6, label=r"Patched $v_{\mathrm{ECT}}$"),
        Line2D([0], [0], color=ECT_WINS_COLOR, linewidth=2, label=r"$v_{\mathrm{ECT}} > v_{\mathrm{trad}}$ (blue)"),
        Line2D([0], [0], color=TRAD_WINS_COLOR, linewidth=2, label=r"$v_{\mathrm{trad}} > v_{\mathrm{ECT}}$ (red)"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="best", fontsize=9)

    fig.tight_layout()
    dest = out_dir / f"layer{layer_index}_ect_vs_trad_sorted_a{alpha_used:g}.png"
    fig.savefig(dest, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return dest


def plot_layer_samples(
    df: pd.DataFrame,
    *,
    layer_index: int = 12,
    alpha: float | None = 1.0,
    out_dir: Path | str,
) -> list[Path]:
    merged = _merge_trad_ect(df, layer_index=layer_index, alpha=alpha)
    return [
        plot_layer_per_sample_all(merged, layer_index=layer_index, out_dir=out_dir),
        plot_layer_ect_vs_trad_sorted(merged, layer_index=layer_index, out_dir=out_dir),
    ]


def plot_layer_samples_by_alpha(
    df: pd.DataFrame,
    *,
    layer_index: int = 12,
    out_dir: Path | str,
) -> list[Path]:
    alphas = sorted(
        df[(df["layer_index"] == layer_index) & (df["vector_type"] == "trad")]["alpha"].unique()
    )
    if not alphas:
        raise ValueError(f"No trad rows at layer {layer_index}")
    paths: list[Path] = []
    for alpha in alphas:
        paths.extend(plot_layer_samples(df, layer_index=layer_index, alpha=float(alpha), out_dir=out_dir))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-sample layer plots: baselines + v_trad + v_ECT; sorted ECT vs trad diff."
    )
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Single α to plot (default: all alphas in separate figures).",
    )
    args = parser.parse_args()

    csv_path = args.csv or results_csv(args.output_dir)
    if not csv_path.exists():
        raise SystemExit(f"Results CSV not found: {csv_path}")

    df = load_results(csv_path)
    out_dir = figures_dir(args.output_dir) / f"layer{args.layer}_per_sample"

    if args.alpha is not None:
        paths = plot_layer_samples(df, layer_index=args.layer, alpha=args.alpha, out_dir=out_dir)
    else:
        paths = plot_layer_samples_by_alpha(df, layer_index=args.layer, out_dir=out_dir)

    print(f"Loaded {csv_path} ({df['pair_id'].nunique()} pairs)")
    for path in paths:
        print(f"Saved plot → {path}")


if __name__ == "__main__":
    main()
