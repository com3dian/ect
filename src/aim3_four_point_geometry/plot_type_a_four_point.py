"""Type A plot: four-point slack natural vs shuffled controls."""

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
    summary_json,
)

COLORS = {
    "natural": "#1f4e5f",
    "shuffled_middles": "#8a4b2b",
    "reversed": "#5c4d7a",
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


def _fp_slacks(rows: list[dict[str, Any]], variant: str) -> np.ndarray:
    vals: list[float] = []
    for row in rows:
        pack = (row.get("variants") or {}).get(variant)
        if not pack:
            continue
        fp = pack.get("four_point") or {}
        v = fp.get("fp_slack")
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    return np.asarray(vals, dtype=float)


def plot_type_a_four_point(
    rows: list[dict[str, Any]],
    *,
    out_path: Path,
    layer_index: int,
    model_id: str = "google/gemma-2-2b",
    summary_path: Path | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    variants = ["natural", "shuffled_middles", "reversed"]
    present = [v for v in variants if _fp_slacks(rows, v).size > 0]
    if not present:
        raise SystemExit("No Type A four-point slack data found.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: box/violin by variant
    ax = axes[0]
    data = [_fp_slacks(rows, v) for v in present]
    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(COLORS.get(present[i], "#666666"))
        body.set_alpha(0.75)
    ax.set_xticks(range(1, len(present) + 1))
    ax.set_xticklabels([v.replace("_", "\n") for v in present], fontsize=9)
    ax.set_ylabel("fp_slack (lower = more tree-like)")
    ax.set_title("Four-point slack by ordering")
    ax.grid(axis="y", alpha=0.3)

    # Panel B: paired natural vs shuffled scatter
    ax = axes[1]
    nat = _fp_slacks(rows, "natural")
    shuf = _fp_slacks(rows, "shuffled_middles")
    n = min(nat.size, shuf.size)
    if n:
        nat, shuf = nat[:n], shuf[:n]
        ax.scatter(nat, shuf, alpha=0.35, s=18, c="#4a5568", edgecolors="none")
        lo = float(min(nat.min(), shuf.min()))
        hi = float(max(nat.max(), shuf.max()))
        pad = 0.05 * max(hi - lo, 1e-3)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="#999", lw=1)
        frac = float(np.mean(nat < shuf))
        title = f"Paired natural vs shuffled (n={n})\n% natural lower: {100 * frac:.1f}%"
        if summary_path and summary_path.exists():
            summ = json.loads(summary_path.read_text(encoding="utf-8"))
            paired = (
                summ.get("by_type", {})
                .get("A", {})
                .get("paired_tests", {})
                .get("natural_vs_shuffled_middles", {})
            )
            wp = paired.get("wilcoxon", {}).get("pvalue")
            bp = paired.get("binomial_sign", {}).get("pvalue")
            if wp is not None and bp is not None:
                title += f"\nWilcoxon p={wp:.2e}  binomial p={bp:.2e}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("fp_slack natural")
        ax.set_ylabel("fp_slack shuffled_middles")
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Need natural + shuffled_middles", ha="center", va="center")
        ax.set_axis_off()

    slug = model_id.split("/")[-1]
    fig.suptitle(
        f"Type A four-point geometry — {slug} layer {layer_index}",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot Type A four-point slack.")
    p.add_argument("--layer-index", type=int, default=LAYER_INDEX)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--model-id", default="google/gemma-2-2b")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    layer = int(args.layer_index)
    jsonl = distances_jsonl(layer, args.output_dir)
    rows = load_type_a(jsonl)
    if not rows:
        raise SystemExit(f"No Type A rows in {jsonl}")
    out = figures_dir_for_layer(layer, args.output_dir) / "type_a_four_point_slack.png"
    plot_type_a_four_point(
        rows,
        out_path=out,
        layer_index=layer,
        model_id=args.model_id,
        summary_path=summary_json(layer, args.output_dir),
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
