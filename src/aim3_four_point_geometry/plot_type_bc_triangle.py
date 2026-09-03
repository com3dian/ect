"""Type B/C plot: triangle slack histograms."""

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


def load_by_type(path: Path, type_key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type_key") == type_key:
                rows.append(row)
    return rows


def plot_type_bc_triangle(
    b_rows: list[dict[str, Any]],
    c_rows: list[dict[str, Any]],
    *,
    out_path: Path,
    layer_index: int,
    model_id: str = "google/gemma-2-2b",
    summary_path: Path | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, rows, label, color in (
        (axes[0], b_rows, "Type B (narrative disruption)", "#2a6f6f"),
        (axes[1], c_rows, "Type C (adversarial framing)", "#8a4b2b"),
    ):
        slacks = np.array([float(r["triangle_slack"]) for r in rows], dtype=float)
        slacks = slacks[np.isfinite(slacks)]
        if slacks.size == 0:
            ax.text(0.5, 0.5, f"No {label} data", ha="center", va="center")
            continue
        viol = float(np.mean(slacks > 1e-9))
        ax.hist(slacks, bins=40, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(0.0, color="#333", lw=1.2, ls="--")
        ax.set_xlabel("triangle slack (≤0 = TI holds)")
        ax.set_ylabel("count")
        ax.set_title(f"{label}\nviolation rate = {100 * viol:.1f}% (n={slacks.size})")
        ax.grid(axis="y", alpha=0.3)

    slug = model_id.split("/")[-1]
    fig.suptitle(
        f"Triangle slack — {slug} layer {layer_index}",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot Type B/C triangle slack.")
    p.add_argument("--layer-index", type=int, default=LAYER_INDEX)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--model-id", default="google/gemma-2-2b")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    layer = int(args.layer_index)
    jsonl = distances_jsonl(layer, args.output_dir)
    b_rows = load_by_type(jsonl, "B")
    c_rows = load_by_type(jsonl, "C")
    out = figures_dir_for_layer(layer, args.output_dir) / "type_bc_triangle_slack.png"
    plot_type_bc_triangle(
        b_rows,
        c_rows,
        out_path=out,
        layer_index=layer,
        model_id=args.model_id,
        summary_path=summary_json(layer, args.output_dir),
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
