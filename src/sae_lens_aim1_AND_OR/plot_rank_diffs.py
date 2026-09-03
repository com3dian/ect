"""Generate SAE AND/OR rank-diff plots from extracted feature distributions."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import (
    DEFAULT_OUTPUT_DIR,
    LAYER_INDEX,
    TOP_K_FEATURES,
    distributions_jsonl_for_layer,
    figures_rank_diff_dir_for_layer,
    rank_diff_npz_for_layer,
)
from .rank_diff import (
    compute_rank_diff_summary,
    load_rank_diff_summary,
    plot_category_rank_diffs,
    save_rank_diff_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot min / mean / max SAE feature-rank diffs vs V_AND and V_OR "
            "(three panels per category, matching logit / logit-lens style)."
        )
    )
    parser.add_argument("--layer-index", type=int, default=LAYER_INDEX)
    parser.add_argument("--distributions-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--ops",
        nargs="*",
        default=["and", "or"],
        choices=["and", "or"],
        help="Which empirical compositions to plot (default: both).",
    )
    parser.add_argument("--categories", type=int, nargs="*", default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None, help=f"Default: from JSONL / {TOP_K_FEATURES}")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute summary from JSONL even if NPZ exists.",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    layer = args.layer_index
    distributions = args.distributions_jsonl or distributions_jsonl_for_layer(
        layer, args.output_dir
    )
    if not distributions.exists():
        raise SystemExit(
            f"Feature distributions not found: {distributions}\n"
            "Run: python -m sae_lens_aim1_AND_OR.extract_feature_distributions"
        )

    for op in args.ops:
        summary_npz = rank_diff_npz_for_layer(layer, op, args.output_dir)
        out_dir = figures_rank_diff_dir_for_layer(layer, op, args.output_dir)

        if args.recompute or not summary_npz.exists():
            print(f"[{op}] Computing rank-aligned diffs from", distributions)
            summary = compute_rank_diff_summary(
                distributions,
                op=op,
                categories=args.categories,
                max_pairs=args.max_pairs,
                top_k=args.top_k,
                q_low=0.05,
                q_high=0.95,
            )
            save_rank_diff_summary(summary, summary_npz)
            print(f"[{op}] Wrote", summary_npz)
        else:
            print(f"[{op}] Loading cached summary", summary_npz)
            summary = load_rank_diff_summary(summary_npz)

        for cid, blob in sorted(summary.items()):
            print(
                f"[{op}] cat={cid} n={blob['n_pairs']} K={blob['top_k']} | "
                f"{blob['category_name']}"
            )

        paths = plot_category_rank_diffs(
            summary,
            op=op,
            out_dir=out_dir,
            show=args.show,
            log_x=True,
            layer_index=layer,
        )
        for p in paths:
            print(f"[{op}] Saved plot →", p)


if __name__ == "__main__":
    main()
