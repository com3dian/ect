"""Generate OR rank-diff plots from extracted distributions."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import (
    DEFAULT_DISTRIBUTIONS_JSONL,
    DEFAULT_FIGURES_RANK_DIFF_DIR,
    DEFAULT_RANK_DIFF_NPZ,
)
from .rank_diff import (
    compute_rank_diff_summary,
    load_rank_diff_summary,
    plot_category_rank_diffs,
    save_rank_diff_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot min / mean / max rank diffs vs P_{A∨B} (three panels per category)."
    )
    parser.add_argument(
        "--distributions-jsonl",
        type=Path,
        default=DEFAULT_DISTRIBUTIONS_JSONL,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_FIGURES_RANK_DIFF_DIR,
    )
    parser.add_argument(
        "--summary-npz",
        type=Path,
        default=DEFAULT_RANK_DIFF_NPZ,
    )
    parser.add_argument("--categories", type=int, nargs="*", default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute summary from JSONL even if NPZ exists.",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.recompute or not args.summary_npz.exists():
        print("Computing rank-aligned diffs from", args.distributions_jsonl)
        summary = compute_rank_diff_summary(
            args.distributions_jsonl,
            categories=args.categories,
            max_pairs=args.max_pairs,
            q_low=0.05,
            q_high=0.95,
        )
        save_rank_diff_summary(summary, args.summary_npz)
        print("Wrote", args.summary_npz)
    else:
        print("Loading cached summary", args.summary_npz)
        summary = load_rank_diff_summary(args.summary_npz)

    for cid, blob in sorted(summary.items()):
        print(f"cat={cid} n={blob['n_pairs']} K={blob['top_k']} | {blob['category_name']}")

    paths = plot_category_rank_diffs(
        summary,
        out_dir=args.out_dir,
        show=args.show,
        log_x=True,
    )
    for p in paths:
        print("Saved plot →", p)


if __name__ == "__main__":
    main()
