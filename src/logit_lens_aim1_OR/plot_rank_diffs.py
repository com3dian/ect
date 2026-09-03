"""Generate OR rank-diff plots from middle-layer extracted distributions."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import (
    DEFAULT_OUTPUT_DIR,
    LAYER_INDEX,
    distributions_jsonl_for_layer,
    figures_dir_for_layer,
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
        description="Plot min / mean / max rank diffs vs P_{A∨B} (three panels per category)."
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=LAYER_INDEX,
        help="Layer used to resolve default paths (required if paths omitted).",
    )
    parser.add_argument("--distributions-jsonl", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--summary-npz", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--categories", type=int, nargs="*", default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute summary from JSONL even if NPZ exists.",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.layer_index is None and (
        args.distributions_jsonl is None
        or args.out_dir is None
        or args.summary_npz is None
    ):
        raise SystemExit(
            "Provide --layer-index (or LOGIT_LENS_LAYER) so default paths "
            "can be resolved, or pass all of --distributions-jsonl, "
            "--summary-npz, and --out-dir explicitly."
        )

    layer = args.layer_index
    distributions = args.distributions_jsonl
    summary_npz = args.summary_npz
    out_dir = args.out_dir
    if layer is not None:
        if distributions is None:
            distributions = distributions_jsonl_for_layer(layer, args.output_dir)
        if summary_npz is None:
            summary_npz = rank_diff_npz_for_layer(layer, args.output_dir)
        if out_dir is None:
            out_dir = figures_dir_for_layer(layer, args.output_dir)

    assert distributions is not None and summary_npz is not None and out_dir is not None

    if args.recompute or not summary_npz.exists():
        print("Computing rank-aligned diffs from", distributions)
        summary = compute_rank_diff_summary(
            distributions,
            categories=args.categories,
            max_pairs=args.max_pairs,
            q_low=0.05,
            q_high=0.95,
        )
        save_rank_diff_summary(summary, summary_npz)
        print("Wrote", summary_npz)
    else:
        print("Loading cached summary", summary_npz)
        summary = load_rank_diff_summary(summary_npz)

    for cid, blob in sorted(summary.items()):
        print(f"cat={cid} n={blob['n_pairs']} K={blob['top_k']} | {blob['category_name']}")

    paths = plot_category_rank_diffs(
        summary,
        out_dir=out_dir,
        show=args.show,
        log_x=True,
        layer_index=layer,
    )
    for p in paths:
        print("Saved plot →", p)


if __name__ == "__main__":
    main()
