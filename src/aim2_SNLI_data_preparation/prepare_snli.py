"""
Download SNLI and keep entailment-labeled pairs for Aim-2.

Uses the namespaced HuggingFace id ``stanfordnlp/snli`` (the un-namespaced
``snli`` id fails on current ``datasets`` versions). Walks validation, then
test, then train. If the entailment count exceeds 1000, keeps the first 1000;
otherwise writes the full entailment subset.

Output schema matches ``aim2_entailment`` ``nli_pairs.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PACKAGE_DIR / "data"
DEFAULT_JSONL = DEFAULT_DATA_DIR / "snli_entailment.jsonl"
DEFAULT_COUNTS = DEFAULT_DATA_DIR / "snli_entailment_counts.json"

# Cap requested for Aim-2. Override with --max-pairs.
DEFAULT_MAX_PAIRS = 1000
HF_DATASET_IDS = ("stanfordnlp/snli", "snli")
DEFAULT_SPLITS = ("validation", "test", "train")
CORRUPTED_PREMISE = "It is the case that:"


def _is_entailment(example: dict[str, Any]) -> bool:
    gold = str(example.get("gold_label") or "").strip().lower()
    if gold == "entailment":
        return True
    if gold in {"neutral", "contradiction", "-"}:
        return False
    label = example.get("label")
    return label in (0, "entailment")


def _clean_sentence(text: Any) -> str:
    return str(text or "").strip()


def _is_usable(premise: str, hypothesis: str) -> bool:
    if not premise or not hypothesis:
        return False
    if premise == "-" or hypothesis == "-":
        return False
    return True


def _target_word_from_hypothesis(hypothesis: str) -> str:
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", hypothesis or "")
    return words[-1] if words else ""


def _prompt_bundle(premise: str, hypothesis: str) -> dict[str, str]:
    hyp = hypothesis if hypothesis.endswith(".") else f"{hypothesis}."
    return {
        "premise": premise,
        "corrupted": CORRUPTED_PREMISE,
        "hypothesis": hyp,
    }


def _load_snli() -> tuple[Any, str]:
    from datasets import load_dataset

    last_error: Exception | None = None
    for dataset_id in HF_DATASET_IDS:
        try:
            ds = load_dataset(dataset_id)
            return ds, dataset_id
        except Exception as exc:
            last_error = exc
            print(f"Failed to load {dataset_id!r}: {exc}")
    raise RuntimeError(
        "Could not download SNLI. Tried: " + ", ".join(HF_DATASET_IDS)
    ) from last_error


def _iter_split(
    dataset: Any,
    split: str,
) -> Iterable[tuple[int, dict[str, Any]]]:
    if split not in dataset:
        print(f"Split {split!r} not in dataset; skipping.")
        return
    for i, example in enumerate(dataset[split]):
        yield i, example


def collect_entailment_rows(
    *,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    splits: Iterable[str] = DEFAULT_SPLITS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset, dataset_id = _load_snli()
    split_counts: dict[str, int] = {}
    kept: list[dict[str, Any]] = []
    total_entailment = 0

    for split in splits:
        n_split = 0
        for i, example in _iter_split(dataset, split):
            if not _is_entailment(example):
                continue
            premise = _clean_sentence(example.get("premise"))
            hypothesis = _clean_sentence(example.get("hypothesis"))
            if not _is_usable(premise, hypothesis):
                continue
            n_split += 1
            total_entailment += 1
            if max_pairs > 0 and len(kept) >= max_pairs:
                continue
            kept.append(
                {
                    "pair_id": f"snli_{split}_{i}",
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "target_word": _target_word_from_hypothesis(hypothesis),
                    "label": "entailment",
                    "source": "snli",
                    "split": split,
                    "snli_index": i,
                    "prompts": _prompt_bundle(premise, hypothesis),
                }
            )
        split_counts[split] = n_split

    summary = {
        "dataset_id": dataset_id,
        "splits": list(splits),
        "entailment_per_split": split_counts,
        "entailment_total": total_entailment,
        "max_pairs": max_pairs,
        "n_written": len(kept),
        "capped": bool(max_pairs > 0 and total_entailment > max_pairs),
    }
    return kept, summary


def prepare_snli_entailment(
    *,
    out_path: Path | str = DEFAULT_JSONL,
    counts_path: Path | str = DEFAULT_COUNTS,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    splits: Iterable[str] = DEFAULT_SPLITS,
) -> list[dict[str, Any]]:
    out_path = Path(out_path)
    counts_path = Path(counts_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows, summary = collect_entailment_rows(max_pairs=max_pairs, splits=splits)
    summary["out_path"] = str(out_path)

    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"SNLI entailment total={summary['entailment_total']} "
        f"per_split={summary['entailment_per_split']}; "
        f"wrote {len(rows)} pairs → {out_path}"
        + (" (capped at first 1000)" if summary["capped"] else "")
    )
    print(f"Counts → {counts_path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download SNLI entailment pairs (cap at 1000)."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--counts", type=Path, default=DEFAULT_COUNTS)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=DEFAULT_MAX_PAIRS,
        help="Keep at most this many entailment pairs (0 = no cap).",
    )
    parser.add_argument(
        "--splits",
        default="validation,test,train",
        help="Comma-separated SNLI splits, in walk order.",
    )
    args = parser.parse_args()
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    prepare_snli_entailment(
        out_path=args.out,
        counts_path=args.counts,
        max_pairs=args.max_pairs,
        splits=splits,
    )


if __name__ == "__main__":
    main()
