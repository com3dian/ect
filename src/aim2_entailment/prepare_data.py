"""
Prepare a subset of NLI entailment pairs for Aim-2 patching.

Downloads SNLI (validation, entailment label) via HuggingFace `datasets` when
available; falls back to a small bundled-style seed set otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import DEFAULT_DATA_DIR, DEFAULT_PAIRS_JSONL
from .prompts import build_prompt_bundle


SEED_PAIRS: list[dict[str, Any]] = [
    {
        "pair_id": "seed_001",
        "premise": "A man is playing a guitar on stage.",
        "hypothesis": "Someone is performing music.",
        "target_word": "music",
        "label": "entailment",
        "source": "seed",
    },
    {
        "pair_id": "seed_002",
        "premise": "A poodle runs through a grassy park.",
        "hypothesis": "A dog is outdoors.",
        "target_word": "dog",
        "label": "entailment",
        "source": "seed",
    },
    {
        "pair_id": "seed_003",
        "premise": "Two children are building a sandcastle at the beach.",
        "hypothesis": "Kids are playing outside.",
        "target_word": "Kids",
        "label": "entailment",
        "source": "seed",
    },
    {
        "pair_id": "seed_004",
        "premise": "A chef is slicing vegetables in a kitchen.",
        "hypothesis": "A person is cooking food.",
        "target_word": "cooking",
        "label": "entailment",
        "source": "seed",
    },
    {
        "pair_id": "seed_005",
        "premise": "A student reads a textbook in the library.",
        "hypothesis": "Someone is studying.",
        "target_word": "studying",
        "label": "entailment",
        "source": "seed",
    },
]


def _load_snli_entailment(max_pairs: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("snli", split="validation")
    rows: list[dict[str, Any]] = []
    for i, ex in enumerate(ds):
        label = ex.get("label")
        if label not in (0, "entailment"):
            continue
        premise = str(ex.get("premise", "")).strip()
        hypothesis = str(ex.get("hypothesis", "")).strip()
        if not premise or not hypothesis or premise == "-":
            continue
        rows.append(
            {
                "pair_id": f"snli_{i}",
                "premise": premise,
                "hypothesis": hypothesis,
                "label": "entailment",
                "source": "snli",
            }
        )
        if len(rows) >= max_pairs:
            break
    return rows


def prepare_pairs(
    *,
    out_path: Path | str | None = None,
    max_pairs: int = 50,
    use_snli: bool = True,
) -> list[dict[str, Any]]:
    out_path = Path(out_path) if out_path is not None else DEFAULT_DATA_DIR / "nli_pairs.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    if use_snli:
        try:
            rows = _load_snli_entailment(max_pairs)
        except Exception as exc:
            print(f"SNLI download failed ({exc}); using seed pairs.")

    if not rows:
        rows = SEED_PAIRS[: max_pairs]

    for row in rows:
        row["prompts"] = build_prompt_bundle(row)

    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} pairs → {out_path}")
    return rows


def load_pairs(path: Path | str = DEFAULT_PAIRS_JSONL) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NLI entailment pairs for Aim-2.")
    parser.add_argument("--out", type=Path, default=DEFAULT_DATA_DIR / "nli_pairs.jsonl")
    parser.add_argument("--max-pairs", type=int, default=50)
    parser.add_argument("--no-snli", action="store_true", help="Use bundled seed pairs only.")
    args = parser.parse_args()
    prepare_pairs(
        out_path=args.out,
        max_pairs=args.max_pairs,
        use_snli=not args.no_snli,
    )


if __name__ == "__main__":
    main()
