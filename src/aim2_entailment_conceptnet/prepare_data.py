"""
Load ConceptNet entailment pairs for Aim-2 patching.

Pairs are prepared by ``aim2_ConceptNet_data_preparation`` (IsA / PartOf).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import DEFAULT_PAIRS_JSONL
from .prompts import build_prompt_bundle


def load_pairs(path: Path | str = DEFAULT_PAIRS_JSONL) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"ConceptNet pairs not found: {path}\n"
            "Run: sbatch src/aim2_ConceptNet_data_preparation/run_prepare_conceptnet.slurm"
        )
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "prompts" not in row:
                row["prompts"] = build_prompt_bundle(row)
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect ConceptNet entailment pairs for Aim-2."
    )
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS_JSONL)
    parser.add_argument("--max-pairs", type=int, default=5)
    args = parser.parse_args()
    rows = load_pairs(args.pairs)
    print(f"Loaded {len(rows)} pairs from {args.pairs}")
    for row in rows[: args.max_pairs]:
        print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
