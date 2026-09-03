"""
Extract ConceptNet entailment pairs for Aim-2 activation patching.

Uses strict entailment edges only: IsA and PartOf.
Output JSONL matches ``aim2_entailment`` / SNLI pair schema.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from .config import (
    CONCEPTNET_ASSERTIONS_URL,
    CORRUPTED_PREMISE,
    DEFAULT_ASSERTIONS_GZ,
    DEFAULT_COUNTS_JSON,
    DEFAULT_DATA_DIR,
    DEFAULT_MAX_PAIRS,
    DEFAULT_MIN_WEIGHT,
    DEFAULT_OUTPUT_JSONL,
    ENTAILMENT_RELATIONS,
)

_EN_URI_RE = re.compile(r"^/c/en/([^/]+)")
_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z0-9' -]*$")


def label_from_uri(uri: str) -> str | None:
    """Map `/c/en/hot_dog` → `hot dog`; return None for non-English nodes."""
    match = _EN_URI_RE.match(uri.strip())
    if not match:
        return None
    raw = match.group(1).replace("_", " ").strip()
    if not raw or len(raw) < 2 or len(raw) > 48:
        return None
    if not _WORD_RE.match(raw):
        return None
    return raw


def _with_article(noun: str) -> str:
    word = noun.strip()
    if not word:
        return word
    article = "an" if word[0].lower() in "aeiou" else "a"
    return f"{article} {word}"


def _edge_name(relation: str) -> str:
    return relation.rsplit("/", 1)[-1]


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


def entailment_templates(
    *,
    relation: str,
    start_label: str,
    end_label: str,
) -> tuple[str, str] | None:
    """
    Map ConceptNet edges to (premise, hypothesis) sentence pairs.

    IsA:   [dog] --IsA--> [animal]  →  "This is a dog." ⊢ "This is an animal."
    PartOf:[wheel] --PartOf--> [car] → "This is a car." ⊢ "This has a wheel."
    """
    rel = _edge_name(relation)
    if rel == "IsA":
        premise = f"This is {_with_article(start_label)}."
        hypothesis = f"This is {_with_article(end_label)}."
        return premise, hypothesis
    if rel == "PartOf":
        part, whole = start_label, end_label
        premise = f"This is {_with_article(whole)}."
        hypothesis = f"This has {_with_article(part)}."
        return premise, hypothesis
    return None


def _parse_weight(metadata: str) -> float:
    metadata = (metadata or "").strip()
    if not metadata:
        return 1.0
    try:
        payload = json.loads(metadata)
        return float(payload.get("weight", 1.0))
    except json.JSONDecodeError:
        return 1.0


def download_assertions(dest: Path, url: str = CONCEPTNET_ASSERTIONS_URL) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Using cached assertions: {dest}")
        return dest
    print(f"Downloading ConceptNet assertions → {dest}")
    tmp = dest.with_suffix(dest.suffix + ".partial")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310
    tmp.replace(dest)
    print(f"Download complete ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def iter_assertions(path: Path, *, min_weight: float) -> Iterable[tuple[str, str, str, float]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 5:
                continue
            _uri, relation, start, end, metadata = row[:5]
            if relation not in ENTAILMENT_RELATIONS:
                continue
            weight = _parse_weight(metadata)
            if weight < min_weight:
                continue
            yield relation, start, end, weight


def collect_entailment_rows(
    assertions_path: Path,
    *,
    max_pairs: int,
    min_weight: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    counts = {"IsA": 0, "PartOf": 0, "skipped": 0}

    for relation, start, end, weight in iter_assertions(assertions_path, min_weight=min_weight):
        if max_pairs > 0 and len(rows) >= max_pairs:
            break

        start_label = label_from_uri(start)
        end_label = label_from_uri(end)
        if not start_label or not end_label:
            counts["skipped"] += 1
            continue
        if start_label.casefold() == end_label.casefold():
            counts["skipped"] += 1
            continue

        templates = entailment_templates(
            relation=relation,
            start_label=start_label,
            end_label=end_label,
        )
        if templates is None:
            counts["skipped"] += 1
            continue
        premise, hypothesis = templates
        dedup_key = (premise.casefold(), hypothesis.casefold())
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        edge = _edge_name(relation)
        pair_id = f"conceptnet_{edge.lower()}_{len(rows)}"
        row: dict[str, Any] = {
            "pair_id": pair_id,
            "premise": premise,
            "hypothesis": hypothesis,
            "target_word": _target_word_from_hypothesis(hypothesis),
            "label": "entailment",
            "source": "conceptnet",
            "conceptnet_edge": edge,
            "conceptnet_start": start_label,
            "conceptnet_end": end_label,
            "conceptnet_weight": weight,
            "prompts": _prompt_bundle(premise, hypothesis),
        }
        rows.append(row)
        counts[edge] = counts.get(edge, 0) + 1

    summary = {
        "assertions_path": str(assertions_path),
        "max_pairs": max_pairs,
        "min_weight": min_weight,
        "n_written": len(rows),
        "edges": counts,
    }
    return rows, summary


def prepare_conceptnet_entailment(
    *,
    assertions_path: Path | str,
    out_path: Path | str = DEFAULT_OUTPUT_JSONL,
    counts_path: Path | str = DEFAULT_COUNTS_JSON,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    min_weight: float = DEFAULT_MIN_WEIGHT,
) -> list[dict[str, Any]]:
    assertions_path = Path(assertions_path)
    out_path = Path(out_path)
    counts_path = Path(counts_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows, summary = collect_entailment_rows(
        assertions_path,
        max_pairs=max_pairs,
        min_weight=min_weight,
    )
    summary["out_path"] = str(out_path)

    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"ConceptNet entailment: wrote {len(rows)} pairs → {out_path} "
        f"(IsA={summary['edges'].get('IsA', 0)}, "
        f"PartOf={summary['edges'].get('PartOf', 0)})"
    )
    print(f"Counts → {counts_path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract ConceptNet IsA/PartOf entailment pairs for Aim-2."
    )
    parser.add_argument(
        "--assertions",
        type=Path,
        default=DEFAULT_ASSERTIONS_GZ,
        help="Local ConceptNet assertions .csv.gz (downloaded if missing).",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--counts", type=Path, default=DEFAULT_COUNTS_JSON)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=DEFAULT_MAX_PAIRS,
        help="Maximum entailment pairs to keep (0 = no cap).",
    )
    parser.add_argument("--min-weight", type=float, default=DEFAULT_MIN_WEIGHT)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not download assertions; fail if --assertions is missing.",
    )
    args = parser.parse_args()

    assertions_path = args.assertions
    if not assertions_path.exists():
        if args.skip_download:
            raise SystemExit(f"Assertions file not found: {assertions_path}")
        download_assertions(assertions_path)

    prepare_conceptnet_entailment(
        assertions_path=assertions_path,
        out_path=args.out,
        counts_path=args.counts,
        max_pairs=args.max_pairs,
        min_weight=args.min_weight,
    )


if __name__ == "__main__":
    main()
