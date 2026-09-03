"""
Generate the 5-dimensional orthogonality spectrum corpus via SURF AI Hub.

Nested loops:
  outer  → categories 1–5
  inner  → domain seeds for that category

Each API call requests PAIRS_PER_BATCH (default 50) JSON concept pairs.
Uses OpenAI structured outputs (json_schema), with json_object fallback.
"""

from __future__ import annotations

import csv
import json
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI
from openai import BadRequestError

from .config import (
    BASE_URL,
    CATEGORY_NAMES,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_JSONL,
    MAX_PAIRS_PER_CATEGORY,
    MAX_RETRIES,
    MAX_TOKENS,
    MODEL,
    PAIRS_PER_BATCH,
    REQUEST_TIMEOUT_S,
    SHUFFLE_SEEDS,
    TARGET_PAIRS_PER_CATEGORY,
    TEMPERATURE,
    resolve_api_key,
)
from .domain_seed import domain_seeds_map
from .prompts import (
    build_messages,
    response_format_json_object,
    response_format_json_schema,
)

_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")
_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")

# Prefer strict schema; fall back to json_object if the gateway rejects it.
_USE_JSON_SCHEMA = True


@dataclass(frozen=True)
class ConceptPair:
    category_id: int
    category_name: str
    domain_seed: str
    word_a: str
    word_b: str

    def key(self) -> tuple[int, str, str]:
        return (
            self.category_id,
            self.word_a.casefold().strip(),
            self.word_b.casefold().strip(),
        )


def make_client(api_key: str | None = None, base_url: str = BASE_URL) -> OpenAI:
    """Create an OpenAI-compatible client pointed at SURF AI Hub / WiLLMa."""
    return OpenAI(
        api_key=resolve_api_key(api_key),
        base_url=base_url,
        timeout=REQUEST_TIMEOUT_S,
    )


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _normalize_pairs_payload(data: Any) -> list[dict[str, str]]:
    """Accept {"pairs": [...]} or a bare [...]; return cleaned pair dicts."""
    if isinstance(data, dict):
        if "pairs" in data:
            data = data["pairs"]
        else:
            raise ValueError(
                f"JSON object missing 'pairs' key; got keys={list(data.keys())}"
            )

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array of pairs, got {type(data).__name__}")

    pairs: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        word_a = str(item.get("word_a", "")).strip()
        word_b = str(item.get("word_b", "")).strip()
        if not word_a or not word_b or word_a.casefold() == word_b.casefold():
            continue
        pairs.append({"word_a": word_a, "word_b": word_b})
    return pairs


def parse_pairs_json(raw: str) -> list[dict[str, str]]:
    """Parse model output into a list of {word_a, word_b} dicts."""
    text = _strip_code_fences(raw)
    if not text:
        raise ValueError("Empty model response")

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        obj_match = _JSON_OBJECT_RE.search(text)
        arr_match = _JSON_ARRAY_RE.search(text)
        if obj_match:
            try:
                data = json.loads(obj_match.group(0))
            except json.JSONDecodeError:
                data = None
        else:
            data = None
        if data is None and arr_match:
            data = json.loads(arr_match.group(0))
        if data is None:
            raise ValueError("No JSON object/array found in model response") from None

    return _normalize_pairs_payload(data)


def _chat_completion(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    response_format: dict,
):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        response_format=response_format,
    )


def generate_batch(
    client: OpenAI,
    category_id: int,
    seed: str,
    *,
    n_pairs: int = PAIRS_PER_BATCH,
    model: str = MODEL,
    temperature: float = TEMPERATURE,
) -> list[dict[str, str]]:
    """Request one batch of concept pairs using structured JSON output."""
    global _USE_JSON_SCHEMA

    messages = build_messages(category_id, seed, n_pairs=n_pairs)
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        response_format = (
            response_format_json_schema()
            if _USE_JSON_SCHEMA
            else response_format_json_object()
        )
        try:
            response = _chat_completion(
                client,
                model=model,
                messages=messages,
                temperature=temperature,
                response_format=response_format,
            )
            content = response.choices[0].message.content or ""
            pairs = parse_pairs_json(content)
            if not pairs:
                raise ValueError("Parsed zero valid pairs from structured response")
            return pairs
        except BadRequestError as exc:
            last_error = exc
            # Gateway may not support json_schema; fall back once to json_object.
            msg = str(exc).lower()
            if _USE_JSON_SCHEMA and (
                "response_format" in msg
                or "json_schema" in msg
                or "invalid" in msg
                or "unsupported" in msg
            ):
                _USE_JSON_SCHEMA = False
                if attempt < MAX_RETRIES:
                    time.sleep(0.5)
                    continue
            if attempt < MAX_RETRIES:
                time.sleep(1.5 * attempt)
        except Exception as exc:  # noqa: BLE001 — retry transient API/parse failures
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(1.5 * attempt)

    assert last_error is not None
    raise last_error


def _seed_order(seeds: list[str], shuffle: bool, rng: random.Random) -> list[str]:
    ordered = list(seeds)
    if shuffle:
        rng.shuffle(ordered)
    return ordered


def _append_jsonl(path: Path, rows: Iterable[ConceptPair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: Iterable[ConceptPair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "category_id",
        "category_name",
        "domain_seed",
        "word_a",
        "word_b",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def load_existing_jsonl(path: Path) -> list[ConceptPair]:
    if not path.exists():
        return []
    rows: list[ConceptPair] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            rows.append(ConceptPair(**data))
    return rows


def run_generation(
    *,
    api_key: str | None = None,
    base_url: str = BASE_URL,
    model: str = MODEL,
    pairs_per_batch: int = PAIRS_PER_BATCH,
    target_per_category: int = TARGET_PAIRS_PER_CATEGORY,
    max_per_category: int = MAX_PAIRS_PER_CATEGORY,
    shuffle_seeds: bool = SHUFFLE_SEEDS,
    categories: Iterable[int] | None = None,
    output_jsonl: Path | str = DEFAULT_OUTPUT_JSONL,
    output_csv: Path | str = DEFAULT_OUTPUT_CSV,
    resume: bool = True,
    dry_run: bool = False,
    seed: int = 42,
    verbose: bool = True,
    skip_failed_batches: bool = True,
) -> list[ConceptPair]:
    """
    Run the nested category × domain-seed generation loop.

    Parameters
    ----------
    dry_run:
        If True, build prompts only and do not call the API.
    resume:
        If True, load existing JSONL and skip already-seen pairs / filled categories.
    skip_failed_batches:
        If True, log and continue after a seed fails all retries (default).
        If False, raise and stop the run.
    """
    output_jsonl = Path(output_jsonl)
    output_csv = Path(output_csv)
    rng = random.Random(seed)

    existing = load_existing_jsonl(output_jsonl) if resume else []
    corpus: list[ConceptPair] = list(existing)
    seen = {p.key() for p in corpus}
    counts: dict[int, int] = {c: 0 for c in range(1, 6)}
    for p in corpus:
        counts[p.category_id] = counts.get(p.category_id, 0) + 1

    category_ids = list(categories) if categories is not None else list(range(1, 6))
    client = None if dry_run else make_client(api_key=api_key, base_url=base_url)
    failed_seeds: list[tuple[int, str, str]] = []

    if verbose:
        print(f"base_url={base_url}")
        print(f"model={model}")
        print(f"existing_pairs={len(corpus)}")
        print(f"target_per_category={target_per_category}")
        print("structured_output=json_schema (fallback: json_object)")
        if dry_run:
            print("DRY RUN — no API calls")

    for category_id in category_ids:
        seeds = domain_seeds_map.get(category_id, [])
        if not seeds:
            raise KeyError(f"No domain seeds for category {category_id}")

        cat_name = CATEGORY_NAMES[category_id]
        if counts.get(category_id, 0) >= target_per_category:
            if verbose:
                print(
                    f"[cat {category_id}] already have "
                    f"{counts[category_id]} pairs (≥ target); skipping"
                )
            continue

        for domain_seed in _seed_order(seeds, shuffle_seeds, rng):
            if counts.get(category_id, 0) >= max_per_category:
                if verbose:
                    print(f"[cat {category_id}] hit max_per_category={max_per_category}")
                break
            if counts.get(category_id, 0) >= target_per_category:
                break

            if verbose:
                print(
                    f"[cat {category_id}] seed={domain_seed!r} "
                    f"(have {counts.get(category_id, 0)})"
                )

            if dry_run:
                messages = build_messages(category_id, domain_seed, n_pairs=pairs_per_batch)
                if verbose:
                    print("  system:", messages[0]["content"][:80], "...")
                    print("  user:", messages[1]["content"][:120].replace("\n", " "), "...")
                continue

            assert client is not None
            try:
                raw_pairs = generate_batch(
                    client,
                    category_id,
                    domain_seed,
                    n_pairs=pairs_per_batch,
                    model=model,
                )
            except Exception as exc:  # noqa: BLE001
                failed_seeds.append((category_id, domain_seed, str(exc)))
                if verbose:
                    print(f"  SKIP after retries: {exc}")
                if not skip_failed_batches:
                    raise
                continue

            new_rows: list[ConceptPair] = []
            for item in raw_pairs:
                row = ConceptPair(
                    category_id=category_id,
                    category_name=cat_name,
                    domain_seed=domain_seed,
                    word_a=item["word_a"],
                    word_b=item["word_b"],
                )
                if row.key() in seen:
                    continue
                seen.add(row.key())
                new_rows.append(row)

            if new_rows:
                _append_jsonl(output_jsonl, new_rows)
                corpus.extend(new_rows)
                counts[category_id] = counts.get(category_id, 0) + len(new_rows)
                # Keep CSV current so crashes still leave a usable table.
                _write_csv(output_csv, corpus)

            if verbose:
                print(
                    f"  +{len(new_rows)} new "
                    f"(parsed {len(raw_pairs)}; cat total {counts[category_id]})"
                )

    _write_csv(output_csv, corpus)
    if verbose:
        print("Done. Per-category counts:", dict(sorted(counts.items())))
        if failed_seeds:
            print(f"Skipped {len(failed_seeds)} failed seed(s):")
            for cat, seed_name, err in failed_seeds:
                print(f"  - cat {cat}: {seed_name!r} → {err}")
        print(f"Wrote JSONL → {output_jsonl}")
        print(f"Wrote CSV   → {output_csv}")
    return corpus


def corpus_summary(rows: list[ConceptPair]) -> dict[str, Any]:
    counts: dict[int, int] = {}
    for row in rows:
        counts[row.category_id] = counts.get(row.category_id, 0) + 1
    return {
        "total": len(rows),
        "per_category": {CATEGORY_NAMES[k]: counts.get(k, 0) for k in range(1, 6)},
        "output_dir": str(DEFAULT_OUTPUT_DIR),
    }


if __name__ == "__main__":
    run_generation()
