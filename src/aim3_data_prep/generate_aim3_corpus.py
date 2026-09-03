"""
Generate the Aim-3 tropical geometry corpus via SURF AI Hub (WiLLMa).

Loops over topological types A/B/C. Each API call requests ITEMS_PER_BATCH
(default 50) JSON structures. Always uses forced OpenAI structured outputs
(``response_format=json_schema``, strict=True)—no json_object fallback.
Async client with optional concurrency. Qwen thinking mode is disabled by
default so answers land in ``message.content``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, BadRequestError, OpenAI

from .config import (
    BASE_URL,
    BATCHES_PER_TYPE,
    CONCURRENCY,
    DEFAULT_OUTPUT_JSON,
    DEFAULT_OUTPUT_JSONL,
    DISABLE_THINKING,
    FORCE_STRUCTURED_OUTPUT,
    ITEMS_PER_BATCH,
    MAX_RETRIES,
    MAX_TOKENS,
    MODEL,
    REQUEST_TIMEOUT_S,
    TARGET_ITEMS_PER_TYPE,
    TEMPERATURE,
    resolve_api_key,
)
from .prompts import (
    STRUCTURE_TYPES,
    StructureType,
    build_messages,
    response_format_json_schema,
)

_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")
_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def make_client(api_key: str | None = None, base_url: str = BASE_URL) -> OpenAI:
    """Sync OpenAI-compatible client pointed at SURF AI Hub / WiLLMa."""
    return OpenAI(
        api_key=resolve_api_key(api_key),
        base_url=base_url,
        timeout=REQUEST_TIMEOUT_S,
    )


def make_async_client(
    api_key: str | None = None, base_url: str = BASE_URL
) -> AsyncOpenAI:
    """Async OpenAI-compatible client pointed at SURF AI Hub / WiLLMa."""
    return AsyncOpenAI(
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


def _normalize_items_payload(
    data: Any, structure: StructureType
) -> list[dict[str, str]]:
    """Accept {"items": [...]} / {"examples": [...]} / bare [...]; validate fields."""
    if isinstance(data, dict):
        for key in ("items", "examples", "structures", "data"):
            if key in data:
                data = data[key]
                break
        else:
            raise ValueError(
                f"JSON object missing items key; got keys={list(data.keys())}"
            )

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got {type(data).__name__}")

    cleaned: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        row: dict[str, str] = {}
        ok = True
        for field in structure.required_fields:
            if field == "structure_type":
                row[field] = structure.name
                continue
            value = str(item.get(field, "")).strip()
            if not value:
                ok = False
                break
            row[field] = value
        if ok:
            cleaned.append(row)
    return cleaned


def parse_items_json(raw: str, structure: StructureType) -> list[dict[str, str]]:
    """Parse model output into validated item dicts."""
    text = _strip_code_fences(raw)
    if not text:
        raise ValueError("Empty model response")

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        obj_match = _JSON_OBJECT_RE.search(text)
        arr_match = _JSON_ARRAY_RE.search(text)
        data = None
        if obj_match:
            try:
                data = json.loads(obj_match.group(0))
            except json.JSONDecodeError:
                data = None
        if data is None and arr_match:
            data = json.loads(arr_match.group(0))
        if data is None:
            raise ValueError("No JSON object/array found in model response") from None

    return _normalize_items_payload(data, structure)


def dedup_key(item: dict[str, str]) -> str:
    """Canonical uniqueness key across the full corpus."""
    structure_type = item.get("structure_type", "")
    payload = {
        k: v.casefold().strip()
        for k, v in item.items()
        if k != "structure_type"
    }
    return structure_type + "|" + json.dumps(payload, sort_keys=True, ensure_ascii=False)


def load_existing_json(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return [row for row in data if isinstance(row, dict)]


def load_existing_jsonl(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if isinstance(data, dict):
                rows.append(data)
    return rows


def write_corpus_json(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def append_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _avoid_echo(existing: list[dict[str, str]], structure: StructureType, n: int = 8) -> str:
    """Short sample of existing items to discourage repetition."""
    same = [r for r in existing if r.get("structure_type") == structure.name]
    if not same:
        return ""
    snippets: list[str] = []
    for row in same[-n:]:
        if "context_sequence" in row:
            snippets.append(f"- {row['context_sequence'][:120]}")
        elif "combined_sequence" in row:
            snippets.append(f"- {row['combined_sequence'][:120]}")
        elif "framed_sequence" in row:
            snippets.append(f"- {row['framed_sequence'][:120]}")
    return "\n".join(snippets)


def _extra_body_for_model(model: str) -> dict[str, Any]:
    """Gateway extras: keep structured JSON answers out of Qwen thinking traces."""
    extra: dict[str, Any] = {}
    if DISABLE_THINKING and "qwen" in model.casefold():
        # Common OpenAI-compatible knobs used by vLLM / WiLLMa for Qwen3.
        extra["chat_template_kwargs"] = {"enable_thinking": False}
        extra["enable_thinking"] = False
    return extra


def _message_text(message: Any) -> str:
    """Extract JSON text from a chat message, including structured-output fields."""
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
        joined = "".join(parts).strip()
        if joined:
            return joined

    parsed = getattr(message, "parsed", None)
    if parsed is not None:
        if hasattr(parsed, "model_dump"):
            return json.dumps(parsed.model_dump(), ensure_ascii=False)
        if isinstance(parsed, dict):
            return json.dumps(parsed, ensure_ascii=False)

    refusal = getattr(message, "refusal", None)
    if refusal:
        raise ValueError(f"Model refusal under structured output: {refusal}")

    # Last resort: some gateways put the answer only in reasoning fields.
    for attr in ("reasoning_content", "reasoning"):
        alt = getattr(message, attr, None)
        if isinstance(alt, str) and alt.strip():
            return alt
    return ""


async def _achat_completion(
    client: AsyncOpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    response_format: dict,
):
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
        "response_format": response_format,
    }
    extra = _extra_body_for_model(model)
    if extra:
        kwargs["extra_body"] = extra
    return await client.chat.completions.create(**kwargs)


async def generate_batch(
    client: AsyncOpenAI,
    structure: StructureType,
    *,
    n_items: int = ITEMS_PER_BATCH,
    model: str = MODEL,
    temperature: float = TEMPERATURE,
    avoid_echo: str = "",
) -> list[dict[str, str]]:
    """Request one batch via forced json_schema structured output (no fallback)."""
    if not FORCE_STRUCTURED_OUTPUT:
        raise RuntimeError("FORCE_STRUCTURED_OUTPUT is disabled; refusing to generate.")

    messages = build_messages(structure, n_items=n_items, avoid_echo=avoid_echo)
    response_format = response_format_json_schema(structure)
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await _achat_completion(
                client,
                model=model,
                messages=messages,
                temperature=temperature,
                response_format=response_format,
            )
            choice = response.choices[0]
            finish = getattr(choice, "finish_reason", None)
            content = _message_text(choice.message)
            if finish == "length":
                raise ValueError(
                    "Structured output truncated (finish_reason=length); "
                    "try fewer items_per_batch or higher max_tokens."
                )
            if not content.strip():
                raise ValueError(
                    "Empty model response under forced json_schema "
                    f"(finish_reason={finish!r})."
                )
            items = parse_items_json(content, structure)
            if not items:
                preview = content[:240].replace("\n", " ")
                raise ValueError(
                    "Parsed zero valid items from structured response; "
                    f"preview={preview!r}"
                )
            return items
        except BadRequestError as exc:
            # Do not fall back to json_object — surface schema/gateway errors.
            last_error = RuntimeError(
                f"Forced json_schema structured output rejected by gateway: {exc}"
            )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(1.5 * attempt)
        except Exception as exc:  # noqa: BLE001 — retry transient API/parse failures
            last_error = exc
            if attempt < MAX_RETRIES:
                await asyncio.sleep(1.5 * attempt)

    assert last_error is not None
    raise last_error


async def run_generation_async(
    *,
    api_key: str | None = None,
    base_url: str = BASE_URL,
    model: str = MODEL,
    items_per_batch: int = ITEMS_PER_BATCH,
    batches_per_type: int = BATCHES_PER_TYPE,
    target_per_type: int | None = None,
    type_keys: list[str] | None = None,
    output_json: Path | str = DEFAULT_OUTPUT_JSON,
    output_jsonl: Path | str = DEFAULT_OUTPUT_JSONL,
    resume: bool = True,
    dry_run: bool = False,
    concurrency: int = CONCURRENCY,
    verbose: bool = True,
    skip_failed_batches: bool = True,
) -> list[dict[str, str]]:
    """
    Async loop over topological types A/B/C with batch generation.

    Writes the canonical JSON array to ``output_json`` and appends new rows to
    ``output_jsonl`` for crash-safe resume.
    """
    output_json = Path(output_json)
    output_jsonl = Path(output_jsonl)
    target = target_per_type if target_per_type is not None else TARGET_ITEMS_PER_TYPE

    if resume and output_json.exists():
        corpus = load_existing_json(output_json)
    elif resume and output_jsonl.exists():
        corpus = load_existing_jsonl(output_jsonl)
    else:
        corpus = []

    seen = {dedup_key(row) for row in corpus}
    counts: dict[str, int] = {k: 0 for k in STRUCTURE_TYPES}
    for row in corpus:
        for key, st in STRUCTURE_TYPES.items():
            if row.get("structure_type") == st.name:
                counts[key] += 1
                break

    keys = type_keys or list(STRUCTURE_TYPES.keys())
    for key in keys:
        if key not in STRUCTURE_TYPES:
            raise KeyError(f"Unknown structure type key={key!r}; expected A/B/C")

    client: AsyncOpenAI | None = None if dry_run else make_async_client(
        api_key=api_key, base_url=base_url
    )
    sem = asyncio.Semaphore(max(1, concurrency))
    failed: list[tuple[str, int, str]] = []

    if verbose:
        print(f"base_url={base_url}")
        print(f"model={model}")
        print(f"existing_items={len(corpus)}")
        print(f"target_per_type={target}")
        print(f"items_per_batch={items_per_batch}")
        print(f"batches_per_type={batches_per_type}")
        print(f"concurrency={concurrency}")
        print(
            "structured_output=json_schema strict "
            f"(forced={FORCE_STRUCTURED_OUTPUT}, disable_thinking={DISABLE_THINKING})"
        )
        if dry_run:
            print("DRY RUN — no API calls")

    async def _one_batch(structure: StructureType, batch_idx: int) -> list[dict[str, str]]:
        assert client is not None
        async with sem:
            if verbose:
                print(
                    f"[{structure.key}] batch {batch_idx + 1} "
                    f"(have {counts[structure.key]})"
                )
            return await generate_batch(
                client,
                structure,
                n_items=items_per_batch,
                model=model,
                avoid_echo=_avoid_echo(corpus, structure),
            )

    for key in keys:
        structure = STRUCTURE_TYPES[key]
        if counts[key] >= target:
            if verbose:
                print(
                    f"[{key}] already have {counts[key]} items (≥ target); skipping"
                )
            continue

        for batch_idx in range(batches_per_type):
            if counts[key] >= target:
                break

            if dry_run:
                messages = build_messages(structure, n_items=items_per_batch)
                if verbose:
                    print(f"[{key}] dry-run batch {batch_idx + 1}")
                    print("  system:", messages[0]["content"][:80], "...")
                    print(
                        "  user:",
                        messages[1]["content"][:140].replace("\n", " "),
                        "...",
                    )
                continue

            try:
                raw_items = await _one_batch(structure, batch_idx)
            except Exception as exc:  # noqa: BLE001
                failed.append((key, batch_idx, str(exc)))
                if verbose:
                    print(f"  SKIP after retries: {exc}")
                if not skip_failed_batches:
                    raise
                continue

            new_rows: list[dict[str, str]] = []
            for item in raw_items:
                key_str = dedup_key(item)
                if key_str in seen:
                    continue
                seen.add(key_str)
                new_rows.append(item)

            if new_rows:
                append_jsonl(output_jsonl, new_rows)
                corpus.extend(new_rows)
                counts[key] += len(new_rows)
                write_corpus_json(output_json, corpus)

            if verbose:
                print(
                    f"  +{len(new_rows)} new "
                    f"(parsed {len(raw_items)}; type total {counts[key]})"
                )

            # Small pause between batches to be polite to the gateway.
            await asyncio.sleep(0.4)

    if not dry_run:
        write_corpus_json(output_json, corpus)

    if verbose:
        print("Done. Per-type counts:", dict(sorted(counts.items())))
        if failed:
            print(f"Skipped {len(failed)} failed batch(es):")
            for type_key, batch_idx, err in failed:
                print(f"  - type {type_key} batch {batch_idx}: {err}")
        if dry_run:
            print("Dry run complete (no files written).")
        else:
            print(f"Wrote JSON  → {output_json}")
            print(f"Wrote JSONL → {output_jsonl}")
    return corpus


def run_generation(**kwargs: Any) -> list[dict[str, str]]:
    """Sync wrapper around the async generation loop."""
    return asyncio.run(run_generation_async(**kwargs))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate Aim-3 tropical geometry corpus via SURF / WiLLMa."
    )
    p.add_argument(
        "--types",
        default="A,B,C",
        help="Comma-separated type keys to generate (default: A,B,C).",
    )
    p.add_argument(
        "--items-per-batch",
        type=int,
        default=ITEMS_PER_BATCH,
        help="Structures requested per API call (default: 50).",
    )
    p.add_argument(
        "--batches-per-type",
        type=int,
        default=BATCHES_PER_TYPE,
        help="Max API batches per topological type.",
    )
    p.add_argument(
        "--target-per-type",
        type=int,
        default=TARGET_ITEMS_PER_TYPE,
        help="Stop a type once this many unique items are collected.",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=CONCURRENCY,
        help="Max concurrent API calls (default: 1).",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Canonical JSON array output path.",
    )
    p.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_OUTPUT_JSONL,
        help="Append-only JSONL for resume safety.",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing output and start fresh.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts only; do not call the API.",
    )
    p.add_argument(
        "--model",
        default=MODEL,
        help=f"WiLLMa / SURF model id (default: {MODEL}).",
    )
    p.add_argument(
        "--base-url",
        default=BASE_URL,
        help=f"OpenAI-compatible base URL (default: {BASE_URL}).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    type_keys = [t.strip().upper() for t in args.types.split(",") if t.strip()]
    t0 = time.time()
    run_generation(
        base_url=args.base_url,
        model=args.model,
        items_per_batch=args.items_per_batch,
        batches_per_type=args.batches_per_type,
        target_per_type=args.target_per_type,
        type_keys=type_keys,
        output_json=args.output_json,
        output_jsonl=args.output_jsonl,
        resume=not args.no_resume,
        dry_run=args.dry_run,
        concurrency=args.concurrency,
    )
    if not args.dry_run:
        print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
