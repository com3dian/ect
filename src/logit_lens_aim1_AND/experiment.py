"""
Run the Aim-1 logit-lens experiment over the orthogonality corpus.

For each (word_a, word_b):
  1) build P_A, P_B, P_AB
  2) extract Top-K softmax via logit lens at a chosen middle (or user) layer
  3) compute min / mean / max composition metrics vs p_AB
  4) append sparse distributions + metrics to disk (resumable, per layer)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .config import (
    APPLY_FINAL_NORM,
    DEFAULT_CORPUS_JSONL,
    DEFAULT_OUTPUT_DIR,
    DEVICE,
    DTYPE,
    LAYER_INDEX,
    MODEL_ID,
    SAVE_EVERY,
    TOP_K,
    distributions_jsonl_for_layer,
    metrics_csv_for_layer,
    metrics_jsonl_for_layer,
)
from .metrics import compute_composition_metrics
from .model_io import (
    extract_topk_logit_lens,
    load_model_and_tokenizer,
    num_transformer_layers,
    resolve_layer_index,
)
from .prompts import build_prompts


def load_corpus(path: Path | str = DEFAULT_CORPUS_JSONL) -> list[dict[str, Any]]:
    """Load concept pairs from the vocabulary_gen JSONL."""
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["category_id"]),
        str(row["word_a"]).casefold().strip(),
        str(row["word_b"]).casefold().strip(),
        str(row.get("domain_seed", "")),
    )


def _load_done_keys(metrics_jsonl: Path) -> set[tuple[Any, ...]]:
    """Resume helper: keys already present in the metrics file."""
    done: set[tuple[Any, ...]] = set()
    if not metrics_jsonl.exists():
        return done
    with metrics_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.add(_pair_key(row))
    return done


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _rewrite_metrics_csv(metrics_jsonl: Path, metrics_csv: Path) -> None:
    """Rebuild CSV from the JSONL so both stay aligned."""
    if not metrics_jsonl.exists():
        return
    rows = [json.loads(line) for line in metrics_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return
    fieldnames: list[str] = []
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        flat = {
            "category_id": row.get("category_id"),
            "category_name": row.get("category_name"),
            "domain_seed": row.get("domain_seed"),
            "word_a": row.get("word_a"),
            "word_b": row.get("word_b"),
            "model_id": row.get("model_id"),
            "layer_index": row.get("layer_index"),
            "n_layers": row.get("n_layers"),
            "apply_final_norm": row.get("apply_final_norm"),
            "top_k": row.get("top_k"),
        }
        metrics = row.get("metrics", {})
        for k, v in metrics.items():
            flat[k] = v
        for k in flat:
            if k not in fieldnames:
                fieldnames.append(k)
        flat_rows.append(flat)

    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    with metrics_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def run_experiment(
    *,
    corpus_path: Path | str = DEFAULT_CORPUS_JSONL,
    model_id: str = MODEL_ID,
    top_k: int = TOP_K,
    layer_index: int | None = LAYER_INDEX,
    apply_final_norm: bool = APPLY_FINAL_NORM,
    device: str = DEVICE,
    dtype_name: str = DTYPE,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    distributions_jsonl: Path | str | None = None,
    metrics_jsonl: Path | str | None = None,
    metrics_csv: Path | str | None = None,
    max_pairs: int | None = None,
    resume: bool = True,
    save_every: int = SAVE_EVERY,
    categories: Iterable[int] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Full logit-lens extraction + metrics loop.

    layer_index: transformer block index, or None for the middle layer.
    max_pairs: optional cap for a smoke test before a long GPU job.
    resume: skip pairs already written to metrics_jsonl.
    """
    corpus_path = Path(corpus_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus(corpus_path)
    if categories is not None:
        wanted = set(categories)
        corpus = [r for r in corpus if int(r["category_id"]) in wanted]
    if max_pairs is not None:
        corpus = corpus[: max_pairs]

    if verbose:
        print(f"model_id={model_id}")
        print(f"corpus={corpus_path} (n={len(corpus)})")
        print(f"top_k={top_k} device={device} dtype={dtype_name}")
        print(f"layer_index request={layer_index!r} (None → middle)")

    tokenizer, model, device_t = load_model_and_tokenizer(
        model_id=model_id,
        device=device,
        dtype_name=dtype_name,
    )
    n_layers = num_transformer_layers(model)
    resolved_layer = resolve_layer_index(model, layer_index)

    if distributions_jsonl is None:
        distributions_jsonl = distributions_jsonl_for_layer(resolved_layer, output_dir)
    if metrics_jsonl is None:
        metrics_jsonl = metrics_jsonl_for_layer(resolved_layer, output_dir)
    if metrics_csv is None:
        metrics_csv = metrics_csv_for_layer(resolved_layer, output_dir)
    distributions_jsonl = Path(distributions_jsonl)
    metrics_jsonl = Path(metrics_jsonl)
    metrics_csv = Path(metrics_csv)

    done = _load_done_keys(metrics_jsonl) if resume else set()
    todo = [r for r in corpus if _pair_key(r) not in done]

    if verbose:
        print(f"loaded on {device_t}, vocab_size={len(tokenizer)}")
        print(f"n_layers={n_layers} → using layer_index={resolved_layer}")
        print(f"apply_final_norm={apply_final_norm}")
        print(f"distributions → {distributions_jsonl}")
        print(f"todo={len(todo)} done={len(done)}")

    if not todo:
        _rewrite_metrics_csv(metrics_jsonl, metrics_csv)
        return {
            "n_total": len(corpus),
            "n_done": len(done),
            "n_new": 0,
            "layer_index": resolved_layer,
            "n_layers": n_layers,
        }

    dist_buf: list[dict[str, Any]] = []
    metrics_buf: list[dict[str, Any]] = []
    n_new = 0

    for i, row in enumerate(todo, start=1):
        word_a = str(row["word_a"])
        word_b = str(row["word_b"])
        prompts = build_prompts(word_a, word_b)

        dist_a = extract_topk_logit_lens(
            tokenizer,
            model,
            prompts["P_A"],
            device=device_t,
            layer_index=resolved_layer,
            top_k=top_k,
            apply_final_norm=apply_final_norm,
        )
        dist_b = extract_topk_logit_lens(
            tokenizer,
            model,
            prompts["P_B"],
            device=device_t,
            layer_index=resolved_layer,
            top_k=top_k,
            apply_final_norm=apply_final_norm,
        )
        dist_ab = extract_topk_logit_lens(
            tokenizer,
            model,
            prompts["P_AB"],
            device=device_t,
            layer_index=resolved_layer,
            top_k=top_k,
            apply_final_norm=apply_final_norm,
        )

        metrics = compute_composition_metrics(dist_a, dist_b, dist_ab)

        meta = {
            "category_id": row.get("category_id"),
            "category_name": row.get("category_name"),
            "domain_seed": row.get("domain_seed"),
            "word_a": word_a,
            "word_b": word_b,
            "model_id": model_id,
            "layer_index": resolved_layer,
            "n_layers": n_layers,
            "apply_final_norm": apply_final_norm,
            "top_k": top_k,
            "prompts": prompts,
        }

        dist_buf.append(
            {
                **meta,
                "P_A": {"token_ids": dist_a.token_ids, "probs": dist_a.probs},
                "P_B": {"token_ids": dist_b.token_ids, "probs": dist_b.probs},
                "P_AB": {"token_ids": dist_ab.token_ids, "probs": dist_ab.probs},
            }
        )
        metrics_buf.append({**meta, "metrics": metrics})
        n_new += 1

        if verbose and (i == 1 or i % 10 == 0 or i == len(todo)):
            print(
                f"[{i}/{len(todo)}] L={resolved_layer} cat={meta['category_id']} "
                f"{word_a!r} ∧ {word_b!r} | "
                f"min_cos={metrics['min_cosine_to_ab']:.4f} "
                f"mean_cos={metrics['mean_cosine_to_ab']:.4f} "
                f"max_cos={metrics['max_cosine_to_ab']:.4f}"
            )

        if len(metrics_buf) >= save_every:
            _append_jsonl(distributions_jsonl, dist_buf)
            _append_jsonl(metrics_jsonl, metrics_buf)
            dist_buf.clear()
            metrics_buf.clear()
            _rewrite_metrics_csv(metrics_jsonl, metrics_csv)

    if metrics_buf:
        _append_jsonl(distributions_jsonl, dist_buf)
        _append_jsonl(metrics_jsonl, metrics_buf)
        _rewrite_metrics_csv(metrics_jsonl, metrics_csv)

    if verbose:
        print(f"Wrote distributions → {distributions_jsonl}")
        print(f"Wrote metrics JSONL → {metrics_jsonl}")
        print(f"Wrote metrics CSV   → {metrics_csv}")

    return {
        "n_total": len(corpus),
        "n_done_before": len(done),
        "n_new": n_new,
        "layer_index": resolved_layer,
        "n_layers": n_layers,
        "distributions_jsonl": str(distributions_jsonl),
        "metrics_csv": str(metrics_csv),
    }


if __name__ == "__main__":
    run_experiment()
