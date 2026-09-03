"""
Incremental OR middle-layer extraction + colimit metrics.

For each cached (P_A, P_B) row from logit_lens_aim1_AND at layer L:
  1) extract only P_{A∨B} via middle-layer logit lens
  2) merge into the OR distributions JSONL under output/layer_{L}/
  3) score normalized max(P_A, P_B) vs empirical P_{A∨B}
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import (
    APPLY_FINAL_NORM,
    DEFAULT_OUTPUT_DIR,
    DEVICE,
    DTYPE,
    LAYER_INDEX,
    MODEL_ID,
    SAVE_EVERY,
    TOP_K,
    cached_distributions_for_layer,
    distributions_jsonl_for_layer,
    metrics_csv_for_layer,
    metrics_jsonl_for_layer,
)
from .metrics import compute_or_metrics
from .model_io import (
    TopKDistribution,
    extract_topk_logit_lens,
    load_model_and_tokenizer,
    num_transformer_layers,
    resolve_layer_index,
)
from .prompts import build_or_prompt


def _pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["category_id"]),
        str(row["word_a"]).casefold().strip(),
        str(row["word_b"]).casefold().strip(),
        str(row.get("domain_seed", "")),
    )


def _load_done_keys(metrics_jsonl: Path) -> set[tuple[Any, ...]]:
    done: set[tuple[Any, ...]] = set()
    if not metrics_jsonl.exists():
        return done
    with metrics_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            done.add(_pair_key(json.loads(line)))
    return done


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _rewrite_metrics_csv(metrics_jsonl: Path, metrics_csv: Path) -> None:
    if not metrics_jsonl.exists():
        return
    rows = [
        json.loads(line)
        for line in metrics_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
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
            "cached_top_k": row.get("cached_top_k"),
            "cached_layer_index": row.get("cached_layer_index"),
        }
        for k, v in row.get("metrics", {}).items():
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


def _distribution_from_blob(blob: dict[str, Any]) -> TopKDistribution:
    return TopKDistribution(
        token_ids=[int(t) for t in blob["token_ids"]],
        probs=[float(p) for p in blob["probs"]],
    )


def iter_cached_pairs(
    cached_jsonl: Path | str,
    *,
    categories: Iterable[int] | None = None,
    max_pairs: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream cached AND rows that supply P_A and P_B."""
    path = Path(cached_jsonl)
    if not path.exists():
        raise FileNotFoundError(
            f"AND logit-lens cache not found: {path}\n"
            "Run logit_lens_aim1_AND for this layer first."
        )
    wanted = set(categories) if categories is not None else None
    n = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if wanted is not None and int(row["category_id"]) not in wanted:
                continue
            yield row
            n += 1
            if max_pairs is not None and n >= max_pairs:
                break


def run_experiment(
    *,
    cached_distributions_jsonl: Path | str | None = None,
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
    Incremental OR middle-layer extraction loop.

    Only P_{A∨B} is forwarded through the model; P_A and P_B are inherited
    from the sibling AND logit-lens cache at the same layer.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"model_id={model_id}")
        print(f"top_k={top_k} device={device} dtype={dtype_name}")
        print(f"layer_index request={layer_index!r} (None → middle)")

    tokenizer, model, device_t = load_model_and_tokenizer(
        model_id=model_id,
        device=device,
        dtype_name=dtype_name,
    )
    n_layers = num_transformer_layers(model)
    resolved_layer = resolve_layer_index(model, layer_index)

    if cached_distributions_jsonl is None:
        cached_distributions_jsonl = cached_distributions_for_layer(resolved_layer)
    if distributions_jsonl is None:
        distributions_jsonl = distributions_jsonl_for_layer(resolved_layer, output_dir)
    if metrics_jsonl is None:
        metrics_jsonl = metrics_jsonl_for_layer(resolved_layer, output_dir)
    if metrics_csv is None:
        metrics_csv = metrics_csv_for_layer(resolved_layer, output_dir)

    cached_distributions_jsonl = Path(cached_distributions_jsonl)
    distributions_jsonl = Path(distributions_jsonl)
    metrics_jsonl = Path(metrics_jsonl)
    metrics_csv = Path(metrics_csv)

    cached_rows = list(
        iter_cached_pairs(
            cached_distributions_jsonl,
            categories=categories,
            max_pairs=max_pairs,
        )
    )
    done = _load_done_keys(metrics_jsonl) if resume else set()
    todo = [r for r in cached_rows if _pair_key(r) not in done]

    if verbose:
        print(f"loaded on {device_t}, vocab_size={len(tokenizer)}")
        print(f"n_layers={n_layers} → using layer_index={resolved_layer}")
        print(f"apply_final_norm={apply_final_norm}")
        print(
            f"cached={cached_distributions_jsonl} "
            f"(n={len(cached_rows)}, todo={len(todo)}, done={len(done)})"
        )
        print(f"distributions → {distributions_jsonl}")

    if not todo:
        _rewrite_metrics_csv(metrics_jsonl, metrics_csv)
        return {
            "n_total": len(cached_rows),
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
        prompt_or = build_or_prompt(word_a, word_b)

        dist_a = _distribution_from_blob(row["P_A"])
        dist_b = _distribution_from_blob(row["P_B"])
        dist_or = extract_topk_logit_lens(
            tokenizer,
            model,
            prompt_or,
            device=device_t,
            layer_index=resolved_layer,
            top_k=top_k,
            apply_final_norm=apply_final_norm,
        )

        metrics = compute_or_metrics(dist_a, dist_b, dist_or)

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
            "cached_top_k": row.get("top_k"),
            "cached_model_id": row.get("model_id"),
            "cached_layer_index": row.get("layer_index"),
            "prompts": {
                "P_A": row.get("prompts", {}).get("P_A"),
                "P_B": row.get("prompts", {}).get("P_B"),
                "P_A_or_B": prompt_or,
            },
        }

        dist_buf.append(
            {
                **meta,
                "P_A": {"token_ids": dist_a.token_ids, "probs": dist_a.probs},
                "P_B": {"token_ids": dist_b.token_ids, "probs": dist_b.probs},
                "P_A_or_B": {"token_ids": dist_or.token_ids, "probs": dist_or.probs},
            }
        )
        metrics_buf.append({**meta, "metrics": metrics})
        n_new += 1

        if verbose and (i == 1 or i % 10 == 0 or i == len(todo)):
            print(
                f"[{i}/{len(todo)}] L={resolved_layer} cat={meta['category_id']} "
                f"{word_a!r} ∨ {word_b!r} | "
                f"js={metrics['max_norm_js_to_or']:.4f} "
                f"cos={metrics['max_norm_cosine_to_or']:.4f}"
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
        "n_total": len(cached_rows),
        "n_done_before": len(done),
        "n_new": n_new,
        "layer_index": resolved_layer,
        "n_layers": n_layers,
        "distributions_jsonl": str(distributions_jsonl),
        "metrics_csv": str(metrics_csv),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract middle-layer P_{A∨B} and score vs max(P_A, P_B) "
            "using cached AND logit-lens distributions."
        )
    )
    parser.add_argument(
        "--cached-distributions",
        type=Path,
        default=None,
        help="AND logit-lens JSONL (default: sibling layer_{L} cache).",
    )
    parser.add_argument("--distributions-jsonl", type=Path, default=None)
    parser.add_argument("--metrics-jsonl", type=Path, default=None)
    parser.add_argument("--metrics-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--layer-index",
        type=int,
        default=LAYER_INDEX,
        help="Transformer block index (default: middle layer).",
    )
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--dtype", default=DTYPE)
    parser.add_argument(
        "--no-final-norm",
        action="store_true",
        help="Disable final RMSNorm before lm_head.",
    )
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--categories", type=int, nargs="*", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    parser.add_argument("-q", "--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_experiment(
        cached_distributions_jsonl=args.cached_distributions,
        model_id=args.model_id,
        top_k=args.top_k,
        layer_index=args.layer_index,
        apply_final_norm=APPLY_FINAL_NORM and not args.no_final_norm,
        device=args.device,
        dtype_name=args.dtype,
        output_dir=args.output_dir,
        distributions_jsonl=args.distributions_jsonl,
        metrics_jsonl=args.metrics_jsonl,
        metrics_csv=args.metrics_csv,
        max_pairs=args.max_pairs,
        resume=not args.no_resume,
        save_every=args.save_every,
        categories=args.categories,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
