"""
Run Aim-1 SAE-lens AND/OR composition over the orthogonality corpus.

For each (word_a, word_b):
  1) build P_A, P_B, P_AND, P_OR (same typicality templates as logit packages)
  2) extract SAE feature activations at residual layer L (final token)
  3) L1-normalize; score TVD vs min/mean/max/prod + Jaccard(A,B)
  4) append metrics to disk (resumable) and rewrite category summary
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .config import (
    DEFAULT_CORPUS_JSONL,
    DEFAULT_OUTPUT_DIR,
    DEVICE,
    DTYPE,
    JACCARD_THRESHOLD,
    LAYER_INDEX,
    MODEL_ID,
    SAE_ID,
    SAE_RELEASE,
    SAVE_EVERY,
    category_summary_csv_for_layer,
    metrics_csv_for_layer,
    metrics_jsonl_for_layer,
    sae_id_for_layer,
)
from .metrics import compute_composition_metrics
from .model_io import (
    extract_sae_features,
    load_model_and_tokenizer,
    load_sae,
    num_transformer_layers,
    resolve_layer_index,
)
from .prompts import build_prompts


def load_corpus(path: Path | str = DEFAULT_CORPUS_JSONL) -> list[dict[str, Any]]:
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
            "sae_release": row.get("sae_release"),
            "sae_id": row.get("sae_id"),
            "layer_index": row.get("layer_index"),
            "n_layers": row.get("n_layers"),
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


def _rewrite_category_summary(metrics_jsonl: Path, summary_csv: Path) -> None:
    if not metrics_jsonl.exists():
        return
    buckets: dict[int, dict[str, Any]] = {}
    metric_sums: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    metric_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    with metrics_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cid = int(row["category_id"])
            if cid not in buckets:
                buckets[cid] = {
                    "category_id": cid,
                    "category_name": row.get("category_name"),
                    "n_pairs": 0,
                }
            buckets[cid]["n_pairs"] += 1
            for k, v in row.get("metrics", {}).items():
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                metric_sums[cid][k] += fv
                metric_counts[cid][k] += 1

    if not buckets:
        return

    metric_keys = sorted({k for sums in metric_sums.values() for k in sums})
    fieldnames = ["category_id", "category_name", "n_pairs"] + [f"mean_{k}" for k in metric_keys]
    rows_out: list[dict[str, Any]] = []
    for cid in sorted(buckets):
        flat = dict(buckets[cid])
        for k in metric_keys:
            n = metric_counts[cid][k]
            flat[f"mean_{k}"] = (metric_sums[cid][k] / n) if n else None
        rows_out.append(flat)

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)


def run_experiment(
    *,
    corpus_path: Path | str = DEFAULT_CORPUS_JSONL,
    model_id: str = MODEL_ID,
    sae_release: str = SAE_RELEASE,
    sae_id: str | None = None,
    layer_index: int = LAYER_INDEX,
    device: str = DEVICE,
    dtype_name: str = DTYPE,
    jaccard_threshold: float = JACCARD_THRESHOLD,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    metrics_jsonl: Path | str | None = None,
    metrics_csv: Path | str | None = None,
    category_summary_csv: Path | str | None = None,
    max_pairs: int | None = None,
    resume: bool = True,
    save_every: int = SAVE_EVERY,
    categories: Iterable[int] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    corpus_path = Path(corpus_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if sae_id is None:
        sae_id = sae_id_for_layer(layer_index)

    corpus = load_corpus(corpus_path)
    if categories is not None:
        wanted = set(categories)
        corpus = [r for r in corpus if int(r["category_id"]) in wanted]
    if max_pairs is not None:
        corpus = corpus[: max_pairs]

    if verbose:
        print(f"model_id={model_id}")
        print(f"sae_release={sae_release} sae_id={sae_id}")
        print(f"corpus={corpus_path} (n={len(corpus)})")
        print(f"device={device} dtype={dtype_name} layer={layer_index}")

    tokenizer, model, device_t = load_model_and_tokenizer(
        model_id=model_id,
        device=device,
        dtype_name=dtype_name,
    )
    n_layers = num_transformer_layers(model)
    resolved_layer = resolve_layer_index(model, layer_index)

    if metrics_jsonl is None:
        metrics_jsonl = metrics_jsonl_for_layer(resolved_layer, output_dir)
    if metrics_csv is None:
        metrics_csv = metrics_csv_for_layer(resolved_layer, output_dir)
    if category_summary_csv is None:
        category_summary_csv = category_summary_csv_for_layer(resolved_layer, output_dir)
    metrics_jsonl = Path(metrics_jsonl)
    metrics_csv = Path(metrics_csv)
    category_summary_csv = Path(category_summary_csv)

    sae = load_sae(release=sae_release, sae_id=sae_id, device=device_t)

    done = _load_done_keys(metrics_jsonl) if resume else set()
    todo = [r for r in corpus if _pair_key(r) not in done]

    if verbose:
        print(f"loaded on {device_t}, vocab_size={len(tokenizer)}")
        print(f"n_layers={n_layers} → using layer_index={resolved_layer}")
        print(f"todo={len(todo)} done={len(done)}")
        print(f"metrics → {metrics_jsonl}")

    if not todo:
        _rewrite_metrics_csv(metrics_jsonl, metrics_csv)
        _rewrite_category_summary(metrics_jsonl, category_summary_csv)
        return {
            "n_total": len(corpus),
            "n_done": len(done),
            "n_new": 0,
            "layer_index": resolved_layer,
        }

    metrics_buf: list[dict[str, Any]] = []
    n_new = 0

    for i, row in enumerate(todo, start=1):
        word_a = str(row["word_a"])
        word_b = str(row["word_b"])
        prompts = build_prompts(word_a, word_b)

        v_a = extract_sae_features(
            tokenizer, model, sae, prompts["P_A"], device=device_t, layer_index=resolved_layer
        )
        v_b = extract_sae_features(
            tokenizer, model, sae, prompts["P_B"], device=device_t, layer_index=resolved_layer
        )
        v_and = extract_sae_features(
            tokenizer, model, sae, prompts["P_AND"], device=device_t, layer_index=resolved_layer
        )
        v_or = extract_sae_features(
            tokenizer, model, sae, prompts["P_OR"], device=device_t, layer_index=resolved_layer
        )

        metrics = compute_composition_metrics(
            v_a, v_b, v_and, v_or, jaccard_threshold=jaccard_threshold
        )

        meta = {
            "category_id": row.get("category_id"),
            "category_name": row.get("category_name"),
            "domain_seed": row.get("domain_seed"),
            "word_a": word_a,
            "word_b": word_b,
            "model_id": model_id,
            "sae_release": sae_release,
            "sae_id": sae_id,
            "layer_index": resolved_layer,
            "n_layers": n_layers,
            "prompts": prompts,
            "metrics": metrics,
        }
        metrics_buf.append(meta)
        n_new += 1

        if verbose and (i == 1 or i % 10 == 0 or i == len(todo)):
            print(
                f"[{i}/{len(todo)}] L={resolved_layer} cat={meta['category_id']} "
                f"{word_a!r} / {word_b!r} | "
                f"tvd_and_min={metrics['tvd_and_vs_min']:.4f} "
                f"tvd_or_max={metrics['tvd_or_vs_max']:.4f} "
                f"jaccard={metrics['jaccard_a_b']:.4f}"
            )

        if len(metrics_buf) >= save_every:
            _append_jsonl(metrics_jsonl, metrics_buf)
            metrics_buf.clear()
            _rewrite_metrics_csv(metrics_jsonl, metrics_csv)
            _rewrite_category_summary(metrics_jsonl, category_summary_csv)

    if metrics_buf:
        _append_jsonl(metrics_jsonl, metrics_buf)
        _rewrite_metrics_csv(metrics_jsonl, metrics_csv)
        _rewrite_category_summary(metrics_jsonl, category_summary_csv)

    if verbose:
        print(f"Wrote metrics JSONL → {metrics_jsonl}")
        print(f"Wrote metrics CSV   → {metrics_csv}")
        print(f"Wrote category summary → {category_summary_csv}")

    return {
        "n_total": len(corpus),
        "n_done_before": len(done),
        "n_new": n_new,
        "layer_index": resolved_layer,
        "metrics_csv": str(metrics_csv),
        "category_summary_csv": str(category_summary_csv),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAE-lens AND/OR composition metrics on the orthogonality corpus."
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_JSONL)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--sae-release", default=SAE_RELEASE)
    parser.add_argument("--sae-id", default=None, help="Default: layer_{L}/width_16k/canonical")
    parser.add_argument("--layer-index", type=int, default=LAYER_INDEX)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--dtype", default=DTYPE)
    parser.add_argument("--jaccard-threshold", type=float, default=JACCARD_THRESHOLD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metrics-jsonl", type=Path, default=None)
    parser.add_argument("--metrics-csv", type=Path, default=None)
    parser.add_argument("--category-summary-csv", type=Path, default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--categories", type=int, nargs="*", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    parser.add_argument("-q", "--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_experiment(
        corpus_path=args.corpus,
        model_id=args.model_id,
        sae_release=args.sae_release,
        sae_id=args.sae_id,
        layer_index=args.layer_index,
        device=args.device,
        dtype_name=args.dtype,
        jaccard_threshold=args.jaccard_threshold,
        output_dir=args.output_dir,
        metrics_jsonl=args.metrics_jsonl,
        metrics_csv=args.metrics_csv,
        category_summary_csv=args.category_summary_csv,
        max_pairs=args.max_pairs,
        resume=not args.no_resume,
        save_every=args.save_every,
        categories=args.categories,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
