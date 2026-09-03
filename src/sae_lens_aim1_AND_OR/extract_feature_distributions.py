"""
Extract sparse Top-K SAE feature distributions for rank-diff plots.

Writes one JSONL row per corpus pair with L1-normalized Top-K features for
V_A, V_B, V_AND, V_OR. Resumable independently of the metrics JSONL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .config import (
    DEFAULT_CORPUS_JSONL,
    DEFAULT_OUTPUT_DIR,
    DEVICE,
    DTYPE,
    LAYER_INDEX,
    MODEL_ID,
    SAE_ID,
    SAE_RELEASE,
    SAVE_EVERY,
    TOP_K_FEATURES,
    distributions_jsonl_for_layer,
    sae_id_for_layer,
)
from .experiment import _append_jsonl, _load_done_keys, _pair_key, load_corpus
from .metrics import sparse_topk
from .model_io import (
    extract_sae_features,
    load_model_and_tokenizer,
    load_sae,
    num_transformer_layers,
    resolve_layer_index,
)
from .prompts import build_prompts


def run_extract(
    *,
    corpus_path: Path | str = DEFAULT_CORPUS_JSONL,
    model_id: str = MODEL_ID,
    sae_release: str = SAE_RELEASE,
    sae_id: str | None = None,
    layer_index: int = LAYER_INDEX,
    device: str = DEVICE,
    dtype_name: str = DTYPE,
    top_k: int = TOP_K_FEATURES,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    distributions_jsonl: Path | str | None = None,
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

    tokenizer, model, device_t = load_model_and_tokenizer(
        model_id=model_id,
        device=device,
        dtype_name=dtype_name,
    )
    n_layers = num_transformer_layers(model)
    resolved_layer = resolve_layer_index(model, layer_index)

    if distributions_jsonl is None:
        distributions_jsonl = distributions_jsonl_for_layer(resolved_layer, output_dir)
    distributions_jsonl = Path(distributions_jsonl)

    sae = load_sae(release=sae_release, sae_id=sae_id, device=device_t)

    done = _load_done_keys(distributions_jsonl) if resume else set()
    todo = [r for r in corpus if _pair_key(r) not in done]

    if verbose:
        print(f"model_id={model_id}")
        print(f"sae_release={sae_release} sae_id={sae_id}")
        print(f"corpus={corpus_path} (n={len(corpus)})")
        print(f"device={device_t} dtype={dtype_name} layer={resolved_layer} top_k={top_k}")
        print(f"todo={len(todo)} done={len(done)}")
        print(f"distributions → {distributions_jsonl}")

    if not todo:
        return {
            "n_total": len(corpus),
            "n_done": len(done),
            "n_new": 0,
            "layer_index": resolved_layer,
            "distributions_jsonl": str(distributions_jsonl),
        }

    buf: list[dict[str, Any]] = []
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

        rec = {
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
            "top_k": int(top_k),
            "V_A": sparse_topk(v_a, top_k),
            "V_B": sparse_topk(v_b, top_k),
            "V_AND": sparse_topk(v_and, top_k),
            "V_OR": sparse_topk(v_or, top_k),
        }
        buf.append(rec)
        n_new += 1

        if verbose and (i == 1 or i % 10 == 0 or i == len(todo)):
            print(
                f"[{i}/{len(todo)}] L={resolved_layer} cat={rec['category_id']} "
                f"{word_a!r} / {word_b!r} | "
                f"|A|={len(rec['V_A']['feature_ids'])} "
                f"|AND|={len(rec['V_AND']['feature_ids'])} "
                f"|OR|={len(rec['V_OR']['feature_ids'])}"
            )

        if len(buf) >= save_every:
            _append_jsonl(distributions_jsonl, buf)
            buf.clear()

    if buf:
        _append_jsonl(distributions_jsonl, buf)

    if verbose:
        print(f"Wrote distributions → {distributions_jsonl}")

    return {
        "n_total": len(corpus),
        "n_done_before": len(done),
        "n_new": n_new,
        "layer_index": resolved_layer,
        "distributions_jsonl": str(distributions_jsonl),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract sparse SAE Top-K feature distributions for rank-diff plots."
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_JSONL)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--sae-release", default=SAE_RELEASE)
    parser.add_argument("--sae-id", default=None)
    parser.add_argument("--layer-index", type=int, default=LAYER_INDEX)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--dtype", default=DTYPE)
    parser.add_argument("--top-k", type=int, default=TOP_K_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--distributions-jsonl", type=Path, default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--categories", type=int, nargs="*", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    parser.add_argument("-q", "--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_extract(
        corpus_path=args.corpus,
        model_id=args.model_id,
        sae_release=args.sae_release,
        sae_id=args.sae_id,
        layer_index=args.layer_index,
        device=args.device,
        dtype_name=args.dtype,
        top_k=args.top_k,
        output_dir=args.output_dir,
        distributions_jsonl=args.distributions_jsonl,
        max_pairs=args.max_pairs,
        resume=not args.no_resume,
        save_every=args.save_every,
        categories=args.categories,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
