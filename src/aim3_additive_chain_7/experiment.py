"""
Aim-3 v7 calibration experiment.

1) Chain-rule tautology on Type A (all items): joint vs factorized log π.
2) Markov cloze (v6 claim) restricted to short-BPE concept chains.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .config import (
    DEFAULT_CORPUS_JSON,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VARIANTS,
    DEVICE,
    DTYPE,
    LAYER_INDEX,
    MAX_BPE_PER_NODE,
    MAX_TARGET_TOKENS,
    MODEL_ID,
    SAVE_EVERY,
    distances_jsonl,
    summary_json,
)
from .distances import chain_rule_sanity, type_a_markov_pack
from .model_io import (
    concept_bpe_len,
    load_model_and_tokenizer,
    nodes_within_bpe_budget,
    num_transformer_layers,
    resolve_layer_index,
)
from .prompts import item_id, parse_variants, type_a_nodes, type_key


def load_corpus(path: Path | str = DEFAULT_CORPUS_JSON) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return [r for r in data if isinstance(r, dict)]


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                done.add(str(json.loads(line)["item_id"]))
    return done


def score_item(
    row: dict[str, Any],
    *,
    index: int,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    layer_index: int,
    variants: list[str],
    max_tokens: int,
    max_bpe: int,
    run_markov: bool,
) -> dict[str, Any]:
    key = type_key(row)
    if key != "A":
        raise ValueError("v7 calibration is Type A only")

    nodes = type_a_nodes(row)
    bpe_lens = [concept_bpe_len(tokenizer, n) for n in nodes]
    short = nodes_within_bpe_budget(tokenizer, nodes, max_bpe=max_bpe)

    out: dict[str, Any] = {
        "item_id": item_id(row, index),
        "index": index,
        "type_key": key,
        "structure_type": row.get("structure_type"),
        "layer_index": layer_index,
        "prompt_design": "calibration_chain_rule_and_short_markov_v7",
        "nodes": list(nodes),
        "bpe_lens": bpe_lens,
        "short_bpe": short,
        "max_bpe_per_node": max_bpe,
    }

    out["chain_rule"] = chain_rule_sanity(
        tokenizer, model, row, device=device, max_tokens=max_tokens
    )

    if run_markov and short:
        packs = {
            v: type_a_markov_pack(
                tokenizer,
                model,
                row,
                device=device,
                variant=v,
                max_tokens=max_tokens,
            )
            for v in variants
        }
        out["markov_variants"] = packs
        primary = packs.get("natural") or next(iter(packs.values()))
        out["markov_additive_gap"] = primary["distances"]["additive_gap"]
        out["markov_relative_gap"] = primary["distances"]["relative_gap"]
    return out


def _finite(a: np.ndarray) -> np.ndarray:
    return a[np.isfinite(a)]


def _write_summary(rows: list[dict[str, Any]], path: Path, layer_index: int) -> None:
    cr_gaps = _finite(
        np.asarray([float(r["chain_rule"]["chain_rule_gap"]) for r in rows], dtype=float)
    )
    short_rows = [r for r in rows if r.get("short_bpe") and r.get("markov_variants")]

    summary: dict[str, Any] = {
        "layer_index": layer_index,
        "n_items": len(rows),
        "prompt_design": "calibration_chain_rule_and_short_markov_v7",
        "chain_rule": {
            "n": int(cr_gaps.size),
            "gap_mean": float(cr_gaps.mean()) if cr_gaps.size else None,
            "gap_median": float(np.median(cr_gaps)) if cr_gaps.size else None,
            "gap_mae": float(np.mean(np.abs(cr_gaps))) if cr_gaps.size else None,
            "gap_max_abs": float(np.max(np.abs(cr_gaps))) if cr_gaps.size else None,
            "frac_abs_lt_1e-3": float(np.mean(np.abs(cr_gaps) < 1e-3)) if cr_gaps.size else None,
            "frac_abs_lt_1e-2": float(np.mean(np.abs(cr_gaps) < 1e-2)) if cr_gaps.size else None,
            "frac_abs_lt_0_1": float(np.mean(np.abs(cr_gaps) < 0.1)) if cr_gaps.size else None,
            "interpretation": "gap≈0 ⇒ joint vs factorized scoring is consistent (tautology)",
        },
        "short_markov": {"n_short": len(short_rows), "by_variant": {}},
    }

    if short_rows:
        variant_names: set[str] = set()
        for r in short_rows:
            variant_names.update((r.get("markov_variants") or {}).keys())
        by_v: dict[str, Any] = {}
        for vname in sorted(variant_names):
            gaps, rels = [], []
            for r in short_rows:
                pack = (r.get("markov_variants") or {}).get(vname)
                if not pack:
                    continue
                gaps.append(float(pack["distances"]["additive_gap"]))
                rels.append(float(pack["distances"]["relative_gap"]))
            g = _finite(np.asarray(gaps, dtype=float))
            rel = _finite(np.asarray(rels, dtype=float))
            if not g.size:
                continue
            by_v[vname] = {
                "n": int(g.size),
                "gap_mean": float(g.mean()),
                "gap_median": float(np.median(g)),
                "gap_near_zero_abs_lt_1": float(np.mean(np.abs(g) < 1.0)),
                "gap_near_zero_abs_lt_2": float(np.mean(np.abs(g) < 2.0)),
                "relative_gap_mean": float(rel.mean()) if rel.size else None,
            }
        summary["short_markov"]["by_variant"] = by_v
        if "natural" in by_v and "shuffled_middles" in by_v:
            nat = np.asarray(
                [
                    float(r["markov_variants"]["natural"]["distances"]["additive_gap"])
                    for r in short_rows
                    if "natural" in (r.get("markov_variants") or {})
                ],
                dtype=float,
            )
            shuf = np.asarray(
                [
                    float(r["markov_variants"]["shuffled_middles"]["distances"]["additive_gap"])
                    for r in short_rows
                    if "shuffled_middles" in (r.get("markov_variants") or {})
                ],
                dtype=float,
            )
            m = min(nat.size, shuf.size)
            if m:
                summary["short_markov"]["paired_natural_closer_than_shuffled"] = float(
                    np.mean(np.abs(nat[:m]) < np.abs(shuf[:m]))
                )
                summary["short_markov"]["paired_natural_less_negative_than_shuffled"] = float(
                    np.mean(nat[:m] > shuf[:m])
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def run_experiment(
    *,
    corpus_path: Path | str = DEFAULT_CORPUS_JSON,
    model_id: str = MODEL_ID,
    layer_index: int = LAYER_INDEX,
    device: str = DEVICE,
    dtype_name: str = DTYPE,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    max_items: int | None = None,
    variants: Iterable[str] | str | None = DEFAULT_VARIANTS,
    resume: bool = True,
    save_every: int = SAVE_EVERY,
    max_tokens: int = MAX_TARGET_TOKENS,
    max_bpe: int = MAX_BPE_PER_NODE,
    skip_markov: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    corpus_path = Path(corpus_path)
    output_dir = Path(output_dir)
    variant_list = parse_variants(variants)

    corpus = load_corpus(corpus_path)
    indexed = [(i, r) for i, r in enumerate(corpus) if type_key(r) == "A"]
    if max_items is not None:
        indexed = indexed[: int(max_items)]

    tokenizer, model, device_t = load_model_and_tokenizer(
        model_id=model_id, device=device, dtype_name=dtype_name
    )
    resolved = resolve_layer_index(model, layer_index)
    n_layers = num_transformer_layers(model)

    out_jsonl = distances_jsonl(resolved, output_dir)
    done = _load_done(out_jsonl) if resume else set()
    todo = [(i, r) for i, r in indexed if item_id(r, i) not in done]

    if verbose:
        print(f"model_id={model_id}")
        print(f"design=calibration_v7 layer={resolved}/{n_layers}")
        print(f"chain_rule=all Type A; markov=short BPE≤{max_bpe} variants={variant_list}")
        print(f"corpus={corpus_path} todo={len(todo)} done={len(done)}")

    buf: list[dict[str, Any]] = []
    n_new = 0
    for step, (index, row) in enumerate(todo, start=1):
        iid = item_id(row, index)
        if verbose:
            print(f"[{step}/{len(todo)}] {iid} …", flush=True)
        try:
            result = score_item(
                row,
                index=index,
                tokenizer=tokenizer,
                model=model,
                device=device_t,
                layer_index=resolved,
                variants=variant_list,
                max_tokens=max_tokens,
                max_bpe=max_bpe,
                run_markov=not skip_markov,
            )
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  SKIP {iid}: {exc}", flush=True)
            continue
        buf.append(result)
        n_new += 1
        if verbose:
            cr = result["chain_rule"]["chain_rule_gap"]
            msg = f"  chain_rule_gap={cr:.4g}"
            if "markov_additive_gap" in result:
                msg += (
                    f"  markov_gap={result['markov_additive_gap']:.3f}"
                    f"  rel={result['markov_relative_gap']:.3f}"
                )
            elif result.get("short_bpe") is False:
                msg += "  (skip markov: long BPE)"
            print(msg, flush=True)
        if device_t.type == "cuda":
            torch.cuda.empty_cache()
        if len(buf) >= save_every:
            _append_jsonl(out_jsonl, buf)
            buf.clear()

    if buf:
        _append_jsonl(out_jsonl, buf)

    all_rows: list[dict[str, Any]] = []
    if out_jsonl.exists():
        with out_jsonl.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    all_rows.append(json.loads(line))
    sum_path = summary_json(resolved, output_dir)
    _write_summary(all_rows, sum_path, resolved)

    if verbose:
        print(f"Wrote distances → {out_jsonl}")
        print(f"Wrote summary   → {sum_path}")
        if sum_path.exists():
            print(sum_path.read_text(encoding="utf-8"))

    return {"n_new": n_new, "n_total_file": len(all_rows), "layer_index": resolved}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aim-3 v7 calibration.")
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_JSON)
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--layer-index", type=int, default=LAYER_INDEX)
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--dtype", default=DTYPE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--variants", default=DEFAULT_VARIANTS)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--save-every", type=int, default=SAVE_EVERY)
    p.add_argument("--max-tokens", type=int, default=MAX_TARGET_TOKENS)
    p.add_argument("--max-bpe", type=int, default=MAX_BPE_PER_NODE)
    p.add_argument("--skip-markov", action="store_true")
    p.add_argument("-q", "--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_experiment(
        corpus_path=args.corpus,
        model_id=args.model_id,
        layer_index=args.layer_index,
        device=args.device,
        dtype_name=args.dtype,
        output_dir=args.output_dir,
        max_items=args.max_items,
        variants=args.variants,
        resume=not args.no_resume,
        save_every=args.save_every,
        max_tokens=args.max_tokens,
        max_bpe=args.max_bpe,
        skip_markov=args.skip_markov,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
