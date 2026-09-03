"""
Aim-3 v6 experiment: Markov cloze + true sequence log π factorization.

Primary claim (Type A):
  π(n4|n1)  ?≈  π(n2|n1) · π(n3|n2) · π(n4|n3)
  ⇔  d(n1,n4) ?≈ d(n1,n2)+d(n2,n3)+d(n3,n4)
  with d_M = -sum_i log π(token_i | prefix, tokens_<i).

Controls: shuffled_middles and reversed node orders.
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
    MAX_TARGET_TOKENS,
    MODEL_ID,
    SAVE_EVERY,
    distances_jsonl,
    summary_json,
)
from .distances import type_a_markov_pack, type_b_triangle, type_c_triangle
from .model_io import load_model_and_tokenizer, num_transformer_layers, resolve_layer_index
from .prompts import (
    build_type_b_queries,
    build_type_c_queries,
    item_id,
    parse_variants,
    type_key,
)


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
) -> dict[str, Any]:
    key = type_key(row)
    out: dict[str, Any] = {
        "item_id": item_id(row, index),
        "index": index,
        "type_key": key,
        "structure_type": row.get("structure_type"),
        "layer_index": layer_index,
        "prompt_design": "markov_cloze_sequence_logprob_v6",
        "metric": "d_M=-sum log π(target tokens | cloze prefix)",
        "claim": "π(n4|n1) ?≈ Π hop π(n_{i+1}|n_i)",
    }

    if key == "A":
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
        out["variants"] = packs
        # Convenience: natural (or first) summary fields for plotting/resume
        primary = packs.get("natural") or next(iter(packs.values()))
        out["additive_gap"] = primary["distances"]["additive_gap"]
        out["relative_gap"] = primary["distances"]["relative_gap"]
        out["token_distances"] = primary["distances"]
    elif key == "B":
        pack = type_b_triangle(
            tokenizer,
            model,
            build_type_b_queries(row),
            device=device,
            max_tokens=max_tokens,
        )
        out["token_distances"] = pack["distances"]
        out["triangle_slack"] = pack["distances"]["triangle_slack"]
        out["queries"] = pack["queries"]
    else:
        pack = type_c_triangle(
            tokenizer,
            model,
            build_type_c_queries(row),
            device=device,
            max_tokens=max_tokens,
        )
        out["token_distances"] = pack["distances"]
        out["triangle_slack"] = pack["distances"]["triangle_slack"]
        out["queries"] = pack["queries"]
    return out


def _finite(arr: np.ndarray) -> np.ndarray:
    return arr[np.isfinite(arr)]


def _write_summary(rows: list[dict[str, Any]], path: Path, layer_index: int) -> None:
    by_type: dict[str, list[dict[str, Any]]] = {"A": [], "B": [], "C": []}
    for r in rows:
        by_type.setdefault(str(r["type_key"]), []).append(r)

    summary: dict[str, Any] = {
        "layer_index": layer_index,
        "n_items": len(rows),
        "prompt_design": "markov_cloze_sequence_logprob_v6",
        "metric": "d_M=-sum log π(target|cloze)",
        "claim": "π(n4|n1) ?≈ Π π(n_{i+1}|n_i)",
        "by_type": {},
    }

    a_rows = by_type.get("A", [])
    if a_rows:
        variant_names: set[str] = set()
        for r in a_rows:
            variant_names.update((r.get("variants") or {}).keys())
        by_variant: dict[str, Any] = {}
        for vname in sorted(variant_names):
            gaps, rels, log_ratios = [], [], []
            for r in a_rows:
                pack = (r.get("variants") or {}).get(vname)
                if not pack:
                    continue
                d = pack["distances"]
                gaps.append(float(d["additive_gap"]))
                rels.append(float(d["relative_gap"]))
                log_ratios.append(
                    float(pack["sum_logprobs"]["log_ratio_direct_over_product"])
                )
            g = _finite(np.asarray(gaps, dtype=float))
            rel = _finite(np.asarray(rels, dtype=float))
            lr = _finite(np.asarray(log_ratios, dtype=float))
            if g.size == 0:
                continue
            by_variant[vname] = {
                "n": int(g.size),
                "gap_mean": float(g.mean()),
                "gap_median": float(np.median(g)),
                "gap_frac_le_0": float(np.mean(g <= 0)),
                "gap_near_zero_abs_lt_1": float(np.mean(np.abs(g) < 1.0)),
                "gap_near_zero_abs_lt_2": float(np.mean(np.abs(g) < 2.0)),
                "relative_gap_mean": float(rel.mean()) if rel.size else None,
                "relative_gap_median": float(np.median(rel)) if rel.size else None,
                "log_ratio_direct_over_product_mean": float(lr.mean()) if lr.size else None,
            }
        summary["by_type"]["A"] = {
            "n": len(a_rows),
            "by_variant": by_variant,
            # headline = natural if present
            **(
                {"headline_natural": by_variant["natural"]}
                if "natural" in by_variant
                else {}
            ),
        }

    for key in ("B", "C"):
        sub = by_type.get(key, [])
        if not sub:
            continue
        slacks = np.array([float(r["triangle_slack"]) for r in sub], dtype=float)
        summary["by_type"][key] = {
            "n": len(sub),
            "slack_mean": float(slacks.mean()),
            "n_violations": int(np.sum(slacks > 1e-9)),
            "violation_rate": float(np.mean(slacks > 1e-9)),
        }

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
    types: Iterable[str] | None = None,
    variants: Iterable[str] | str | None = DEFAULT_VARIANTS,
    resume: bool = True,
    save_every: int = SAVE_EVERY,
    max_tokens: int = MAX_TARGET_TOKENS,
    verbose: bool = True,
) -> dict[str, Any]:
    corpus_path = Path(corpus_path)
    output_dir = Path(output_dir)
    variant_list = parse_variants(variants)

    corpus = load_corpus(corpus_path)
    indexed = list(enumerate(corpus))
    if types is not None:
        wanted = {t.strip().upper() for t in types}
        indexed = [(i, r) for i, r in indexed if type_key(r) in wanted]
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
        print(f"design=markov_cloze_sequence_logprob_v6 layer={resolved}/{n_layers}")
        print(f"variants={variant_list}")
        print("metric=d_M=-sum log π(target|cloze)  claim: π(n4|n1)?≈Π hops")
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
            )
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  SKIP {iid}: {exc}", flush=True)
            continue
        buf.append(result)
        n_new += 1
        if verbose and type_key(row) == "A":
            nat = (result.get("variants") or {}).get("natural") or {}
            d = nat.get("distances") or result.get("token_distances") or {}
            print(
                f"  gap={d.get('additive_gap', float('nan')):.3f} "
                f"rel={d.get('relative_gap', float('nan')):.3f} "
                f"D={d.get('n1_n4', float('nan')):.2f} "
                f"L={d.get('sum_hops', float('nan')):.2f}",
                flush=True,
            )
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
    p = argparse.ArgumentParser(
        description="Aim-3 v6: Markov cloze sequence-logprob factorization."
    )
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_JSON)
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--layer-index", type=int, default=LAYER_INDEX)
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--dtype", default=DTYPE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--types", default="A,B,C")
    p.add_argument("--variants", default=DEFAULT_VARIANTS)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--save-every", type=int, default=SAVE_EVERY)
    p.add_argument("--max-tokens", type=int, default=MAX_TARGET_TOKENS)
    p.add_argument("-q", "--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    types = [t.strip().upper() for t in args.types.split(",") if t.strip()]
    run_experiment(
        corpus_path=args.corpus,
        model_id=args.model_id,
        layer_index=args.layer_index,
        device=args.device,
        dtype_name=args.dtype,
        output_dir=args.output_dir,
        max_items=args.max_items,
        types=types,
        variants=args.variants,
        resume=not args.no_resume,
        save_every=args.save_every,
        max_tokens=args.max_tokens,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
