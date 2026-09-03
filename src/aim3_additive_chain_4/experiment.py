"""
Shared-trajectory true -ln π experiment (Aim-3 v4).

d(n_i → n_j) = -ln π(label_j | prefix through n_i) from the LM head.
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
    DEVICE,
    DTYPE,
    LAYER_INDEX,
    MODEL_ID,
    SAVE_EVERY,
    distances_jsonl,
    summary_json,
)
from .distances import type_a_ln_distances, type_b_ln_distances, type_c_ln_distances
from .model_io import load_model_and_tokenizer, num_transformer_layers, resolve_layer_index
from .prompts import build_trajectory, item_id, type_key


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
) -> dict[str, Any]:
    key = type_key(row)
    traj = build_trajectory(row)
    out: dict[str, Any] = {
        "item_id": item_id(row, index),
        "index": index,
        "type_key": key,
        "structure_type": row.get("structure_type"),
        "layer_index": layer_index,
        "trajectory_text": traj.text,
        "waypoint_names": list(traj.waypoint_names),
        "labels": list(traj.labels),
        "prompt_design": "shared_trajectory_true_ln_pi_v4",
        "metric": "d_M=-ln π(label|prefix)",
    }

    if key == "A":
        d = type_a_ln_distances(tokenizer, model, traj, device=device)
        out["token_distances"] = {
            k: d[k]
            for k in ("n1_n2", "n2_n3", "n3_n4", "n1_n4", "sum_hops", "additive_gap")
        }
        out["logprobs"] = {k: d[k] for k in d if k.startswith("logprob_")}
        out["additive_gap"] = d["additive_gap"]
    elif key == "B":
        d = type_b_ln_distances(tokenizer, model, traj, device=device)
        out["token_distances"] = d
        out["triangle_slack"] = d["triangle_slack"]
    elif key == "C":
        d = type_c_ln_distances(tokenizer, model, traj, device=device)
        out["token_distances"] = d
        out["triangle_slack"] = d["triangle_slack"]
    return out


def _write_summary(rows: list[dict[str, Any]], path: Path, layer_index: int) -> None:
    by_type: dict[str, list[dict[str, Any]]] = {"A": [], "B": [], "C": []}
    for r in rows:
        by_type.setdefault(str(r["type_key"]), []).append(r)

    summary: dict[str, Any] = {
        "layer_index": layer_index,
        "n_items": len(rows),
        "prompt_design": "shared_trajectory_true_ln_pi_v4",
        "metric": "d_M=-ln π(label|prefix through waypoint)",
        "by_type": {},
    }
    a_rows = by_type.get("A", [])
    if a_rows:
        gaps = np.array([float(r["additive_gap"]) for r in a_rows], dtype=float)
        summary["by_type"]["A"] = {
            "n": len(a_rows),
            "gap_mean": float(gaps.mean()),
            "gap_median": float(np.median(gaps)),
            "gap_frac_le_0": float(np.mean(gaps <= 0)),
            "gap_near_zero_abs_lt_1": float(np.mean(np.abs(gaps) < 1.0)),
            "gap_near_zero_abs_lt_2": float(np.mean(np.abs(gaps) < 2.0)),
        }
    for key in ("B", "C"):
        sub = by_type.get(key, [])
        if not sub:
            continue
        slacks = np.array([float(r["triangle_slack"]) for r in sub], dtype=float)
        summary["by_type"][key] = {
            "n": len(sub),
            "slack_mean": float(slacks.mean()),
            "n_violations": int(np.sum(slacks > 0)),
            "violation_rate": float(np.mean(slacks > 0)),
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
    resume: bool = True,
    save_every: int = SAVE_EVERY,
    verbose: bool = True,
) -> dict[str, Any]:
    corpus_path = Path(corpus_path)
    output_dir = Path(output_dir)

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
        print(f"design=shared_trajectory_true_ln_pi_v4 layer={resolved}/{n_layers}")
        print("metric=d_M=-ln π(label | prefix through waypoint)")
        print(f"corpus={corpus_path} todo={len(todo)} done={len(done)}")
        for i, r in indexed:
            if type_key(r) == "A":
                traj = build_trajectory(r)
                print("--- example ---")
                print("prefix through n1:")
                print(traj.prefix_through(0))
                print("target hop n1→n2:", traj.labels[1])
                print("target shortcut n1→n4:", traj.labels[3])
                print("----------------")
                break

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
            )
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  SKIP {iid}: {exc}", flush=True)
            continue
        buf.append(result)
        n_new += 1
        if verbose and type_key(row) == "A":
            td = result["token_distances"]
            print(
                f"  hops={td['n1_n2']:.2f}/{td['n2_n3']:.2f}/{td['n3_n4']:.2f} "
                f"direct={td['n1_n4']:.2f} gap={td['additive_gap']:.2f}",
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
        description="Aim-3 v4: true -ln π on shared-trajectory prefixes."
    )
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_JSON)
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--layer-index", type=int, default=LAYER_INDEX)
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--dtype", default=DTYPE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--types", default="A,B,C")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--save-every", type=int, default=SAVE_EVERY)
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
        resume=not args.no_resume,
        save_every=args.save_every,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
