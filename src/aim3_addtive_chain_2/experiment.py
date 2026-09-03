"""
Shared-trajectory additive-chain experiment (Aim-3 v2).

For each corpus item: build one trajectory prompt → one forward → distances
between layer-L states (SAE tropical + Type-A logit-lens), not independent probes.
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
    SAE_ID,
    SAE_RELEASE,
    SAVE_EVERY,
    distances_jsonl,
    sae_id_for_layer,
    summary_json,
)
from .distances import (
    type_a_logit_lens_distances,
    type_a_sae_distances,
    type_b_sae_distances,
    type_c_sae_distances,
)
from .model_io import (
    extract_trajectory_states,
    load_model_and_tokenizer,
    load_sae,
    num_transformer_layers,
    resolve_layer_index,
)
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
    sae: Any,
    device: torch.device,
    layer_index: int,
) -> dict[str, Any]:
    key = type_key(row)
    traj = build_trajectory(row)
    states = extract_trajectory_states(
        tokenizer,
        model,
        sae,
        traj,
        device=device,
        layer_index=layer_index,
    )
    out: dict[str, Any] = {
        "item_id": item_id(row, index),
        "index": index,
        "type_key": key,
        "structure_type": row.get("structure_type"),
        "layer_index": layer_index,
        "trajectory_text": traj.text,
        "waypoint_names": list(traj.waypoint_names),
        "labels": list(traj.labels),
        "token_indices": states["token_indices"],
        "prompt_design": "shared_trajectory_v2",
    }

    if key == "A":
        sae_d = type_a_sae_distances(states["sae_vecs"])
        lens_d = type_a_logit_lens_distances(
            model,
            tokenizer,
            states["residuals"],
            traj.labels,
            device=device,
        )
        out["sae_distances"] = sae_d
        out["logit_lens_distances"] = lens_d
        out["additive_gap_sae"] = sae_d["additive_gap"]
        out["additive_gap_logit_lens"] = lens_d["additive_gap"]
    elif key == "B":
        sae_d = type_b_sae_distances(states["sae_vecs"])
        out["sae_distances"] = sae_d
        out["triangle_slack_sae"] = sae_d["triangle_slack"]
    elif key == "C":
        sae_d = type_c_sae_distances(states["sae_vecs"])
        out["sae_distances"] = sae_d
        out["triangle_slack_sae"] = sae_d["triangle_slack"]
    return out


def _write_summary(rows: list[dict[str, Any]], path: Path, layer_index: int) -> None:
    by_type: dict[str, list[dict[str, Any]]] = {"A": [], "B": [], "C": []}
    for r in rows:
        by_type.setdefault(str(r["type_key"]), []).append(r)

    summary: dict[str, Any] = {
        "layer_index": layer_index,
        "n_items": len(rows),
        "prompt_design": "shared_trajectory_v2",
        "by_type": {},
    }
    a_rows = by_type.get("A", [])
    if a_rows:
        gaps_sae = np.array([float(r["additive_gap_sae"]) for r in a_rows], dtype=float)
        gaps_ll = np.array(
            [float(r["additive_gap_logit_lens"]) for r in a_rows], dtype=float
        )
        summary["by_type"]["A"] = {
            "n": len(a_rows),
            "sae_gap_mean": float(gaps_sae.mean()),
            "sae_gap_frac_le_0": float(np.mean(gaps_sae <= 0)),
            "logit_lens_gap_mean": float(gaps_ll.mean()),
            "logit_lens_gap_frac_le_0": float(np.mean(gaps_ll <= 0)),
            "sae_gap_near_zero_abs_lt_1": float(np.mean(np.abs(gaps_sae) < 1.0)),
            "logit_lens_gap_near_zero_abs_lt_1": float(np.mean(np.abs(gaps_ll) < 1.0)),
        }
    for key in ("B", "C"):
        sub = by_type.get(key, [])
        if not sub:
            continue
        slacks = np.array([float(r["triangle_slack_sae"]) for r in sub], dtype=float)
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
    sae_release: str = SAE_RELEASE,
    sae_id: str | None = None,
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
    if sae_id is None:
        sae_id = sae_id_for_layer(layer_index)

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
    sae = load_sae(release=sae_release, sae_id=sae_id, device=device_t)

    out_jsonl = distances_jsonl(resolved, output_dir)
    done = _load_done(out_jsonl) if resume else set()
    todo = [(i, r) for i, r in indexed if item_id(r, i) not in done]

    if verbose:
        print(f"model_id={model_id}")
        print(f"sae={sae_release} / {sae_id}")
        print(f"design=shared_trajectory_v2 layer={resolved}/{n_layers}")
        print(f"corpus={corpus_path} todo={len(todo)} done={len(done)}")
        # Show one Type A prompt example
        for i, r in indexed:
            if type_key(r) == "A":
                ex = build_trajectory(r)
                print("--- example Type A trajectory ---")
                print(ex.text[:500])
                print("-------------------------------")
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
                sae=sae,
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
            print(
                f"  SAE gap={result['additive_gap_sae']:.3f}  "
                f"logit-lens gap={result['additive_gap_logit_lens']:.3f}",
                flush=True,
            )
        if len(buf) >= save_every:
            _append_jsonl(out_jsonl, buf)
            buf.clear()

    if buf:
        _append_jsonl(out_jsonl, buf)

    # Rewrite summary from full file
    all_rows = []
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
        description="Aim-3 v2 shared-trajectory additive chain experiment."
    )
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_JSON)
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--sae-release", default=SAE_RELEASE)
    p.add_argument("--sae-id", default=None)
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
        sae_release=args.sae_release,
        sae_id=args.sae_id,
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
