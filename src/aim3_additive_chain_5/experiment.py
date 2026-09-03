"""
Shared-trajectory TVD / half-L1 path-cost experiment (Aim-3 v5).

δ = (1/2)||π_a - π_b||_1 on SAE (and J-lens) fingerprints along one forward.
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
    TOP_K_RESID,
    TOP_K_VOCAB,
    distances_jsonl,
    sae_id_for_layer,
    summary_json,
)
from .distances import (
    to_fingerprint,
    type_a_tvd_path,
    type_b_tvd_path,
    type_c_tvd_path,
)
from .jacobian import compute_truncated_jacobian_at_token, residuals_to_j_features
from .model_io import (
    extract_trajectory_fingerprints,
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
    top_k_vocab: int,
    top_k_resid: int,
    compute_j: bool,
) -> dict[str, Any]:
    key = type_key(row)
    traj = build_trajectory(row)
    states = extract_trajectory_fingerprints(
        tokenizer,
        model,
        sae,
        traj,
        device=device,
        layer_index=layer_index,
    )
    sae_fps = [to_fingerprint(v, kind="sae") for v in states["sae_vecs"]]

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
        "prompt_design": "shared_trajectory_tvd_path_v5",
        "metric": "TVD = (1/2)||π_a - π_b||_1",
    }

    if key == "A":
        sae_d = type_a_tvd_path(sae_fps)
        out["sae_tvd"] = sae_d
        out["additive_gap_sae"] = sae_d["additive_gap"]
    elif key == "B":
        sae_d = type_b_tvd_path(sae_fps)
        out["sae_tvd"] = sae_d
        out["triangle_slack_sae"] = sae_d["triangle_slack"]
    else:
        sae_d = type_c_tvd_path(sae_fps)
        out["sae_tvd"] = sae_d
        out["triangle_slack_sae"] = sae_d["triangle_slack"]

    if compute_j:
        ref_t = int(states["token_indices"][-1])
        j_pack = compute_truncated_jacobian_at_token(
            tokenizer,
            model,
            traj.text,
            ref_t,
            device=device,
            layer_index=layer_index,
            top_k_vocab=top_k_vocab,
            top_k_resid=top_k_resid,
        )
        j_raw = residuals_to_j_features(states["residuals"], j_pack)
        j_fps = [to_fingerprint(v, kind="j") for v in j_raw]
        if key == "A":
            j_d = type_a_tvd_path(j_fps)
            out["j_tvd"] = j_d
            out["additive_gap_j"] = j_d["additive_gap"]
        elif key == "B":
            j_d = type_b_tvd_path(j_fps)
            out["j_tvd"] = j_d
            out["triangle_slack_j"] = j_d["triangle_slack"]
        else:
            j_d = type_c_tvd_path(j_fps)
            out["j_tvd"] = j_d
            out["triangle_slack_j"] = j_d["triangle_slack"]

    return out


def _write_summary(rows: list[dict[str, Any]], path: Path, layer_index: int) -> None:
    by_type: dict[str, list[dict[str, Any]]] = {"A": [], "B": [], "C": []}
    for r in rows:
        by_type.setdefault(str(r["type_key"]), []).append(r)

    summary: dict[str, Any] = {
        "layer_index": layer_index,
        "n_items": len(rows),
        "prompt_design": "shared_trajectory_tvd_path_v5",
        "metric": "TVD=(1/2)||π_a-π_b||_1 on fingerprints",
        "by_type": {},
    }
    a_rows = by_type.get("A", [])
    if a_rows:
        g_sae = np.array([float(r["additive_gap_sae"]) for r in a_rows], dtype=float)
        blob: dict[str, Any] = {
            "n": len(a_rows),
            "sae_gap_mean": float(g_sae.mean()),
            "sae_gap_median": float(np.median(g_sae)),
            "sae_gap_frac_le_0": float(np.mean(g_sae <= 0)),
            "sae_gap_near_zero_abs_lt_0_05": float(np.mean(np.abs(g_sae) < 0.05)),
            "sae_gap_near_zero_abs_lt_0_1": float(np.mean(np.abs(g_sae) < 0.1)),
        }
        if all("additive_gap_j" in r for r in a_rows):
            g_j = np.array([float(r["additive_gap_j"]) for r in a_rows], dtype=float)
            blob.update(
                {
                    "j_gap_mean": float(g_j.mean()),
                    "j_gap_median": float(np.median(g_j)),
                    "j_gap_frac_le_0": float(np.mean(g_j <= 0)),
                    "j_gap_near_zero_abs_lt_0_05": float(np.mean(np.abs(g_j) < 0.05)),
                    "j_gap_near_zero_abs_lt_0_1": float(np.mean(np.abs(g_j) < 0.1)),
                }
            )
        summary["by_type"]["A"] = blob

    for key in ("B", "C"):
        sub = by_type.get(key, [])
        if not sub:
            continue
        slacks = np.array([float(r["triangle_slack_sae"]) for r in sub], dtype=float)
        summary["by_type"][key] = {
            "n": len(sub),
            "sae_slack_mean": float(slacks.mean()),
            "n_violations": int(np.sum(slacks > 1e-9)),
            "violation_rate": float(np.mean(slacks > 1e-9)),
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
    top_k_vocab: int = TOP_K_VOCAB,
    top_k_resid: int = TOP_K_RESID,
    compute_j: bool = True,
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
        print(f"design=shared_trajectory_tvd_path_v5 layer={resolved}/{n_layers}")
        print("metric=TVD=(1/2)||π_a-π_b||_1  (SAE" + ("+J" if compute_j else "") + ")")
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
                sae=sae,
                device=device_t,
                layer_index=resolved,
                top_k_vocab=top_k_vocab,
                top_k_resid=top_k_resid,
                compute_j=compute_j,
            )
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  SKIP {iid}: {exc}", flush=True)
            continue
        buf.append(result)
        n_new += 1
        if verbose and type_key(row) == "A":
            msg = f"  SAE gap={result['additive_gap_sae']:.4f}"
            if "additive_gap_j" in result:
                msg += f"  J gap={result['additive_gap_j']:.4f}"
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
    p = argparse.ArgumentParser(
        description="Aim-3 v5: shared-trajectory TVD / half-L1 path costs."
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
    p.add_argument("--top-k-vocab", type=int, default=TOP_K_VOCAB)
    p.add_argument("--top-k-resid", type=int, default=TOP_K_RESID)
    p.add_argument("--no-j", action="store_true", help="Skip J-lens TVD (SAE only).")
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
        top_k_vocab=args.top_k_vocab,
        top_k_resid=args.top_k_resid,
        compute_j=not args.no_j,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
