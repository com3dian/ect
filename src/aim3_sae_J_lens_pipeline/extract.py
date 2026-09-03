"""
Phase 3.1 — GPU extraction: SAE features + truncated J-lens Jacobian.

Iterates Aim-3 tropical_geometry_corpus.json, writes per-waypoint safetensors
and tropical log-probs, then the caller should free GPU memory.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors.torch import save_file

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
    TOP_K_SAE,
    TOP_K_VOCAB,
    manifest_jsonl,
    sae_id_for_layer,
    tensor_path,
    tensors_dir,
    tropical_logprobs_jsonl,
)
from .jacobian import compute_truncated_jacobian
from .model_io import (
    extract_sae_features,
    load_model_and_tokenizer,
    load_sae,
    mean_target_logprob,
    num_transformer_layers,
    resolve_layer_index,
)
from .prompts import (
    jacobian_waypoint_name,
    item_id,
    tropical_pairs_for_row,
    type_key,
    waypoints_for_row,
)
from .tropical import tropical_distance_from_logprob


def load_corpus(path: Path | str = DEFAULT_CORPUS_JSON) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return [row for row in data if isinstance(row, dict)]


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_done_item_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.add(str(row["item_id"]))
    return done


def _topk_sae(acts: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    k = min(int(k), int(acts.numel()))
    values, indices = torch.topk(acts.float(), k=k, largest=True, sorted=True)
    return indices.long(), values.float()


def _clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_item(
    row: dict[str, Any],
    *,
    index: int,
    tokenizer: Any,
    model: Any,
    sae: Any,
    device: torch.device,
    layer_index: int,
    output_dir: Path,
    top_k_sae: int,
    top_k_vocab: int,
    top_k_resid: int,
    compute_jacobian: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    iid = item_id(row, index)
    key = type_key(row)
    jac_name = jacobian_waypoint_name(key)
    waypoints_meta: list[dict[str, Any]] = []

    for wp in waypoints_for_row(row):
        acts = extract_sae_features(
            tokenizer,
            model,
            sae,
            wp["prompt"],
            device=device,
            layer_index=layer_index,
        )
        sae_idx, sae_val = _topk_sae(acts, top_k_sae)
        payload: dict[str, torch.Tensor] = {
            "sae_indices": sae_idx.cpu(),
            "sae_values": sae_val.cpu(),
            "sae_l1": acts.float().sum().reshape(()),
            "d_sae": torch.tensor([int(acts.numel())], dtype=torch.int64),
            "layer_index": torch.tensor([int(layer_index)], dtype=torch.int64),
        }

        do_j = compute_jacobian and wp["name"] == jac_name
        if do_j:
            j_pack = compute_truncated_jacobian(
                tokenizer,
                model,
                wp["prompt"],
                device=device,
                layer_index=layer_index,
                top_k_vocab=top_k_vocab,
                top_k_resid=top_k_resid,
            )
            payload.update(j_pack)

        dest = tensor_path(iid, wp["name"], layer_index, output_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        save_file(payload, str(dest))
        waypoints_meta.append(
            {
                "name": wp["name"],
                "path": str(dest),
                "has_jacobian": bool(do_j),
                "prompt": wp["prompt"],
                "concept": wp.get("concept", ""),
            }
        )
        del acts, payload
        _clear_cuda()

    trop_rows: list[dict[str, Any]] = []
    for pair in tropical_pairs_for_row(row):
        logp = mean_target_logprob(
            tokenizer,
            model,
            pair["source_prompt"],
            pair["target_text"],
            device=device,
        )
        dist = tropical_distance_from_logprob(logp)
        trop_rows.append(
            {
                "item_id": iid,
                "structure_type": row.get("structure_type"),
                "type_key": key,
                "pair_name": pair["pair_name"],
                "source_label": pair["source_label"],
                "target_label": pair["target_label"],
                "logprob": logp,
                "tropical_distance": dist,
                "layer_index": layer_index,
            }
        )
        _clear_cuda()

    manifest = {
        "item_id": iid,
        "index": index,
        "type_key": key,
        "structure_type": row.get("structure_type"),
        "layer_index": layer_index,
        "waypoints": waypoints_meta,
        "corpus_fields": {
            k: row[k]
            for k in row
            if k != "structure_type" and isinstance(row[k], str)
        },
    }
    return manifest, trop_rows


def run_extraction(
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
    top_k_sae: int = TOP_K_SAE,
    top_k_vocab: int = TOP_K_VOCAB,
    top_k_resid: int = TOP_K_RESID,
    compute_jacobian: bool = True,
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
    n_layers = num_transformer_layers(model)
    resolved_layer = resolve_layer_index(model, layer_index)
    sae = load_sae(release=sae_release, sae_id=sae_id, device=device_t)

    tensors_dir(resolved_layer, output_dir).mkdir(parents=True, exist_ok=True)
    man_path = manifest_jsonl(resolved_layer, output_dir)
    trop_path = tropical_logprobs_jsonl(resolved_layer, output_dir)
    done = _load_done_item_ids(man_path) if resume else set()
    todo = [(i, r) for i, r in indexed if item_id(r, i) not in done]

    if verbose:
        print(f"model_id={model_id}")
        print(f"sae_release={sae_release} sae_id={sae_id}")
        print(f"corpus={corpus_path} n={len(indexed)} todo={len(todo)} done={len(done)}")
        print(f"device={device_t} dtype={dtype_name} layer={resolved_layer}/{n_layers}")
        print(
            f"top_k_sae={top_k_sae} top_k_vocab={top_k_vocab} "
            f"top_k_resid={top_k_resid} jacobian={compute_jacobian}"
        )
        print(f"tensors → {tensors_dir(resolved_layer, output_dir)}")

    man_buf: list[dict[str, Any]] = []
    trop_buf: list[dict[str, Any]] = []
    n_new = 0

    for step, (index, row) in enumerate(todo, start=1):
        iid = item_id(row, index)
        if verbose:
            print(f"[{step}/{len(todo)}] {iid} {type_key(row)} …", flush=True)
        try:
            manifest, trop_rows = extract_item(
                row,
                index=index,
                tokenizer=tokenizer,
                model=model,
                sae=sae,
                device=device_t,
                layer_index=resolved_layer,
                output_dir=output_dir,
                top_k_sae=top_k_sae,
                top_k_vocab=top_k_vocab,
                top_k_resid=top_k_resid,
                compute_jacobian=compute_jacobian,
            )
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  SKIP {iid}: {exc}", flush=True)
            continue

        man_buf.append(manifest)
        trop_buf.extend(trop_rows)
        n_new += 1
        if verbose:
            print(
                f"  waypoints={len(manifest['waypoints'])} "
                f"tropical_pairs={len(trop_rows)}",
                flush=True,
            )

        if len(man_buf) >= save_every:
            _append_jsonl(man_path, man_buf)
            _append_jsonl(trop_path, trop_buf)
            man_buf.clear()
            trop_buf.clear()

    if man_buf:
        _append_jsonl(man_path, man_buf)
        _append_jsonl(trop_path, trop_buf)

    if verbose:
        print(f"Wrote manifest → {man_path}")
        print(f"Wrote tropical logprobs → {trop_path}")

    return {
        "n_total": len(indexed),
        "n_done_before": len(done),
        "n_new": n_new,
        "layer_index": resolved_layer,
        "manifest": str(man_path),
    }


def release_gpu(*modules: Any) -> None:
    """Drop model/SAE references and empty CUDA cache (Phase 3.1 → 3.2 handoff)."""
    for obj in modules:
        del obj
    _clear_cuda()
