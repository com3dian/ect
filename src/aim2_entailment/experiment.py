"""
Aim-2 entailment experiment: vector extraction + activation patching sweep.

Pipeline (README §3):
  1) prepare_data  → nli_pairs.jsonl
  2) extract v_trad, v_ECT (Top-K safetensors per pair × layer)
  3) patch + sweep α, layers; log target hypothesis log-probs → CSV
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from .config import (
    DEFAULT_ALPHAS,
    DEFAULT_DATA_DIR,
    DEFAULT_LAYERS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PAIRS_JSONL,
    MODEL_ID,
    SAVE_EVERY,
    TOP_K_DIMS,
    DEVICE,
    DTYPE,
    parse_alphas,
    results_csv,
    results_jsonl,
    vector_path,
    vectors_dir,
)
from .model_io import (
    load_model_and_tokenizer,
    mean_target_logprob,
    resolve_layer_index,
    resolve_target_word,
    target_token_ids,
    target_word_token_id,
)
from .patching import patch_residual_stream
from .prepare_data import load_pairs
from .prompts import build_prompt_bundle
from .vectors import extract_steering_vectors, load_vectors, save_vectors


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _rewrite_results_csv(jsonl_path: Path, csv_path: Path) -> None:
    if not jsonl_path.exists():
        return
    rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_done_patch_keys(jsonl_path: Path) -> set[tuple[str, int, str, float, str]]:
    done: set[tuple[str, int, str, float, str]] = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.add(
                (
                    str(row["pair_id"]),
                    int(row["layer_index"]),
                    str(row["vector_type"]),
                    float(row["alpha"]),
                    str(row.get("patch_input", "clean")),
                )
            )
    return done


def extract_vectors_for_pairs(
    pairs: list[dict[str, Any]],
    *,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    layers: Iterable[int],
    output_dir: Path,
    top_k_dims: int = TOP_K_DIMS,
    verbose: bool = True,
) -> None:
    out_v = vectors_dir(output_dir)
    out_v.mkdir(parents=True, exist_ok=True)

    for row in pairs:
        pair_id = str(row["pair_id"])
        prompts = row.get("prompts") or build_prompt_bundle(row)
        for layer in layers:
            resolved = resolve_layer_index(model, layer)
            dest = vector_path(pair_id, resolved, output_dir)
            if dest.exists():
                if verbose:
                    print(f"skip extract {pair_id} L={resolved} (exists)")
                continue
            if verbose:
                print(f"extract {pair_id} L={resolved} …")
            payload = extract_steering_vectors(
                tokenizer,
                model,
                premise_prompt=prompts["premise"],
                hypothesis_prompt=prompts["hypothesis"],
                hypothesis=str(row["hypothesis"]),
                device=device,
                layer_index=resolved,
                top_k_dims=top_k_dims,
                target_word=row.get("target_word"),
            )
            save_vectors(payload, dest)
            del payload
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()


@torch.inference_mode()
def evaluate_patch_sweep(
    pairs: list[dict[str, Any]],
    *,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    layers: Iterable[int],
    alphas: Iterable[float],
    output_dir: Path,
    resume: bool = True,
    save_every: int = SAVE_EVERY,
    verbose: bool = True,
) -> None:
    jsonl_out = results_jsonl(output_dir)
    csv_out = results_csv(output_dir)
    done = _load_done_patch_keys(jsonl_out) if resume else set()
    buf: list[dict[str, Any]] = []

    for row in pairs:
        pair_id = str(row["pair_id"])
        prompts = row.get("prompts") or build_prompt_bundle(row)
        hypothesis = str(row["hypothesis"])
        target_ids = target_token_ids(tokenizer, hypothesis)
        target_word = resolve_target_word(hypothesis, row.get("target_word"))
        target_tid, target_piece = target_word_token_id(tokenizer, target_word)

        baseline_clean = mean_target_logprob(
            tokenizer, model, prompts["premise"], target_ids, device=device
        )
        baseline_corrupted = mean_target_logprob(
            tokenizer, model, prompts["corrupted"], target_ids, device=device
        )

        for layer in layers:
            resolved = resolve_layer_index(model, layer)
            vpath = vector_path(pair_id, resolved, output_dir)
            if not vpath.exists():
                if verbose:
                    print(f"missing vectors {vpath}; run extract first")
                continue

            vecs = load_vectors(vpath)
            dim_indices = vecs["dim_indices"]
            # Causal test: steer the *corrupted* prompt and ask whether Y recovers.
            patch_prompt = prompts["corrupted"]
            prompt_len = len(tokenizer.encode(patch_prompt, add_special_tokens=False))
            token_start = max(0, prompt_len - 1)

            for vector_type, key in (("trad", "v_trad"), ("ect", "v_ect")):
                steering = vecs[key]
                for alpha in alphas:
                    key_tuple = (pair_id, resolved, vector_type, float(alpha), "corrupted")
                    if key_tuple in done:
                        continue

                    fired: list[int] = []
                    with patch_residual_stream(
                        model,
                        resolved,
                        steering,
                        alpha=float(alpha),
                        dim_indices=dim_indices,
                        token_start=token_start,
                        fired=fired,
                    ):
                        patched_logprob = mean_target_logprob(
                            tokenizer,
                            model,
                            patch_prompt,
                            target_ids,
                            device=device,
                        )
                    if not fired:
                        raise RuntimeError(
                            f"patch hook did not fire at layer {resolved} "
                            f"(pair={pair_id}, {vector_type}, α={alpha})"
                        )

                    rec = {
                        "pair_id": pair_id,
                        "premise": row.get("premise"),
                        "hypothesis": hypothesis,
                        "target_word": target_word,
                        "target_token_id": target_tid,
                        "target_token_str": target_piece,
                        "label": row.get("label"),
                        "source": row.get("source"),
                        "layer_index": resolved,
                        "alpha": float(alpha),
                        "vector_type": vector_type,
                        "patch_input": "corrupted",
                        "top_k_dims": int(vecs["top_k_dims"].item()),
                        "baseline_logprob_clean": baseline_clean,
                        "baseline_logprob_corrupted": baseline_corrupted,
                        "patched_logprob": patched_logprob,
                        "delta_logprob": patched_logprob - baseline_corrupted,
                        "delta_vs_clean": patched_logprob - baseline_clean,
                        "delta_vs_corrupted": patched_logprob - baseline_corrupted,
                        "norm_trad": float(vecs["norm_trad"]) if "norm_trad" in vecs else None,
                        "norm_ect_raw": float(vecs["norm_ect_raw"]) if "norm_ect_raw" in vecs else None,
                        "norm_ect_slice": float(vecs["norm_ect_slice"]) if "norm_ect_slice" in vecs else None,
                        "norm_trad_slice": float(vecs["norm_trad_slice"]) if "norm_trad_slice" in vecs else None,
                        "model_id": MODEL_ID,
                    }
                    buf.append(rec)
                    done.add(key_tuple)

                    if verbose and len(buf) == 1:
                        print(
                            f"patch {pair_id} L={resolved} {vector_type} α={alpha} "
                            f"on corrupted  Δ_corr={rec['delta_vs_corrupted']:.4f} "
                            f"hooks={len(fired)}"
                        )

                    if len(buf) >= save_every:
                        _append_jsonl(jsonl_out, buf)
                        buf.clear()
                        _rewrite_results_csv(jsonl_out, csv_out)

    if buf:
        _append_jsonl(jsonl_out, buf)
    _rewrite_results_csv(jsonl_out, csv_out)

    if verbose:
        print(f"Results → {csv_out}")


def run_experiment(
    *,
    pairs_path: Path | str = DEFAULT_PAIRS_JSONL,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    model_id: str = MODEL_ID,
    device: str = DEVICE,
    dtype_name: str = DTYPE,
    layers: Iterable[int] | None = None,
    alphas: Iterable[float] | None = None,
    top_k_dims: int = TOP_K_DIMS,
    max_pairs: int | None = None,
    extract: bool = True,
    patch: bool = True,
    plot: bool = True,
    resume: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = Path(pairs_path)

    if not pairs_path.exists():
        raise FileNotFoundError(
            f"SNLI pairs not found: {pairs_path}\n"
            "Run: sbatch src/aim2_SNLI_data_preperation/run_prepare_snli.slurm"
        )

    pairs = load_pairs(pairs_path)
    if max_pairs is not None:
        pairs = pairs[: max_pairs]

    layer_list = list(layers if layers is not None else DEFAULT_LAYERS)
    alpha_list = list(alphas if alphas is not None else DEFAULT_ALPHAS)

    if verbose:
        print(f"model_id={model_id}")
        print(f"pairs={pairs_path} (n={len(pairs)})")
        print(f"layers={layer_list} alphas={alpha_list} top_k={top_k_dims}")

    tokenizer, model, device_t = load_model_and_tokenizer(
        model_id=model_id, device=device, dtype_name=dtype_name
    )

    if extract:
        extract_vectors_for_pairs(
            pairs,
            tokenizer=tokenizer,
            model=model,
            device=device_t,
            layers=layer_list,
            output_dir=output_dir,
            top_k_dims=top_k_dims,
            verbose=verbose,
        )

    if patch:
        evaluate_patch_sweep(
            pairs,
            tokenizer=tokenizer,
            model=model,
            device=device_t,
            layers=layer_list,
            alphas=alpha_list,
            output_dir=output_dir,
            resume=resume,
            verbose=verbose,
        )

    figure_paths: list[str] = []
    if plot and results_csv(output_dir).exists():
        from .plot_patching import plot_patching_results

        figure_paths = [str(p) for p in plot_patching_results(results_csv(output_dir), output_dir)]
        if verbose:
            for path in figure_paths:
                print(f"Figure → {path}")

    return {
        "n_pairs": len(pairs),
        "layers": layer_list,
        "alphas": alpha_list,
        "output_dir": str(output_dir),
        "results_csv": str(results_csv(output_dir)),
        "figures": figure_paths,
    }


def _parse_layers(s: str) -> list[int]:
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aim-2 entailment: Jacobian vectors + activation patching sweep."
    )
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--dtype", default=DTYPE)
    parser.add_argument("--layers", type=str, default=None, help="e.g. 10-20 or 10,12,14")
    parser.add_argument(
        "--alphas",
        type=str,
        default=None,
        help="colon/comma/space-separated (prefer ':' with sbatch --export)",
    )
    parser.add_argument("--top-k-dims", type=int, default=TOP_K_DIMS)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--patch-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="Skip model; plot existing CSV.")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.plot_only:
        from .plot_patching import plot_patching_results

        csv_path = results_csv(args.output_dir)
        if not csv_path.exists():
            raise SystemExit(f"Results CSV not found: {csv_path}")
        paths = plot_patching_results(csv_path, args.output_dir)
        for path in paths:
            print(f"Figure → {path}")
        return

    layers = _parse_layers(args.layers) if args.layers else None
    alphas = parse_alphas(args.alphas) if args.alphas else None

    extract = not args.patch_only
    patch = not args.extract_only

    run_experiment(
        pairs_path=args.pairs,
        output_dir=args.output_dir,
        model_id=args.model_id,
        device=args.device,
        dtype_name=args.dtype,
        layers=layers,
        alphas=alphas,
        top_k_dims=args.top_k_dims,
        max_pairs=args.max_pairs,
        extract=extract,
        patch=patch,
        plot=not args.no_plot,
        resume=not args.no_resume,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
