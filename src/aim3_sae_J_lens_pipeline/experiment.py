"""
Aim-3 SAE + J-lens pipeline: GPU extract then CPU tropical analysis.

Default: extract SAE/Jacobian tensors, free VRAM, then run Neighbor-Joining
and tropical triangle-inequality checks.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import torch

from .analyze_tropical_topology import run_analysis
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
    TRIANGLE_ATOL,
    sae_id_for_layer,
)
from .extract import release_gpu, run_extraction


def run_pipeline(
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
    types: list[str] | None = None,
    resume: bool = True,
    save_every: int = SAVE_EVERY,
    top_k_sae: int = TOP_K_SAE,
    top_k_vocab: int = TOP_K_VOCAB,
    top_k_resid: int = TOP_K_RESID,
    compute_jacobian: bool = True,
    extract: bool = True,
    analyze: bool = True,
    atol: float = TRIANGLE_ATOL,
    verbose: bool = True,
) -> dict[str, Any]:
    if sae_id is None:
        sae_id = sae_id_for_layer(layer_index)

    extract_stats: dict[str, Any] | None = None
    if extract:
        extract_stats = run_extraction(
            corpus_path=corpus_path,
            model_id=model_id,
            sae_release=sae_release,
            sae_id=sae_id,
            layer_index=layer_index,
            device=device,
            dtype_name=dtype_name,
            output_dir=output_dir,
            max_items=max_items,
            types=types,
            resume=resume,
            save_every=save_every,
            top_k_sae=top_k_sae,
            top_k_vocab=top_k_vocab,
            top_k_resid=top_k_resid,
            compute_jacobian=compute_jacobian,
            verbose=verbose,
        )
        if verbose:
            print("Clearing GPU VRAM before offline analysis …", flush=True)
        release_gpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    analysis_stats: dict[str, Any] | None = None
    if analyze:
        analysis_stats = run_analysis(
            layer_index=layer_index,
            output_dir=output_dir,
            atol=atol,
            verbose=verbose,
        )
    return {"extract": extract_stats, "analysis": analysis_stats}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aim-3 SAE + J-lens tropical geometry extract/analyze."
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
    p.add_argument("--types", default="A,B,C", help="Comma-separated type keys.")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--save-every", type=int, default=SAVE_EVERY)
    p.add_argument("--top-k-sae", type=int, default=TOP_K_SAE)
    p.add_argument("--top-k-vocab", type=int, default=TOP_K_VOCAB)
    p.add_argument("--top-k-resid", type=int, default=TOP_K_RESID)
    p.add_argument("--no-jacobian", action="store_true")
    p.add_argument("--extract-only", action="store_true")
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--atol", type=float, default=TRIANGLE_ATOL)
    p.add_argument("-q", "--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    type_keys = [t.strip().upper() for t in args.types.split(",") if t.strip()]
    run_pipeline(
        corpus_path=args.corpus,
        model_id=args.model_id,
        sae_release=args.sae_release,
        sae_id=args.sae_id,
        layer_index=args.layer_index,
        device=args.device,
        dtype_name=args.dtype,
        output_dir=args.output_dir,
        max_items=args.max_items,
        types=type_keys,
        resume=not args.no_resume,
        save_every=args.save_every,
        top_k_sae=args.top_k_sae,
        top_k_vocab=args.top_k_vocab,
        top_k_resid=args.top_k_resid,
        compute_jacobian=not args.no_jacobian,
        extract=not args.analyze_only,
        analyze=not args.extract_only,
        atol=args.atol,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
