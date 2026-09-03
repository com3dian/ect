"""Defaults for Aim-3 SAE + J-lens tropical geometry pipeline."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent

DEFAULT_CORPUS_JSON = (
    SRC_DIR / "aim3_data_prep" / "data" / "tropical_geometry_corpus.json"
)

# Gemma Scope SAEs are trained on Gemma 2; keep model/SAE paired.
MODEL_ID = os.getenv("SAE_MODEL", "google/gemma-2-2b")
SAE_RELEASE = os.getenv("SAE_RELEASE", "gemma-scope-2b-pt-res-canonical")
SAE_WIDTH = os.getenv("SAE_WIDTH", "16k")

_layer_env = os.getenv("SAE_LAYER", "14").strip()
LAYER_INDEX: int = int(_layer_env) if _layer_env else 14


def sae_id_for_layer(layer_index: int = LAYER_INDEX, width: str = SAE_WIDTH) -> str:
    return f"layer_{int(layer_index)}/width_{width}/canonical"


SAE_ID = os.getenv("SAE_ID", sae_id_for_layer())

DEVICE = os.getenv("SAE_DEVICE", "auto")
DTYPE = os.getenv("SAE_DTYPE", "bfloat16")

# Top-K truncation of the softmax manifold (drop long-tail vocab mass).
TOP_K_VOCAB = int(os.getenv("AIM3_TOP_K_VOCAB", "32"))
# Column truncation of J_ℓ over residual dimensions (VRAM / disk).
TOP_K_RESID = int(os.getenv("AIM3_TOP_K_RESID", "256"))
# Sparse SAE features kept per waypoint.
TOP_K_SAE = int(os.getenv("AIM3_TOP_K_SAE", "256"))

TROPICAL_EPS = float(os.getenv("AIM3_TROPICAL_EPS", "1e-12"))
TRIANGLE_ATOL = float(os.getenv("AIM3_TRIANGLE_ATOL", "1e-4"))

DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "output"
SAVE_EVERY = int(os.getenv("AIM3_SAVE_EVERY", "10"))


def layer_output_dir(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return Path(base) / f"layer_{int(layer_index)}"


def tensors_dir(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return layer_output_dir(layer_index, base) / "tensors"


def tensor_path(
    item_id: str,
    waypoint: str,
    layer_index: int,
    base: Path | str = DEFAULT_OUTPUT_DIR,
) -> Path:
    safe_wp = waypoint.replace("/", "_")
    return tensors_dir(layer_index, base) / f"{item_id}__{safe_wp}.safetensors"


def manifest_jsonl(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return layer_output_dir(layer_index, base) / "extract_manifest.jsonl"


def tropical_logprobs_jsonl(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return layer_output_dir(layer_index, base) / "tropical_logprobs.jsonl"


def distances_jsonl(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return layer_output_dir(layer_index, base) / "tropical_distances.jsonl"


def nj_newick_jsonl(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return layer_output_dir(layer_index, base) / "neighbor_joining_newick.jsonl"


def triangle_csv(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return layer_output_dir(layer_index, base) / "triangle_inequality.csv"


def analysis_summary_json(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return layer_output_dir(layer_index, base) / "tropical_analysis_summary.json"


def figures_dir_for_layer(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return layer_output_dir(layer_index, base) / "figures"
