"""Aim-3 additive-chain v2: shared-trajectory tropical geometry (Type A focus)."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent

DEFAULT_CORPUS_JSON = (
    SRC_DIR / "aim3_data_prep" / "data" / "tropical_geometry_corpus.json"
)

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

TOP_K_SAE = int(os.getenv("AIM3_TOP_K_SAE", "256"))
TROPICAL_EPS = float(os.getenv("AIM3_TROPICAL_EPS", "1e-12"))

DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "output"
SAVE_EVERY = int(os.getenv("AIM3_SAVE_EVERY", "25"))


def layer_output_dir(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return Path(base) / f"layer_{int(layer_index)}"


def distances_jsonl(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return layer_output_dir(layer_index, base) / "trajectory_distances.jsonl"


def summary_json(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return layer_output_dir(layer_index, base) / "trajectory_summary.json"


def figures_dir_for_layer(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return layer_output_dir(layer_index, base) / "figures"
