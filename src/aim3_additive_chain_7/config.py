"""Aim-3 v7: calibration — chain-rule tautology + short-token Markov cloze."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent

DEFAULT_CORPUS_JSON = (
    SRC_DIR / "aim3_data_prep" / "data" / "tropical_geometry_corpus.json"
)

MODEL_ID = os.getenv("SAE_MODEL", "google/gemma-2-9b")
DEVICE = os.getenv("SAE_DEVICE", "auto")
DTYPE = os.getenv("SAE_DTYPE", "bfloat16")

_layer_env = os.getenv("SAE_LAYER", "14").strip()
LAYER_INDEX: int = int(_layer_env) if _layer_env else 14

TROPICAL_EPS = float(os.getenv("AIM3_TROPICAL_EPS", "1e-12"))
MAX_TARGET_TOKENS = int(os.getenv("AIM3_MAX_TARGET_TOKENS", "12"))
# Short-concept filter for Markov re-run (BPE pieces with leading space).
MAX_BPE_PER_NODE = int(os.getenv("AIM3_MAX_BPE_PER_NODE", "2"))


def model_slug(model_id: str = MODEL_ID) -> str:
    """HF id → folder-safe name, e.g. google/gemma-2-9b → gemma-2-9b."""
    return str(model_id).strip().rstrip("/").split("/")[-1].replace(" ", "_")


DEFAULT_OUTPUT_DIR = PACKAGE_DIR / f"output_{model_slug(MODEL_ID)}"
SAVE_EVERY = int(os.getenv("AIM3_SAVE_EVERY", "25"))
DEFAULT_VARIANTS = os.getenv("AIM3_VARIANTS", "natural,shuffled_middles")


def layer_output_dir(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return Path(base) / f"layer_{int(layer_index)}"


def distances_jsonl(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return layer_output_dir(layer_index, base) / "calibration_distances.jsonl"


def summary_json(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return layer_output_dir(layer_index, base) / "calibration_summary.json"


def figures_dir_for_layer(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return layer_output_dir(layer_index, base) / "figures"
