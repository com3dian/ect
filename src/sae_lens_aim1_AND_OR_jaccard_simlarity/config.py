"""Defaults for Aim-1 SAE Jaccard (set-theory) AND/OR experiment."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent

DEFAULT_CORPUS_JSONL = (
    SRC_DIR
    / "vocabulary_gen_aim1_AND"
    / "output"
    / "orthogonality_spectrum_corpus.jsonl"
)

MODEL_ID = os.getenv("SAE_MODEL", "google/gemma-2-2b")
SAE_RELEASE = os.getenv("SAE_RELEASE", "gemma-scope-2b-pt-res-canonical")
SAE_WIDTH = os.getenv("SAE_WIDTH", "16k")
SAE_L0 = os.getenv("SAE_L0", "71")

_layer_env = os.getenv("SAE_LAYER", "14").strip()
LAYER_INDEX: int = int(_layer_env) if _layer_env else 14


def sae_id_for_layer(layer_index: int = LAYER_INDEX, width: str = SAE_WIDTH) -> str:
    release = SAE_RELEASE.lower()
    if release.endswith("canonical") or "/canonical" in release:
        return f"layer_{int(layer_index)}/width_{width}/canonical"
    return f"layer_{int(layer_index)}/width_{width}/average_l0_{SAE_L0}"


SAE_ID = os.getenv("SAE_ID", sae_id_for_layer())

DEVICE = os.getenv("SAE_DEVICE", "auto")
DTYPE = os.getenv("SAE_DTYPE", "bfloat16")

# Strict absolute activation threshold for discrete active sets (README: theta = 1.0).
ACTIVATION_THRESHOLD = float(os.getenv("SAE_JACCARD_THRESHOLD", "1.0"))

DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "output"
SAVE_EVERY = 25


def layer_output_dir(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return Path(base) / f"layer_{int(layer_index)}"


def metrics_jsonl_for_layer(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return layer_output_dir(layer_index, base) / "results_sae_jaccard.jsonl"


def metrics_csv_for_layer(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return layer_output_dir(layer_index, base) / "results_sae_jaccard.csv"


def metrics_json_for_layer(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return layer_output_dir(layer_index, base) / "results_sae_jaccard.json"


def category_summary_csv_for_layer(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return layer_output_dir(layer_index, base) / "category_summary.csv"


def figures_dir_for_layer(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return layer_output_dir(layer_index, base) / "figures_jaccard"
