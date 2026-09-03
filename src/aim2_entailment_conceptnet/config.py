"""Defaults for Aim-2 contextual entailment via activation patching (ConceptNet pairs)."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent

MODEL_ID = os.getenv("AIM2_CN_MODEL", os.getenv("AIM2_MODEL", "Qwen/Qwen3.5-2B"))

DEVICE = os.getenv("AIM2_CN_DEVICE", os.getenv("AIM2_DEVICE", "auto"))
DTYPE = os.getenv("AIM2_CN_DTYPE", os.getenv("AIM2_DTYPE", "bfloat16"))

DEFAULT_DATA_DIR = PACKAGE_DIR / "data"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "output"

# Prepared by src/aim2_ConceptNet_data_preparation (IsA / PartOf entailment).
DEFAULT_CONCEPTNET_JSONL = (
    PACKAGE_DIR.parent
    / "aim2_ConceptNet_data_preparation"
    / "data"
    / "conceptnet_entailment.jsonl"
)
DEFAULT_PAIRS_JSONL = Path(
    os.getenv("AIM2_CN_PAIRS", os.getenv("AIM2_PAIRS", str(DEFAULT_CONCEPTNET_JSONL)))
)

TOP_K_DIMS = int(os.getenv("AIM2_CN_TOP_K_DIMS", os.getenv("AIM2_TOP_K_DIMS", "1000")))

_layer_start = int(os.getenv("AIM2_CN_LAYER_START", os.getenv("AIM2_LAYER_START", "10")))
_layer_end = int(os.getenv("AIM2_CN_LAYER_END", os.getenv("AIM2_LAYER_END", "20")))
DEFAULT_LAYERS = list(range(_layer_start, _layer_end + 1))


def parse_alphas(raw: str | None = None) -> list[float]:
    """Parse alphas; sbatch --export splits on commas, so ':' / space are also accepted."""
    default = os.getenv("AIM2_CN_ALPHAS", os.getenv("AIM2_ALPHAS", "0.1:0.5:1.0:2.0:5.0"))
    text = default if raw is None else raw
    parts = [p for p in text.replace(",", " ").replace(":", " ").split() if p]
    return [float(p) for p in parts]


DEFAULT_ALPHAS = parse_alphas()

SAVE_EVERY = int(os.getenv("AIM2_CN_SAVE_EVERY", os.getenv("AIM2_SAVE_EVERY", "10")))


def vectors_dir(base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return Path(base) / "vectors"


def vector_path(pair_id: str, layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return vectors_dir(base) / f"{pair_id}_L{layer_index}.safetensors"


def results_csv(base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return Path(base) / "patching_results.csv"


def results_jsonl(base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return Path(base) / "patching_results.jsonl"


def figures_dir(base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return Path(base) / "figures"
