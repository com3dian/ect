"""Defaults for Aim-2 contextual entailment via activation patching."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent

# README specifies Qwen3.5-1.5B; use nearest HF checkpoint (override via AIM2_MODEL).
MODEL_ID = os.getenv("AIM2_MODEL", "Qwen/Qwen3.5-2B")

DEVICE = os.getenv("AIM2_DEVICE", "auto")
DTYPE = os.getenv("AIM2_DTYPE", "bfloat16")

DEFAULT_DATA_DIR = PACKAGE_DIR / "data"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "output"

# Prepared by src/aim2_SNLI_data_preperation (entailment, capped at 1000).
DEFAULT_SNLI_JSONL = (
    PACKAGE_DIR.parent / "aim2_SNLI_data_preperation" / "data" / "snli_entailment.jsonl"
)
DEFAULT_PAIRS_JSONL = Path(os.getenv("AIM2_PAIRS", str(DEFAULT_SNLI_JSONL)))

# Jacobian / vector storage: keep Top-K residual dimensions (README: 1000).
TOP_K_DIMS = int(os.getenv("AIM2_TOP_K_DIMS", "1000"))

# Layer sweep (README: e.g. 10–20).
_layer_start = int(os.getenv("AIM2_LAYER_START", "10"))
_layer_end = int(os.getenv("AIM2_LAYER_END", "20"))
DEFAULT_LAYERS = list(range(_layer_start, _layer_end + 1))

# Alpha sweep (README: 0.1 – 5.0).
def parse_alphas(raw: str | None = None) -> list[float]:
    """Parse alphas; sbatch --export splits on commas, so ':' / space are also accepted."""
    text = (raw if raw is not None else os.getenv("AIM2_ALPHAS", "0.1:0.5:1.0:2.0:5.0"))
    parts = [p for p in text.replace(",", " ").replace(":", " ").split() if p]
    return [float(p) for p in parts]


DEFAULT_ALPHAS = parse_alphas()

SAVE_EVERY = int(os.getenv("AIM2_SAVE_EVERY", "10"))


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
