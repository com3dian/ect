"""Defaults for Aim-1 middle-layer logit-lens extraction + min/mean/max metrics."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent

# Corpus produced by vocabulary_gen_aim1_AND
DEFAULT_CORPUS_JSONL = (
    SRC_DIR
    / "vocabulary_gen_aim1_AND"
    / "output"
    / "orthogonality_spectrum_corpus.jsonl"
)

# Same model as output-logit baseline (override with LOGIT_LENS_MODEL).
MODEL_ID = os.getenv("LOGIT_LENS_MODEL", "Qwen/Qwen3.5-2B")

# Keep only the top-K softmax probabilities.
TOP_K = int(os.getenv("LOGIT_LENS_TOP_K", "4096"))

# Transformer-block index for the logit lens (0 .. n_layers-1).
# None / unset → middle layer (n_layers // 2), resolved after the model loads.
_layer_env = os.getenv("LOGIT_LENS_LAYER", "").strip()
LAYER_INDEX: int | None = int(_layer_env) if _layer_env else None

# Device: "cuda", "cpu", or "auto"
DEVICE = os.getenv("LOGIT_LENS_DEVICE", "auto")

# Load dtype hint for GPU runs ("bfloat16", "float16", "float32")
DTYPE = os.getenv("LOGIT_LENS_DTYPE", "bfloat16")

# Apply the model's final RMSNorm/LayerNorm before lm_head (standard logit lens).
APPLY_FINAL_NORM = os.getenv("LOGIT_LENS_APPLY_NORM", "1") not in {"0", "false", "False"}

# Outputs (layer-specific subdirs are chosen at runtime once LAYER_INDEX is known)
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "output"


def layer_output_dir(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    """Per-layer output root: output/layer_{L}/."""
    return Path(base) / f"layer_{int(layer_index)}"


def distributions_jsonl_for_layer(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return layer_output_dir(layer_index, base) / "extracted_distributions_top4096.jsonl"


def metrics_jsonl_for_layer(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return layer_output_dir(layer_index, base) / "composition_metrics_min_mean_max.jsonl"


def metrics_csv_for_layer(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return layer_output_dir(layer_index, base) / "composition_metrics_min_mean_max.csv"


# How often to flush metrics/distributions to disk during a long GPU run
SAVE_EVERY = 25
