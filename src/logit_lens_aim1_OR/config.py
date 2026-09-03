"""Defaults for Aim-1 OR middle-layer logit-lens extraction + colimit metrics."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent

# Sibling AND middle-layer cache root (P_A / P_B inherited per layer).
AND_LOGIT_LENS_OUTPUT_DIR = SRC_DIR / "logit_lens_aim1_AND" / "output"

MODEL_ID = os.getenv("LOGIT_LENS_MODEL", "Qwen/Qwen3.5-2B")

# Match AND logit-lens Top-K so cached supports stay comparable.
TOP_K = int(os.getenv("LOGIT_LENS_TOP_K", "4096"))

# Transformer-block index for the logit lens (0 .. n_layers-1).
# None / unset → middle layer (n_layers // 2), resolved after the model loads.
_layer_env = os.getenv("LOGIT_LENS_LAYER", "").strip()
LAYER_INDEX: int | None = int(_layer_env) if _layer_env else None

DEVICE = os.getenv("LOGIT_LENS_DEVICE", "auto")
DTYPE = os.getenv("LOGIT_LENS_DTYPE", "bfloat16")

# Apply the model's final RMSNorm/LayerNorm before lm_head (standard logit lens).
APPLY_FINAL_NORM = os.getenv("LOGIT_LENS_APPLY_NORM", "1") not in {"0", "false", "False"}

DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "output"

SAVE_EVERY = 25


def layer_output_dir(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    """Per-layer output root: output/layer_{L}/."""
    return Path(base) / f"layer_{int(layer_index)}"


def cached_distributions_for_layer(layer_index: int) -> Path:
    """AND sibling cache for the same layer L."""
    return (
        AND_LOGIT_LENS_OUTPUT_DIR
        / f"layer_{int(layer_index)}"
        / "extracted_distributions_top4096.jsonl"
    )


def distributions_jsonl_for_layer(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return layer_output_dir(layer_index, base) / f"extracted_distributions_top{TOP_K}.jsonl"


def metrics_jsonl_for_layer(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return layer_output_dir(layer_index, base) / "composition_metrics_or_max.jsonl"


def metrics_csv_for_layer(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return layer_output_dir(layer_index, base) / "composition_metrics_or_max.csv"


def rank_diff_npz_for_layer(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return layer_output_dir(layer_index, base) / "rank_diff_summary_q05_q95.npz"


def figures_dir_for_layer(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return layer_output_dir(layer_index, base) / "figures_rank_diff"
