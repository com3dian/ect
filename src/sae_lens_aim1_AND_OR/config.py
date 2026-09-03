"""Defaults for Aim-1 SAE-lens AND/OR composition experiment."""

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

# Gemma Scope SAEs are trained on Gemma 2; keep model/SAE paired.
MODEL_ID = os.getenv("SAE_MODEL", "google/gemma-2-2b")

SAE_RELEASE = os.getenv("SAE_RELEASE", "gemma-scope-2b-pt-res-canonical")
SAE_WIDTH = os.getenv("SAE_WIDTH", "16k")

# Residual-stream SAE layer (Gemma-2-2B has 26 layers: 0..25).
_layer_env = os.getenv("SAE_LAYER", "14").strip()
LAYER_INDEX: int = int(_layer_env) if _layer_env else 14


def sae_id_for_layer(layer_index: int = LAYER_INDEX, width: str = SAE_WIDTH) -> str:
    return f"layer_{int(layer_index)}/width_{width}/canonical"


SAE_ID = os.getenv("SAE_ID", sae_id_for_layer())

DEVICE = os.getenv("SAE_DEVICE", "auto")
DTYPE = os.getenv("SAE_DTYPE", "bfloat16")

# Treat feature activations above this as "active" for Jaccard.
JACCARD_THRESHOLD = float(os.getenv("SAE_JACCARD_THRESHOLD", "1e-6"))

DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "output"
SAVE_EVERY = 25

# Sparse Top-K SAE features kept for rank-diff plots (dictionary width is 16k).
TOP_K_FEATURES = int(os.getenv("SAE_TOP_K_FEATURES", "256"))


def layer_output_dir(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return Path(base) / f"layer_{int(layer_index)}"


def metrics_jsonl_for_layer(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return layer_output_dir(layer_index, base) / "sae_composition_metrics.jsonl"


def metrics_csv_for_layer(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return layer_output_dir(layer_index, base) / "sae_composition_metrics.csv"


def category_summary_csv_for_layer(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return layer_output_dir(layer_index, base) / "category_summary.csv"


def figures_tvd_dir_for_layer(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return layer_output_dir(layer_index, base) / "figures_tvd"


def distributions_jsonl_for_layer(layer_index: int, base: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    return layer_output_dir(layer_index, base) / "sae_feature_distributions.jsonl"


def rank_diff_npz_for_layer(
    layer_index: int,
    op: str,
    base: Path | str = DEFAULT_OUTPUT_DIR,
) -> Path:
    return layer_output_dir(layer_index, base) / f"rank_diff_{op}_summary_q05_q95.npz"


def figures_rank_diff_dir_for_layer(
    layer_index: int,
    op: str,
    base: Path | str = DEFAULT_OUTPUT_DIR,
) -> Path:
    return layer_output_dir(layer_index, base) / "figures_rank_diff" / op
