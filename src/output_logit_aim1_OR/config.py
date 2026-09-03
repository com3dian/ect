"""Defaults for Aim-1 OR logit-lens extraction + colimit (max) metrics."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent

# Cached P_A / P_B from the AND extraction run (README: top8192 when available).
DEFAULT_CACHED_DISTRIBUTIONS_JSONL = (
    SRC_DIR
    / "output_logit_aim1_AND"
    / "output"
    / "extracted_distributions_top4096.jsonl"
)

# README asked for Qwen3.5-1.5B (not on HF). Use the nearest official Qwen3.5:
MODEL_ID = os.getenv("LOGIT_LENS_MODEL", "Qwen/Qwen3.5-2B")

# Top-K for the new P_{A∨B} extraction (README default: 8192).
TOP_K = int(os.getenv("LOGIT_LENS_TOP_K", "8192"))

DEVICE = os.getenv("LOGIT_LENS_DEVICE", "auto")
DTYPE = os.getenv("LOGIT_LENS_DTYPE", "bfloat16")

DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "output"
DEFAULT_DISTRIBUTIONS_JSONL = DEFAULT_OUTPUT_DIR / f"extracted_distributions_top{TOP_K}.jsonl"
DEFAULT_METRICS_CSV = DEFAULT_OUTPUT_DIR / "composition_metrics_or_max.csv"
DEFAULT_METRICS_JSONL = DEFAULT_OUTPUT_DIR / "composition_metrics_or_max.jsonl"
DEFAULT_RANK_DIFF_NPZ = DEFAULT_OUTPUT_DIR / "rank_diff_summary_q05_q95.npz"
DEFAULT_FIGURES_RANK_DIFF_DIR = DEFAULT_OUTPUT_DIR / "figures_rank_diff"

SAVE_EVERY = 25
