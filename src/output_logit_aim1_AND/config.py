"""Defaults for Aim-1 logit-lens extraction + min/mean/max metrics."""

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

# README asked for Qwen3.5-1.5B (not on HF). Use the nearest official Qwen3.5:
MODEL_ID = os.getenv("LOGIT_LENS_MODEL", "Qwen/Qwen3.5-2B")

# Keep only the top-K softmax probabilities (README default).
TOP_K = int(os.getenv("LOGIT_LENS_TOP_K", "4096"))

# Device: "cuda", "cpu", or "auto"
DEVICE = os.getenv("LOGIT_LENS_DEVICE", "auto")

# Load dtype hint for GPU runs ("bfloat16", "float16", "float32")
DTYPE = os.getenv("LOGIT_LENS_DTYPE", "bfloat16")

# Outputs
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "output"
DEFAULT_DISTRIBUTIONS_JSONL = DEFAULT_OUTPUT_DIR / "extracted_distributions_top4096.jsonl"
DEFAULT_METRICS_CSV = DEFAULT_OUTPUT_DIR / "composition_metrics_min_mean_max.csv"
DEFAULT_METRICS_JSONL = DEFAULT_OUTPUT_DIR / "composition_metrics_min_mean_max.jsonl"

# How often to flush metrics/distributions to disk during a long GPU run
SAVE_EVERY = 25
