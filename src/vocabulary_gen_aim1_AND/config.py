"""SURF AI Hub / WiLLMa client settings and generation defaults."""

from __future__ import annotations

import os
from pathlib import Path

# OpenAI-compatible SURF AI Hub endpoint (user-specified)
BASE_URL = os.getenv("WILLMA_BASE_URL", "https://api.willma.surf.nl/v0")
MODEL = os.getenv("WILLMA_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")

# Prefer WILLMA_API_KEY; fall back to OPENAI_API_KEY for OpenAI client compatibility
API_KEY_ENV_VARS = ("WILLMA_API_KEY", "OPENAI_API_KEY")

# Batch size per API call (README: 50 pairs to avoid output token cut-offs)
PAIRS_PER_BATCH = 50

# Target corpus size per orthogonality category
TARGET_PAIRS_PER_CATEGORY = 750  # midpoint of 500–1000
MIN_PAIRS_PER_CATEGORY = 500
MAX_PAIRS_PER_CATEGORY = 1000

# Generation behaviour
SHUFFLE_SEEDS = True
TEMPERATURE = 0.8
MAX_TOKENS = 8192
REQUEST_TIMEOUT_S = 180.0
MAX_RETRIES = 3

# Output
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "output"
DEFAULT_OUTPUT_JSONL = DEFAULT_OUTPUT_DIR / "orthogonality_spectrum_corpus.jsonl"
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "orthogonality_spectrum_corpus.csv"

CATEGORY_NAMES = {
    1: "Redundant / Strict Entailment",
    2: "Typical Composition",
    3: "Uncommon / Orthogonal",
    4: "Surreal / Out-of-Distribution",
    5: "Mutually Exclusive / Paradoxical",
}


def resolve_api_key(explicit: str | None = None) -> str:
    """Resolve API key from an explicit value or environment variables."""
    if explicit:
        return explicit.strip()
    for name in API_KEY_ENV_VARS:
        value = os.getenv(name)
        if value:
            return value.strip()
    raise EnvironmentError(
        "Missing API key. Set WILLMA_API_KEY (preferred) or OPENAI_API_KEY, "
        "or pass api_key=... explicitly."
    )
