"""SURF AI Hub / WiLLMa settings and Aim-3 generation defaults."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_SRC = PACKAGE_DIR.parent

# Reuse the Aim-1 WiLLMa / SURF credentials (do not duplicate secrets here).
_AIM1_ENV = REPO_SRC / "vocabulary_gen_aim1_AND" / ".env"
_LOCAL_ENV = PACKAGE_DIR / ".env"
load_dotenv(_AIM1_ENV)
load_dotenv(_LOCAL_ENV)  # optional local overrides
load_dotenv()  # repo-root .env if present

# OpenAI-compatible SURF AI Hub endpoint
BASE_URL = os.getenv("WILLMA_BASE_URL", "https://api.willma.surf.nl/v0")
MODEL = os.getenv("WILLMA_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")

API_KEY_ENV_VARS = ("WILLMA_API_KEY", "OPENAI_API_KEY")

# Batch size per API call (README meta-prompt: 50 structures)
ITEMS_PER_BATCH = 50

# Default: a few batches per topological type
BATCHES_PER_TYPE = 3
TARGET_ITEMS_PER_TYPE = ITEMS_PER_BATCH * BATCHES_PER_TYPE  # 150

TEMPERATURE = 0.85
MAX_TOKENS = 8192
REQUEST_TIMEOUT_S = 180.0
MAX_RETRIES = 4
CONCURRENCY = 1  # sequential by default; raise for parallel type batches

# Always require OpenAI-style structured outputs (json_schema + strict).
FORCE_STRUCTURED_OUTPUT = True
# Qwen3 thinking mode often leaves message.content empty / burns max_tokens.
DISABLE_THINKING = os.getenv("WILLMA_DISABLE_THINKING", "1").strip() not in {
    "0",
    "false",
    "False",
    "no",
}

DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "data"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_DIR / "tropical_geometry_corpus.json"
DEFAULT_OUTPUT_JSONL = DEFAULT_OUTPUT_DIR / "tropical_geometry_corpus.jsonl"


def resolve_api_key(explicit: str | None = None) -> str:
    """Resolve API key from an explicit value or environment variables."""
    if explicit:
        return explicit.strip()
    for name in API_KEY_ENV_VARS:
        value = os.getenv(name)
        if value:
            return value.strip()
    raise EnvironmentError(
        "Missing API key. Set WILLMA_API_KEY (preferred) or OPENAI_API_KEY "
        f"(loaded from {_AIM1_ENV} when present), or pass api_key=... explicitly."
    )
