"""ConceptNet entailment extraction settings for Aim 2."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PACKAGE_DIR / "data"
DEFAULT_OUTPUT_JSONL = DEFAULT_DATA_DIR / "conceptnet_entailment.jsonl"
DEFAULT_COUNTS_JSON = DEFAULT_DATA_DIR / "conceptnet_entailment_counts.json"
DEFAULT_ASSERTIONS_GZ = DEFAULT_DATA_DIR / "conceptnet-assertions-5.7.0.csv.gz"

CONCEPTNET_ASSERTIONS_URL = (
    "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/"
    "conceptnet-assertions-5.7.0.csv.gz"
)

# Strict entailment edges only (ConceptNet category 1).
ENTAILMENT_RELATIONS = frozenset({"/r/IsA", "/r/PartOf"})

DEFAULT_MAX_PAIRS = 1000
DEFAULT_MIN_WEIGHT = 1.0
CORRUPTED_PREMISE = "It is the case that:"
