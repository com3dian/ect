"""Aim 1 orthogonality-spectrum vocabulary generation package."""

from .config import BASE_URL, MODEL, resolve_api_key
from .domain_seed import domain_seeds_map
from .generate_corpus import ConceptPair, corpus_summary, make_client, run_generation

__all__ = [
    "BASE_URL",
    "MODEL",
    "ConceptPair",
    "corpus_summary",
    "domain_seeds_map",
    "make_client",
    "resolve_api_key",
    "run_generation",
]
