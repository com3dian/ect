"""Typicality prompt templates for logical disjunction (OR)."""

from __future__ import annotations

TEMPLATE_A_OR_B = (
    "Describe the typical properties and context of an entity that is either "
    "a {word_a} or a {word_b}:"
)


def build_or_prompt(word_a: str, word_b: str) -> str:
    """Build the disjunction prompt P_{A∨B} for one concept pair."""
    return TEMPLATE_A_OR_B.format(word_a=word_a, word_b=word_b)
