"""Concept prompts + shared syntactic baseline carrier templates."""

from __future__ import annotations

TEMPLATE_A = "Describe the typical properties and context of a {word_a}:"
TEMPLATE_B = "Describe the typical properties and context of a {word_b}:"
TEMPLATE_AND = (
    "Describe the typical properties and context of an entity that is both "
    "a {word_a} and a {word_b}:"
)
TEMPLATE_OR = (
    "Describe the typical properties and context of an entity that is either "
    "a {word_a} or a {word_b}:"
)

# Carrier templates with neutral placeholders (no concept nouns).
TEMPLATE_BASE_SINGLE = "Describe the typical properties and context of an entity:"
TEMPLATE_BASE_COMPOSED = (
    "Describe the typical properties and context of an entity that is:"
)


def build_prompts(word_a: str, word_b: str) -> dict[str, str]:
    """Build concept prompts for one pair (baselines are constant; see build_baselines)."""
    return {
        "P_A": TEMPLATE_A.format(word_a=word_a),
        "P_B": TEMPLATE_B.format(word_b=word_b),
        "P_AND": TEMPLATE_AND.format(word_a=word_a, word_b=word_b),
        "P_OR": TEMPLATE_OR.format(word_a=word_a, word_b=word_b),
    }


def build_baselines() -> dict[str, str]:
    """Shared syntactic baselines (independent of the concept pair)."""
    return {
        "P_BASE_SINGLE": TEMPLATE_BASE_SINGLE,
        "P_BASE_COMPOSED": TEMPLATE_BASE_COMPOSED,
    }
