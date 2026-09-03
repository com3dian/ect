"""Typicality prompt templates (control variables from the README)."""

from __future__ import annotations

# Unified framing for A, B, and A∧B. All prompts end with ":" so the
# logit lens reads the model's expectation at that final colon token.
TEMPLATE_A = "Describe the typical properties and context of a {word_a}:"
TEMPLATE_B = "Describe the typical properties and context of a {word_b}:"
TEMPLATE_AB = (
    "Describe the typical properties and context of an entity that is both "
    "a {word_a} and a {word_b}:"
)


def build_prompts(word_a: str, word_b: str) -> dict[str, str]:
    """Build the three parallel prompts for one concept pair."""
    return {
        "P_A": TEMPLATE_A.format(word_a=word_a),
        "P_B": TEMPLATE_B.format(word_b=word_b),
        "P_AB": TEMPLATE_AB.format(word_a=word_a, word_b=word_b),
    }
