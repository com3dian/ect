"""Typicality prompt templates (same as logit / SAE Aim-1 packages)."""

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


def build_prompts(word_a: str, word_b: str) -> dict[str, str]:
    return {
        "P_A": TEMPLATE_A.format(word_a=word_a),
        "P_B": TEMPLATE_B.format(word_b=word_b),
        "P_AND": TEMPLATE_AND.format(word_a=word_a, word_b=word_b),
        "P_OR": TEMPLATE_OR.format(word_a=word_a, word_b=word_b),
    }
