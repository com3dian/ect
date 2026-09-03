"""Prompt templates for clean / corrupted premise and isolated hypothesis Y."""

from __future__ import annotations

from typing import Any, Mapping


# Neutral corrupted baseline (no entailment content).
CORRUPTED_PREMISE = "It is the case that:"


def premise_prompt(premise: str) -> str:
    """Clean premise X — Jacobian is evaluated in this context."""
    return premise.strip()


def corrupted_prompt(_premise: str = "") -> str:
    """Corrupted / neutral carrier prompt for patching baseline."""
    return CORRUPTED_PREMISE


def hypothesis_prompt(hypothesis: str) -> str:
    """Isolated target Y for h_Y extraction."""
    hyp = hypothesis.strip()
    if hyp.endswith("."):
        return hyp
    return f"{hyp}."


def build_prompt_bundle(row: Mapping[str, Any]) -> dict[str, str]:
    premise = str(row["premise"])
    hypothesis = str(row["hypothesis"])
    return {
        "premise": premise_prompt(premise),
        "corrupted": corrupted_prompt(premise),
        "hypothesis": hypothesis_prompt(hypothesis),
    }
