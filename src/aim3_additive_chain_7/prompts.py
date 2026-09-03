"""Prompts for chain-rule sanity and short-token Markov cloze (v7)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


def type_key(row: dict[str, Any]) -> str:
    name = str(row.get("structure_type", "")).strip()
    if name.startswith("Type A") or name == "A":
        return "A"
    if name.startswith("Type B") or name == "B":
        return "B"
    if name.startswith("Type C") or name == "C":
        return "C"
    raise ValueError(f"Unknown structure_type={name!r}")


def item_id(row: dict[str, Any], index: int) -> str:
    return f"{type_key(row)}_{int(index):05d}"


def type_a_nodes(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return tuple(str(row[f"node_{i}"]).strip() for i in range(1, 5))  # type: ignore[return-value]


def permute_nodes(
    nodes: tuple[str, str, str, str],
    variant: str,
) -> tuple[str, str, str, str]:
    n1, n2, n3, n4 = nodes
    key = variant.strip().lower()
    if key == "natural":
        return nodes
    if key in ("shuffled_middles", "shuffle_middles", "shuffled"):
        return (n1, n3, n2, n4)
    if key in ("reversed", "reverse"):
        return (n4, n3, n2, n1)
    raise ValueError(f"Unknown variant={variant!r}")


@dataclass(frozen=True)
class ClozeQuery:
    name: str
    prefix: str
    target: str
    cond_concept: str
    target_concept: str


def chain_rule_stem(n1: str) -> str:
    """Stem continued by n2, then n3, then n4 (one concept per line)."""
    return (
        "Taxonomic refinement chain. After the broad concept, write each more "
        "specific concept on its own line.\n"
        f'Broad concept: "{n1.strip()}"\n'
    )


def chain_rule_line(concept: str) -> str:
    return f" {concept.strip()}"


def markov_hop_prefix(cond_concept: str, *, hop: str = "next") -> str:
    cond = cond_concept.strip()
    if hop == "far":
        return (
            "Taxonomic refinement in English.\n"
            f'Broad concept: "{cond}"\n'
            "A much more specific instance of that concept is"
        )
    return (
        "Taxonomic refinement in English.\n"
        f'Concept: "{cond}"\n'
        "A more specific kind of that concept is"
    )


def build_type_a_markov_queries(
    nodes: tuple[str, str, str, str],
    *,
    variant: str = "natural",
) -> list[ClozeQuery]:
    a, b, c, d = permute_nodes(nodes, variant)
    return [
        ClozeQuery("n1_n2", markov_hop_prefix(a, hop="next"), b, a, b),
        ClozeQuery("n2_n3", markov_hop_prefix(b, hop="next"), c, b, c),
        ClozeQuery("n3_n4", markov_hop_prefix(c, hop="next"), d, c, d),
        ClozeQuery("n1_n4", markov_hop_prefix(a, hop="far"), d, a, d),
    ]


def parse_variants(spec: str | Iterable[str] | None) -> list[str]:
    if spec is None:
        return ["natural", "shuffled_middles"]
    if isinstance(spec, str):
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        return parts or ["natural"]
    return [str(p).strip() for p in spec if str(p).strip()]
