"""Prompts for four-point Type A and triangle Type B/C."""

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


# Six undirected pairs among four nodes (i,j) with i<j.
PAIR_INDICES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
)


def pair_key(i: int, j: int) -> str:
    a, b = (int(i), int(j))
    if a > b:
        a, b = b, a
    return f"d_{a}{b}"


def cloze_prefix(cond_concept: str) -> str:
    """One template for all six pairwise conditionals (fixed design)."""
    return (
        "Taxonomic refinement in English.\n"
        f'Concept: "{cond_concept.strip()}"\n'
        "A more specific kind of that concept is"
    )


def build_type_a_six_pair_queries(
    nodes: tuple[str, str, str, str],
    *,
    variant: str = "natural",
) -> list[ClozeQuery]:
    chain = permute_nodes(nodes, variant)
    queries: list[ClozeQuery] = []
    for i, j in PAIR_INDICES:
        queries.append(
            ClozeQuery(
                name=pair_key(i, j),
                prefix=cloze_prefix(chain[i]),
                target=chain[j],
                cond_concept=chain[i],
                target_concept=chain[j],
            )
        )
    return queries


def build_type_b_queries(row: dict[str, Any]) -> list[ClozeQuery]:
    base = str(row["base_context"]).strip()
    shift = str(row["shift_context"]).strip()
    combined = str(row.get("combined_sequence", f"{base} {shift}")).strip()

    def cont_prefix(ctx: str) -> str:
        return (
            "Continue the narrative naturally.\n"
            f"Context: {ctx}\n"
            "Continuation:"
        )

    return [
        ClozeQuery("base_shift", cont_prefix(base), shift, "base", "shift"),
        ClozeQuery("shift_combined", cont_prefix(shift), combined, "shift", "combined"),
        ClozeQuery("base_combined", cont_prefix(base), combined, "base", "combined"),
    ]


def build_type_c_queries(row: dict[str, Any]) -> list[ClozeQuery]:
    entailed = str(row["entailed_concept"]).strip()
    superord = str(row["superordinate"]).strip()
    framed = str(row["framed_sequence"]).strip()

    def kind_of(x: str) -> str:
        return (
            "Lexical / taxonomic relation.\n"
            f'Concept: "{x}"\n'
            "A closely related broader or entailed concept is"
        )

    def from_frame(frame: str) -> str:
        return (
            "Given the following framed sentence, a key concept it involves is\n"
            f"Frame: {frame}\n"
            "Concept:"
        )

    return [
        ClozeQuery("entailed_superordinate", kind_of(entailed), superord, entailed, superord),
        ClozeQuery("framed_entailed", from_frame(framed), entailed, "framed", entailed),
        ClozeQuery("framed_superordinate", from_frame(framed), superord, "framed", superord),
    ]


def parse_variants(spec: str | Iterable[str] | None) -> list[str]:
    if spec is None:
        return ["natural", "shuffled_middles", "reversed"]
    if isinstance(spec, str):
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        return parts or ["natural"]
    return [str(p).strip() for p in spec if str(p).strip()]
