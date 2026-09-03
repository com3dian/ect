"""
Markov cloze prompts for Type A phylogenetic chains (Aim-3 v6).

Primary test (non-tautological):
  π(n2|n1) · π(n3|n2) · π(n4|n3)  ?≈  π(n4|n1)

i.e. with d_M = -ln π(·|·) (sequence logprob of the concept string):
  d(n1,n4)  ?≈  d(n1,n2)+d(n2,n3)+d(n3,n4)

Each conditional uses a short cloze that mentions only the conditioning concept
(not a shared multi-node trajectory listing). Controls permute the node order.
"""

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
    if key in ("n1_n4_n2_n3", "broken"):
        return (n1, n4, n2, n3)
    raise ValueError(f"Unknown variant={variant!r}")


@dataclass(frozen=True)
class ClozeQuery:
    """One conditional: score ``target`` as continuation of ``prefix``."""

    name: str
    prefix: str
    target: str
    cond_concept: str
    target_concept: str


def markov_hop_prefix(cond_concept: str, *, hop: str = "next") -> str:
    """
    Prefix that elicits the next, more specific concept after ``cond_concept``.

    ``hop='next'`` for consecutive refinement; ``hop='far'`` for a much more
    specific instance (used for the n1→n4 shortcut).
    """
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
    """
    Four queries for path hops + shortcut under Markov cloze.

    Order after permutation is treated as the chain (a,b,c,d):
      d(a,b), d(b,c), d(c,d), d(a,d).
    """
    a, b, c, d = permute_nodes(nodes, variant)
    return [
        ClozeQuery(
            name="n1_n2",
            prefix=markov_hop_prefix(a, hop="next"),
            target=b,
            cond_concept=a,
            target_concept=b,
        ),
        ClozeQuery(
            name="n2_n3",
            prefix=markov_hop_prefix(b, hop="next"),
            target=c,
            cond_concept=b,
            target_concept=c,
        ),
        ClozeQuery(
            name="n3_n4",
            prefix=markov_hop_prefix(c, hop="next"),
            target=d,
            cond_concept=c,
            target_concept=d,
        ),
        ClozeQuery(
            name="n1_n4",
            prefix=markov_hop_prefix(a, hop="far"),
            target=d,
            cond_concept=a,
            target_concept=d,
        ),
    ]


def build_type_b_queries(row: dict[str, Any]) -> list[ClozeQuery]:
    """Light Type B triangle on cloze-style conditionals (optional)."""
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
        ClozeQuery(
            name="base_shift",
            prefix=cont_prefix(base),
            target=shift,
            cond_concept="base",
            target_concept="shift",
        ),
        ClozeQuery(
            name="shift_combined",
            prefix=cont_prefix(shift),
            target=combined,
            cond_concept="shift",
            target_concept="combined",
        ),
        ClozeQuery(
            name="base_combined",
            prefix=cont_prefix(base),
            target=combined,
            cond_concept="base",
            target_concept="combined",
        ),
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
        ClozeQuery(
            name="entailed_superordinate",
            prefix=kind_of(entailed),
            target=superord,
            cond_concept=entailed,
            target_concept=superord,
        ),
        ClozeQuery(
            name="framed_entailed",
            prefix=from_frame(framed),
            target=entailed,
            cond_concept="framed",
            target_concept=entailed,
        ),
        ClozeQuery(
            name="framed_superordinate",
            prefix=from_frame(framed),
            target=superord,
            cond_concept="framed",
            target_concept=superord,
        ),
    ]


def parse_variants(spec: str | Iterable[str] | None) -> list[str]:
    if spec is None:
        return ["natural", "shuffled_middles", "reversed"]
    if isinstance(spec, str):
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        return parts or ["natural"]
    return [str(p).strip() for p in spec if str(p).strip()]
