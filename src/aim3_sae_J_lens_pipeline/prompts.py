"""Waypoint prompts for Type A/B/C tropical-geometry corpus items."""

from __future__ import annotations

from typing import Any

CONCEPT_TEMPLATE = "Describe the typical properties and context of a {concept}:"

TYPE_A_NAME = "Type A: Strict Phylogenetic Chains"
TYPE_B_NAME = "Type B: Sudden Contextual Shifts"
TYPE_C_NAME = "Type C: Adversarial Functors (Contextual Distortions)"

TYPE_KEY_BY_NAME = {
    TYPE_A_NAME: "A",
    TYPE_B_NAME: "B",
    TYPE_C_NAME: "C",
}


def type_key(row: dict[str, Any]) -> str:
    name = str(row.get("structure_type", "")).strip()
    if name in TYPE_KEY_BY_NAME:
        return TYPE_KEY_BY_NAME[name]
    raw = name[:6].upper()
    if raw.startswith("TYPE A") or raw == "A":
        return "A"
    if raw.startswith("TYPE B") or raw == "B":
        return "B"
    if raw.startswith("TYPE C") or raw == "C":
        return "C"
    raise ValueError(f"Unknown structure_type={name!r}")


def novel_suffix(prefix: str, full: str) -> str:
    """Text in ``full`` that is not already in ``prefix`` (for continuation scoring)."""
    prefix = prefix.strip()
    full = full.strip()
    if full.startswith(prefix):
        return full[len(prefix) :].lstrip(" ,.;:!?")
    return full


def concept_prompt(concept: str) -> str:
    return CONCEPT_TEMPLATE.format(concept=str(concept).strip())


def item_id(row: dict[str, Any], index: int) -> str:
    return f"{type_key(row)}_{int(index):05d}"


def waypoints_for_row(row: dict[str, Any]) -> list[dict[str, str]]:
    """
    Named extraction waypoints for one corpus item.

    Each waypoint has ``name``, ``prompt`` (model input), and optional
    ``concept`` (word whose next-token tropical log-prob may be scored).
    """
    key = type_key(row)
    if key == "A":
        nodes = [str(row[f"node_{i}"]).strip() for i in range(1, 5)]
        points = [
            {"name": f"node_{i}", "prompt": concept_prompt(node), "concept": node}
            for i, node in enumerate(nodes, start=1)
        ]
        points.append(
            {
                "name": "context",
                "prompt": str(row["context_sequence"]).strip(),
                "concept": nodes[-1],
            }
        )
        return points
    if key == "B":
        return [
            {
                "name": "base",
                "prompt": str(row["base_context"]).strip(),
                "concept": "",
            },
            {
                "name": "shift",
                "prompt": str(row["shift_context"]).strip(),
                "concept": "",
            },
            {
                "name": "combined",
                "prompt": str(row["combined_sequence"]).strip(),
                "concept": "",
            },
        ]
    if key == "C":
        entailed = str(row["entailed_concept"]).strip()
        superord = str(row["superordinate"]).strip()
        return [
            {
                "name": "entailed",
                "prompt": concept_prompt(entailed),
                "concept": entailed,
            },
            {
                "name": "superordinate",
                "prompt": concept_prompt(superord),
                "concept": superord,
            },
            {
                "name": "framed",
                "prompt": str(row["framed_sequence"]).strip(),
                "concept": entailed,
            },
        ]
    raise ValueError(f"Unhandled type key={key!r}")


def jacobian_waypoint_name(key: str) -> str:
    """Single VRAM-heavy Jacobian site per item (the full contextual sequence)."""
    return {"A": "context", "B": "combined", "C": "framed"}[key]


def tropical_pairs_for_row(row: dict[str, Any]) -> list[dict[str, str]]:
    """
    Directed pairs (source_prompt, target_text) for d_M = -ln π(target | source).
    """
    key = type_key(row)
    if key == "A":
        nodes = [str(row[f"node_{i}"]).strip() for i in range(1, 5)]
        pairs = [
            {
                "pair_name": "n1_n2",
                "source_prompt": concept_prompt(nodes[0]),
                "target_text": nodes[1],
                "source_label": nodes[0],
                "target_label": nodes[1],
            },
            {
                "pair_name": "n2_n3",
                "source_prompt": concept_prompt(nodes[1]),
                "target_text": nodes[2],
                "source_label": nodes[1],
                "target_label": nodes[2],
            },
            {
                "pair_name": "n3_n4",
                "source_prompt": concept_prompt(nodes[2]),
                "target_text": nodes[3],
                "source_label": nodes[2],
                "target_label": nodes[3],
            },
            {
                "pair_name": "n1_n4",
                "source_prompt": concept_prompt(nodes[0]),
                "target_text": nodes[3],
                "source_label": nodes[0],
                "target_label": nodes[3],
            },
        ]
        return pairs
    if key == "B":
        base = str(row["base_context"]).strip()
        shift = str(row["shift_context"]).strip()
        combined = str(row["combined_sequence"]).strip()
        return [
            {
                "pair_name": "base_shift",
                "source_prompt": base,
                "target_text": shift,
                "source_label": "base",
                "target_label": "shift",
            },
            {
                "pair_name": "base_combined",
                "source_prompt": base,
                "target_text": novel_suffix(base, combined) or shift,
                "source_label": "base",
                "target_label": "combined",
            },
            {
                "pair_name": "shift_combined",
                "source_prompt": shift,
                "target_text": novel_suffix(shift, combined) or combined,
                "source_label": "shift",
                "target_label": "combined",
            },
        ]
    if key == "C":
        entailed = str(row["entailed_concept"]).strip()
        superord = str(row["superordinate"]).strip()
        framed = str(row["framed_sequence"]).strip()
        return [
            {
                "pair_name": "entailed_superordinate",
                "source_prompt": concept_prompt(entailed),
                "target_text": superord,
                "source_label": entailed,
                "target_label": superord,
            },
            {
                "pair_name": "framed_entailed",
                "source_prompt": framed,
                "target_text": entailed,
                "source_label": "framed",
                "target_label": entailed,
            },
            {
                "pair_name": "framed_superordinate",
                "source_prompt": framed,
                "target_text": superord,
                "source_label": "framed",
                "target_label": superord,
            },
        ]
    raise ValueError(f"Unhandled type key={key!r}")
