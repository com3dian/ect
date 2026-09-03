"""Shared-trajectory prompts (same design as aim3_addtive_chain_2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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


@dataclass(frozen=True)
class TrajectoryPrompt:
    text: str
    labels: tuple[str, ...]
    char_spans: tuple[tuple[int, int], ...]
    waypoint_names: tuple[str, ...]


def build_type_a_trajectory(row: dict[str, Any]) -> TrajectoryPrompt:
    nodes = tuple(str(row[f"node_{i}"]).strip() for i in range(1, 5))
    narrative = str(row.get("context_sequence", "")).strip()

    header = "Phylogenetic semantic chain (shared trajectory):\n"
    text = header
    spans: list[tuple[int, int]] = []
    names: list[str] = []

    for i, node in enumerate(nodes, start=1):
        prefix = f"{i}. "
        start = len(text) + len(prefix)
        text += f"{prefix}{node}\n"
        spans.append((start, start + len(node)))
        names.append(f"node_{i}")

    if narrative:
        text += "\nNarrative: " + narrative + "\n"

    return TrajectoryPrompt(
        text=text,
        labels=nodes,
        char_spans=tuple(spans),
        waypoint_names=tuple(names),
    )


def build_type_b_trajectory(row: dict[str, Any]) -> TrajectoryPrompt:
    base = str(row["base_context"]).strip()
    shift = str(row["shift_context"]).strip()
    combined = str(row.get("combined_sequence", f"{base} {shift}")).strip()

    text = "Contextual trajectory (shared):\n"
    base_h = "Base: "
    b0 = len(text) + len(base_h)
    text += base_h + base + "\n"
    b1 = b0 + len(base)

    shift_h = "Shift: "
    s0 = len(text) + len(shift_h)
    text += shift_h + shift + "\n"
    s1 = s0 + len(shift)

    comb_h = "Combined: "
    c0 = len(text) + len(comb_h)
    text += comb_h + combined + "\n"
    c1 = c0 + len(combined)

    return TrajectoryPrompt(
        text=text,
        labels=("base", "shift", "combined"),
        char_spans=((b0, b1), (s0, s1), (c0, c1)),
        waypoint_names=("base", "shift", "combined"),
    )


def build_type_c_trajectory(row: dict[str, Any]) -> TrajectoryPrompt:
    entailed = str(row["entailed_concept"]).strip()
    superord = str(row["superordinate"]).strip()
    framed = str(row["framed_sequence"]).strip()

    text = "Framed entailment trajectory (shared):\n"
    e_h = "Entailed: "
    e0 = len(text) + len(e_h)
    text += e_h + entailed + "\n"
    e1 = e0 + len(entailed)

    s_h = "Superordinate: "
    s0 = len(text) + len(s_h)
    text += s_h + superord + "\n"
    s1 = s0 + len(superord)

    f_h = "Framed: "
    f0 = len(text) + len(f_h)
    text += f_h + framed + "\n"
    f1 = f0 + len(framed)

    return TrajectoryPrompt(
        text=text,
        labels=(entailed, superord, framed),
        char_spans=((e0, e1), (s0, s1), (f0, f1)),
        waypoint_names=("entailed", "superordinate", "framed"),
    )


def build_trajectory(row: dict[str, Any]) -> TrajectoryPrompt:
    key = type_key(row)
    if key == "A":
        return build_type_a_trajectory(row)
    if key == "B":
        return build_type_b_trajectory(row)
    if key == "C":
        return build_type_c_trajectory(row)
    raise ValueError(f"Unhandled type {key}")
