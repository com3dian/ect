"""
Shared-trajectory prompts for Type A phylogenetic chains.

v1 (aim3_sae_J_lens_pipeline) used *independent* concept probes:
  prompt(n_i) → score n_{i+1}
so path sum and shortcut were unrelated experiments.

v2 builds *one* string that walks n1→n2→n3→n4, records the token index
at the end of each node mention, and measures distances between those
states on the same forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


TYPE_A_NAME = "Type A: Strict Phylogenetic Chains"
TYPE_B_NAME = "Type B: Sudden Contextual Shifts"
TYPE_C_NAME = "Type C: Adversarial Functors (Contextual Distortions)"


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
    """One shared string + character spans for each waypoint label."""

    text: str
    labels: tuple[str, ...]
    # Inclusive char span [start, end) of each label mention used as a waypoint.
    char_spans: tuple[tuple[int, int], ...]
    waypoint_names: tuple[str, ...]


def build_type_a_trajectory(row: dict[str, Any]) -> TrajectoryPrompt:
    """
    Controlled shared trajectory.

    Layout (offsets are exact because we concatenate ourselves)::

        Phylogenetic semantic chain (shared trajectory):
        1. {node_1}
        2. {node_2}
        3. {node_3}
        4. {node_4}

        Narrative: {context_sequence}

    Waypoints sit at the end of each ``N. {node}`` line (on the node text),
    so all four states come from one evolving residual stream.
    """
    nodes = tuple(str(row[f"node_{i}"]).strip() for i in range(1, 5))
    narrative = str(row.get("context_sequence", "")).strip()

    header = "Phylogenetic semantic chain (shared trajectory):\n"
    text = header
    spans: list[tuple[int, int]] = []
    names: list[str] = []

    for i, node in enumerate(nodes, start=1):
        prefix = f"{i}. "
        start = len(text) + len(prefix)
        line = f"{prefix}{node}\n"
        text += line
        end = start + len(node)
        spans.append((start, end))
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
    """
    Shared trajectory for contextual shift: base then shift as one sequence.

    Waypoints: end of base block, end of shift block, end of full combined text.
    """
    base = str(row["base_context"]).strip()
    shift = str(row["shift_context"]).strip()
    combined = str(row.get("combined_sequence", f"{base} {shift}")).strip()

    header = "Contextual trajectory (shared):\n"
    text = header

    base_header = "Base: "
    start_base = len(text) + len(base_header)
    text += base_header + base + "\n"
    end_base = start_base + len(base)

    shift_header = "Shift: "
    start_shift = len(text) + len(shift_header)
    text += shift_header + shift + "\n"
    end_shift = start_shift + len(shift)

    comb_header = "Combined: "
    start_comb = len(text) + len(comb_header)
    text += comb_header + combined + "\n"
    end_comb = start_comb + len(combined)

    return TrajectoryPrompt(
        text=text,
        labels=("base", "shift", "combined"),
        char_spans=((start_base, end_base), (start_shift, end_shift), (start_comb, end_comb)),
        waypoint_names=("base", "shift", "combined"),
    )


def build_type_c_trajectory(row: dict[str, Any]) -> TrajectoryPrompt:
    """
    Shared trajectory: bare entailment labels, then framed sequence.

    Waypoints: entailed label, superordinate label, end of framed sentence.
    """
    entailed = str(row["entailed_concept"]).strip()
    superord = str(row["superordinate"]).strip()
    framed = str(row["framed_sequence"]).strip()

    header = "Framed entailment trajectory (shared):\n"
    text = header

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
