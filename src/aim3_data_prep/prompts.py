"""Meta-prompts and JSON schemas for tropical-geometry corpus generation."""

from __future__ import annotations

from dataclasses import dataclass

from .config import ITEMS_PER_BATCH


@dataclass(frozen=True)
class StructureType:
    """One topological taxonomy class from the Aim-3 README."""

    key: str
    name: str
    description: str
    schema_name: str
    item_schema: dict
    required_fields: tuple[str, ...]


STRUCTURE_TYPES: dict[str, StructureType] = {
    "A": StructureType(
        key="A",
        name="Type A: Strict Phylogenetic Chains",
        description=(
            "Pure hierarchical semantic chains (Entity → Category → Specific → "
            "Instance). Purpose: establish a baseline for additive distances in a "
            "tropical metric space. Example chain: Animal → Mammal → Dog → Poodle. "
            "Each example must include four nested nodes and a short natural-language "
            "context_sequence that walks the chain."
        ),
        schema_name="type_a_batch",
        required_fields=(
            "structure_type",
            "node_1",
            "node_2",
            "node_3",
            "node_4",
            "context_sequence",
        ),
        item_schema={
            "type": "object",
            "properties": {
                "structure_type": {"type": "string"},
                "node_1": {"type": "string"},
                "node_2": {"type": "string"},
                "node_3": {"type": "string"},
                "node_4": {"type": "string"},
                "context_sequence": {"type": "string"},
            },
            "required": [
                "structure_type",
                "node_1",
                "node_2",
                "node_3",
                "node_4",
                "context_sequence",
            ],
            "additionalProperties": False,
        },
    ),
    "B": StructureType(
        key="B",
        name="Type B: Sudden Contextual Shifts",
        description=(
            "Text sequences that contain sudden contextual or topical shifts "
            "(surprisal). Purpose: map theoretical distances against LLM hidden-state "
            "geometry when the semantic trajectory is violently disrupted. Example: "
            "'The surgeon carefully picked up the scalpel, and then suddenly started "
            "juggling glowing chainsaws.' Each example must provide base_context, "
            "shift_context, and combined_sequence."
        ),
        schema_name="type_b_batch",
        required_fields=(
            "structure_type",
            "base_context",
            "shift_context",
            "combined_sequence",
        ),
        item_schema={
            "type": "object",
            "properties": {
                "structure_type": {"type": "string"},
                "base_context": {"type": "string"},
                "shift_context": {"type": "string"},
                "combined_sequence": {"type": "string"},
            },
            "required": [
                "structure_type",
                "base_context",
                "shift_context",
                "combined_sequence",
            ],
            "additionalProperties": False,
        },
    ),
    "C": StructureType(
        key="C",
        name="Type C: Adversarial Functors (Contextual Distortions)",
        description=(
            "Standard logical entailments placed inside extreme, reality-distorting "
            "framing. Purpose: test topological resilience of semantic space against "
            "extreme contextual variables. Example: 'In a surreal dream sequence on "
            "Mars, a poodle is barking.' (testing whether Dog↔Poodle distance remains "
            "rigid). Each example must include entailed_concept, superordinate, "
            "distorting_frame, and framed_sequence."
        ),
        schema_name="type_c_batch",
        required_fields=(
            "structure_type",
            "entailed_concept",
            "superordinate",
            "distorting_frame",
            "framed_sequence",
        ),
        item_schema={
            "type": "object",
            "properties": {
                "structure_type": {"type": "string"},
                "entailed_concept": {"type": "string"},
                "superordinate": {"type": "string"},
                "distorting_frame": {"type": "string"},
                "framed_sequence": {"type": "string"},
            },
            "required": [
                "structure_type",
                "entailed_concept",
                "superordinate",
                "distorting_frame",
                "framed_sequence",
            ],
            "additionalProperties": False,
        },
    ),
}


SYSTEM_PROMPT = (
    "You are an expert computational linguist building a dataset for testing the "
    "tropical geometry and topological properties of Large Language Models. "
    "Follow the requested topological relationship exactly. Return only structured "
    "JSON that matches the schema—no markdown fences, no commentary."
)


def items_json_schema(structure: StructureType) -> dict:
    """OpenAI-compatible JSON Schema (root must be an object)."""
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": structure.item_schema,
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }


def response_format_json_schema(structure: StructureType) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": structure.schema_name,
            "strict": True,
            "schema": items_json_schema(structure),
        },
    }


def response_format_json_object() -> dict:
    """Fallback for gateways that only support JSON mode."""
    return {"type": "json_object"}


def build_user_prompt(
    structure: StructureType,
    n_items: int = ITEMS_PER_BATCH,
    *,
    avoid_echo: str = "",
) -> str:
    """Build the user message matching the Aim-3 README meta-prompt."""
    field_list = ", ".join(f'"{f}"' for f in structure.required_fields)
    avoid_block = ""
    if avoid_echo:
        avoid_block = (
            "\nAvoid near-duplicates of these already-collected examples "
            f"(paraphrase freely; do not copy):\n{avoid_echo}\n"
        )

    return (
        "You are an expert computational linguist building a dataset for testing "
        "the tropical geometry and topological properties of Large Language Models. "
        f"Your task is to generate {n_items} unique text structures that strictly "
        f"exhibit the following topological relationship: {structure.name}.\n"
        f"Definition: {structure.description}\n"
        "Ensure the examples are highly diverse and do not repeat. "
        "Output NOTHING ELSE but a valid JSON object.\n\n"
        "Output requirements:\n"
        '- Return a single JSON object with key "items".\n'
        f'- "items" must be an array of length {n_items}.\n'
        f'- Every item must set "structure_type" exactly to "{structure.name}".\n'
        f"- Required fields per item: {field_list}.\n"
        "- No markdown fences, no commentary, no trailing text.\n"
        "- Items must be distinct within the batch (case-insensitive).\n"
        f"{avoid_block}"
    )


def build_messages(
    structure: StructureType,
    n_items: int = ITEMS_PER_BATCH,
    *,
    avoid_echo: str = "",
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                structure, n_items=n_items, avoid_echo=avoid_echo
            ),
        },
    ]
