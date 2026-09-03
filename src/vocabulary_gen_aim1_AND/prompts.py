"""Dynamic category meta-prompts for orthogonality spectrum corpus generation."""

from __future__ import annotations

from .config import PAIRS_PER_BATCH

# Meta-prompt templates from README §3. `{seed}` is injected per domain seed.
CATEGORY_CONSTRAINTS: dict[int, str] = {
    1: (
        "Word A MUST be a specific instance, sub-part, or strictly entailed subset "
        "of Word B within the domain of [{seed}]. Do NOT generate lateral "
        "co-occurring items. Example: (Right Triangle, Polygon)."
    ),
    2: (
        "Word A and Word B must heavily co-occur in typical daily life within "
        "the domain of [{seed}]."
    ),
    3: (
        "Word A and Word B must come from two completely different sub-fields "
        "within the prompt [{seed}], representing a highly unusual but physically "
        "possible combination."
    ),
    4: (
        "Word A and Word B must represent a surreal, hallucinatory, or "
        "category-error combination based on the domain [{seed}]."
    ),
    5: (
        "Word A and Word B must be absolute antonyms or represent a "
        "physical/logical impossibility when combined within the domain of "
        "[{seed}]. They must destroy each other conceptually."
    ),
}


SYSTEM_PROMPT = (
    "You are a careful linguistic dataset architect generating controlled "
    "concept pairs for evaluating Large Language Models under Enriched "
    "Category Theory. Follow the category constraints exactly. Prefer short, "
    "concrete concept phrases (1–4 words). Return only structured JSON that "
    "matches the requested schema."
)


def pairs_json_schema() -> dict:
    """OpenAI-compatible JSON Schema for structured outputs (root must be an object)."""
    return {
        "type": "object",
        "properties": {
            "pairs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "word_a": {"type": "string"},
                        "word_b": {"type": "string"},
                    },
                    "required": ["word_a", "word_b"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["pairs"],
        "additionalProperties": False,
    }


def response_format_json_schema() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "concept_pairs_batch",
            "strict": True,
            "schema": pairs_json_schema(),
        },
    }


def response_format_json_object() -> dict:
    """Fallback for gateways that only support JSON mode (not full json_schema)."""
    return {"type": "json_object"}


def build_user_prompt(category_id: int, seed: str, n_pairs: int = PAIRS_PER_BATCH) -> str:
    """Build the user message for one (category, seed) generation batch."""
    if category_id not in CATEGORY_CONSTRAINTS:
        raise KeyError(f"Unknown category_id={category_id}")

    constraint = CATEGORY_CONSTRAINTS[category_id].format(seed=seed)
    return (
        f"Category {category_id} constraint:\n{constraint}\n\n"
        f"Task: Generate exactly {n_pairs} unique concept pairs for this category "
        f"and domain seed.\n"
        "Output requirements:\n"
        "- Return a single JSON object with key \"pairs\".\n"
        f'- \"pairs\" must be an array of length {n_pairs}.\n'
        '- Each element must be: {{"word_a": "...", "word_b": "..."}}.\n'
        "- No markdown fences, no commentary, no trailing text.\n"
        "- Pairs must be distinct within the batch (case-insensitive).\n"
        "- word_a and word_b must be different strings.\n"
    )


def build_messages(
    category_id: int,
    seed: str,
    n_pairs: int = PAIRS_PER_BATCH,
) -> list[dict[str, str]]:
    """OpenAI chat messages for one generation call."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(category_id, seed, n_pairs)},
    ]
