"""Model I/O: true next-token π(y|x) for shared-trajectory prefixes."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import DEVICE, DTYPE, MAX_TARGET_TOKENS, MODEL_ID


def resolve_device(device: str = DEVICE) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(dtype_name: str = DTYPE, device: torch.device | None = None) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    name = dtype_name.lower()
    if name not in mapping:
        raise ValueError(f"Unknown dtype: {dtype_name}")
    dtype = mapping[name]
    if device is not None and device.type == "cpu" and dtype != torch.float32:
        return torch.float32
    return dtype


def load_model_and_tokenizer(
    model_id: str = MODEL_ID,
    device: str | torch.device = DEVICE,
    dtype_name: str = DTYPE,
) -> tuple[Any, Any, torch.device]:
    device_t = resolve_device(
        str(device) if not isinstance(device, torch.device) else device.type
    )
    dtype = resolve_dtype(dtype_name, device_t)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device_t)
    model.eval()
    return tokenizer, model, device_t


def num_transformer_layers(model: Any) -> int:
    n = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    if n is None:
        raise ValueError("model.config.num_hidden_layers is required")
    return int(n)


def resolve_layer_index(model: Any, layer_index: int) -> int:
    n_layers = num_transformer_layers(model)
    idx = int(layer_index)
    if idx < 0:
        idx = n_layers + idx
    if not (0 <= idx < n_layers):
        raise ValueError(f"layer_index={layer_index} out of range for n_layers={n_layers}")
    return idx


def target_token_ids(
    tokenizer: Any, text: str, *, max_tokens: int = MAX_TARGET_TOKENS
) -> list[int]:
    """Prefer a leading-space form (typical in-context continuation)."""
    text = text.strip()
    for candidate in (f" {text}", text, f"\n{text}"):
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if ids:
            return [int(t) for t in ids[: max(1, int(max_tokens))]]
    return []


@torch.inference_mode()
def mean_target_logprob(
    tokenizer: Any,
    model: Any,
    prefix: str,
    target_text: str,
    *,
    device: torch.device,
    max_tokens: int = MAX_TARGET_TOKENS,
) -> float:
    """
    Mean log π(target subwords | prefix).

    This is the real LM conditional used for d_M = -ln π(y|x).
    """
    target_ids = target_token_ids(tokenizer, target_text, max_tokens=max_tokens)
    if not target_ids:
        return float("-inf")

    # Match trajectory construction: special tokens on the prefix.
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
    if not prefix_ids:
        return float("-inf")

    full_ids = prefix_ids + target_ids
    input_ids = torch.tensor([full_ids], device=device)
    outputs = model(input_ids=input_ids, use_cache=False)
    logits = outputs.logits[0].float()

    logps: list[float] = []
    for i, tid in enumerate(target_ids):
        # Logits at position p predict token at p+1.
        pos = len(prefix_ids) + i - 1
        if pos < 0:
            continue
        lp = F.log_softmax(logits[pos], dim=-1)[tid]
        logps.append(float(lp.item()))
    if not logps:
        return float("-inf")
    return float(sum(logps) / len(logps))
