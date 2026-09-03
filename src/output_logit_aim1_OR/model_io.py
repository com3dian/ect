"""Load the local HF model and extract Top-K softmax at the final token."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import DEVICE, DTYPE, MODEL_ID, TOP_K


@dataclass
class TopKDistribution:
    """Sparse Top-K probability distribution over vocabulary token ids."""

    token_ids: list[int]
    probs: list[float]

    def as_dict(self) -> dict[int, float]:
        return {int(t): float(p) for t, p in zip(self.token_ids, self.probs)}


def resolve_device(device: str = DEVICE) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(dtype_name: str = DTYPE, device: torch.device | None = None) -> torch.dtype:
    name = dtype_name.lower()
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
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
    """Load tokenizer + causal LM for offline logit-lens extraction."""
    device_t = resolve_device(str(device) if not isinstance(device, torch.device) else device.type)
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


@torch.inference_mode()
def extract_topk_at_final_token(
    tokenizer: Any,
    model: Any,
    prompt: str,
    *,
    device: torch.device,
    top_k: int = TOP_K,
) -> TopKDistribution:
    """Forward pass → logits at the last prompt token → softmax → Top-K."""
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    final_logits = outputs.logits[0, -1, :].float()
    probs = F.softmax(final_logits, dim=-1)

    k = min(top_k, probs.numel())
    values, indices = torch.topk(probs, k=k, largest=True, sorted=True)
    return TopKDistribution(
        token_ids=[int(i) for i in indices.tolist()],
        probs=[float(v) for v in values.tolist()],
    )
