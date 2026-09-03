"""Load the local HF model and extract Top-K via middle-layer logit lens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import APPLY_FINAL_NORM, DEVICE, DTYPE, LAYER_INDEX, MODEL_ID, TOP_K


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


def num_transformer_layers(model: Any) -> int:
    """Number of transformer blocks (excludes the embedding slot in hidden_states)."""
    cfg = getattr(model, "config", None)
    n = getattr(cfg, "num_hidden_layers", None) if cfg is not None else None
    if n is not None:
        return int(n)
    raise ValueError("model.config.num_hidden_layers is required for logit-lens layer selection")


def resolve_layer_index(model: Any, layer_index: int | None = LAYER_INDEX) -> int:
    """
    Resolve the transformer-block index used for the logit lens.

    - Explicit int is clamped into [0, n_layers - 1]
    - None → middle layer = n_layers // 2
    """
    n_layers = num_transformer_layers(model)
    if layer_index is None:
        return int(n_layers // 2)
    idx = int(layer_index)
    if idx < 0:
        idx = n_layers + idx
    if not (0 <= idx < n_layers):
        raise ValueError(f"layer_index={layer_index} out of range for n_layers={n_layers}")
    return idx


def _final_norm_module(model: Any) -> Any | None:
    """Locate the pre-lm_head RMSNorm / LayerNorm used by the causal LM."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "norm"):
        return inner.norm
    transformer = getattr(model, "transformer", None)
    if transformer is not None and hasattr(transformer, "ln_f"):
        return transformer.ln_f
    return None


def hidden_to_logits(
    model: Any,
    hidden: torch.Tensor,
    *,
    apply_final_norm: bool = APPLY_FINAL_NORM,
) -> torch.Tensor:
    """Classic logit lens: optional final norm → lm_head → vocab logits."""
    h = hidden
    if apply_final_norm:
        norm = _final_norm_module(model)
        if norm is not None:
            h = norm(h)
    if not hasattr(model, "lm_head"):
        raise AttributeError("model has no lm_head; cannot apply logit lens")
    return model.lm_head(h)


@torch.inference_mode()
def extract_topk_logit_lens(
    tokenizer: Any,
    model: Any,
    prompt: str,
    *,
    device: torch.device,
    layer_index: int,
    top_k: int = TOP_K,
    apply_final_norm: bool = APPLY_FINAL_NORM,
) -> TopKDistribution:
    """
    Forward pass → hidden state at `layer_index` (final prompt token) →
    logit lens (norm + lm_head) → softmax → Top-K.

    `layer_index` indexes transformer blocks in [0, n_layers-1].
    HuggingFace `hidden_states[layer_index + 1]` is the residual stream
    after that block (`hidden_states[0]` is the embedding output).
    """
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden_states = outputs.hidden_states
    hs_index = int(layer_index) + 1
    if hs_index >= len(hidden_states):
        raise ValueError(
            f"layer_index={layer_index} → hidden_states[{hs_index}] "
            f"but only {len(hidden_states)} states were returned"
        )

    hidden = hidden_states[hs_index][0, -1, :]
    logits = hidden_to_logits(model, hidden, apply_final_norm=apply_final_norm).float()
    probs = F.softmax(logits, dim=-1)

    k = min(top_k, probs.numel())
    values, indices = torch.topk(probs, k=k, largest=True, sorted=True)
    return TopKDistribution(
        token_ids=[int(i) for i in indices.tolist()],
        probs=[float(v) for v in values.tolist()],
    )
