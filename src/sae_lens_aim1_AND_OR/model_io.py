"""Load Gemma-2 + Gemma Scope SAE and extract L1-normalized feature vectors."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import DEVICE, DTYPE, LAYER_INDEX, MODEL_ID, SAE_ID, SAE_RELEASE


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


def load_sae(
    release: str = SAE_RELEASE,
    sae_id: str = SAE_ID,
    device: torch.device | str = "cuda",
) -> Any:
    """Load a Gemma Scope residual SAE via SAELens."""
    from sae_lens import SAE

    device_str = str(device) if isinstance(device, torch.device) else device
    result = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device_str,
    )
    # Older SAELens returned (sae, cfg, sparsity); newer may return sae only.
    if isinstance(result, tuple):
        return result[0]
    return result


def num_transformer_layers(model: Any) -> int:
    cfg = getattr(model, "config", None)
    n = getattr(cfg, "num_hidden_layers", None) if cfg is not None else None
    if n is not None:
        return int(n)
    raise ValueError("model.config.num_hidden_layers is required")


def resolve_layer_index(model: Any, layer_index: int = LAYER_INDEX) -> int:
    n_layers = num_transformer_layers(model)
    idx = int(layer_index)
    if idx < 0:
        idx = n_layers + idx
    if not (0 <= idx < n_layers):
        raise ValueError(f"layer_index={layer_index} out of range for n_layers={n_layers}")
    return idx


@torch.inference_mode()
def extract_sae_features(
    tokenizer: Any,
    model: Any,
    sae: Any,
    prompt: str,
    *,
    device: torch.device,
    layer_index: int,
) -> torch.Tensor:
    """
    Forward pass → residual at `layer_index` (final token) → SAE encode →
    non-negative feature activations (raw, not yet L1-normalized).

    Returns 1-D float32 CPU tensor of shape [d_sae].
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
    # hidden_states[0] = embeddings; hidden_states[L+1] = after block L
    hs_index = int(layer_index) + 1
    if hs_index >= len(hidden_states):
        raise ValueError(
            f"layer_index={layer_index} → hidden_states[{hs_index}] "
            f"but only {len(hidden_states)} states were returned"
        )

    hidden = hidden_states[hs_index][0, -1, :].to(device)
    # SAE encode expects [batch, d_model] or [d_model]
    acts = sae.encode(hidden.unsqueeze(0) if hidden.ndim == 1 else hidden)
    if acts.ndim == 2:
        acts = acts[0]
    # JumpReLU / ReLU SAEs should already be non-negative; clamp for safety.
    acts = torch.clamp(acts.float(), min=0.0)
    return acts.detach().cpu()
