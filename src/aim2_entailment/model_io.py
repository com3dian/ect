"""Load Qwen and extract hidden states / target-token log-probabilities."""

from __future__ import annotations

import re
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import DEVICE, DTYPE, MODEL_ID


def resolve_target_word(hypothesis: str, target_word: str | None = None) -> str:
    """
    Concept word for J-lens unembedding pullback.

    Prefer an explicit `target_word`; otherwise the last alphabetic word of
    the hypothesis (e.g. "Someone is performing music." → "music").
    """
    if target_word and str(target_word).strip():
        return str(target_word).strip()
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", hypothesis or "")
    if not words:
        raise ValueError(f"cannot infer target word from hypothesis={hypothesis!r}")
    return words[-1]


def target_word_token_id(tokenizer: Any, word: str) -> tuple[int, str]:
    """
    First subword id of `word`. Try a leading space (typical in-context form).
    Returns (token_id, decoded_piece).
    """
    word = word.strip()
    for candidate in (f" {word}", word):
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if ids:
            tid = int(ids[0])
            piece = tokenizer.decode([tid])
            return tid, piece
    raise ValueError(f"tokenizer produced no ids for target word {word!r}")


def unembedding_row(
    model: Any,
    token_id: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """W_U[token_id]: unembedding row, float32 CPU (or `device` if given)."""
    emb = model.get_output_embeddings()
    if emb is None or not hasattr(emb, "weight"):
        raise ValueError("model has no output embedding (lm_head) for W_U")
    row = emb.weight[int(token_id)].detach().float().contiguous()
    if device is not None:
        row = row.to(device)
    return row.cpu() if device is None else row


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


def num_transformer_layers(model: Any) -> int:
    cfg = getattr(model, "config", None)
    n = getattr(cfg, "num_hidden_layers", None) if cfg is not None else None
    if n is not None:
        return int(n)
    raise ValueError("model.config.num_hidden_layers is required")


def resolve_layer_index(model: Any, layer_index: int) -> int:
    n_layers = num_transformer_layers(model)
    idx = int(layer_index)
    if idx < 0:
        idx = n_layers + idx
    if not (0 <= idx < n_layers):
        raise ValueError(f"layer_index={layer_index} out of range for n_layers={n_layers}")
    return idx


def get_transformer_layers(model: Any) -> Any:
    """Return nn.ModuleList of transformer blocks (Qwen / Llama style)."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unsupported model architecture: cannot locate transformer layers")


def hidden_state_index(layer_index: int) -> int:
    """hidden_states[k] is post-block (k-1); block L → index L+1."""
    return int(layer_index) + 1


@torch.inference_mode()
def extract_hidden_at_final_token(
    tokenizer: Any,
    model: Any,
    prompt: str,
    *,
    device: torch.device,
    layer_index: int | None = None,
) -> torch.Tensor:
    """
    Return final-token hidden vector at `layer_index` (or final layer if None).
    Shape: [hidden_dim], float32 CPU.
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
    hs = outputs.hidden_states
    if layer_index is None:
        h = hs[-1][0, -1, :]
    else:
        idx = hidden_state_index(layer_index)
        if idx >= len(hs):
            raise ValueError(f"layer_index={layer_index} → hs[{idx}] but len={len(hs)}")
        h = hs[idx][0, -1, :]
    return h.float().detach().cpu()


def target_token_ids(tokenizer: Any, hypothesis: str, *, max_tokens: int = 8) -> list[int]:
    """Token ids for the hypothesis span (used for log-prob measurement)."""
    ids = tokenizer.encode(hypothesis, add_special_tokens=False)
    return [int(t) for t in ids[: max(1, int(max_tokens))]]


@torch.inference_mode()
def mean_target_logprob(
    tokenizer: Any,
    model: Any,
    prompt: str,
    target_ids: list[int],
    *,
    device: torch.device,
) -> float:
    """
    Mean log P(target_token_i | prompt + target_prefix) over hypothesis tokens.

    For each target token t_i, score logits at the position immediately before t_i.
    """
    if not target_ids:
        return float("-inf")

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = prompt_ids + target_ids
    input_ids = torch.tensor([full_ids], device=device)
    outputs = model(input_ids=input_ids, use_cache=False)
    logits = outputs.logits[0].float()

    logps: list[float] = []
    for i, tid in enumerate(target_ids):
        pos = len(prompt_ids) + i - 1
        if pos < 0:
            continue
        lp = F.log_softmax(logits[pos], dim=-1)[tid]
        logps.append(float(lp.item()))
    if not logps:
        return float("-inf")
    return float(sum(logps) / len(logps))
