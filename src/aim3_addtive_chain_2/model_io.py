"""Model / SAE loading and shared-trajectory hidden-state extraction."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import DEVICE, DTYPE, LAYER_INDEX, MODEL_ID, SAE_ID, SAE_RELEASE
from .prompts import TrajectoryPrompt


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


def load_sae(
    release: str = SAE_RELEASE,
    sae_id: str = SAE_ID,
    device: torch.device | str = "cuda",
) -> Any:
    from sae_lens import SAE

    device_str = str(device) if isinstance(device, torch.device) else device
    result = SAE.from_pretrained(release=release, sae_id=sae_id, device=device_str)
    if isinstance(result, tuple):
        return result[0]
    return result


def num_transformer_layers(model: Any) -> int:
    n = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    if n is None:
        raise ValueError("model.config.num_hidden_layers is required")
    return int(n)


def resolve_layer_index(model: Any, layer_index: int = LAYER_INDEX) -> int:
    n_layers = num_transformer_layers(model)
    idx = int(layer_index)
    if idx < 0:
        idx = n_layers + idx
    if not (0 <= idx < n_layers):
        raise ValueError(f"layer_index={layer_index} out of range for n_layers={n_layers}")
    return idx


def char_span_to_token_index(
    tokenizer: Any,
    text: str,
    char_start: int,
    char_end: int,
) -> int:
    """
    Token index of the last token that overlaps ``[char_start, char_end)``.

    Uses offset mapping when available; falls back to prefix encoding.
    """
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    offsets = encoded["offset_mapping"][0].tolist()
    last = None
    for ti, (a, b) in enumerate(offsets):
        if b <= a:
            continue  # special token
        # overlap with [char_start, char_end)
        if a < char_end and b > char_start:
            last = ti
    if last is not None:
        return int(last)

    # Fallback: encode prefix through char_end
    prefix = text[:char_end]
    ids = tokenizer.encode(prefix, add_special_tokens=True)
    return max(0, len(ids) - 1)


def waypoint_token_indices(
    tokenizer: Any, traj: TrajectoryPrompt
) -> list[int]:
    return [
        char_span_to_token_index(tokenizer, traj.text, a, b)
        for a, b in traj.char_spans
    ]


@torch.inference_mode()
def extract_trajectory_states(
    tokenizer: Any,
    model: Any,
    sae: Any,
    traj: TrajectoryPrompt,
    *,
    device: torch.device,
    layer_index: int,
) -> dict[str, Any]:
    """
    One forward over ``traj.text``; residual + SAE vector at each waypoint token.
    """
    token_idx = waypoint_token_indices(tokenizer, traj)
    encoded = tokenizer(traj.text, return_tensors="pt", add_special_tokens=True)
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
    hs_index = int(layer_index) + 1
    if hs_index >= len(hs):
        raise ValueError(f"layer_index={layer_index} → hs[{hs_index}] len={len(hs)}")

    layer_h = hs[hs_index][0]  # [seq, d]
    seq_len = int(layer_h.shape[0])
    token_idx = [min(max(0, t), seq_len - 1) for t in token_idx]

    residuals: list[torch.Tensor] = []
    sae_vecs: list[torch.Tensor] = []
    for ti in token_idx:
        h = layer_h[ti].to(device)
        residuals.append(h.detach().float().cpu())
        acts = sae.encode(h.unsqueeze(0) if h.ndim == 1 else h)
        if acts.ndim == 2:
            acts = acts[0]
        acts = torch.clamp(acts.float(), min=0.0)
        sae_vecs.append(acts.detach().cpu())

    return {
        "token_indices": token_idx,
        "residuals": residuals,
        "sae_vecs": sae_vecs,
        "input_ids": input_ids[0].detach().cpu(),
        "seq_len": seq_len,
    }


def unembedding_matrix(model: Any) -> torch.Tensor:
    emb = model.get_output_embeddings()
    if emb is None or not hasattr(emb, "weight"):
        raise ValueError("model has no output embedding for logit lens")
    return emb.weight


@torch.inference_mode()
def logit_lens_logprob_for_text(
    model: Any,
    tokenizer: Any,
    residual: torch.Tensor,
    target_text: str,
    *,
    device: torch.device,
    max_tokens: int = 8,
) -> float:
    """
    Logit lens at a trajectory state: apply W_U to residual, score target label.

    Uses mean log-softmax over the first ``max_tokens`` subwords of ``target_text``.
    """
    ids = tokenizer.encode(target_text, add_special_tokens=False)
    if not ids:
        ids = tokenizer.encode(f" {target_text}", add_special_tokens=False)
    ids = [int(t) for t in ids[: max(1, int(max_tokens))]]
    if not ids:
        return float("-inf")

    w_u = unembedding_matrix(model)
    h = residual.to(device=device, dtype=torch.float32)
    # Match common logit-lens practice: project with lm_head weight.
    logits = F.linear(h, w_u.float())
    logp = F.log_softmax(logits, dim=-1)
    vals = [float(logp[tid].item()) for tid in ids]
    return float(sum(vals) / len(vals))
