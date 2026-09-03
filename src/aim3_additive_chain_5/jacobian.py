"""
J-lens fingerprint projection (optional second metric alongside SAE TVD).
"""

from __future__ import annotations

from typing import Any

import torch

from .config import TOP_K_RESID, TOP_K_VOCAB
from .model_io import get_transformer_layers


def top_k_abs_indices(vec: torch.Tensor, k: int) -> torch.Tensor:
    k = min(int(k), int(vec.numel()))
    _, idx = torch.topk(vec.abs().float(), k=k, largest=True, sorted=True)
    return idx.sort().values


def compute_truncated_jacobian_at_token(
    tokenizer: Any,
    model: Any,
    text: str,
    token_index: int,
    *,
    device: torch.device,
    layer_index: int,
    top_k_vocab: int = TOP_K_VOCAB,
    top_k_resid: int = TOP_K_RESID,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    captured: dict[str, torch.Tensor] = {}

    def _hook(_module: Any, _inputs: Any, output: Any) -> Any:
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h
        return output

    handle = get_transformer_layers(model)[int(layer_index)].register_forward_hook(_hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
    finally:
        handle.remove()

    h_stream = captured["h"]
    seq_len = int(h_stream.shape[1])
    t = min(max(0, int(token_index)), seq_len - 1)
    logits = outputs.logits[0, t, :].float()
    k_vocab = min(int(top_k_vocab), int(logits.numel()))
    _, top_ids = torch.topk(logits, k=k_vocab, largest=True, sorted=True)
    h_ell = h_stream[0, t, :].detach().float()
    resid_idx = top_k_abs_indices(h_ell, top_k_resid)

    rows: list[torch.Tensor] = []
    for i in range(k_vocab):
        scalar = logits[top_ids[i]]
        retain = i < k_vocab - 1
        (grad_stream,) = torch.autograd.grad(
            scalar, h_stream, retain_graph=retain, allow_unused=False
        )
        rows.append(grad_stream[0, t, :].detach().float())

    j_trunc = torch.stack(rows, dim=0)[:, resid_idx].contiguous().cpu()
    return {
        "j_trunc": j_trunc,
        "resid_indices": resid_idx.detach().cpu().long(),
    }


def residuals_to_j_features(
    residuals: list[torch.Tensor],
    j_pack: dict[str, torch.Tensor],
) -> list[torch.Tensor]:
    j_trunc = j_pack["j_trunc"].float()
    idx = j_pack["resid_indices"].long()
    out: list[torch.Tensor] = []
    for h in residuals:
        h_s = h.float().reshape(-1)[idx]
        out.append((j_trunc @ h_s).detach().cpu().float())
    return out
