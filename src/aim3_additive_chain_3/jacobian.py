"""
J-lens on a shared trajectory.

Compute a truncated Jacobian J_ℓ at a reference token (last waypoint), then
map every waypoint residual into J-space:

    z_i = J_trunc @ h_i[resid_indices]     # shape [K_vocab]

Treat softplus(z_i) as a non-negative energy distribution and use tropical
d_M = -ln ⟨π_i, π_j⟩ (same formula as SAE v2, but in causal J-coordinates).
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
    """
    J[k, r] ≈ ∂ logit[vocab_k] / ∂ h_ℓ[r] at ``token_index``.

    Vocab Top-K from logits at that position; residual Top-K from |h_ℓ|.
    """
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

    if "h" not in captured:
        raise RuntimeError(f"forward hook did not capture layer {layer_index}")

    h_stream = captured["h"]  # [1, seq, d]
    if not h_stream.requires_grad:
        raise RuntimeError(f"captured residual at layer {layer_index} has no grad")

    seq_len = int(h_stream.shape[1])
    t = min(max(0, int(token_index)), seq_len - 1)

    logits = outputs.logits[0, t, :].float()
    k_vocab = min(int(top_k_vocab), int(logits.numel()))
    top_vals, top_ids = torch.topk(logits, k=k_vocab, largest=True, sorted=True)

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

    j_full = torch.stack(rows, dim=0)  # [K, d]
    j_trunc = j_full[:, resid_idx].contiguous()

    return {
        "j_trunc": j_trunc.cpu(),
        "vocab_ids": top_ids.detach().cpu().long(),
        "vocab_logits": top_vals.detach().cpu().float(),
        "resid_indices": resid_idx.detach().cpu().long(),
        "ref_token_index": torch.tensor([t], dtype=torch.int64),
        "top_k_vocab": torch.tensor([k_vocab], dtype=torch.int64),
        "top_k_resid": torch.tensor([int(resid_idx.numel())], dtype=torch.int64),
    }


def project_residual_to_j_space(
    residual: torch.Tensor,
    j_trunc: torch.Tensor,
    resid_indices: torch.Tensor,
) -> torch.Tensor:
    """
    z = J_trunc @ h[resid_indices]  →  [K_vocab] float32 CPU.
    """
    h = residual.float().reshape(-1)
    idx = resid_indices.long()
    h_s = h[idx]
    z = j_trunc.float() @ h_s
    return z.detach().cpu().float()


def residuals_to_j_features(
    residuals: list[torch.Tensor],
    j_pack: dict[str, torch.Tensor],
) -> list[torch.Tensor]:
    j_trunc = j_pack["j_trunc"]
    resid_idx = j_pack["resid_indices"]
    return [
        project_residual_to_j_space(h, j_trunc, resid_idx) for h in residuals
    ]
