"""
Truncated J-lens Jacobian: J_ℓ rows = ∇_{h_ℓ} logit[top-k vocab].

Top-K vocab truncates the softmax long tail; Top-K residual columns isolate
the core semantic manifold and keep the saved matrix small.
"""

from __future__ import annotations

from typing import Any

import torch

from .config import TOP_K_RESID, TOP_K_VOCAB
from .model_io import encode_prompt, get_transformer_layers


def top_k_abs_indices(vec: torch.Tensor, k: int) -> torch.Tensor:
    k = min(int(k), int(vec.numel()))
    _, idx = torch.topk(vec.abs().float(), k=k, largest=True, sorted=True)
    return idx.sort().values


def compute_truncated_jacobian(
    tokenizer: Any,
    model: Any,
    prompt: str,
    *,
    device: torch.device,
    layer_index: int,
    top_k_vocab: int = TOP_K_VOCAB,
    top_k_resid: int = TOP_K_RESID,
) -> dict[str, torch.Tensor]:
    """
    J_trunc[k, r] ≈ ∂ logit[vocab_k] / ∂ h_ℓ[resid_r] at the final token.

    Returns CPU tensors ready for safetensors.
    """
    input_ids, attention_mask = encode_prompt(tokenizer, prompt, device)
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

    h_stream = captured["h"]
    if not h_stream.requires_grad:
        raise RuntimeError(
            f"captured residual at layer {layer_index} does not require grad"
        )

    logits = outputs.logits[0, -1, :].float()
    k_vocab = min(int(top_k_vocab), int(logits.numel()))
    top_vals, top_ids = torch.topk(logits, k=k_vocab, largest=True, sorted=True)
    logprobs = torch.log_softmax(logits, dim=-1)[top_ids]

    h_ell = h_stream[0, -1, :].detach().float()
    resid_idx = top_k_abs_indices(h_ell, top_k_resid)

    rows: list[torch.Tensor] = []
    for i in range(k_vocab):
        scalar = logits[top_ids[i]]
        retain = i < k_vocab - 1
        (grad_stream,) = torch.autograd.grad(
            scalar, h_stream, retain_graph=retain, allow_unused=False
        )
        rows.append(grad_stream[0, -1, :].detach().float())

    j_full = torch.stack(rows, dim=0)  # [K_vocab, d_model]
    j_trunc = j_full[:, resid_idx].contiguous().cpu()

    h_L = outputs.hidden_states[-1][0, -1, :].detach().float().cpu()
    hs_index = int(layer_index) + 1
    h_ell_cpu = outputs.hidden_states[hs_index][0, -1, :].detach().float().cpu()

    return {
        "j_trunc": j_trunc,
        "vocab_ids": top_ids.detach().cpu().long(),
        "vocab_logits": top_vals.detach().cpu().float(),
        "vocab_logprobs": logprobs.detach().cpu().float(),
        "resid_indices": resid_idx.detach().cpu().long(),
        "h_ell": h_ell_cpu,
        "h_L": h_L,
        "layer_index": torch.tensor([int(layer_index)], dtype=torch.int64),
        "top_k_vocab": torch.tensor([k_vocab], dtype=torch.int64),
        "top_k_resid": torch.tensor([int(resid_idx.numel())], dtype=torch.int64),
    }
