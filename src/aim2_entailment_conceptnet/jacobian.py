"""
Jacobian pullback for v_ECT = J_ell^T · W_U[target_token] without materializing full J.

Uses autograd: v_ECT = ∇_{h_ell} (h_L · W_U[t]) evaluated on premise X.
"""

from __future__ import annotations

from typing import Any

import torch

from .model_io import get_transformer_layers


def compute_v_ect_vjp(
    tokenizer: Any,
    model: Any,
    premise_prompt: str,
    pullback_vector: torch.Tensor,
    *,
    device: torch.device,
    layer_index: int,
) -> torch.Tensor:
    """
    v_ECT = J_ell^T · u  via ∇_{h_ell} (h_L · u), u = W_U[target_token].

    Jacobian is local to premise context X (one forward + backward).
    Returns float32 CPU vector [hidden_dim].
    """
    encoded = tokenizer(premise_prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Capture the residual tensor in-graph via a hook. Slicing hidden_states
    # yields a non-leaf view whose .grad is never populated.
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
            f"captured residual at layer {layer_index} does not require grad; "
            "cannot compute v_ECT VJP"
        )

    h_l = outputs.hidden_states[-1][0, -1, :].float()
    u = pullback_vector.to(device=h_l.device, dtype=torch.float32)
    scalar = (h_l * u).sum()
    (grad_stream,) = torch.autograd.grad(scalar, h_stream, retain_graph=False)
    return grad_stream[0, -1, :].detach().float().cpu()


def top_k_dim_indices(
    v_trad: torch.Tensor,
    v_ect: torch.Tensor,
    *,
    top_k: int,
) -> torch.Tensor:
    """
    Select Top-K concept-relevant residual dimensions by max |component|
    across v_trad and v_ECT (README slicing to manage VRAM / disk).
    """
    k = min(int(top_k), int(v_trad.numel()))
    scores = torch.maximum(v_trad.abs(), v_ect.abs())
    _, idx = torch.topk(scores, k=k, largest=True, sorted=True)
    return idx.sort().values


def slice_vector(v: torch.Tensor, dim_indices: torch.Tensor) -> torch.Tensor:
    return v[dim_indices].float().clone()
