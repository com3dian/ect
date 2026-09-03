"""Forward-hook activation patching: h_patched = h + alpha * v."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch

from .model_io import get_transformer_layers


def _unpack_hidden(output: Any) -> tuple[torch.Tensor, Any]:
    """Split a decoder-layer output into residual hidden states + remainder."""
    if isinstance(output, tuple):
        if not output or not torch.is_tensor(output[0]):
            raise TypeError(f"unexpected layer output tuple: {type(output[0]) if output else output}")
        return output[0], output[1:]
    if torch.is_tensor(output):
        return output, None
    if isinstance(output, dict):
        for key in ("hidden_states", "last_hidden_state"):
            if key in output and torch.is_tensor(output[key]):
                return output[key], ("dict", key, output)
        raise TypeError(f"dict layer output missing hidden tensor: {list(output)}")
    raise TypeError(f"unsupported layer output type: {type(output)}")


def _repack_hidden(hidden: torch.Tensor, rest: Any, original: Any) -> Any:
    if rest is None:
        return hidden
    if isinstance(rest, tuple) and len(rest) == 3 and rest[0] == "dict":
        _, key, mapping = rest
        out = dict(mapping)
        out[key] = hidden
        return out
    return (hidden,) + rest


def _apply_steering(
    hidden: torch.Tensor,
    *,
    steering: torch.Tensor,
    alpha: float,
    dim_indices: torch.Tensor | None,
    token_start: int,
) -> torch.Tensor:
    """
    Add alpha * v to residual positions [token_start:] (all batch rows).

    Works in-place on a clone so the original graph tensor is not mutated
    until we copy back.
    """
    h_out = hidden.clone()
    patch = steering.to(device=h_out.device, dtype=h_out.dtype)
    start = int(token_start)
    if start < 0:
        start = h_out.shape[1] + start
    start = max(0, min(start, h_out.shape[1] - 1))

    if dim_indices is None:
        h_out[:, start:, :] = h_out[:, start:, :] + float(alpha) * patch
    else:
        idx = dim_indices.to(device=h_out.device)
        h_out[:, start:, idx] = h_out[:, start:, idx] + float(alpha) * patch
    return h_out


@contextmanager
def patch_residual_stream(
    model: Any,
    layer_index: int,
    steering_vector: torch.Tensor,
    *,
    alpha: float,
    dim_indices: torch.Tensor | None = None,
    token_start: int = 0,
    fired: list[int] | None = None,
) -> Iterator[None]:
    """
    Register a forward hook on transformer block `layer_index` that adds
    alpha * v to the residual stream from `token_start` through the last token.

    The hook both writes in-place onto the module output AND returns the
    patched tensor, so downstream layers cannot ignore the intervention.
    """
    layers = get_transformer_layers(model)
    block = layers[int(layer_index)]
    v = steering_vector.detach().float().contiguous()
    idx = None if dim_indices is None else dim_indices.detach().long().contiguous()

    def _hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden, rest = _unpack_hidden(output)
        patched = _apply_steering(
            hidden,
            steering=v,
            alpha=alpha,
            dim_indices=idx,
            token_start=token_start,
        )
        # In-place so any aliased references see the patch.
        hidden.copy_(patched)
        if fired is not None:
            fired.append(1)
        return _repack_hidden(hidden, rest, output)

    handle = block.register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()
