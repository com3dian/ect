"""Extract v_trad / v_ECT and persist Top-K sliced vectors as safetensors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .config import TOP_K_DIMS, vector_path
from .jacobian import (
    compute_v_ect_vjp,
    slice_vector,
    top_k_dim_indices,
)
from .model_io import (
    extract_hidden_at_final_token,
    resolve_target_word,
    target_word_token_id,
    unembedding_row,
)


def match_ect_magnitude_to_trad(
    v_ect: torch.Tensor,
    v_trad: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    L2-normalize v_ECT and rescale to ||v_trad|| so alpha is the same force.

    v_ect <- (v_ect / ||v_ect||) * ||v_trad||
    """
    trad = v_trad.float()
    ect = v_ect.float()
    n_ect = torch.linalg.vector_norm(ect)
    n_trad = torch.linalg.vector_norm(trad)
    if float(n_ect) <= eps:
        return torch.zeros_like(ect)
    return ect * (n_trad / n_ect)


def extract_steering_vectors(
    tokenizer: Any,
    model: Any,
    *,
    premise_prompt: str,
    hypothesis_prompt: str,
    hypothesis: str,
    device: torch.device,
    layer_index: int,
    top_k_dims: int = TOP_K_DIMS,
    target_word: str | None = None,
) -> dict[str, torch.Tensor]:
    """
    Compute full v_trad and v_ECT, then slice to Top-K dims.

    v_trad = h_Y^ell - h_X^ell
    v_ECT  = J_ell^T · W_U[target_token]  (J-lens unembedding pullback on premise X)
    """
    h_x = extract_hidden_at_final_token(
        tokenizer, model, premise_prompt, device=device, layer_index=layer_index
    )
    h_y_layer = extract_hidden_at_final_token(
        tokenizer, model, hypothesis_prompt, device=device, layer_index=layer_index
    )
    v_trad = (h_y_layer - h_x).float()

    word = resolve_target_word(hypothesis, target_word)
    token_id, _piece = target_word_token_id(tokenizer, word)
    w_u_t = unembedding_row(model, token_id)
    v_ect = compute_v_ect_vjp(
        tokenizer,
        model,
        premise_prompt,
        w_u_t,
        device=device,
        layer_index=layer_index,
    )

    n_trad = torch.linalg.vector_norm(v_trad)
    n_ect_raw = torch.linalg.vector_norm(v_ect)
    # Equalize physical force before any slicing / alpha.
    v_ect = match_ect_magnitude_to_trad(v_ect, v_trad)

    dim_indices = top_k_dim_indices(v_trad, v_ect, top_k=top_k_dims)
    v_trad_s = slice_vector(v_trad, dim_indices)
    v_ect_s = slice_vector(v_ect, dim_indices)
    # Re-match on the injected Top-K slices so alpha is fair at patch time.
    v_ect_s = match_ect_magnitude_to_trad(v_ect_s, v_trad_s)

    return {
        "v_trad_full": v_trad,
        "v_ect_full": v_ect,
        "v_trad": v_trad_s,
        "v_ect": v_ect_s,
        "dim_indices": dim_indices.long(),
        "target_token_id": torch.tensor([token_id], dtype=torch.int64),
        "layer_index": torch.tensor([layer_index], dtype=torch.int64),
        "top_k_dims": torch.tensor([len(dim_indices)], dtype=torch.int64),
        "norm_trad": n_trad.detach().cpu().float().reshape(()),
        "norm_ect_raw": n_ect_raw.detach().cpu().float().reshape(()),
        "norm_ect_matched": torch.linalg.vector_norm(v_ect).detach().cpu().float().reshape(()),
        "norm_trad_slice": torch.linalg.vector_norm(v_trad_s).detach().cpu().float().reshape(()),
        "norm_ect_slice": torch.linalg.vector_norm(v_ect_s).detach().cpu().float().reshape(()),
    }


def save_vectors(payload: dict[str, torch.Tensor], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_save = {
        "v_trad": payload["v_trad"],
        "v_ect": payload["v_ect"],
        "dim_indices": payload["dim_indices"],
        "target_token_id": payload["target_token_id"],
        "layer_index": payload["layer_index"],
        "top_k_dims": payload["top_k_dims"],
        "norm_trad": payload["norm_trad"],
        "norm_ect_raw": payload["norm_ect_raw"],
        "norm_ect_matched": payload["norm_ect_matched"],
        "norm_trad_slice": payload["norm_trad_slice"],
        "norm_ect_slice": payload["norm_ect_slice"],
    }
    save_file(to_save, str(path))
    return path


def load_vectors(path: Path | str) -> dict[str, torch.Tensor]:
    return load_file(str(path))


def vectors_exist(pair_id: str, layer_index: int, base: Path | str) -> bool:
    return vector_path(pair_id, layer_index, base).exists()
