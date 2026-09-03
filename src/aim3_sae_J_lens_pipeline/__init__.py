"""Aim 3: SAE + J-lens tropical geometry of Gemma-2 hidden states."""

from .config import LAYER_INDEX, MODEL_ID, SAE_ID, SAE_RELEASE

__all__ = [
    "LAYER_INDEX",
    "MODEL_ID",
    "SAE_ID",
    "SAE_RELEASE",
    "run_pipeline",
]


def run_pipeline(*args, **kwargs):
    from .experiment import run_pipeline as _run_pipeline

    return _run_pipeline(*args, **kwargs)
