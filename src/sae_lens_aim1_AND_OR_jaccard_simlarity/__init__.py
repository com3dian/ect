"""Aim 1 SAE Jaccard set-theory AND/OR experiment."""

from .config import ACTIVATION_THRESHOLD, LAYER_INDEX, MODEL_ID, SAE_ID, SAE_RELEASE

__all__ = [
    "ACTIVATION_THRESHOLD",
    "LAYER_INDEX",
    "MODEL_ID",
    "SAE_ID",
    "SAE_RELEASE",
    "run_experiment",
]


def run_experiment(*args, **kwargs):
    from .experiment import run_experiment as _run_experiment

    return _run_experiment(*args, **kwargs)
