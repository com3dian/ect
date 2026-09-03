"""Aim 1 OR middle-layer logit-lens extraction and colimit (max) metrics."""

from .config import LAYER_INDEX, MODEL_ID, TOP_K
from .metrics import compute_or_metrics, compute_theoretical_or

__all__ = [
    "LAYER_INDEX",
    "MODEL_ID",
    "TOP_K",
    "compute_or_metrics",
    "compute_theoretical_or",
    "run_experiment",
]


def run_experiment(*args, **kwargs):
    from .experiment import run_experiment as _run_experiment

    return _run_experiment(*args, **kwargs)
