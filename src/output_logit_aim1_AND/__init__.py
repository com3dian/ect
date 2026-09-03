"""Aim 1 output-logit extraction and composition metrics."""

from .config import MODEL_ID, TOP_K
from .metrics import compute_composition_metrics

__all__ = [
    "MODEL_ID",
    "TOP_K",
    "compute_composition_metrics",
    "run_experiment",
]


def run_experiment(*args, **kwargs):
    # Lazy import so metrics/config work before torch is installed.
    from .experiment import run_experiment as _run_experiment

    return _run_experiment(*args, **kwargs)
