"""Aim 1 differential SAE-lens AND/OR with syntactic baseline subtraction."""

from .config import LAYER_INDEX, MODEL_ID, SAE_ID, SAE_RELEASE

__all__ = [
    "LAYER_INDEX",
    "MODEL_ID",
    "SAE_ID",
    "SAE_RELEASE",
    "run_experiment",
]


def run_experiment(*args, **kwargs):
    from .experiment import run_experiment as _run_experiment

    return _run_experiment(*args, **kwargs)
