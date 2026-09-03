"""Aim-3 additive-chain v5: shared-trajectory TVD / half-L1 path costs."""

from .config import LAYER_INDEX, MODEL_ID, SAE_ID, SAE_RELEASE

__all__ = ["LAYER_INDEX", "MODEL_ID", "SAE_ID", "SAE_RELEASE", "run_experiment"]


def run_experiment(*args, **kwargs):
    from .experiment import run_experiment as _run

    return _run(*args, **kwargs)
