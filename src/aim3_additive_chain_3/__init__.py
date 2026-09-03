"""Aim-3 additive-chain v3: shared-trajectory J-lens distances."""

from .config import LAYER_INDEX, MODEL_ID

__all__ = ["LAYER_INDEX", "MODEL_ID", "run_experiment"]


def run_experiment(*args, **kwargs):
    from .experiment import run_experiment as _run

    return _run(*args, **kwargs)
