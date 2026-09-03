"""Aim-3 additive-chain v7: chain-rule calibration + short-token Markov."""

from .config import LAYER_INDEX, MODEL_ID

__all__ = ["LAYER_INDEX", "MODEL_ID", "run_experiment"]


def run_experiment(*args, **kwargs):
    from .experiment import run_experiment as _run

    return _run(*args, **kwargs)
