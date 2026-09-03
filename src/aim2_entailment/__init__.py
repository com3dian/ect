"""Aim-2 contextual entailment via Jacobian activation patching."""

from .config import DEFAULT_ALPHAS, DEFAULT_LAYERS, MODEL_ID

__all__ = ["DEFAULT_ALPHAS", "DEFAULT_LAYERS", "MODEL_ID", "run_experiment"]


def run_experiment(*args, **kwargs):
    from .experiment import run_experiment as _run

    return _run(*args, **kwargs)
