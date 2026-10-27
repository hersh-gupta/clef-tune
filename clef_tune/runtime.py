"""Runtime constants shared by the trainer and the HF Jobs launcher. Torch-free."""

import importlib.util
import math
import os

BASE_IMAGE = "pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel"
KERNEL_PACKAGES = ("flash-linear-attention", "causal-conv1d")
KERNEL_MODULES = ("fla", "causal_conv1d")
PRICE_ENV = "CLEF_TUNE_PRICE_PER_HOUR"
ESTIMATE_ENV = "CLEF_TUNE_ESTIMATED_HOURS"


def kernels_available() -> bool:
    """True when every fast linear-attention kernel module is importable."""
    return all(importlib.util.find_spec(name) is not None for name in KERNEL_MODULES)


def env_float(name: str) -> float | None:
    """A positive finite float from the environment, or None when unset or invalid."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) and value > 0 else None
