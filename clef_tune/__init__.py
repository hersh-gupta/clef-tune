"""clef-tune: fine-tuning harness for Cloudflare's Clef decision models."""

from typing import TYPE_CHECKING

try:
    from clef_tune._version import __version__
except ImportError:  # source checkout that was never built or installed
    __version__ = "0.0.0"

if TYPE_CHECKING:
    from clef_tune.predictor import Predictor

__all__ = ["Predictor", "__version__"]


def __getattr__(name: str):
    # Lazy (PEP 562) so that `import clef_tune` never imports torch.
    if name == "Predictor":
        from clef_tune.predictor import Predictor

        return Predictor
    raise AttributeError(f"module 'clef_tune' has no attribute {name!r}")
