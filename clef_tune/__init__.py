"""clef-tune: fine-tuning harness for Cloudflare's Clef decision models."""

try:
    from clef_tune._version import __version__
except ImportError:  # source checkout that was never built or installed
    __version__ = "0.0.0"
