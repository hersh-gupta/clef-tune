"""How the UI runs: as a public Hugging Face Space or on someone's own machine, and whose
Hugging Face token a user action may use."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from huggingface_hub import get_token

MAX_UPLOAD_ENV = "CLEF_TUNE_MAX_UPLOAD_MB"
DEFAULT_MAX_UPLOAD_MB = 200
# Gradio's stand-in token when sign-in runs outside a Space; it cannot call the Hub.
MOCK_OAUTH_TOKEN = "mock-oauth-token-for-local-dev"
CUDA_CHECK_SECONDS = 120


class SignInRequired(RuntimeError):
    """A user action needs the user's own Hugging Face token."""


def cuda_available() -> bool:
    """Whether torch sees a CUDA GPU, checked in a subprocess so the UI never imports torch."""
    check = "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
    try:
        result = subprocess.run(
            [sys.executable, "-c", check], capture_output=True, timeout=CUDA_CHECK_SECONDS
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


@dataclass(frozen=True)
class Settings:
    space: bool  # running as a Hugging Face Space (public, multi-user)
    max_upload_mb: int
    local_training: bool  # show "Train locally": never on a Space, elsewhere only with CUDA

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 2**20

    @classmethod
    def from_env(
        cls, env: Mapping[str, str] = os.environ, *, cuda: Callable[[], bool] = cuda_available
    ) -> Settings:
        space = bool(env.get("SPACE_ID"))
        raw = env.get(MAX_UPLOAD_ENV, str(DEFAULT_MAX_UPLOAD_MB))
        try:
            cap = int(raw)
        except ValueError:
            cap = 0
        if cap <= 0:
            raise ValueError(f"{MAX_UPLOAD_ENV} must be a whole number of megabytes (got {raw!r})")
        return cls(space=space, max_upload_mb=cap, local_training=not space and cuda())


def user_token(
    settings: Settings,
    oauth_token: str | None,
    pasted: str | None,
    *,
    local_token: Callable[[], str | None] | None = None,
) -> str:
    """The token for a user action: a pasted token first, then the Hugging Face sign-in on a
    Space or the machine's own login elsewhere. On a Space the server's own token (its HF_TOKEN
    secret) is never used: visitors must not act as the Space's owner."""
    if pasted and pasted.strip():
        return pasted.strip()
    if settings.space:
        if oauth_token:
            return oauth_token
        raise SignInRequired("Sign in with Hugging Face, or paste a token")
    token = (local_token or get_token)()
    if token:
        return token
    raise SignInRequired("Run `hf auth login`, or paste a token")
