"""Locate the pinned model snapshot and import Cloudflare's own joint_schema_model.py from it."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from huggingface_hub import snapshot_download

from clef_tune.config import Config

CLEF_CODE_FILE = "joint_schema_model.py"


class ClefCodeError(RuntimeError):
    """Clef's code or model directory could not be found or imported."""


def _is_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def _download(config: Config) -> Path:
    return Path(snapshot_download(config.model.id, revision=config.model.resolved_revision()))


def model_dir(config: Config) -> Path:
    local_path = config.model.local_path
    if local_path:
        path = config.resolve_path(local_path)
        if not _is_model_dir(path):
            raise ClefCodeError(
                f"model.local_path {local_path!r} is not a model directory "
                f"(expected a directory containing config.json): {path}"
            )
        return path
    return _download(config)


def adapter_base_dir(config: Config, base_model_path: str | Path | None = None) -> Path:
    """The base model of a trained adapter: `base_model_path` if given, else the recorded
    model.local_path while it is still a model directory here, else the pinned Hub snapshot
    (an adapter trained on another machine, or on HF Jobs from /model)."""
    if base_model_path is not None:
        path = Path(base_model_path)
        if not _is_model_dir(path):
            raise ClefCodeError(
                f"base model path {str(base_model_path)!r} is not a model directory "
                "(expected a directory containing config.json)"
            )
        return path
    local_path = config.model.local_path
    if local_path and not _is_model_dir(config.resolve_path(local_path)):
        print(
            f"note: the adapter's base model {local_path!r} is not a model directory here; "
            f"using {config.model.id} @ {config.model.resolved_revision()} from the Hub",
            file=sys.stderr,
        )
        return _download(config)
    return model_dir(config)


def load_clef_code(model_path: str | Path) -> ModuleType:
    file = Path(model_path) / CLEF_CODE_FILE
    if not file.is_file():
        raise ClefCodeError(f"{CLEF_CODE_FILE} not found in {Path(model_path)}")
    digest = hashlib.sha256(file.read_bytes()).hexdigest()[:12]
    name = f"clef_joint_schema_model_{digest}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, file)
    if spec is None or spec.loader is None:
        raise ClefCodeError(f"cannot import {file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # its dataclasses look themselves up here during exec
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module
