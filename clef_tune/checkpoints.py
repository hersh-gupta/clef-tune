"""Step checkpoints, best-eval state and baseline metrics under ``output_dir / "checkpoints"``."""

from __future__ import annotations

import json
import random
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from clef_tune.model import load_trainable_state, trainable_state_dict

KEEP_CHECKPOINTS = 2
_STEP_DIR = re.compile(r"step-(\d+)")


@dataclass
class TrainState:
    step: int
    epoch: int
    batch_in_epoch: int
    best_metric: float | None
    best_step: int | None
    bad_evals: int
    config_hash: str
    tokens_seen: int
    train_seconds: float
    world_size: int = 1  # processes the run trains with: data sharding and the step count


class ConfigMismatchError(RuntimeError):
    pass


def _root(output_dir: Path) -> Path:
    return Path(output_dir) / "checkpoints"


def _step_dirs(output_dir: Path) -> list[tuple[int, Path]]:
    root = _root(output_dir)
    if not root.is_dir():
        return []
    found = []
    for path in root.iterdir():
        match = _STEP_DIR.fullmatch(path.name)
        if match and path.is_dir():
            found.append((int(match.group(1)), path))
    return sorted(found)


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace_dir(tmp: Path, final: Path) -> None:
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)


def save_checkpoint(output_dir, model, optimizer, scheduler, state: TrainState) -> Path:
    root = _root(output_dir)
    final = root / f"step-{state.step}"
    tmp = root / f"step-{state.step}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    save_file(trainable_state_dict(model), str(tmp / "trainable.safetensors"))
    torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    torch.save(scheduler.state_dict(), tmp / "scheduler.pt")
    torch.save(_rng_state(), tmp / "rng.pt")
    _write_json(tmp / "state.json", asdict(state))
    _replace_dir(tmp, final)
    for _, old in _step_dirs(output_dir)[:-KEEP_CHECKPOINTS]:
        shutil.rmtree(old)
    return final


def latest_checkpoint(output_dir) -> Path | None:
    dirs = _step_dirs(output_dir)
    return dirs[-1][1] if dirs else None


def load_checkpoint(path, model, optimizer, scheduler) -> TrainState:
    path = Path(path)
    load_trainable_state(model, load_file(str(path / "trainable.safetensors")))
    optimizer.load_state_dict(torch.load(path / "optimizer.pt", weights_only=True))
    scheduler.load_state_dict(torch.load(path / "scheduler.pt", weights_only=True))
    _restore_rng(torch.load(path / "rng.pt", weights_only=False))
    return read_state(path)


def read_state(path) -> TrainState:
    """A checkpoint's TrainState, without loading anything else (older ones: one process)."""
    return TrainState(**_read_json(Path(path) / "state.json"))


def check_resume(output_dir, config_hash: str) -> Path | None:
    latest = latest_checkpoint(output_dir)
    if latest is None:
        return None
    if _read_json(latest / "state.json")["config_hash"] != config_hash:
        raise ConfigMismatchError(
            f"checkpoints in {_root(output_dir)} were made with a different config or data; "
            "run with --restart to discard them, or set a new output.dir"
        )
    return latest


def save_best(output_dir, model, eval_logits: Any, eval_targets: Any, state: TrainState) -> None:
    """eval_logits/eval_targets: any JSON-serializable eval output, returned as-is by load_best."""
    root = _root(output_dir)
    final = root / "best"
    tmp = root / "best.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    save_file(trainable_state_dict(model), str(tmp / "trainable.safetensors"))
    _write_json(tmp / "eval.json", {"logits": eval_logits, "targets": eval_targets})
    _write_json(tmp / "state.json", asdict(state))
    _replace_dir(tmp, final)


def load_best(output_dir, model) -> tuple[Any, Any]:
    best = _root(output_dir) / "best"
    load_trainable_state(model, load_file(str(best / "trainable.safetensors")))
    data = _read_json(best / "eval.json")
    return data["logits"], data["targets"]


def save_baseline(output_dir, metrics: dict) -> None:
    root = _root(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "baseline.json", metrics)


def load_baseline(output_dir) -> dict | None:
    path = _root(output_dir) / "baseline.json"
    return _read_json(path) if path.is_file() else None


def clear_checkpoints(output_dir) -> None:
    shutil.rmtree(_root(output_dir), ignore_errors=True)
