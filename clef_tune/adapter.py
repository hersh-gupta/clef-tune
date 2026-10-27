"""Portable adapter directories: save after training, read back without torch."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import clef_tune
from clef_tune.card import render_card
from clef_tune.config import Config, load_config
from clef_tune.schema import Schema
from clef_tune.yamlio import dump_yaml

LORA_DIR = "adapter"
HEAD_FILE = "joint_head.safetensors"
HEAD_CONFIG_FILE = "joint_head_config.json"
CALIBRATION_FILE = "calibration.json"
SCHEMA_FILE = "schema.yaml"
CONFIG_FILE = "train_config.yaml"
METRICS_FILE = "metrics.json"
CARD_FILE = "README.md"


@dataclass
class AdapterInfo:
    path: Path
    config: Config
    schema: Schema | None
    calibration: dict[str, float]
    head_config: dict
    has_lora: bool


def save_adapter(
    out_dir: str | Path,
    model: Any,
    head_config: dict,
    config: Config,
    *,
    calibration: dict[str, float],
    metrics: dict,
) -> Path:
    from peft import PeftModel
    from safetensors.torch import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if isinstance(model.language_model, PeftModel):
        model.language_model.save_pretrained(out / LORA_DIR)
    state = {k: v.detach().float().cpu().contiguous() for k, v in model.head.state_dict().items()}
    save_file(state, out / HEAD_FILE)
    (out / HEAD_CONFIG_FILE).write_text(json.dumps(head_config, indent=2), encoding="utf-8")
    (out / CALIBRATION_FILE).write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    schema = config.get_schema()
    if schema is not None:
        schema.save(out / SCHEMA_FILE)

    revision = config.model.resolved_revision()
    data = config.to_dict()
    data.setdefault("model", {})["revision"] = revision
    header = f"# clef-tune {clef_tune.__version__}\n# base model: {config.model.id} @ {revision}\n"
    (out / CONFIG_FILE).write_text(header + dump_yaml(data), encoding="utf-8")
    (out / METRICS_FILE).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / CARD_FILE).write_text(
        render_card(metrics, config, clef_tune.__version__), encoding="utf-8"
    )
    return out


def read_adapter(path: str | Path) -> AdapterInfo:
    path = Path(path)
    config = load_config(path / CONFIG_FILE)
    schema = Schema.load(path / SCHEMA_FILE) if (path / SCHEMA_FILE).exists() else None
    return AdapterInfo(
        path=path,
        config=config,
        schema=schema,
        calibration=json.loads((path / CALIBRATION_FILE).read_text(encoding="utf-8")),
        head_config=json.loads((path / HEAD_CONFIG_FILE).read_text(encoding="utf-8")),
        has_lora=(path / LORA_DIR / "adapter_config.json").exists(),
    )
