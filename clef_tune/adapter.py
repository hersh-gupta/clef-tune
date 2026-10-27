"""Portable adapter directories: save after training, read back without torch."""

from __future__ import annotations

import json
import shutil
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
RELEASE_COPY_FILES = (
    "joint_schema_model.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "chat_template.jinja",
    "LICENSE",
)

EXPORT_README = """# {name}: merged Clef release checkpoint

Fine-tuned with clef-tune from `{base}`. This directory uses Cloudflare's release
layout (merged backbone shards, `joint_head.safetensors`, `joint_schema_model.py`),
so Clef's `load_release_model()` and SGLang load it unchanged.

## Calibration

SGLang serves **uncalibrated** probabilities. Per-question temperatures are stored
in `calibration.json` (question id -> T; a question not listed uses T = 1). A client
applies them to the served probabilities of each question's options:

    p_i^(1/T) / Σ_j p_j^(1/T)

where p_i is the served probability of option i and the sum runs over that
question's options.
"""


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


def export_merged(
    adapter: str | Path,
    out_dir: str | Path,
    *,
    dtype: Any = None,
    device: str = "cpu",
    max_shard_size: str = "5GB",
) -> Path:
    """Write a merged checkpoint in Cloudflare's release layout (loads in Clef and SGLang)."""
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file, save_file

    from clef_tune.clef_code import model_dir
    from clef_tune.model import load_backbone

    dtype = torch.bfloat16 if dtype is None else dtype
    info = read_adapter(adapter)
    base = model_dir(info.config)
    backbone = load_backbone(base, quantization="none", device=torch.device(device), dtype=dtype)
    if info.has_lora:
        backbone = PeftModel.from_pretrained(backbone, info.path / LORA_DIR).merge_and_unload()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    backbone.save_pretrained(out, max_shard_size=max_shard_size)

    head = {k: v.to(dtype).contiguous() for k, v in load_file(info.path / HEAD_FILE).items()}
    save_file(head, out / HEAD_FILE, metadata={"format": "pt"})
    shutil.copyfile(info.path / HEAD_CONFIG_FILE, out / HEAD_CONFIG_FILE)
    for name in RELEASE_COPY_FILES:
        if (base / name).is_file():
            shutil.copyfile(base / name, out / name)
    (out / CALIBRATION_FILE).write_text(json.dumps(info.calibration, indent=2), encoding="utf-8")
    readme = EXPORT_README.format(name=out.name, base=info.config.model.id)
    (out / CARD_FILE).write_text(readme, encoding="utf-8")
    return out
