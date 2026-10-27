"""Training configuration (config.yaml): models, presets, path resolution and hashing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from clef_tune.schema import Schema
from clef_tune.yamlio import dump_yaml, load_yaml_file

PINNED_REVISIONS = {
    "Cloudflare/clef": "a30162b99d68a481d8bd75d38df069a0cf83df40",
    "Cloudflare/clef-flash": "f0b7d3cd65296abfe459e6c228b618f5d8b26d2f",
}

PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "train.mode": "head_only",
        "train.epochs": 1,
        "train.max_length": 2048,
        "eval.early_stopping_patience": None,
    },
    "balanced": {
        "train.mode": "lora",
        "train.lora.r": 16,
        "train.lora.alpha": 32,
        "train.epochs": 3,
        "train.max_length": 4096,
        "eval.early_stopping_patience": None,
    },
    "thorough": {
        "train.mode": "lora",
        "train.lora.r": 32,
        "train.lora.alpha": 64,
        "train.epochs": 5,
        "train.max_length": 4096,
        "eval.early_stopping_patience": 3,
    },
}


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(_Section):
    id: str = "Cloudflare/clef"
    revision: str | None = None
    local_path: str | None = None

    def resolved_revision(self) -> str:
        return self.revision or PINNED_REVISIONS.get(self.id, "main")


class DataConfig(_Section):
    train: str
    eval: str | None = None
    eval_fraction: float = Field(0.1, gt=0, lt=1)
    id_column: str | None = None
    state_columns: list[str] | None = None  # None: every column not used for id/labels/images
    label_columns: dict[str, str] = Field(default_factory=dict)
    image_columns: list[str] = Field(default_factory=list)


class LoraConfig(_Section):
    r: int = Field(16, ge=1)
    alpha: int = Field(32, ge=1)
    dropout: float = Field(0.05, ge=0, lt=1)


class TrainSection(_Section):
    mode: Literal["lora", "head_only"] = "lora"
    quantization: Literal["none", "4bit"] = "none"
    lora: LoraConfig = Field(default_factory=LoraConfig)
    lora_lr: float = Field(1e-4, gt=0)
    head_lr: float = Field(5e-5, gt=0)
    epochs: int = Field(3, ge=1)
    max_length: int = Field(4096, ge=256, le=16384)
    max_tokens_per_batch: int = Field(16384, ge=256)
    grad_accum_steps: int = Field(4, ge=1)
    warmup_ratio: float = Field(0.03, ge=0, lt=1)
    weight_decay: float = Field(0.0, ge=0)
    label_smoothing: float = Field(0.0, ge=0, lt=1)
    class_weighting: Literal["none", "balanced"] = "none"
    question_weights: dict[str, float] = Field(default_factory=dict)
    gradient_checkpointing: bool = True
    seed: int = 42


class EvalSection(_Section):
    eval_every: int = Field(200, ge=1)
    select_metric: Literal["mean_macro_f1", "exact_match", "mean_log_loss"] = "mean_macro_f1"
    early_stopping_patience: int | None = Field(None, ge=1)
    calibrate: bool = True


class OutputSection(_Section):
    dir: str | None = None  # default: out/<run_name> next to the config
    save_every: int = Field(200, ge=1)
    push_to_hub: str | None = None
    report_to: list[Literal["tensorboard", "wandb"]] = Field(default_factory=list)


class Config(_Section):
    model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

    run_name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig
    schema_def: dict[str, Any] | None = Field(default=None, alias="schema")
    schema_file: str | None = None
    train: TrainSection = Field(default_factory=TrainSection)
    eval: EvalSection = Field(default_factory=EvalSection)
    output: OutputSection = Field(default_factory=OutputSection)
    _base_dir: Path = PrivateAttr(default_factory=Path.cwd)

    @model_validator(mode="after")
    def _one_schema_source(self) -> Config:
        if self.schema_def is not None and self.schema_file is not None:
            raise ValueError("set either 'schema' or 'schema_file', not both")
        if self.schema_def is not None:
            Schema.from_dict(self.schema_def)  # fail fast with a clear message
        return self

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def with_base_dir(self, base_dir: str | Path) -> Config:
        self._base_dir = Path(base_dir).resolve()
        return self

    def resolve_path(self, value: str) -> Path:
        """Resolve a config path against the config's directory (accepts Windows separators)."""
        path = Path(value.replace("\\", "/"))
        return path if path.is_absolute() else self._base_dir / path

    def get_schema(self) -> Schema | None:
        if self.schema_def is not None:
            return Schema.from_dict(self.schema_def)
        if self.schema_file is not None:
            return Schema.load(self.resolve_path(self.schema_file))
        return None

    def output_dir(self) -> Path:
        return self.resolve_path(self.output.dir or f"out/{self.run_name}")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)

    def to_yaml(self) -> str:
        return dump_yaml(self.to_dict())

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_yaml(), encoding="utf-8")

    def with_preset(self, name: str) -> Config:
        if name not in PRESETS:
            raise ValueError(f"unknown preset {name!r} (choose from {', '.join(PRESETS)})")
        data = self.to_dict()
        for dotted, value in PRESETS[name].items():
            *parents, leaf = dotted.split(".")
            section = data
            for parent in parents:
                section = section.setdefault(parent, {})
            section[leaf] = value
        return Config.model_validate(data).with_base_dir(self._base_dir)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    data = load_yaml_file(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: config must be a YAML mapping")
    return Config.model_validate(data).with_base_dir(path.parent)


def file_digest(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def config_hash(config: Config, files: Iterable[str | Path] = ()) -> str:
    """Hash of everything that changes training results (not run name, output, or weight paths)."""
    payload = config.to_dict()
    payload.pop("run_name", None)
    payload.pop("output", None)
    payload.pop("schema_file", None)
    payload["model"]["revision"] = config.model.resolved_revision()
    payload["model"].pop("local_path", None)
    schema = config.get_schema()
    payload["schema"] = schema.to_dict() if schema else None
    payload["files"] = sorted(file_digest(path) for path in files)
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
