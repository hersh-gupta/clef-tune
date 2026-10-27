"""Load Clef for training: backbone, LoRA or head-only, fp32 head, trainable state."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

from clef_tune.clef_code import load_clef_code
from clef_tune.config import TrainSection

# All 12 Linear kinds in Qwen3.5 decoder layers; the vision tower uses other names.
LORA_TARGET_MODULES = (
    r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj"
    r"|linear_attn\.(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)"
    r"|mlp\.(gate|up|down)_proj)"
)
DEVICE_ENV = "CLEF_TUNE_DEVICE"


def default_device() -> torch.device:
    if DEVICE_ENV in os.environ:
        return torch.device(os.environ[DEVICE_ENV])
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32


def quantization_config(quantization: str) -> BitsAndBytesConfig | None:
    if quantization != "4bit":
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["lm_head"],
    )


def load_backbone(
    model_path: str | Path, *, quantization: str, device: torch.device, dtype: torch.dtype
) -> Qwen3_5ForConditionalGeneration:
    kwargs: dict[str, Any] = {}
    quant = quantization_config(quantization)
    if quant is not None:
        kwargs["quantization_config"] = quant
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
        Path(model_path), dtype=dtype, device_map={"": str(device)}, **kwargs
    )
    backbone.config.use_cache = False
    for param in backbone.parameters():
        param.requires_grad = False
    return backbone


def load_head(
    model_path: str | Path, clef_code: ModuleType, *, device: torch.device, dtype: torch.dtype
) -> tuple[Any, dict]:
    path = Path(model_path)
    head_config = json.loads((path / "joint_head_config.json").read_text())
    head = clef_code.JointSchemaHead(**head_config)
    head.load_state_dict(load_file(path / "joint_head.safetensors"), strict=True)
    return head.to(device=device, dtype=dtype), head_config


@dataclass
class LoadedClef:
    model: Any
    processor: Any
    clef_code: ModuleType
    model_path: Path
    head_config: dict


def load_for_training(
    model_path: str | Path, train: TrainSection, *, device: torch.device
) -> LoadedClef:
    path = Path(model_path)
    clef_code = load_clef_code(path)
    backbone = load_backbone(
        path, quantization=train.quantization, device=device, dtype=compute_dtype(device)
    )
    if train.mode == "lora":
        from peft import LoraConfig, get_peft_model

        if train.gradient_checkpointing:
            backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        backbone = get_peft_model(
            backbone,
            LoraConfig(
                r=train.lora.r,
                lora_alpha=train.lora.alpha,
                lora_dropout=train.lora.dropout,
                target_modules=LORA_TARGET_MODULES,
            ),
        )
    head, head_config = load_head(path, clef_code, device=device, dtype=torch.float32)
    for param in head.parameters():
        param.requires_grad = True
    model = clef_code.ClefModel(backbone, head)
    processor = AutoProcessor.from_pretrained(path)
    return LoadedClef(model, processor, clef_code, path, head_config)


def trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def load_trainable_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    missing = sorted(set(params) - set(state))
    extra = sorted(set(state) - set(params))
    if missing or extra:
        raise KeyError(f"trainable state mismatch: missing={missing[:5]} extra={extra[:5]}")
    with torch.no_grad():
        for name, param in params.items():
            param.copy_(state[name].to(device=param.device, dtype=param.dtype))
