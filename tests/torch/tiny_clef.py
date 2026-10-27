"""A tiny random Clef in the exact release layout: sharded safetensors + index, joint head,
Clef's own joint_schema_model.py, tokenizer and processor files, LICENSE."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from clef_tune.yamlio import dump_yaml

FIXTURES = Path(__file__).parent / "fixtures"
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
]
TINY_HEAD = {
    "hidden_size": 64,
    "width": 32,
    "routing_layers": 1,
    "layers": 1,
    "heads": 2,
    "feedforward": 64,
}
TINY_SCHEMA = {
    "questions": {
        "outage": {"type": "noul", "instructions": "Is a service down?"},
        "department": {"type": "choice", "criteria": {"billing": "Money", "technical": "Bugs"}},
    }
}
_CORPUS = [
    "Read the complete state and schema. Decide every field jointly.",
    "SCHEMA FIELDS",
    "OPTION",
    "END FIELD",
    '{"option_id":"true","description":"The proposition is true"}',
    "JOINT SCHEMA DECISIONS:",
    "<think>",
    "the site is down fine invoice server matters",
]


def build_tiny_release(out_dir: Path, seed: int = 0) -> Path:
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    from clef_tune.clef_code import load_clef_code

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(_CORPUS * 20, trainer)
    tokenizer.save(str(out_dir / "tokenizer.json"))
    for name in (
        "tokenizer_config.json",
        "processor_config.json",
        "chat_template.jinja",
        "LICENSE",
        "joint_schema_model.py",
    ):
        shutil.copyfile(FIXTURES / name, out_dir / name)
    token = tokenizer.token_to_id
    config = Qwen3_5Config(
        text_config=dict(
            vocab_size=tokenizer.get_vocab_size(),
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            layer_types=["linear_attention", "full_attention"],
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
            max_position_embeddings=4096,
            tie_word_embeddings=False,
            eos_token_id=token("<|im_end|>"),
            rope_parameters={
                "mrope_interleaved": True,
                "mrope_section": [2, 2, 0],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default",
            },
        ),
        vision_config=dict(
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            out_hidden_size=64,
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            num_position_embeddings=64,
            deepstack_visual_indexes=[],
        ),
        image_token_id=token("<|image_pad|>"),
        video_token_id=token("<|video_pad|>"),
        vision_start_token_id=token("<|vision_start|>"),
        vision_end_token_id=token("<|vision_end|>"),
        tie_word_embeddings=False,
    )
    torch.manual_seed(seed)
    Qwen3_5ForConditionalGeneration(config).save_pretrained(out_dir, max_shard_size="200KB")
    head = load_clef_code(out_dir).JointSchemaHead(**TINY_HEAD)
    save_file(
        {k: v.contiguous() for k, v in head.state_dict().items()},
        out_dir / "joint_head.safetensors",
    )
    (out_dir / "joint_head_config.json").write_text(json.dumps(TINY_HEAD, indent=2))
    return out_dir


def _rows(prefix: str) -> list[dict[str, str]]:
    rows = []
    for i in range(16):
        down = i % 2 == 0
        invoice = (i // 2) % 2 == 0
        rows.append(
            {
                "id": f"{prefix}{i}",
                "subject": f"ticket {i}",
                "body": f"the site is {'down' if down else 'fine'} and the "
                f"{'invoice' if invoice else 'server'} matters",
                "outage": "yes" if down else "no",
                "department": "billing" if invoice else "technical",
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def write_tiny_project(project_dir: Path, release_dir: Path, overrides: dict | None = None) -> Path:
    project_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(project_dir / "train.csv", _rows("t"))
    _write_csv(project_dir / "eval.csv", _rows("e"))
    config: dict[str, Any] = {
        "run_name": "tiny",
        "model": {"id": "tiny/clef", "local_path": str(Path(release_dir).resolve())},
        "data": {
            "train": "train.csv",
            "eval": "eval.csv",
            "id_column": "id",
            "state_columns": ["subject", "body"],
        },
        "schema": TINY_SCHEMA,
        "train": {
            "mode": "lora",
            "lora": {"r": 8, "alpha": 16, "dropout": 0.05},
            "lora_lr": 3.0e-3,
            "head_lr": 3.0e-3,
            "epochs": 24,
            "max_length": 1024,
            "max_tokens_per_batch": 1600,
            "grad_accum_steps": 1,
            "seed": 0,
        },
        "eval": {"eval_every": 24, "select_metric": "exact_match", "calibrate": True},
        "output": {"dir": "out", "save_every": 1000},
    }
    config = _merge(config, overrides or {})
    path = project_dir / "config.yaml"
    path.write_text(dump_yaml(config), encoding="utf-8")
    return path
