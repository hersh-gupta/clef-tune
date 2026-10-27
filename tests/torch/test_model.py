import re

import pytest

torch = pytest.importorskip("torch")
from clef_tune.config import TrainSection  # noqa: E402
from clef_tune.model import (  # noqa: E402
    compute_dtype,
    default_device,
    load_for_training,
    load_trainable_state,
    quantization_config,
    trainable_state_dict,
)

pytestmark = pytest.mark.torch
CPU = torch.device("cpu")
KINDS = {
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.in_proj_b",
    "linear_attn.in_proj_a",
    "linear_attn.out_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
}


def test_lora_targets_only_language_model_linears(tiny_release):
    loaded = load_for_training(tiny_release, TrainSection(lora={"r": 4, "alpha": 8}), device=CPU)
    names = [n for n, p in loaded.model.named_parameters() if p.requires_grad and "lora_" in n]
    assert {re.sub(r".*layers\.\d+\.", "", n).split(".lora_")[0] for n in names} == KINDS
    assert not any("visual" in n or "lm_head" in n for n in names)
    text_model = loaded.model.language_model.get_base_model().model.language_model
    assert text_model.gradient_checkpointing


def test_head_only_freezes_the_backbone_and_trains_an_fp32_head(tiny_release):
    loaded = load_for_training(tiny_release, TrainSection(mode="head_only"), device=CPU)
    trainable = [n for n, p in loaded.model.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith("head.") for n in trainable)
    assert all(p.dtype == torch.float32 for p in loaded.model.head.parameters())
    assert not hasattr(loaded.model.language_model, "get_base_model")


def test_quantization_config():
    assert quantization_config("none") is None
    q = quantization_config("4bit")
    assert q.load_in_4bit and q.bnb_4bit_quant_type == "nf4" and q.bnb_4bit_use_double_quant
    assert q.bnb_4bit_compute_dtype == torch.bfloat16 and q.llm_int8_skip_modules == ["lm_head"]


def test_compute_dtype():
    assert compute_dtype(CPU) == torch.float32
    assert compute_dtype(torch.device("cuda")) == torch.bfloat16


def test_device_env_overrides_the_default(monkeypatch):
    monkeypatch.setenv("CLEF_TUNE_DEVICE", "cpu")
    assert default_device() == CPU


def test_trainable_state_round_trip(tiny_release):
    loaded = load_for_training(tiny_release, TrainSection(lora={"r": 4, "alpha": 8}), device=CPU)
    saved = trainable_state_dict(loaded.model)
    with torch.no_grad():
        for p in loaded.model.parameters():
            if p.requires_grad:
                p.add_(1.0)
    load_trainable_state(loaded.model, saved)
    for name, p in loaded.model.named_parameters():
        if p.requires_grad:
            assert torch.equal(saved[name], p.detach().cpu())
    with pytest.raises(KeyError):
        load_trainable_state(loaded.model, {**saved, "bogus": torch.zeros(1)})
