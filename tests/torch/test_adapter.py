from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from clef_tune.adapter import read_adapter, save_adapter  # noqa: E402
from clef_tune.config import load_config  # noqa: E402
from clef_tune.model import load_for_training  # noqa: E402
from tests.test_card import METRICS  # noqa: E402
from tests.torch.tiny_clef import TINY_HEAD  # noqa: E402

pytestmark = pytest.mark.torch
CPU = torch.device("cpu")


def test_save_and_read_adapter_round_trip(tiny_release, tiny_project, tmp_path):
    cfg = load_config(tiny_project())
    loaded = load_for_training(tiny_release, cfg.train, device=CPU)
    out = save_adapter(
        tmp_path / "final",
        loaded.model,
        loaded.head_config,
        cfg,
        calibration={"outage": 1.5},
        metrics=METRICS,
    )
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [
            "README.md",
            "adapter",
            "calibration.json",
            "joint_head.safetensors",
            "joint_head_config.json",
            "metrics.json",
            "schema.yaml",
            "train_config.yaml",
        ]
    )
    assert (out / "adapter" / "adapter_model.safetensors").exists()
    assert (out / "train_config.yaml").read_text(encoding="utf-8").startswith("# clef-tune ")
    info = read_adapter(out)
    assert info.has_lora and info.calibration == {"outage": 1.5}
    assert info.schema == cfg.get_schema() and info.head_config == TINY_HEAD
    assert info.config.model.revision == cfg.model.resolved_revision()


def test_head_only_adapter_has_no_lora_dir(tiny_release, tiny_project, tmp_path):
    cfg = load_config(tiny_project(train={"mode": "head_only"}))
    loaded = load_for_training(tiny_release, cfg.train, device=CPU)
    out = save_adapter(
        tmp_path / "final", loaded.model, loaded.head_config, cfg, calibration={}, metrics=METRICS
    )
    assert not (out / "adapter").exists() and not read_adapter(out).has_lora
