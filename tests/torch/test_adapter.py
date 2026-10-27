from __future__ import annotations

import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from clef_tune import clef_code  # noqa: E402
from clef_tune.adapter import CONFIG_FILE, export_merged, read_adapter, save_adapter  # noqa: E402
from clef_tune.config import load_config  # noqa: E402
from clef_tune.model import load_for_training  # noqa: E402
from clef_tune.predictor import Predictor, predict_file  # noqa: E402
from clef_tune.train import train  # noqa: E402
from clef_tune.yamlio import dump_yaml, load_yaml_file  # noqa: E402
from tests.test_card import METRICS  # noqa: E402
from tests.torch.test_predictor import ROWS  # noqa: E402
from tests.torch.tiny_clef import TINY_HEAD, TINY_SCHEMA  # noqa: E402

pytestmark = pytest.mark.torch
CPU = torch.device("cpu")


def moved_adapter(adapter: Path, dest: Path, local_path: str) -> Path:
    """A copy of the adapter whose recorded model.local_path is `local_path`."""
    shutil.copytree(adapter, dest)
    data = load_yaml_file(dest / CONFIG_FILE)
    data["model"]["local_path"] = local_path
    (dest / CONFIG_FILE).write_text(dump_yaml(data), encoding="utf-8")
    return dest


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


def test_adapter_trained_with_a_relative_local_path_loads(tiny_project, tiny_release, tmp_path):
    shutil.copytree(tiny_release, tmp_path / "run" / "weights")
    config = tiny_project(model={"local_path": "weights"}, train={"epochs": 1})
    result = train(config)
    local_path = read_adapter(result.adapter_dir).config.model.local_path
    assert Path(local_path) == (config.parent / "weights").resolve()
    answers = Predictor.load(result.adapter_dir, device="cpu").predict(ROWS)[0]["answers"]
    assert set(answers) == {"outage", "department"}


def test_adapter_trained_with_a_schema_file_is_self_contained(tiny_project, tmp_path):
    config = tiny_project(train={"epochs": 1})
    (config.parent / "schemas").mkdir()
    (config.parent / "schemas" / "tickets.yaml").write_text(
        dump_yaml(TINY_SCHEMA), encoding="utf-8"
    )
    data = load_yaml_file(config)
    del data["schema"]
    data["schema_file"] = "schemas/tickets.yaml"
    config.write_text(dump_yaml(data), encoding="utf-8")
    result = train(config)
    info = read_adapter(result.adapter_dir)
    assert info.config.schema_file is None
    assert info.config.get_schema() == load_config(config).get_schema()
    (tmp_path / "in.csv").write_text("id,subject,body\n1,a,down\n", encoding="utf-8")
    predictor = Predictor.load(result.adapter_dir, device="cpu")
    assert predict_file(predictor, tmp_path / "in.csv", tmp_path / "out.jsonl") == 1


def test_a_missing_local_path_falls_back_to_the_hub_snapshot(
    adapter, tiny_release, tmp_path, monkeypatch, capsys
):
    moved = moved_adapter(adapter, tmp_path / "adapter", "/model")  # an HF Jobs mount
    calls = []

    def fake_snapshot_download(repo_id, revision=None, **kwargs):
        calls.append((repo_id, revision))
        return str(tiny_release)

    monkeypatch.setattr(clef_code, "snapshot_download", fake_snapshot_download)
    Predictor.load(moved, device="cpu").predict(ROWS)
    assert calls == [("tiny/clef", "main")]
    assert "/model" in capsys.readouterr().err


def test_base_model_path_overrides_the_recorded_model_location(
    adapter, tiny_release, tmp_path, monkeypatch
):
    moved = moved_adapter(adapter, tmp_path / "adapter", "/model")
    monkeypatch.setattr(
        clef_code, "snapshot_download", lambda *a, **k: pytest.fail("downloaded the base model")
    )
    predictor = Predictor.load(moved, device="cpu", base_model_path=tiny_release)
    assert set(predictor.predict(ROWS)[0]["answers"]) == {"outage", "department"}
    out = export_merged(
        moved,
        tmp_path / "merged",
        dtype=torch.float32,
        max_shard_size="200KB",
        base_model_path=tiny_release,
    )
    assert (out / "joint_head.safetensors").exists() and (out / "config.json").exists()
