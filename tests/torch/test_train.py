import json

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import load_file  # noqa: E402

from clef_tune import train as train_module  # noqa: E402
from clef_tune.train import TrainingError, train  # noqa: E402

pytestmark = pytest.mark.torch


def events(config_path):
    lines = (config_path.parent / "out" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_lora_overfits_sixteen_rows(tiny_project):
    config = tiny_project()
    result = train(config)
    final, baseline = result.metrics["final"], result.metrics["baseline"]
    assert final["exact_match"] == 1.0
    assert final["mean_log_loss"] < baseline["mean_log_loss"]
    assert result.adapter_dir == config.parent / "out" / "final"
    assert json.loads((result.adapter_dir / "calibration.json").read_text()) == {
        "outage": 1.0,
        "department": 1.0,
    }  # 16 eval rows < 20: T stays 1
    kinds = [e["event"] for e in events(config)]
    assert kinds[0] == "baseline" and kinds[-1] == "done" and "eval" in kinds
    train_events = [e for e in events(config) if e["event"] == "train"]
    assert {"step", "loss", "lr", "tokens_per_sec", "eta_s", "projected_cost"} <= set(
        train_events[-1]
    )
    assert train_events[-1]["step"] == result.metrics["run"]["steps"] >= 24 * 4
    assert train_events[-1]["eta_s"] is not None
    assert not (config.parent / "out" / "checkpoints").exists()


def test_head_only_trains_only_the_head(tiny_project, tiny_release):
    before = load_file(tiny_release / "joint_head.safetensors")
    result = train(tiny_project(train={"mode": "head_only", "epochs": 2}))
    after = load_file(result.adapter_dir / "joint_head.safetensors")
    assert any(not torch.allclose(before[k].float(), after[k]) for k in before)
    assert not (result.adapter_dir / "adapter").exists()


def test_balanced_class_weighting_trains(tiny_project):
    result = train(tiny_project(train={"class_weighting": "balanced", "epochs": 2}))
    assert result.metrics["run"]["class_weighting"] == "balanced"
    assert "Balanced class weighting" in (result.adapter_dir / "README.md").read_text(
        encoding="utf-8"
    )


def test_early_stopping_stops_when_the_metric_stalls(tiny_project):
    config = tiny_project(
        train={"lora_lr": 1e-12, "head_lr": 1e-12, "epochs": 2},
        eval={"eval_every": 1, "early_stopping_patience": 2},
    )
    run = train(config).metrics["run"]
    assert run["stopped_early"] and run["steps"] == 3 and run["best_step"] == 1


def test_price_and_estimate_feed_projected_and_actual_costs(tiny_project, monkeypatch):
    monkeypatch.setenv("CLEF_TUNE_PRICE_PER_HOUR", "3.6")
    monkeypatch.setenv("CLEF_TUNE_ESTIMATED_HOURS", "0.5")
    config = tiny_project(train={"epochs": 5})  # at least 20 steps, so the ETA starts
    run = train(config).metrics["run"]
    assert run["price_per_hour"] == 3.6 and run["estimated_hours"] == 0.5
    assert run["estimated_cost"] == pytest.approx(1.8)
    assert run["cost"] == pytest.approx(run["hours"] * 3.6)
    last = [e for e in events(config) if e["event"] == "train"][-1]
    assert last["eta_s"] == 0.0 and last["projected_cost"] > 0


def test_validation_errors_stop_before_the_model_loads(tiny_project, monkeypatch):
    config = tiny_project()
    csv = config.parent / "train.csv"
    csv.write_text(
        csv.read_text(encoding="utf-8").replace(",billing", ",bogus", 1), encoding="utf-8"
    )
    monkeypatch.setattr(
        train_module,
        "load_for_training",
        lambda *a, **k: pytest.fail("model loaded despite validation errors"),
    )
    with pytest.raises(TrainingError, match="validation failed"):
        train(config)
