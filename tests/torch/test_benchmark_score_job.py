"""The benchmark's scoring job (benchmarks/cfpb/score_job.py) on the tiny model: the "out of the
box" adapter it saves must answer exactly as the model training starts from."""

from __future__ import annotations

import csv
import json

import pytest
import torch
from typer.testing import CliRunner

from benchmarks.cfpb import score_job
from clef_tune.calibration import softmax
from clef_tune.cli import app
from clef_tune.config import load_config
from clef_tune.data import load_file
from clef_tune.encoding import prepare_items
from clef_tune.evaluation import predict_logits
from clef_tune.model import load_for_training
from tests.torch.tiny_clef import write_tiny_project


def test_untrained_adapter_answers_like_the_training_baseline(tmp_path, tiny_release, monkeypatch):
    config_path = write_tiny_project(tmp_path / "project", tiny_release)
    base = tmp_path / "base"
    score_job.base(str(config_path), str(base))
    monkeypatch.setattr(score_job, "MODEL_MOUNT", str(tiny_release))
    eval_csv = tmp_path / "project" / "eval.csv"
    out_csv, timing = tmp_path / "predictions.csv", tmp_path / "timing.json"
    score_job.predict(str(base), str(eval_csv), str(out_csv), str(timing))

    # What training starts from: the release with untouched LoRA weights, before any update.
    config = load_config(config_path)
    loaded = load_for_training(tiny_release, config.train, device=torch.device("cpu"))
    records = load_file(config, str(eval_csv)).records
    items = prepare_items(loaded.clef_code, loaded.processor, records, config.train.max_length)
    logits = predict_logits(
        loaded.model.eval(),
        items,
        loaded.clef_code,
        loaded.processor,
        max_length=config.train.max_length,
        max_tokens_per_batch=config.train.max_tokens_per_batch,
        device=torch.device("cpu"),
    )
    with out_csv.open(encoding="utf-8", newline="") as rows:
        written = list(csv.DictReader(rows))
    assert len(written) == len(records) == json.loads(timing.read_text())["rows"]
    for row, item_logits in zip(written, logits, strict=True):
        probs = softmax(item_logits["department"])
        options = ["billing", "technical"]
        assert row["department_predicted"] == options[max(range(2), key=probs.__getitem__)]
        assert float(row["department_confidence"]) == pytest.approx(max(probs), abs=1e-3)


def test_clef_tune_eval_reads_the_untrained_adapter(tmp_path, tiny_release):
    config_path = write_tiny_project(tmp_path / "project", tiny_release)
    score_job.base(str(config_path), str(tmp_path / "base"))
    report = tmp_path / "eval.json"
    result = CliRunner().invoke(
        app,
        [
            "eval",
            str(tmp_path / "base"),
            str(tmp_path / "project" / "eval.csv"),
            "--output",
            str(report),
        ],
    )
    assert result.exit_code == 0, result.output
    assert set(json.loads(report.read_text())["questions"]) == {"outage", "department"}
