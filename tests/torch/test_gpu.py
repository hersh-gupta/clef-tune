"""Opt-in: QLoRA-trains the real Clef-Flash on a >=16 GB GPU (downloads about 19 GB).

Run with: uv run pytest tests/torch/test_gpu.py -m gpu -v
"""

from __future__ import annotations

import csv
import itertools
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from clef_tune import Predictor  # noqa: E402
from clef_tune.config import load_config  # noqa: E402
from clef_tune.train import train  # noqa: E402
from clef_tune.yamlio import dump_yaml, load_yaml_file  # noqa: E402

pytestmark = [pytest.mark.torch, pytest.mark.gpu]

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
MIN_MEMORY = 16e9


def _finite(value) -> bool:
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return True
    return math.isfinite(value)


def _write_project(project: Path) -> tuple[Path, list[dict]]:
    project.mkdir()
    with (EXAMPLES / "support_tickets.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = [row for row in itertools.islice(reader, 50)]
    with (project / "train.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    example = load_yaml_file(EXAMPLES / "support_tickets.yaml")
    config = {
        "run_name": "gpu",
        "model": {"id": "Cloudflare/clef-flash"},
        "data": {**example["data"], "train": "train.csv", "eval_fraction": 0.2},
        "schema": example["schema"],
        "train": {
            "quantization": "4bit",
            "mode": "lora",
            "epochs": 1,
            "max_length": 1024,
            "max_tokens_per_batch": 1024,
        },
        "eval": {"eval_every": 1000},
        "output": {"dir": "out"},
    }
    path = project / "config.yaml"
    path.write_text(dump_yaml(config), encoding="utf-8")
    load_config(path)  # fail early on a malformed config
    return path, rows


def test_clef_flash_qlora_trains_and_predicts(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    if torch.cuda.get_device_properties(0).total_memory < MIN_MEMORY:
        pytest.skip("needs a GPU with at least 16 GB of memory")
    config, rows = _write_project(tmp_path / "project")

    result = train(config)

    assert (result.adapter_dir / "joint_head.safetensors").exists()
    assert result.metrics["final"] and _finite(result.metrics["final"])
    predictor = Predictor.load(result.adapter_dir)
    state_columns = ["subject", "body", "customer_tier"]
    out = predictor.predict(
        [{"id": r["ticket_id"], "state": {c: r[c] for c in state_columns}} for r in rows[:3]]
    )
    assert len(out) == 3
    for item in out:
        assert set(item["answers"]) == {"department", "urgency", "outage"}
