import json
from pathlib import Path

from clef_tune.card import render_card
from clef_tune.config import load_config

RUN = Path(__file__).parent / "fixtures" / "recorded_run"
EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "support_tickets.yaml"
TRAIN_KEYS = {
    "event",
    "step",
    "loss",
    "lr",
    "tokens_per_sec",
    "eta_s",
    "projected_cost",
    "total_steps",
}


def test_the_recorded_run_has_the_trainers_shapes():
    events = [json.loads(line) for line in (RUN / "metrics.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "baseline" and events[-1]["event"] == "done"
    assert all(set(e) == TRAIN_KEYS for e in events if e["event"] == "train")
    assert [e["step"] for e in events if e["event"] == "eval"] == [40, 80, 120]
    metrics = json.loads((RUN / "metrics.json").read_text())
    assert set(metrics) == {"baseline", "final", "calibration", "data", "run"}
    assert "## Confusion matrices" in render_card(metrics, load_config(EXAMPLE), "0.0.0")
