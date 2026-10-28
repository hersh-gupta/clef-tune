from __future__ import annotations

import yaml

from clef_tune.card import confusion_markdown, render_card
from clef_tune.config import PINNED_REVISIONS
from tests.helpers import make_config


def _question(qid, qtype, labels, matrix, **values):
    base = {"accuracy": None, "f1": None, "log_loss": None, "mae": None}
    base.update(values)
    return {
        "question_id": qid,
        "question_type": qtype,
        "n": 10,
        **base,
        "confusion": {"labels": labels, "matrix": matrix},
    }


def _report(department_accuracy):
    return {
        "questions": {
            "department": _question(
                "department",
                "choice",
                ["billing", "technical"],
                [[3, 1], [1, 5]],
                accuracy=department_accuracy,
                f1=0.4,
                log_loss=0.9,
            ),
            "outage": _question(
                "outage",
                "noul",
                ["no", "yes"],
                [[6, 1], [1, 2]],
                accuracy=0.8,
                f1=0.6,
                log_loss=0.5,
            ),
        },
        "exact_match": 0.3,
        "mean_macro_f1": 0.5,
        "mean_log_loss": 0.7,
        "n_records": 10,
    }


def _calibration(t):
    return {
        "temperature": t,
        "n": 10,
        "ece_before": 0.2,
        "ece_after": 0.1,
        "log_loss_before": 0.9,
        "log_loss_after": 0.8,
        "fitted": True,
    }


METRICS = {
    "baseline": _report(0.5),
    "final": _report(0.9),
    "calibration": {"department": _calibration(1.2), "outage": _calibration(1.5)},
    "data": {"train_rows": 80, "eval_rows": 20},
    "run": {
        "steps": 100,
        "best_step": 80,
        "stopped_early": False,
        "select_metric": "mean_macro_f1",
        "class_weighting": "balanced",
        "hours": 1.0,
        "estimated_hours": 0.8,
        "price_per_hour": 2.5,
        "cost": 2.5,
        "estimated_cost": 2.0,
    },
}


def test_card_reports_model_metrics_and_caveats(tmp_path):
    cfg = make_config(tmp_path, data={"train": "t.csv"}, train={"class_weighting": "balanced"})
    text = render_card(METRICS, cfg, "1.2.3")
    assert f"Base model: Cloudflare/clef @ {PINNED_REVISIONS['Cloudflare/clef']}" in text
    assert "clef-tune 1.2.3" in text
    assert "| department | accuracy | 0.500 | 0.900 |" in text
    assert "| all | exact match |" in text
    assert "slightly optimistic" in text
    calibration_header = (
        "| question | temperature | ECE before | ECE after | log-loss before | log-loss after |"
    )
    assert calibration_header in text
    assert "| department | 1.200 | 0.200 | 0.100 | 0.900 | 0.800 |" in text
    assert "Balanced class weighting shifts predicted probabilities" in text
    assert "Estimated time: 0.8 h" in text and "Actual time: 1.0 h" in text


def test_card_omits_the_weighting_note_without_balanced_weighting(tmp_path):
    cfg = make_config(tmp_path, data={"train": "t.csv"})
    assert "Balanced class weighting" not in render_card(METRICS, cfg, "1.2.3")


def test_confusion_markdown_keeps_the_top_classes_and_merges_the_rest():
    labels = [f"c{i:02d}" for i in range(12)]
    matrix = [[(12 - i) if i == j else 0 for j in range(12)] for i in range(12)]
    lines = confusion_markdown(labels, matrix).splitlines()
    assert lines[0] == "| true \\ predicted | " + " | ".join(labels[:10] + ["other"]) + " |"
    assert lines[-1].startswith("| other | ") and lines[-1].endswith(
        " | 3 |"
    )  # 2 + 1 on the diagonal


def test_the_card_starts_with_hub_metadata(tmp_path):
    config = make_config(tmp_path, data={"train": "t.csv"}, model={"id": "Cloudflare/clef-flash"})
    card = render_card(METRICS, config, "1.0")
    front, _ = card.removeprefix("---\n").split("---\n", 1)
    assert yaml.safe_load(front) == {
        "base_model": "Cloudflare/clef-flash",
        "base_model_relation": "adapter",
        "tags": ["clef", "clef-tune"],
    }
