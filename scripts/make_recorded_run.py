"""Write tests/ui/fixtures/recorded_run/: a deterministic example run in the trainer's exact
output format (metrics.jsonl events and the adapter's metrics.json), for the UI's tests, its
browser test and its screenshots. The numbers are synthetic; the metrics and calibration come
from clef_tune.metrics and clef_tune.calibration, so their shapes match what training writes."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path

from clef_tune.calibration import calibrate, softmax
from clef_tune.metrics import MetricsReport, Prediction, summarize

OUT = Path(__file__).resolve().parents[1] / "tests" / "ui" / "fixtures" / "recorded_run"
QUESTIONS = {  # id -> (type, option ids, how often each option is the answer)
    "department": ("choice", ["account", "billing", "technical"], [0.2, 0.4, 0.4]),
    "urgency": ("score", ["0", "1", "2"], [0.4, 0.33, 0.27]),
    "outage": ("noul", ["true", "false"], [0.2, 0.8]),
}
TRAIN_ROWS, EVAL_ROWS = 170, 30
STEPS, EVAL_EVERY, LOG_EVERY, WARMUP_STEPS = 120, 40, 10, 4
SECONDS_PER_STEP = 7.5
PRICE_PER_HOUR = 0.8
LORA_LR = 1e-4
SKILL = {0: 0.4, 40: 1.3, 80: 1.7, 120: 2.0}  # how far the answer's logit stands out
SCALE = {0: 1.0, 40: 1.6, 80: 2.0, 120: 2.4}  # > 1: overconfident, as fine-tuned models are


def evaluate(
    rng: random.Random, answers: list[dict[str, int]], skill: float, scale: float
) -> tuple[MetricsReport, dict[str, list[list[float]]], dict[str, list[list[float]]]]:
    records, logits, targets = [], {q: [] for q in QUESTIONS}, {q: [] for q in QUESTIONS}
    for row in answers:
        predictions = []
        for qid, (qtype, options, _) in QUESTIONS.items():
            z = [rng.gauss(0.0, 1.0) for _ in options]
            z[row[qid]] += skill
            z = [scale * value for value in z]
            target = [1.0 if i == row[qid] else 0.0 for i in range(len(options))]
            predictions.append(Prediction(qid, qtype, options, softmax(z), target))
            logits[qid].append(z)
            targets[qid].append(target)
        records.append(predictions)
    return summarize(records), logits, targets


def headline(report: MetricsReport) -> dict[str, float]:
    return {
        "exact_match": report.exact_match,
        "mean_macro_f1": report.mean_macro_f1,
        "mean_log_loss": report.mean_log_loss,
    }


def lr(step: int) -> float:
    if step < WARMUP_STEPS:
        return LORA_LR * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (STEPS - WARMUP_STEPS)
    return LORA_LR * 0.5 * (1 + math.cos(math.pi * progress))


def main() -> None:
    rng = random.Random(7)
    answers = [
        {
            qid: rng.choices(range(len(options)), weights)[0]
            for qid, (_, options, weights) in QUESTIONS.items()
        }
        for _ in range(EVAL_ROWS)
    ]
    reports = {step: evaluate(rng, answers, SKILL[step], SCALE[step]) for step in SKILL}
    events: list[dict] = [{"event": "baseline", "step": 0, **headline(reports[0][0])}]
    for step in range(LOG_EVERY, STEPS + 1, LOG_EVERY):
        loss = (math.exp(-step / 45) + 0.22) * (1 + rng.uniform(-0.03, 0.03))
        eta = None if step < 20 else (STEPS - step) * SECONDS_PER_STEP
        cost = None if eta is None else STEPS * SECONDS_PER_STEP / 3600 * PRICE_PER_HOUR
        events.append(
            {
                "event": "train",
                "step": step,
                "loss": round(loss, 4),
                "lr": lr(step),
                "tokens_per_sec": round(380 + rng.gauss(0, 8), 1),
                "eta_s": eta,
                "projected_cost": cost,
                "total_steps": STEPS,
            }
        )
        if step % EVAL_EVERY == 0:
            events.append({"event": "eval", "step": step, **headline(reports[step][0])})
    events.append({"event": "done", "adapter": "/work/out/final"})

    final, logits, targets = reports[STEPS]
    hours = 0.42
    metrics = {
        "baseline": reports[0][0].to_dict(),
        "final": final.to_dict(),
        "calibration": {qid: asdict(item) for qid, item in calibrate(logits, targets).items()},
        "data": {"train_rows": TRAIN_ROWS, "eval_rows": EVAL_ROWS},
        "run": {
            "steps": STEPS,
            "best_step": STEPS,
            "stopped_early": False,
            "select_metric": "mean_macro_f1",
            "class_weighting": "none",
            "hours": hours,
            "estimated_hours": 0.5,
            "price_per_hour": PRICE_PER_HOUR,
            "cost": hours * PRICE_PER_HOUR,
            "estimated_cost": 0.5 * PRICE_PER_HOUR,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps(event) + "\n" for event in events)
    (OUT / "metrics.jsonl").write_text(lines, encoding="utf-8", newline="\n")
    (OUT / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {len(events)} events and metrics.json to {OUT}")


if __name__ == "__main__":
    main()
