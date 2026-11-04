"""Step 5: score every contender the same way, from its per-row answers.

  costs    gather measured cost, speed and training spend into results/costs.json
  score    results/metrics.json, results/results.csv and results/results.md

Every contender is scored from results/answers/<contender>.csv (one row per test complaint with
each answer, its confidence and the truth). A missing or failed answer counts as wrong.
95% ranges come from 2,000 bootstrap resamples of the test rows; the same resamples are used for
every contender, so differences between two contenders are paired.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.cfpb.common import ANSWERS, DATA, PREPARED, RESULTS, SEED
from clef_tune.calibration import ECE_BINS

QUESTIONS = ("issue", "product", "outcome")
BOOTSTRAP = 2000
CONTENDERS = {  # answers file stem -> how the write-up names it
    "clef-flash-untrained": "Clef-Flash, out of the box",
    "clef-flash-tuned": "Clef-Flash, tuned",
    "clef-27b-untrained": "Clef 27B, out of the box",
    "clef-27b-tuned": "Clef 27B, tuned",
    "claude-zero": "Claude Opus 5.5, instructions only",
    "claude-few": "Claude Opus 5.5, with examples",
}
PAIRS = [  # (a, b): the paired difference a - b in "all three right"
    ("clef-flash-tuned", "claude-few"),
    ("clef-27b-tuned", "claude-few"),
    ("clef-flash-tuned", "claude-zero"),
    ("clef-27b-tuned", "claude-zero"),
    ("clef-flash-tuned", "clef-flash-untrained"),
    ("clef-27b-tuned", "clef-27b-untrained"),
    ("claude-few", "claude-zero"),
]


@dataclass
class Answers:
    ids: list[str]
    options: dict[str, list[str]]  # question -> option ids, in a fixed order
    truth: dict[str, np.ndarray]  # question -> option index per row
    predicted: dict[str, np.ndarray]  # question -> option index, or -1 for no answer
    confidence: dict[str, np.ndarray]  # question -> 0..1, NaN for no answer


def question_options() -> dict[str, list[str]]:
    from clef_tune.config import load_config

    questions = load_config(PREPARED / "cfpb_flash.yaml").schema_def["questions"]
    return {q: list(questions[q]["criteria"]) for q in QUESTIONS}


def read_answers(path: Path, ids: list[str], options: dict[str, list[str]]) -> Answers:
    """Answers aligned to `ids` (the test rows, in order); a row missing from the file counts
    as unanswered."""
    with path.open(encoding="utf-8", newline="") as rows:
        by_id = {r["complaint_id"]: r for r in csv.DictReader(rows)}
    truth, predicted, confidence = {}, {}, {}
    for q in QUESTIONS:
        index = {option: k for k, option in enumerate(options[q])}
        t, p, c = [], [], []
        for complaint in ids:
            row = by_id.get(complaint)
            if row is None:
                raise ValueError(f"{path.name}: no row for complaint {complaint}")
            t.append(index[row[f"{q}_true"]])
            answer = row[f"{q}_answer"] if row.get("ok", "1") == "1" else ""
            p.append(index.get(answer, -1))
            value = row.get(f"{q}_confidence", "")
            c.append(float(value) if value not in ("", None) and answer else np.nan)
        truth[q], predicted[q], confidence[q] = np.array(t), np.array(p), np.array(c)
    return Answers(ids, options, truth, predicted, confidence)


def macro_f1(truth: np.ndarray, predicted: np.ndarray, classes: int) -> float:
    """clef-tune's macro-F1: the mean F1 over classes that occur in the truth or the answers.
    An unanswered row (-1) is wrong and adds no class."""
    hit = truth == predicted
    tp = np.bincount(truth[hit], minlength=classes)
    true_n = np.bincount(truth, minlength=classes)
    answered = predicted[predicted >= 0]
    pred_n = np.bincount(answered, minlength=classes)
    present = (true_n + pred_n) > 0
    f1 = np.where(present, 2 * tp / np.maximum(true_n + pred_n, 1), 0.0)
    return float(f1[present].mean()) if present.any() else 0.0


def ece(confidence: np.ndarray, correct: np.ndarray, bins: int = ECE_BINS) -> float | None:
    """Expected calibration error over answered rows, binned as clef-tune bins it."""
    answered = ~np.isnan(confidence)
    if not answered.any():
        return None
    conf, ok = confidence[answered], correct[answered].astype(float)
    which = np.minimum((conf * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        inside = which == b
        if inside.any():
            total += abs(conf[inside].sum() - ok[inside].sum())
    return total / len(confidence)


def resamples(n: int, count: int = BOOTSTRAP, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(count, n))


def interval(values: np.ndarray) -> list[float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def metrics(answers: Answers, draws: np.ndarray) -> dict:
    correct = {q: answers.truth[q] == answers.predicted[q] for q in QUESTIONS}
    all_right = np.all([correct[q] for q in QUESTIONS], axis=0)
    out = {
        "rows": len(answers.ids),
        "unanswered": int((answers.predicted[QUESTIONS[0]] < 0).sum()),
        "all_three_right": float(all_right.mean()),
        "all_three_right_95": interval(all_right[draws].mean(axis=1)),
        "questions": {},
    }
    for q in QUESTIONS:
        k = len(answers.options[q])
        t, p = answers.truth[q], answers.predicted[q]
        f1_draws = np.array([macro_f1(t[d], p[d], k) for d in draws])
        out["questions"][q] = {
            "accuracy": float(correct[q].mean()),
            "accuracy_95": interval(correct[q][draws].mean(axis=1)),
            "macro_f1": macro_f1(t, p, k),
            "macro_f1_95": interval(f1_draws),
            "ece": ece(answers.confidence[q], correct[q]),
        }
    return out


def paired_difference(a: Answers, b: Answers, draws: np.ndarray) -> dict:
    def right(x: Answers) -> np.ndarray:
        return np.all([x.truth[q] == x.predicted[q] for q in QUESTIONS], axis=0).astype(float)

    diff = right(a) - right(b)
    spread = diff[draws].mean(axis=1)
    return {
        "difference": float(diff.mean()),
        "difference_95": interval(spread),
        "share_of_resamples_above_zero": float((spread > 0).mean()),
    }


def test_ids() -> list[str]:
    with (PREPARED / "test.csv").open(encoding="utf-8", newline="") as rows:
        return [r["complaint_id"] for r in csv.DictReader(rows)]


def cross_check(name: str, scored: dict) -> list[str]:
    """Clef contenders: our accuracy must equal `clef-tune eval`'s own report."""
    model, arm = name.split("-")[1], "tuned" if name.endswith("tuned") else "base"
    report = DATA / "clef" / model / f"{arm}-eval.json"
    if not report.exists():
        return [f"{name}: no clef-tune eval report to cross-check"]
    theirs = json.loads(report.read_text(encoding="utf-8"))
    problems = []
    for q in QUESTIONS:
        ours = scored["questions"][q]["accuracy"]
        if abs(ours - theirs["questions"][q]["accuracy"]) > 1e-6:
            problems.append(f"{name} {q}: accuracy {ours} vs clef-tune eval's {theirs}")
    if abs(scored["all_three_right"] - theirs["exact_match"]) > 1e-6:
        problems.append(f"{name}: all three right differs from clef-tune eval's exact match")
    return problems


# ---------------------------------------------------------------------------------------- costs
def costs_command() -> None:
    from benchmarks.cfpb.common import machine_env
    from clef_tune.launch.hf_jobs import job_status

    token = machine_env("HF_TOKEN")
    jobs = json.loads((DATA / "jobs.json").read_text(encoding="utf-8"))
    costs: dict = {}
    for variant in ("zero", "few"):
        path = DATA / "claude" / variant / "test-summary.json"
        if path.exists():
            s = json.loads(path.read_text(encoding="utf-8"))
            costs[f"claude-{variant}"] = {
                "usd_per_1000": s["usd_per_1000"],
                "batch_usd_per_1000": s["batch_usd_per_1000"],
                "complaints_per_minute": s["complaints_per_minute"],
                "median_seconds": s["median_seconds"],
                "concurrency": s["concurrency"],
                "spent_usd": s["usd"],
                "prices": s["prices"],
            }
    for model in ("flash", "27b"):
        train_job = jobs.get(f"train-{model}")
        training = job_status(train_job["id"], token=token) if train_job else None
        for arm, stem in (("base", "untrained"), ("tuned", "tuned")):
            path = DATA / "clef" / model / f"{arm}-summary.json"
            if not path.exists():
                continue
            s = json.loads(path.read_text(encoding="utf-8"))
            entry = {
                "usd_per_1000": s["usd_per_1000"],
                "complaints_per_minute": s["complaints_per_minute"],
                "flavor": s["flavor"],
                "gpu": s["gpu"],
                "price_per_hour": s["price_per_hour"],
            }
            if arm == "tuned" and training is not None:
                entry["training_hours"] = training.hours
                entry["training_usd"] = training.cost
                entry["training_flavor"] = training.flavor
            costs[f"clef-{model}-{stem}"] = entry
    (RESULTS / "costs.json").write_text(json.dumps(costs, indent=2), encoding="utf-8")
    print(json.dumps(costs, indent=2))


# ---------------------------------------------------------------------------------------- score
def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def span(r: list[float]) -> str:
    return f"{100 * r[0]:.1f}–{100 * r[1]:.1f}"


def money(x: float | None) -> str:
    if x is None:
        return "–"
    return f"${x:,.2f}" if x >= 0.1 else f"${x:.3f}"


def score_command() -> None:
    ids = test_ids()
    options = question_options()
    draws = resamples(len(ids))
    present = {n: ANSWERS / f"{n}.csv" for n in CONTENDERS if (ANSWERS / f"{n}.csv").exists()}
    answers = {n: read_answers(p, ids, options) for n, p in present.items()}
    scored = {n: metrics(a, draws) for n, a in answers.items()}
    problems = [p for n in scored if n.startswith("clef") for p in cross_check(n, scored[n])]
    pairs = {
        f"{a} minus {b}": paired_difference(answers[a], answers[b], draws)
        for a, b in PAIRS
        if a in answers and b in answers
    }
    costs_path = RESULTS / "costs.json"
    costs = json.loads(costs_path.read_text(encoding="utf-8")) if costs_path.exists() else {}
    split = json.loads((PREPARED / "split.json").read_text(encoding="utf-8"))
    report = {
        "test_rows": len(ids),
        "test_dates": split["dates"]["test"],
        "bootstrap_resamples": BOOTSTRAP,
        "contenders": {
            n: {"name": CONTENDERS[n], **scored[n], "cost": costs.get(n)} for n in scored
        },
        "paired_differences": pairs,
        "cross_check_problems": problems,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(report)
    write_markdown(report)
    for problem in problems:
        print("CROSS-CHECK:", problem)
    print((RESULTS / "results.md").read_text(encoding="utf-8"))


def write_csv(report: dict) -> None:
    header = ["contender", "all_three_right", "all_three_right_low", "all_three_right_high"]
    for q in QUESTIONS:
        header += [f"{q}_accuracy", f"{q}_macro_f1", f"{q}_ece"]
    header += ["usd_per_1000", "batch_usd_per_1000", "complaints_per_minute", "training_usd"]
    with (RESULTS / "results.csv").open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(header)
        for c in report["contenders"].values():
            cost = c.get("cost") or {}
            line = [c["name"], c["all_three_right"], *c["all_three_right_95"]]
            for q in QUESTIONS:
                m = c["questions"][q]
                line += [m["accuracy"], m["macro_f1"], m["ece"]]
            line += [
                cost.get("usd_per_1000"),
                cost.get("batch_usd_per_1000"),
                cost.get("complaints_per_minute"),
                cost.get("training_usd"),
            ]
            writer.writerow(["" if v is None else v for v in line])


def write_markdown(report: dict) -> None:
    rows = report["contenders"]
    lines = [
        f"Test set: {report['test_rows']} complaints received "
        f"{report['test_dates'][0]} to {report['test_dates'][1]}. "
        f"Ranges are 95% bootstrap ranges ({report['bootstrap_resamples']} resamples).",
        "",
        "| Contender | All three right | Issue | Product | Outcome | Cost per 1,000 | "
        "Complaints per minute |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in rows.values():
        cost = c.get("cost") or {}
        q = c["questions"]
        per_1000 = money(cost.get("usd_per_1000"))
        if cost.get("batch_usd_per_1000") is not None:
            per_1000 += f" ({money(cost['batch_usd_per_1000'])} batch)"
        speed = cost.get("complaints_per_minute")
        lines.append(
            f"| {c['name']} | **{pct(c['all_three_right'])}** ({span(c['all_three_right_95'])}) "
            f"| {pct(q['issue']['accuracy'])} | {pct(q['product']['accuracy'])} "
            f"| {pct(q['outcome']['accuracy'])} | {per_1000} "
            f"| {'–' if speed is None else f'{speed:,.0f}'} |"
        )
    lines += [
        "",
        "Macro-F1 (the average over each question's answer choices, so rare choices count as "
        "much as common ones) and calibration error:",
        "",
        "| Contender | Issue F1 | Product F1 | Outcome F1 | Calibration error (issue / product /"
        " outcome) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c in rows.values():
        q = c["questions"]
        errors = " / ".join("–" if q[k]["ece"] is None else f"{q[k]['ece']:.3f}" for k in QUESTIONS)
        lines.append(
            f"| {c['name']} | {q['issue']['macro_f1']:.3f} | {q['product']['macro_f1']:.3f} "
            f"| {q['outcome']['macro_f1']:.3f} | {errors} |"
        )
    if report["paired_differences"]:
        lines += [
            "",
            'Paired differences in "all three right" (same resamples for both):',
            "",
            "| Comparison | Difference | 95% range |",
            "| --- | --- | --- |",
        ]
        for label, d in report["paired_differences"].items():
            a, b = label.split(" minus ")
            lines.append(
                f"| {CONTENDERS[a]} vs {CONTENDERS[b]} | {100 * d['difference']:+.1f} points "
                f"| {100 * d['difference_95'][0]:+.1f} to {100 * d['difference_95'][1]:+.1f} |"
            )
    (RESULTS / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("costs")
    sub.add_parser("score")
    args = parser.parse_args()
    if args.command == "costs":
        costs_command()
    else:
        score_command()


if __name__ == "__main__":
    main()
