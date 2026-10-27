"""The adapter's README card: metrics tables, confusion matrices and calibration caveats.

Torch-free: it only formats the metrics dict that `train` also writes as metrics.json."""

from __future__ import annotations

from typing import Any

from clef_tune.config import Config

TOP_CLASSES = 10

# question type -> (label, key in the per-question metrics) rows shown in the table
_ROWS: dict[str, list[tuple[str, str]]] = {
    "choice": [("accuracy", "accuracy"), ("macro-F1", "f1"), ("log-loss", "log_loss")],
    "score": [("accuracy", "accuracy"), ("MAE", "mae")],
    "noul": [("accuracy", "accuracy"), ("F1", "f1"), ("log-loss", "log_loss")],
}
_CALIBRATION_CAVEAT = (
    "Temperatures were fit on the same eval set used to choose the checkpoint, "
    "so post-calibration ECE and log-loss are slightly optimistic."
)
_WEIGHTING_NOTE = (
    "Balanced class weighting shifts predicted probabilities toward rare classes; "
    "temperature calibration rescales confidence but does not undo that shift."
)


def _num(value: Any) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def confusion_markdown(labels: list[str], matrix: list[list[int]], top: int = TOP_CLASSES) -> str:
    """A Markdown confusion matrix; classes beyond the `top` by true rows merge into `other`."""
    order = sorted(range(len(labels)), key=lambda i: (-sum(matrix[i]), i))
    keep = sorted(order[:top])
    rest = [i for i in range(len(labels)) if i not in keep]
    names = [labels[i] for i in keep] + (["other"] if rest else [])
    rows = [[sum(matrix[i][j] for j in group) for group in _groups(keep, rest)] for i in keep]
    if rest:
        rows.append(
            [sum(matrix[i][j] for i in rest for j in group) for group in _groups(keep, rest)]
        )
    lines = ["| true \\ predicted | " + " | ".join(names) + " |"]
    lines.append("|" + " --- |" * (len(names) + 1))
    for name, row in zip(names, rows, strict=True):
        lines.append(f"| {name} | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def _groups(keep: list[int], rest: list[int]) -> list[list[int]]:
    return [[j] for j in keep] + ([rest] if rest else [])


def _metric_table(baseline: dict, final: dict) -> list[str]:
    lines = ["| question | metric | baseline | fine-tuned |", "| --- | --- | --- | --- |"]
    for qid, after in final["questions"].items():
        before = baseline["questions"].get(qid, {})
        for label, key in _ROWS.get(after["question_type"], [("accuracy", "accuracy")]):
            lines.append(f"| {qid} | {label} | {_num(before.get(key))} | {_num(after.get(key))} |")
    before, after = _num(baseline.get("exact_match")), _num(final.get("exact_match"))
    lines.append(f"| all | exact match | {before} | {after} |")
    return lines


def _time_line(name: str, hours: float, cost: float | None) -> str:
    return f"{name}: {hours:.1f} h" + (f" (${cost:.2f})" if cost is not None else "")


def render_card(metrics: dict, config: Config, version: str) -> str:
    run = metrics.get("run", {})
    data = metrics.get("data", {})
    out = [
        f"# {config.run_name}",
        "",
        f"Base model: {config.model.id} @ {config.model.resolved_revision()}",
        "",
        f"Fine-tuned with clef-tune {version}.",
        "",
        f"Train rows: {data.get('train_rows', 'n/a')}; eval rows: {data.get('eval_rows', 'n/a')}.",
        "",
        "## Results (eval set)",
        "",
        *_metric_table(metrics["baseline"], metrics["final"]),
        "",
        "## Confusion matrices",
    ]
    for qid, question in metrics["final"]["questions"].items():
        confusion = question["confusion"]
        out += ["", f"### {qid}", "", confusion_markdown(confusion["labels"], confusion["matrix"])]
    calibration = metrics.get("calibration", {})
    if calibration:
        out += [
            "",
            "## Calibration",
            "",
            "| question | temperature | ECE before | ECE after "
            "| log-loss before | log-loss after |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for qid, cal in calibration.items():
            keys = ("temperature", "ece_before", "ece_after", "log_loss_before", "log_loss_after")
            cells = tuple(cal.get(key) for key in keys)
            out.append(f"| {qid} | " + " | ".join(_num(v) for v in cells) + " |")
        out += ["", _CALIBRATION_CAVEAT]
    if config.train.class_weighting == "balanced":
        out += ["", _WEIGHTING_NOTE]
    out += ["", "## Run", ""]
    if run.get("estimated_hours") is not None:
        out += [_time_line("Estimated time", run["estimated_hours"], run.get("estimated_cost")), ""]
    if run.get("hours") is not None:
        out += [_time_line("Actual time", run["hours"], run.get("cost")), ""]
    return "\n".join(out).rstrip() + "\n"
