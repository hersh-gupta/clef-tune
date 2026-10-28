"""Small HTML pieces the UI is built from: stat tiles, issue lists, the prompt preview, tables,
badges and callouts. Everything that came from data, the Hub or a log is escaped here."""

from __future__ import annotations

import html
import re
from typing import Any, Literal

from clef_tune.card import METRIC_ROWS
from clef_tune.data import MAX_LISTED_ROWS, Issue
from clef_tune.estimate import EstimateRow
from clef_tune.prompt import Piece

LOWER_IS_BETTER = {"log_loss", "mae"}
_BADGES = {
    "COMPLETED": "ct-badge-success",
    "RUNNING": "ct-badge-active",
    "SCHEDULING": "ct-badge-active",
    "ERROR": "ct-badge-error",
    "CANCELED": "ct-badge-muted",
    "DELETED": "ct-badge-muted",
}
_CALLOUT_ICONS = {"info": "i", "warning": "!", "error": "×", "success": "✓"}


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _number(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _money(value: float | None) -> str:
    return "—" if value is None else f"${value:,.2f}"


def stat_tiles(tiles: list[tuple[str, str]]) -> str:
    items = "".join(
        f'<div class="ct-tile"><div class="ct-tile-value">{esc(value)}</div>'
        f'<div class="ct-tile-label">{esc(label)}</div></div>'
        for label, value in tiles
    )
    return f'<div class="ct-tiles">{items}</div>'


def empty_state(title: str, body: str) -> str:
    return (
        f'<div class="ct-empty-state"><div class="ct-empty-title">{esc(title)}</div>'
        f"<p>{esc(body)}</p></div>"
    )


def callout(
    kind: Literal["info", "warning", "error", "success"],
    text: str,
    *,
    link: tuple[str, str] | None = None,
) -> str:
    role = "alert" if kind == "error" else "status"
    anchor = ""
    if link is not None:
        label, url = link
        anchor = f' <a href="{esc(url)}" target="_blank" rel="noopener">{esc(label)}</a>'
    return (
        f'<div class="ct-callout is-{kind}" role="{role}">'
        f'<span class="ct-callout-icon" aria-hidden="true">{_CALLOUT_ICONS[kind]}</span>'
        f"<div>{esc(text)}{anchor}</div></div>"
    )


def _rows_text(rows: list[int]) -> str:
    shown = ", ".join(str(row) for row in rows[:MAX_LISTED_ROWS])
    extra = len(rows) - MAX_LISTED_ROWS
    word = "Row" if len(rows) == 1 else "Rows"
    return f"{word} {shown}" + (f" +{extra} more" if extra > 0 else "")


def issue_list(issues: list[Issue]) -> str:
    if not issues:
        return empty_state("No problems found", "The data and schema passed every check.")
    ordered = sorted(issues, key=lambda issue: issue.level != "error")
    items = []
    for issue in ordered:
        icon = "×" if issue.level == "error" else "!"
        question = (
            f'<span class="ct-issue-question">{esc(issue.question)}</span> '
            if issue.question
            else ""
        )
        rows = (
            f'<div class="ct-issue-rows">{esc(_rows_text(issue.rows))}</div>' if issue.rows else ""
        )
        items.append(
            f'<li class="ct-issue is-{issue.level}">'
            f'<span class="ct-issue-icon" aria-hidden="true">{icon}</span>'
            f'<div><div class="ct-issue-text">{question}{esc(issue.message)}</div>{rows}</div></li>'
        )
    return f'<ul class="ct-issues">{"".join(items)}</ul>'


def prompt_preview(pieces: list[Piece]) -> str:
    spans = "".join(f'<span class="pp-{piece.kind}">{esc(piece.text)}</span>' for piece in pieces)
    return f'<pre class="ct-prompt">{spans}</pre>'


def stage_badge(stage: str) -> str:
    css = _BADGES.get(stage, "ct-badge-muted")
    return f'<span class="ct-badge {css}">{esc(stage.capitalize())}</span>'


def progress(fraction: float | None, label: str) -> str:
    if fraction is None:
        bar = (
            f'<div class="ct-progress ct-progress-unknown" role="progressbar" '
            f'aria-label="{esc(label)}"><div class="ct-progress-bar"></div></div>'
        )
    else:
        percent = round(min(max(fraction, 0.0), 1.0) * 100)
        bar = (
            f'<div class="ct-progress" role="progressbar" aria-label="{esc(label)}" '
            f'aria-valuemin="0" aria-valuemax="100" aria-valuenow="{percent}">'
            f'<div class="ct-progress-bar" style="width:{percent}%"></div></div>'
        )
    return f'{bar}<div class="ct-progress-label">{esc(label)}</div>'


def _change(key: str, before: Any, after: Any) -> str:
    if before is None or after is None:
        return '<td class="ct-num">—</td>'
    delta = after - before
    better = delta < 0 if key in LOWER_IS_BETTER else delta > 0
    css = "is-better" if better else ("is-worse" if delta else "")
    return f'<td class="ct-num {css}">{delta:+.3f}</td>'


def metrics_table(baseline: dict, final: dict) -> str:
    head = (
        "<thead><tr><th>Question</th><th>Metric</th><th>Baseline</th><th>Fine-tuned</th>"
        "<th>Change</th></tr></thead>"
    )
    rows = []
    before_questions = baseline.get("questions", {})
    for qid, after in final.get("questions", {}).items():
        before = before_questions.get(qid, {})
        metrics = METRIC_ROWS.get(after.get("question_type"), [("accuracy", "accuracy")])
        for index, (label, key) in enumerate(metrics):
            name = f'<th scope="row" rowspan="{len(metrics)}">{esc(qid)}</th>' if not index else ""
            rows.append(
                f"<tr>{name}<td>{esc(label)}</td>"
                f'<td class="ct-num">{_number(before.get(key))}</td>'
                f'<td class="ct-num">{_number(after.get(key))}</td>'
                f"{_change(key, before.get(key), after.get(key))}</tr>"
            )
    before, after = baseline.get("exact_match"), final.get("exact_match")
    rows.append(
        '<tr class="ct-total"><th scope="row">All questions</th><td>exact match</td>'
        f'<td class="ct-num">{_number(before)}</td><td class="ct-num">{_number(after)}</td>'
        f"{_change('exact_match', before, after)}</tr>"
    )
    return f'<table class="ct-table">{head}<tbody>{"".join(rows)}</tbody></table>'


def calibration_table(calibration: dict) -> str:
    if not calibration:
        return empty_state("No calibration", "Calibration was turned off for this run.")
    head = (
        "<thead><tr><th>Question</th><th>Temperature</th><th>ECE before</th><th>ECE after</th>"
        "<th>Log-loss before</th><th>Log-loss after</th></tr></thead>"
    )
    rows = []
    for qid, item in calibration.items():
        keys = ("temperature", "ece_before", "ece_after", "log_loss_before", "log_loss_after")
        cells = "".join(f'<td class="ct-num">{_number(item.get(key))}</td>' for key in keys)
        note = "" if item.get("fitted", True) else ' <span class="ct-muted">(too few rows)</span>'
        rows.append(f'<tr><th scope="row">{esc(qid)}{note}</th>{cells}</tr>')
    return f'<table class="ct-table">{head}<tbody>{"".join(rows)}</tbody></table>'


def _estimate_row(row: EstimateRow, recommended: str | None) -> str:
    tag = ' <span class="ct-tag">Recommended</span>' if row.flavor == recommended else ""
    css = "ct-recommended" if row.flavor == recommended else ("" if row.fits else "is-unfit")
    gpus = f"{row.count} × {row.gpu}"
    memory = f"{row.memory_needed_gb:.0f} / {row.memory_available_gb:.0f} GB"
    hours = f"{row.hours_low:.1f}–{row.hours_high:.1f} h"
    cost = "—" if row.cost_low is None else f"{_money(row.cost_low)}–{_money(row.cost_high)}"
    note = f'<div class="ct-muted">{esc(row.note)}</div>' if row.note else ""
    return (
        f'<tr class="{css}"><th scope="row">{esc(row.flavor)}{tag}{note}</th>'
        f'<td>{esc(gpus)}</td><td class="ct-num">{memory}</td>'
        f'<td class="ct-num">{hours}</td><td class="ct-num">{cost}</td></tr>'
    )


def estimate_table(rows: list[EstimateRow], recommended: str | None) -> str:
    if not rows:
        return empty_state("No estimate yet", "Fix the validation errors to see run times.")
    head = (
        "<thead><tr><th>Hardware</th><th>GPUs</th><th>Memory needed</th><th>Time</th>"
        "<th>Cost</th></tr></thead>"
    )
    fitting = "".join(_estimate_row(row, recommended) for row in rows if row.fits)
    unfit = [row for row in rows if not row.fits]
    table = f'<table class="ct-table ct-estimates">{head}<tbody>{fitting}</tbody></table>'
    if unfit:
        body = "".join(_estimate_row(row, recommended) for row in unfit)
        table += (
            f'<details class="ct-more"><summary>{len(unfit)} more that cannot run this config'
            f'</summary><table class="ct-table ct-estimates">{head}<tbody>{body}</tbody>'
            "</table></details>"
        )
    return table


def column_chips(columns: list[str], roles: dict[str, str]) -> str:
    chips = []
    for column in columns:
        role = roles.get(column, "state")
        css = role if re.fullmatch(r"[a-z]+", role) else "other"
        chips.append(
            f'<span class="ct-chip is-{css}"><span class="ct-chip-name">{esc(column)}</span>'
            f'<span class="ct-chip-role">{esc(role)}</span></span>'
        )
    return f'<div class="ct-chips">{"".join(chips)}</div>'
