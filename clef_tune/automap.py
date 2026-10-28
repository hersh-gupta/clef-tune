"""Propose a schema from an uploaded file (spec §4.7): which columns are questions, which are
the state, and which holds the row ids. The UI shows the proposal for the person to edit."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

from clef_tune.labels import NOUL_FALSE, NOUL_TRUE, is_missing
from clef_tune.schema import option_key

MAX_CHOICE_OPTIONS = 50
MAX_UNIQUE_SHARE = 0.5
NOUL_OPTIONS = ["true", "false"]
_ID_NAME = re.compile(r"^(id|.*[_ -]id)$", re.IGNORECASE)


@dataclass(frozen=True)
class QuestionProposal:
    id: str
    column: str
    type: Literal["noul", "choice"]
    options: list[str]


@dataclass(frozen=True)
class Proposal:
    id_column: str | None
    state_columns: list[str]
    questions: list[QuestionProposal]
    ignored: list[str]  # columns without any values
    native: bool  # rows are native JSONL ({"state", "labels"})


def _value(raw: Any) -> str | None:
    if isinstance(raw, bool):
        return option_key(raw)
    if is_missing(raw):
        return None
    text = str(raw).strip()
    return text or None


def _by_count(counts: Counter[str]) -> list[str]:
    return [value for value, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def _is_noul(counts: Counter[str]) -> bool:
    return all(value.casefold() in NOUL_TRUE | NOUL_FALSE for value in counts)


def propose_column(rows: list[dict[str, Any]], column: str) -> QuestionProposal | None:
    """A question for the column, or None when it has no values or looks like free text."""
    values = [v for v in (_value(row.get(column)) for row in rows) if v is not None]
    if not values:
        return None
    counts = Counter(values)
    if _is_noul(counts):
        return QuestionProposal(column, column, "noul", list(NOUL_OPTIONS))
    if len(counts) <= MAX_CHOICE_OPTIONS and len(counts) / len(values) < MAX_UNIQUE_SHARE:
        return QuestionProposal(column, column, "choice", _by_count(counts))
    return None


def _is_id_column(rows: list[dict[str, Any]], column: str) -> bool:
    if not _ID_NAME.match(column):
        return False
    values = [_value(row.get(column)) for row in rows]
    return bool(values) and None not in values and len(set(values)) == len(values)


def _propose_native(rows: list[dict[str, Any]]) -> Proposal:
    counts: dict[str, Counter[str]] = {}
    for row in rows:
        labels = row.get("labels")
        if not isinstance(labels, dict):
            continue
        for qid, raw in labels.items():
            seen = counts.setdefault(str(qid), Counter())
            if isinstance(raw, dict):  # a soft label names each option once
                seen.update(option_key(key) for key in raw)
            elif (value := _value(raw)) is not None:
                seen[value] += 1
    questions = [
        QuestionProposal(qid, qid, "noul", list(NOUL_OPTIONS))
        if seen and _is_noul(seen)
        else QuestionProposal(qid, qid, "choice", _by_count(seen))
        for qid, seen in counts.items()
    ]
    return Proposal(None, [], questions, [], native=True)


def propose(rows: list[dict[str, Any]], columns: list[str]) -> Proposal:
    if any(isinstance(row.get("labels"), dict) for row in rows):
        return _propose_native(rows)
    id_column = next((c for c in columns if _is_id_column(rows, c)), None)
    state: list[str] = []
    questions: list[QuestionProposal] = []
    ignored: list[str] = []
    for column in columns:
        if column == id_column:
            continue
        if all(_value(row.get(column)) is None for row in rows):
            ignored.append(column)
        elif (question := propose_column(rows, column)) is not None:
            questions.append(question)
        else:
            state.append(column)
    return Proposal(id_column, state, questions, ignored, native=False)
