"""Prediction spreadsheets, without torch: check a file before predicting, build the output
table (the person's columns, then each question's answer, its confidence and a review mark),
write it as CSV, Excel, TSV or JSON, and summarize it for the web UI."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clef_tune.config import Config
from clef_tune.data import JSONL_SUFFIXES, Issue, Row, normalize_rows, read_rows
from clef_tune.labels import LabelError, parse_label
from clef_tune.prompt import RecordTokenCounter, TokenCounter
from clef_tune.validation import media_tokens

REVIEW_COLUMN = "review"
DEFAULT_REVIEW_BELOW = 0.7
PREVIEW_ROWS = 5
NOUL_WORDS = {"true": "yes", "false": "no"}

Probabilities = dict[str, dict[str, float]]  # question id -> option id -> probability


def answer_column(qid: str, columns: list[str]) -> str:
    """A question's answer column: its id, unless the file already has a column of that name
    (usually its expected outputs)."""
    return f"{qid}_predicted" if qid in columns else qid


def _number(value: float) -> str:
    return format(round(value, 4), "g")


def prediction_table(
    columns: list[str],
    rows: list[Row],
    answers: dict[int, Probabilities],
    options: dict[str, list[str]],
    *,
    review_below: float = DEFAULT_REVIEW_BELOW,
    probabilities: bool = False,
) -> tuple[list[str], list[list[str]]]:
    """The output header and rows. `answers` is keyed by each row's number in the file; a row
    without answers keeps its cells and gets empty answer columns."""
    header = list(columns)
    for qid, option_ids in options.items():
        header += [answer_column(qid, columns), f"{qid}_confidence"]
        if probabilities:
            header += [f"{qid}.p.{option}" for option in option_ids]
    header.append(REVIEW_COLUMN)
    out = []
    for number, row in rows:
        line = [str(row.get(column, "") or "") for column in columns]
        per_question = answers.get(number)
        unsure = False
        for qid, option_ids in options.items():
            probs = per_question.get(qid) if per_question else None
            if probs is None:
                line += [""] * (2 + (len(option_ids) if probabilities else 0))
                continue
            top = max(option_ids, key=lambda option: probs.get(option, 0.0))
            confidence = probs.get(top, 0.0)
            unsure = unsure or confidence < review_below
            line += [
                NOUL_WORDS.get(top, top) if set(option_ids) == set(NOUL_WORDS) else top,
                _number(confidence),
            ]
            if probabilities:
                line += [_number(probs.get(option, 0.0)) for option in option_ids]
        line.append("yes" if unsure else "")
        out.append(line)
    return header, out


TABLE_SUFFIXES = (".csv", ".xlsx", ".tsv", ".jsonl", ".json")


def _numeric(header: list[str]) -> set[int]:
    """The columns holding numbers: confidences and per-answer probabilities."""
    return {k for k, name in enumerate(header) if name.endswith("_confidence") or ".p." in name}


def _records(header: list[str], rows: list[list[str]]) -> list[dict[str, Any]]:
    """Rows as JSON objects: numbers as numbers, and an empty cell as null."""
    numeric = _numeric(header)
    return [
        {
            name: None if value == "" else float(value) if k in numeric else value
            for k, (name, value) in enumerate(zip(header, row, strict=True))
        }
        for row in rows
    ]


def write_table(header: list[str], rows: list[list[str]], out: Path) -> Path:
    """Write a table as .csv (UTF-8 with a BOM, so Excel reads it right), .xlsx, .tsv, or as
    JSON records: .jsonl (one object per line) or .json (one list)."""
    out = Path(out)
    suffix = out.suffix.lower()
    out.parent.mkdir(parents=True, exist_ok=True)
    if suffix in (".csv", ".tsv"):
        encoding = "utf-8-sig" if suffix == ".csv" else "utf-8"
        with open(out, "w", encoding=encoding, newline="") as handle:
            writer = csv.writer(handle, delimiter="," if suffix == ".csv" else "\t")
            writer.writerow(header)
            writer.writerows(rows)
        return out
    if suffix == ".xlsx":
        from openpyxl import Workbook

        numeric = _numeric(header)
        book = Workbook()
        sheet = book.active
        sheet.title = "predictions"
        sheet.append(header)
        for row in rows:
            sheet.append([float(v) if k in numeric and v != "" else v for k, v in enumerate(row)])
        book.save(out)
        return out
    if suffix == ".jsonl":
        with open(out, "w", encoding="utf-8") as handle:
            for record in _records(header, rows):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return out
    if suffix == ".json":
        out.write_text(json.dumps(_records(header, rows), ensure_ascii=False), encoding="utf-8")
        return out
    raise ValueError(f"unsupported output type {suffix!r} (use {', '.join(TABLE_SUFFIXES)})")


def read_table(path: Path) -> tuple[list[str], list[list[str]]]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        records = list(csv.reader(handle))
    return (records[0], records[1:]) if records else ([], [])


@dataclass
class PredictionSummary:
    rows: int
    predicted: int
    review: int
    shares: dict[str, list[tuple[str, float]]]  # question -> (answer, share), largest first
    matches: dict[str, tuple[int, int]]  # question -> (matched, compared) when outputs exist
    preview: list[dict[str, str]]


def summarize(header: list[str], rows: list[list[str]], config: Config) -> PredictionSummary:
    schema = config.get_schema()
    questions = schema.questions if schema else {}
    index = {name: k for k, name in enumerate(header)}
    review = index.get(REVIEW_COLUMN)
    shares: dict[str, list[tuple[str, float]]] = {}
    matches: dict[str, tuple[int, int]] = {}
    predicted_rows: set[int] = set()
    original = header[: header.index(REVIEW_COLUMN)] if REVIEW_COLUMN in header else header
    for qid, question in questions.items():
        predicted_name = f"{qid}_predicted" if f"{qid}_predicted" in index else qid
        column = index.get(predicted_name)
        if column is None:
            continue
        answers = [(k, row[column]) for k, row in enumerate(rows) if row[column]]
        predicted_rows.update(k for k, _ in answers)
        counts = Counter(answer for _, answer in answers)
        total = sum(counts.values())
        shares[qid] = sorted(
            ((answer, count / total) for answer, count in counts.items()),
            key=lambda item: (-item[1], item[0]),
        )
        expected = index.get(config.data.label_columns.get(qid, qid))
        if expected is None or expected == column or header[expected] not in original:
            continue
        matched = compared = 0
        for k, answer in answers:
            try:
                label, _ = parse_label(question, rows[k][expected])
            except LabelError:
                continue
            if label is None:
                continue
            compared += 1
            matched += NOUL_WORDS.get(label.top(), label.top()) == answer
        if compared:
            matches[qid] = (matched, compared)
    return PredictionSummary(
        rows=len(rows),
        predicted=len(predicted_rows),
        review=sum(1 for row in rows if review is not None and row[review] == "yes"),
        shares=shares,
        matches=matches,
        preview=[dict(zip(header, row, strict=False)) for row in rows[:PREVIEW_ROWS]],
    )


@dataclass
class InputCheck:
    columns: list[str]  # the file's own columns
    missing: list[str]  # columns the model reads that the file lacks (after renaming)
    rows: int
    tokens: int  # prompt tokens over all rows; 0 when unknown
    truncated: int  # rows too long to read in full
    expected: list[str]  # questions whose expected outputs the file has
    issues: list[Issue] = field(default_factory=list)
    numbered_ids: bool = False  # the file had no ID column, so rows were numbered
    table: list[Row] = field(default_factory=list, repr=False)
    table_columns: list[str] = field(default_factory=list, repr=False)

    @property
    def ok(self) -> bool:
        return not self.missing and not any(issue.level == "error" for issue in self.issues)


def check_input(
    config: Config,
    path: Path,
    counter: TokenCounter | None,
    rename: dict[str, str] | None = None,
) -> InputCheck:
    """Check a file to predict on against the model's config. `rename` maps the file's column
    names to the ones the model read (when they differ). A missing ID column is fine: rows are
    numbered instead."""
    rows, columns, issues = read_rows(Path(path))
    rename = {old: new for old, new in (rename or {}).items() if old in columns and new}
    renamed = [rename.get(column, column) for column in columns]
    table = [(number, {rename.get(k, k): v for k, v in row.items()}) for number, row in rows]
    data = config.data
    numbered = bool(data.id_column) and data.id_column not in renamed
    if numbered:
        renamed = [data.id_column, *renamed]
        table = [
            (number, {data.id_column: str(k), **row})
            for k, (number, row) in enumerate(table, start=1)
        ]
    needed = list(data.state_columns or []) + list(data.image_columns)
    missing = [column for column in needed if column not in renamed]
    if missing:
        return InputCheck(columns, missing, len(table), 0, 0, [], issues, numbered, table, renamed)
    jsonl = Path(path).suffix.lower() in JSONL_SUFFIXES
    result = normalize_rows(
        table, renamed, data, config.get_schema(), Path(path).parent, True, jsonl
    )
    issues = issues + result.issues
    tokens = truncated = 0
    if counter is not None and result.records:
        counting = RecordTokenCounter(counter, config.train.max_length)
        counts = counting.count_records(
            [(r.state, r.questions, media_tokens(r)) for r in result.records]
        )
        tokens = sum(c.total for c in counts)
        truncated = sum(1 for c in counts if c.truncated)
    labeled = {qid for record in result.records for qid in record.labels}
    schema = config.get_schema()
    expected = [qid for qid in (schema.questions if schema else {}) if qid in labeled]
    return InputCheck(
        columns,
        [],
        len(result.records),
        tokens,
        truncated,
        expected,
        issues,
        numbered,
        table,
        renamed,
    )


def write_input_csv(check: InputCheck, out: Path) -> Path:
    """The checked (renamed, numbered) rows as a plain UTF-8 CSV for the prediction job."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=check.table_columns, extrasaction="ignore")
        writer.writeheader()
        for _, row in check.table:
            writer.writerow({k: "" if v is None else v for k, v in row.items()})
    return out


def records_to_answers(records: list[Any], probabilities: list[Probabilities]) -> dict[int, Any]:
    """Answers keyed by each record's row number, for prediction_table."""
    return {record.source_row: probs for record, probs in zip(records, probabilities, strict=True)}
