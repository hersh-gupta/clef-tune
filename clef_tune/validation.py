"""Validate a config and its data. Errors block training; warnings inform; the stats feed
the UI and the cost estimates."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from clef_tune.config import Config
from clef_tune.data import URL_PREFIXES, DataError, Issue, Record, load_file, split_records
from clef_tune.prompt import (
    DEFAULT_URL_IMAGE_SIZE,
    RecordTokenCounter,
    TokenCounter,
    media_token_count,
)
from clef_tune.schema import Question, SchemaError

IMBALANCE_SHARE = 0.05
MIN_CALIBRATION_ROWS = 20


@dataclass
class QuestionStats:
    question_id: str
    type: str
    option_ids: list[str]
    train_counts: dict[str, int]
    eval_counts: dict[str, int]
    train_missing: int
    eval_labeled: int


def _percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


@dataclass
class TokenStats:
    train_lengths: list[int]
    eval_lengths: list[int]
    truncated_rows: int

    @property
    def p50(self) -> int:
        return _percentile(self.train_lengths + self.eval_lengths, 0.50)

    @property
    def p95(self) -> int:
        return _percentile(self.train_lengths + self.eval_lengths, 0.95)

    @property
    def max(self) -> int:
        return max(self.train_lengths + self.eval_lengths, default=0)

    @property
    def train_tokens(self) -> int:
        return sum(self.train_lengths)

    @property
    def eval_tokens(self) -> int:
        return sum(self.eval_lengths)


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)
    train: list[Record] = field(default_factory=list)
    eval: list[Record] = field(default_factory=list)
    questions: dict[str, QuestionStats] = field(default_factory=dict)
    tokens: TokenStats | None = None
    columns: list[str] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors


def validate(config: Config, counter: TokenCounter | None) -> ValidationReport:
    report = ValidationReport()
    try:
        config.get_schema()
    except (SchemaError, OSError) as exc:
        report.issues.append(Issue("error", f"schema: {exc}"))
        return report
    try:
        train_result = load_file(config, config.data.train)
        eval_result = load_file(config, config.data.eval) if config.data.eval else None
    except DataError as exc:
        report.issues.append(Issue("error", str(exc)))
        return report
    report.columns = train_result.columns
    report.issues.extend(train_result.issues)
    if eval_result is not None:
        report.issues.extend(
            Issue(i.level, f"eval file: {i.message}", i.rows, i.question)
            for i in eval_result.issues
        )
        report.train, report.eval = train_result.records, eval_result.records
    else:
        report.train, report.eval = split_records(
            train_result.records, config.data.eval_fraction, config.train.seed
        )
    if not report.train:
        report.issues.append(Issue("error", "no labeled training rows"))
    if not report.eval:
        message = "eval set is empty (need at least 2 labeled rows, or set data.eval)"
        report.issues.append(Issue("error", message))
    _check_ids(report)
    _question_stats(report)
    _check_images(report)
    if counter is None:
        message = "tokenizer unavailable; token statistics and truncation checks skipped"
        report.issues.append(Issue("warning", message))
    else:
        _token_stats(report, config, counter)
    return report


def _check_ids(report: ValidationReport) -> None:
    records = report.train + report.eval
    counts = Counter(record.id for record in records)
    duplicates = [record_id for record_id, n in counts.items() if n > 1]
    if duplicates:
        rows = sorted(r.source_row for r in records if counts[r.id] > 1)
        examples = ", ".join(duplicates[:5])
        message = f"{len(duplicates)} duplicate ID(s) (e.g. {examples})"
        report.issues.append(Issue("error", message, rows))


def _question_stats(report: ValidationReport) -> None:
    records = report.train + report.eval
    questions: dict[str, Question] = {}
    option_ids: dict[str, dict[str, None]] = {}
    for record in records:
        for qid, question in record.questions.items():
            questions.setdefault(qid, question)
            option_ids.setdefault(qid, {}).update(dict.fromkeys(question.option_ids()))
    for qid, question in questions.items():
        options = list(option_ids[qid])
        train = Counter(r.labels[qid].top() for r in report.train if qid in r.labels)
        evals = Counter(r.labels[qid].top() for r in report.eval if qid in r.labels)
        stats = QuestionStats(
            question_id=qid,
            type=question.type,
            option_ids=options,
            train_counts={option: train.get(option, 0) for option in options},
            eval_counts={option: evals.get(option, 0) for option in options},
            train_missing=sum(
                1 for r in report.train if qid in r.questions and qid not in r.labels
            ),
            eval_labeled=sum(evals.values()),
        )
        report.questions[qid] = stats
        observed = [option for option in options if train.get(option)]
        if len(observed) < 2:
            message = f"needs at least 2 classes in training data (found {len(observed)})"
            report.issues.append(Issue("error", message, question=qid))
            continue
        total = sum(train.values())
        rare = [option for option in observed if train[option] / total < IMBALANCE_SHARE]
        if rare:
            message = (
                f"rare classes (<5% of labels): {', '.join(rare)}; "
                "consider class_weighting: balanced"
            )
            report.issues.append(Issue("warning", message, question=qid))
        missing_eval = [option for option in observed if not evals.get(option)]
        if missing_eval and report.eval:
            message = f"eval set has no examples of: {', '.join(missing_eval)}"
            report.issues.append(Issue("warning", message, question=qid))
        if stats.eval_labeled < MIN_CALIBRATION_ROWS:
            message = (
                f"only {stats.eval_labeled} labeled eval rows; "
                "calibration will be skipped for this question"
            )
            report.issues.append(Issue("warning", message, question=qid))


def _check_images(report: ValidationReport) -> None:
    missing: list[int] = []
    remote: list[int] = []
    for record in report.train + report.eval:
        for image in record.images:
            if image.startswith(URL_PREFIXES):
                remote.append(record.source_row)
            elif not Path(image).exists():
                missing.append(record.source_row)
    if missing:
        report.issues.append(Issue("error", "image file not found", missing))
    if remote:
        message = "image URLs are not checked; assumed 1024x1024 for token estimates"
        report.issues.append(Issue("warning", message, remote))


def _media_tokens(record: Record) -> int:
    from PIL import Image

    sizes: list[tuple[int, int]] = []
    for image in record.images:
        size = DEFAULT_URL_IMAGE_SIZE
        if not image.startswith(URL_PREFIXES) and Path(image).exists():
            try:
                with Image.open(image) as opened:
                    size = opened.size
            except OSError:
                pass
        sizes.append(size)
    return media_token_count(sizes)


def _token_stats(report: ValidationReport, config: Config, counter: TokenCounter) -> None:
    max_length = config.train.max_length
    counting = RecordTokenCounter(counter, max_length)
    records = report.train + report.eval
    counts = counting.count_records([(r.state, r.questions, _media_tokens(r)) for r in records])
    too_long = [r.source_row for r, c in zip(records, counts, strict=True) if c.schema_too_long]
    if too_long:
        worst = max(c.fixed for c in counts)
        message = f"schema needs up to {worst} tokens before the state; max_length is {max_length}"
        report.issues.append(Issue("error", message, too_long))
    truncated = [r.source_row for r, c in zip(records, counts, strict=True) if c.truncated]
    if truncated:
        message = (
            f"{len(truncated)} rows exceed max_length ({max_length}); their state will be truncated"
        )
        report.issues.append(Issue("warning", message, truncated))
    split = len(report.train)
    report.tokens = TokenStats(
        train_lengths=[c.total for c in counts[:split]],
        eval_lengths=[c.total for c in counts[split:]],
        truncated_rows=len(truncated),
    )
