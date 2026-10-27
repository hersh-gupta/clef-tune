"""Evaluation metrics: per-question accuracy / F1 / log-loss / MAE, confusion matrices, and
per-record exact match."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Any, Literal

EPS = 1e-12
SelectMetric = Literal["mean_macro_f1", "exact_match", "mean_log_loss"]
HIGHER_IS_BETTER: dict[str, bool] = {
    "mean_macro_f1": True,
    "exact_match": True,
    "mean_log_loss": False,
}


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


@dataclass(frozen=True)
class Prediction:
    question_id: str
    question_type: str
    option_ids: list[str]
    probs: list[float]
    target: list[float]  # one-hot for hard labels, a distribution for soft labels

    @property
    def predicted(self) -> str:
        return self.option_ids[_argmax(self.probs)]

    @property
    def actual(self) -> str:
        return self.option_ids[_argmax(self.target)]

    @property
    def correct(self) -> bool:
        return self.predicted == self.actual

    def log_loss(self) -> float:
        return -sum(
            t * math.log(max(p, EPS)) for p, t in zip(self.probs, self.target, strict=True) if t > 0
        )


@dataclass
class ConfusionMatrix:
    labels: list[str]
    matrix: list[list[int]]  # matrix[true][predicted]


@dataclass
class QuestionMetrics:
    question_id: str
    question_type: str
    n: int
    accuracy: float
    f1: float  # macro-F1 for choice/score; binary F1 with positive "true" for noul
    log_loss: float
    mae: float | None
    confusion: ConfusionMatrix


@dataclass
class MetricsReport:
    questions: dict[str, QuestionMetrics]
    exact_match: float
    n_records: int

    @property
    def mean_macro_f1(self) -> float:
        return fmean(q.f1 for q in self.questions.values()) if self.questions else 0.0

    @property
    def mean_log_loss(self) -> float:
        return fmean(q.log_loss for q in self.questions.values()) if self.questions else 0.0

    def value(self, metric: SelectMetric) -> float:
        return float(getattr(self, metric))

    def to_dict(self) -> dict[str, Any]:
        return {
            "questions": {qid: asdict(metrics) for qid, metrics in self.questions.items()},
            "exact_match": self.exact_match,
            "mean_macro_f1": self.mean_macro_f1,
            "mean_log_loss": self.mean_log_loss,
            "n_records": self.n_records,
        }


def is_better(metric: str, new: float, best: float | None) -> bool:
    if best is None:
        return True
    return new > best if HIGHER_IS_BETTER[metric] else new < best


def confusion_matrix(preds: list[Prediction]) -> ConfusionMatrix:
    labels = list(dict.fromkeys(option for p in preds for option in p.option_ids))
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for p in preds:
        matrix[index[p.actual]][index[p.predicted]] += 1
    return ConfusionMatrix(labels, matrix)


def _f1(label: str, preds: list[Prediction]) -> float:
    tp = sum(1 for p in preds if p.predicted == label and p.actual == label)
    fp = sum(1 for p in preds if p.predicted == label and p.actual != label)
    fn = sum(1 for p in preds if p.predicted != label and p.actual == label)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def _expected_level(values: list[float]) -> float:
    return sum(index * value for index, value in enumerate(values))


def question_metrics(preds: list[Prediction]) -> QuestionMetrics:
    first = preds[0]
    if first.question_type == "noul":
        f1 = _f1("true", preds)
    else:
        classes = sorted({p.actual for p in preds} | {p.predicted for p in preds})
        f1 = fmean(_f1(label, preds) for label in classes)
    mae = None
    if first.question_type == "score":
        mae = fmean(abs(_expected_level(p.probs) - _expected_level(p.target)) for p in preds)
    return QuestionMetrics(
        question_id=first.question_id,
        question_type=first.question_type,
        n=len(preds),
        accuracy=fmean(1.0 if p.correct else 0.0 for p in preds),
        f1=f1,
        log_loss=fmean(p.log_loss() for p in preds),
        mae=mae,
        confusion=confusion_matrix(preds),
    )


def summarize(records: list[list[Prediction]]) -> MetricsReport:
    """records: the labeled-question predictions of each eval record."""
    by_question: dict[str, list[Prediction]] = defaultdict(list)
    for preds in records:
        for p in preds:
            by_question[p.question_id].append(p)
    labeled = [preds for preds in records if preds]
    exact = (
        fmean(1.0 if all(p.correct for p in preds) else 0.0 for preds in labeled)
        if labeled
        else 0.0
    )
    return MetricsReport(
        questions={qid: question_metrics(preds) for qid, preds in by_question.items()},
        exact_match=exact,
        n_records=len(labeled),
    )
