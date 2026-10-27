"""Per-question temperature scaling and top-label expected calibration error (ECE).

Rows within one question may have different option counts (multi-task JSONL), so
everything works on lists of rows rather than fixed-width arrays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

T_MIN = 0.05
T_MAX = 20.0
MIN_ROWS = 20
ECE_BINS = 15
_INV_GOLDEN = (math.sqrt(5) - 1) / 2


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    scaled = [z / temperature for z in logits]
    top = max(scaled)
    exps = [math.exp(z - top) for z in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def nll(logits_rows: list[list[float]], targets: list[list[float]], temperature: float) -> float:
    if not logits_rows:
        return 0.0  # a question with no labeled eval rows
    total = 0.0
    for logits, target in zip(logits_rows, targets, strict=True):
        scaled = [z / temperature for z in logits]
        top = max(scaled)
        log_norm = top + math.log(sum(math.exp(z - top) for z in scaled))
        total -= sum(t * (z - log_norm) for z, t in zip(scaled, target, strict=True) if t > 0)
    return total / len(logits_rows)


def fit_temperature(
    logits_rows: list[list[float]],
    targets: list[list[float]],
    t_min: float = T_MIN,
    t_max: float = T_MAX,
    tol: float = 1e-4,
) -> float:
    """Golden-section search on log T. NLL is convex in 1/T, hence unimodal in log T."""
    lo, hi = math.log(t_min), math.log(t_max)

    def loss(u: float) -> float:
        return nll(logits_rows, targets, math.exp(u))

    c = hi - _INV_GOLDEN * (hi - lo)
    d = lo + _INV_GOLDEN * (hi - lo)
    fc, fd = loss(c), loss(d)
    while hi - lo > tol:
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - _INV_GOLDEN * (hi - lo)
            fc = loss(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + _INV_GOLDEN * (hi - lo)
            fd = loss(d)
    return math.exp((lo + hi) / 2)


def expected_calibration_error(
    probs_rows: list[list[float]], targets: list[list[float]], n_bins: int = ECE_BINS
) -> float:
    bins = [[0, 0.0, 0.0] for _ in range(n_bins)]  # count, sum of confidence, sum of correct
    for probs, target in zip(probs_rows, targets, strict=True):
        predicted = max(range(len(probs)), key=probs.__getitem__)
        actual = max(range(len(target)), key=target.__getitem__)
        confidence = probs[predicted]
        index = min(int(confidence * n_bins), n_bins - 1)
        bins[index][0] += 1
        bins[index][1] += confidence
        bins[index][2] += 1.0 if predicted == actual else 0.0
    n = len(probs_rows)
    if n == 0:
        return 0.0
    return sum(abs(conf - correct) / n for count, conf, correct in bins if count)


def temper_probabilities(probs: list[float], temperature: float) -> list[float]:
    """Calibrate already-softmaxed probabilities: p_i^(1/T) / sum_j p_j^(1/T)."""
    powered = [max(p, 1e-300) ** (1 / temperature) for p in probs]
    total = sum(powered)
    return [value / total for value in powered]


@dataclass
class QuestionCalibration:
    temperature: float
    n: int
    ece_before: float
    ece_after: float
    log_loss_before: float
    log_loss_after: float
    fitted: bool


def calibrate(
    logits_by_question: dict[str, list[list[float]]],
    targets_by_question: dict[str, list[list[float]]],
    min_rows: int = MIN_ROWS,
) -> dict[str, QuestionCalibration]:
    result: dict[str, QuestionCalibration] = {}
    for qid, rows in logits_by_question.items():
        targets = targets_by_question[qid]
        fitted = len(rows) >= min_rows
        temperature = fit_temperature(rows, targets) if fitted else 1.0
        before = [softmax(z) for z in rows]
        after = [softmax(z, temperature) for z in rows]
        result[qid] = QuestionCalibration(
            temperature=temperature,
            n=len(rows),
            ece_before=expected_calibration_error(before, targets),
            ece_after=expected_calibration_error(after, targets),
            log_loss_before=nll(rows, targets, 1.0),
            log_loss_after=nll(rows, targets, temperature),
            fitted=fitted,
        )
    return result


def temperatures(calibration: dict[str, QuestionCalibration]) -> dict[str, float]:
    return {qid: item.temperature for qid, item in calibration.items()}
