"""The CFPB benchmark's scorer and split (benchmarks/cfpb/): the scorer must agree with
clef-tune's own metrics, and the split must keep the test months out of training."""

from __future__ import annotations

import random

import numpy as np
import pytest

from benchmarks.cfpb import claude_arm, prepare, score
from benchmarks.cfpb.common import TEST_WINDOW, TRAIN_WINDOW
from clef_tune.calibration import expected_calibration_error
from clef_tune.metrics import Prediction, question_metrics


def _predictions(truth, predicted, k):
    out = []
    for t, p in zip(truth, predicted, strict=True):
        probs = [0.0] * k
        probs[p] = 1.0
        target = [0.0] * k
        target[t] = 1.0
        out.append(Prediction("q", "choice", [str(i) for i in range(k)], probs, target))
    return out


@pytest.mark.parametrize("seed", range(5))
def test_macro_f1_matches_clef_tunes(seed):
    rng = random.Random(seed)
    k = 4
    truth = [rng.randrange(k) for _ in range(60)]
    predicted = [rng.choice([t, rng.randrange(k)]) for t in truth]
    ours = score.macro_f1(np.array(truth), np.array(predicted), k)
    assert ours == pytest.approx(question_metrics(_predictions(truth, predicted, k)).f1)


def test_an_unanswered_row_is_wrong_and_adds_no_class():
    truth = np.array([0, 0, 1, 1])
    # Answering class 1 where unanswered would add a false positive; -1 must not.
    assert score.macro_f1(truth, np.array([0, -1, 1, 1]), 3) == pytest.approx((2 / 3 + 1.0) / 2)


def test_ece_matches_clef_tunes():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet([1, 1, 1], size=80)
    targets = rng.integers(0, 3, size=80)
    predicted = probs.argmax(axis=1)
    theirs = expected_calibration_error(
        [list(p) for p in probs], [list(np.eye(3)[t]) for t in targets]
    )
    assert score.ece(probs.max(axis=1), predicted == targets) == pytest.approx(theirs)


def test_paired_difference_uses_the_same_rows():
    options = {q: ["a", "b"] for q in score.QUESTIONS}
    n = 10
    truth = {q: np.zeros(n, dtype=int) for q in score.QUESTIONS}

    def answers(right: int):
        predicted = {q: np.array([0] * right + [1] * (n - right)) for q in score.QUESTIONS}
        conf = {q: np.full(n, 0.9) for q in score.QUESTIONS}
        return score.Answers([str(i) for i in range(n)], options, truth, predicted, conf)

    d = score.paired_difference(answers(7), answers(4), score.resamples(n, 200))
    assert d["difference"] == pytest.approx(0.3)
    assert d["difference_95"][0] >= 0  # a never loses to b on any row


def _pool():
    rng = random.Random(1)
    pool = []
    days = ["2025-01-02", "2025-08-09", "2026-05-31", "2026-06-15", "2026-07-03", "2026-07-30"]
    issues = list(prepare.ISSUES)
    for k in range(4000):
        pool.append(
            {
                "complaint_id": str(100000 + k),
                "date_received": rng.choice(days),
                "cfpb_product": rng.choice(list(prepare.PRODUCTS)),
                "cfpb_issue": rng.choice(issues),
                "cfpb_response": rng.choice(list(prepare.OUTCOMES)),
                "narrative": "x" * rng.choice([100, 500, prepare.MAX_CHARS + 1]),
            }
        )
    return pool


def test_split_keeps_test_months_apart_and_drops_long_complaints():
    train, validation, test, info = prepare.split(_pool())
    for part, window in ((train, TRAIN_WINDOW), (validation, TRAIN_WINDOW), (test, TEST_WINDOW)):
        first, last = (day.isoformat() for day in window)
        assert all(first <= r["date_received"] <= last for r in part)
        assert all(len(r["complaint"]) <= prepare.MAX_CHARS for r in part)
    ids = [{r["complaint_id"] for r in part} for part in (train, validation, test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert len(info["issues"]) == prepare.ISSUE_CHOICES
    assert {r["issue"] for r in train} <= set(info["issues"]) | {prepare.OTHER}
    assert all("2026-06" not in r["date_received"] for r in train + validation + test)


def test_split_is_repeatable():
    assert prepare.split(_pool())[0] == prepare.split(_pool())[0]


def test_few_shot_examples_come_from_train_and_cover_every_option():
    questions = {
        q: {"criteria": {option: "" for option in options}}
        for q, options in (
            ("issue", ["a", "b"]),
            ("product", ["c", "d"]),
            ("outcome", ["e", "f", "g"]),
        )
    }
    rng = random.Random(0)
    train = [
        {
            "complaint_id": str(k),
            "issue": rng.choice("ab"),
            "product": rng.choice("cd"),
            "outcome": rng.choice("efg"),
        }
        for k in range(200)
    ]
    picked = claude_arm.pick_examples(train, questions, n=10)
    assert {r["complaint_id"] for r in picked} <= {r["complaint_id"] for r in train}
    for q, question in questions.items():
        assert {r[q] for r in picked} == set(question["criteria"])
    assert picked == claude_arm.pick_examples(train, questions, n=10)
