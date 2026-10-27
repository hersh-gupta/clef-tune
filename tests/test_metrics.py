import math

import pytest

from clef_tune.metrics import Prediction, is_better, question_metrics, summarize


def pred(actual, predicted, options=("a", "b", "c"), qid="q", qtype="choice", conf=0.7):
    other = (1 - conf) / (len(options) - 1)
    probs = [conf if option == predicted else other for option in options]
    target = [1.0 if option == actual else 0.0 for option in options]
    return Prediction(qid, qtype, list(options), probs, target)


def test_choice_accuracy_macro_f1_and_confusion():
    pairs = [("a", "a"), ("a", "a"), ("a", "b"), ("b", "b"), ("b", "a"), ("c", "c")]
    metrics = question_metrics([pred(actual, predicted) for actual, predicted in pairs])
    assert metrics.n == 6
    assert metrics.accuracy == pytest.approx(4 / 6)
    assert metrics.f1 == pytest.approx((2 / 3 + 1 / 2 + 1) / 3)
    assert metrics.confusion.labels == ["a", "b", "c"]
    assert metrics.confusion.matrix == [[2, 1, 0], [1, 1, 0], [0, 0, 1]]
    assert metrics.mae is None


def test_log_loss():
    metrics = question_metrics([pred("a", "a"), pred("a", "b")])
    assert metrics.log_loss == pytest.approx((-math.log(0.7) - math.log(0.15)) / 2)


def test_noul_uses_binary_f1_on_true():
    options = ("true", "false")
    pairs = [("true", "true"), ("true", "false"), ("false", "true"), ("false", "false")]
    preds = [pred(a, p, options, qtype="noul") for a, p in pairs]
    assert question_metrics(preds).f1 == pytest.approx(0.5)


def test_noul_f1_zero_without_true_positives():
    preds = [pred("true", "false", ("true", "false"), qtype="noul")]
    assert question_metrics(preds).f1 == 0.0


def test_score_mae_uses_expected_level():
    p = Prediction("u", "score", ["0", "1", "2"], [0.2, 0.3, 0.5], [0.0, 0.0, 1.0])
    assert question_metrics([p]).mae == pytest.approx(0.7)


def test_soft_target_actual_is_argmax():
    p = Prediction("q", "choice", ["a", "b"], [0.4, 0.6], [0.3, 0.7])
    assert p.actual == "b" and p.correct


def test_summarize_exact_match_and_aggregates():
    records = [
        [pred("a", "a"), pred("true", "true", ("true", "false"), qid="n", qtype="noul")],
        [pred("a", "b"), pred("true", "true", ("true", "false"), qid="n", qtype="noul")],
        [],
    ]
    report = summarize(records)
    assert report.n_records == 2
    assert report.exact_match == pytest.approx(0.5)
    assert set(report.questions) == {"q", "n"}
    assert report.mean_macro_f1 == pytest.approx(
        (report.questions["q"].f1 + report.questions["n"].f1) / 2
    )
    assert report.value("exact_match") == report.exact_match
    data = report.to_dict()
    assert data["questions"]["q"]["confusion"]["labels"] == ["a", "b", "c"]
    assert data["mean_macro_f1"] == report.mean_macro_f1


def test_is_better():
    assert is_better("mean_macro_f1", 0.8, None)
    assert is_better("mean_macro_f1", 0.8, 0.7)
    assert not is_better("exact_match", 0.6, 0.7)
    assert is_better("mean_log_loss", 0.3, 0.4)
    assert not is_better("mean_log_loss", 0.5, 0.4)
