import random

import pytest

from clef_tune.calibration import (
    calibrate,
    expected_calibration_error,
    fit_temperature,
    softmax,
    temper_probabilities,
    temperatures,
)


def synthetic(n, scale, seed=0):
    """Labels drawn from softmax(z); the model reports scale * z (overconfident when scale > 1)."""
    rng = random.Random(seed)
    logits, targets = [], []
    for _ in range(n):
        true = [rng.gauss(0, 1.5) for _ in range(3)]
        probs = softmax(true)
        label = rng.choices(range(3), weights=probs)[0]
        logits.append([scale * z for z in true])
        targets.append([1.0 if i == label else 0.0 for i in range(3)])
    return logits, targets


def test_fit_temperature_recovers_overconfidence():
    logits, targets = synthetic(4000, scale=3.0)
    assert fit_temperature(logits, targets) == pytest.approx(3.0, rel=0.1)


def test_fit_temperature_near_one_when_calibrated():
    logits, targets = synthetic(4000, scale=1.0, seed=1)
    assert fit_temperature(logits, targets) == pytest.approx(1.0, rel=0.1)


def test_ece_hand_example():
    probs = [[0.9, 0.1], [0.6, 0.4], [0.8, 0.2]]
    targets = [[1, 0], [0, 1], [1, 0]]
    assert expected_calibration_error(probs, targets, n_bins=10) == pytest.approx(0.3)


def test_temper_probabilities_matches_softmax_with_temperature():
    logits = [2.0, -1.0, 0.5]
    assert temper_probabilities(softmax(logits), 2.5) == pytest.approx(softmax(logits, 2.5))


def test_calibrate_improves_ece_and_skips_small_questions():
    logits, targets = synthetic(2000, scale=3.0, seed=2)
    result = calibrate({"big": logits, "small": logits[:5]}, {"big": targets, "small": targets[:5]})
    big, small = result["big"], result["small"]
    assert big.fitted and big.n == 2000
    assert big.ece_after < big.ece_before
    assert big.log_loss_after < big.log_loss_before
    assert not small.fitted and small.temperature == 1.0
    assert small.ece_after == small.ece_before
    assert temperatures(result) == {"big": big.temperature, "small": 1.0}
