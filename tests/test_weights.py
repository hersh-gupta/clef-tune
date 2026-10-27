import pytest

from clef_tune.data import Record
from clef_tune.labels import Label
from clef_tune.weights import class_weights, target_weight


def rec(i, dept):
    return Record(str(i), "s", {}, {"department": Label(hard=dept)}, [], i + 2)


def test_balanced_weights_hand_computed():
    records = [rec(i, "billing") for i in range(8)] + [rec(i, "technical") for i in range(8, 10)]
    assert class_weights(records) == {
        "department": {"billing": pytest.approx(0.625), "technical": pytest.approx(2.5)}
    }


def test_weights_clip_at_both_ends():
    records = [rec(i, "billing") for i in range(1000)] + [rec(1000, "technical")]
    assert class_weights(records)["department"]["technical"] == 10.0
    many = [rec(i, "big") for i in range(1000)] + [rec(1000 + k, f"c{k}") for k in range(10)]
    assert class_weights(many)["department"]["big"] == 0.1  # 1010 / (11 * 1000) < 0.1


def test_soft_labels_count_by_top_option_and_use_the_expected_weight():
    soft = Record(
        "s", "s", {}, {"department": Label(soft={"billing": 0.6, "technical": 0.4})}, [], 2
    )
    assert class_weights([soft, rec(1, "technical")])["department"] == {
        "billing": 1.0,
        "technical": 1.0,
    }
    weights = {"billing": 0.625, "technical": 2.5}
    assert target_weight(weights, ["billing", "technical"], [0.5, 0.5]) == pytest.approx(1.5625)
    assert target_weight(weights, ["NA", "billing", "technical"], [1.0, 0.0, 0.0]) == 1.0
