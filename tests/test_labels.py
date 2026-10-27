import math

import pytest

from clef_tune.labels import Label, LabelError, parse_label
from clef_tune.schema import Question

CHOICE = Question.from_dict(
    "dept", {"type": "choice", "criteria": {"billing": "", "technical": "", "NA": ""}}
)
SCORE = Question.from_dict("urgency", {"type": "score", "criteria": ["Low", "Medium", "High"]})
NOUL = Question.from_dict("outage", {"type": "noul"})


@pytest.mark.parametrize("raw", [None, "", "   ", math.nan])
def test_missing_values_mask_the_question(raw):
    assert parse_label(CHOICE, raw) == (None, None)


@pytest.mark.parametrize(
    "raw, expected",
    [("billing", "billing"), ("  technical ", "technical"), ("NA", "NA")],
)
def test_choice_exact(raw, expected):
    assert parse_label(CHOICE, raw) == (Label(hard=expected), None)


def test_choice_case_fix_warns():
    label, warning = parse_label(CHOICE, "Billing")
    assert label == Label(hard="billing")
    assert "matched option 'billing' by case" in warning


def test_choice_unknown_value():
    with pytest.raises(LabelError, match="'biling' is not an option"):
        parse_label(CHOICE, "biling")


@pytest.mark.parametrize(
    "raw, expected",
    [("2", "2"), (" 0 ", "0"), ("1.0", "1"), (2, "2"), (1.0, "1"), ("High", "2")],
)
def test_score_index_or_text(raw, expected):
    assert parse_label(SCORE, raw)[0] == Label(hard=expected)


def test_score_text_case_fix_warns():
    label, warning = parse_label(SCORE, "medium")
    assert label == Label(hard="1") and "by case" in warning


@pytest.mark.parametrize("raw", ["3", -1, True, "urgent", 1.5])
def test_score_invalid(raw):
    with pytest.raises(LabelError, match="not a score level"):
        parse_label(SCORE, raw)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("true", "true"),
        ("YES", "true"),
        ("1", "true"),
        ("t", "true"),
        (True, "true"),
        (1, "true"),
        ("1.0", "true"),
        ("false", "false"),
        ("No", "false"),
        ("0", "false"),
        ("F", "false"),
        (False, "false"),
        (0, "false"),
    ],
)
def test_noul_values(raw, expected):
    assert parse_label(NOUL, raw) == (Label(hard=expected), None)


def test_noul_invalid():
    with pytest.raises(LabelError, match="not true/false"):
        parse_label(NOUL, "maybe")


def test_soft_label():
    label, _ = parse_label(CHOICE, {"billing": 0.7, "technical": 0.3})
    assert label.soft == {"billing": 0.7, "technical": 0.3}
    assert label.distribution(CHOICE.option_ids()) == [0.0, 0.7, 0.3]
    assert label.top() == "billing"
    assert label.to_json() == {"billing": 0.7, "technical": 0.3}


def test_soft_label_score_int_keys():
    label, _ = parse_label(SCORE, {0: 0.5, 2: 0.5})
    assert label.soft == {"0": 0.5, "2": 0.5}


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"billing": 0.5}, "sum to 0.5000"),
        ({"bogus": 1.0}, "'bogus' is not an option"),
        ({"billing": -0.2, "technical": 1.2}, "non-negative number"),
        ({"billing": math.nan, "technical": 1.0}, "non-negative number"),
        ({"billing": "x"}, "non-negative number"),
    ],
)
def test_soft_label_invalid(raw, message):
    with pytest.raises(LabelError, match=message):
        parse_label(CHOICE, raw)


def test_hard_label_distribution_and_json():
    label = Label(hard="technical")
    assert label.distribution(CHOICE.option_ids()) == [0.0, 0.0, 1.0]
    assert label.to_json() == "technical"
