import pytest

from clef_tune.schema import Question, Schema, SchemaError
from clef_tune.yamlio import load_yaml_text


def test_choice_options_are_sorted_like_clef():
    q = Question.from_dict(
        "dept", {"type": "choice", "criteria": {"technical": "Bugs", "billing": "Money"}}
    )
    assert q.options() == [("billing", "Money"), ("technical", "Bugs")]


def test_score_options_are_indices():
    q = Question.from_dict("urgency", {"type": "score", "criteria": ["Low", "High"]})
    assert q.option_ids() == ["0", "1"]
    assert q.options() == [("0", "Low"), ("1", "High")]


def test_noul_options_default_and_override():
    q = Question.from_dict("outage", {"type": "noul", "criteria": {"true": "Down"}})
    assert q.options() == [
        ("true", "Down"),
        ("false", "The proposition is false or the answer is no."),
    ]


@pytest.mark.parametrize(
    "data, message",
    [
        ({"type": "bogus"}, "type must be one of"),
        ({"type": "choice"}, "non-empty criteria mapping"),
        ({"type": "score", "criteria": []}, "non-empty criteria list"),
        ({"type": "noul", "criteria": {"maybe": "?"}}, "must be true/false"),
        ({"type": "noul", "extra": 1}, "unknown keys"),
        ({"type": "noul", "instructions": 5}, "instructions must be text"),
        ("noul", "must be a mapping"),
    ],
)
def test_invalid_questions(data, message):
    with pytest.raises(SchemaError, match=message):
        Question.from_dict("q", data)


def test_empty_schema_rejected():
    with pytest.raises(SchemaError, match="non-empty 'questions'"):
        Schema.from_dict({"questions": {}})


def test_yaml_yes_no_keys_stay_strings():
    data = load_yaml_text(
        "questions:\n"
        "  decision:\n"
        "    type: choice\n"
        "    criteria: {yes: Approve, no: Reject, on: Hold, off: Drop}\n"
    )
    question = Schema.from_dict(data).questions["decision"]
    assert question.option_ids() == ["no", "off", "on", "yes"]


def test_yaml_bool_and_int_keys_normalize():
    # Not `1:` next to `true:`: Python treats True == 1, so those two keys would collide.
    data = load_yaml_text("questions:\n  q:\n    type: choice\n    criteria: {true: T, 2: two}\n")
    assert Schema.from_dict(data).questions["q"].option_ids() == ["2", "true"]


def test_to_clef_round_trips_through_from_dict():
    q = Question.from_dict(
        "d", {"type": "choice", "instructions": "Which?", "criteria": {"a": "A", "b": None}}
    )
    assert Question.from_dict("d", q.to_clef()) == q


def test_save_and_load_preserve_order(tmp_path):
    schema = Schema.from_dict(
        {
            "questions": {
                "d": {"type": "choice", "criteria": {"yes": "Y", "no": "N"}},
                "u": {"type": "score", "criteria": ["a", "b"]},
            }
        }
    )
    path = tmp_path / "schema.yaml"
    schema.save(path)
    loaded = Schema.load(path)
    assert loaded == schema
    assert list(loaded.questions) == ["d", "u"]
