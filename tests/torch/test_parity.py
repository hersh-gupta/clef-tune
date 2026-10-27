"""Plan 1's torch-free token counter must agree with Clef's encode_record on the real tokenizer."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
from transformers import AutoTokenizer  # noqa: E402

from clef_tune.clef_code import load_clef_code  # noqa: E402
from clef_tune.config import PINNED_REVISIONS  # noqa: E402
from clef_tune.prompt import RecordTokenCounter, get_token_counter  # noqa: E402
from clef_tune.schema import Schema  # noqa: E402

pytestmark = [pytest.mark.torch, pytest.mark.network]

STATES = [
    "hello world",
    "",
    "  leading and trailing  ",
    "Ünïcödé — “quotes” and emoji 🚀🔥",
    "中文测试，日本語のテスト",
    "line1\nline2\r\n\ttab",
    {
        "subject": "Card declined",
        "body": "Help!",
        "n": 3,
        "x": None,
        "nested": {"a": [1, 2.5, True]},
    },
    "x" * 3000,
    '{"already": "json"}',
    "<|im_start|> literal special",
    " sep ",
]
QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "Money", "Technical": "Bugs 🐛", "NA": None},
    },
    "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
    "outage": {"type": "noul"},
    "spam": {
        "type": "noul",
        "instructions": "",
        "criteria": {"true": "Unsolicited", "false": "Legit"},
    },
}


@pytest.mark.parametrize("subset", [list(QUESTIONS), ["outage"], ["urgency", "department"]])
def test_token_counter_matches_clefs_encode_record(subset):
    revision = PINNED_REVISIONS["Cloudflare/clef"]
    tokenizer = AutoTokenizer.from_pretrained("Cloudflare/clef", revision=revision)
    jsm = load_clef_code(Path(__file__).parent / "fixtures")
    questions = Schema.from_dict({"questions": {q: QUESTIONS[q] for q in subset}}).questions
    counter = RecordTokenCounter(get_token_counter("Cloudflare/clef", revision), max_length=1024)
    for state in STATES:
        record = {
            "id": "r",
            "state": state,
            "questions": {q: v.to_clef() for q, v in questions.items()},
        }
        expected = len(jsm.encode_record(tokenizer, record, max_length=1024).input_ids)
        assert counter.count_records([(state, questions, 0)])[0].total == expected, repr(state)[:40]
