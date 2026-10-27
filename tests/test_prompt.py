import pytest

from clef_tune.config import PINNED_REVISIONS
from clef_tune.prompt import (
    PREFIX,
    SUFFIX,
    RecordTokenCounter,
    get_token_counter,
    image_token_count,
    media_token_count,
    prompt_pieces,
    render,
    schema_pieces,
)
from clef_tune.schema import Question
from tests.helpers import CharCounter

URGENT = {"urgent": Question.from_dict("urgent", {"type": "noul", "instructions": "Is it urgent?"})}


def test_prompt_matches_clef_encode_record_layout():
    text = "".join(piece.text for piece in prompt_pieces("hi", URGENT))
    expected = (
        PREFIX
        + "hi"
        + "\n\nSCHEMA FIELDS:\n"
        + "\nFIELD 1\nID: urgent\nTYPE: noul\nINSTRUCTION: "
        + "Is it urgent?"
        + "\nALLOWED OPTIONS:\n"
        + 'OPTION 1: {"description":"The proposition is true or the answer is yes.",'
        + '"option_id":"true"}\n'
        + 'OPTION 2: {"description":"The proposition is false or the answer is no.",'
        + '"option_id":"false"}\n'
        + "END FIELD\n"
        + SUFFIX
    )
    assert text == expected


def test_prefix_and_suffix_are_clefs():
    assert PREFIX.startswith("<|im_start|>system\nRead the complete state and schema.")
    assert PREFIX.endswith("<|im_end|>\n<|im_start|>user\nSTATE:\n")
    assert SUFFIX == (
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
    )


def test_piece_kinds_mark_spans_for_the_preview():
    kinds = [piece.kind for piece in prompt_pieces({"a": 1}, URGENT, n_images=2)]
    assert kinds[:3] == ["prefix", "media", "state"]
    assert kinds.count("question") == 1 and kinds.count("option") == 2
    assert kinds[-1] == "suffix"


def test_missing_instructions_fall_back_to_question_id():
    question = Question.from_dict("refund", {"type": "noul"})
    pieces = schema_pieces({"refund": question})
    assert [p.text for p in pieces if p.kind == "question"] == ["refund"]


def test_option_without_description_omits_it():
    question = Question.from_dict("d", {"type": "choice", "criteria": {"x": None}})
    assert [p.text for p in schema_pieces({"d": question}) if p.kind == "option"] == [
        '{"option_id":"x"}'
    ]


def test_render_is_compact_sorted_json():
    assert render({"b": 1, "a": "é"}) == '{"a":"é","b":1}'
    assert render("plain") == "plain"


def test_token_counts_and_truncation():
    counter = RecordTokenCounter(CharCounter(), max_length=1000)
    fixed = len("".join(p.text for p in prompt_pieces("", URGENT)))
    short, long = counter.count_records([("hi", URGENT, 0), ("x" * 2000, URGENT, 0)])
    assert short.fixed == fixed and short.state == 2
    assert short.total == fixed + 2 and not short.truncated
    assert long.truncated and long.total == 1000


def test_media_tokens_add_to_fixed():
    counter = RecordTokenCounter(CharCounter(), max_length=10_000)
    (plain,) = counter.count_records([("hi", URGENT, 0)])
    (with_image,) = counter.count_records([("hi", URGENT, 1027)])
    assert with_image.fixed - plain.fixed == 1027


def test_schema_too_long():
    counter = RecordTokenCounter(CharCounter(), max_length=256)
    (count,) = counter.count_records([("hi", URGENT, 0)])
    assert count.schema_too_long and not count.truncated and count.total == count.fixed


def test_image_token_count_matches_qwen_smart_resize():
    assert image_token_count(1024, 1024) == 1024
    assert image_token_count(100, 100) == 64  # upscaled to the 65536-pixel minimum: 256x256
    assert image_token_count(8000, 6000) == 16170  # downscaled to the 16777216-pixel maximum


def test_media_token_count_adds_vision_markers_and_newline():
    assert media_token_count([]) == 0
    assert media_token_count([(1024, 1024), (100, 100)]) == (1024 + 2) + (64 + 2) + 1


@pytest.mark.network
def test_real_clef_tokenizer():
    counter = get_token_counter("Cloudflare/clef", PINNED_REVISIONS["Cloudflare/clef"])
    assert counter.count("<|im_start|>") == 1
    assert counter.count_many(["hello world", ""]) == [2, 0]
