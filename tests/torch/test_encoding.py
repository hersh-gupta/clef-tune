import dataclasses

import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

from clef_tune.clef_code import load_clef_code  # noqa: E402
from clef_tune.config import load_config  # noqa: E402
from clef_tune.encoding import collate, materialize, prepare_items  # noqa: E402
from clef_tune.validation import validate  # noqa: E402

pytestmark = pytest.mark.torch


def setup(tiny_release, tiny_project):
    jsm = load_clef_code(tiny_release)
    processor = AutoProcessor.from_pretrained(tiny_release)
    records = validate(load_config(tiny_project()), None).train
    return jsm, processor, records


def test_items_follow_clefs_question_and_option_order(tiny_release, tiny_project):
    jsm, processor, records = setup(tiny_release, tiny_project)
    item = prepare_items(jsm, processor, records[:1], max_length=1024)[0]
    assert item.question_ids == ["outage", "department"]
    assert item.option_ids == {"outage": ["true", "false"], "department": ["billing", "technical"]}
    assert item.targets == {"outage": [1.0, 0.0], "department": [1.0, 0.0]}  # row t0: down, invoice
    assert item.length == len(item.encoded.input_ids)


def test_long_state_is_truncated_to_max_length(tiny_release, tiny_project):
    jsm, processor, records = setup(tiny_release, tiny_project)
    long = dataclasses.replace(records[0], state="word " * 2000)
    assert prepare_items(jsm, processor, [long], max_length=512)[0].length == 512


def test_image_records_encode_media_lazily(tiny_release, tiny_project, tmp_path):
    jsm, processor, records = setup(tiny_release, tiny_project)
    Image.new("RGB", (64, 64)).save(tmp_path / "a.png")
    with_image = dataclasses.replace(records[0], images=[str(tmp_path / "a.png")])
    (item,) = prepare_items(jsm, processor, [with_image], max_length=1024)
    (plain,) = prepare_items(jsm, processor, records[:1], max_length=1024)
    assert (
        item.encoded is None and item.length == plain.length + 67
    )  # 64 pads + 2 markers + newline
    encoded = materialize(item, jsm, processor, 1024)
    assert len(encoded.input_ids) == item.length and encoded.media["pixel_values"].shape[0] > 0


def test_collate_materializes_each_item(tiny_release, tiny_project):
    jsm, processor, records = setup(tiny_release, tiny_project)
    items = prepare_items(jsm, processor, records[:2], max_length=1024)
    batch = collate(items, jsm, processor, 1024, torch.device("cpu"))
    assert batch["input_ids"].shape[0] == 2
