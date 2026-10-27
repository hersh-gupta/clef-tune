import dataclasses

import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402
from transformers import AutoProcessor, image_utils  # noqa: E402

from clef_tune import encoding  # noqa: E402
from clef_tune.clef_code import load_clef_code  # noqa: E402
from clef_tune.config import load_config  # noqa: E402
from clef_tune.encoding import (  # noqa: E402
    ImageLoadError,
    collate,
    materialize,
    open_image,
    prepare_items,
)
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


def test_exif_orientation_is_applied_when_measuring_and_encoding(
    tiny_release, tiny_project, tmp_path
):
    jsm, processor, records = setup(tiny_release, tiny_project)
    exif = Image.Exif()
    exif[0x0112] = 6  # stored 200x100, shown rotated 90 degrees: 100 wide, 200 tall
    Image.new("RGB", (200, 100)).save(tmp_path / "rotated.jpg", exif=exif.tobytes())
    Image.new("RGB", (100, 200)).save(tmp_path / "upright.png")
    rotated, upright = (
        dataclasses.replace(records[0], images=[str(tmp_path / name)])
        for name in ("rotated.jpg", "upright.png")
    )
    assert encoding._image_size(str(tmp_path / "rotated.jpg")) == (100, 200)
    assert open_image(str(tmp_path / "rotated.jpg")).size == (100, 200)
    items = prepare_items(jsm, processor, [rotated, upright], max_length=1024)
    encoded = [materialize(item, jsm, processor, 1024) for item in items]
    assert [len(e.input_ids) for e in encoded] == [item.length for item in items]
    assert torch.equal(encoded[0].media["image_grid_thw"], encoded[1].media["image_grid_thw"])


URL = "https://example.com/receipt.png"


def test_url_images_are_retried_with_a_timeout(monkeypatch):
    calls = []

    def flaky(source, timeout=None):
        calls.append(timeout)
        if len(calls) < 3:
            raise OSError("connection reset")
        return Image.new("RGB", (8, 8))

    monkeypatch.setattr(image_utils, "load_image", flaky)
    monkeypatch.setattr(encoding, "IMAGE_RETRY_DELAY_S", 0.0)
    assert open_image(URL).size == (8, 8)
    assert calls == [encoding.IMAGE_TIMEOUT_S] * 3


def test_a_url_that_keeps_failing_names_the_url(monkeypatch):
    def down(source, timeout=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr(image_utils, "load_image", down)
    monkeypatch.setattr(encoding, "IMAGE_RETRY_DELAY_S", 0.0)
    with pytest.raises(ImageLoadError, match=f"{URL}.*3 attempts.*timed out"):
        open_image(URL)
