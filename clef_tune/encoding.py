"""Training items: Clef's own encoding, with media encoded lazily per batch."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from clef_tune.data import URL_PREFIXES, Record
from clef_tune.prompt import DEFAULT_URL_IMAGE_SIZE, media_token_count

IMAGE_TIMEOUT_S = 30.0
IMAGE_ATTEMPTS = 3  # the first try and 2 retries
IMAGE_RETRY_DELAY_S = 2.0  # times the attempt number
_EXIF_ORIENTATION = 0x0112
_SWAPS_WIDTH_AND_HEIGHT = {5, 6, 7, 8}  # EXIF orientations that turn the image by 90 degrees


class ImageLoadError(OSError):
    """An image URL could not be fetched; the message names it."""


@dataclass
class TrainItem:
    record: Record | None
    length: int
    question_ids: list[str]
    option_ids: dict[str, list[str]]
    targets: dict[str, list[float]]
    encoded: Any | None  # Clef's EncodedRecord; None for records with images


def _is_url(source: Any) -> bool:
    return isinstance(source, str) and source.startswith(URL_PREFIXES)


def open_image(source: Any):
    """Load an image the way Clef's processor does, with transformers' load_image: a path, an
    http(s) URL, a base64 or data-URI string, or a PIL image; EXIF orientation applied, RGB.
    URLs get a timeout and retries, so a stalled host cannot hang a run."""
    from transformers.image_utils import load_image

    if not _is_url(source):
        return load_image(source)
    for attempt in range(1, IMAGE_ATTEMPTS):
        try:
            return load_image(source, timeout=IMAGE_TIMEOUT_S)
        except Exception:
            time.sleep(IMAGE_RETRY_DELAY_S * attempt)
    try:
        return load_image(source, timeout=IMAGE_TIMEOUT_S)
    except Exception as error:
        raise ImageLoadError(
            f"cannot load image {source} after {IMAGE_ATTEMPTS} attempts: {error}"
        ) from error


def record_payload(record: Record, *, with_images: bool) -> dict:
    payload: dict[str, Any] = {
        "id": record.id,
        "state": record.state,
        "questions": {qid: q.to_clef() for qid, q in record.questions.items()},
    }
    if with_images and record.images:
        payload["images"] = [open_image(source) for source in record.images]
    return payload


def _image_size(source: Any) -> tuple[int, int]:
    """(width, height) as the model sees it, after EXIF orientation."""
    if _is_url(source):
        return DEFAULT_URL_IMAGE_SIZE  # fetched only when its batch is encoded
    if isinstance(source, str) and os.path.isfile(source):
        from PIL import Image

        with Image.open(source) as image:  # reads the header, not the pixels
            width, height = image.size
            turned = image.getexif().get(_EXIF_ORIENTATION) in _SWAPS_WIDTH_AND_HEIGHT
        return (height, width) if turned else (width, height)
    return open_image(source).size  # a PIL image or a base64 string


def prepare_items(clef_code, processor, records: list[Record], max_length: int) -> list[TrainItem]:
    items = []
    for record in records:
        encoded = clef_code.encode_record(
            processor.tokenizer, record_payload(record, with_images=False), max_length=max_length
        )
        question_ids = [q.question_id for q in encoded.questions]
        option_ids = {q.question_id: list(q.option_ids) for q in encoded.questions}
        targets = {
            qid: record.labels[qid].distribution(option_ids[qid])
            for qid in question_ids
            if qid in record.labels
        }
        length = len(encoded.input_ids)
        keep = encoded
        if record.images:
            length += media_token_count([_image_size(source) for source in record.images])
            keep = None
        items.append(TrainItem(record, length, question_ids, option_ids, targets, keep))
    return items


def materialize(item: TrainItem, clef_code, processor, max_length: int):
    if item.encoded is not None:
        return item.encoded
    return clef_code.encode_record(
        processor.tokenizer,
        record_payload(item.record, with_images=True),
        max_length=max_length,
        processor=processor,
    )


def collate(items: list[TrainItem], clef_code, processor, max_length: int, device) -> dict:
    encoded = [materialize(item, clef_code, processor, max_length) for item in items]
    return clef_code.collate_records(encoded, processor.tokenizer.pad_token_id, device)
