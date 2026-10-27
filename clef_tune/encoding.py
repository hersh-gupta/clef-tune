"""Training items: Clef's own encoding, with media encoded lazily per batch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from clef_tune.data import URL_PREFIXES, Record
from clef_tune.prompt import DEFAULT_URL_IMAGE_SIZE, media_token_count


@dataclass
class TrainItem:
    record: Record | None
    length: int
    question_ids: list[str]
    option_ids: dict[str, list[str]]
    targets: dict[str, list[float]]
    encoded: Any | None  # Clef's EncodedRecord; None for records with images


def _load_image(source: str):
    from PIL import Image

    if source.startswith(URL_PREFIXES):
        from transformers.image_utils import load_image

        return load_image(source)
    return Image.open(source).convert("RGB")


def record_payload(record: Record, *, with_images: bool) -> dict:
    payload: dict[str, Any] = {
        "id": record.id,
        "state": record.state,
        "questions": {qid: q.to_clef() for qid, q in record.questions.items()},
    }
    if with_images and record.images:
        payload["images"] = [_load_image(source) for source in record.images]
    return payload


def _image_size(source: str) -> tuple[int, int]:
    if source.startswith(URL_PREFIXES):
        return DEFAULT_URL_IMAGE_SIZE
    from PIL import Image

    with Image.open(source) as image:
        return image.size


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
