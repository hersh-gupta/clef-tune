"""Clef's exact prompt layout (mirrors encode_record in joint_schema_model.py), token
counting with the real Clef tokenizer, and Qwen vision token estimates for images.

encode_record tokenizes each piece separately and concatenates the ids, so a record's
token count is the sum of the pieces' counts, not the count of the joined string.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from clef_tune.schema import Question

if TYPE_CHECKING:
    from clef_tune.config import Config

SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
PREFIX = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
SUFFIX = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

# Qwen2VLImageProcessor settings from Clef's processor_config.json.
PATCH_SIZE = 16
MERGE_SIZE = 2
MIN_PIXELS = 65536
MAX_PIXELS = 16777216
DEFAULT_URL_IMAGE_SIZE = (1024, 1024)  # assumed for images we can't open (URLs)
COUNT_BATCH = 2048

PieceKind = Literal["prefix", "media", "state", "schema", "question", "option", "suffix"]


def render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True)
class Piece:
    kind: PieceKind
    text: str
    question_id: str | None = None


def schema_pieces(questions: dict[str, Question]) -> list[Piece]:
    pieces = [Piece("schema", "\n\nSCHEMA FIELDS:\n")]
    for index, (qid, question) in enumerate(questions.items()):
        header = f"\nFIELD {index + 1}\nID: {qid}\nTYPE: {question.type}\nINSTRUCTION: "
        pieces.append(Piece("schema", header, qid))
        instructions = question.instructions if question.instructions else str(qid)
        pieces.append(Piece("question", render(instructions), qid))
        pieces.append(Piece("schema", "\nALLOWED OPTIONS:\n", qid))
        for option_index, (option_id, description) in enumerate(question.options()):
            pieces.append(Piece("schema", f"OPTION {option_index + 1}: ", qid))
            semantics: dict[str, Any] = {"option_id": option_id}
            if description is not None:
                semantics["description"] = description
            pieces.append(Piece("option", render(semantics), qid))
            pieces.append(Piece("schema", "\n", qid))
        pieces.append(Piece("schema", "END FIELD\n", qid))
    return pieces


def prompt_pieces(state: Any, questions: dict[str, Question], n_images: int = 0) -> list[Piece]:
    pieces = [Piece("prefix", PREFIX)]
    if n_images:
        pieces.append(Piece("media", IMAGE_PLACEHOLDER * n_images + "\n"))
    pieces.append(Piece("state", render(state)))
    pieces.extend(schema_pieces(questions))
    pieces.append(Piece("suffix", SUFFIX))
    return pieces


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

    def count_many(self, texts: list[str]) -> list[int]: ...


class HFTokenCounter:
    """Counts tokens with a `tokenizer.json` (the same tokenizer Clef's processor uses)."""

    def __init__(self, tokenizer_path: str | Path) -> None:
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def count_many(self, texts: list[str]) -> list[int]:
        encodings = self._tokenizer.encode_batch(list(texts), add_special_tokens=False)
        return [len(encoding.ids) for encoding in encodings]


@lru_cache(maxsize=4)
def get_token_counter(
    model_id: str, revision: str, token: bool | str | None = None
) -> HFTokenCounter:
    """The model's tokenizer. `token` is passed to the Hub as it is: None uses this machine's
    login, and False sends none (a public download)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(model_id, "tokenizer.json", revision=revision, token=token)
    return HFTokenCounter(path)


@lru_cache(maxsize=4)
def _counter_from_file(path: str) -> HFTokenCounter:
    return HFTokenCounter(path)


def counter_for_config(config: Config, *, token: bool | str | None = None) -> HFTokenCounter:
    if config.model.local_path:
        local = config.resolve_path(config.model.local_path) / "tokenizer.json"
        if local.exists():
            return _counter_from_file(str(local))
    return get_token_counter(config.model.id, config.model.resolved_revision(), token)


@dataclass(frozen=True)
class TokenCount:
    fixed: int  # prefix + media + schema + suffix
    state: int  # state tokens before truncation
    max_length: int

    @property
    def schema_too_long(self) -> bool:
        return self.fixed > self.max_length

    @property
    def truncated(self) -> bool:
        return not self.schema_too_long and self.fixed + self.state > self.max_length

    @property
    def total(self) -> int:
        if self.schema_too_long:
            return self.fixed
        return min(self.fixed + self.state, self.max_length)


class RecordTokenCounter:
    """Counts full Clef prompts, caching schema counts (shared by most records)."""

    def __init__(self, counter: TokenCounter, max_length: int) -> None:
        self.counter = counter
        self.max_length = max_length
        self._frame = counter.count(PREFIX) + counter.count(SUFFIX)
        self._schema_cache: dict[str, int] = {}

    def schema_tokens(self, questions: dict[str, Question]) -> int:
        key = json.dumps([[qid, q.to_clef()] for qid, q in questions.items()], default=str)
        if key not in self._schema_cache:
            self._schema_cache[key] = sum(
                self.counter.count(piece.text) for piece in schema_pieces(questions)
            )
        return self._schema_cache[key]

    def count_records(self, items: list[tuple[Any, dict[str, Question], int]]) -> list[TokenCount]:
        """items: (state, questions, media_tokens) per record."""
        states: list[int] = []
        for start in range(0, len(items), COUNT_BATCH):
            chunk = items[start : start + COUNT_BATCH]
            states.extend(self.counter.count_many([render(state) for state, _, _ in chunk]))
        return [
            TokenCount(
                fixed=self._frame + media + self.schema_tokens(questions),
                state=n_state,
                max_length=self.max_length,
            )
            for (_, questions, media), n_state in zip(items, states, strict=True)
        ]


def image_token_count(
    width: int, height: int, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> int:
    """Vision tokens for one image after Qwen's smart_resize (patch 16, merge 2)."""
    factor = PATCH_SIZE * MERGE_SIZE
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return (h_bar // PATCH_SIZE) * (w_bar // PATCH_SIZE) // (MERGE_SIZE * MERGE_SIZE)


def media_token_count(sizes: list[tuple[int, int]]) -> int:
    """Tokens for a record's images: each image's pads + vision_start/end, plus one newline."""
    if not sizes:
        return 0
    return sum(image_token_count(width, height) + 2 for width, height in sizes) + 1
