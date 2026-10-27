"""Parse raw label values (CSV cells, JSON values) into option IDs or soft distributions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from clef_tune.schema import Question, option_key

NOUL_TRUE = {"true", "yes", "1", "t"}
NOUL_FALSE = {"false", "no", "0", "f"}
SOFT_SUM_TOLERANCE = 1e-3


class LabelError(ValueError):
    """A label value is present but cannot be interpreted for its question."""


@dataclass(frozen=True)
class Label:
    hard: str | None = None
    soft: dict[str, float] | None = None

    def distribution(self, option_ids: list[str]) -> list[float]:
        if self.soft is not None:
            return [float(self.soft.get(option, 0.0)) for option in option_ids]
        return [1.0 if option == self.hard else 0.0 for option in option_ids]

    def top(self) -> str:
        """The single most likely option: the hard label, or the soft argmax."""
        if self.hard is not None:
            return self.hard
        return max(self.soft, key=self.soft.__getitem__)

    def to_json(self) -> str | dict[str, float]:
        return self.hard if self.hard is not None else dict(self.soft)


def is_missing(raw: Any) -> bool:
    if raw is None:
        return True
    if isinstance(raw, float) and math.isnan(raw):
        return True
    return isinstance(raw, str) and raw.strip() == ""


def parse_label(question: Question, raw: Any) -> tuple[Label | None, str | None]:
    """Return ``(label, warning)``. ``label`` is None when the value is missing.

    Raises LabelError when the value is present but invalid for the question.
    """
    if is_missing(raw):
        return None, None
    if isinstance(raw, dict):
        return _parse_soft(question, raw), None
    if question.type == "choice":
        return _parse_choice(question, raw)
    if question.type == "score":
        return _parse_score(question, raw)
    return _parse_noul(raw), None


def _parse_choice(question: Question, raw: Any) -> tuple[Label, str | None]:
    value = raw.strip() if isinstance(raw, str) else option_key(raw)
    options = question.option_ids()
    if value in options:
        return Label(hard=value), None
    folded = [option for option in options if option.casefold() == value.casefold()]
    if len(folded) == 1:
        return Label(hard=folded[0]), f"'{value}' matched option '{folded[0]}' by case"
    raise LabelError(f"'{value}' is not an option (options: {', '.join(options)})")


def _parse_score(question: Question, raw: Any) -> tuple[Label, str | None]:
    levels = len(question.criteria)
    index: int | None = None
    warning = None
    if isinstance(raw, bool):
        index = None
    elif isinstance(raw, int | float):
        if float(raw).is_integer():
            index = int(raw)
    else:
        text = str(raw).strip()
        try:
            number: float | None = float(text)
        except ValueError:
            number = None
        if number is not None and number.is_integer():
            index = int(number)
        elif number is None:
            texts = [str(criterion).strip() for criterion in question.criteria]
            if text in texts:
                index = texts.index(text)
            else:
                folded = [i for i, t in enumerate(texts) if t.casefold() == text.casefold()]
                if len(folded) == 1:
                    index = folded[0]
                    warning = f"'{text}' matched level '{texts[index]}' by case"
    if index is None or not 0 <= index < levels:
        raise LabelError(
            f"{raw!r} is not a score level (expected 0..{levels - 1} or a criteria text)"
        )
    return Label(hard=str(index)), warning


def _parse_noul(raw: Any) -> Label:
    if isinstance(raw, bool):
        return Label(hard="true" if raw else "false")
    if isinstance(raw, int | float) and raw in (0, 1):
        return Label(hard="true" if raw == 1 else "false")
    text = str(raw).strip().casefold()
    try:  # spreadsheets sometimes write 1.0 / 0.0
        number = float(text)
        if number in (0.0, 1.0):
            text = "1" if number == 1.0 else "0"
    except ValueError:
        pass
    if text in NOUL_TRUE:
        return Label(hard="true")
    if text in NOUL_FALSE:
        return Label(hard="false")
    raise LabelError(f"{raw!r} is not true/false (use true/false, yes/no, 1/0 or t/f)")


def _parse_soft(question: Question, raw: dict[Any, Any]) -> Label:
    options = set(question.option_ids())
    distribution: dict[str, float] = {}
    for key, value in raw.items():
        option = option_key(key)
        if option not in options:
            raise LabelError(f"soft label key '{option}' is not an option")
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or value < 0
        ):
            raise LabelError(f"soft label for '{option}' must be a non-negative number")
        distribution[option] = float(value)
    total = sum(distribution.values())
    if abs(total - 1.0) > SOFT_SUM_TOLERANCE:
        raise LabelError(f"soft label probabilities sum to {total:.4f}, not 1")
    return Label(soft=distribution)
