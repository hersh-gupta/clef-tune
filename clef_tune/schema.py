"""Clef's typed questions (noul / choice / score) and the schema that groups them."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clef_tune.yamlio import dump_yaml, load_yaml_file

QUESTION_TYPES = ("noul", "choice", "score")
NOUL_DEFAULT_CRITERIA = {
    "true": "The proposition is true or the answer is yes.",
    "false": "The proposition is false or the answer is no.",
}


class SchemaError(ValueError):
    """A schema or question definition is invalid."""


def option_key(key: Any) -> str:
    """Normalize a criteria key the way the user meant it (YAML/JSON may give bools or ints)."""
    if key is True:
        return "true"
    if key is False:
        return "false"
    return str(key)


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    instructions: str | None = None
    criteria: Any = None  # dict[str, Any] for choice/noul, list[Any] for score

    @classmethod
    def from_dict(cls, question_id: Any, data: Any) -> Question:
        qid = str(question_id).strip()
        if not qid:
            raise SchemaError("question IDs must be non-empty")
        if not isinstance(data, dict):
            raise SchemaError(f"{qid}: question must be a mapping with a 'type'")
        unknown = set(data) - {"type", "instructions", "criteria"}
        if unknown:
            raise SchemaError(f"{qid}: unknown keys {sorted(unknown)}")
        qtype = data.get("type")
        if qtype not in QUESTION_TYPES:
            raise SchemaError(f"{qid}: type must be one of noul, choice, score (got {qtype!r})")
        instructions = data.get("instructions")
        if instructions is not None and not isinstance(instructions, str):
            raise SchemaError(f"{qid}: instructions must be text")
        raw = data.get("criteria")
        if qtype == "choice":
            if not isinstance(raw, dict) or not raw:
                raise SchemaError(f"{qid}: choice questions need a non-empty criteria mapping")
            criteria: Any = {option_key(k): v for k, v in raw.items()}
            if len(criteria) != len(raw):
                raise SchemaError(f"{qid}: criteria keys collide after normalization")
        elif qtype == "score":
            if not isinstance(raw, list) or not raw:
                raise SchemaError(f"{qid}: score questions need a non-empty criteria list")
            criteria = list(raw)
        elif raw is None:
            criteria = None
        elif not isinstance(raw, dict):
            raise SchemaError(f"{qid}: noul criteria must map true/false to descriptions")
        else:
            criteria = {option_key(k): v for k, v in raw.items()}
            bad = set(criteria) - {"true", "false"}
            if bad:
                raise SchemaError(
                    f"{qid}: noul criteria keys must be true/false (got {sorted(bad)})"
                )
        return cls(id=qid, type=qtype, instructions=instructions or None, criteria=criteria)

    def options(self) -> list[tuple[str, Any]]:
        """Allowed (option_id, description) pairs in Clef's order (mirrors question_options)."""
        if self.type == "noul":
            criteria = dict(NOUL_DEFAULT_CRITERIA)
            criteria.update(self.criteria or {})
            return [(key, criteria[key]) for key in ("true", "false")]
        if self.type == "choice":
            return sorted((str(key), value) for key, value in self.criteria.items())
        return [(str(index), value) for index, value in enumerate(self.criteria)]

    def option_ids(self) -> list[str]:
        return [option_id for option_id, _ in self.options()]

    def to_clef(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type}
        if self.instructions:
            out["instructions"] = self.instructions
        if self.criteria is not None:
            out["criteria"] = (
                dict(self.criteria) if isinstance(self.criteria, dict) else list(self.criteria)
            )
        return out


@dataclass(frozen=True)
class Schema:
    questions: dict[str, Question]  # insertion order is the prompt order

    @classmethod
    def from_dict(cls, data: Any) -> Schema:
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("questions"), dict)
            or not data["questions"]
        ):
            raise SchemaError("schema must have a non-empty 'questions' mapping")
        questions: dict[str, Question] = {}
        for question_id, question_data in data["questions"].items():
            question = Question.from_dict(question_id, question_data)
            if question.id in questions:
                raise SchemaError(f"duplicate question ID {question.id!r}")
            questions[question.id] = question
        return cls(questions)

    @classmethod
    def load(cls, path: str | Path) -> Schema:
        return cls.from_dict(load_yaml_file(path))

    def to_dict(self) -> dict[str, Any]:
        return {"questions": {qid: q.to_clef() for qid, q in self.questions.items()}}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(dump_yaml(self.to_dict()), encoding="utf-8")
