"""Shared test helpers. Import explicitly: `from tests.helpers import ...`."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from clef_tune.config import Config

SCHEMA = {
    "questions": {
        "department": {
            "type": "choice",
            "criteria": {"billing": "Money", "technical": "Bugs", "NA": "Not applicable"},
        },
        "urgency": {"type": "score", "criteria": ["Low", "Medium", "High"]},
        "outage": {"type": "noul"},
    }
}


class CharCounter:
    """Token counter stand-in: one token per character."""

    def count(self, text: str) -> int:
        return len(text)

    def count_many(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def make_config(base_dir: Path, schema: dict | None = SCHEMA, **sections: Any) -> Config:
    data: dict[str, Any] = {"run_name": "test", **sections}
    if schema is not None:
        data["schema"] = schema
    return Config.model_validate(data).with_base_dir(base_dir)


def ticket_rows(n: int, start: int = 1) -> list[dict[str, str]]:
    """Labeled rows: departments alternate, urgency cycles 0-2, every 4th row is an outage."""
    departments = ["billing", "technical"]
    return [
        {
            "id": str(i),
            "subject": f"subject {i}",
            "body": f"body text {i}",
            "department": departments[i % 2],
            "urgency": str(i % 3),
            "outage": "yes" if i % 4 == 0 else "no",
        }
        for i in range(start, start + n)
    ]
