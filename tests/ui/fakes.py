"""Fake services for UI tests, the browser test and the screenshot script: no network, and
every call is recorded as (name, args, kwargs)."""

from __future__ import annotations

from typing import Any

from clef_tune.ui.services import Services
from tests.helpers import CharCounter


def fake_services(**overrides: Any) -> tuple[Services, list[tuple]]:
    calls: list[tuple] = []

    def recorded(name: str, function):
        def call(*args, **kwargs):
            calls.append((name, args, kwargs))
            return function(*args, **kwargs)

        return call

    answers: dict[str, Any] = {
        "counter": lambda config: CharCounter(),
        "prices": lambda: {},
    }
    answers.update(overrides)
    fields = {name: recorded(name, function) for name, function in answers.items()}
    return Services(**fields), calls
