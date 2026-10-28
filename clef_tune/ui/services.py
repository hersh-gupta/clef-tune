"""Everything the UI calls outside itself: the tokenizer, live prices, and (later) the Hub and
Jobs APIs. Handlers take a Services object, so tests, the browser test and the screenshot
script can replay recorded answers instead of calling the network."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from clef_tune.config import Config
from clef_tune.estimate import fetch_hf_prices
from clef_tune.prompt import TokenCounter, counter_for_config

PRICES_TTL_SECONDS = 3600
_prices: dict[str, object] = {"at": 0.0, "values": {}}


def cached_prices() -> dict[str, float]:
    """Live HF Jobs prices, fetched at most once an hour per process. An empty answer (offline)
    is not cached, so the next call tries again."""
    now = time.monotonic()
    values = _prices["values"]
    if values and now - float(_prices["at"]) < PRICES_TTL_SECONDS:
        return dict(values)  # type: ignore[arg-type]
    fetched = fetch_hf_prices()
    if fetched:
        _prices.update(at=now, values=fetched)
    return dict(fetched)


@dataclass(frozen=True)
class Services:
    counter: Callable[[Config], TokenCounter] = counter_for_config
    prices: Callable[[], dict[str, float]] = cached_prices
