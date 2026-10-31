"""Everything the UI calls outside itself: the tokenizer, live prices, and the Hub and Jobs
APIs. Handlers take a Services object, so tests, the browser test and the screenshot script
can replay recorded answers instead of calling the network. Every Hub call gets the user's
token explicitly (ui.settings.user_token), never None."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from huggingface_hub import HfApi

from clef_tune import hub
from clef_tune.config import Config
from clef_tune.estimate import fetch_hf_prices
from clef_tune.launch import hf_jobs
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


def hub_namespace(token: str) -> str:
    """The user name a token belongs to."""
    return HfApi().whoami(token=token)["name"]


@dataclass(frozen=True)
class Services:
    counter: Callable[[Config], TokenCounter] = counter_for_config
    prices: Callable[[], dict[str, float]] = cached_prices
    whoami: Callable[[str], str] = hub_namespace
    plan_launch: Callable[..., Any] = hf_jobs.plan_launch
    submit_launch: Callable[..., Any] = hf_jobs.submit_launch
    list_jobs: Callable[..., list[Any]] = hf_jobs.list_jobs
    job_status: Callable[..., hf_jobs.JobStatus] = hf_jobs.job_status
    job_logs: Callable[..., Iterable[str]] = hf_jobs.job_logs
    cancel_job: Callable[..., None] = hf_jobs.cancel_job
    quote_resume: Callable[..., hf_jobs.ResumeQuote] = hf_jobs.quote_resume
    resume_job: Callable[..., Any] = hf_jobs.resume_job
    pull_metrics: Callable[..., dict] = hub.pull_metrics
    pull_run_files: Callable[..., hub.RunFiles] = hub.pull_run_files
    plan_prediction: Callable[..., hf_jobs.PredictionPlan] = hf_jobs.plan_prediction
    submit_prediction: Callable[..., Any] = hf_jobs.submit_prediction
    download_predictions: Callable[..., Any] = hf_jobs.download_predictions
    delete_job_files: Callable[..., list[str]] = hf_jobs.delete_job_files
