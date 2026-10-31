"""Fake services for UI tests, the browser test and the screenshot script: no network, and
every call is recorded as (name, args, kwargs). The Jobs answers replay the recorded run in
tests/ui/fixtures/recorded_run/."""

from __future__ import annotations

import json
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from clef_tune.config import load_config
from clef_tune.hub import RunFiles
from clef_tune.launch.hf_jobs import JobStatus, ResumeQuote
from clef_tune.ui.services import Services
from clef_tune.yamlio import load_yaml_file
from tests.helpers import CharCounter

RECORDED_RUN = Path(__file__).parent / "fixtures" / "recorded_run"
EXAMPLE_CONFIG = Path(__file__).resolve().parents[2] / "examples" / "support_tickets.yaml"
NOW = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)
SHA = "0123456789abcdef0123456789abcdef01234567"
RUNNING_UNTIL_STEP = 70
FAILED_AT_STEP = 50
JOBS = {  # id -> (run, stage, minutes ago it was created)
    "job-a": ("run-a", "RUNNING", 30),
    "job-p": ("run-b-predict", "COMPLETED", 60),
    "job-b": ("run-b", "COMPLETED", 120),
    "job-c": ("run-c", "ERROR", 300),
}
PREDICTIONS = (  # what the fake prediction job wrote
    "ticket_id,subject,body,customer_tier,department,department_confidence,urgency,"
    "urgency_confidence,outage,outage_confidence,review\n"
    "T2041,Refund for duplicate charge,Charged twice.,pro,billing,0.97,1,0.81,no,0.99,\n"
    "T2042,Dashboard blank after update,Blank since 9am.,enterprise,technical,0.94,2,0.72,"
    "yes,0.64,yes\n"
    "T2043,Add a teammate,How do I invite?,free,account,0.91,0,0.88,no,0.98,\n"
)


def kind_of(job_id: str) -> str:
    return "predict" if JOBS[job_id][0].endswith("-predict") else "train"


def recorded_events() -> list[dict]:
    lines = (RECORDED_RUN / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def recorded_metrics() -> dict:
    return json.loads((RECORDED_RUN / "metrics.json").read_text(encoding="utf-8"))


def _events_until(step: int) -> list[dict]:
    """The log as it stood just after the train event for `step` (its eval not yet run)."""
    events = []
    for event in recorded_events():
        if event["event"] != "baseline" and event.get("step", 0) > step:
            break
        if event["event"] == "eval" and event["step"] == step:
            break
        events.append(event)
    return events


def job_info(job_id: str) -> SimpleNamespace:
    run, stage, minutes = JOBS[job_id]
    return SimpleNamespace(
        id=job_id,
        url=f"https://huggingface.co/jobs/me/{job_id}",
        status=SimpleNamespace(stage=stage, message=None),
        labels={"tool": "clef-tune", "run": run, "timeout": "1h"}
        | ({"kind": "predict"} if kind_of(job_id) == "predict" else {}),
        flavor="l4x1",
        created_at=NOW - timedelta(minutes=minutes),
    )


def job_status(job_id: str) -> JobStatus:
    run, stage, minutes = JOBS.get(job_id, ("run-new", "RUNNING", 0))
    if job_id in JOBS and kind_of(job_id) == "predict":
        return JobStatus(
            id=job_id,
            url=f"https://huggingface.co/jobs/me/{job_id}",
            run=run,
            flavor="l40sx1",
            stage=stage,
            message=None,
            hours=0.18,
            cost=0.33,
            progress=[],
            adapter_repo="me/clef-tune-run-b",
            kind="predict",
            price_per_hour=1.8,
            timeout="30m",
            started_at=NOW - timedelta(minutes=minutes),
            created_at=NOW - timedelta(minutes=minutes),
        )
    if stage == "COMPLETED":
        events, hours = recorded_events(), 0.42
    elif stage == "ERROR":
        events, hours = _events_until(FAILED_AT_STEP), 0.2
    else:
        events, hours = _events_until(RUNNING_UNTIL_STEP), 0.25
    return JobStatus(
        id=job_id,
        url=f"https://huggingface.co/jobs/me/{job_id}",
        run=run,
        flavor="l4x1",
        stage=stage,
        message="exit code 1" if stage == "ERROR" else None,
        hours=hours,
        cost=hours * 0.8,
        progress=events,
        adapter_repo=f"me/clef-tune-{run}",
        price_per_hour=0.8,
        timeout="1h",
        started_at=NOW - timedelta(minutes=minutes),
        created_at=NOW - timedelta(minutes=minutes),
    )


def fake_plan(config_path, request, plan_warning=None, plan_error=None) -> SimpleNamespace:
    if plan_error is not None:
        raise plan_error
    if plan_warning:
        warnings.warn(plan_warning, stacklevel=2)
    config = load_config(config_path)
    timeout = request.timeout or "1h"
    return SimpleNamespace(
        config=config,
        source=SimpleNamespace(repo_url="https://github.com/hersh-gupta/clef-tune", ref=SHA),
        namespace="me",
        adapter_repo=request.adapter_repo or f"me/clef-tune-{config.run_name}",
        flavor=request.flavor or "l4x1",
        gpus=1,
        timeout=timeout,
        price_per_hour=0.8,
        estimate=SimpleNamespace(hours_low=0.35, hours_high=0.75, cost_low=0.28, cost_high=0.6),
        max_cost=0.8,
        kernels=True,
        kernels_cached=False,
        keep_data=request.keep_data,
    )


def fake_services(**overrides: Any) -> tuple[Services, list[tuple]]:
    calls: list[tuple] = []
    plan_warning = overrides.pop("plan_warning", None)
    plan_error = overrides.pop("plan_error", None)

    def recorded(name: str, function):
        def call(*args, **kwargs):
            calls.append((name, args, kwargs))
            return function(*args, **kwargs)

        return call

    answers: dict[str, Any] = {
        "counter": lambda config: CharCounter(),
        "prices": lambda: {},
        "whoami": lambda token: "me",
        "plan_launch": lambda config_path, request, **k: fake_plan(
            config_path, request, plan_warning, plan_error
        ),
        "submit_launch": lambda plan, **k: SimpleNamespace(
            id="job-new", url="https://huggingface.co/jobs/me/job-new", plan=plan
        ),
        "list_jobs": lambda **k: [job_info(job_id) for job_id in JOBS],
        "job_status": lambda job_id, **k: job_status(job_id),
        "job_logs": lambda job_id, **k: ["line 1", "<b>line 2</b>"],
        "cancel_job": lambda job_id, **k: None,
        "quote_resume": lambda job_id, timeout=None, **k: ResumeQuote(
            job_id, "l4x1", timeout or "2h", 0.8
        ),
        "resume_job": lambda job_id, **k: SimpleNamespace(
            id="job-resumed", url="https://huggingface.co/jobs/me/job-resumed"
        ),
        "pull_metrics": lambda repo_id, **k: recorded_metrics(),
        "pull_run_files": lambda repo_id, **k: RunFiles(
            recorded_metrics(), load_yaml_file(EXAMPLE_CONFIG), None
        ),
        "plan_prediction": lambda adapter_repo, input_csv, **k: SimpleNamespace(
            adapter_repo=adapter_repo,
            input_csv=Path(input_csv),
            rows=k.get("rows", 0),
            review_below=k.get("review_below", 0.7),
            flavor="l40sx1",
            timeout="30m",
            price_per_hour=1.8,
            max_cost=0.9,
            estimate=SimpleNamespace(
                hours_low=0.17, hours_high=0.25, cost_low=0.3, cost_high=0.45, hours=0.2
            ),
            kernels_cached=True,
        ),
        "submit_prediction": lambda plan, **k: SimpleNamespace(
            id="job-pnew", url="https://huggingface.co/jobs/me/job-pnew", plan=plan
        ),
        "download_predictions": lambda job_id, out, **k: _write(Path(out), PREDICTIONS),
        "delete_job_files": lambda job_id, **k: ["clef-tune-runs/x/input/a.csv"],
    }
    answers.update(overrides)
    fields = {name: recorded(name, function) for name, function in answers.items()}
    return Services(**fields), calls


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
