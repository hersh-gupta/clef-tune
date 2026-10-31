"""Runs: reviewing and starting a training job, following any run (training or prediction),
stopping, resuming and cleaning it up, and what its results say. Plain data for the pages;
every Hub call uses the person's own token, and nothing that costs money starts without a
review that shows the most it can cost."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from clef_tune.estimate import HF_FLAVORS
from clef_tune.hub import HubError
from clef_tune.launch.hf_jobs import JobRequest, JobStatus, LaunchError, timeout_hours
from clef_tune.ui import wording as w
from clef_tune.ui.services import Services
from clef_tune.ui.session import Session

POLL_SECONDS = 5
TERMINAL = ("COMPLETED", "CANCELED", "ERROR", "DELETED")
RESUMABLE = ("ERROR", "CANCELED")
DOTS_MAX = 60
# What a Hub or Jobs call can raise that the person can act on. HfHubHTTPError is an OSError.
_ERRORS = (LaunchError, HubError, OSError, ValueError)
Status = Literal["running", "done", "stopped", "failed"]
StageState = Literal["done", "now", "todo", "stopped"]


def _status(stage: str) -> Status:
    if stage == "COMPLETED":
        return "done"
    if stage == "CANCELED":
        return "stopped"
    if stage in ("ERROR", "DELETED"):
        return "failed"
    return "running"


def hardware(flavor: str | None) -> str:
    """A flavor in words: "one L40S GPU"."""
    spec = HF_FLAVORS.get(flavor or "")
    if spec is None:
        return "a GPU"
    count = "one" if spec.count == 1 else str(spec.count)
    return f"{count} {spec.gpu.replace('-', ' ')} GPU{'s' if spec.count > 1 else ''}"


def cap_cost(price_per_hour: float | None, timeout: str | None) -> float | None:
    if price_per_hour is None or not timeout:
        return None
    try:
        return timeout_hours(timeout) * price_per_hour
    except LaunchError:
        return None


# ---------------------------------------------------------------------------- following a run


@dataclass
class Stage:
    label: str
    detail: str = ""
    state: StageState = "todo"


@dataclass
class RunState:
    id: str
    kind: str  # "train" or "predict"
    name: str
    status: Status
    stages: list[Stage]
    fraction: float  # how far along, 0-1
    time_left: str | None
    eta_seconds: float | None
    spent: str
    cap: str | None
    message: str | None
    can_stop: bool
    can_resume: bool
    loss: list[tuple[int, float]] = field(default_factory=list)
    adapter_repo: str | None = None
    url: str | None = None
    hours: float | None = None
    cost: float | None = None


def _latest(events: list[dict], event: str) -> dict | None:
    return next((e for e in reversed(events) if e.get("event") == event), None)


def _stages(labels: list[tuple[str, str, str, str]], reached: list[bool], running: bool):
    """Stages from (done, now, todo, detail) labels and which of them were reached: the first
    one not reached is where the run is now, or where it stopped."""
    first_todo = next((k for k, ok in enumerate(reached) if not ok), len(reached))
    stages = []
    for k, (done, now, todo, detail) in enumerate(labels):
        if k < first_todo:
            stages.append(Stage(done, detail, "done"))
        elif k == first_todo:
            stages.append(Stage(now, detail, "now" if running else "stopped"))
        else:
            stages.append(Stage(todo, "", "todo"))
    return stages


def _machine(flavor: str | None) -> str:
    return f"{hardware(flavor)[0].upper()}{hardware(flavor)[1:]} on Hugging Face"


def train_stages(
    stage: str, events: list[dict], model: str, noun: str, flavor: str | None
) -> list[Stage]:
    """The six training stages, from the job's stage and its metrics.jsonl events."""
    baseline, train = _latest(events, "baseline"), _latest(events, "train")
    total = (train or {}).get("total_steps")
    trained = bool(train and total and train["step"] >= total)
    tested = _latest(events, "done") is not None
    complete = stage == "COMPLETED"
    starting = ""
    if baseline and baseline.get("exact_match") is not None:
        share = round(baseline["exact_match"] * 100)
        starting = (
            f"Before training it matched every expected output on {share}% of the test "
            f"{w.plural(noun)}"
        )
    steps = ""
    if train:
        steps = f"Step {train['step']} of {total}" if total else f"Step {train['step']}"
    labels = [
        ("Started a machine", "Starting a machine", "Start a machine", _machine(flavor)),
        (f"Loaded {model}", f"Loading {model}", f"Load {model}", ""),
        (
            "Measured the starting point",
            "Measuring the starting point",
            "Measure the starting point",
            starting,
        ),
        (
            "Learned from your examples",
            "Learning from your examples",
            "Learn from your examples",
            steps,
        ),
        ("Tested the result", "Testing the result", "Test the result", ""),
        ("Saved to your account", "Saving to your account", "Save to your account", ""),
    ]
    reached = [
        stage != "SCHEDULING",
        bool(events) or complete,
        baseline is not None or complete,
        trained or tested or complete,
        tested or complete,
        complete,
    ]
    return _stages(labels, reached, _status(stage) == "running")


def predict_stages(stage: str, model: str, noun: str, flavor: str | None) -> list[Stage]:
    things = w.plural(noun)
    labels = [
        ("Started a machine", "Starting a machine", "Start a machine", _machine(flavor)),
        (
            f"Read your {things} with {model}",
            f"Reading your {things} with {model}",
            f"Read your {things}",
            "",
        ),
        ("Prepared your spreadsheet", "Preparing your spreadsheet", "Prepare your spreadsheet", ""),
    ]
    complete = stage == "COMPLETED"
    reached = [stage != "SCHEDULING", complete, complete]
    return _stages(labels, reached, _status(stage) == "running")


def _fraction(stages: list[Stage], events: list[dict]) -> float:
    done = sum(1 for s in stages if s.state == "done")
    if done == len(stages):
        return 1.0
    train = _latest(events, "train")
    if (
        train
        and train.get("total_steps")
        and any(s.state == "now" and s.label.startswith("Learning") for s in stages)
    ):
        return 0.15 + 0.75 * min(train["step"] / train["total_steps"], 1.0)
    return max(0.04, done / len(stages))


def _failure(status: JobStatus) -> str | None:
    if status.stage == "CANCELED":
        return "You stopped this run."
    if status.stage in ("ERROR", "DELETED"):
        reason = f" ({status.message})" if status.message else ""
        return (
            f"It stopped with an error{reason}. Its logs on Hugging Face say more; you can try "
            "again from where it stopped."
        )
    return None


def run_state(status: JobStatus, *, model: str = "the model", noun: str = "example") -> RunState:
    noun = w.noun_or_default(noun)
    events = status.progress or []
    if status.kind == "predict":
        stages = predict_stages(status.stage, model, noun, status.flavor)
    else:
        stages = train_stages(status.stage, events, model, noun, status.flavor)
    train = _latest(events, "train")
    running = _status(status.stage) == "running"
    eta = train.get("eta_s") if train and running else None
    cap = cap_cost(status.price_per_hour, status.timeout)
    return RunState(
        id=status.id,
        kind=status.kind,
        name=status.run or status.id,
        status=_status(status.stage),
        stages=stages,
        fraction=_fraction(stages, events),
        time_left=w.time_left(eta) if eta is not None else None,
        eta_seconds=eta,
        spent=w.money(status.cost) if status.cost is not None else "$0.00",
        cap=w.money(cap) if cap is not None else None,
        message=_failure(status),
        can_stop=running,
        can_resume=status.stage in RESUMABLE and status.kind == "train",
        loss=[(e["step"], e["loss"]) for e in events if e.get("event") == "train"],
        adapter_repo=status.adapter_repo,
        url=status.url,
        hours=status.hours,
        cost=status.cost,
    )


def job_state(
    session: Session, services: Services, job_id: str, *, token: str, model: str, noun: str
) -> tuple[RunState | None, str | None]:
    """A job's state, and a notice when it could not be read. A failed poll keeps the last
    state it read, so the page stays put while Hugging Face is briefly unreachable."""
    try:
        status = services.job_status(job_id, token=token)
    except _ERRORS as exc:
        last = session.last_states.get(job_id)
        return last, f"Could not reach Hugging Face ({exc}); trying again."
    state = run_state(status, model=model, noun=noun)
    session.last_states[job_id] = state
    return state, None


# ---------------------------------------------------------------------------- results


@dataclass
class QuestionResult:
    id: str
    ask: str
    kind: str
    n: int
    before_right: int
    after_right: int
    before_pct: int
    after_pct: int
    summary: w.Phrase
    names: list[str]  # answer names, in matrix order
    matrix: list[list[int]]  # rows: expected output; columns: the model's answer
    tech: str


@dataclass
class ResultsView:
    name: str
    noun: str
    n: int
    headline: w.Phrase
    dots: tuple[int, int, int] | None  # before, after, total (in dots)
    dots_caption: str
    questions: list[QuestionResult]
    run_line: str | None
    repo: str | None
    local_path: str | None
    snippet: str
    calibration_note: str | None
    state_columns: list[str] = field(default_factory=list)


def predictor_snippet(repo: str, state_columns: list[str] | None) -> str:
    if state_columns and len(state_columns) > 1:
        state = "{" + ", ".join(f'{json.dumps(column)}: "..."' for column in state_columns) + "}"
    else:
        state = '"..."'
    return (
        "from clef_tune import Predictor\n\n"
        f"predictor = Predictor.load({json.dumps(repo)})\n"
        f'answers = predictor.predict([{{"state": {state}}}])\n'
        "print(answers[0])\n"
    )


def _descriptions(definition: dict, labels: list[str]) -> list[str]:
    criteria = definition.get("criteria")
    if isinstance(criteria, list):
        return [
            str(criteria[int(label)]) if label.isdigit() and int(label) < len(criteria) else ""
            for label in labels
        ]
    if isinstance(criteria, dict):
        return [str(criteria.get(label) or "") for label in labels]
    return ["" for _ in labels]


def _tech(before: dict, after: dict, calibration: dict | None) -> str:
    parts = []
    if before.get("f1") is not None and after.get("f1") is not None:
        parts.append(f"Macro-F1 {before['f1']:.2f} → {after['f1']:.2f}")
    if before.get("log_loss") is not None and after.get("log_loss") is not None:
        parts.append(f"log-loss {before['log_loss']:.2f} → {after['log_loss']:.2f}")
    if before.get("mae") is not None and after.get("mae") is not None:
        parts.append(f"average error {before['mae']:.2f} → {after['mae']:.2f} steps")
    if calibration and calibration.get("fitted"):
        ece = (calibration.get("ece_before"), calibration.get("ece_after"))
        if None not in ece:
            parts.append(f"calibration error {ece[0]:.1%} → {ece[1]:.1%}")
    return " · ".join(parts)


def results_view(
    metrics: dict,
    config: dict,
    schema: dict | None,
    *,
    name: str,
    repo: str | None = None,
    local_path: str | None = None,
) -> ResultsView:
    data = config.get("data") or {}
    noun = w.noun_or_default(data.get("row_noun"))
    definitions = (schema or config.get("schema") or {}).get("questions") or {}
    baseline, final = metrics.get("baseline") or {}, metrics.get("final") or {}
    n = int(final.get("n_records") or baseline.get("n_records") or 0)
    calibration = metrics.get("calibration") or {}
    questions = []
    for qid, after in (final.get("questions") or {}).items():
        before = (baseline.get("questions") or {}).get(qid, {})
        definition = definitions.get(qid) or {}
        kind = after.get("question_type") or definition.get("type") or "choice"
        confusion = after.get("confusion") or {}
        labels = [str(label) for label in confusion.get("labels", [])]
        matrix = confusion.get("matrix", [])
        names = (
            [{"true": "yes", "false": "no"}.get(label, label) for label in labels]
            if kind == "noul"
            else w.answer_names(kind, labels, _descriptions(definition, labels))
        )
        count = int(after.get("n") or n)
        before_acc, after_acc = before.get("accuracy") or 0.0, after.get("accuracy") or 0.0
        questions.append(
            QuestionResult(
                id=qid,
                ask=str(definition.get("instructions") or qid),
                kind=kind,
                n=count,
                before_right=round(before_acc * count),
                after_right=round(after_acc * count),
                before_pct=round(before_acc * 100),
                after_pct=round(after_acc * 100),
                summary=w.mistake_summary(kind, labels if kind == "noul" else names, matrix, noun),
                names=names,
                matrix=matrix,
                tech=_tech(before, after, calibration.get(qid)),
            )
        )
    exact_before = round((baseline.get("exact_match") or 0.0) * n)
    exact_after = round((final.get("exact_match") or 0.0) * n)
    mean = lambda key: (  # noqa: E731
        sum(q.before_pct if key == "b" else q.after_pct for q in questions) / 100 / len(questions)
        if questions
        else 0.0
    )
    dots = None
    caption = w.dots_caption(len(questions), noun)
    if len(questions) <= 3 and n:
        unit = math.ceil(n / DOTS_MAX)
        dots = (round(exact_before / unit), round(exact_after / unit), math.ceil(n / unit))
        if unit > 1:
            caption = caption.replace(f"one test {noun}", f"{unit} test {w.plural(noun)}")
    run = metrics.get("run") or {}
    run_line = None
    if run.get("hours") is not None:
        run_line = f"Training took {w.duration(run['hours'])}"
        if run.get("cost") is not None:
            run_line += f" and cost {w.money(run['cost'])}"
            if run.get("estimated_cost") is not None:
                run_line += f", against an estimate of {w.money(run['estimated_cost'])}"
        run_line += "."
    note = None
    if calibration and not any(entry.get("fitted") for entry in calibration.values()):
        note = (
            "Its confidence wasn't calibrated: that needs at least 20 test "
            f"{w.plural(noun)} per question."
        )
    state_columns = list(data.get("state_columns") or [])
    return ResultsView(
        name=name,
        noun=noun,
        n=n,
        headline=w.headline(
            questions=len(questions),
            exact_before=exact_before,
            exact_after=exact_after,
            n=n,
            noun=noun,
            mean_before=mean("b"),
            mean_after=mean("a"),
        ),
        dots=dots,
        dots_caption=caption,
        questions=questions,
        run_line=run_line,
        repo=repo,
        local_path=local_path,
        snippet=predictor_snippet(repo or local_path or "path/to/adapter", state_columns),
        calibration_note=note,
        state_columns=state_columns,
    )


def job_results(
    session: Session, services: Services, status: JobStatus, *, token: str
) -> tuple[ResultsView | None, str | None]:
    """A completed training job's results (read once per session), or why they can't be read."""
    if status.stage != "COMPLETED" or not status.adapter_repo or status.kind != "train":
        return None, None
    files = session.results.get(status.id)
    if files is None:
        try:
            files = services.pull_run_files(status.adapter_repo, token=token)
        except _ERRORS as exc:
            return None, f"The results aren't readable yet ({exc}); trying again."
        session.results[status.id] = files
    return (
        results_view(
            files.metrics,
            files.config,
            files.schema,
            name=status.run or status.id,
            repo=status.adapter_repo,
        ),
        None,
    )


# ---------------------------------------------------------------------------- starting a run


@dataclass
class PendingLaunch:
    plan: Any  # hf_jobs.LaunchPlan
    fingerprint: str  # the draft it was reviewed for


@dataclass
class ReadyView:
    can_start: bool
    error: str | None
    model: str = ""
    preset: str = ""
    data_line: str = ""
    split_line: str = ""
    repo: str = ""
    time: str = ""
    cost: str = ""
    cap_cost: str | None = None
    cap_time: str = ""
    first_build: bool = False
    notes: list[str] = field(default_factory=list)


def review_launch(
    session: Session,
    services: Services,
    *,
    token: str,
    adapter_repo: str | None,
    delete_data: bool,
    model: str,
    preset: str,
    flavor: str | None = None,
) -> ReadyView:
    """Check everything a launch needs and what it will cost. Uploads and starts nothing."""
    session.pending = None
    try:
        if session.draft is None:
            raise ValueError("Load data on the first step first")
        config_path = session.write_config()
        try:
            counter = services.counter(session.config())
        except Exception:  # plan_launch retries and reports why it could not
            counter = None
        request = JobRequest(
            flavor=flavor, keep_data=not delete_data, adapter_repo=adapter_repo or None
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plan = services.plan_launch(
                config_path, request, token=token, counter=counter, prices=services.prices()
            )
    except _ERRORS as exc:
        return ReadyView(False, str(exc))
    session.pending = PendingLaunch(plan, session.draft.fingerprint())
    noun = w.noun_or_default(session.draft.row_noun)
    report = session.report
    rows = len(report.train) + len(report.eval) if report else 0
    questions = len(session.draft.questions)
    estimate = plan.estimate
    return ReadyView(
        can_start=True,
        error=None,
        model=model,
        preset=preset,
        data_line=f"{w.counted(rows, noun)} · {questions} question{'s' if questions != 1 else ''}",
        split_line=(
            f"{len(report.train):,} to learn from, {len(report.eval):,} set aside to test it"
            if report
            else ""
        ),
        repo=plan.adapter_repo,
        time=w.minutes_range(estimate.hours_low, estimate.hours_high) if estimate else "",
        cost=w.money_range(estimate.cost_low, estimate.cost_high) if estimate else "",
        cap_cost=w.money(plan.max_cost) if plan.max_cost is not None else None,
        cap_time=w.timeout_text(plan.timeout),
        first_build=bool(plan.kernels and not plan.kernels_cached),
        notes=[str(item.message) for item in caught],
    )


def confirm_launch(
    session: Session, services: Services, *, token: str
) -> tuple[str | None, str | None]:
    """Start the reviewed job; return its id, or why it did not start. A review is used once,
    and only for the draft it reviewed."""
    pending, session.pending = session.pending, None
    if pending is None:
        return None, "Nothing to start; check the details again."
    if session.draft is None or pending.fingerprint != session.draft.fingerprint():
        return None, "The settings changed after the check; check the details again."
    try:
        job = services.submit_launch(pending.plan, token=token)
    except _ERRORS as exc:
        return None, f"Training did not start: {exc}"
    session.job_id = job.id
    return job.id, None


# ---------------------------------------------------------------------------- your models


@dataclass
class ModelRow:
    id: str
    name: str
    kind: str
    status: Status
    label: str  # "Training", "Ready", "Predictions ready", ...
    created: datetime | None


_LABELS = {
    ("train", "running"): "Training",
    ("train", "done"): "Ready",
    ("train", "stopped"): "Stopped",
    ("train", "failed"): "Didn't finish",
    ("predict", "running"): "Predicting",
    ("predict", "done"): "Predictions ready",
    ("predict", "stopped"): "Stopped",
    ("predict", "failed"): "Didn't finish",
}


def _stage_of(info: Any) -> str:
    stage = info.status.stage
    return str(getattr(stage, "value", stage))


def list_models(services: Services, *, token: str) -> tuple[list[ModelRow], str | None]:
    """Every clef-tune run on the account, newest first."""
    try:
        jobs = services.list_jobs(token=token)
    except _ERRORS as exc:
        return [], f"Could not list your runs: {exc}"
    rows = []
    for info in jobs:
        labels = info.labels or {}
        kind = labels.get("kind") or "train"
        status = _status(_stage_of(info))
        rows.append(
            ModelRow(
                id=info.id,
                name=labels.get("run") or info.id,
                kind=kind,
                status=status,
                label=_LABELS[(kind, status)],
                created=getattr(info, "created_at", None),
            )
        )
    return rows, None


def model_detail(state: RunState, results: ResultsView | None) -> str:
    """One line for a run in Your models."""
    if state.status == "running":
        return state.time_left or "Getting started"
    if results is not None:
        return w.plain(results.headline)
    if state.status == "done" and state.kind == "predict":
        return f"Took {w.duration(state.hours or 0)} · {state.spent}"
    if state.status == "stopped":
        return f"Stopped after {w.duration(state.hours or 0)} · {state.spent}"
    if state.status == "failed":
        return state.message or "It stopped with an error."
    return ""


# ---------------------------------------------------------------------------- managing runs


def stop_run(services: Services, job_id: str, *, token: str) -> str | None:
    try:
        services.cancel_job(job_id, token=token)
    except _ERRORS as exc:
        return f"Could not stop it: {exc}"
    return None


@dataclass
class ResumeView:
    can_resume: bool
    error: str | None
    cap_cost: str | None = None
    cap_time: str = ""
    hardware: str = ""


def review_resume(services: Services, job_id: str, *, token: str) -> ResumeView:
    try:
        quote = services.quote_resume(job_id, token=token)
    except _ERRORS as exc:
        return ResumeView(False, str(exc))
    return ResumeView(
        True,
        None,
        w.money(quote.max_cost) if quote.max_cost is not None else None,
        w.timeout_text(quote.timeout),
        hardware(quote.flavor),
    )


def resume_run(services: Services, job_id: str, *, token: str) -> tuple[str | None, str | None]:
    try:
        job = services.resume_job(job_id, token=token)
    except _ERRORS as exc:
        return None, f"Could not try again: {exc}"
    return job.id, None


def delete_files(services: Services, job_id: str, *, token: str) -> tuple[int, str | None]:
    try:
        return len(services.delete_job_files(job_id, token=token)), None
    except _ERRORS as exc:
        return 0, f"Could not delete the files: {exc}"
