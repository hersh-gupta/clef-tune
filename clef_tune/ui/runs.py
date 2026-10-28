"""Event handlers for the Launch step's HF Jobs tab: the launch review and confirmation, the
runs list, the run monitor, cancel and resume, and the results view. Every Hub call uses the
user's own token; nothing starts or resumes without a confirmation that shows its cost."""

from __future__ import annotations

import dataclasses
import json
import warnings
from dataclasses import dataclass
from typing import Any

from clef_tune.hub import HubError
from clef_tune.launch.hf_jobs import JobRequest, LaunchError
from clef_tune.ui import charts, fragments
from clef_tune.ui.services import Services
from clef_tune.ui.session import Session
from clef_tune.ui.settings import Settings, SignInRequired, user_token

LOG_TAIL = 200
POLL_SECONDS = 10
TERMINAL = ("COMPLETED", "CANCELED", "ERROR", "DELETED")
RESUMABLE = ("ERROR", "CANCELED")
# What a Hub or Jobs call can raise that the person can act on. HfHubHTTPError is an OSError.
_ERRORS = (SignInRequired, LaunchError, HubError, OSError, ValueError)


@dataclass
class PendingLaunch:
    plan: Any  # hf_jobs.LaunchPlan
    fingerprint: str  # the draft it was reviewed for


@dataclass
class LaunchReview:
    summary: str
    confirm_label: str
    can_confirm: bool
    timeout: str
    error: str | None


@dataclass
class ResultsView:
    table: str
    heatmaps: str
    calibration: str
    repo_link: str
    snippet: str


@dataclass
class RunView:
    badge: str
    progress: str
    stats: str
    loss_chart: str
    eval_chart: str
    logs: str
    can_cancel: bool
    can_resume: bool
    polling: bool
    results: ResultsView | None
    notice: str | None


def _money(value: float | None) -> str:
    return "—" if value is None else f"${value:,.2f}"


def _stage(info: Any) -> str:
    stage = info.status.stage
    return str(getattr(stage, "value", stage))


def _token(settings: Settings, oauth_token: str | None, pasted: str | None) -> str:
    return user_token(settings, oauth_token, pasted)


# ---------------------------------------------------------------------------- launching


def account(
    settings: Settings, services: Services, *, oauth_token: str | None, pasted: str | None
) -> str | None:
    """The signed-in user's name, or None."""
    try:
        return services.whoami(_token(settings, oauth_token, pasted))
    except Exception:  # not signed in, or the token cannot be read: the UI says sign in
        return None


def default_adapter_repo(session: Session, namespace: str) -> str:
    return f"{namespace}/clef-tune-{session.draft.run_name}"


def _fact(label: str, value: str) -> str:
    return f"<div><dt>{fragments.esc(label)}</dt><dd>{fragments.esc(value)}</dd></div>"


def _plan_summary(plan: Any, notes: list[str]) -> str:
    gpus = f"{plan.gpus} GPU" + ("s" if plan.gpus > 1 else "")
    estimate = plan.estimate
    if estimate is None:
        expected = "No estimate for this model"
    else:
        cost = (
            ""
            if estimate.cost_low is None
            else (f" · {_money(estimate.cost_low)}–{_money(estimate.cost_high)}")
        )
        expected = f"{estimate.hours_low:.1f}–{estimate.hours_high:.1f} h{cost}"
    ceiling = (
        "Unknown (no price for this hardware)"
        if plan.max_cost is None
        else f"{_money(plan.max_cost)}, if it runs until the time limit"
    )
    if not plan.kernels:
        kernels = "Off (slower training)"
    elif plan.kernels_cached:
        kernels = "Cached from an earlier job"
    else:
        kernels = "Built once (about 12 min), then cached"
    facts = [
        _fact("Hardware", f"{plan.flavor} ({gpus})"),
        _fact("Time limit", plan.timeout),
        _fact("Expected", expected),
        _fact("Most it can cost", ceiling),
        _fact("Adapter repo", f"{plan.adapter_repo} (private)"),
        _fact("Training data", "Kept" if plan.keep_data else "Deleted after a successful run"),
        _fact("Fast kernels", kernels),
        _fact("clef-tune code", f"{plan.source.repo_url} @ {plan.source.ref[:12]}"),
    ]
    callouts = "".join(fragments.callout("warning", note) for note in notes)
    return f'<div class="ct-review"><dl class="ct-facts">{"".join(facts)}</dl>{callouts}</div>'


def review_launch(
    session: Session,
    settings: Settings,
    services: Services,
    *,
    oauth_token: str | None,
    pasted: str | None,
    flavor: str | None,
    timeout: str,
    adapter_repo: str,
    delete_data: bool,
) -> LaunchReview:
    """Check everything a launch needs and show what it will cost. Uploads and starts nothing."""
    session.pending = None
    timeout = (timeout or "").strip()
    try:
        token = _token(settings, oauth_token, pasted)
        if session.draft is None:
            raise ValueError("Load data on the Data step first")
        config_path = session.write_config()
        try:
            counter = services.counter(session.config())
        except Exception:  # plan_launch retries and reports why it could not
            counter = None
        request = JobRequest(
            flavor=flavor or None,
            timeout=timeout or None,
            keep_data=not delete_data,
            adapter_repo=(adapter_repo or "").strip() or None,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plan = services.plan_launch(
                config_path, request, token=token, counter=counter, prices=services.prices()
            )
    except _ERRORS as exc:
        return LaunchReview("", "Launch", False, timeout, str(exc))
    session.pending = PendingLaunch(plan, session.draft.fingerprint())
    label = "Launch" if plan.max_cost is None else f"Launch for up to {_money(plan.max_cost)}"
    notes = [str(item.message) for item in caught]
    return LaunchReview(_plan_summary(plan, notes), label, True, plan.timeout, None)


def confirm_launch(
    session: Session,
    settings: Settings,
    services: Services,
    *,
    oauth_token: str | None,
    pasted: str | None,
) -> str:
    """Start the reviewed job. The review is used once, and only for the draft it reviewed."""
    pending, session.pending = session.pending, None
    if pending is None:
        return fragments.callout("info", "Nothing to launch; review the launch first.")
    if session.draft is None or pending.fingerprint != session.draft.fingerprint():
        return fragments.callout(
            "warning", "The settings changed after the review; review the launch again."
        )
    try:
        job = services.submit_launch(pending.plan, token=_token(settings, oauth_token, pasted))
    except _ERRORS as exc:
        return fragments.callout("error", f"The job did not start: {exc}")
    session.job_id, session.last_view = job.id, None
    return fragments.callout(
        "success",
        f"Job {job.id} started. Follow it below, under Your runs.",
        link=("Open it on Hugging Face", job.url),
    )


# ---------------------------------------------------------------------------- runs list


def list_runs(
    settings: Settings, services: Services, *, oauth_token: str | None, pasted: str | None
) -> tuple[list[list[str]], str | None]:
    """The user's clef-tune jobs, newest first, as rows: id, run, hardware, stage, created."""
    try:
        jobs = services.list_jobs(token=_token(settings, oauth_token, pasted))
    except _ERRORS as exc:
        return [], str(exc)
    rows = []
    for info in jobs:
        created = info.created_at
        rows.append(
            [
                info.id,
                (info.labels or {}).get("run", "—"),
                info.flavor or "—",
                _stage(info).capitalize(),
                f"{created:%Y-%m-%d %H:%M}" if created else "—",
            ]
        )
    return rows, None


def select_run(session: Session, job_id: str) -> None:
    session.job_id, session.last_view = job_id, None


# ---------------------------------------------------------------------------- monitor


def _latest(events: list[dict], event: str) -> dict | None:
    return next((e for e in reversed(events) if e.get("event") == event), None)


def _minutes(seconds: float) -> str:
    minutes = max(1, round(seconds / 60))
    if minutes < 60:
        return f"about {minutes} min left"
    return f"about {minutes / 60:.1f} h left"


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


def results_view(
    metrics: dict, adapter_repo: str | None, state_columns: list[str] | None
) -> ResultsView:
    baseline, final = metrics.get("baseline", {}), metrics.get("final", {})
    run = metrics.get("run", {})

    def change(key: str) -> str:
        before, after = baseline.get(key), final.get(key)
        if before is None or after is None:
            return "—"
        return f"{before:.2f} → {after:.2f}"

    tiles = [("Exact match", change("exact_match")), ("Mean macro-F1", change("mean_macro_f1"))]
    if run.get("hours") is not None:
        tiles.append(("Run time", f"{run['hours']:.1f} h"))
    if run.get("cost") is not None:
        tiles.append(("Cost", _money(run["cost"])))
    table = fragments.stat_tiles(tiles) + fragments.metrics_table(baseline, final)
    figures = []
    for qid, question in final.get("questions", {}).items():
        confusion = question.get("confusion") or {}
        heat = charts.heatmap(confusion.get("labels", []), confusion.get("matrix", []))
        figures.append(
            f'<figure class="ct-heatmap"><figcaption>{fragments.esc(qid)}</figcaption>'
            f"{heat}</figure>"
        )
    repo_link = ""
    if adapter_repo:
        url = f"https://huggingface.co/{adapter_repo}"
        repo_link = (
            f'<a class="ct-repo-link" href="{fragments.esc(url)}" target="_blank" '
            f'rel="noopener">{fragments.esc(adapter_repo)}</a>'
        )
    return ResultsView(
        table,
        f'<div class="ct-heatmaps">{"".join(figures)}</div>',
        fragments.calibration_table(metrics.get("calibration", {})),
        repo_link,
        predictor_snippet(adapter_repo or "path/to/adapter", state_columns),
    )


def run_view_from(
    *,
    stage: str,
    events: list[dict],
    hours: float | None,
    cost: float | None,
    logs: list[str],
    metrics: dict | None,
    adapter_repo: str | None,
    state_columns: list[str] | None,
) -> RunView:
    """The monitor for one run (an HF Job or a local run) from its metrics.jsonl events."""
    running = stage not in TERMINAL
    train = _latest(events, "train")
    if train is None:
        fraction, label = None, "Waiting for the first steps" if running else "No steps logged"
    elif train.get("total_steps"):
        fraction = train["step"] / train["total_steps"]
        label = f"Step {train['step']} of {train['total_steps']}"
    else:
        fraction, label = None, f"Step {train['step']}"
    if stage == "COMPLETED":
        fraction = 1.0
    eta = train.get("eta_s") if train else None
    speed = train.get("tokens_per_sec") if train else None
    stats = [
        ("Time", "—" if hours is None else f"{hours:.2f} h"),
        ("Cost so far", _money(cost)),
        ("Projected cost", _money(train.get("projected_cost") if train else None)),
        ("Speed", "—" if speed is None else f"{speed:,.0f} tokens/s"),
        ("Time left", _minutes(eta) if running and eta is not None else "—"),
    ]
    losses = [(e["step"], e["loss"]) for e in events if e.get("event") == "train"]
    evals = [e for e in events if e.get("event") == "eval"]
    baseline = _latest(events, "baseline")
    eval_series = {
        "mean macro-F1": [(e["step"], e["mean_macro_f1"]) for e in evals],
        "exact match": [(e["step"], e["exact_match"]) for e in evals],
    }
    reference = {"baseline macro-F1": baseline["mean_macro_f1"]} if baseline else None
    if logs:
        log_html = f'<pre class="ct-logs">{fragments.esc(chr(10).join(logs[-LOG_TAIL:]))}</pre>'
    else:
        log_html = fragments.empty_state("No logs yet", "Logs appear once the job starts.")
    return RunView(
        badge=fragments.stage_badge(stage),
        progress=fragments.progress(fraction, label),
        stats=fragments.stat_tiles(stats),
        loss_chart=charts.line_chart({"training loss": losses}),
        eval_chart=charts.line_chart(eval_series, reference=reference),
        logs=log_html,
        can_cancel=running,
        can_resume=stage in RESUMABLE,
        polling=running,
        results=results_view(metrics, adapter_repo, state_columns) if metrics else None,
        notice=None,
    )


def no_run_view() -> RunView:
    empty = fragments.empty_state(
        "No run selected", "Launch a run, or pick one under Your runs, to follow it here."
    )
    return RunView("", empty, "", "", "", "", False, False, False, None, None)


def _failed(session: Session, message: str, *, keep_polling: bool) -> RunView:
    notice = fragments.callout("warning", message)
    base = session.last_view if isinstance(session.last_view, RunView) else no_run_view()
    return dataclasses.replace(base, notice=notice, polling=keep_polling)


def _state_columns(session: Session) -> list[str] | None:
    return list(session.draft.state_columns) if session.draft else None


def job_view(
    session: Session,
    settings: Settings,
    services: Services,
    *,
    oauth_token: str | None,
    pasted: str | None,
    with_logs: bool,
) -> RunView:
    """The selected job's stage, progress, charts, logs and (once it completes) results. A
    failed poll keeps the last view and keeps polling."""
    if not session.job_id:
        return no_run_view()
    try:
        token = _token(settings, oauth_token, pasted)
        status = services.job_status(session.job_id, token=token)
        logs = list(services.job_logs(session.job_id, token=token)) if with_logs else []
    except SignInRequired as exc:
        return _failed(session, str(exc), keep_polling=False)
    except _ERRORS as exc:
        return _failed(
            session, f"Could not reach Hugging Face ({exc}); trying again.", keep_polling=True
        )
    metrics, notice = None, None
    if status.stage == "COMPLETED" and status.adapter_repo:
        metrics = session.results.get(status.id)
        if metrics is None:
            try:
                metrics = services.pull_metrics(status.adapter_repo, token=token)
                session.results[status.id] = metrics
            except _ERRORS as exc:
                notice = fragments.callout("warning", f"The results are not readable yet: {exc}")
    view = run_view_from(
        stage=status.stage,
        events=status.progress,
        hours=status.hours,
        cost=status.cost,
        logs=logs[-LOG_TAIL:],
        metrics=metrics,
        adapter_repo=status.adapter_repo,
        state_columns=_state_columns(session),
    )
    if not with_logs and isinstance(session.last_view, RunView):
        view.logs = session.last_view.logs
    if notice is not None:
        view.notice, view.polling = notice, True  # try the results again on the next poll
    elif status.stage == "ERROR" and status.message:
        view.notice = fragments.callout("error", f"The job failed: {status.message}")
    session.last_view = view
    return view


def cancel_run(
    session: Session,
    settings: Settings,
    services: Services,
    *,
    oauth_token: str | None,
    pasted: str | None,
) -> str:
    if not session.job_id:
        return fragments.callout("info", "Pick a run under Your runs first.")
    try:
        services.cancel_job(session.job_id, token=_token(settings, oauth_token, pasted))
    except _ERRORS as exc:
        return fragments.callout("error", f"Could not cancel {session.job_id}: {exc}")
    return fragments.callout(
        "info",
        "Cancel requested. The job can take up to a minute to stop; "
        "wait that long before resuming it. Its checkpoints are kept.",
    )


def review_resume(
    session: Session,
    settings: Settings,
    services: Services,
    *,
    oauth_token: str | None,
    pasted: str | None,
    timeout: str,
) -> LaunchReview:
    """What resuming the selected run costs at most. Starts nothing."""
    timeout = (timeout or "").strip()
    if not session.job_id:
        return LaunchReview("", "Resume", False, timeout, "Pick a run under Your runs first.")
    try:
        quote = services.quote_resume(
            session.job_id,
            timeout=timeout or None,
            token=_token(settings, oauth_token, pasted),
        )
    except _ERRORS as exc:
        return LaunchReview("", "Resume", False, timeout, str(exc))
    ceiling = (
        "Unknown (no price for this hardware)"
        if quote.max_cost is None
        else f"{_money(quote.max_cost)}, if it runs until the time limit"
    )
    facts = [
        _fact("Hardware", quote.flavor or "—"),
        _fact("Time limit", quote.timeout),
        _fact("Most it can cost", ceiling),
        _fact("Starts from", "Its newest complete checkpoint"),
    ]
    summary = f'<div class="ct-review"><dl class="ct-facts">{"".join(facts)}</dl></div>'
    label = "Resume" if quote.max_cost is None else f"Resume for up to {_money(quote.max_cost)}"
    return LaunchReview(summary, label, True, quote.timeout, None)


def resume_run(
    session: Session,
    settings: Settings,
    services: Services,
    *,
    oauth_token: str | None,
    pasted: str | None,
    timeout: str,
) -> str:
    if not session.job_id:
        return fragments.callout("info", "Pick a run under Your runs first.")
    try:
        job = services.resume_job(
            session.job_id,
            timeout=(timeout or "").strip() or None,
            token=_token(settings, oauth_token, pasted),
        )
    except _ERRORS as exc:
        return fragments.callout("error", f"Could not resume {session.job_id}: {exc}")
    session.job_id, session.last_view = job.id, None
    return fragments.callout(
        "success", f"Resumed as job {job.id}.", link=("Open it on Hugging Face", job.url)
    )
