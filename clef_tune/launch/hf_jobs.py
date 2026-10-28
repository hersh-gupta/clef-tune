"""Train on Hugging Face Jobs: plan a launch (validate, pin the source, estimate, pick hardware)
before anything is uploaded or started, then stage the bundle and start the job; list, watch,
cancel and resume jobs. Torch-free."""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi, JobInfo, Volume, constants, get_token

from clef_tune.bundle import write_bundle
from clef_tune.config import Config, load_config
from clef_tune.estimate import (
    HF_FLAVORS,
    EstimateRow,
    estimate_hf_flavors,
    fetch_hf_prices,
    model_spec,
    recommend,
    suggest_timeout,
    workload_from,
)
from clef_tune.hub import HubError, check_write_access
from clef_tune.launch.job_script import (
    JOB_OUTPUT_DIR,
    KERNEL_MOUNT,
    MODEL_MOUNT,
    WORK_MOUNT,
    job_command,
    job_env,
)
from clef_tune.launch.source import Source, SourceError, pin_commit, resolve_source
from clef_tune.prompt import TokenCounter, counter_for_config
from clef_tune.runtime import BASE_IMAGE, KERNEL_PINS, PRICE_ENV
from clef_tune.validation import ValidationReport, validate
from clef_tune.yamlio import load_yaml_file

TOOL_LABEL = {"tool": "clef-tune"}
LABEL_KEYS = ("tool", "run", "timeout")
RUNS_FOLDER = "clef-tune-runs"  # one folder per launch in the jobs-artifacts bucket
KERNEL_CACHE_ROOT = "clef-tune-kernels"
KERNEL_BUILD_MINUTES = 15  # causal-conv1d compiles in about 12 minutes on 8 vCPUs
BILLING_URL = "https://huggingface.co/settings/billing"
SCOPE_ADVICE = 'use a "Write" token, or a fine-grained token with job.write and repo.write'
METRICS_LOG = "metrics.jsonl"  # what clef_tune.train writes into the output dir
RESUMABLE = ("ERROR", "CANCELED")
CONTINUING = ("SCHEDULING", "RUNNING", "COMPLETED")  # a job in these stages owns its run's folder

_TIMEOUT = re.compile(r"^(\d+(?:\.\d+)?)([smhd])$")
_HOURS_PER_UNIT = {"s": 1 / 3600, "m": 1 / 60, "h": 1, "d": 24}


class LaunchError(RuntimeError):
    """A job cannot be launched or managed; the message says what to do."""


@dataclass(frozen=True)
class JobRequest:
    flavor: str | None = None
    timeout: str | None = None
    keep_data: bool = False
    kernels: bool = True
    adapter_repo: str | None = None
    source_ref: str | None = None


@dataclass(frozen=True)
class LaunchPlan:
    config: Config
    report: ValidationReport
    source: Source
    namespace: str
    adapter_repo: str
    flavor: str
    gpus: int
    timeout: str
    price_per_hour: float | None
    estimate: EstimateRow | None
    kernels: bool
    kernels_cached: bool
    keep_data: bool

    @property
    def estimated_hours(self) -> float | None:
        return self.estimate.hours if self.estimate is not None else None

    @property
    def max_cost(self) -> float | None:
        """What the job costs if it runs until its timeout."""
        if self.price_per_hour is None:
            return None
        return timeout_hours(self.timeout) * self.price_per_hour


def jobs_bucket(namespace: str) -> str:
    return f"{namespace}/{constants.HF_JOBS_ARTIFACTS_BUCKET_NAME}"


def kernel_cache_folder() -> str:
    """Where the causal-conv1d wheel is cached; it is only valid for one image and version."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", BASE_IMAGE)
    return f"{KERNEL_CACHE_ROOT}/{slug}/causal-conv1d-{KERNEL_PINS['causal-conv1d']}"


def timeout_hours(timeout: str) -> float:
    match = _TIMEOUT.match(timeout)
    if match is None:
        raise LaunchError(
            f"invalid timeout {timeout!r}: use a number and a unit (s, m, h or d), like 90m or 2h"
        )
    return float(match.group(1)) * _HOURS_PER_UNIT[match.group(2)]


def _kernel_wheel_cached(api: HfApi, namespace: str, token: str) -> bool:
    try:
        files = api.list_bucket_tree(
            jobs_bucket(namespace), kernel_cache_folder(), recursive=True, token=token
        )
        return any(f.path.endswith(".whl") for f in files)
    except Exception:  # no bucket yet, or it cannot be read: the job builds the wheel
        return False


def plan_launch(
    config_path: str | Path,
    request: JobRequest = JobRequest(),
    *,
    api: HfApi | None = None,
    token: str | None = None,
    counter: TokenCounter | None = None,
    prices: dict[str, float] | None = None,
    source: Source | None = None,
) -> LaunchPlan:
    """Everything a launch needs, checked in order of cost: local validation first, then the
    source, the token and the hardware. Uploads nothing and starts nothing."""
    try:
        config = load_config(config_path)
    except (ValueError, OSError) as exc:
        raise LaunchError(f"invalid config: {exc}") from exc
    if counter is None:
        try:
            counter = counter_for_config(config)
        except Exception:  # offline or no tokenizer: validate without token statistics
            counter = None
    report = validate(config, counter)
    if not report.ok:
        raise LaunchError("validation failed:\n" + "\n".join(str(i) for i in report.errors))

    try:
        source = source or pin_commit(resolve_source(request.source_ref))
    except SourceError as exc:
        raise LaunchError(str(exc)) from exc

    api = api or HfApi()
    token = token or get_token()
    if token is None:
        raise LaunchError("no Hugging Face token: run `hf auth login` or set HF_TOKEN")
    try:
        namespace = api.whoami(token=token)["name"]
    except Exception as exc:
        raise LaunchError(f"cannot check the Hugging Face token: {exc}") from exc
    adapter_repo = (
        request.adapter_repo
        or config.output.push_to_hub
        or f"{namespace}/clef-tune-{config.run_name}"
    )
    try:
        check_write_access(adapter_repo, token=token, api=api)
    except HubError as exc:
        raise LaunchError(str(exc)) from exc

    cached = request.kernels and _kernel_wheel_cached(api, namespace, token)

    flavor = request.flavor
    if flavor is not None and flavor not in HF_FLAVORS:
        raise LaunchError(f"unknown flavor {flavor!r}; choose one of {', '.join(HF_FLAVORS)}")
    live = prices if prices is not None else fetch_hf_prices()
    row = None
    spec = model_spec(config.model.id)
    if spec is not None and report.tokens is not None:
        rows = estimate_hf_flavors(spec, workload_from(config, report.tokens), live)
        row = next(r for r in rows if r.flavor == flavor) if flavor else recommend(rows)
        if row is None:
            raise LaunchError("no HF Jobs flavor fits this config; see `clef-tune estimate`")
        if not row.fits:
            raise LaunchError(f"{flavor} cannot run this config: {row.note}")
        flavor = row.flavor
    elif flavor is None or request.timeout is None:
        raise LaunchError(
            f"cannot estimate {config.model.id} for this config; pass --flavor and --timeout"
        )

    if request.timeout is not None:
        timeout_hours(request.timeout)
        timeout = request.timeout
    else:
        build = KERNEL_BUILD_MINUTES if request.kernels and not cached else 0
        timeout = suggest_timeout(row, extra_minutes=build)

    return LaunchPlan(
        config=config,
        report=report,
        source=source,
        namespace=namespace,
        adapter_repo=adapter_repo,
        flavor=flavor,
        gpus=HF_FLAVORS[flavor].count,
        timeout=timeout,
        price_per_hour=live.get(flavor, HF_FLAVORS[flavor].price_per_hour),
        estimate=row,
        kernels=request.kernels,
        kernels_cached=cached,
        keep_data=request.keep_data,
    )


@dataclass(frozen=True)
class LaunchedJob:
    id: str
    url: str
    work_folder: str  # the run's folder in the jobs-artifacts bucket, mounted at /work
    plan: LaunchPlan


def run_folder(run_name: str, now: datetime) -> str:
    """A folder per launch, so launching a run_name again never resumes from or collides with
    an earlier job's checkpoints."""
    return f"{RUNS_FOLDER}/{run_name}-{now:%Y%m%d-%H%M%S}"


def stage_bundle(plan: LaunchPlan, staging_dir: Path) -> Path:
    """Write the bundle with its config pointed at the job's mounts. The job passes
    --push-to-hub itself, so the config's own target is dropped."""
    config_path = write_bundle(plan.report, plan.config, staging_dir / "bundle")
    data = load_yaml_file(config_path)
    data.setdefault("model", {})["local_path"] = MODEL_MOUNT
    output = data.setdefault("output", {})
    output["dir"] = JOB_OUTPUT_DIR
    output.pop("push_to_hub", None)
    Config.model_validate(data).save(config_path)
    return config_path


def _status_code(error: Exception) -> int | None:
    return getattr(getattr(error, "response", None), "status_code", None)


def _scope_advice(error: Exception) -> str:
    """What to do about a 401 or 403: the usual cause is a token without a needed permission."""
    if _status_code(error) in (401, 403):
        return f"; the token may lack a permission: {SCOPE_ADVICE}"
    return ""


def start_job(api: HfApi, **run_job_kwargs) -> JobInfo:
    try:
        return api.run_job(**run_job_kwargs)
    except Exception as error:
        if _status_code(error) == 402:
            raise LaunchError(
                f"Hugging Face Jobs needs pre-paid credits; add them at {BILLING_URL}"
            ) from error
        raise LaunchError(f"could not start the job: {error}{_scope_advice(error)}") from error


def _remove_folder(api: HfApi, bucket: str, folder: str, token: str) -> None:
    try:
        files = api.list_bucket_tree(bucket, folder, recursive=True, token=token)
        paths = [f.path for f in files if f.type == "file"]
        if paths:
            api.batch_bucket_files(bucket, delete=paths, token=token)
    except Exception:  # best effort: the launch error is what the person needs to see
        pass


def submit_launch(
    plan: LaunchPlan,
    *,
    api: HfApi | None = None,
    token: str | None = None,
    now: datetime | None = None,
) -> LaunchedJob:
    """Upload the bundle to a fresh folder in the private jobs-artifacts bucket and start the
    job. A failed upload or a rejected job removes what was uploaded."""
    api = api or HfApi()
    token = token or get_token()
    if token is None:
        raise LaunchError("no Hugging Face token: run `hf auth login` or set HF_TOKEN")
    config = plan.config
    folder = run_folder(config.run_name, now or datetime.now(UTC))
    bucket = jobs_bucket(plan.namespace)
    sync = {"read_only": False, "namespace": plan.namespace, "token": token}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                stage_bundle(plan, Path(tmp) / "stage")
                work = api.sync_job_volume(
                    Path(tmp) / "stage", WORK_MOUNT, remote_name=folder, **sync
                )
                volumes = [
                    work,
                    Volume(
                        type="model",
                        source=config.model.id,
                        mount_path=MODEL_MOUNT,
                        revision=config.model.resolved_revision(),
                    ),
                ]
                if plan.kernels:
                    # an empty folder uploads a .keep, and syncing never deletes, so a cached
                    # wheel survives
                    empty = Path(tmp) / "kernels"
                    empty.mkdir()
                    volumes.append(
                        api.sync_job_volume(
                            empty, KERNEL_MOUNT, remote_name=kernel_cache_folder(), **sync
                        )
                    )
            except Exception as exc:
                raise LaunchError(
                    f"could not upload the bundle to {bucket}: {exc}{_scope_advice(exc)}"
                ) from exc
        info = start_job(
            api,
            image=BASE_IMAGE,
            command=job_command(
                plan.source,
                gpus=plan.gpus,
                adapter_repo=plan.adapter_repo,
                keep_data=plan.keep_data,
                kernels=plan.kernels,
            ),
            env=job_env(
                plan.source,
                price_per_hour=plan.price_per_hour,
                estimated_hours=plan.estimated_hours,
            ),
            secrets={"HF_TOKEN": token},
            flavor=plan.flavor,
            timeout=plan.timeout,
            labels={**TOOL_LABEL, "run": config.run_name, "timeout": plan.timeout},
            volumes=volumes,
            token=token,
        )
    except LaunchError:
        _remove_folder(api, bucket, folder, token)
        raise
    return LaunchedJob(id=info.id, url=info.url, work_folder=folder, plan=plan)


@dataclass(frozen=True)
class JobStatus:
    id: str
    url: str
    run: str | None
    flavor: str | None
    stage: str  # COMPLETED, CANCELED, ERROR, DELETED, SCHEDULING or RUNNING
    message: str | None
    hours: float | None
    cost: float | None
    progress: list[dict]  # the metrics.jsonl records so far

    def latest(self, event: str) -> dict | None:
        return next((r for r in reversed(self.progress) if r.get("event") == event), None)


def _stage(info: JobInfo) -> str:
    stage = info.status.stage
    return str(getattr(stage, "value", stage))  # the real JobStage is an enum


def _price(info: JobInfo) -> float | None:
    try:
        return float((info.environment or {})[PRICE_ENV])
    except (KeyError, ValueError):
        return None


def list_jobs(*, api: HfApi | None = None, token: str | None = None) -> list[JobInfo]:
    """clef-tune jobs, newest first."""
    api = api or HfApi()
    jobs = api.list_jobs(labels=dict(TOOL_LABEL), token=token)
    oldest = datetime.min.replace(tzinfo=UTC)
    return sorted(jobs, key=lambda info: info.created_at or oldest, reverse=True)


def _work_volume(info: JobInfo) -> Volume | None:
    """The bucket folder mounted at /work: the run's bundle, output and checkpoints."""
    return next(
        (v for v in info.volumes or [] if v.type == "bucket" and v.mount_path == WORK_MOUNT), None
    )


def job_progress(
    info: JobInfo, *, api: HfApi | None = None, token: str | None = None
) -> list[dict]:
    """The metrics.jsonl records the job's trainer has written to its work folder so far."""
    api = api or HfApi()
    work = _work_volume(info)
    if work is None:
        return []
    output = JOB_OUTPUT_DIR.removeprefix(WORK_MOUNT + "/")
    remote = "/".join(part for part in (work.path, output, METRICS_LOG) if part)
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / METRICS_LOG
        try:
            api.download_bucket_files(
                work.source, [(remote, local)], raise_on_missing_files=False, token=token
            )
        except Exception:  # progress is best effort; the job's stage still shows
            return []
        if not local.exists():
            return []
        records = []
        for line in local.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:  # a line still being written
                continue
            if isinstance(record, dict):
                records.append(record)
        return records


def job_status(
    job_id: str,
    *,
    api: HfApi | None = None,
    token: str | None = None,
    now: datetime | None = None,
) -> JobStatus:
    api = api or HfApi()
    info = api.inspect_job(job_id=job_id, token=token)
    hours = cost = None
    stage = _stage(info)
    # A canceled job never gets finished_at, so only a running job is timed up to now.
    end = info.finished_at or ((now or datetime.now(UTC)) if stage == "RUNNING" else None)
    if info.started_at is not None and end is not None:
        hours = max((end - info.started_at).total_seconds(), 0.0) / 3600
        price = _price(info)
        cost = hours * price if price is not None else None
    started = info.started_at is not None
    return JobStatus(
        id=info.id,
        url=info.url,
        run=(info.labels or {}).get("run"),
        flavor=info.flavor,
        stage=stage,
        message=info.status.message,
        hours=hours,
        cost=cost,
        progress=job_progress(info, api=api, token=token) if started else [],
    )


def job_logs(
    job_id: str, *, follow: bool = False, api: HfApi | None = None, token: str | None = None
) -> Iterable[str]:
    api = api or HfApi()
    return api.fetch_job_logs(job_id=job_id, follow=follow, token=token)


def cancel_job(job_id: str, *, api: HfApi | None = None, token: str | None = None) -> None:
    api = api or HfApi()
    api.cancel_job(job_id=job_id, token=token)


def _check_no_job_continues(info: JobInfo, *, api: HfApi, token: str) -> None:
    """Refuse a resume while another job runs in the same work folder, or once one finished the
    run: two jobs would write the same checkpoints, or retrain over the pushed adapter."""
    work = _work_volume(info)
    if work is None:
        return
    for other in list_jobs(api=api, token=token):
        other_work = _work_volume(other)
        if other.id == info.id or other_work is None:
            continue
        if (other_work.source, other_work.path) != (work.source, work.path):
            continue
        stage = _stage(other)
        if stage in CONTINUING:
            hint = "the run is finished" if stage == "COMPLETED" else "resume it if it fails"
            raise LaunchError(
                f"{other.id} already continues {info.id}'s run in the same work folder "
                f"and is {stage}; {hint}"
            )


def resume_job(
    job_id: str,
    *,
    timeout: str | None = None,
    api: HfApi | None = None,
    token: str | None = None,
) -> JobInfo:
    """Start a failed or canceled job again with the same command, hardware and /work folder,
    so training resumes from its newest checkpoint. JobInfo has only secret names, so the
    token is always fresh."""
    api = api or HfApi()
    token = token or get_token()
    if token is None:
        raise LaunchError("no Hugging Face token: run `hf auth login` or set HF_TOKEN")
    info = api.inspect_job(job_id=job_id, token=token)
    labels = info.labels or {}
    if labels.get("tool") != TOOL_LABEL["tool"]:
        raise LaunchError(f"{job_id} is not a clef-tune job")
    stage = _stage(info)
    if stage not in RESUMABLE:
        raise LaunchError(f"{job_id} is {stage}; only failed or canceled jobs can be resumed")
    _check_no_job_continues(info, api=api, token=token)
    timeout = timeout or labels.get("timeout")  # JobInfo has no timeout, so launches label it
    if not timeout:
        raise LaunchError(f"{job_id} has no timeout label; pass --timeout")
    timeout_hours(timeout)
    return start_job(
        api,
        image=info.docker_image,
        command=info.command,
        env=info.environment,
        secrets={"HF_TOKEN": token},
        flavor=info.flavor,
        timeout=timeout,
        labels={**{k: labels[k] for k in LABEL_KEYS if k in labels}, "timeout": timeout},
        volumes=info.volumes,
        token=token,
    )
