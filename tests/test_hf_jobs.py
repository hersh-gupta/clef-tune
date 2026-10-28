from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub import Volume
from huggingface_hub.errors import HfHubHTTPError

from clef_tune.config import PINNED_REVISIONS, load_config
from clef_tune.estimate import HF_FLAVORS, suggest_timeout
from clef_tune.launch import hf_jobs
from clef_tune.launch.hf_jobs import (
    JobRequest,
    LaunchError,
    cancel_job,
    clean_job,
    job_logs,
    job_status,
    kernel_cache_folder,
    leftover_files,
    plan_launch,
    resume_job,
    submit_launch,
)
from clef_tune.launch.job_script import job_command, job_env
from clef_tune.launch.source import Source, SourceError
from clef_tune.runtime import BASE_IMAGE, PRICE_ENV
from clef_tune.yamlio import dump_yaml
from tests.helpers import SCHEMA, CharCounter, ticket_rows, write_csv

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "support_tickets.yaml"
SOURCE = Source(
    "https://github.com/hersh-gupta/clef-tune", "0123456789abcdef0123456789abcdef01234567"
)
WRITE = {"name": "me", "auth": {"type": "access_token", "accessToken": {"role": "write"}}}
READ = {"name": "me", "auth": {"type": "access_token", "accessToken": {"role": "read"}}}
TOKEN = "hf_test"


class FakeApi:
    """Records every Hub call; returns canned answers."""

    def __init__(
        self,
        *,
        token_info=WRITE,
        bucket_files=(),
        run_error=None,
        job_info=None,
        metrics=None,
        jobs=(),
        logs=(),
    ):
        self.calls, self.staged = [], {}
        self.token_info, self.bucket_files, self.run_error = (
            token_info,
            list(bucket_files),
            run_error,
        )
        self.job_info, self.metrics, self.jobs, self.logs = (
            job_info,
            metrics,
            list(jobs),
            list(logs),
        )

    def whoami(self, token=None, **kwargs):
        self.calls.append(("whoami",))
        return self.token_info

    def list_bucket_tree(self, bucket_id, prefix=None, *, recursive=None, token=None):
        self.calls.append(("list_bucket_tree", bucket_id, prefix))
        return [
            SimpleNamespace(type="file", path=p)
            for p in self.bucket_files
            if p.startswith(prefix or "")
        ]

    def sync_job_volume(
        self, source, mount_path, *, remote_name=None, read_only=True, namespace=None, token=None
    ):
        self.calls.append(("sync_job_volume", mount_path, remote_name, read_only))
        bundle = Path(source) / "bundle"
        if (bundle / "config.yaml").exists():
            self.staged = {
                "config": load_config(bundle / "config.yaml"),
                "files": sorted(p.name for p in bundle.iterdir()),
            }
        return Volume(
            type="bucket",
            source="me/jobs-artifacts",
            mount_path=mount_path,
            path=remote_name,
            read_only=read_only,
        )

    def run_job(self, **kwargs):
        self.calls.append(("run_job", kwargs))
        if self.run_error:
            raise self.run_error
        return SimpleNamespace(id="job123", url="https://huggingface.co/jobs/me/job123")

    def batch_bucket_files(self, bucket_id, *, add=None, delete=None, token=None, **kwargs):
        self.calls.append(("batch_bucket_files", bucket_id, delete))

    def inspect_job(self, *, job_id, token=None, **kwargs):
        self.calls.append(("inspect_job", job_id))
        return self.job_info

    def list_jobs(self, *, labels=None, token=None, **kwargs):
        self.calls.append(("list_jobs", labels))
        return list(self.jobs)

    def download_bucket_files(self, bucket_id, files, *, raise_on_missing_files=False, token=None):
        for remote, local in files:
            self.calls.append(("download_bucket_files", bucket_id, remote))
            if self.metrics is not None:
                Path(local).write_text(
                    "\n".join(json.dumps(m) for m in self.metrics), encoding="utf-8"
                )

    def fetch_job_logs(self, *, job_id, follow=False, token=None, **kwargs):
        self.calls.append(("fetch_job_logs", job_id, follow))
        return iter(self.logs)

    def cancel_job(self, *, job_id, token=None, **kwargs):
        self.calls.append(("cancel_job", job_id))


def project(tmp_path, rows=None, **sections):
    write_csv(tmp_path / "train.csv", rows or ticket_rows(40))
    data = {
        "run_name": "t",
        "data": {"train": "train.csv", "id_column": "id"},
        "schema": SCHEMA,
        **sections,
    }
    path = tmp_path / "config.yaml"
    path.write_text(dump_yaml(data), encoding="utf-8")
    return path


def plan(api=None, request=JobRequest(), config=EXAMPLE, source=SOURCE):
    return plan_launch(
        config,
        request,
        api=api or FakeApi(),
        token=TOKEN,
        counter=CharCounter(),
        prices={},
        source=source,
    )


def test_the_plan_picks_the_cheapest_fitting_flavor_and_allows_for_the_kernel_build():
    api = FakeApi()
    p = plan(api)
    assert (
        p.estimate.fits and p.flavor == p.estimate.flavor and p.gpus == HF_FLAVORS[p.flavor].count
    )
    assert p.timeout == suggest_timeout(p.estimate, extra_minutes=hf_jobs.KERNEL_BUILD_MINUTES)
    assert p.adapter_repo == "me/clef-tune-support-tickets" and p.kernels and not p.kernels_cached
    assert p.price_per_hour == HF_FLAVORS[p.flavor].price_per_hour
    assert {call[0] for call in api.calls} <= {
        "whoami",
        "list_bucket_tree",
    }  # nothing uploaded or started


def test_the_kernel_cache_folder_names_the_image_and_version():
    assert kernel_cache_folder() == (
        "clef-tune-kernels/pytorch-pytorch-2.11.0-cuda13.0-cudnn9-devel/causal-conv1d-1.7.0"
    )


def test_a_cached_kernel_wheel_needs_no_build_time():
    folder = kernel_cache_folder()
    api = FakeApi(
        bucket_files=[
            f"{folder}/.keep",
            f"{folder}/causal_conv1d-1.7.0-cp312-cp312-linux_x86_64.whl",
        ]
    )
    p = plan(api)
    assert p.kernels_cached and p.timeout == suggest_timeout(p.estimate)
    assert ("list_bucket_tree", "me/jobs-artifacts", folder) in api.calls


def test_without_kernels_there_is_no_cache_lookup_or_build_time():
    api = FakeApi()
    p = plan(api, JobRequest(kernels=False))
    assert not p.kernels and p.timeout == suggest_timeout(p.estimate)
    assert all(call[0] != "list_bucket_tree" for call in api.calls)


def test_a_multi_gpu_flavor_runs_one_process_per_gpu():
    assert plan(request=JobRequest(flavor="h200x2")).gpus == 2


def test_an_explicit_timeout_and_its_maximum_cost():
    p = plan(request=JobRequest(flavor="h200", timeout="2h"))
    assert p.timeout == "2h" and p.max_cost == pytest.approx(2 * HF_FLAVORS["h200"].price_per_hour)


@pytest.mark.parametrize("timeout", ["90", "1 h", "h", "2w"])
def test_a_timeout_needs_a_number_and_a_unit(timeout):
    with pytest.raises(LaunchError, match="timeout"):
        plan(request=JobRequest(flavor="h200", timeout=timeout))


def test_an_unknown_flavor_lists_the_known_ones():
    with pytest.raises(LaunchError, match="a10g-small"):
        plan(request=JobRequest(flavor="tpu-v5"))


@pytest.mark.parametrize("flavor, reason", [("t4-small", "bf16"), ("a10g-small", "needs ~")])
def test_a_flavor_that_cannot_run_the_config_is_refused(tmp_path, flavor, reason):
    config = project(tmp_path)  # Cloudflare/clef in bf16: ~87 GB
    with pytest.raises(LaunchError, match=f"cannot run this config.*{reason}"):
        plan(config=config, request=JobRequest(flavor=flavor))


def test_invalid_data_is_refused_before_touching_the_hub(tmp_path, monkeypatch):
    rows = ticket_rows(20)
    rows[1]["department"] = "biling"
    monkeypatch.setattr(hf_jobs, "resolve_source", lambda ref: pytest.fail("resolved the source"))
    api = FakeApi()
    with pytest.raises(LaunchError, match="validation failed"):
        plan(api, config=project(tmp_path, rows), source=None)
    assert api.calls == []


def test_a_dirty_checkout_is_refused_before_touching_the_hub(monkeypatch):
    def dirty(ref):
        raise SourceError("uncommitted changes in /src; commit and push them, or pass --source-ref")

    monkeypatch.setattr(hf_jobs, "resolve_source", dirty)
    api = FakeApi()
    with pytest.raises(LaunchError, match="uncommitted changes"):
        plan(api, source=None)
    assert api.calls == []


def test_the_source_is_resolved_and_pinned_when_not_given(monkeypatch):
    monkeypatch.setattr(
        hf_jobs, "resolve_source", lambda ref: Source(SOURCE.repo_url, ref or "main")
    )
    monkeypatch.setattr(hf_jobs, "pin_commit", lambda source: SOURCE)
    assert plan(request=JobRequest(source_ref="v1"), source=None).source == SOURCE


def test_the_adapter_repo_needs_write_access():
    with pytest.raises(LaunchError, match="write access"):
        plan(FakeApi(token_info=READ))


def test_the_requested_adapter_repo_wins(tmp_path):
    config = project(
        tmp_path, model={"id": "Cloudflare/clef-flash"}, output={"push_to_hub": "me/cfg"}
    )
    assert plan(config=config).adapter_repo == "me/cfg"
    assert plan(config=config, request=JobRequest(adapter_repo="me/x")).adapter_repo == "me/x"


def test_a_token_is_required(monkeypatch):
    monkeypatch.setattr(hf_jobs, "get_token", lambda: None)
    with pytest.raises(LaunchError, match="hf auth login"):
        plan_launch(
            EXAMPLE, api=FakeApi(), token=None, counter=CharCounter(), prices={}, source=SOURCE
        )


def test_without_a_tokenizer_the_flavor_and_timeout_must_be_given(monkeypatch):
    def offline(config):
        raise OSError("offline")

    monkeypatch.setattr(hf_jobs, "counter_for_config", offline)
    kwargs = {"api": FakeApi(), "token": TOKEN, "prices": {}, "source": SOURCE}
    with pytest.raises(LaunchError, match="--flavor and --timeout"):
        plan_launch(EXAMPLE, **kwargs)
    p = plan_launch(EXAMPLE, JobRequest(flavor="l4x1", timeout="1h"), **kwargs)
    assert p.estimate is None and p.estimated_hours is None and p.gpus == 1


NOW = datetime(2026, 10, 28, 12, 0, 0, tzinfo=UTC)
FOLDER = "clef-tune-runs/support-tickets-20261028-120000"


def run_call(api):
    return next(call[1] for call in api.calls if call[0] == "run_job")


def test_submit_uploads_the_bundle_and_starts_the_job():
    api = FakeApi()
    p = plan(api)
    job = submit_launch(p, api=api, token=TOKEN, now=NOW)
    assert (job.id, job.url, job.work_folder) == (
        "job123",
        "https://huggingface.co/jobs/me/job123",
        FOLDER,
    )
    syncs = [call for call in api.calls if call[0] == "sync_job_volume"]
    assert syncs == [
        ("sync_job_volume", "/work", FOLDER, False),
        ("sync_job_volume", "/kernels", kernel_cache_folder(), False),
    ]
    run = run_call(api)
    assert run["image"] == BASE_IMAGE and run["flavor"] == p.flavor and run["timeout"] == p.timeout
    assert run["secrets"] == {"HF_TOKEN": TOKEN} and run["token"] == TOKEN
    assert run["labels"] == {"tool": "clef-tune", "run": "support-tickets", "timeout": p.timeout}
    assert run["command"] == job_command(
        SOURCE,
        gpus=p.gpus,
        adapter_repo="me/clef-tune-support-tickets",
        keep_data=False,
        kernels=True,
    )
    assert run["env"] == job_env(
        SOURCE, price_per_hour=p.price_per_hour, estimated_hours=p.estimated_hours
    )
    work, model, kernels = run["volumes"]
    assert (work.mount_path, work.path, work.read_only) == ("/work", FOLDER, False)
    assert model == Volume(
        type="model",
        source="Cloudflare/clef-flash",
        mount_path="/model",
        revision=PINNED_REVISIONS["Cloudflare/clef-flash"],
    )
    assert kernels.mount_path == "/kernels"


def test_the_staged_bundle_points_at_the_job_mounts(tmp_path):
    config = project(
        tmp_path,
        model={"id": "Cloudflare/clef-flash", "local_path": "weights"},
        output={"dir": "runs/x", "push_to_hub": "me/other"},
    )
    api = FakeApi()
    submit_launch(plan(api, config=config), api=api, token=TOKEN, now=NOW)
    staged = api.staged["config"]
    assert staged.model.local_path == "/model" and staged.output.dir == "/work/out"
    assert staged.output.push_to_hub is None  # the job passes --push-to-hub
    assert api.staged["files"] == ["config.yaml", "eval.jsonl", "train.jsonl"]


def test_keep_data_keeps_the_bundle():
    api = FakeApi()
    submit_launch(plan(api, JobRequest(keep_data=True)), api=api, token=TOKEN, now=NOW)
    assert "--delete-bundle-on-success" not in run_call(api)["command"][-1]


def test_without_kernels_no_cache_is_mounted():
    api = FakeApi()
    submit_launch(plan(api, JobRequest(kernels=False)), api=api, token=TOKEN, now=NOW)
    run = run_call(api)
    assert len(run["volumes"]) == 2 and "causal" not in run["command"][-1]


def test_each_launch_gets_its_own_work_folder():
    api = FakeApi()
    p = plan(api)
    first = submit_launch(p, api=api, token=TOKEN, now=NOW)
    second = submit_launch(p, api=api, token=TOKEN, now=NOW + timedelta(seconds=1))
    assert first.work_folder != second.work_folder


@pytest.mark.parametrize(
    "status, message",
    [
        (402, r"pre-paid credits.*settings/billing"),
        (500, "could not start the job"),
        (403, r"could not start the job.*job\.write and repo\.write"),
        (401, r"could not start the job.*job\.write and repo\.write"),
    ],
)
def test_out_of_credits_is_explained_and_the_upload_removed(status, message):
    request = httpx.Request("POST", "https://huggingface.co/api/jobs/me")
    error = HfHubHTTPError(f"{status} error", response=httpx.Response(status, request=request))
    staged = [f"{FOLDER}/bundle/config.yaml", f"{FOLDER}/bundle/train.jsonl"]
    api = FakeApi(run_error=error, bucket_files=staged)
    with pytest.raises(LaunchError, match=message):
        submit_launch(plan(api), api=api, token=TOKEN, now=NOW)
    assert ("batch_bucket_files", "me/jobs-artifacts", staged) in api.calls


def test_a_failed_upload_is_explained_and_cleaned_up():
    staged = [f"{FOLDER}/bundle/config.yaml"]
    api = FakeApi(bucket_files=staged)
    p = plan(api)

    def refuse(*args, **kwargs):
        request = httpx.Request("POST", "https://huggingface.co/api/buckets/me/jobs-artifacts")
        raise HfHubHTTPError("403 Forbidden", response=httpx.Response(403, request=request))

    api.sync_job_volume = refuse
    with pytest.raises(LaunchError, match=r"could not upload the bundle.*403.*job\.write"):
        submit_launch(p, api=api, token=TOKEN, now=NOW)
    assert ("batch_bucket_files", "me/jobs-artifacts", staged) in api.calls
    assert all(call[0] != "run_job" for call in api.calls)


WORK = Volume(
    type="bucket",
    source="me/jobs-artifacts",
    mount_path="/work",
    path="clef-tune-runs/r-1",
    read_only=False,
)


def job(stage="RUNNING", **overrides):
    values = {
        "id": "job123",
        "url": "https://huggingface.co/jobs/me/job123",
        "status": SimpleNamespace(stage=stage, message=None),
        "labels": {"tool": "clef-tune", "run": "r", "timeout": "1h", "name": "set-by-hf"},
        "docker_image": BASE_IMAGE,
        "command": ["bash", "-c", "echo train"],
        "environment": {"PYTHONUNBUFFERED": "1", PRICE_ENV: "0.8"},
        "flavor": "l4x1",
        "volumes": [
            WORK,
            Volume(type="model", source="Cloudflare/clef-flash", mount_path="/model"),
        ],
        "created_at": NOW,
        "started_at": NOW - timedelta(minutes=30),
        "finished_at": None,
    }
    return SimpleNamespace(**(values | overrides))


def test_list_jobs_filters_by_the_tool_label_newest_first():
    api = FakeApi(jobs=[job(id="a", created_at=NOW - timedelta(hours=1)), job(id="b")])
    assert [j.id for j in hf_jobs.list_jobs(api=api, token=TOKEN)] == ["b", "a"]
    assert ("list_jobs", {"tool": "clef-tune"}) in api.calls


def test_status_reads_progress_from_the_work_folder_and_prices_the_time():
    metrics = [
        {"event": "baseline", "step": 0, "exact_match": 0.1},
        {"event": "train", "step": 20, "loss": 0.5, "tokens_per_sec": 400.0, "eta_s": 600.0},
        {"event": "eval", "step": 20, "exact_match": 0.4},
    ]
    api = FakeApi(job_info=job(), metrics=metrics)
    status = job_status("job123", api=api, token=TOKEN, now=NOW)
    assert (status.stage, status.run, status.flavor) == ("RUNNING", "r", "l4x1")
    assert status.hours == pytest.approx(0.5) and status.cost == pytest.approx(0.4)
    assert status.latest("train")["step"] == 20 and status.latest("eval")["exact_match"] == 0.4
    assert (
        "download_bucket_files",
        "me/jobs-artifacts",
        "clef-tune-runs/r-1/out/metrics.jsonl",
    ) in api.calls


def test_status_before_the_job_starts_has_no_time_cost_or_progress():
    api = FakeApi(job_info=job(stage="SCHEDULING", started_at=None))
    status = job_status("job123", api=api, token=TOKEN, now=NOW)
    assert (status.hours, status.cost, status.progress) == (None, None, [])


def test_a_finished_job_is_priced_to_its_finish_time():
    finished = job(stage="COMPLETED", finished_at=NOW - timedelta(minutes=15))
    status = job_status("job123", api=FakeApi(job_info=finished), token=TOKEN, now=NOW)
    assert status.hours == pytest.approx(0.25)


def test_resume_rebuilds_the_failed_job_with_a_fresh_token():
    original = job(stage="ERROR")
    api = FakeApi(job_info=original)
    resume_job("job123", api=api, token=TOKEN)
    assert run_call(api) == {
        "image": BASE_IMAGE,
        "command": original.command,
        "env": original.environment,
        "secrets": {"HF_TOKEN": TOKEN},
        "flavor": "l4x1",
        "timeout": "1h",
        "labels": {"tool": "clef-tune", "run": "r", "timeout": "1h"},  # HF's own labels dropped
        "volumes": original.volumes,  # same /work folder: training auto-resumes from its checkpoint
        "token": TOKEN,
    }


def test_resume_can_extend_the_timeout():
    api = FakeApi(job_info=job(stage="CANCELED"))
    resume_job("job123", timeout="3h", api=api, token=TOKEN)
    run = run_call(api)
    assert run["timeout"] == "3h" and run["labels"]["timeout"] == "3h"


@pytest.mark.parametrize("stage", ["RUNNING", "SCHEDULING", "COMPLETED"])
def test_only_failed_or_canceled_jobs_resume(stage):
    with pytest.raises(LaunchError, match="only failed or canceled"):
        resume_job("job123", api=FakeApi(job_info=job(stage=stage)), token=TOKEN)


def test_other_tools_jobs_do_not_resume():
    with pytest.raises(LaunchError, match="not a clef-tune job"):
        resume_job("job123", api=FakeApi(job_info=job(stage="ERROR", labels={})), token=TOKEN)


def test_logs_and_cancel_pass_the_job_through():
    api = FakeApi(logs=["line 1", "line 2"])
    assert list(job_logs("job123", follow=True, api=api, token=TOKEN)) == ["line 1", "line 2"]
    cancel_job("job123", api=api, token=TOKEN)
    assert ("fetch_job_logs", "job123", True) in api.calls and ("cancel_job", "job123") in api.calls


@pytest.mark.parametrize("stage", ["RUNNING", "SCHEDULING", "COMPLETED"])
def test_resume_refuses_when_another_job_continues_the_same_run(stage):
    canceled = job(stage="CANCELED")
    api = FakeApi(job_info=canceled, jobs=[canceled, job(id="job456", stage=stage)])
    with pytest.raises(LaunchError, match=f"job456 .*same work folder.*{stage}"):
        resume_job("job123", api=api, token=TOKEN)
    assert all(call[0] != "run_job" for call in api.calls)


def test_resume_ignores_other_folders_and_failed_resumes():
    other = Volume(
        type="bucket",
        source="me/jobs-artifacts",
        mount_path="/work",
        path="clef-tune-runs/other-1",
        read_only=False,
    )
    canceled = job(stage="CANCELED")
    elsewhere = job(id="job456", stage="RUNNING", volumes=[other])
    api = FakeApi(job_info=canceled, jobs=[canceled, elsewhere, job(id="job789", stage="ERROR")])
    resume_job("job123", api=api, token=TOKEN)
    assert run_call(api)["volumes"] == canceled.volumes


def test_a_canceled_job_without_a_finish_time_has_no_time_or_cost():
    # HF leaves finished_at unset on canceled jobs; counting to now would grow forever.
    canceled = job(stage="CANCELED", finished_at=None)
    status = job_status("job123", api=FakeApi(job_info=canceled), token=TOKEN, now=NOW)
    assert (status.hours, status.cost) == (None, None)


TRAIN = "clef-tune train /work/bundle/config.yaml --push-to-hub me/x --delete-bundle-on-success"
FINISHED = [
    "clef-tune-runs/r-1/bundle/config.yaml",
    "clef-tune-runs/r-1/out/checkpoints/best/trainable.safetensors",
    "clef-tune-runs/r-1/out/final/joint_head.safetensors",
    "clef-tune-runs/r-1/out/metrics.jsonl",
]


def test_clean_removes_what_a_completed_job_left_behind():
    # The bucket mount can finish an upload after the job deleted the file.
    done = job(stage="COMPLETED", command=["bash", "-c", TRAIN])
    api = FakeApi(job_info=done, bucket_files=FINISHED)
    assert leftover_files("job123", api=api, token=TOKEN) == FINISHED[:2]
    assert clean_job("job123", api=api, token=TOKEN) == FINISHED[:2]
    assert ("batch_bucket_files", "me/jobs-artifacts", FINISHED[:2]) in api.calls


def test_clean_keeps_a_bundle_launched_with_keep_data():
    done = job(
        stage="COMPLETED", command=["bash", "-c", TRAIN.removesuffix(" --delete-bundle-on-success")]
    )
    api = FakeApi(job_info=done, bucket_files=FINISHED)
    assert clean_job("job123", api=api, token=TOKEN) == [FINISHED[1]]


def test_clean_with_nothing_left_deletes_nothing():
    done = job(stage="COMPLETED", command=["bash", "-c", TRAIN])
    api = FakeApi(job_info=done, bucket_files=FINISHED[2:])
    assert clean_job("job123", api=api, token=TOKEN) == []
    assert all(call[0] != "batch_bucket_files" for call in api.calls)


@pytest.mark.parametrize("stage", ["RUNNING", "SCHEDULING", "CANCELED", "ERROR"])
def test_only_completed_jobs_are_cleaned(stage):
    api = FakeApi(job_info=job(stage=stage, command=["bash", "-c", TRAIN]), bucket_files=FINISHED)
    assert leftover_files("job123", api=api, token=TOKEN) == []  # checkpoints for a resume
    with pytest.raises(LaunchError, match="only completed jobs"):
        clean_job("job123", api=api, token=TOKEN)
    assert all(call[0] != "batch_bucket_files" for call in api.calls)


def test_job_status_names_the_adapter_repo():
    command = job_command(
        SOURCE, gpus=1, adapter_repo="me/clef-tune-x", keep_data=False, kernels=True
    )
    api = FakeApi(job_info=job(command=command))
    assert job_status("job123", api=api, token=TOKEN).adapter_repo == "me/clef-tune-x"
    assert hf_jobs.adapter_repo_of(job()) is None  # "echo train" pushes nowhere


def test_quote_resume_prices_the_timeout_and_starts_nothing():
    info = job(
        "ERROR",
        labels={"tool": "clef-tune", "run": "r", "timeout": "2h"},
        environment={PRICE_ENV: "1.5"},
    )
    api = FakeApi(job_info=info, jobs=[info])
    quote = hf_jobs.quote_resume("job123", api=api, token=TOKEN)
    assert (quote.timeout, quote.price_per_hour, quote.max_cost) == ("2h", 1.5, 3.0)
    assert hf_jobs.quote_resume("job123", timeout="3h", api=api, token=TOKEN).max_cost == 4.5
    assert not [c for c in api.calls if c[0] == "run_job"]


def test_quote_resume_refuses_what_resume_refuses():
    api = FakeApi(job_info=job("RUNNING"))
    with pytest.raises(LaunchError, match="only failed or canceled"):
        hf_jobs.quote_resume("job123", api=api, token=TOKEN)
