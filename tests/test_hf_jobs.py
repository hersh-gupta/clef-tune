from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub import Volume

from clef_tune.config import load_config
from clef_tune.estimate import HF_FLAVORS, suggest_timeout
from clef_tune.launch import hf_jobs
from clef_tune.launch.hf_jobs import JobRequest, LaunchError, kernel_cache_folder, plan_launch
from clef_tune.launch.source import Source, SourceError
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
