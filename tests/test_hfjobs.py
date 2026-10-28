"""Opt-in: one real HF Jobs run on l4x1 (about 30 minutes, about $0.40), canceled after its first
checkpoint and resumed. Needs a Hugging Face token with repo.write and job.write, pre-paid Jobs
credits, and the current commit pushed to GitHub.

Run with: uv run pytest tests/test_hfjobs.py -m hfjobs -v -s
"""

from __future__ import annotations

import re
import time
import warnings
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pytest
from huggingface_hub import HfApi, get_token

from clef_tune.launch.hf_jobs import (
    JobRequest,
    cancel_job,
    clean_job,
    job_logs,
    job_status,
    jobs_bucket,
    kernel_cache_folder,
    leftover_files,
    plan_launch,
    resume_job,
    submit_launch,
)
from clef_tune.yamlio import dump_yaml, load_yaml_file

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
POLL_SECONDS = 15
ACTIVE = ("SCHEDULING", "RUNNING")
HARDWARE_WAIT_S = 2 * 60 * 60  # queueing for a GPU costs nothing, so it gets its own budget
CANCEL_SETTLE_S = 3 * 60  # a canceled job shows CANCELED at once but can run on for a minute
_STEP_FILE = re.compile(r"/checkpoints/step-(\d+)/([^/]+)$")  # not step-N.tmp, not best/


def _wait_for(predicate, timeout_s, what, poll_s=POLL_SECONDS, detail=None):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(poll_s)
    pytest.fail(f"timed out waiting for {what}" + (f": {detail()}" if detail else ""))


def _wait_until_running(stage, job_id, poll_s=POLL_SECONDS):
    """Wait while the job queues for hardware, so the queue doesn't eat the job's own budget."""

    def running():
        current = stage(job_id)
        if current not in ACTIVE:
            pytest.fail(f"{job_id} ended as {current} before it started")
        return current == "RUNNING"

    _wait_for(running, HARDWARE_WAIT_S, f"hardware for {job_id}", poll_s)


def _bucket_paths(api, bucket, prefix):
    """Every file path under prefix; [] while the folder does not exist yet."""
    try:
        return [f.path for f in api.list_bucket_tree(bucket, prefix, recursive=True)]
    except Exception:
        return []


def _complete_steps(paths, required):
    """Step numbers whose checkpoint folder holds every required file. Uploads finish in the
    background, so a step folder can exist with only some of its files."""
    found = defaultdict(set)
    for path in paths:
        match = _STEP_FILE.search(path)
        if match:
            found[int(match.group(1))].add(match.group(2))
    return {step for step, names in found.items() if set(required) <= names}


def _quietly(action):
    """Cleanup: one failing step must not skip the others."""
    try:
        action()
    except Exception as error:
        print(f"cleanup step failed: {error}")


@pytest.mark.hfjobs
def test_a_job_resumes_after_cancel_and_pushes_the_adapter(tmp_path):
    from clef_tune.checkpoints import CHECKPOINT_FILES  # imports torch; this test runs with it

    token = get_token()
    if token is None:
        pytest.skip("needs a Hugging Face token")
    api = HfApi(token=token)
    run = f"hfjobs-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    (tmp_path / "support_tickets.csv").write_bytes((EXAMPLES / "support_tickets.csv").read_bytes())
    example = load_yaml_file(EXAMPLES / "support_tickets.yaml")
    config = tmp_path / "config.yaml"
    config.write_text(
        dump_yaml(
            {
                **example,
                "run_name": run,
                "data": {**example["data"], "eval_fraction": 0.2},
                "train": {
                    "quantization": "4bit",
                    "epochs": 2,
                    "max_length": 1024,
                    "max_tokens_per_batch": 1024,
                },
                "eval": {"eval_every": 1000},
                "output": {"save_every": 4},
            }
        ),
        encoding="utf-8",
    )

    plan = plan_launch(config, JobRequest(flavor="l4x1", timeout="1h"), api=api, token=token)
    bucket = jobs_bucket(plan.namespace)
    first = submit_launch(plan, api=api, token=token)
    jobs = [first.id]
    print(f"first job: {first.id} {first.url} (work folder {first.work_folder})")
    saw_progress = []

    def stage(job_id):
        return job_status(job_id, api=api, token=token).stage

    try:
        _wait_until_running(stage, first.id)

        def checkpointed():
            status = job_status(first.id, api=api, token=token)
            if status.stage not in ACTIVE:
                pytest.fail(f"the first job ended as {status.stage} before its first checkpoint")
            if status.latest("baseline") is not None:
                saw_progress.append(True)
            paths = _bucket_paths(api, bucket, f"{first.work_folder}/out")
            return bool(_complete_steps(paths, CHECKPOINT_FILES))

        _wait_for(checkpointed, 45 * 60, "the first complete checkpoint")
        # Fail before paying for the resume if the first job did not cache the kernel wheel.
        _wait_for(
            lambda: any(
                p.endswith(".whl") for p in _bucket_paths(api, bucket, kernel_cache_folder())
            ),
            5 * 60,
            "the cached causal-conv1d wheel",
        )
        cancel_job(first.id, api=api, token=token)
        _wait_for(lambda: stage(first.id) == "CANCELED", 5 * 60, "the cancel")
        time.sleep(CANCEL_SETTLE_S)  # so the old container no longer writes to /work

        resumed = resume_job(first.id, api=api, token=token)
        jobs.append(resumed.id)
        print(f"resumed job: {resumed.id} {resumed.url}")
        _wait_until_running(stage, resumed.id)
        _wait_for(lambda: stage(resumed.id) not in ACTIVE, 60 * 60, "the resumed job")
        logs = "\n".join(job_logs(resumed.id, api=api, token=token))
        assert stage(resumed.id) == "COMPLETED", logs[-3000:]
        assert "Resuming from step" in logs
        assert "Using the cached causal-conv1d wheel" in logs
        assert "fast path is not available" not in logs

        files = set(api.list_repo_files(plan.adapter_repo))
        assert {
            "joint_head.safetensors",
            "calibration.json",
            "adapter/adapter_config.json",
        } <= files

        left = []

        def bundle_deleted():  # the mount's deletes also land in the background
            left[:] = [p for p in _bucket_paths(api, bucket, first.work_folder) if "/bundle/" in p]
            return not left

        _wait_for(bundle_deleted, 5 * 60, "the bundle to be deleted", detail=lambda: left)
        # The mount can finish uploading a checkpoint file after the job deleted it.
        leftovers = leftover_files(resumed.id, api=api, token=token)
        if leftovers:
            warnings.warn(
                f"the bucket kept {leftovers}; `clef-tune jobs clean {resumed.id}` removes them",
                stacklevel=1,
            )
        assert clean_job(resumed.id, api=api, token=token) == leftovers
        assert leftover_files(resumed.id, api=api, token=token) == []
        assert saw_progress, "jobs status never showed metrics.jsonl progress while the job ran"
    finally:
        for job_id in jobs:  # only active ones: canceling a finished job relabels it CANCELED

            def stop(job_id=job_id):
                if stage(job_id) in ACTIVE:
                    cancel_job(job_id, api=api, token=token)

            _quietly(stop)
        _quietly(lambda: api.delete_repo(plan.adapter_repo, missing_ok=True))

        def remove_work_folder():
            leftover = [
                f.path
                for f in api.list_bucket_tree(bucket, first.work_folder, recursive=True)
                if f.type == "file"
            ]
            if leftover:
                api.batch_bucket_files(bucket, delete=leftover)

        _quietly(remove_work_folder)


def test_only_complete_step_folders_count_as_checkpoints():
    names = ("a.pt", "state.json")
    paths = [
        "w/out/checkpoints/step-4.tmp/a.pt",  # written, not yet renamed
        "w/out/checkpoints/step-4.tmp/state.json",
        "w/out/checkpoints/step-8/state.json",  # an upload still under way
        "w/out/checkpoints/step-2/a.pt",
        "w/out/checkpoints/step-2/state.json",
        "w/out/checkpoints/best/state.json",
    ]
    assert _complete_steps(paths, names) == {2}


def test_a_folder_that_does_not_exist_yet_lists_as_empty():
    class Api:
        def list_bucket_tree(self, *args, **kwargs):
            raise OSError("404 Not Found")
            yield  # a generator, like the real one: the error comes on iteration

    assert _bucket_paths(Api(), "me/jobs-artifacts", "clef-tune-runs/x/out") == []


def test_waiting_for_hardware_does_not_use_the_job_budget():
    stages = iter(["SCHEDULING", "SCHEDULING", "RUNNING"])
    _wait_until_running(lambda job_id: next(stages), "job1", poll_s=0)
    assert next(stages, None) is None
    with pytest.raises(pytest.fail.Exception, match="job1 ended as CANCELED"):
        _wait_until_running(lambda job_id: "CANCELED", "job1", poll_s=0)


def test_a_timed_out_wait_says_what_it_last_saw():
    with pytest.raises(pytest.fail.Exception, match=r"the cleanup: \['w/bundle/config.yaml'\]"):
        _wait_for(
            lambda: False, 0, "the cleanup", poll_s=0, detail=lambda: ["w/bundle/config.yaml"]
        )
