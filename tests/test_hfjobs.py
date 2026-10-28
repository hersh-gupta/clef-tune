"""Opt-in: one real HF Jobs run on l4x1 (about 30 minutes, about $0.40), canceled after its first
checkpoint and resumed. Needs a Hugging Face token with repo.write and job.write, pre-paid Jobs
credits, and the current commit pushed to GitHub.

Run with: uv run pytest tests/test_hfjobs.py -m hfjobs -v -s
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from huggingface_hub import HfApi, get_token

from clef_tune.launch.hf_jobs import (
    JobRequest,
    cancel_job,
    job_logs,
    job_status,
    jobs_bucket,
    plan_launch,
    resume_job,
    submit_launch,
)
from clef_tune.yamlio import dump_yaml, load_yaml_file

pytestmark = pytest.mark.hfjobs
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
POLL_SECONDS = 15
ACTIVE = ("SCHEDULING", "RUNNING")


def _wait_for(predicate, timeout_s, what):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(POLL_SECONDS)
    pytest.fail(f"timed out waiting for {what}")


def test_a_job_resumes_after_cancel_and_pushes_the_adapter(tmp_path):
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

    def stage(job_id):
        return job_status(job_id, api=api, token=token).stage

    try:

        def checkpointed():
            current = stage(first.id)
            if current not in ACTIVE:
                pytest.fail(f"the first job ended as {current} before its first checkpoint")
            files = api.list_bucket_tree(bucket, prefix=f"{first.work_folder}/out", recursive=True)
            return any(
                f.path.endswith("/state.json") and "/checkpoints/step-" in f.path for f in files
            )

        _wait_for(checkpointed, 45 * 60, "the first checkpoint")
        cancel_job(first.id, api=api, token=token)
        _wait_for(lambda: stage(first.id) == "CANCELED", 5 * 60, "the cancel")

        resumed = resume_job(first.id, api=api, token=token)
        jobs.append(resumed.id)
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
        left = [
            f.path for f in api.list_bucket_tree(bucket, prefix=first.work_folder, recursive=True)
        ]
        assert not any("/bundle/" in p or "/checkpoints/" in p for p in left), left
    finally:
        for job_id in jobs:
            if stage(job_id) in ACTIVE:
                cancel_job(job_id, api=api, token=token)
        api.delete_repo(plan.adapter_repo, missing_ok=True)
        leftover = [
            f.path
            for f in api.list_bucket_tree(bucket, prefix=first.work_folder, recursive=True)
            if f.type == "file"
        ]
        if leftover:
            api.batch_bucket_files(bucket, delete=leftover)
