import subprocess
from types import SimpleNamespace

import pytest
from huggingface_hub import Volume

from clef_tune import hub
from clef_tune.config import PINNED_REVISIONS, load_config
from clef_tune.estimate import estimate_prediction, model_spec, prediction_workload
from clef_tune.launch import hf_jobs
from clef_tune.launch.hf_jobs import LaunchError
from clef_tune.launch.job_script import predict_script
from tests.helpers import write_csv
from tests.test_hf_jobs import EXAMPLE, SOURCE, TOKEN, FakeApi
from tests.test_job_script import BASH, POSIX_BASH

CONFIG = load_config(EXAMPLE)


def test_a_prediction_estimate_is_one_forward_pass_plus_setup():
    workload = prediction_workload(CONFIG, tokens=1_240 * 360)
    assert workload.mode == "head_only" and workload.epochs == 1
    row = estimate_prediction(model_spec(CONFIG.model.id), CONFIG, 1_240 * 360, {})
    assert row is not None and row.fits
    assert 10 / 60 <= row.hours < 15 / 60  # mostly the 10 minutes of start, install and load


def test_the_prediction_script_predicts_into_the_work_folder():
    script = predict_script(
        SOURCE,
        adapter_repo="me/support-tickets",
        input_name="new.csv",
        review_below=0.7,
        keep_input=False,
        kernels=True,
    )
    assert "clef-tune predict me/support-tickets /work/input/new.csv" in script
    assert "-o /work/out/predictions.csv --review-below 0.7 --base-model /model" in script
    assert "rm -f /work/input/new.csv" in script and "causal-conv1d" in script
    kept = predict_script(
        SOURCE,
        adapter_repo="me/x",
        input_name="a b.csv",
        review_below=0.5,
        keep_input=True,
        kernels=False,
    )
    assert "rm -f" not in kept and "'/work/input/a b.csv'" in kept
    if BASH:
        subprocess.run([BASH, "-n"], input=script, text=True, check=True)


def plan(tmp_path, api=None, **kwargs):
    csv_path = write_csv(tmp_path / "new tickets.csv", [{"subject": "a", "body": "b"}])
    return hf_jobs.plan_prediction(
        "me/support-tickets",
        csv_path,
        config=CONFIG,
        tokens=1000,
        rows=1,
        review_below=0.7,
        token=TOKEN,
        api=api or FakeApi(),
        prices={},
        source=SOURCE,
        **kwargs,
    )


def test_a_prediction_plan_prices_the_run_and_its_cap(tmp_path):
    prediction = plan(tmp_path)
    assert prediction.namespace == "me" and prediction.flavor == "l40sx1"
    assert prediction.base_model == "Cloudflare/clef-flash"
    assert prediction.max_cost is not None and prediction.max_cost > 0
    assert prediction.run_name == "support-tickets-predict"


def test_submitting_a_prediction_stages_the_input_and_labels_the_job(tmp_path):
    api = FakeApi()
    job = hf_jobs.submit_prediction(plan(tmp_path, api), api=api, token=TOKEN)
    assert job.id == "job123"
    run = next(call[1] for call in api.calls if call[0] == "run_job")
    assert run["labels"]["kind"] == "predict" and run["labels"]["tool"] == "clef-tune"
    command = " ".join(run["command"])
    assert "clef-tune predict me/support-tickets /work/input/new-tickets.csv" in command
    model = next(v for v in run["volumes"] if v.mount_path == "/model")
    assert model.source == "Cloudflare/clef-flash"
    assert model.revision == PINNED_REVISIONS["Cloudflare/clef-flash"]
    assert run["secrets"] == {"HF_TOKEN": TOKEN}


def job(stage="COMPLETED", kind="predict", command=None):
    return SimpleNamespace(
        id="job9",
        status=SimpleNamespace(stage=stage, message=None),
        labels={"tool": "clef-tune", "run": "r", "kind": kind},
        command=command or ["bash", "-c", "clef-tune predict me/support-tickets /work/input/a.csv"],
        volumes=[
            Volume(
                type="bucket",
                source="me/jobs-artifacts",
                mount_path="/work",
                path="clef-tune-runs/r-1",
            )
        ],
    )


def test_a_jobs_kind_and_the_model_a_prediction_used():
    assert hf_jobs.job_kind(job()) == "predict"
    assert hf_jobs.job_kind(job(kind=None)) == "train"
    assert hf_jobs.prediction_adapter_of(job()) == "me/support-tickets"


def test_predictions_are_downloaded_from_the_work_folder(tmp_path):
    api = FakeApi(job_info=job(), metrics=[{"x": 1}])
    out = hf_jobs.download_predictions("job9", tmp_path / "p.csv", api=api, token=TOKEN)
    assert out == tmp_path / "p.csv" and out.exists()
    assert (
        "download_bucket_files",
        "me/jobs-artifacts",
        "clef-tune-runs/r-1/out/predictions.csv",
    ) in api.calls


def test_missing_predictions_are_an_error(tmp_path):
    api = FakeApi(job_info=job())
    with pytest.raises(LaunchError, match="no predictions"):
        hf_jobs.download_predictions("job9", tmp_path / "p.csv", api=api, token=TOKEN)


def test_deleting_a_runs_files_removes_its_whole_folder():
    files = ["clef-tune-runs/r-1/input/a.csv", "clef-tune-runs/r-1/out/predictions.csv"]
    api = FakeApi(job_info=job(), bucket_files=files + ["clef-tune-runs/other/x"])
    assert hf_jobs.delete_job_files("job9", api=api, token=TOKEN) == files
    assert ("batch_bucket_files", "me/jobs-artifacts", files) in api.calls


@pytest.mark.parametrize("stage", ["RUNNING", "SCHEDULING"])
def test_a_running_jobs_files_are_kept(stage):
    with pytest.raises(LaunchError, match="still"):
        hf_jobs.delete_job_files("job9", api=FakeApi(job_info=job(stage)), token=TOKEN)


def test_a_runs_files_come_from_its_adapter_repo(tmp_path):
    (tmp_path / "metrics.json").write_text('{"final": {}}', encoding="utf-8")
    (tmp_path / "train_config.yaml").write_text("run_name: x\n", encoding="utf-8")
    (tmp_path / "schema.yaml").write_text("questions: {}\n", encoding="utf-8")

    def download(repo_id, filename, token=None):
        return str(tmp_path / filename)

    files = hub.pull_run_files("me/x", token=TOKEN, download=download)
    assert files.metrics == {"final": {}} and files.config == {"run_name": "x"}
    assert files.schema == {"questions": {}}


def test_a_prediction_job_remembers_its_threshold_and_the_persons_column_names(tmp_path):
    from tests.test_hf_jobs import job as hf_job

    api = FakeApi()
    names = {"ticket_id": "row", "subject": "title"}
    hf_jobs.submit_prediction(plan(tmp_path, api, output_names=names), api=api, token=TOKEN)
    run = next(call[1] for call in api.calls if call[0] == "run_job")
    info = hf_job("COMPLETED", command=run["command"], environment=run["env"], labels=run["labels"])
    status = hf_jobs.job_status("job123", api=FakeApi(job_info=info), token=TOKEN)
    assert status.output_names == names and status.review_below == 0.7
    assert hf_jobs.review_below_of(job()) is None
    bare = hf_job("COMPLETED", labels=run["labels"])
    assert hf_jobs.job_status("job123", api=FakeApi(job_info=bare), token=TOKEN).output_names == {}


def test_a_failed_or_stopped_prediction_still_removes_the_uploaded_input():
    script = predict_script(
        SOURCE,
        adapter_repo="me/support-tickets",
        input_name="new.csv",
        review_below=0.7,
        keep_input=False,
        kernels=False,
    )
    lines = script.splitlines()
    cleanup = lines.index("trap 'rm -f /work/input/new.csv' EXIT")
    assert cleanup < next(k for k, line in enumerate(lines) if line.startswith("clef-tune predict"))
    kept = predict_script(
        SOURCE,
        adapter_repo="me/x",
        input_name="a.csv",
        review_below=0.5,
        keep_input=True,
        kernels=False,
    )
    assert "EXIT" not in kept


@POSIX_BASH
@pytest.mark.parametrize("outcome", ["exit 1", "sleep 30"])
def test_the_input_is_removed_however_predicting_ends(tmp_path, outcome):
    import signal

    work = tmp_path / "work"
    (work / "input").mkdir(parents=True)
    uploaded = work / "input" / "new.csv"
    uploaded.write_text("a\n1\n", encoding="utf-8")
    stubs = tmp_path / "bin"
    stubs.mkdir()
    (stubs / "pip").write_text("#!/bin/sh\nexit 0\n")
    (stubs / "clef-tune").write_text(f"#!/bin/sh\necho started\n{outcome}\n")
    for stub in stubs.iterdir():
        stub.chmod(0o755)
    script = predict_script(
        SOURCE,
        adapter_repo="me/x",
        input_name="new.csv",
        review_below=0.7,
        keep_input=False,
        kernels=False,
    ).replace("/work", str(work))
    job = subprocess.Popen(
        ["bash", "-c", script],
        env={"PATH": f"{stubs}:/usr/bin:/bin"},
        stdout=subprocess.PIPE,
        text=True,
    )
    while job.stdout.readline().strip() != "started":
        pass
    if outcome.startswith("sleep"):
        job.send_signal(signal.SIGTERM)
    assert job.wait(timeout=10) != 0
    assert not uploaded.exists()
