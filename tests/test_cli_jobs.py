from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from clef_tune import cli
from clef_tune.launch.hf_jobs import JobRequest, JobStatus, LaunchError

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "support_tickets.yaml"
runner = CliRunner()


def fake_plan(**overrides):
    values = {
        "config": SimpleNamespace(run_name="support-tickets"),
        "source": SimpleNamespace(
            repo_url="https://github.com/hersh-gupta/clef-tune",
            ref="0123456789abcdef0123456789abcdef01234567",
        ),
        "adapter_repo": "me/clef-tune-support-tickets",
        "flavor": "l40sx1",
        "gpus": 1,
        "timeout": "1h",
        "estimate": SimpleNamespace(hours_low=0.2, hours_high=0.45, cost_low=0.36, cost_high=0.81),
        "max_cost": 1.8,
        "kernels": True,
        "kernels_cached": False,
        "keep_data": False,
    }
    return SimpleNamespace(**(values | overrides))


@pytest.fixture
def launcher(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli,
        "plan_launch",
        lambda config, request, **k: calls.append(("plan", request)) or fake_plan(),
    )
    monkeypatch.setattr(
        cli,
        "submit_launch",
        lambda plan, **k: (
            calls.append(("submit",))
            or SimpleNamespace(
                id="job123",
                url="https://huggingface.co/jobs/me/job123",
                work_folder="clef-tune-runs/x",
            )
        ),
    )
    return calls


def test_launch_shows_the_estimate_and_asks_before_spending(launcher):
    result = runner.invoke(cli.app, ["jobs", "launch", str(EXAMPLE)], input="n\n")
    assert result.exit_code == 1
    for text in (
        "l40sx1",
        "1h",
        "$0.36",
        "$0.81",
        "$1.80",
        "me/clef-tune-support-tickets",
        "0123456789ab",
    ):
        assert text in result.output
    assert launcher == [("plan", JobRequest())]


def test_launch_with_yes_passes_every_option_and_prints_the_job(launcher):
    args = [
        "jobs",
        "launch",
        str(EXAMPLE),
        "--yes",
        "--flavor",
        "h200",
        "--timeout",
        "6h",
        "--keep-data",
        "--no-kernels",
        "--source-ref",
        "v1",
        "--push-to-hub",
        "org/x",
    ]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    request = JobRequest(
        flavor="h200",
        timeout="6h",
        keep_data=True,
        kernels=False,
        adapter_repo="org/x",
        source_ref="v1",
    )
    assert launcher == [("plan", request), ("submit",)]
    assert "job123" in result.output and "clef-tune jobs logs job123 --follow" in result.output


def test_launch_errors_exit_1(monkeypatch):
    def refuse(config, request, **kwargs):
        raise LaunchError("v1 is not on GitHub at https://github.com/hersh-gupta/clef-tune")

    monkeypatch.setattr(cli, "plan_launch", refuse)
    result = runner.invoke(cli.app, ["jobs", "launch", str(EXAMPLE)])
    assert result.exit_code == 1 and "not on GitHub" in result.output


def test_list_shows_each_job(monkeypatch):
    job = SimpleNamespace(
        id="job123",
        labels={"run": "r"},
        flavor="l4x1",
        status=SimpleNamespace(stage="RUNNING"),
        created_at=datetime(2026, 10, 28, tzinfo=UTC),
    )
    monkeypatch.setattr(cli, "list_jobs", lambda **k: [job])
    result = runner.invoke(cli.app, ["jobs", "list"])
    assert result.exit_code == 0 and all(
        t in result.output for t in ("job123", "r", "l4x1", "RUNNING")
    )


def test_status_shows_progress_and_cost(monkeypatch):
    status = JobStatus(
        id="job123",
        url="https://huggingface.co/jobs/me/job123",
        run="r",
        flavor="l4x1",
        stage="RUNNING",
        message=None,
        hours=0.5,
        cost=0.4,
        progress=[
            {"event": "train", "step": 20, "loss": 0.5, "tokens_per_sec": 400.0, "eta_s": 600.0},
            {"event": "eval", "step": 20, "exact_match": 0.4, "mean_macro_f1": 0.7},
        ],
    )
    monkeypatch.setattr(cli, "job_status", lambda job_id, **k: status)
    result = runner.invoke(cli.app, ["jobs", "status", "job123"])
    assert result.exit_code == 0
    assert all(t in result.output for t in ("RUNNING", "$0.40", "step 20", "ETA 10 min", "0.400"))


def test_logs_print_lines_verbatim(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        cli, "job_logs", lambda job_id, follow, **k: seen.update(follow=follow) or iter(["[red]x"])
    )
    result = runner.invoke(cli.app, ["jobs", "logs", "job123", "-f"])
    assert result.exit_code == 0 and "[red]x" in result.output and seen == {"follow": True}


def test_cancel_and_resume(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "cancel_job", lambda job_id, **k: calls.append(("cancel", job_id)))
    monkeypatch.setattr(
        cli,
        "resume_job",
        lambda job_id, timeout=None, **k: (
            calls.append(("resume", job_id, timeout))
            or SimpleNamespace(id="job456", url="https://huggingface.co/jobs/me/job456")
        ),
    )
    assert runner.invoke(cli.app, ["jobs", "cancel", "job123"]).exit_code == 0
    result = runner.invoke(cli.app, ["jobs", "resume", "job123", "--timeout", "3h"])
    assert result.exit_code == 0 and "job456" in result.output
    assert calls == [("cancel", "job123"), ("resume", "job123", "3h")]


def test_cancel_says_to_wait_before_resuming(monkeypatch):
    monkeypatch.setattr(cli, "cancel_job", lambda job_id, **k: None)
    result = runner.invoke(cli.app, ["jobs", "cancel", "job123"])
    assert result.exit_code == 0 and "a minute" in result.output and "jobs resume" in result.output


def test_clean_reports_what_it_removed(monkeypatch):
    monkeypatch.setattr(cli, "clean_job", lambda job_id, **k: ["a/bundle/x", "a/out/checkpoints/y"])
    result = runner.invoke(cli.app, ["jobs", "clean", "job123"])
    assert result.exit_code == 0 and "Removed 2 leftover files" in result.output
    monkeypatch.setattr(cli, "clean_job", lambda job_id, **k: [])
    assert "Nothing to clean" in runner.invoke(cli.app, ["jobs", "clean", "job123"]).output


def test_status_of_a_completed_job_points_to_clean(monkeypatch):
    status = JobStatus(
        id="job123",
        url="u",
        run="r",
        flavor="l4x1",
        stage="COMPLETED",
        message=None,
        hours=0.2,
        cost=0.16,
        progress=[],
    )
    monkeypatch.setattr(cli, "job_status", lambda job_id, **k: status)
    monkeypatch.setattr(cli, "leftover_files", lambda job_id, **k: ["a/out/checkpoints/best/w"])
    result = runner.invoke(cli.app, ["jobs", "status", "job123"])
    assert result.exit_code == 0 and "clef-tune jobs clean job123" in result.output
