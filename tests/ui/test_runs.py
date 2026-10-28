import pytest

from clef_tune.launch.hf_jobs import LaunchError
from clef_tune.ui import handlers, runs
from clef_tune.ui.session import Session
from clef_tune.ui.settings import Settings
from tests.ui.fakes import fake_services

LOCAL = Settings(space=False, max_upload_mb=200, local_training=False)
SPACE = Settings(space=True, max_upload_mb=200, local_training=False)
SIGNED_IN = {"oauth_token": None, "pasted": "hf_user"}


@pytest.fixture
def session():
    s = Session()
    handlers.example(s)
    yield s
    s.close()


def review(session, services, **overrides):
    values = {"flavor": "l4x1", "timeout": "", "adapter_repo": "", "delete_data": True}
    return runs.review_launch(session, LOCAL, services, **SIGNED_IN, **(values | overrides))


def test_space_actions_need_sign_in_and_call_nothing(session, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_server")
    services, calls = fake_services()
    anonymous = {"oauth_token": None, "pasted": ""}
    launch = runs.review_launch(
        session,
        SPACE,
        services,
        **anonymous,
        flavor=None,
        timeout="",
        adapter_repo="",
        delete_data=True,
    )
    assert "Sign in" in launch.error
    assert "Sign in" in runs.list_runs(SPACE, services, **anonymous)[1]
    session.job_id = "job-a"
    assert "Sign in" in runs.job_view(session, SPACE, services, **anonymous, with_logs=True).notice
    assert [c for c in calls if c[0] not in ("counter", "prices")] == []


def test_review_shows_the_maximum_cost_and_warnings(session):
    services, calls = fake_services(plan_warning="could not check main on GitHub")
    result = review(session, services)
    assert result.can_confirm and result.confirm_label.startswith("Launch for up to $")
    assert "could not check main on GitHub" in result.summary
    request = next(c for c in calls if c[0] == "plan_launch")[1][1]
    assert (request.flavor, request.timeout, request.keep_data) == ("l4x1", None, False)


def test_a_launch_error_is_shown_and_nothing_is_pending(session):
    error = LaunchError("Hugging Face Jobs needs pre-paid credits")
    services, _ = fake_services(plan_error=error)
    result = review(session, services)
    assert not result.can_confirm and "pre-paid credits" in result.error
    assert session.pending is None


def test_a_stale_review_is_not_launched(session):
    services, calls = fake_services()
    review(session, services)
    session.draft.questions[0].instructions = "changed"
    message = runs.confirm_launch(session, LOCAL, services, **SIGNED_IN)
    assert "review the launch again" in message
    assert not [c for c in calls if c[0] == "submit_launch"]


def test_confirming_twice_launches_once(session):
    services, calls = fake_services()
    review(session, services)
    assert "job-new" in runs.confirm_launch(session, LOCAL, services, **SIGNED_IN)
    assert "review the launch first" in runs.confirm_launch(session, LOCAL, services, **SIGNED_IN)
    assert len([c for c in calls if c[0] == "submit_launch"]) == 1
    assert session.job_id == "job-new"


def test_a_running_job_shows_progress_charts_and_cost(session):
    services, _ = fake_services()
    session.job_id = "job-a"
    view = runs.job_view(session, LOCAL, services, **SIGNED_IN, with_logs=False)
    assert view.polling and view.can_cancel and not view.can_resume
    assert "Step 70 of 120" in view.progress and "<polyline" in view.loss_chart
    assert "ct-reference" in view.eval_chart and view.results is None


def test_a_completed_job_shows_results(session):
    services, calls = fake_services()
    session.job_id = "job-b"
    view = runs.job_view(session, LOCAL, services, **SIGNED_IN, with_logs=False)
    assert not view.polling and view.results is not None
    assert view.results.heatmaps.count("<svg") == 3
    assert 'Predictor.load("me/clef-tune-run-b")' in view.results.snippet
    runs.job_view(session, LOCAL, services, **SIGNED_IN, with_logs=False)
    assert len([c for c in calls if c[0] == "pull_metrics"]) == 1  # cached in the session


def test_log_tail_is_escaped(session):
    services, _ = fake_services()
    session.job_id = "job-a"
    logs = runs.job_view(session, LOCAL, services, **SIGNED_IN, with_logs=True).logs
    assert "&lt;b&gt;line 2&lt;/b&gt;" in logs and "<b>" not in logs


def test_a_failed_poll_keeps_the_last_view_and_polling(session):
    services, _ = fake_services()
    session.job_id = "job-a"
    good = runs.job_view(session, LOCAL, services, **SIGNED_IN, with_logs=False)

    def down(job_id, **kwargs):
        raise OSError("connection reset")

    failing, _ = fake_services(job_status=down)
    view = runs.job_view(session, LOCAL, failing, **SIGNED_IN, with_logs=False)
    assert view.polling and "connection reset" in view.notice
    assert view.progress == good.progress


def test_resume_shows_its_cost_before_starting(session):
    services, calls = fake_services()
    session.job_id = "job-c"
    quote = runs.review_resume(session, LOCAL, services, **SIGNED_IN, timeout="")
    assert quote.confirm_label == "Resume for up to $1.60"
    assert not [c for c in calls if c[0] == "resume_job"]
    runs.resume_run(session, LOCAL, services, **SIGNED_IN, timeout="")
    assert session.job_id == "job-resumed"
