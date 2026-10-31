import pytest

from clef_tune.launch.hf_jobs import LaunchError
from clef_tune.ui import handlers, runs
from clef_tune.ui import wording as w
from clef_tune.ui.session import Session
from clef_tune.yamlio import load_yaml_file
from tests.ui.fakes import EXAMPLE_CONFIG, fake_services, job_status, recorded_metrics

TOKEN = "hf_user"


@pytest.fixture
def session():
    s = Session()
    handlers.sample(s)
    handlers.refresh(s, fake_services()[0])
    yield s
    s.close()


def review(session, services, **overrides):
    values = {
        "adapter_repo": "me/support-tickets",
        "delete_data": True,
        "model": "Clef-Flash",
        "preset": "Balanced",
    }
    return runs.review_launch(session, services, token=TOKEN, **(values | overrides))


def test_review_shows_time_cost_cap_and_the_split(session):
    services, calls = fake_services(plan_warning="could not confirm write access to me/x")
    ready = review(session, services)
    assert ready.can_start and ready.cap_cost == "$0.80" and ready.cap_time == "1 hour"
    assert ready.time == "21–45 minutes" and ready.cost == "$0.28–$0.60"
    assert ready.data_line == "200 tickets · 3 questions"
    assert ready.split_line == "169 to learn from, 31 set aside to test it"
    assert ready.first_build and ready.notes == ["could not confirm write access to me/x"]
    request = next(c for c in calls if c[0] == "plan_launch")[1][1]
    assert (request.keep_data, request.adapter_repo) == (False, "me/support-tickets")


def test_a_launch_error_is_shown_and_nothing_is_pending(session):
    services, _ = fake_services(plan_error=LaunchError("Hugging Face Jobs needs pre-paid credits"))
    ready = review(session, services)
    assert not ready.can_start and "pre-paid credits" in ready.error and session.pending is None


def test_a_stale_review_is_not_started(session):
    services, calls = fake_services()
    review(session, services)
    session.draft.questions[0].instructions = "changed"
    job, error = runs.confirm_launch(session, services, token=TOKEN)
    assert job is None and "check the details again" in error
    assert not [c for c in calls if c[0] == "submit_launch"]


def test_confirming_twice_starts_once(session):
    services, calls = fake_services()
    review(session, services)
    assert runs.confirm_launch(session, services, token=TOKEN) == ("job-new", None)
    assert runs.confirm_launch(session, services, token=TOKEN)[0] is None
    assert len([c for c in calls if c[0] == "submit_launch"]) == 1


def test_a_running_job_shows_its_stage_time_left_and_cost(session):
    services, _ = fake_services()
    state, notice = runs.job_state(
        session, services, "job-a", token=TOKEN, model="Clef-Flash", noun="ticket"
    )
    assert notice is None and state.status == "running" and state.can_stop
    assert [s.state for s in state.stages] == ["done", "done", "done", "now", "todo", "todo"]
    assert state.stages[3].detail == "Step 70 of 120"
    assert state.stages[2].detail.endswith("of the test tickets")
    assert state.time_left.startswith("About") and state.cap == "$0.80"
    assert state.spent == "$0.20" and 0.15 < state.fraction < 0.9 and state.loss


def test_a_scheduled_job_is_starting_a_machine():
    status = job_status("job-a")
    starting = runs.run_state(
        status.__class__(**{**status.__dict__, "stage": "SCHEDULING", "progress": []})
    )
    assert [s.state for s in starting.stages][:2] == ["now", "todo"]
    assert starting.stages[0].label == "Starting a machine"


def test_a_failed_job_says_so_and_can_resume():
    state = runs.run_state(job_status("job-c"))
    assert state.status == "failed" and state.can_resume and not state.can_stop
    assert "exit code 1" in state.message and "stopped" in [s.state for s in state.stages]


def test_a_failed_poll_keeps_the_last_state(session):
    services, _ = fake_services()
    good, _ = runs.job_state(session, services, "job-a", token=TOKEN, model="m", noun="ticket")

    def down(job_id, **kwargs):
        raise OSError("connection reset")

    state, notice = runs.job_state(
        session, fake_services(job_status=down)[0], "job-a", token=TOKEN, model="m", noun="ticket"
    )
    assert state is good and "connection reset" in notice


def test_a_prediction_job_has_its_own_stages():
    state = runs.run_state(job_status("job-p"), noun="ticket")
    assert state.kind == "predict" and state.status == "done"
    assert [s.label for s in state.stages] == [
        "Started a machine",
        "Read your tickets with the model",
        "Prepared your spreadsheet",
    ]


def test_results_read_like_the_design(session):
    services, calls = fake_services()
    status = job_status("job-b")
    view, notice = runs.job_results(session, services, status, token=TOKEN)
    assert notice is None and view.n == 30 and view.noun == "ticket"
    assert w.plain(view.headline) == (
        "It now matches every expected output on 24 of 30 tickets. Before training, it managed 2."
    )
    assert view.dots == (2, 24, 30)
    department = view.questions[0]
    assert department.ask == "Which team should handle this ticket?"
    assert (department.before_pct, department.after_pct) == (37, 93)
    assert (department.before_right, department.after_right) == (11, 28)
    assert w.plain(department.summary).startswith("Its 2 mistakes: one account ticket marked")
    urgency = view.questions[1]
    assert urgency.names == ["Can wait", "This week", "Today"]
    assert "leans toward the top of the scale (Today)" in w.plain(urgency.summary)
    assert "average error" in urgency.tech and "calibration error" in urgency.tech
    outage = view.questions[2]
    assert outage.names == ["yes", "no"] and "caught all 5 tickets" in w.plain(outage.summary)
    assert view.run_line == (
        "Training took 25 minutes and cost $0.34, against an estimate of $0.40."
    )
    assert 'Predictor.load("me/clef-tune-run-b")' in view.snippet
    runs.job_results(session, services, status, token=TOKEN)
    assert len([c for c in calls if c[0] == "pull_run_files"]) == 1  # read once per session


def test_many_questions_use_the_averaged_headline():
    metrics = recorded_metrics()
    final = metrics["final"]["questions"]
    for k in range(5):
        final[f"q{k}"] = final["department"]
        metrics["baseline"]["questions"][f"q{k}"] = metrics["baseline"]["questions"]["department"]
    view = runs.results_view(metrics, load_yaml_file(EXAMPLE_CONFIG), None, name="x")
    assert w.plain(view.headline).startswith("Across its 8 questions, it's now right")
    assert view.dots is None


def test_a_quick_test_says_it_barely_trained():
    config = load_yaml_file(EXAMPLE_CONFIG)
    assert runs.results_view(recorded_metrics(), config, None, name="x").quick_note is None
    config["train"] = {"mode": "head_only", "epochs": 1}
    assert runs.results_view(recorded_metrics(), config, None, name="x").quick_note == (
        "This was a quick test, so it barely trained. "
        "For a real model, train again and choose Balanced."
    )


def test_your_models_lists_every_run_with_a_plain_status():
    services, _ = fake_services()
    rows, error = runs.list_models(services, token=TOKEN)
    assert error is None
    assert [(r.id, r.kind, r.label) for r in rows] == [
        ("job-a", "train", "Training"),
        ("job-p", "predict", "Predictions ready"),
        ("job-b", "train", "Ready"),
        ("job-c", "train", "Didn't finish"),
    ]


def test_resume_shows_its_cost_before_starting():
    services, calls = fake_services()
    quote = runs.review_resume(services, "job-c", token=TOKEN)
    assert quote.can_resume and quote.cap_cost == "$1.60" and quote.cap_time == "2 hours"
    assert not [c for c in calls if c[0] == "resume_job"]
    assert runs.resume_run(services, "job-c", token=TOKEN) == ("job-resumed", None)


def test_files_can_be_deleted_and_runs_stopped():
    services, calls = fake_services()
    assert runs.delete_files(services, "job-p", token=TOKEN) == (1, None)
    assert runs.stop_run(services, "job-a", token=TOKEN) is None
    assert ("cancel_job", ("job-a",), {"token": TOKEN}) in calls


def test_a_github_check_that_could_not_run_stays_in_the_logs(session, caplog):
    from clef_tune.launch.source import SourceWarning

    unchecked = SourceWarning("could not check abc on GitHub (rate limited); launching anyway")
    services, _ = fake_services(plan_warning=unchecked)
    with caplog.at_level("WARNING"):
        ready = review(session, services)
    assert ready.can_start and ready.notes == []
    assert "could not check abc on GitHub" in caplog.text


def test_a_run_says_how_long_it_should_take():
    scheduled = {**job_status("job-a").__dict__, "stage": "SCHEDULING", "progress": []}
    status_type = job_status("job-a").__class__
    estimated = runs.run_state(status_type(**{**scheduled, "estimated_hours": 0.25}))
    assert estimated.usual == "Expected to take about 15 minutes"
    limited = runs.run_state(status_type(**scheduled))
    assert limited.usual == "It stops itself after 1 hour at the latest"
    unknown = runs.run_state(status_type(**{**scheduled, "timeout": None}))
    assert unknown.usual is None


def test_a_models_line_reads_its_run_once(session):
    services, calls = fake_services()
    line = runs.model_line(session, services, "job-b", token=TOKEN)
    assert line.startswith("It now matches every expected output on 24 of 30 tickets.")
    assert len([c for c in calls if c[0] == "job_status"]) == 1
    assert runs.model_line(session, services, "job-a", token=TOKEN).startswith("About")


def test_whether_a_model_was_calibrated():
    metrics = recorded_metrics()
    assert runs.was_calibrated(metrics)
    for entry in metrics["calibration"].values():
        entry["fitted"] = False
    assert not runs.was_calibrated(metrics)
    assert not runs.was_calibrated({"final": {}})
