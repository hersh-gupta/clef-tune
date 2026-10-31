import pytest

from clef_tune.config import load_config
from clef_tune.predictions import check_input, prediction_table, summarize
from clef_tune.ui import handlers, views
from clef_tune.ui.session import Session
from clef_tune.ui.settings import Settings
from tests.helpers import CharCounter, ticket_rows, write_csv
from tests.ui.fakes import EXAMPLE_CONFIG, fake_services

LOCAL = Settings(space=False, max_upload_mb=200, local_training=False)


@pytest.fixture
def services():
    return fake_services()[0]


@pytest.fixture
def session(services):
    s = Session()
    handlers.sample(s)
    handlers.refresh(s, services)
    yield s
    s.close()


def test_the_step_bar_unlocks_as_the_data_gets_ready(session):
    empty = views.step_bar(Session(), "data")
    assert [i.state for i in empty] == ["now", "locked", "locked", "locked"]
    assert [i.label for i in empty] == ["Your data", "What to predict", "Settings", "Train"]
    ready = views.step_bar(session, "settings")
    assert [i.state for i in ready] == ["done", "done", "now", "todo"]
    handlers.set_role(session, "department", "skip")
    handlers.set_role(session, "urgency", "skip")
    handlers.set_role(session, "outage", "skip")
    assert [i.state for i in views.step_bar(session, "data")] == ["now", "todo", "locked", "locked"]


def test_the_data_page_before_and_after_a_file(session):
    empty = views.data_page(Session(), LOCAL)
    assert not empty.has_data and empty.max_mb == 200
    assert [label for _, label in empty.samples] == [
        "support tickets",
        "claim triage",
        "server logs",
    ]
    page = views.data_page(session, LOCAL)
    assert (page.file_name, page.rows, page.columns) == ("support_tickets.csv", 200, 7)
    assert page.noun == "ticket" and page.noun_guessed
    assert dict(page.headers)["department"] == "question" and dict(page.headers)["body"] == "state"
    assert len(page.preview) == 4 and page.summary == "3 expected outputs · 3 columns read"
    names = {row.name: row.role_name for row in page.column_rows}
    assert names["ticket_id"] == "Row ID" and names["outage"] == "Expected output"
    assert [c.text for c in page.checks if c.level == "ok"] == [
        "All 200 rows read, with 7 columns",
        "Every row has its own ID",
        "Every row is short enough for the model to read in full",
    ]
    assert not page.blocking


def test_a_renamed_noun_is_no_longer_marked_as_guessed(session):
    handlers.set_noun(session, "case")
    page = views.data_page(session, LOCAL)
    assert page.noun == "case" and not page.noun_guessed
    handlers.set_noun(session, "  ")
    assert views.data_page(session, LOCAL).noun == ""


def test_without_an_expected_output_continue_is_blocked(session, services):
    for column in ("department", "urgency", "outage"):
        handlers.set_role(session, column, "skip")
    handlers.refresh(session, services)
    page = views.data_page(session, LOCAL)
    assert page.blocking
    assert any(c.text == "Choose at least one expected output" for c in page.checks)


def test_empty_answers_are_flagged_under_their_column(tmp_path, services):
    rows = ticket_rows(40)
    for row in rows[:3]:
        row["urgency"] = ""
    s = Session()
    handlers.upload(s, write_csv(tmp_path / "tickets.csv", rows), LOCAL)
    handlers.refresh(s, services)
    flags = {row.name: row.flags for row in views.data_page(s, LOCAL).column_rows}
    assert any(flag.startswith("Empty in") for flag in flags["urgency"])
    s.close()


def test_the_question_editor_counts_answers_and_explains_types(session):
    page = views.questions_page(session)
    assert [i.id for i in page.items] == ["department", "urgency", "outage"]
    assert page.items[0].ask == "Which team should handle this ticket?"
    editor = page.selected
    assert editor.id == "department" and editor.counted == "3 answers in your data"
    assert sum(a.count for a in editor.answers) == 200 and max(a.bar for a in editor.answers) == 1
    assert page.reads == ["subject", "body", "customer_tier"] and page.ignores == []
    assert views.questions_page(session, "urgency").selected.hint.startswith("Lowest first")
    assert "optional" in views.questions_page(session, "outage").selected.hint


def test_answers_missing_from_the_list_are_flagged(session):
    question = session.draft.question("department")
    question.options = [o for o in question.options if o.id != "account"]
    flags = views.questions_page(session, "department").selected.flags
    assert any("isn't in this list: account" in flag for flag in flags)


def test_the_prompt_preview_marks_each_part(session):
    kinds = {kind for kind, _ in views.prompt_preview(session)}
    assert {"prefix", "state", "question", "option", "suffix"} <= kinds


def test_settings_show_presets_with_prices_and_a_capped_estimate(session, services):
    page = views.settings_page(session, services, "me")
    assert page.repo == "me/support-tickets" and page.model.name == "Clef-Flash"
    assert page.preset.key == "balanced" and page.presets[1].recommended
    assert all(choice.extra.startswith("About ") for choice in page.presets)
    assert page.estimate.cap_cost == "$0.90" and page.estimate.cap_time == "30 minutes"
    assert page.estimate.hardware == "one L40S GPU"
    assert (page.estimate.test_rows, page.estimate.total_rows) == (31, 200)
    assert page.advanced[0].label == "Passes over your data"
    assert any(row.recommended for row in page.flavors)


def test_a_chosen_machine_changes_the_estimate(session, services):
    session.flavor = "a100-large"
    page = views.settings_page(session, services, None)
    assert page.estimate.hardware == "one A100 80GB GPU" and page.repo == "support-tickets"


def test_predicting_checks_the_file_and_summarizes_the_result(tmp_path):
    config = load_config(EXAMPLE_CONFIG)
    rows = [{"subject": "s", "body": "b", "customer_tier": "pro"} for _ in range(3)]
    path = write_csv(tmp_path / "new.csv", rows)
    draft = views.PredictionDraft("job-b", "me/m", "support-tickets", config, "new.csv", path)
    draft.check = check_input(config, path, CharCounter())
    page = views.predict_page(draft)
    assert page.has_file and page.rows == 3 and page.can_check_cost and page.threshold == 70
    assert page.checks[0].text == "Found subject, body and customer_tier"
    assert any(c.text == "No ID column, so rows are numbered from 1" for c in page.checks)
    header, table = prediction_table(
        ["subject"],
        [(2, {"subject": "a"}), (3, {"subject": "b"})],
        {
            2: {"department": {"account": 0.1, "billing": 0.9, "technical": 0.0}},
            3: {"department": {"account": 0.5, "billing": 0.3, "technical": 0.2}},
        },
        {"department": ["account", "billing", "technical"]},
    )
    result = views.prediction_result(summarize(header, table, config), config, threshold=0.7)
    assert result.review == 1 and result.noun == "ticket"
    assert result.added == {"department", "department_confidence", "review"}
    assert (1, "department_confidence") in result.low
    assert result.shares[0].ask == "Which team should handle this ticket?"
    assert result.shares[0].parts == [("account", 50), ("billing", 50)]


def test_settings_hold_out_a_share_set_in_the_sentence(session, services):
    page = views.settings_page(session, services, "me")
    assert (page.holdout.percent, page.holdout.test_rows, page.holdout.total_rows) == (
        "15",
        31,
        200,
    )
    assert "data.eval_fraction" not in [row.path for row in page.advanced]


def test_every_advanced_setting_has_a_tip_and_a_range(session, services):
    page = views.settings_page(session, services, "me")
    for row in page.advanced:
        assert row.tip, row.path
        if row.kind in ("int", "float", "optional_int"):
            assert row.range, row.path
    rows = {row.path: row for row in page.advanced}
    assert rows["train.lora.r"].range == "Usually 8–64"
    assert "Allowed: 256 to 16,384." in rows["train.max_length"].tip
    assert "Default: 4,096." in rows["train.max_length"].tip
    assert rows["eval.eval_every"].range.startswith("This run has about ")


def test_allowed_values_are_read_from_the_config():
    assert views.allowed("train.epochs") == "1 or more"
    assert views.allowed("train.lora.dropout") == "0 to just under 1"
    assert views.allowed("train.lora_lr") == "more than 0"
    assert views.allowed("train.max_length") == "256 to 16,384"
    assert views.allowed("eval.early_stopping_patience") == "1 or more, or empty"
