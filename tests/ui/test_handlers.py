import zipfile

import pytest

from clef_tune.ui import handlers
from clef_tune.ui.session import Session
from clef_tune.ui.settings import Settings
from tests.ui.fakes import fake_services

LOCAL = Settings(space=False, max_upload_mb=200, local_training=False)


@pytest.fixture
def services():
    return fake_services()[0]


@pytest.fixture
def session():
    s = Session()
    handlers.sample(s)
    yield s
    s.close()


def test_upload_errors_are_returned_not_raised(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("hi")
    s = Session()
    error = handlers.upload(s, str(bad), LOCAL)
    assert "Could not use notes.txt" in error and "unsupported file type" in error
    assert s.table is None
    s.close()


def test_refresh_validates_and_estimates(session, services):
    handlers.refresh(session, services)
    assert session.report.ok and session.report.tokens is not None and session.estimates


def test_a_broken_schema_is_an_error_not_a_crash(session, services):
    handlers.update_question(session, "department", options=[])
    handlers.refresh(session, services)
    assert not session.report.ok
    assert any("criteria" in issue.message for issue in session.report.errors)


def test_a_tokenizer_failure_says_why(session):
    def offline(config):
        raise OSError("hub unreachable")

    handlers.refresh(session, fake_services(counter=offline)[0])
    assert any("hub unreachable" in issue.message for issue in session.report.issues)


def test_presets_and_advanced_fields(session):
    handlers.set_preset(session, "thorough")
    assert session.config().train.epochs == 5
    assert handlers.set_field(session, "train.epochs", 2) is None
    assert session.config().train.epochs == 2
    handlers.set_preset(session, "quick")
    assert session.config().train.epochs == 1
    assert "max_length" in handlers.set_field(session, "train.max_length", 100)
    assert session.config().train.max_length == 2048


def test_every_config_field_is_editable_or_deliberately_excluded():
    from clef_tune.config import Config

    def leaves(model, prefix=""):
        for name, field in model.model_fields.items():
            key = field.alias or name
            annotation = field.annotation
            if isinstance(annotation, type) and hasattr(annotation, "model_fields"):
                yield from leaves(annotation, f"{prefix}{key}.")
            else:
                yield f"{prefix}{key}"

    editable = {spec.path for spec in handlers.ADVANCED_FIELDS} | {"train.question_weights"}
    assert set(leaves(Config)) == editable | handlers.EXCLUDED_FIELDS
    assert not editable & handlers.EXCLUDED_FIELDS


def test_the_bundle_zip_holds_a_self_contained_config(session, services):
    handlers.refresh(session, services)
    names = zipfile.ZipFile(handlers.bundle(session)).namelist()
    assert {"config.yaml", "train.jsonl", "eval.jsonl"} <= set(names)


def test_no_bundle_while_validation_has_errors(session, services):
    handlers.update_question(session, "department", options=[])
    handlers.refresh(session, services)
    with pytest.raises(ValueError, match="validation"):
        handlers.bundle(session)


def test_every_column_has_one_role_and_sample_values(session):
    rows = {row.name: row for row in handlers.column_rows(session)}
    assert {name: row.role for name, row in rows.items()} == {
        "ticket_id": "id",
        "subject": "state",
        "body": "state",
        "customer_tier": "state",
        "department": "question",
        "urgency": "question",
        "outage": "question",
    }
    assert "billing" in rows["department"].samples and len(rows["body"].samples) <= 80
    assert [name for _, name, _ in handlers.ROLE_CHOICES] == [
        "Model reads it",
        "Expected output",
        "Image",
        "Row ID",
        "Ignore",
    ]


def test_setting_a_role_moves_the_column(session):
    assert handlers.set_role(session, "customer_tier", "question") is None
    draft = session.draft
    tier = draft.question("customer_tier")
    assert "customer_tier" not in draft.state_columns
    assert tier is not None and "enterprise" in [o.id for o in tier.options]
    handlers.set_role(session, "department", "skip")
    assert draft.question("department") is None and "department" not in draft.state_columns
    handlers.set_role(session, "subject", "id")
    roles = {row.name: row.role for row in handlers.column_rows(session)}
    assert (roles["subject"], roles["ticket_id"]) == ("id", "skip")
    handlers.set_role(session, "body", "image")
    assert draft.image_columns == ["body"] and "body" not in draft.state_columns


def test_state_columns_keep_the_file_order(session):
    handlers.set_role(session, "subject", "skip")
    handlers.set_role(session, "subject", "state")
    assert session.draft.state_columns == ["subject", "body", "customer_tier"]


def test_an_unknown_role_is_refused(session):
    assert "role" in handlers.set_role(session, "subject", "label")


def test_answers_get_descriptions_and_scales_can_be_reordered(session):
    assert handlers.describe_option(session, "department", "billing", " Money ") is None
    options = {o.id: o.description for o in session.draft.question("department").options}
    assert options["billing"] == "Money"
    assert "no answer" in handlers.describe_option(session, "department", "nope", "x")
    handlers.move_option(session, "urgency", "2", -1)
    assert [o.id for o in session.draft.question("urgency").options] == ["0", "2", "1"]
    handlers.move_option(session, "urgency", "0", -5)
    assert [o.id for o in session.draft.question("urgency").options][0] == "0"


def test_the_noun_is_trimmed_and_can_be_cleared(session):
    handlers.set_noun(session, " case ")
    assert session.draft.row_noun == "case"
    handlers.set_noun(session, "")
    assert session.draft.row_noun is None
