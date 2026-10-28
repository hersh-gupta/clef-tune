import zipfile

import pytest

from clef_tune.ui import handlers
from clef_tune.ui.session import Session
from clef_tune.ui.settings import Settings
from tests.helpers import ticket_rows, write_csv
from tests.ui.fakes import fake_services

LOCAL = Settings(space=False, max_upload_mb=200, local_training=False)


@pytest.fixture
def services():
    return fake_services()[0]


@pytest.fixture
def session():
    s = Session()
    handlers.example(s)
    yield s
    s.close()


def test_the_preview_is_paginated_and_cells_are_shortened(tmp_path):
    rows = ticket_rows(120)
    rows[0]["body"] = "x" * 500
    s = Session()
    handlers.upload(s, str(write_csv(tmp_path / "t.csv", rows)), LOCAL)
    first, second = handlers.data_view(s, 0), handlers.data_view(s, 1)
    assert len(second.rows) == 50 and second.page_label == "Rows 51–100 of 120"
    assert len(first.rows[0][2]) == handlers.CELL_CHARS + 1 and first.rows[0][2].endswith("…")
    assert handlers.data_view(s, 99).page == 2
    s.close()


def test_upload_errors_are_shown_not_raised(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("hi")
    s = Session()
    view = handlers.upload(s, str(bad), LOCAL)
    assert "unsupported file type" in view.error and s.table is None
    s.close()


def test_schema_cards_count_labels_per_option(session):
    cards = {card.id: card for card in handlers.schema_view(session).cards}
    department = cards["department"]
    assert sum(row[2] for row in department.options) == 200
    assert department.unknown == ""


def test_values_that_are_not_options_are_listed(session):
    card = handlers.schema_view(session).cards[0]
    kept = [row for row in card.options if row[0] != "account"]
    handlers.update_question(session, "department", options=kept)
    unknown = handlers.schema_view(session).cards[0].unknown
    assert "account" in unknown and "not an option" in unknown


def test_prompt_preview_escapes_the_row(tmp_path):
    rows = ticket_rows(30)
    rows[0]["subject"] = "<script>alert(1)</script>"
    s = Session()
    handlers.upload(s, str(write_csv(tmp_path / "t.csv", rows)), LOCAL)
    preview = handlers.schema_view(s, preview_row=0).preview
    assert "&lt;script&gt;" in preview and "<script>" not in preview
    assert 'class="pp-state"' in preview
    s.close()


def test_refresh_validates_and_estimates(session, services):
    handlers.refresh(session, services)
    view = handlers.validate_view(session)
    assert view.ok and "Train rows" in view.tiles and 'class="ct-cutoff"' in view.histogram
    assert session.estimates and "$" in handlers.sidebar_estimate(session)
    assert handlers.step_states(session)[:3] in (
        ["done", "done", "done"],
        ["done", "done", "warning"],
    )


def test_a_broken_schema_is_an_error_not_a_crash(session, services):
    handlers.update_question(session, "department", options=[])
    handlers.refresh(session, services)
    assert not handlers.validate_view(session).ok
    assert "criteria" in handlers.validate_view(session).issues
    assert handlers.step_states(session)[1] == "error"


def test_a_tokenizer_failure_says_why(session):
    def offline(config):
        raise OSError("hub unreachable")

    handlers.refresh(session, fake_services(counter=offline)[0])
    assert "hub unreachable" in handlers.validate_view(session).issues


def test_presets_and_advanced_fields(session, services):
    handlers.set_preset(session, "thorough")
    assert handlers.configure_view(session).values["train.epochs"] == 5
    assert handlers.set_field(session, "train.epochs", 2) is None
    assert handlers.configure_view(session).values["train.epochs"] == 2
    handlers.set_preset(session, "quick")
    assert handlers.configure_view(session).values["train.epochs"] == 1
    error = handlers.set_field(session, "train.max_length", 100)
    assert "max_length" in error
    assert handlers.configure_view(session).values["train.max_length"] == 2048


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
