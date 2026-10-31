import os
import zipfile

import pytest

from clef_tune.ui.session import (
    SAMPLES,
    OptionDraft,
    QuestionDraft,
    Session,
    UploadError,
    clear_data,
    extract_zip,
    load_example,
    load_sample,
    load_upload,
    slug,
    sweep_stale,
)
from clef_tune.ui.settings import MOCK_OAUTH_TOKEN, Settings, SignInRequired, user_token
from tests.helpers import ticket_rows, write_csv

LOCAL = Settings(space=False, max_upload_mb=200, local_training=False)
SPACE = Settings(space=True, max_upload_mb=200, local_training=False)


@pytest.fixture
def session():
    s = Session()
    yield s
    s.close()


def test_settings_read_space_mode_and_the_upload_cap():
    def no_cuda():
        pytest.fail("checked for CUDA on a Space")

    env = {"SPACE_ID": "me/clef-tune", "CLEF_TUNE_MAX_UPLOAD_MB": "50"}
    assert Settings.from_env(env, cuda=no_cuda) == Settings(True, 50, False)
    assert Settings.from_env({}, cuda=lambda: True) == Settings(False, 200, True)
    for bad in ("lots", "0"):
        with pytest.raises(ValueError, match="CLEF_TUNE_MAX_UPLOAD_MB"):
            Settings.from_env({"CLEF_TUNE_MAX_UPLOAD_MB": bad}, cuda=lambda: False)


def test_a_space_never_uses_the_server_token():
    with pytest.raises(SignInRequired, match="Sign in"):
        user_token(SPACE, None, " ", local_token=lambda: "hf_server")
    assert user_token(SPACE, "hf_oauth", None, local_token=lambda: "hf_server") == "hf_oauth"
    assert user_token(SPACE, "hf_oauth", " hf_pasted ") == "hf_pasted"


def test_locally_the_mock_sign_in_is_ignored():
    assert user_token(LOCAL, MOCK_OAUTH_TOKEN, None, local_token=lambda: "hf_local") == "hf_local"
    with pytest.raises(SignInRequired, match="hf auth login"):
        user_token(LOCAL, None, None, local_token=lambda: None)


def test_an_upload_becomes_a_draft(session, tmp_path):
    path = write_csv(tmp_path / "tickets.csv", ticket_rows(30))
    load_upload(session, path, LOCAL)
    draft = session.draft
    assert session.table.columns == ["id", "subject", "body", "department", "urgency", "outage"]
    assert (draft.run_name, draft.id_column, draft.state_columns) == (
        "tickets",
        "id",
        ["subject", "body"],
    )
    assert [(q.id, q.type) for q in draft.questions] == [
        ("department", "choice"),
        ("urgency", "choice"),
        ("outage", "noul"),
    ]
    assert session.reached == 1 and session.table.path.is_relative_to(session.workdir())


def test_run_names_are_made_valid():
    assert slug("My Data (v2)") == "My-Data-v2"
    assert slug("___") == "run" and slug("é") == "run"
    assert len(slug("x" * 100)) == 64


def test_load_example_uses_the_example_config(session):
    load_example(session)
    draft = session.draft
    assert [q.id for q in draft.questions] == ["department", "urgency", "outage"]
    assert draft.questions[1].type == "score" and draft.id_column == "ticket_id"
    assert draft.model_id == "Cloudflare/clef-flash"
    assert len(session.table.rows) == 200


def test_a_zip_keeps_its_folders_so_image_paths_resolve(session, tmp_path):
    archive = tmp_path / "upload.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("set/tickets.csv", "id,text,label,img\n1,a,x,img/a.png\n2,b,y,img/b.png\n")
        z.writestr("set/img/a.png", b"png")
        z.writestr("set/img/b.png", b"png")
        z.writestr("__MACOSX/set/._tickets.csv", b"junk")
    load_upload(session, archive, LOCAL)
    assert session.table.path.name == "tickets.csv"
    assert (session.table.path.parent / "img" / "a.png").exists()


@pytest.mark.parametrize("name", ["../evil.csv", "/abs/evil.csv", "a/../../evil.csv"])
def test_hostile_zips_are_refused(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(name, "id,x\n1,2\n")
    dest = tmp_path / "out" / "dest"
    with pytest.raises(UploadError, match="outside"):
        extract_zip(archive, dest, max_bytes=10_000)
    assert not (tmp_path / "evil.csv").exists() and not (tmp_path / "out" / "evil.csv").exists()


def test_zips_that_expand_too_far_or_hold_the_wrong_files_are_refused(tmp_path):
    bomb = tmp_path / "bomb.zip"
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("big.csv", "0" * 50_000)
    with pytest.raises(UploadError, match="expands to more than"):
        extract_zip(bomb, tmp_path / "a", max_bytes=10_000)
    for names in ([], ["a.csv", "b.jsonl"]):
        archive = tmp_path / f"n{len(names)}.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("readme.txt", "hi")
            for name in names:
                z.writestr(name, "id,x\n1,2\n")
        with pytest.raises(UploadError, match="exactly one .csv, .tsv, .xlsx or .jsonl"):
            extract_zip(archive, tmp_path / f"d{len(names)}", max_bytes=10_000)


def test_an_upload_over_the_cap_is_refused(session, tmp_path):
    path = tmp_path / "big.csv"
    path.write_bytes(b"id,x\n" + b"1,2\n" * 400_000)  # 1.6 MB
    with pytest.raises(UploadError, match="limit is 1 MB"):
        load_upload(session, path, Settings(space=True, max_upload_mb=1, local_training=False))
    assert session.table is None


def test_a_file_that_cannot_be_read_is_an_upload_error(session, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    with pytest.raises(UploadError, match="unsupported file type"):
        load_upload(session, path, LOCAL)


def test_sessions_are_isolated_and_closed():
    first, second = Session(), Session()
    assert first.workdir() != second.workdir() and first.workdir().exists()
    first.close()
    first.close()
    assert not first.root.exists() and second.workdir().exists()
    second.close()


def test_stale_session_folders_are_swept(tmp_path):
    old, fresh = tmp_path / "old", tmp_path / "fresh"
    old.mkdir(), fresh.mkdir()
    os.utime(old, (1_000, 1_000))
    assert sweep_stale(tmp_path, max_age_hours=24, now=1_000 + 25 * 3600) == 1
    assert not old.exists() and fresh.exists()


def test_the_draft_builds_a_config(session):
    load_example(session)
    draft = session.draft
    draft.preset, draft.overrides = "thorough", {"train.epochs": 2}
    draft.questions[0].id = "team"  # column stays "department"
    config = draft.to_config(session.workdir(), "data/support_tickets.csv")
    assert config.data.label_columns == {"team": "department"}
    assert (config.train.lora.r, config.train.epochs) == (32, 2)
    schema = config.get_schema()
    assert schema.questions["urgency"].criteria == ["Can wait", "This week", "Today"]


def test_switching_types_keeps_sensible_options():
    options = [OptionDraft(i) for i in "201"]
    question = QuestionDraft("urgency", "urgency", "choice", options=options)
    question.set_type("score", observed=["0", "1", "2"])
    assert [o.id for o in question.options] == ["0", "1", "2"]
    question.set_type("noul", observed=["0", "1", "2"])
    assert [o.id for o in question.options] == ["true", "false"]
    question.set_type("choice", observed=["a", "b"])
    assert [o.id for o in question.options] == ["a", "b"]


def test_the_fingerprint_changes_with_any_edit(session):
    load_example(session)
    before = session.draft.fingerprint()
    session.draft.questions[0].instructions = "Which team?"
    assert session.draft.fingerprint() != before


def test_an_upload_guesses_its_row_noun(session, tmp_path):
    load_upload(session, write_csv(tmp_path / "tickets.csv", ticket_rows(30)), LOCAL)
    assert session.draft.row_noun == "ticket"
    assert session.config().data.row_noun == "ticket"
    load_upload(session, write_csv(tmp_path / "data.csv", ticket_rows(30)), LOCAL)
    assert session.draft.row_noun is None and session.config().data.row_noun is None


@pytest.mark.parametrize(
    ("name", "noun", "first"),
    [
        ("support_tickets", "ticket", "department"),
        ("claims_triage", "claim", "reviewer"),
        ("server_logs", "log line", "component"),
    ],
)
def test_each_sample_loads_with_its_noun(session, name, noun, first):
    load_sample(session, name)
    assert session.draft.row_noun == noun
    assert session.draft.questions[0].id == first and len(session.draft.questions) == 3
    assert len(session.table.rows) == 200 and name in SAMPLES


def test_an_unknown_sample_is_refused(session):
    with pytest.raises(ValueError, match="no sample"):
        load_sample(session, "nope")


def test_excel_uploads_and_zipped_excel_work(session, tmp_path):
    from openpyxl import Workbook

    book = Workbook()
    rows = ticket_rows(30)
    book.active.append(list(rows[0]))
    for row in rows:
        book.active.append(list(row.values()))
    book.save(tmp_path / "claims.xlsx")
    load_upload(session, tmp_path / "claims.xlsx", LOCAL)
    assert len(session.table.rows) == 30 and session.draft.row_noun == "claim"
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.write(tmp_path / "claims.xlsx", "set/claims.xlsx")
    load_upload(session, archive, LOCAL)
    assert session.table.path.name == "claims.xlsx"


def test_removing_the_data_starts_the_first_step_over(session):
    load_sample(session, "claims_triage")
    folder = session.table.path.parent
    clear_data(session)
    assert session.table is None and session.draft is None and session.report is None
    assert session.reached == 0 and not folder.exists()
    load_sample(session, "server_logs")
    assert session.table.path.name == "server_logs.csv"
