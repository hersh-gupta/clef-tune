from PIL import Image

from clef_tune.validation import validate
from tests.helpers import CharCounter, make_config, ticket_rows, write_csv


def config_for(tmp_path, rows, **data):
    write_csv(tmp_path / "train.csv", rows)
    return make_config(tmp_path, data={"train": "train.csv", "id_column": "id", **data})


def messages(issues):
    return [str(issue) for issue in issues]


def test_valid_dataset(tmp_path):
    report = validate(config_for(tmp_path, ticket_rows(40)), CharCounter())
    assert report.ok, messages(report.errors)
    assert len(report.train) == 36 and len(report.eval) == 4
    dept = report.questions["department"]
    assert dept.train_counts == {"NA": 0, "billing": 18, "technical": 18}
    assert dept.eval_labeled == 4
    assert report.tokens is not None and report.tokens.max >= report.tokens.p95 >= report.tokens.p50
    assert report.tokens.train_tokens == sum(report.tokens.train_lengths)
    assert "subject" in report.columns
    assert any("calibration will be skipped" in m for m in messages(report.warnings))


def test_single_class_question_is_an_error(tmp_path):
    rows = ticket_rows(20)
    for row in rows:
        row["outage"] = "no"
    report = validate(config_for(tmp_path, rows), CharCounter())
    assert any(m.startswith("outage: needs at least 2 classes") for m in messages(report.errors))


def test_rare_class_warning(tmp_path):
    rows = ticket_rows(60)
    for row in rows:
        row["outage"] = "no"
    rows[0]["outage"] = "yes"
    rows[1]["outage"] = "yes"  # 2 of 60 training rows (3%)
    write_csv(tmp_path / "eval.csv", ticket_rows(10, start=100))
    report = validate(config_for(tmp_path, rows, eval="eval.csv"), CharCounter())
    assert any("rare classes" in m and "class_weighting" in m for m in messages(report.warnings))


def test_duplicate_ids_are_an_error(tmp_path):
    rows = ticket_rows(20)
    rows[5]["id"] = rows[4]["id"]
    report = validate(config_for(tmp_path, rows), CharCounter())
    assert any("duplicate ID" in m for m in messages(report.errors))


def test_eval_file_is_used_and_missing_class_warns(tmp_path):
    write_csv(
        tmp_path / "eval.csv",
        [r for r in ticket_rows(10, start=100) if r["department"] == "billing"],
    )
    report = validate(config_for(tmp_path, ticket_rows(40), eval="eval.csv"), CharCounter())
    assert len(report.train) == 40 and len(report.eval) == 5
    assert any(
        m.startswith("department: eval set has no examples of: technical")
        for m in messages(report.warnings)
    )


def test_schema_too_long_is_an_error(tmp_path):
    cfg = config_for(tmp_path, ticket_rows(20))
    cfg.train.max_length = 256
    report = validate(cfg, CharCounter())
    assert any("max_length is 256" in m for m in messages(report.errors))


def test_truncation_warning(tmp_path):
    rows = ticket_rows(20)
    rows[3]["body"] = "x" * 5000
    report = validate(config_for(tmp_path, rows), CharCounter())
    assert report.tokens.truncated_rows == 1
    assert any("will be truncated" in m for m in messages(report.warnings))


def test_images_missing_and_present(tmp_path):
    rows = ticket_rows(20)
    for row in rows:
        row["receipt"] = ""
    Image.new("RGB", (1024, 1024)).save(tmp_path / "ok.png")
    rows[0]["receipt"] = "ok.png"
    rows[1]["receipt"] = "missing.png"
    rows[2]["receipt"] = "https://example.com/r.png"
    report = validate(config_for(tmp_path, rows, image_columns=["receipt"]), CharCounter())
    assert any("image file not found" in m and "rows 3" in m for m in messages(report.errors))
    assert any("not checked" in m for m in messages(report.warnings))


def test_without_counter_tokens_are_skipped(tmp_path):
    report = validate(config_for(tmp_path, ticket_rows(20)), None)
    assert report.tokens is None
    assert any("tokenizer unavailable" in m for m in messages(report.warnings))


def test_missing_data_file_and_bad_schema_file(tmp_path):
    cfg = make_config(tmp_path, data={"train": "nope.csv"})
    assert any("file not found" in m for m in messages(validate(cfg, None).errors))
    (tmp_path / "s.yaml").write_text("questions: {q: {type: bogus}}\n", encoding="utf-8")
    cfg = make_config(tmp_path, schema=None, data={"train": "x.csv"}, schema_file="s.yaml")
    assert any(m.startswith("schema:") for m in messages(validate(cfg, None).errors))


def test_too_few_rows(tmp_path):
    report = validate(config_for(tmp_path, ticket_rows(1)), CharCounter())
    assert any("eval set is empty" in m for m in messages(report.errors))


def test_schema_file_with_yaml_syntax_error_is_an_issue(tmp_path):
    (tmp_path / "bad.yaml").write_text("questions: {q: [unclosed\n", encoding="utf-8")
    cfg = make_config(tmp_path, schema=None, data={"train": "x.csv"}, schema_file="bad.yaml")
    errors = messages(validate(cfg, None).errors)
    assert len(errors) == 1 and errors[0].startswith("schema:") and "YAML" in errors[0]


def test_question_weights_for_unknown_questions_are_a_warning(tmp_path):
    cfg = config_for(tmp_path, ticket_rows(40))
    cfg.train.question_weights = {"departmnt": 2.0, "outage": 0.5}
    report = validate(cfg, CharCounter())
    assert report.ok
    warning = [m for m in messages(report.warnings) if "question_weights" in m]
    assert len(warning) == 1 and "departmnt" in warning[0] and "outage" not in warning[0]
