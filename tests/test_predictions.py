import csv
import json

import pytest

from clef_tune import predictions as p
from tests.helpers import SCHEMA, CharCounter, make_config, ticket_rows, write_csv

OPTIONS = {
    "department": ["NA", "billing", "technical"],
    "urgency": ["0", "1", "2"],
    "outage": ["true", "false"],
}
ROWS = [
    (2, {"id": "1", "subject": "Card declined", "department": "billing"}),
    (3, {"id": "2", "subject": "API down", "department": ""}),
    (5, {"id": "3", "subject": "???", "department": ""}),
]
ANSWERS = {
    2: {
        "department": {"NA": 0.01, "billing": 0.97, "technical": 0.02},
        "urgency": {"0": 0.1, "1": 0.81, "2": 0.09},
        "outage": {"true": 0.01, "false": 0.99},
    },
    3: {
        "department": {"NA": 0.0, "billing": 0.06, "technical": 0.94},
        "urgency": {"0": 0.0, "1": 0.28, "2": 0.72},
        "outage": {"true": 0.64, "false": 0.36},
    },
}
COLUMNS = ["id", "subject", "department"]


def table(**kwargs):
    return p.prediction_table(COLUMNS, ROWS, ANSWERS, OPTIONS, review_below=0.7, **kwargs)


def test_the_output_keeps_the_columns_and_adds_answers_confidence_and_review():
    header, rows = table()
    assert header == [
        "id",
        "subject",
        "department",
        "department_predicted",
        "department_confidence",
        "urgency",
        "urgency_confidence",
        "outage",
        "outage_confidence",
        "review",
    ]
    assert rows[0] == [
        "1",
        "Card declined",
        "billing",
        "billing",
        "0.97",
        "1",
        "0.81",
        "no",
        "0.99",
        "",
    ]
    assert rows[1] == ["2", "API down", "", "technical", "0.94", "2", "0.72", "yes", "0.64", "yes"]
    assert rows[2] == ["3", "???", "", "", "", "", "", "", "", ""]


def test_every_answers_probability_is_optional():
    header, rows = table(probabilities=True)
    assert header[4:8] == [
        "department_confidence",
        "department.p.NA",
        "department.p.billing",
        "department.p.technical",
    ]
    assert rows[0][5:8] == ["0.01", "0.97", "0.02"]
    assert "outage.p.true" in header and "outage.p.yes" not in header


def test_tables_are_written_as_csv_or_excel(tmp_path):
    header, rows = table()
    out = p.write_table(header, rows, tmp_path / "out.csv")
    raw = out.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # Excel opens UTF-8 CSV with a BOM correctly
    with open(out, encoding="utf-8-sig", newline="") as handle:
        assert list(csv.reader(handle))[1] == rows[0]
    from openpyxl import load_workbook

    book = load_workbook(p.write_table(header, rows, tmp_path / "out.xlsx"))
    sheet = book.active
    assert [c.value for c in sheet[1]] == header
    assert sheet["E2"].value == 0.97 and sheet["A2"].value == "1"
    with pytest.raises(ValueError, match=r"\.txt.*\.jsonl"):
        p.write_table(header, rows, tmp_path / "out.txt")


def test_tables_are_written_as_tsv_json_lines_or_json(tmp_path):
    header, rows = table()
    with open(p.write_table(header, rows, tmp_path / "out.tsv"), encoding="utf-8") as handle:
        assert list(csv.reader(handle, delimiter="\t")) == [header, *rows]
    lines = p.write_table(header, rows, tmp_path / "out.jsonl").read_text("utf-8").splitlines()
    first = json.loads(lines[0])
    assert len(lines) == 3 and list(first) == header
    assert first["department_confidence"] == 0.97 and first["id"] == "1"
    assert json.loads(lines[1])["department"] is None  # an empty cell is null
    records = json.loads(p.write_table(header, rows, tmp_path / "out.json").read_text("utf-8"))
    assert records[0] == first and len(records) == 3


def test_the_summary_counts_answers_reviews_and_matches(tmp_path):
    config = make_config(tmp_path, data={"train": "t.csv", "id_column": "id"})
    header, rows = table()
    summary = p.summarize(header, rows, config)
    assert summary.rows == 3 and summary.predicted == 2 and summary.review == 1
    assert summary.shares["department"] == [("billing", 0.5), ("technical", 0.5)]
    assert summary.shares["outage"] == [("no", 0.5), ("yes", 0.5)]
    assert summary.matches == {"department": (1, 1)}
    assert summary.preview[0]["department_predicted"] == "billing"


def test_an_input_check_finds_the_columns_the_model_reads(tmp_path):
    config = make_config(
        tmp_path, data={"train": "t.csv", "id_column": "id", "state_columns": ["subject", "body"]}
    )
    rows = [{k: v for k, v in r.items() if k in ("id", "subject", "body")} for r in ticket_rows(4)]
    renamed = [{"id": r["id"], "title": r["subject"], "body": r["body"]} for r in rows]
    path = write_csv(tmp_path / "new.csv", renamed)
    check = p.check_input(config, path, CharCounter())
    assert check.missing == ["subject"] and check.columns == ["id", "title", "body"]
    assert not check.ok
    fixed = p.check_input(config, path, CharCounter(), rename={"title": "subject"})
    assert fixed.ok and fixed.missing == [] and fixed.rows == 4
    assert fixed.tokens > 0 and fixed.truncated == 0 and fixed.expected == []
    out = p.write_input_csv(fixed, tmp_path / "staged.csv")
    with open(out, encoding="utf-8", newline="") as handle:
        assert next(csv.reader(handle)) == ["id", "subject", "body"]


def test_an_input_with_expected_outputs_is_noticed(tmp_path):
    config = make_config(
        tmp_path, data={"train": "t.csv", "id_column": "id", "state_columns": ["subject", "body"]}
    )
    check = p.check_input(config, write_csv(tmp_path / "l.csv", ticket_rows(5)), CharCounter())
    assert check.ok and check.expected == ["department", "urgency", "outage"]


def test_schema_fixture_matches_the_options():
    assert set(OPTIONS) == set(SCHEMA["questions"])


def test_rows_without_the_id_column_are_numbered(tmp_path):
    config = make_config(
        tmp_path, data={"train": "t.csv", "id_column": "id", "state_columns": ["subject", "body"]}
    )
    rows = [{"subject": r["subject"], "body": r["body"]} for r in ticket_rows(3)]
    check = p.check_input(config, write_csv(tmp_path / "noid.csv", rows), CharCounter())
    assert check.ok and check.numbered_ids and check.missing == []
    with open(p.write_input_csv(check, tmp_path / "s.csv"), encoding="utf-8", newline="") as f:
        assert [row["id"] for row in csv.DictReader(f)] == ["1", "2", "3"]


def test_the_output_gives_the_persons_own_column_names_back(tmp_path):
    config = make_config(
        tmp_path, data={"train": "t.csv", "id_column": "id", "state_columns": ["subject", "body"]}
    )
    rows = [{"title": r["subject"], "body": r["body"]} for r in ticket_rows(3)]
    path = write_csv(tmp_path / "renamed.csv", rows)
    check = p.check_input(config, path, CharCounter(), rename={"title": "subject"})
    assert check.table_columns == ["id", "subject", "body"]
    assert check.output_names() == {"id": "row", "subject": "title"}
    own = [{"row": str(k), "title": r["title"], "body": r["body"]} for k, r in enumerate(rows)]
    clash = p.check_input(
        config, write_csv(tmp_path / "row.csv", own), CharCounter(), {"title": "subject"}
    )
    assert clash.output_names() == {"id": "row_number", "subject": "title"}
    with_ids = [{"id": "a", "subject": "s", "body": "b"}]
    same = p.check_input(config, write_csv(tmp_path / "same.csv", with_ids), CharCounter())
    assert same.output_names() == {}
