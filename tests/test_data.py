import json

from clef_tune.data import Record, load_file, split_records
from clef_tune.labels import Label
from tests.helpers import make_config, ticket_rows, write_csv

HEADER = "id,subject,body,department,urgency,outage\n"


def load_text(tmp_path, text, name="train.csv", encoding="utf-8", **data):
    (tmp_path / name).write_bytes(text.encode(encoding))
    cfg = make_config(tmp_path, data={"train": name, "id_column": "id", **data})
    return load_file(cfg, name)


def test_flat_csv_basic(tmp_path):
    result = load_text(
        tmp_path,
        HEADER + "1,Card declined,Help,billing,2,yes\n2,Site down,Now,technical,High,no\n",
        state_columns=["subject", "body"],
    )
    assert [i for i in result.issues if i.level == "error"] == []
    first, second = result.records
    assert first.id == "1" and first.source_row == 2
    assert first.state == {"subject": "Card declined", "body": "Help"}
    assert first.labels == {
        "department": Label(hard="billing"),
        "urgency": Label(hard="2"),
        "outage": Label(hard="true"),
    }
    assert second.labels["urgency"] == Label(hard="2")
    assert second.labels["outage"] == Label(hard="false")
    assert list(first.questions) == ["department", "urgency", "outage"]


def test_single_state_column_is_raw_string(tmp_path):
    result = load_text(tmp_path, HEADER + "1,Hi,There,billing,0,no\n", state_columns=["body"])
    assert result.records[0].state == "There"


def test_default_state_columns_are_the_unused_columns(tmp_path):
    result = load_text(tmp_path, HEADER + "1,Hi,There,billing,0,no\n")
    assert result.records[0].state == {"subject": "Hi", "body": "There"}


def test_na_like_values_stay_literal(tmp_path):
    result = load_text(tmp_path, HEADER + "1,N/A,None,NA,0,no\n2,null,NaN,billing,1,no\n")
    first, second = result.records
    assert first.labels["department"] == Label(hard="NA")
    assert first.state == {"subject": "N/A", "body": "None"}
    assert second.state == {"subject": "null", "body": "NaN"}


def test_empty_label_masks_question_and_unlabeled_rows_are_dropped(tmp_path):
    result = load_text(tmp_path, HEADER + "1,a,b,billing,,\n2,c,d,,,\n")
    assert len(result.records) == 1
    assert set(result.records[0].labels) == {"department"}
    dropped = [i for i in result.issues if "no labels" in i.message]
    assert dropped and dropped[0].rows == [3] and dropped[0].level == "warning"


def test_unknown_label_values_are_grouped_with_rows(tmp_path):
    result = load_text(tmp_path, HEADER + "1,a,b,biling,0,no\n2,c,d,biling,0,no\n")
    errors = [i for i in result.issues if i.level == "error"]
    assert len(errors) == 1
    assert errors[0].question == "department" and errors[0].rows == [2, 3]
    assert "'biling' is not an option" in str(errors[0])
    assert "(rows 2, 3)" in str(errors[0])


def test_case_fix_warning(tmp_path):
    result = load_text(tmp_path, HEADER + "1,a,b,Billing,0,no\n")
    assert result.records[0].labels["department"] == Label(hard="billing")
    assert any(i.level == "warning" and "by case" in i.message for i in result.issues)


def test_missing_label_column_suggests_close_match(tmp_path):
    text = "id,subject,dept_label,urgency,outage\n1,a,billing,0,no\n"
    result = load_text(tmp_path, text, label_columns={"department": "dept_labl"})
    errors = [str(i) for i in result.issues if i.level == "error"]
    assert any("label column 'dept_labl' not found" in e and "dept_label" in e for e in errors)


def test_header_whitespace_is_stripped(tmp_path):
    text = "id, subject ,body,department ,urgency,outage\n1,a,b,billing,0,no\n"
    result = load_text(tmp_path, text)
    assert result.records[0].labels["department"] == Label(hard="billing")
    assert "subject" in result.columns


def test_wrong_field_count_reports_row(tmp_path):
    result = load_text(tmp_path, HEADER + "1,a,b,billing,0,no\n2,a,billing\n")
    errors = [i for i in result.issues if i.level == "error"]
    assert errors[0].rows == [3] and "fields" in errors[0].message


def test_bom_header_and_cp1252_fallback(tmp_path):
    bom = load_text(tmp_path, "\ufeff" + HEADER + "1,a,b,billing,0,no\n")
    assert bom.columns[0] == "id"
    assert bom.records[0].id == "1"
    legacy = load_text(tmp_path, HEADER + "1,café,b,billing,0,no\n", encoding="cp1252")
    assert legacy.records[0].state["subject"] == "café"
    assert any("Windows-1252" in i.message for i in legacy.issues)


def test_jsonl_native_blank_and_malformed_lines(tmp_path):
    lines = [
        json.dumps({"id": "a", "state": {"text": "hi"}, "labels": {"outage": "yes"}}),
        "",
        "{not json",
        json.dumps(
            {
                "id": "b",
                "state": "free text",
                "questions": {"spam": {"type": "noul", "instructions": "Spam?"}},
                "labels": {"spam": False},
            }
        ),
        "",
    ]
    (tmp_path / "train.jsonl").write_text("\n".join(lines), encoding="utf-8")
    result = load_file(make_config(tmp_path, data={"train": "train.jsonl"}), "train.jsonl")
    assert [r.id for r in result.records] == ["a", "b"]
    assert result.records[0].labels == {"outage": Label(hard="true")}
    assert list(result.records[1].questions) == ["spam"]
    assert result.records[1].source_row == 4
    errors = [i for i in result.issues if i.level == "error"]
    assert len(errors) == 1 and errors[0].rows == [3] and "invalid JSON" in errors[0].message


def test_jsonl_soft_labels_and_unknown_question(tmp_path):
    lines = [
        json.dumps({"state": "x", "labels": {"department": {"billing": 0.6, "technical": 0.4}}}),
        json.dumps({"state": "y", "labels": {"bogus": "yes", "outage": "no"}}),
    ]
    (tmp_path / "train.jsonl").write_text("\n".join(lines), encoding="utf-8")
    result = load_file(make_config(tmp_path, data={"train": "train.jsonl"}), "train.jsonl")
    assert result.records[0].labels["department"].soft == {"billing": 0.6, "technical": 0.4}
    assert result.records[0].id == "row-1"
    errors = [i for i in result.issues if i.level == "error"]
    assert errors[0].question == "bogus" and errors[0].rows == [2]


def test_flat_rows_without_schema_error(tmp_path):
    write_csv(tmp_path / "train.csv", ticket_rows(2))
    cfg = make_config(tmp_path, schema=None, data={"train": "train.csv"})
    result = load_file(cfg, "train.csv")
    assert result.records == []
    assert any("need a schema" in i.message for i in result.issues)


def test_image_paths_resolve_relative_to_data_file(tmp_path):
    text = "id,body,receipt,department,urgency,outage\n1,b,imgs\\a.png,billing,0,no\n"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.csv").write_text(text, encoding="utf-8")
    cfg = make_config(
        tmp_path,
        data={"train": "data/train.csv", "id_column": "id", "image_columns": ["receipt"]},
    )
    record = load_file(cfg, "data/train.csv").records[0]
    assert record.images == [str((data_dir / "imgs" / "a.png").resolve())]
    assert record.state == "b"


def test_record_to_json(tmp_path):
    record = load_text(tmp_path, HEADER + "1,a,b,billing,2,yes\n").records[0]
    data = record.to_json()
    assert data["labels"] == {"department": "billing", "urgency": "2", "outage": "true"}
    assert data["questions"]["urgency"] == {"type": "score", "criteria": ["Low", "Medium", "High"]}
    assert "images" not in data


def _records(departments):
    return [
        Record(str(i), "s", {}, {"department": Label(hard=d)}, [], i + 2)
        for i, d in enumerate(departments)
    ]


def test_split_is_stratified_and_deterministic():
    records = _records(["billing"] * 80 + ["technical"] * 20)
    train, evals = split_records(records, 0.1, seed=42)
    assert len(evals) == 10
    assert sum(r.labels["department"].hard == "technical" for r in evals) == 2
    assert [r.id for r in split_records(records, 0.1, seed=42)[1]] == [r.id for r in evals]
    assert [r.id for r in split_records(records, 0.1, seed=7)[1]] != [r.id for r in evals]
    assert len(train) + len(evals) == 100
    assert [r.source_row for r in train] == sorted(r.source_row for r in train)


def test_split_keeps_singletons_in_train_but_always_makes_an_eval_row():
    records = _records(["billing", "technical", "NA"])
    train, evals = split_records(records, 0.1, seed=1)
    assert len(evals) == 1 and len(train) == 2


def test_jsonl_line_separator_characters_inside_strings(tmp_path):
    lines = [
        json.dumps(
            {"id": "a", "state": "x\u2028y\u2029z\x85", "labels": {"outage": "yes"}},
            ensure_ascii=False,
        ),
        json.dumps({"id": "b", "state": "plain", "labels": {"outage": "no"}}),
    ]
    (tmp_path / "train.jsonl").write_bytes("\r\n".join(lines).encode("utf-8"))
    result = load_file(make_config(tmp_path, data={"train": "train.jsonl"}), "train.jsonl")
    assert [i for i in result.issues if i.level == "error"] == []
    assert [r.id for r in result.records] == ["a", "b"]
    assert result.records[0].state == "x\u2028y\u2029z\x85"
    assert result.records[1].source_row == 2


def test_csv_cells_larger_than_the_default_csv_limit_load(tmp_path):
    big = "x" * 200_000
    result = load_text(tmp_path, HEADER + f"1,a,{big},billing,0,no\n", state_columns=["body"])
    assert result.records[0].state == big


def test_jsonl_native_images_accepts_one_string_and_rejects_other_types(tmp_path):
    lines = [
        json.dumps({"id": "a", "state": "s", "labels": {"outage": "yes"}, "images": "img.png"}),
        json.dumps({"id": "b", "state": "s", "labels": {"outage": "no"}, "images": 5}),
    ]
    (tmp_path / "train.jsonl").write_text("\n".join(lines), encoding="utf-8")
    result = load_file(make_config(tmp_path, data={"train": "train.jsonl"}), "train.jsonl")
    assert [r.id for r in result.records] == ["a"]
    assert result.records[0].images == [str((tmp_path / "img.png").resolve())]
    errors = [i for i in result.issues if i.level == "error"]
    assert len(errors) == 1 and errors[0].rows == [2] and "'images' must be" in errors[0].message
