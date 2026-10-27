import json
import zipfile

import pytest
from PIL import Image

from clef_tune.bundle import write_bundle, zip_bundle
from clef_tune.config import load_config
from clef_tune.validation import validate
from tests.helpers import CharCounter, make_config, ticket_rows, write_csv


def test_bundle_round_trip_preserves_split_and_labels(tmp_path):
    rows = ticket_rows(40)
    for row in rows:
        row["receipt"] = ""
    Image.new("RGB", (64, 64)).save(tmp_path / "r.png")
    rows[0]["receipt"] = "r.png"
    rows[1]["receipt"] = "r.png"
    write_csv(tmp_path / "src" / "train.csv", rows)
    (tmp_path / "src" / "r.png").write_bytes((tmp_path / "r.png").read_bytes())
    cfg = make_config(
        tmp_path / "src",
        data={"train": "train.csv", "id_column": "id", "image_columns": ["receipt"]},
    )
    report = validate(cfg, CharCounter())
    assert report.ok

    config_path = write_bundle(report, cfg, tmp_path / "bundle")
    bundled = load_config(config_path)
    assert bundled.data.train == "train.jsonl" and bundled.data.eval == "eval.jsonl"
    assert bundled.get_schema() == cfg.get_schema()
    again = validate(bundled, CharCounter())
    assert again.ok
    assert [r.id for r in again.train] == [r.id for r in report.train]
    assert [r.id for r in again.eval] == [r.id for r in report.eval]
    assert [r.labels for r in again.train] == [r.labels for r in report.train]

    first = json.loads(
        (tmp_path / "bundle" / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert "questions" not in first  # rows that use the config schema don't repeat it
    images = list((tmp_path / "bundle" / "images").iterdir())
    assert len(images) == 1  # the same file is copied once
    with_image = [r for r in again.train + again.eval if r.images]
    assert all(r.images == [str(images[0].resolve())] for r in with_image)

    archive = zip_bundle(tmp_path / "bundle", tmp_path / "bundle.zip")
    assert "config.yaml" in zipfile.ZipFile(archive).namelist()


def test_bundle_refuses_reports_with_errors(tmp_path):
    rows = ticket_rows(20)
    rows[0]["department"] = "biling"
    write_csv(tmp_path / "train.csv", rows)
    cfg = make_config(tmp_path, data={"train": "train.csv", "id_column": "id"})
    report = validate(cfg, CharCounter())
    with pytest.raises(ValueError, match="validation has errors"):
        write_bundle(report, cfg, tmp_path / "bundle")
