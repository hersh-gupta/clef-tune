import pytest
from PIL import Image

from clef_tune.calibration import temper_probabilities
from clef_tune.predictor import Predictor, predict_file

ROWS = [{"id": "a", "state": {"subject": "x", "body": "the site is down and the invoice matters"}}]


def test_predict_returns_systemone_answers(adapter):
    out = Predictor.load(adapter, device="cpu").predict(ROWS)
    assert out[0]["id"] == "a" and set(out[0]["answers"]) == {"outage", "department"}
    assert out[0]["answers"]["outage"]["type"] == "noul"
    assert set(out[0]["answers"]["department"]["probabilities"]) == {"billing", "technical"}


def test_calibration_applies_the_stored_temperatures(adapter):
    predictor = Predictor.load(adapter, device="cpu")
    raw = predictor.probabilities(ROWS, calibrated=False)[0]
    calibrated = predictor.probabilities(ROWS)[0]
    tempered = temper_probabilities(list(raw["outage"].values()), 2.0)
    assert list(calibrated["outage"].values()) == pytest.approx(tempered, abs=1e-6)
    assert calibrated["department"] == pytest.approx(raw["department"], abs=1e-6)


def test_systemone_keeps_clefs_contract(adapter):
    response = Predictor.load(adapter, device="cpu").systemone(
        {"model": "m", "state": "s", "questions": {"spam": {"type": "noul"}}}
    )
    assert response["model"] == "m" and set(response["answers"]) == {"spam"}
    assert response["usage"]["input_tokens"] > 0 and response["usage"]["output_tokens"] == 0


def test_predict_file_writes_jsonl_and_csv(adapter, tmp_path):
    (tmp_path / "in.csv").write_text(
        "id,subject,body\n1,a,down\n2,b,fine\n3,c,x\n", encoding="utf-8"
    )
    predictor = Predictor.load(adapter, device="cpu")
    assert predict_file(predictor, tmp_path / "in.csv", tmp_path / "out.jsonl") == 3
    assert len((tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    predict_file(predictor, tmp_path / "in.csv", tmp_path / "out.csv")
    header = (tmp_path / "out.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header[:5] == ["id", "outage", "outage.p.true", "outage.p.false", "department"]


def test_image_rows_predict(adapter, tmp_path):
    Image.new("RGB", (64, 64), (200, 10, 10)).save(tmp_path / "i.png")
    out = Predictor.load(adapter, device="cpu").predict(
        [{"id": "i", "state": "s", "images": [str(tmp_path / "i.png")]}]
    )
    assert set(out[0]["answers"]) == {"outage", "department"}


def test_predictor_is_exported_from_the_package():
    from clef_tune import Predictor as Exported

    assert Exported is Predictor
