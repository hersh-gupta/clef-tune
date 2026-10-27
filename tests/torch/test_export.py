import pytest
import torch

from clef_tune.adapter import RELEASE_COPY_FILES, export_merged
from clef_tune.clef_code import load_clef_code
from clef_tune.predictor import Predictor
from tests.torch.test_predictor import ROWS
from tests.torch.tiny_clef import TINY_SCHEMA


def test_merged_export_loads_with_clefs_loader_and_matches_the_adapter(adapter, tmp_path):
    out = export_merged(adapter, tmp_path / "merged", dtype=torch.float32, max_shard_size="200KB")
    names = {p.name for p in out.iterdir()}
    assert {
        "config.json",
        "model.safetensors.index.json",
        "joint_head.safetensors",
        "joint_head_config.json",
        "calibration.json",
        "README.md",
        *RELEASE_COPY_FILES,
    } <= names
    jsm = load_clef_code(out)
    model, processor = jsm.load_release_model(out, device="cpu", dtype=torch.float32)
    request = {"model": "m", "state": ROWS[0]["state"], "questions": TINY_SCHEMA["questions"]}
    encoded = jsm.encode_record(processor.tokenizer, request, max_length=1024)
    with torch.no_grad():
        merged = model(
            jsm.collate_records([encoded], processor.tokenizer.pad_token_id, torch.device("cpu"))
        )[0]
    expected = Predictor.load(adapter, device="cpu").logits(ROWS)[0]
    for question, logits in zip(encoded.questions, merged, strict=True):
        assert logits.tolist() == pytest.approx(expected[question.question_id], abs=1e-4)
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "uncalibrated" in readme and "p_i^(1/T)" in readme
