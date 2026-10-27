import sys

import pytest

torch = pytest.importorskip("torch")
from clef_tune.clef_code import ClefCodeError, load_clef_code  # noqa: E402

pytestmark = pytest.mark.torch


def test_load_clef_code_registers_the_module_once(tiny_release):
    first = load_clef_code(tiny_release)
    assert load_clef_code(tiny_release) is first
    assert first.__name__.startswith("clef_joint_schema_model_")
    assert sys.modules[first.__name__] is first
    assert hasattr(first, "encode_record") and hasattr(first, "ClefModel")


def test_load_clef_code_without_the_file(tmp_path):
    with pytest.raises(ClefCodeError, match="joint_schema_model.py"):
        load_clef_code(tmp_path)


def test_tiny_release_loads_with_clefs_own_loader(tiny_release):
    jsm = load_clef_code(tiny_release)
    model, processor = jsm.load_release_model(tiny_release, device="cpu", dtype=torch.float32)
    request = {
        "model": "tiny",
        "state": "the site is down",
        "questions": {"outage": {"type": "noul"}},
    }
    answer = jsm.systemone(model, processor, request)["answers"]["outage"]
    assert answer["type"] == "noul" and 0.0 <= answer["noul"] <= 1.0
