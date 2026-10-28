from importlib.metadata import version
from pathlib import Path

import yaml

from scripts.prepare_space import prepare

SPACE = Path(__file__).resolve().parents[2] / "deploy" / "space"
SHA = "0123456789abcdef0123456789abcdef01234567"


def front_matter():
    text = (SPACE / "README.md").read_text(encoding="utf-8")
    return yaml.safe_load(text.split("---")[1])


def test_the_space_runs_the_tested_gradio_with_oauth():
    meta = front_matter()
    assert meta["sdk"] == "gradio" and meta["sdk_version"] == version("gradio")
    assert meta["hf_oauth"] is True
    assert {"write-repos", "manage-repos", "jobs"} <= set(meta["hf_oauth_scopes"])
    assert "launch()" in (SPACE / "app.py").read_text()


def test_prepare_pins_the_requirement_to_a_pushed_commit(tmp_path, monkeypatch):
    monkeypatch.delenv("CLEF_TUNE_SOURCE", raising=False)
    out = prepare(tmp_path / "space", "main", fetch=lambda url: {"sha": SHA})
    assert (out / "requirements.txt").read_text().strip() == (
        f"clef-tune[ui] @ git+https://github.com/hersh-gupta/clef-tune@{SHA}"
    )
    assert (out / "app.py").exists() and (out / "README.md").exists()
