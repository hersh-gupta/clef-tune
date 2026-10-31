from pathlib import Path

import yaml

from clef_tune.config import PINNED_REVISIONS
from scripts.prepare_space import prepare

SPACE = Path(__file__).resolve().parents[2] / "deploy" / "space"
SHA = "0123456789abcdef0123456789abcdef01234567"


def front_matter():
    text = (SPACE / "README.md").read_text(encoding="utf-8")
    return yaml.safe_load(text.split("---")[1])


def test_the_space_is_a_docker_app_with_hugging_face_sign_in():
    meta = front_matter()
    assert meta["sdk"] == "docker" and meta["app_port"] == 7860
    assert meta["hf_oauth"] is True
    assert {"read-repos", "write-repos", "manage-repos", "jobs"} <= set(meta["hf_oauth_scopes"])
    assert len(meta["short_description"]) <= 60  # the Hub refuses longer ones
    # Listed under "Spaces using this model" on each model's page.
    assert set(meta["models"]) == set(PINNED_REVISIONS)
    # The social thumbnail is the app's own cover, from GitHub so it shows while the Space sleeps.
    cover = Path(__file__).resolve().parents[2] / "clef_tune" / "ui" / "static" / "cover.png"
    assert meta["thumbnail"].endswith("/main/clef_tune/ui/static/cover.png") and cover.exists()
    dockerfile = (SPACE / "Dockerfile").read_text(encoding="utf-8")
    assert "requirements.txt" in dockerfile and "uvicorn app:app" in dockerfile
    assert "--port 7860" in dockerfile and "--proxy-headers" in dockerfile
    assert "create_app()" in (SPACE / "app.py").read_text(encoding="utf-8")


def test_prepare_pins_the_requirement_to_a_pushed_commit(tmp_path, monkeypatch):
    monkeypatch.delenv("CLEF_TUNE_SOURCE", raising=False)
    out = prepare(tmp_path / "space", "main", fetch=lambda url: {"sha": SHA})
    assert (out / "requirements.txt").read_text().strip() == (
        f"clef-tune[ui] @ git+https://github.com/hersh-gupta/clef-tune@{SHA}"
    )
    assert {"README.md", "Dockerfile", "app.py"} <= {p.name for p in out.iterdir()}
