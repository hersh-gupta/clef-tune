import pytest

from clef_tune import clef_code
from clef_tune.clef_code import ClefCodeError, model_dir
from clef_tune.config import PINNED_REVISIONS
from tests.helpers import make_config


def test_model_dir_uses_local_path(tmp_path):
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "config.json").write_text("{}", encoding="utf-8")
    cfg = make_config(tmp_path, data={"train": "t.csv"}, model={"local_path": "weights"})
    assert model_dir(cfg) == (tmp_path / "weights").resolve()


def test_model_dir_rejects_a_missing_local_path(tmp_path):
    cfg = make_config(tmp_path, data={"train": "t.csv"}, model={"local_path": "nope"})
    with pytest.raises(ClefCodeError, match="local_path"):
        model_dir(cfg)


def test_model_dir_downloads_the_pinned_revision(tmp_path, monkeypatch):
    calls = []

    def fake_snapshot_download(repo_id, revision=None, **kwargs):
        calls.append((repo_id, revision))
        return str(tmp_path)

    monkeypatch.setattr(clef_code, "snapshot_download", fake_snapshot_download)
    cfg = make_config(tmp_path, data={"train": "t.csv"}, model={"id": "Cloudflare/clef-flash"})
    assert model_dir(cfg) == tmp_path
    assert calls == [("Cloudflare/clef-flash", PINNED_REVISIONS["Cloudflare/clef-flash"])]
