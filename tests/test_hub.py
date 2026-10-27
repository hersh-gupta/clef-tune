import pytest

from clef_tune import hub
from clef_tune.hub import HubError, ensure_repo, pull_adapter, push_adapter


class FakeApi:
    def __init__(self, fail=None):
        self.calls, self.fail = [], fail

    def create_repo(self, repo_id, **kwargs):
        self.calls.append(("create_repo", repo_id, kwargs))
        if self.fail:
            raise self.fail
        return f"https://huggingface.co/{repo_id}"

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", kwargs))
        return "https://huggingface.co/me/run/commit/abc"


def test_ensure_repo_creates_a_private_model_repo():
    api = FakeApi()
    ensure_repo("me/run", api=api)
    assert api.calls == [
        (
            "create_repo",
            "me/run",
            {"repo_type": "model", "private": True, "exist_ok": True, "token": None},
        )
    ]


def test_ensure_repo_explains_permission_errors():
    with pytest.raises(HubError, match="write access"):
        ensure_repo("me/run", api=FakeApi(fail=RuntimeError("403 Forbidden")))


def test_push_adapter_uploads_the_folder(tmp_path):
    api = FakeApi()
    assert push_adapter(tmp_path, "me/run", api=api) == "https://huggingface.co/me/run/commit/abc"
    kwargs = api.calls[-1][1]
    assert kwargs["folder_path"] == str(tmp_path) and kwargs["repo_id"] == "me/run"
    assert kwargs["repo_type"] == "model"


def test_push_adapter_wraps_upload_failures(tmp_path):
    class Failing(FakeApi):
        def upload_folder(self, **kwargs):
            raise RuntimeError("503 Service Unavailable")

    with pytest.raises(HubError, match="503"):
        push_adapter(tmp_path, "me/run", api=Failing())


def test_pull_adapter(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        hub, "snapshot_download", lambda repo_id, **kw: calls.append((repo_id, kw)) or str(tmp_path)
    )
    assert pull_adapter("me/run", revision="v1") == tmp_path
    assert calls == [("me/run", {"revision": "v1", "token": None})]
