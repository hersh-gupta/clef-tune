import warnings

import pytest

from clef_tune import hub
from clef_tune.hub import HubError, ensure_repo, pull_adapter, push_adapter

WRITE_TOKEN = {"name": "me", "auth": {"type": "access_token", "accessToken": {"role": "write"}}}


def token(role, **access):
    return {"name": "me", "auth": {"type": "access_token", "accessToken": {"role": role, **access}}}


def fine_grained(*scoped, global_permissions=()):
    grants = {"canReadGatedRepos": False, "global": list(global_permissions), "scoped": []}
    for kind, name, permissions in scoped:
        grants["scoped"].append(
            {"entity": {"type": kind, "name": name}, "permissions": permissions}
        )
    return token("fineGrained", fineGrained=grants)


class FakeApi:
    def __init__(self, fail=None, token_info=WRITE_TOKEN):
        self.calls, self.fail, self.token_info = [], fail, token_info

    def whoami(self, token=None, **kwargs):
        if isinstance(self.token_info, BaseException):
            raise self.token_info
        return self.token_info

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


@pytest.mark.parametrize(
    "info",
    [
        token("read"),
        fine_grained(("user", "me", ["repo.content.read"])),
        fine_grained(("org", "someone-else", ["repo.write"])),
        fine_grained(("model", "me/other", ["repo.write"])),
    ],
)
def test_a_token_without_write_access_is_refused_before_creating_the_repo(info):
    api = FakeApi(token_info=info)
    with pytest.raises(HubError, match="write access"):
        ensure_repo("me/run", api=api)
    assert api.calls == []


@pytest.mark.parametrize(
    "info",
    [
        token("write"),
        token("admin"),
        fine_grained(("user", "me", ["repo.content.read", "repo.write"])),
        fine_grained(("model", "me/run", ["repo.write"])),
        fine_grained(global_permissions=["repo.write"]),
    ],
)
def test_a_token_with_write_access_passes(info):
    api = FakeApi(token_info=info)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ensure_repo("me/run", api=api)
    assert [call[0] for call in api.calls] == ["create_repo"]


def test_a_repo_without_a_namespace_belongs_to_the_token_owner():
    ensure_repo("run", api=FakeApi(token_info=fine_grained(("user", "me", ["repo.write"]))))
    with pytest.raises(HubError, match="write access"):
        ensure_repo("run", api=FakeApi(token_info=fine_grained(("org", "acme", ["repo.write"]))))


@pytest.mark.parametrize(
    "info",
    [
        {"name": "me", "auth": {"type": "oauth"}},
        {"name": "me", "auth": "opaque"},
        token("someNewRole"),
        token("fineGrained", fineGrained={"scoped": "everything"}),
    ],
)
def test_an_unreadable_token_description_warns_instead_of_refusing(info):
    api = FakeApi(token_info=info)
    with pytest.warns(UserWarning, match="could not confirm .* write access to me/run"):
        ensure_repo("me/run", api=api)
    assert [call[0] for call in api.calls] == ["create_repo"]


def test_an_invalid_or_missing_token_is_a_hub_error():
    with pytest.raises(HubError, match="me/run.*Invalid user token"):
        ensure_repo("me/run", api=FakeApi(token_info=OSError("Invalid user token.")))
