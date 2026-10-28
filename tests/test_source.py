import json
import urllib.error

import pytest

from clef_tune.launch import source as src
from clef_tune.launch.source import (
    DEFAULT_SOURCE,
    Source,
    SourceError,
    pin_commit,
    resolve_source,
    running_commit,
    source_repo,
)

SHA = "0123456789abcdef0123456789abcdef01234567"


class FakeGit:
    def __init__(self, status="", head=SHA):
        self.status, self.head, self.calls = status, head, []

    def __call__(self, args, cwd):
        self.calls.append(args)
        return self.status if args[0] == "status" else self.head + "\n"


def test_vcs_install_uses_the_recorded_commit():
    direct_url = json.dumps(
        {
            "url": "https://github.com/hersh-gupta/clef-tune",
            "vcs_info": {"vcs": "git", "commit_id": SHA},
        }
    )
    git = FakeGit()
    assert running_commit(direct_url, None, git=git) == SHA
    assert git.calls == []


def test_a_checkout_uses_git_head_not_the_build_time_version(tmp_path):
    git = FakeGit()
    assert running_commit(None, tmp_path, git=git) == SHA
    assert git.calls == [["status", "--porcelain", "--untracked-files=no"], ["rev-parse", "HEAD"]]


def test_an_editable_install_reads_its_checkout(tmp_path):
    direct_url = json.dumps({"url": tmp_path.as_uri(), "dir_info": {"editable": True}})
    assert running_commit(direct_url, tmp_path, git=FakeGit()) == SHA


def test_uncommitted_changes_are_refused(tmp_path):
    with pytest.raises(SourceError, match="uncommitted changes.*--source-ref"):
        running_commit(None, tmp_path, git=FakeGit(status=" M clef_tune/cli.py\n"))


def test_an_install_without_vcs_information_needs_a_source_ref():
    archive = json.dumps({"url": "https://example.com/clef_tune.tar.gz", "archive_info": {}})
    with pytest.raises(SourceError, match="--source-ref"):
        running_commit(archive, None, git=FakeGit())


def test_a_source_ref_wins(monkeypatch):
    monkeypatch.setattr(src, "running_commit", lambda *a, **k: pytest.fail("looked up the commit"))
    monkeypatch.delenv("CLEF_TUNE_SOURCE", raising=False)
    assert resolve_source("v1.2") == Source(DEFAULT_SOURCE, "v1.2")


@pytest.mark.parametrize(
    "value",
    [
        "https://github.com/me/fork",
        "https://github.com/me/fork.git/",
        " https://github.com/me/fork/ ",
    ],
)
def test_the_source_repo_can_be_overridden(value):
    assert source_repo({"CLEF_TUNE_SOURCE": value}) == "https://github.com/me/fork"
    assert source_repo({}) == DEFAULT_SOURCE


def test_the_source_repo_must_be_a_github_https_url():
    with pytest.raises(SourceError, match="CLEF_TUNE_SOURCE"):
        source_repo({"CLEF_TUNE_SOURCE": "git@github.com:me/fork.git"})


def test_requirement_and_version():
    pinned = Source(DEFAULT_SOURCE, SHA)
    assert pinned.owner_repo == "hersh-gupta/clef-tune"
    assert pinned.requirement() == f"clef-tune[train] @ {DEFAULT_SOURCE}/archive/{SHA}.tar.gz"
    assert pinned.version == "0.0.0+g0123456789ab"
    assert Source(DEFAULT_SOURCE, "main").version == "0.0.0"


def test_pin_commit_resolves_a_ref_to_its_full_sha():
    urls = []

    def fetch(url):
        urls.append(url)
        return {"sha": SHA}

    assert pin_commit(Source(DEFAULT_SOURCE, "main"), fetch=fetch) == Source(DEFAULT_SOURCE, SHA)
    assert urls == ["https://api.github.com/repos/hersh-gupta/clef-tune/commits/main"]


@pytest.mark.parametrize("code", [404, 422])
def test_a_commit_missing_from_github_aborts(code):
    def fetch(url):
        raise urllib.error.HTTPError(url, code, "Not Found", {}, None)

    with pytest.raises(SourceError, match="push your commits"):
        pin_commit(Source(DEFAULT_SOURCE, SHA), fetch=fetch)


@pytest.mark.parametrize(
    "error",
    [urllib.error.URLError("offline"), urllib.error.HTTPError("u", 403, "rate limited", {}, None)],
)
def test_an_unreachable_github_only_warns(error):
    def fetch(url):
        raise error

    with pytest.warns(UserWarning, match="could not check"):
        assert pin_commit(Source(DEFAULT_SOURCE, SHA), fetch=fetch) == Source(DEFAULT_SOURCE, SHA)
