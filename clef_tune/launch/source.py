"""Which clef-tune commit a Hugging Face job installs. Jobs install from a GitHub archive of an
exact commit, so a job runs the same code as the launcher. Torch-free."""

from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

DEFAULT_SOURCE = "https://github.com/hersh-gupta/clef-tune"
SOURCE_ENV = "CLEF_TUNE_SOURCE"  # overrides the GitHub repo jobs install from

_GITHUB_REPO = re.compile(r"^https://github\.com/[\w.-]+/[\w.-]+$")
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_MISSING = (404, 422)  # GitHub's answers for an unknown ref and for a malformed one


class SourceError(RuntimeError):
    """The commit a job should install cannot be determined; the message says what to do."""


@dataclass(frozen=True)
class Source:
    repo_url: str
    ref: str

    @property
    def owner_repo(self) -> str:
        return self.repo_url.removeprefix("https://github.com/")

    @property
    def archive_url(self) -> str:
        return f"{self.repo_url}/archive/{self.ref}.tar.gz"

    def requirement(self, extras: str = "train") -> str:
        return f"clef-tune[{extras}] @ {self.archive_url}"

    @property
    def version(self) -> str:
        """The version to build an archive under; it has no git metadata to derive one from."""
        return "0.0.0+g" + self.ref[:12] if _FULL_SHA.match(self.ref) else "0.0.0"


def source_repo(env: Mapping[str, str] = os.environ) -> str:
    value = env.get(SOURCE_ENV, "").strip().rstrip("/").removesuffix(".git").rstrip("/")
    if not value:
        return DEFAULT_SOURCE
    if not _GITHUB_REPO.match(value):
        raise SourceError(
            f"{SOURCE_ENV} must be a GitHub repo URL like {DEFAULT_SOURCE}, not {value!r}"
        )
    return value


def run_git(args: list[str], cwd: Path) -> str:
    try:
        done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", None) or exc
        raise SourceError(f"git {' '.join(args)} failed in {cwd}: {detail}") from exc
    return done.stdout


def installed_direct_url() -> str | None:
    try:
        return metadata.distribution("clef-tune").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None


def checkout_dir() -> Path | None:
    root = Path(__file__).resolve().parents[2]
    return root if (root / ".git").exists() else None


def running_commit(
    direct_url_json: str | None,
    checkout: Path | None,
    *,
    git: Callable[[list[str], Path], str] = run_git,
) -> str:
    """The commit of the running clef-tune. Never trusts clef_tune._version: an editable install
    freezes it at build time."""
    direct_url = json.loads(direct_url_json) if direct_url_json else {}
    commit = (direct_url.get("vcs_info") or {}).get("commit_id")
    if commit:
        return commit
    if checkout is not None:
        if git(["status", "--porcelain", "--untracked-files=no"], checkout).strip():
            raise SourceError(
                f"uncommitted changes in {checkout}; commit and push them, or pass --source-ref"
            )
        return git(["rev-parse", "HEAD"], checkout).strip()
    raise SourceError("cannot tell which clef-tune commit is installed; pass --source-ref")


def resolve_source(source_ref: str | None = None) -> Source:
    return Source(
        source_repo(), source_ref or running_commit(installed_direct_url(), checkout_dir())
    )


def github_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "clef-tune"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def pin_commit(source: Source, *, fetch: Callable[[str], dict] = github_json) -> Source:
    """Resolve the ref to a full SHA on GitHub, so a job cannot install anything else. A ref
    GitHub doesn't have aborts; a GitHub that cannot be reached only warns."""
    ref = urllib.parse.quote(source.ref, safe="/")
    url = f"https://api.github.com/repos/{source.owner_repo}/commits/{ref}"
    try:
        response = fetch(url)
    except (urllib.error.URLError, TimeoutError) as error:  # HTTPError is a URLError
        if isinstance(error, urllib.error.HTTPError) and error.code in _MISSING:
            raise SourceError(
                f"{source.ref} is not on GitHub at {source.repo_url}; "
                "push your commits first, or pass --source-ref"
            ) from error
        message = f"could not check {source.ref} on GitHub ({error}); launching anyway"
        warnings.warn(message, stacklevel=2)
        return source
    return Source(source.repo_url, response["sha"])
