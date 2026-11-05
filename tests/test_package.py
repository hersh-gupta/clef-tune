import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

import clef_tune
from clef_tune.launch.job_script import job_env
from clef_tune.launch.source import DEFAULT_SOURCE, Source

ROOT = Path(__file__).resolve().parents[1]
SHA = "0123456789abcdef0123456789abcdef01234567"


def test_version_is_a_non_empty_string():
    assert isinstance(clef_tune.__version__, str)
    assert clef_tune.__version__


@pytest.mark.network  # the build fetches hatchling and hatch-vcs, as a job's pip does
@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to build the wheel")
def test_a_job_builds_clef_tune_under_its_commit_version(tmp_path):
    # A job installs a GitHub archive: the project's files with no git metadata, so the version
    # that ends up in its model card can only come from the job's environment.
    archive = tmp_path / "archive"
    archive.mkdir()
    for name in ("pyproject.toml", "README.md"):
        shutil.copy(ROOT / name, archive / name)
    shutil.copytree(ROOT / "examples", archive / "examples")
    shutil.copytree(
        ROOT / "clef_tune",
        archive / "clef_tune",
        ignore=shutil.ignore_patterns("__pycache__", "_version.py"),
    )
    source = Source(DEFAULT_SOURCE, SHA)
    env = {**os.environ, **job_env(source, price_per_hour=None, estimated_hours=None)}
    subprocess.run(
        ["uv", "build", "--wheel", "--no-sources", "-o", str(tmp_path / "out"), str(archive)],
        env=env,
        check=True,
        capture_output=True,
    )
    wheel = next((tmp_path / "out").glob("*.whl"))
    assert wheel.name.startswith(f"clef_tune-{source.version}-")
    version_file = zipfile.ZipFile(wheel).read("clef_tune/_version.py").decode()
    assert f"__version__ = version = '{source.version}'" in version_file
