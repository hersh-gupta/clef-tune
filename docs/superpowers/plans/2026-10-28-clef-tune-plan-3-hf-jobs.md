# clef-tune Plan 3: Hugging Face Jobs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch, watch, cancel and resume clef-tune training runs on Hugging Face Jobs from the CLI (and, in Plan 4, the UI). A job installs the exact clef-tune commit the launcher runs, trains on an uploaded bundle with the model mounted, pushes the adapter to a private repo, and resumes from its last checkpoint after a failure.

**Architecture:** Three torch-free modules under `clef_tune/launch/`:
- `source.py` decides which GitHub commit a job installs.
- `job_script.py` turns that into the job's bash command and environment.
- `hf_jobs.py` drives the Hub API.

Launching has two phases, so a person sees the cost before spending anything: `plan_launch` validates, estimates and picks hardware, uploads nothing and starts nothing; `submit_launch` stages the bundle in the private `jobs-artifacts` bucket and calls `run_job`. Monitoring reads the job record plus the `metrics.jsonl` the trainer writes into the bucket. A final task hardens the trainer for what jobs add: several GPUs per job, and output on a bucket mount.

**Tech Stack:** `huggingface_hub` 1.33 (`run_job`, `inspect_job`, `list_jobs`, `fetch_job_logs`, `cancel_job`, `sync_job_volume`, `Volume`, bucket file APIs), stdlib `urllib` for GitHub, Typer, and everything from Plans 1–2.

**Spec:** `docs/superpowers/specs/2026-10-26-clef-finetuning-harness-design.md` (revision 2), §8 especially. Read `docs/superpowers/plans/2026-10-26-clef-tune-roadmap.md` too. Its "Verified constraints and spec amendments" override the spec where they differ, and items 18–28 were verified on 2026-10-28 with real HF Jobs runs for this plan.

**Branch:** work on `plan-3-hf-jobs`, created from `main`.

## Global Constraints

- Every Plan 1 and Plan 2 global constraint still holds:
  - Python `>=3.11` (dev 3.12), ruff line length 100.
  - Run `uv run ruff format . && uv run ruff check --fix .` before every commit.
  - Commit messages carry no Claude attribution: no `Co-Authored-By` or session trailer.

- **No torch:** `clef_tune.launch`, `clef_tune.launch.source`, `clef_tune.launch.job_script` and `clef_tune.launch.hf_jobs` must not import torch, transformers or peft. Add each one to `NO_TORCH_MODULES` in `tests/test_boundaries.py` in the task that creates it.
- **No new dependencies:** GitHub calls use `urllib.request`. Tests may use `httpx`, which `huggingface_hub` already depends on.
- **No network in unit tests:** every Hub and GitHub call goes through an injectable `api`, `fetch` or `git` parameter, and the tests pass fakes. Only the opt-in `hfjobs` test (Task 8) contacts real services.
- **Tokens:**
  - Every Hub-facing function takes `token: str | None`; `None` means `huggingface_hub.get_token()`.
  - The token reaches a job only through `secrets={"HF_TOKEN": token}`.
  - Tokens are never stored in a dataclass, printed or logged.
- **Job image:** `runtime.BASE_IMAGE` (`pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel`).
- **Job environment:**
  - Always set `PIP_BREAK_SYSTEM_PACKAGES=1` and `PYTHONUNBUFFERED=1`.
  - Never set a name starting with `NVIDIA_`; HF rejects them with 400.
- **Kernels:**
  - Pinned to `causal-conv1d==1.7.0` and `flash-linear-attention==0.6.0`.
  - `causal-conv1d` builds with `--no-build-isolation` and must pass `python -c "import causal_conv1d"`.
  - On any failure, uninstall `causal-conv1d flash-linear-attention fla-core` and continue on the slow path.
- **Labels on every job:** `{"tool": "clef-tune", "run": <run_name>, "timeout": <timeout>}`.
- **Adapter repos:** private. The default name is `<namespace>/clef-tune-<run_name>`.
- **Source:** the source repo defaults to `https://github.com/hersh-gupta/clef-tune`, overridable with `CLEF_TUNE_SOURCE`. Jobs install `clef-tune[train] @ <repo>/archive/<sha>.tar.gz`; the image has no git.
- **Spending:** `clef-tune jobs launch` shows the estimate and asks before submitting unless `--yes` is given.

## Review Focus

1. **A dirty or unpushed checkout** must be refused before anything is uploaded or any money is spent. Pinned by:
   - `test_uncommitted_changes_are_refused` and `test_a_commit_missing_from_github_aborts` (Task 1);
   - `test_a_dirty_checkout_is_refused_before_touching_the_hub` (Task 3).
2. **A kernel build or import that fails** (it did on 2026-10-28) must leave the job training on the slow path, not crash it. Pinned by `test_a_failed_kernel_build_falls_back_to_the_slow_path` (Task 2).
3. **A job canceled or timed out just after a checkpoint** must resume, in the same `/work` folder, from the newest *complete* checkpoint. Bucket uploads finish in the background, so the newest step folder can be partial. Pinned by:
   - `test_resume_rebuilds_the_failed_job_with_a_fresh_token` (Task 5);
   - `test_latest_checkpoint_skips_incomplete_steps` (Task 7);
   - the live test (Task 8).
4. **Launching the same `run_name` twice** must give each job its own `/work` folder. Otherwise the second would auto-resume from, or fail on, the first one's checkpoints. Pinned by `test_each_launch_gets_its_own_work_folder` (Task 4).
5. **A rejected submission** (out of credits, a missing scope, an HTTP error) must say what to do and remove the bundle it uploaded. Pinned by `test_out_of_credits_is_explained_and_the_upload_removed` (Task 4).

---

## File Structure

```
clef_tune/launch/__init__.py         # Task 1 (empty package marker)
clef_tune/launch/source.py           # Task 1: which commit a job installs (no torch)
clef_tune/launch/job_script.py       # Task 2: the job's bash command and environment (no torch)
clef_tune/launch/hf_jobs.py          # Tasks 3-5: plan, submit, list, status, logs, cancel, resume (no torch)
clef_tune/runtime.py                 # Task 2: kernel pins
clef_tune/estimate.py                # Task 2: suggest_timeout(row, extra_minutes)
Dockerfile                           # Task 2: pinned kernel versions
clef_tune/cli.py                     # Task 6: `jobs` sub-commands; Task 7: train prints once
README.md                            # Task 6: HF Jobs quickstart, kernel install fix
clef_tune/train.py                   # Task 7: multi-GPU fixes
clef_tune/checkpoints.py             # Task 7: skip incomplete checkpoints
clef_tune/card.py                    # Task 7: Hub metadata front matter
tests/test_source.py                 # Task 1
tests/test_job_script.py             # Task 2
tests/test_hf_jobs.py                # Tasks 3-5 (FakeApi lives here)
tests/test_cli_jobs.py               # Task 6
tests/test_hfjobs.py                 # Task 8: opt-in live test (marker `hfjobs`)
```

---

### Task 1: Install source

**Files:**
- Create: `clef_tune/launch/__init__.py` (empty), `clef_tune/launch/source.py`, `tests/test_source.py`
- Modify: `tests/test_boundaries.py` (add `clef_tune.launch`, `clef_tune.launch.source`)

**Interfaces:**
- Consumes: nothing.
- Produces, in `clef_tune.launch.source`:
  - **Constants:** `DEFAULT_SOURCE = "https://github.com/hersh-gupta/clef-tune"`, `SOURCE_ENV = "CLEF_TUNE_SOURCE"`.
  - **`SourceError(RuntimeError)`.**
  - **`Source`:** `@dataclass(frozen=True) class Source(repo_url: str, ref: str)` with:
    - `owner_repo -> str`, e.g. `"hersh-gupta/clef-tune"`;
    - `archive_url -> str`: `f"{repo_url}/archive/{ref}.tar.gz"`;
    - `requirement(extras: str = "train") -> str`: `f"clef-tune[{extras}] @ {archive_url}"`;
    - `version -> str`: `"0.0.0+g" + ref[:12]` when `ref` is a 40-hex SHA, else `"0.0.0"`.
  - **`source_repo(env: Mapping[str, str] = os.environ) -> str`.**
  - **`running_commit(direct_url_json: str | None, checkout: Path | None, *, git: Callable[[list[str], Path], str] = run_git) -> str`.**
  - **`installed_direct_url() -> str | None`:** the `direct_url.json` text of the installed `clef-tune` distribution, or `None`.
  - **`checkout_dir() -> Path | None`:** the repo root above `clef_tune/` when it contains `.git`, else `None`.
  - **`run_git(args: list[str], cwd: Path) -> str`:** stdout; failures raise `SourceError`.
  - **`resolve_source(source_ref: str | None = None) -> Source`.**
  - **`pin_commit(source: Source, *, fetch: Callable[[str], dict] = github_json) -> Source`.**
  - **`github_json(url: str) -> dict`:** a GET with a `User-Agent: clef-tune` header and a 10 s timeout.

- [ ] **Step 1: Write the failing tests** `tests/test_source.py`:

```python
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
        {"url": "https://github.com/hersh-gupta/clef-tune", "vcs_info": {"vcs": "git", "commit_id": SHA}}
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
    "value", ["https://github.com/me/fork", "https://github.com/me/fork.git/", " https://github.com/me/fork/ "]
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_source.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clef_tune.launch'`.

- [ ] **Step 3: Implement `clef_tune/launch/source.py`** with the Interfaces above. Each function follows these rules; the test names say the rest.
  - **`running_commit` checks three things in order:**
    1. **Recorded commit:** `vcs_info.commit_id` in the parsed JSON, if present.
    2. **Checkout:** if `checkout` is not `None`, run `git status --porcelain --untracked-files=no`. Non-empty output raises `SourceError(f"uncommitted changes in {checkout}; commit and push them, or pass --source-ref")`. Otherwise return `git rev-parse HEAD`, stripped.
    3. **Otherwise:** raise `SourceError("cannot tell which clef-tune commit is installed; pass --source-ref")`.

    Never read `clef_tune._version`: an editable install freezes it at build time.
  - **`source_repo`:** strip whitespace, then trailing `/` and `.git`. Anything not matching `^https://github\.com/[\w.-]+/[\w.-]+$` raises a `SourceError` that names `CLEF_TUNE_SOURCE`.
  - **`resolve_source`:** `Source(source_repo(), source_ref or running_commit(installed_direct_url(), checkout_dir()))`.
  - **`pin_commit`:** fetch `https://api.github.com/repos/{owner_repo}/commits/{ref}` and return `Source(repo_url, response["sha"])`.
    - HTTP 404 or 422 raises `SourceError(f"{ref} is not on GitHub at {repo_url}; push your commits first, or pass --source-ref")`.
    - Any other `HTTPError`, a `URLError` or `TimeoutError` gives `warnings.warn(f"could not check {ref} on GitHub ({error}); launching anyway")` and returns the source unchanged.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_source.py tests/test_boundaries.py -v`
Expected: all PASS.

- [ ] **Step 5: Lint and commit** (`feat: resolve the exact GitHub commit a job installs`).

---

### Task 2: Job script, kernel pins and setup time in timeouts

**Files:**
- Create: `clef_tune/launch/job_script.py`, `tests/test_job_script.py`
- Modify: `clef_tune/runtime.py`, `Dockerfile`, `clef_tune/estimate.py`, `tests/test_runtime.py`, `tests/test_estimate.py`, `tests/test_boundaries.py` (add `clef_tune.launch.job_script`)

**Interfaces:**
- Consumes: `Source` (Task 1); `runtime.PRICE_ENV`, `runtime.ESTIMATE_ENV`.
- Produces:
  - **`clef_tune.runtime`:**
    - `KERNEL_PINS = {"causal-conv1d": "1.7.0", "flash-linear-attention": "0.6.0"}`;
    - `KERNEL_PACKAGES = ("causal-conv1d==1.7.0", "flash-linear-attention==0.6.0")`, derived from the pins.
  - **`clef_tune.estimate`:** `suggest_timeout(row: EstimateRow, extra_minutes: float = 0) -> str`, where minutes = `ceil((row.hours_high * 60 + extra_minutes) * TIMEOUT_MARGIN / 30) * 30`. The output format is unchanged.
  - **`clef_tune.launch.job_script` constants:**
    - mounts: `WORK_MOUNT = "/work"`, `MODEL_MOUNT = "/model"`, `KERNEL_MOUNT = "/kernels"`;
    - paths: `BUNDLE_CONFIG = "/work/bundle/config.yaml"`, `JOB_OUTPUT_DIR = "/work/out"`.
  - **`clef_tune.launch.job_script` functions:**
    - `kernel_step(cache_dir: str = KERNEL_MOUNT) -> str`;
    - `job_script(source: Source, *, gpus: int, adapter_repo: str, keep_data: bool, kernels: bool) -> str`;
    - `job_command(...same keywords...) -> list[str]`, which returns `["bash", "-c", job_script(...)]`;
    - `job_env(source: Source, *, price_per_hour: float | None, estimated_hours: float | None) -> dict[str, str]`.

- [ ] **Step 1: Write the failing tests.** In `tests/test_job_script.py`:

```python
import shlex
import shutil
import subprocess
import sys

import pytest

from clef_tune.launch.job_script import job_env, job_script, kernel_step
from clef_tune.launch.source import Source
from clef_tune.runtime import ESTIMATE_ENV, KERNEL_PACKAGES, PRICE_ENV

SOURCE = Source("https://github.com/hersh-gupta/clef-tune", "0123456789abcdef0123456789abcdef01234567")
POSIX_BASH = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None, reason="needs a POSIX bash"
)


def script(**overrides):
    options = {"gpus": 1, "adapter_repo": "me/clef-tune-run", "keep_data": False, "kernels": True}
    return job_script(SOURCE, **(options | overrides))


def test_a_single_gpu_job_installs_the_pinned_source_and_trains():
    text = script()
    assert f"pip install --root-user-action=ignore {shlex.quote(SOURCE.requirement())}" in text
    assert text.rstrip().endswith(
        "clef-tune train /work/bundle/config.yaml --push-to-hub me/clef-tune-run "
        "--delete-bundle-on-success"
    )


def test_keep_data_keeps_the_bundle():
    assert script(keep_data=True).rstrip().endswith("--push-to-hub me/clef-tune-run")


def test_multi_gpu_flavors_launch_one_process_per_gpu():
    assert (
        "accelerate launch --multi_gpu --num_processes 4 -m clef_tune train /work/bundle/config.yaml"
        in script(gpus=4)
    )


def test_the_kernel_step_builds_against_the_installed_torch_and_checks_the_import():
    step = kernel_step()
    assert "--no-build-isolation" in step and all(p in step for p in KERNEL_PACKAGES)
    assert 'python -c "import causal_conv1d"' in step
    assert "pip uninstall -y causal-conv1d flash-linear-attention fla-core" in step
    assert step in script()
    assert "causal" not in script(kernels=False)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_job_script_is_valid_bash():
    result = subprocess.run(["bash", "-n"], input=script(gpus=2), text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def _stubs(tmp_path, pip_succeeds):
    """Stand-ins for pip and python that log their arguments."""
    stubs = tmp_path / "bin"
    stubs.mkdir()
    log = tmp_path / "pip.log"
    ok = "exit 0" if pip_succeeds else '[ "$1" = uninstall ]'
    (stubs / "pip").write_text(f'#!/bin/sh\necho "$@" >> {log}\n{ok}\n')
    (stubs / "python").write_text(f"#!/bin/sh\n{'exit 0' if pip_succeeds else 'exit 1'}\n")
    for stub in stubs.iterdir():
        stub.chmod(0o755)
    return stubs, log


@POSIX_BASH
def test_a_failed_kernel_build_falls_back_to_the_slow_path(tmp_path):
    stubs, log = _stubs(tmp_path, pip_succeeds=False)
    cache = tmp_path / "cache"
    cache.mkdir()
    result = subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + kernel_step(str(cache))],
        env={"PATH": f"{stubs}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "slow PyTorch path" in result.stdout
    assert "uninstall -y causal-conv1d flash-linear-attention fla-core" in log.read_text()


@POSIX_BASH
def test_a_cached_kernel_wheel_is_reused(tmp_path):
    stubs, log = _stubs(tmp_path, pip_succeeds=True)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "causal_conv1d-1.7.0-cp312-cp312-linux_x86_64.whl").write_bytes(b"wheel")
    result = subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + kernel_step(str(cache))],
        env={"PATH": f"{stubs}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Using the cached causal-conv1d wheel" in result.stdout
    assert "wheel --no-deps" not in log.read_text()


def test_job_env_feeds_cost_tracking_and_never_sets_reserved_names():
    env = job_env(SOURCE, price_per_hour=0.8, estimated_hours=0.25)
    assert env == {
        "PIP_BREAK_SYSTEM_PACKAGES": "1",
        "PYTHONUNBUFFERED": "1",
        "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CLEF_TUNE": "0.0.0+g0123456789ab",
        PRICE_ENV: "0.8",
        ESTIMATE_ENV: "0.25",
    }
    bare = job_env(SOURCE, price_per_hour=None, estimated_hours=None)
    assert PRICE_ENV not in bare and ESTIMATE_ENV not in bare
    assert not any(name.startswith("NVIDIA_") for name in env)
```

Append to `tests/test_estimate.py`:

```python
def test_suggest_timeout_adds_setup_minutes():
    row = est.EstimateRow(
        flavor="l4x1", gpu="L4", count=1, price_per_hour=0.8, memory_needed_gb=14,
        memory_available_gb=24, fits=True, fits_with_4bit=True, note=None, hours=0.2,
    )
    assert est.suggest_timeout(row) == "30m"  # 0.3 h high x 1.25 = 22.5 min -> 30m
    assert est.suggest_timeout(row, extra_minutes=15) == "1h"  # (18 + 15) x 1.25 = 41.25 -> 60
```

Append to `tests/test_runtime.py`:

```python
def test_the_dockerfile_installs_the_pinned_kernels_without_build_isolation():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "--no-build-isolation" in text
    assert all(package in text for package in runtime.KERNEL_PACKAGES)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_job_script.py tests/test_estimate.py tests/test_runtime.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clef_tune.launch.job_script'`. `test_suggest_timeout_adds_setup_minutes` fails with an unexpected keyword. The Dockerfile test fails because the versions aren't pinned.

- [ ] **Step 3: Implement.**
  - **Runtime pins:** add `KERNEL_PINS` and the derived `KERNEL_PACKAGES` to `runtime.py`.
  - **Dockerfile:** change the kernel `RUN` to install `causal-conv1d==1.7.0 flash-linear-attention==0.6.0`, keeping its `--no-build-isolation`, import check and uninstall fallback.
  - **Timeout:** add `extra_minutes` to `suggest_timeout`.
  - **`job_env`:** fixed names first, then `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CLEF_TUNE = source.version`. Then `PRICE_ENV` and `ESTIMATE_ENV` as `str(value)`, each only when not `None`.
  - **`job_script`:** the lines below, joined with newlines. Quote the requirement and the repo id with `shlex.quote`.

    ```
    set -euo pipefail
    echo "clef-tune job: installing <requirement>"
    pip install --root-user-action=ignore <quoted requirement>
    <kernel_step()>                                   # only when kernels=True
    <launcher> /work/bundle/config.yaml --push-to-hub <quoted repo>[ --delete-bundle-on-success]
    ```

    The launcher is `clef-tune train` for 1 GPU. For more, it is `accelerate launch --multi_gpu --num_processes <gpus> -m clef_tune train`. `--delete-bundle-on-success` is appended unless `keep_data`.

  `kernel_step(cache_dir)` is exactly this, with `{cache}` = `cache_dir` and versions from `KERNEL_PINS`. These lines are the behavior verified on 2026-10-28, so keep them verbatim:

```bash
mkdir -p /tmp/kernel-wheels
if ls {cache}/causal_conv1d-*.whl >/dev/null 2>&1; then
    echo "Using the cached causal-conv1d wheel"
    cp {cache}/causal_conv1d-*.whl /tmp/kernel-wheels/
else
    echo "Building causal-conv1d 1.7.0 (about 12 minutes on 8 vCPUs; cached for later jobs)"
    if pip wheel --no-deps --no-build-isolation -w /tmp/kernel-wheels causal-conv1d==1.7.0; then
        cp /tmp/kernel-wheels/causal_conv1d-*.whl {cache}/ || echo "warning: could not cache the causal-conv1d wheel"
    fi
fi
if ! (pip install --root-user-action=ignore /tmp/kernel-wheels/causal_conv1d-*.whl flash-linear-attention==0.6.0 \
        && python -c "import causal_conv1d"); then
    pip uninstall -y causal-conv1d flash-linear-attention fla-core || true
    echo "warning: kernel packages failed to install; linear attention will use the slow PyTorch path"
fi
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_job_script.py tests/test_estimate.py tests/test_runtime.py tests/test_boundaries.py -v`
Expected: all PASS. On Windows, the two stub tests are SKIPPED; CI on Linux runs them.

- [ ] **Step 5: Lint and commit** (`feat: build the HF Jobs command with pinned, cached and fail-safe kernels`).

---

### Task 3: Launch planning

**Files:**
- Create: `clef_tune/launch/hf_jobs.py`, `tests/test_hf_jobs.py`
- Modify: `tests/test_boundaries.py` (add `clef_tune.launch.hf_jobs`)

**Interfaces:**
- Consumes:
  - `resolve_source`, `pin_commit`, `Source`, `SourceError` (Task 1);
  - `suggest_timeout(row, extra_minutes)` and `KERNEL_PINS` (Task 2);
  - `load_config`, `validate`, `counter_for_config`, the `estimate` functions, `hub.check_write_access`, `runtime.BASE_IMAGE`, `runtime.KERNEL_PINS`.
- Produces, in `clef_tune.launch.hf_jobs`:
  - **Constants:**
    - `TOOL_LABEL = {"tool": "clef-tune"}`, `LABEL_KEYS = ("tool", "run", "timeout")`;
    - `RUNS_FOLDER = "clef-tune-runs"`, `KERNEL_CACHE_ROOT = "clef-tune-kernels"`;
    - `KERNEL_BUILD_MINUTES = 15`, `BILLING_URL = "https://huggingface.co/settings/billing"`.
  - **`LaunchError(RuntimeError)`.**
  - **`JobRequest`:** `@dataclass(frozen=True) class JobRequest(flavor: str | None = None, timeout: str | None = None, keep_data: bool = False, kernels: bool = True, adapter_repo: str | None = None, source_ref: str | None = None)`.
  - **`LaunchPlan`:** `@dataclass(frozen=True) class LaunchPlan(config: Config, report: ValidationReport, source: Source, namespace: str, adapter_repo: str, flavor: str, gpus: int, timeout: str, price_per_hour: float | None, estimate: EstimateRow | None, kernels: bool, kernels_cached: bool, keep_data: bool)`, with:
    - `estimated_hours -> float | None` (from `estimate.hours`);
    - `max_cost -> float | None`: `timeout_hours(timeout) * price_per_hour`.
  - **Helpers:**
    - `jobs_bucket(namespace: str) -> str`: `f"{namespace}/{constants.HF_JOBS_ARTIFACTS_BUCKET_NAME}"`;
    - `kernel_cache_folder() -> str`;
    - `timeout_hours(timeout: str) -> float`.
  - **`plan_launch`:** `plan_launch(config_path: str | Path, request: JobRequest = JobRequest(), *, api: HfApi | None = None, token: str | None = None, counter: TokenCounter | None = None, prices: dict[str, float] | None = None, source: Source | None = None) -> LaunchPlan`.

- [ ] **Step 1: Write the failing tests** `tests/test_hf_jobs.py`. Tasks 4 and 5 extend this file and its `FakeApi`.

```python
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub import Volume

from clef_tune.config import load_config
from clef_tune.estimate import HF_FLAVORS, suggest_timeout
from clef_tune.launch import hf_jobs
from clef_tune.launch.hf_jobs import JobRequest, LaunchError, kernel_cache_folder, plan_launch
from clef_tune.launch.source import Source, SourceError
from clef_tune.yamlio import dump_yaml
from tests.helpers import SCHEMA, CharCounter, ticket_rows, write_csv

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "support_tickets.yaml"
SOURCE = Source("https://github.com/hersh-gupta/clef-tune", "0123456789abcdef0123456789abcdef01234567")
WRITE = {"name": "me", "auth": {"type": "access_token", "accessToken": {"role": "write"}}}
READ = {"name": "me", "auth": {"type": "access_token", "accessToken": {"role": "read"}}}
TOKEN = "hf_test"


class FakeApi:
    """Records every Hub call; returns canned answers."""

    def __init__(self, *, token_info=WRITE, bucket_files=(), run_error=None, job_info=None,
                 metrics=None, jobs=(), logs=()):
        self.calls, self.staged = [], {}
        self.token_info, self.bucket_files, self.run_error = token_info, list(bucket_files), run_error
        self.job_info, self.metrics, self.jobs, self.logs = job_info, metrics, list(jobs), list(logs)

    def whoami(self, token=None, **kwargs):
        self.calls.append(("whoami",))
        return self.token_info

    def list_bucket_tree(self, bucket_id, prefix=None, *, recursive=None, token=None):
        self.calls.append(("list_bucket_tree", bucket_id, prefix))
        return [SimpleNamespace(type="file", path=p) for p in self.bucket_files if p.startswith(prefix or "")]

    def sync_job_volume(self, source, mount_path, *, remote_name=None, read_only=True, namespace=None, token=None):
        self.calls.append(("sync_job_volume", mount_path, remote_name, read_only))
        bundle = Path(source) / "bundle"
        if (bundle / "config.yaml").exists():
            self.staged = {
                "config": load_config(bundle / "config.yaml"),
                "files": sorted(p.name for p in bundle.iterdir()),
            }
        return Volume(type="bucket", source="me/jobs-artifacts", mount_path=mount_path,
                      path=remote_name, read_only=read_only)


def project(tmp_path, rows=None, **sections):
    write_csv(tmp_path / "train.csv", rows or ticket_rows(40))
    data = {"run_name": "t", "data": {"train": "train.csv", "id_column": "id"}, "schema": SCHEMA, **sections}
    path = tmp_path / "config.yaml"
    path.write_text(dump_yaml(data), encoding="utf-8")
    return path


def plan(api=None, request=JobRequest(), config=EXAMPLE, source=SOURCE):
    return plan_launch(config, request, api=api or FakeApi(), token=TOKEN, counter=CharCounter(),
                       prices={}, source=source)


def test_the_plan_picks_the_cheapest_fitting_flavor_and_allows_for_the_kernel_build():
    api = FakeApi()
    p = plan(api)
    assert p.estimate.fits and p.flavor == p.estimate.flavor and p.gpus == HF_FLAVORS[p.flavor].count
    assert p.timeout == suggest_timeout(p.estimate, extra_minutes=hf_jobs.KERNEL_BUILD_MINUTES)
    assert p.adapter_repo == "me/clef-tune-support-tickets" and p.kernels and not p.kernels_cached
    assert p.price_per_hour == HF_FLAVORS[p.flavor].price_per_hour
    assert {call[0] for call in api.calls} <= {"whoami", "list_bucket_tree"}  # nothing uploaded or started


def test_the_kernel_cache_folder_names_the_image_and_version():
    assert kernel_cache_folder() == (
        "clef-tune-kernels/pytorch-pytorch-2.11.0-cuda13.0-cudnn9-devel/causal-conv1d-1.7.0"
    )


def test_a_cached_kernel_wheel_needs_no_build_time():
    folder = kernel_cache_folder()
    api = FakeApi(bucket_files=[f"{folder}/.keep", f"{folder}/causal_conv1d-1.7.0-cp312-cp312-linux_x86_64.whl"])
    p = plan(api)
    assert p.kernels_cached and p.timeout == suggest_timeout(p.estimate)
    assert ("list_bucket_tree", "me/jobs-artifacts", folder) in api.calls


def test_without_kernels_there_is_no_cache_lookup_or_build_time():
    api = FakeApi()
    p = plan(api, JobRequest(kernels=False))
    assert not p.kernels and p.timeout == suggest_timeout(p.estimate)
    assert all(call[0] != "list_bucket_tree" for call in api.calls)


def test_a_multi_gpu_flavor_runs_one_process_per_gpu():
    assert plan(request=JobRequest(flavor="h200x2")).gpus == 2


def test_an_explicit_timeout_and_its_maximum_cost():
    p = plan(request=JobRequest(flavor="h200", timeout="2h"))
    assert p.timeout == "2h" and p.max_cost == pytest.approx(2 * HF_FLAVORS["h200"].price_per_hour)


@pytest.mark.parametrize("timeout", ["90", "1 h", "h", "2w"])
def test_a_timeout_needs_a_number_and_a_unit(timeout):
    with pytest.raises(LaunchError, match="timeout"):
        plan(request=JobRequest(flavor="h200", timeout=timeout))


def test_an_unknown_flavor_lists_the_known_ones():
    with pytest.raises(LaunchError, match="a10g-small"):
        plan(request=JobRequest(flavor="tpu-v5"))


@pytest.mark.parametrize("flavor, reason", [("t4-small", "bf16"), ("a10g-small", "needs ~")])
def test_a_flavor_that_cannot_run_the_config_is_refused(tmp_path, flavor, reason):
    config = project(tmp_path)  # Cloudflare/clef in bf16: ~87 GB
    with pytest.raises(LaunchError, match=f"cannot run this config.*{reason}"):
        plan(config=config, request=JobRequest(flavor=flavor))


def test_invalid_data_is_refused_before_touching_the_hub(tmp_path, monkeypatch):
    rows = ticket_rows(20)
    rows[1]["department"] = "biling"
    monkeypatch.setattr(hf_jobs, "resolve_source", lambda ref: pytest.fail("resolved the source"))
    api = FakeApi()
    with pytest.raises(LaunchError, match="validation failed"):
        plan(api, config=project(tmp_path, rows), source=None)
    assert api.calls == []


def test_a_dirty_checkout_is_refused_before_touching_the_hub(monkeypatch):
    def dirty(ref):
        raise SourceError("uncommitted changes in /src; commit and push them, or pass --source-ref")

    monkeypatch.setattr(hf_jobs, "resolve_source", dirty)
    api = FakeApi()
    with pytest.raises(LaunchError, match="uncommitted changes"):
        plan(api, source=None)
    assert api.calls == []


def test_the_source_is_resolved_and_pinned_when_not_given(monkeypatch):
    monkeypatch.setattr(hf_jobs, "resolve_source", lambda ref: Source(SOURCE.repo_url, ref or "main"))
    monkeypatch.setattr(hf_jobs, "pin_commit", lambda source: SOURCE)
    assert plan(request=JobRequest(source_ref="v1"), source=None).source == SOURCE


def test_the_adapter_repo_needs_write_access():
    with pytest.raises(LaunchError, match="write access"):
        plan(FakeApi(token_info=READ))


def test_the_requested_adapter_repo_wins(tmp_path):
    config = project(tmp_path, model={"id": "Cloudflare/clef-flash"}, output={"push_to_hub": "me/cfg"})
    assert plan(config=config).adapter_repo == "me/cfg"
    assert plan(config=config, request=JobRequest(adapter_repo="me/x")).adapter_repo == "me/x"


def test_a_token_is_required(monkeypatch):
    monkeypatch.setattr(hf_jobs, "get_token", lambda: None)
    with pytest.raises(LaunchError, match="hf auth login"):
        plan_launch(EXAMPLE, api=FakeApi(), token=None, counter=CharCounter(), prices={}, source=SOURCE)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_hf_jobs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clef_tune.launch.hf_jobs'`.

- [ ] **Step 3: Implement `plan_launch`** and the helpers in `clef_tune/launch/hf_jobs.py`.
  - **Imports:** import `resolve_source`, `pin_commit` and `huggingface_hub.get_token` by name into `hf_jobs`, since the tests patch `hf_jobs.resolve_source`, `hf_jobs.pin_commit` and `hf_jobs.get_token`.
  - **Calls:** call them positionally: `pin_commit(resolve_source(request.source_ref))` and `get_token()`.
  - **The CLI (Task 6)** calls `plan_launch(config, request)` and `submit_launch(plan)` positionally too.

  `plan_launch` runs in this order, raising `LaunchError` at the first failure.
  1. **Validate locally.** `load_config` (a `ValueError` or `OSError` becomes `LaunchError(f"invalid config: {exc}")`), then `validate(config, counter or counter_for_config(config))`. Errors become `LaunchError("validation failed:\n" + one issue per line)`. Nothing has touched the network yet.
  2. **Pick the source.** Use `source`, or else `pin_commit(resolve_source(request.source_ref))`. A `SourceError` becomes `LaunchError(str(error))`.
  3. **Find the token and the adapter repo.**
     - Token: `token or get_token()`. `None` raises `LaunchError("no Hugging Face token: run `hf auth login` or set HF_TOKEN")`.
     - Namespace: `api.whoami(token=token)["name"]`.
     - Adapter repo: `request.adapter_repo or config.output.push_to_hub or f"{namespace}/clef-tune-{config.run_name}"`.
     - Write access: `check_write_access(adapter_repo, token=token, api=api)`. A `HubError` becomes a `LaunchError`.
  4. **Check the kernel cache** (when `request.kernels`). The wheel is cached if any path under `kernel_cache_folder()` in `jobs_bucket(namespace)` ends in `.whl`. Any exception, for example no bucket yet, means not cached.
  5. **Choose the hardware.**
     - **Flavor:** it must be a key of `HF_FLAVORS`; otherwise list the known names. When `model_spec` and `report.tokens` exist, estimate with `estimate_hf_flavors(spec, workload_from(config, report.tokens), prices if prices is not None else fetch_hf_prices())`. Use the requested flavor's row, or `recommend(rows)`; `None` means no flavor fits.
     - **Unfit flavor:** a row with `fits=False` raises `LaunchError(f"{flavor} cannot run this config: {row.note}")`.
     - **No estimate:** without one, `request.flavor` and `request.timeout` are both required.
     - **Timeout:** `request.timeout` must match `^\d+(\.\d+)?[smhd]$`. It defaults to `suggest_timeout(row, extra_minutes=KERNEL_BUILD_MINUTES if request.kernels and not cached else 0)`.
     - **GPUs and price:** `gpus = HF_FLAVORS[flavor].count`, and `price_per_hour` comes from the live price, else the fallback table.

  `kernel_cache_folder()` is `f"{KERNEL_CACHE_ROOT}/{slug}/causal-conv1d-{KERNEL_PINS['causal-conv1d']}"`, where `slug = re.sub(r"[^A-Za-z0-9._-]+", "-", BASE_IMAGE)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_hf_jobs.py tests/test_boundaries.py -v`
Expected: all PASS.

- [ ] **Step 5: Lint and commit** (`feat: plan an HF Jobs launch: validate, pin the source, estimate, pick hardware`).

---

### Task 4: Submitting a job

**Files:**
- Modify: `clef_tune/launch/hf_jobs.py`, `tests/test_hf_jobs.py`

**Interfaces:**
- Consumes: `LaunchPlan` and the helpers from Task 3; `job_command`, `job_env`, `WORK_MOUNT`, `MODEL_MOUNT`, `KERNEL_MOUNT`, `JOB_OUTPUT_DIR` (Task 2); `bundle.write_bundle`.
- Produces, in `clef_tune.launch.hf_jobs`:
  - **`LaunchedJob`:** `@dataclass(frozen=True) class LaunchedJob(id: str, url: str, work_folder: str, plan: LaunchPlan)`.
  - **`run_folder(run_name: str, now: datetime) -> str`:** `f"{RUNS_FOLDER}/{run_name}-{now:%Y%m%d-%H%M%S}"`.
  - **`stage_bundle(plan: LaunchPlan, staging_dir: Path) -> Path`:** writes `staging_dir/bundle/` and returns its `config.yaml`.
  - **`submit_launch`:** `submit_launch(plan: LaunchPlan, *, api: HfApi | None = None, token: str | None = None, now: datetime | None = None) -> LaunchedJob`.
  - **`start_job(api: HfApi, **run_job_kwargs) -> JobInfo`:** calls `api.run_job(**run_job_kwargs)`; the kwargs include `token`. HTTP 402 becomes `LaunchError(f"Hugging Face Jobs needs pre-paid credits; add them at {BILLING_URL}")`; any other error becomes `LaunchError(f"could not start the job: {error}")`. Task 5's resume reuses it.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_hf_jobs.py`; add the imports `from datetime import UTC, datetime, timedelta`, `import httpx`, `from huggingface_hub.errors import HfHubHTTPError`, `from clef_tune.config import PINNED_REVISIONS`, `from clef_tune.launch.hf_jobs import submit_launch`, `from clef_tune.launch.job_script import job_command, job_env`, `from clef_tune.runtime import BASE_IMAGE`). Add these `FakeApi` methods:

```python
    def run_job(self, **kwargs):
        self.calls.append(("run_job", kwargs))
        if self.run_error:
            raise self.run_error
        return SimpleNamespace(id="job123", url="https://huggingface.co/jobs/me/job123")

    def batch_bucket_files(self, bucket_id, *, add=None, delete=None, token=None, **kwargs):
        self.calls.append(("batch_bucket_files", bucket_id, delete))
```

Then the tests:

```python
NOW = datetime(2026, 10, 28, 12, 0, 0, tzinfo=UTC)
FOLDER = "clef-tune-runs/support-tickets-20261028-120000"


def run_call(api):
    return next(call[1] for call in api.calls if call[0] == "run_job")


def test_submit_uploads_the_bundle_and_starts_the_job():
    api = FakeApi()
    p = plan(api)
    job = submit_launch(p, api=api, token=TOKEN, now=NOW)
    assert (job.id, job.url, job.work_folder) == ("job123", "https://huggingface.co/jobs/me/job123", FOLDER)
    syncs = [call for call in api.calls if call[0] == "sync_job_volume"]
    assert syncs == [
        ("sync_job_volume", "/work", FOLDER, False),
        ("sync_job_volume", "/kernels", kernel_cache_folder(), False),
    ]
    run = run_call(api)
    assert run["image"] == BASE_IMAGE and run["flavor"] == p.flavor and run["timeout"] == p.timeout
    assert run["secrets"] == {"HF_TOKEN": TOKEN} and run["token"] == TOKEN
    assert run["labels"] == {"tool": "clef-tune", "run": "support-tickets", "timeout": p.timeout}
    assert run["command"] == job_command(
        SOURCE, gpus=p.gpus, adapter_repo="me/clef-tune-support-tickets", keep_data=False, kernels=True
    )
    assert run["env"] == job_env(SOURCE, price_per_hour=p.price_per_hour, estimated_hours=p.estimated_hours)
    work, model, kernels = run["volumes"]
    assert (work.mount_path, work.path, work.read_only) == ("/work", FOLDER, False)
    assert model == Volume(
        type="model", source="Cloudflare/clef-flash", mount_path="/model",
        revision=PINNED_REVISIONS["Cloudflare/clef-flash"],
    )
    assert kernels.mount_path == "/kernels"


def test_the_staged_bundle_points_at_the_job_mounts(tmp_path):
    config = project(
        tmp_path,
        model={"id": "Cloudflare/clef-flash", "local_path": "weights"},
        output={"dir": "runs/x", "push_to_hub": "me/other"},
    )
    api = FakeApi()
    submit_launch(plan(api, config=config), api=api, token=TOKEN, now=NOW)
    staged = api.staged["config"]
    assert staged.model.local_path == "/model" and staged.output.dir == "/work/out"
    assert staged.output.push_to_hub is None  # the job passes --push-to-hub
    assert api.staged["files"] == ["config.yaml", "eval.jsonl", "train.jsonl"]


def test_keep_data_keeps_the_bundle():
    api = FakeApi()
    submit_launch(plan(api, JobRequest(keep_data=True)), api=api, token=TOKEN, now=NOW)
    assert "--delete-bundle-on-success" not in run_call(api)["command"][-1]


def test_without_kernels_no_cache_is_mounted():
    api = FakeApi()
    submit_launch(plan(api, JobRequest(kernels=False)), api=api, token=TOKEN, now=NOW)
    run = run_call(api)
    assert len(run["volumes"]) == 2 and "causal" not in run["command"][-1]


def test_each_launch_gets_its_own_work_folder():
    api = FakeApi()
    p = plan(api)
    first = submit_launch(p, api=api, token=TOKEN, now=NOW)
    second = submit_launch(p, api=api, token=TOKEN, now=NOW + timedelta(seconds=1))
    assert first.work_folder != second.work_folder


@pytest.mark.parametrize(
    "status, message", [(402, r"pre-paid credits.*settings/billing"), (500, "could not start the job")]
)
def test_out_of_credits_is_explained_and_the_upload_removed(status, message):
    request = httpx.Request("POST", "https://huggingface.co/api/jobs/me")
    error = HfHubHTTPError(f"{status} error", response=httpx.Response(status, request=request))
    staged = [f"{FOLDER}/bundle/config.yaml", f"{FOLDER}/bundle/train.jsonl"]
    api = FakeApi(run_error=error, bucket_files=staged)
    with pytest.raises(LaunchError, match=message):
        submit_launch(plan(api), api=api, token=TOKEN, now=NOW)
    assert ("batch_bucket_files", "me/jobs-artifacts", staged) in api.calls
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_hf_jobs.py -v`
Expected: the new tests FAIL with `ImportError: cannot import name 'submit_launch'`.

- [ ] **Step 3: Implement.**
  - **`stage_bundle`:** call `write_bundle(plan.report, plan.config, staging_dir / "bundle")`. Then reload the written YAML, set `model.local_path = MODEL_MOUNT` and `output.dir = JOB_OUTPUT_DIR`, drop `output.push_to_hub`, and save it back through `Config.model_validate(...).save(...)`.
  - **`submit_launch`:** `now` defaults to `datetime.now(UTC)`, and `folder = run_folder(run_name, now)`.
  - **Uploads:** inside a `tempfile.TemporaryDirectory`:
    - stage the bundle in `<tmp>/stage`, then `api.sync_job_volume(<tmp>/stage, WORK_MOUNT, remote_name=folder, read_only=False, token=token)`;
    - when `plan.kernels`, `api.sync_job_volume(<an empty tmp dir>, KERNEL_MOUNT, remote_name=kernel_cache_folder(), read_only=False, token=token)`. An empty folder uploads a `.keep`, and the sync never deletes remote files, so a cached wheel survives.
  - **Volumes:** `[work, Volume(type="model", source=config.model.id, mount_path=MODEL_MOUNT, revision=config.model.resolved_revision()), kernel cache?]`.
  - **Start the job:** `start_job(api, image=BASE_IMAGE, command=job_command(...), env=job_env(...), secrets={"HF_TOKEN": token}, flavor=..., timeout=..., labels={"tool": "clef-tune", "run": run_name, "timeout": timeout}, volumes=..., token=token)`.
  - **On a `LaunchError` from `start_job`:** delete every file under `folder` (`list_bucket_tree(bucket, prefix=folder, recursive=True)`, then `batch_bucket_files(bucket, delete=[paths])`, ignoring cleanup errors). Then re-raise.
  - **Status code:** read it from `getattr(getattr(error, "response", None), "status_code", None)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_hf_jobs.py -v`
Expected: all PASS.

- [ ] **Step 5: Lint and commit** (`feat: stage the bundle in the jobs bucket and start the HF Job`).

---

### Task 5: Listing, status, logs, cancel and resume

**Files:**
- Modify: `clef_tune/launch/hf_jobs.py`, `tests/test_hf_jobs.py`

**Interfaces:**
- Consumes: `start_job`, `LABEL_KEYS`, `TOOL_LABEL` (Tasks 3–4); `WORK_MOUNT` (Task 2); `runtime.PRICE_ENV`.
- Produces, in `clef_tune.launch.hf_jobs`:
  - **`JobStatus`:** `@dataclass(frozen=True) class JobStatus(id: str, url: str, run: str | None, flavor: str | None, stage: str, message: str | None, hours: float | None, cost: float | None, progress: list[dict])`.
    - `latest(event: str) -> dict | None` returns the last progress record with that `event`.
    - `stage` is one of `COMPLETED`, `CANCELED`, `ERROR`, `DELETED`, `SCHEDULING`, `RUNNING`.
  - **`list_jobs(*, api=None, token=None) -> list[JobInfo]`:** clef-tune jobs, newest first.
  - **`job_status(job_id: str, *, api=None, token=None, now: datetime | None = None) -> JobStatus`.**
  - **`job_progress(info: JobInfo, *, api=None, token=None) -> list[dict]`:** the `metrics.jsonl` records written so far.
  - **`job_logs(job_id: str, *, follow: bool = False, api=None, token=None) -> Iterable[str]`.**
  - **`cancel_job(job_id: str, *, api=None, token=None) -> None`.**
  - **`resume_job(job_id: str, *, timeout: str | None = None, api=None, token=None) -> JobInfo`.**

- [ ] **Step 1: Write the failing tests** (append; add `import json`, the new names to the `hf_jobs` import, and `from clef_tune.runtime import PRICE_ENV`). Add these `FakeApi` methods:

```python
    def inspect_job(self, *, job_id, token=None, **kwargs):
        self.calls.append(("inspect_job", job_id))
        return self.job_info

    def list_jobs(self, *, labels=None, token=None, **kwargs):
        self.calls.append(("list_jobs", labels))
        return list(self.jobs)

    def download_bucket_files(self, bucket_id, files, *, raise_on_missing_files=False, token=None):
        for remote, local in files:
            self.calls.append(("download_bucket_files", bucket_id, remote))
            if self.metrics is not None:
                Path(local).write_text("\n".join(json.dumps(m) for m in self.metrics), encoding="utf-8")

    def fetch_job_logs(self, *, job_id, follow=False, token=None, **kwargs):
        self.calls.append(("fetch_job_logs", job_id, follow))
        return iter(self.logs)

    def cancel_job(self, *, job_id, token=None, **kwargs):
        self.calls.append(("cancel_job", job_id))
```

Then the tests:

```python
WORK = Volume(type="bucket", source="me/jobs-artifacts", mount_path="/work", path="clef-tune-runs/r-1", read_only=False)


def job(stage="RUNNING", **overrides):
    values = {
        "id": "job123",
        "url": "https://huggingface.co/jobs/me/job123",
        "status": SimpleNamespace(stage=stage, message=None),
        "labels": {"tool": "clef-tune", "run": "r", "timeout": "1h", "name": "set-by-hf"},
        "docker_image": BASE_IMAGE,
        "command": ["bash", "-c", "echo train"],
        "environment": {"PYTHONUNBUFFERED": "1", PRICE_ENV: "0.8"},
        "flavor": "l4x1",
        "volumes": [WORK, Volume(type="model", source="Cloudflare/clef-flash", mount_path="/model")],
        "created_at": NOW,
        "started_at": NOW - timedelta(minutes=30),
        "finished_at": None,
    }
    return SimpleNamespace(**(values | overrides))


def test_list_jobs_filters_by_the_tool_label_newest_first():
    api = FakeApi(jobs=[job(id="a", created_at=NOW - timedelta(hours=1)), job(id="b")])
    assert [j.id for j in hf_jobs.list_jobs(api=api, token=TOKEN)] == ["b", "a"]
    assert ("list_jobs", {"tool": "clef-tune"}) in api.calls


def test_status_reads_progress_from_the_work_folder_and_prices_the_time():
    metrics = [
        {"event": "baseline", "step": 0, "exact_match": 0.1},
        {"event": "train", "step": 20, "loss": 0.5, "tokens_per_sec": 400.0, "eta_s": 600.0},
        {"event": "eval", "step": 20, "exact_match": 0.4},
    ]
    api = FakeApi(job_info=job(), metrics=metrics)
    status = job_status("job123", api=api, token=TOKEN, now=NOW)
    assert (status.stage, status.run, status.flavor) == ("RUNNING", "r", "l4x1")
    assert status.hours == pytest.approx(0.5) and status.cost == pytest.approx(0.4)
    assert status.latest("train")["step"] == 20 and status.latest("eval")["exact_match"] == 0.4
    assert ("download_bucket_files", "me/jobs-artifacts", "clef-tune-runs/r-1/out/metrics.jsonl") in api.calls


def test_status_before_the_job_starts_has_no_time_cost_or_progress():
    api = FakeApi(job_info=job(stage="SCHEDULING", started_at=None))
    status = job_status("job123", api=api, token=TOKEN, now=NOW)
    assert (status.hours, status.cost, status.progress) == (None, None, [])


def test_a_finished_job_is_priced_to_its_finish_time():
    finished = job(stage="COMPLETED", finished_at=NOW - timedelta(minutes=15))
    status = job_status("job123", api=FakeApi(job_info=finished), token=TOKEN, now=NOW)
    assert status.hours == pytest.approx(0.25)


def test_resume_rebuilds_the_failed_job_with_a_fresh_token():
    original = job(stage="ERROR")
    api = FakeApi(job_info=original)
    resume_job("job123", api=api, token=TOKEN)
    assert run_call(api) == {
        "image": BASE_IMAGE,
        "command": original.command,
        "env": original.environment,
        "secrets": {"HF_TOKEN": TOKEN},
        "flavor": "l4x1",
        "timeout": "1h",
        "labels": {"tool": "clef-tune", "run": "r", "timeout": "1h"},  # HF's own labels dropped
        "volumes": original.volumes,  # same /work folder: training auto-resumes from its checkpoint
        "token": TOKEN,
    }


def test_resume_can_extend_the_timeout():
    api = FakeApi(job_info=job(stage="CANCELED"))
    resume_job("job123", timeout="3h", api=api, token=TOKEN)
    run = run_call(api)
    assert run["timeout"] == "3h" and run["labels"]["timeout"] == "3h"


@pytest.mark.parametrize("stage", ["RUNNING", "SCHEDULING", "COMPLETED"])
def test_only_failed_or_canceled_jobs_resume(stage):
    with pytest.raises(LaunchError, match="only failed or canceled"):
        resume_job("job123", api=FakeApi(job_info=job(stage=stage)), token=TOKEN)


def test_other_tools_jobs_do_not_resume():
    with pytest.raises(LaunchError, match="not a clef-tune job"):
        resume_job("job123", api=FakeApi(job_info=job(stage="ERROR", labels={})), token=TOKEN)


def test_logs_and_cancel_pass_the_job_through():
    api = FakeApi(logs=["line 1", "line 2"])
    assert list(job_logs("job123", follow=True, api=api, token=TOKEN)) == ["line 1", "line 2"]
    cancel_job("job123", api=api, token=TOKEN)
    assert ("fetch_job_logs", "job123", True) in api.calls and ("cancel_job", "job123") in api.calls
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_hf_jobs.py -v`
Expected: the new tests FAIL with `ImportError` for `job_status`.

- [ ] **Step 3: Implement** the Interfaces above.
  - **Stage:** `str(getattr(stage, "value", stage))`, since the real `JobStage` is an enum.
  - **`job_status` time:** `hours` runs from `started_at` to `finished_at`, or to `now` (default `datetime.now(UTC)`) while running; `None` before the job starts.
  - **`job_status` cost:** `cost = hours * float(environment[PRICE_ENV])` when that variable is present.
  - **`job_progress`:** find the bucket volume mounted at `WORK_MOUNT`. Download `<path>/out/metrics.jsonl` with `download_bucket_files(volume.source, [(remote, local)], raise_on_missing_files=False)` into a temporary file. Parse one JSON object per line, skip lines that fail to parse, and return `[]` when the file is missing.
  - **`resume_job` guards:**
    - `labels.get("tool") != "clef-tune"` raises `LaunchError(f"{job_id} is not a clef-tune job")`;
    - a stage other than `ERROR` or `CANCELED` raises `LaunchError(f"{job_id} is {stage}; only failed or canceled jobs can be resumed")`.
  - **`resume_job` call:** `start_job(api, image=info.docker_image, command=info.command, env=info.environment, secrets={"HF_TOKEN": token}, flavor=info.flavor, timeout=<timeout or labels["timeout"]>, labels={k: v for k in LABEL_KEYS} with the new timeout, volumes=info.volumes, token=token)`.

    `JobInfo` has no timeout field, which is why launches store it as a label. `info.secrets` holds only names, so the token is always fresh.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_hf_jobs.py -v`
Expected: all PASS.

- [ ] **Step 5: Lint and commit** (`feat: list, watch, cancel and resume clef-tune HF Jobs`).

---

### Task 6: `clef-tune jobs` commands and the HF Jobs quickstart

**Files:**
- Modify: `clef_tune/cli.py`, `README.md`
- Create: `tests/test_cli_jobs.py`

**Interfaces:**
- Consumes: from `clef_tune.launch.hf_jobs`, `JobRequest`, `LaunchError`, `plan_launch`, `submit_launch`, `list_jobs`, `job_status`, `job_logs`, `cancel_job` and `resume_job`, imported at the top of `cli.py` (they are torch-free).
- Produces:
  - **`jobs` sub-app:** `jobs_app = typer.Typer(no_args_is_help=True, help="Train on Hugging Face Jobs.")`, registered with `app.add_typer(jobs_app, name="jobs")`.
  - **`launch CONFIG`** options: `--flavor`, `--timeout`, `--keep-data`, `--source-ref`, `--push-to-hub REPO` (it maps to `JobRequest.adapter_repo`), `--no-kernels`, `--yes/-y`.
  - **Other commands:** `list`, `status JOB_ID`, `logs JOB_ID [--follow/-f]`, `cancel JOB_ID`, `resume JOB_ID [--timeout]`.
  - **Exit codes:** 0 ok, 1 for a `LaunchError` or a declined confirmation.

- [ ] **Step 1: Write the failing tests** `tests/test_cli_jobs.py`:

```python
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from clef_tune import cli
from clef_tune.launch.hf_jobs import JobRequest, JobStatus, LaunchError

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "support_tickets.yaml"
runner = CliRunner()


def fake_plan(**overrides):
    values = {
        "config": SimpleNamespace(run_name="support-tickets"),
        "source": SimpleNamespace(repo_url="https://github.com/hersh-gupta/clef-tune",
                                  ref="0123456789abcdef0123456789abcdef01234567"),
        "adapter_repo": "me/clef-tune-support-tickets",
        "flavor": "l40sx1", "gpus": 1, "timeout": "1h",
        "estimate": SimpleNamespace(hours_low=0.2, hours_high=0.45, cost_low=0.36, cost_high=0.81),
        "max_cost": 1.8, "kernels": True, "kernels_cached": False, "keep_data": False,
    }
    return SimpleNamespace(**(values | overrides))


@pytest.fixture
def launcher(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "plan_launch", lambda config, request, **k: calls.append(("plan", request)) or fake_plan())
    monkeypatch.setattr(
        cli, "submit_launch",
        lambda plan, **k: calls.append(("submit",)) or SimpleNamespace(
            id="job123", url="https://huggingface.co/jobs/me/job123", work_folder="clef-tune-runs/x"),
    )
    return calls


def test_launch_shows_the_estimate_and_asks_before_spending(launcher):
    result = runner.invoke(cli.app, ["jobs", "launch", str(EXAMPLE)], input="n\n")
    assert result.exit_code == 1
    for text in ("l40sx1", "1h", "$0.36", "$0.81", "$1.80", "me/clef-tune-support-tickets", "0123456789ab"):
        assert text in result.output
    assert launcher == [("plan", JobRequest())]


def test_launch_with_yes_passes_every_option_and_prints_the_job(launcher):
    args = ["jobs", "launch", str(EXAMPLE), "--yes", "--flavor", "h200", "--timeout", "6h", "--keep-data",
            "--no-kernels", "--source-ref", "v1", "--push-to-hub", "org/x"]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    request = JobRequest(flavor="h200", timeout="6h", keep_data=True, kernels=False,
                         adapter_repo="org/x", source_ref="v1")
    assert launcher == [("plan", request), ("submit",)]
    assert "job123" in result.output and "clef-tune jobs logs job123 --follow" in result.output


def test_launch_errors_exit_1(monkeypatch):
    def refuse(config, request, **kwargs):
        raise LaunchError("v1 is not on GitHub at https://github.com/hersh-gupta/clef-tune")

    monkeypatch.setattr(cli, "plan_launch", refuse)
    result = runner.invoke(cli.app, ["jobs", "launch", str(EXAMPLE)])
    assert result.exit_code == 1 and "not on GitHub" in result.output


def test_list_shows_each_job(monkeypatch):
    job = SimpleNamespace(id="job123", labels={"run": "r"}, flavor="l4x1",
                          status=SimpleNamespace(stage="RUNNING"), created_at=datetime(2026, 10, 28, tzinfo=UTC))
    monkeypatch.setattr(cli, "list_jobs", lambda **k: [job])
    result = runner.invoke(cli.app, ["jobs", "list"])
    assert result.exit_code == 0 and all(t in result.output for t in ("job123", "r", "l4x1", "RUNNING"))


def test_status_shows_progress_and_cost(monkeypatch):
    status = JobStatus(
        id="job123", url="https://huggingface.co/jobs/me/job123", run="r", flavor="l4x1", stage="RUNNING",
        message=None, hours=0.5, cost=0.4,
        progress=[{"event": "train", "step": 20, "loss": 0.5, "tokens_per_sec": 400.0, "eta_s": 600.0},
                  {"event": "eval", "step": 20, "exact_match": 0.4, "mean_macro_f1": 0.7}],
    )
    monkeypatch.setattr(cli, "job_status", lambda job_id, **k: status)
    result = runner.invoke(cli.app, ["jobs", "status", "job123"])
    assert result.exit_code == 0
    assert all(t in result.output for t in ("RUNNING", "$0.40", "step 20", "ETA 10 min", "0.400"))


def test_logs_print_lines_verbatim(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "job_logs", lambda job_id, follow, **k: seen.update(follow=follow) or iter(["[red]x"]))
    result = runner.invoke(cli.app, ["jobs", "logs", "job123", "-f"])
    assert result.exit_code == 0 and "[red]x" in result.output and seen == {"follow": True}


def test_cancel_and_resume(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "cancel_job", lambda job_id, **k: calls.append(("cancel", job_id)))
    monkeypatch.setattr(cli, "resume_job", lambda job_id, timeout=None, **k: calls.append(("resume", job_id, timeout))
                        or SimpleNamespace(id="job456", url="https://huggingface.co/jobs/me/job456"))
    assert runner.invoke(cli.app, ["jobs", "cancel", "job123"]).exit_code == 0
    result = runner.invoke(cli.app, ["jobs", "resume", "job123", "--timeout", "3h"])
    assert result.exit_code == 0 and "job456" in result.output
    assert calls == [("cancel", "job123"), ("resume", "job123", "3h")]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_cli_jobs.py -v`
Expected: FAIL with `AttributeError: module 'clef_tune.cli' has no attribute 'plan_launch'`.

- [ ] **Step 3: Implement the commands.** Print with the existing `console`, escaping user and job text with `rich.markup.escape`. Print log lines with `typer.echo`, so markup in logs stays literal.
  - **`jobs launch` summary** (one line each; reuse `_money`):

    ```
    Run support-tickets from https://github.com/hersh-gupta/clef-tune @ 0123456789ab
    Hardware: l40sx1 (1 GPU), timeout 1h
    Estimate: 0.2-0.5 h, $0.36-$0.81 (at most $1.80 if it runs to the timeout)
    Kernels: compiled once (~12 min), then cached   |   cached   |   off
    Adapter: me/clef-tune-support-tickets (private); bundle deleted after success | kept (--keep-data)
    ```

    With `estimate` of `None`, the estimate line reads `Estimate: unavailable`.
  - **Confirmation:** without `--yes`, call `typer.confirm("Launch this job?", abort=True)`. Then submit and print the job id, its URL and `Follow it with: clef-tune jobs logs <id> --follow`.
  - **`jobs list`:** a table of id, run, flavor, stage and created time.
  - **`jobs status`:** the stage (and message), run, flavor, URL, and `running|ran <h> h, about $<cost>` when known. Then the latest `train` record as `step N: loss L, T tokens/s, ETA M min`, and the latest `eval` record with its headline metrics to 3 decimals.
  - **Errors:** each command catches `LaunchError` (and `HubError`) and exits 1 with the message.

- [ ] **Step 4: Update `README.md`.** Add a `## Train on Hugging Face Jobs` section after `## Train on a GPU box`. It must cover:
  - **Prerequisites:**
    - a token with `repo.write` and `job.write` (fine-grained tokens work);
    - pre-paid Jobs credits;
    - a clean working tree whose commit is pushed to GitHub (`CLEF_TUNE_SOURCE` for a fork, `--source-ref` to pick a commit).
  - **Commands:** `jobs launch` with its flags, plus `list`, `status`, `logs -f`, `cancel` and `resume`.
  - **What happens:**
    - the bundle is uploaded to the private `<you>/jobs-artifacts` bucket and the model is mounted at `/model`;
    - the first job compiles `causal-conv1d` for about 12 minutes and caches it for later jobs;
    - multi-GPU flavors run DDP;
    - the adapter is pushed to a private repo, and the bundle is deleted after success unless `--keep-data`;
    - a failed or canceled job keeps its checkpoints, and `jobs resume` continues from them.
  - **Costs:** the confirmation step, and that the timeout caps the cost.

  In `## Train on a GPU box`, replace the kernel sentence with the safe install:

  ```bash
  pip install --no-build-isolation causal-conv1d==1.7.0 flash-linear-attention==0.6.0
  python -c "import causal_conv1d"   # if this fails: pip uninstall -y causal-conv1d flash-linear-attention fla-core
  ```

  Add one sentence of explanation: an isolated build compiles against a different torch and makes model loading crash.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_cli_jobs.py tests/test_cli.py -v`
Expected: all PASS.

- [ ] **Step 6: Lint and commit** (`feat: add clef-tune jobs launch/list/status/logs/cancel/resume`).

---

### Task 7: Training fixes for multi-GPU jobs and bucket-mounted output

**Files:**
- Modify: `clef_tune/train.py`, `clef_tune/checkpoints.py`, `clef_tune/cli.py`, `clef_tune/card.py`
- Test: `tests/torch/test_train.py`, `tests/torch/test_checkpoints.py`, `tests/test_cli.py`, `tests/test_card.py`

**Interfaces:**
- Consumes: Plan 2 `train`, `checkpoints`, `card`.
- Produces:
  - **`clef_tune.train`:**
    - `DIST_TIMEOUT = timedelta(hours=1)`;
    - `TrainResult` gains a last field, `main_process: bool = True`;
    - `_tokens_per_sec(accelerator, tokens: int, seconds: float) -> float`.
  - **`clef_tune.checkpoints`:** `CHECKPOINT_FILES = ("trainable.safetensors", "optimizer.pt", "scheduler.pt", "rng.pt", "state.json")`. `latest_checkpoint()` returns the newest step folder that has all of them.
  - **`render_card`:** its output starts with YAML front matter.

These fix what the 2026-10-28 DDP and bucket runs showed:
1. Every rank printed the CLI result.
2. The process group was never destroyed, so torch warned at exit.
3. `tokens_per_sec` counted rank 0's tokens only.
4. Every rank ran the Hub write check, and a 429 on one rank aborts the pre-flight.
5. The default 10-minute NCCL timeout is too short while rank 0 pushes.
6. The pushed card had no metadata.
7. Bucket uploads finish in the background, so a canceled job can leave a partial newest checkpoint.

- [ ] **Step 1: Write the failing tests.**

`tests/torch/test_train.py` (add `from datetime import timedelta` and `from types import SimpleNamespace`):

```python
def test_the_process_group_gets_a_long_timeout(tiny_project, monkeypatch):
    seen = {}
    real = train_module.Accelerator

    def accelerator(**kwargs):
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(train_module, "Accelerator", accelerator)
    train(tiny_project(train={"epochs": 1}))
    (handler,) = seen["kwargs_handlers"]
    assert handler.timeout == train_module.DIST_TIMEOUT == timedelta(hours=1)


def test_training_always_ends_the_accelerator(tiny_project, monkeypatch):
    ends = []
    monkeypatch.setattr(train_module.Accelerator, "end_training", lambda self: ends.append(1))
    assert train(tiny_project(train={"epochs": 1})).main_process is True
    assert ends == [1]


def test_tokens_per_sec_counts_every_process():
    two_ranks = SimpleNamespace(device=torch.device("cpu"), reduce=lambda tensor, reduction: tensor * 2)
    assert train_module._tokens_per_sec(two_ranks, 300, 1.5) == pytest.approx(400.0)
```

(The existing `test_hub_access_is_checked_before_the_model_loads` and `test_push_to_hub_after_training` must keep passing after the Hub check moves behind the main-process gate.)

`tests/torch/test_checkpoints.py`:

```python
def test_latest_checkpoint_skips_incomplete_steps(tmp_path):
    for step in (2, 4):
        folder = tmp_path / "checkpoints" / f"step-{step}"
        folder.mkdir(parents=True)
        for name in CHECKPOINT_FILES:
            if not (step == 4 and name == "state.json"):  # step 4: a cancel before the upload finished
                (folder / name).write_bytes(b"x")
    assert latest_checkpoint(tmp_path) == tmp_path / "checkpoints" / "step-2"
```

(Import `CHECKPOINT_FILES`, `latest_checkpoint` from `clef_tune.checkpoints`. Match the checkpoint root the module uses, e.g. `_root(output_dir)`, if it is not `output_dir / "checkpoints"`.)

`tests/test_cli.py`: give `fake_training` a `main_process=True` parameter, put it on the returned namespace, and add:

```python
def test_train_command_prints_only_on_the_main_process(monkeypatch, tmp_path):
    fake_training(monkeypatch, tmp_path, main_process=False)
    result = runner.invoke(cli.app, ["train", str(EXAMPLE)])
    assert result.exit_code == 0 and "Adapter written" not in result.output
```

`tests/test_card.py` (add `import yaml`):

```python
def test_the_card_starts_with_hub_metadata(tmp_path):
    config = make_config(tmp_path, data={"train": "t.csv"}, model={"id": "Cloudflare/clef-flash"})
    card = render_card(METRICS, config, "1.0")
    front, _ = card.removeprefix("---\n").split("---\n", 1)
    assert yaml.safe_load(front) == {
        "base_model": "Cloudflare/clef-flash",
        "base_model_relation": "adapter",
        "tags": ["clef", "clef-tune"],
    }
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/torch/test_train.py tests/torch/test_checkpoints.py tests/test_cli.py tests/test_card.py -v`
Expected: the new tests FAIL:
- `KeyError: 'kwargs_handlers'`;
- `ends == []`;
- no `_tokens_per_sec`;
- step 4 returned;
- "Adapter written" printed;
- no front matter.

- [ ] **Step 3: Implement.**
  - **Accelerator:** pass `kwargs_handlers=[InitProcessGroupKwargs(timeout=DIST_TIMEOUT)]` (from `accelerate.utils`).
  - **Hub write check:** move it after the `Accelerator` is created. Run it on the main process only, through a small helper shared with `_push`: run a callable on the main process, then `broadcast_object_list` the error text. Every rank raises `TrainingError` when it is set. The check must still happen before `load_for_training`.
  - **Throughput:** `_tokens_per_sec` returns `accelerator.reduce(torch.tensor(float(tokens), device=accelerator.device), reduction="sum").item() / max(seconds, 1e-9)`. `log_train` uses it. It runs on every rank at the same steps, so the collective is safe.
  - **Teardown:** call `accelerator.end_training()` unconditionally at the end of a successful run. Accelerate 1.15's version finishes trackers and destroys the process group.
  - **Result:** set `TrainResult.main_process` from `accelerator.is_main_process`. `cli.train` prints its result lines only when it is true.
  - **Checkpoints:** `latest_checkpoint()` skips step folders missing any `CHECKPOINT_FILES`.
  - **Card:** `render_card` prepends `---\n` + `dump_yaml({"base_model": config.model.id, "base_model_relation": "adapter", "tags": ["clef", "clef-tune"]})` + `---\n\n`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -v` (the full suite, including `tests/torch`)
Expected: all PASS; the `gpu` and `hfjobs` tests stay deselected.

- [ ] **Step 5: Lint and commit** (`fix: one Hub check, global throughput, clean teardown and complete checkpoints for multi-GPU jobs`).

---

### Task 8: Live HF Jobs test

**Files:**
- Create: `tests/test_hfjobs.py`

**Interfaces:**
- Consumes: from `clef_tune.launch.hf_jobs`, `JobRequest`, `plan_launch`, `submit_launch`, `job_status`, `job_logs`, `cancel_job`, `resume_job` and `jobs_bucket`.
- Produces: the opt-in `hfjobs` test (spec §11.5). *Amends §11.5:* it interrupts the first job with `cancel_job` after the first checkpoint rather than a forced timeout, which is deterministic.

- [ ] **Step 1: Write the test** `tests/test_hfjobs.py`:

```python
"""Opt-in: one real HF Jobs run on l4x1 (about 30 minutes, about $0.40), canceled after its first
checkpoint and resumed. Needs a Hugging Face token with repo.write and job.write, pre-paid Jobs
credits, and the current commit pushed to GitHub.

Run with: uv run pytest tests/test_hfjobs.py -m hfjobs -v -s
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from huggingface_hub import HfApi, get_token

from clef_tune.launch.hf_jobs import (
    JobRequest, cancel_job, job_logs, job_status, jobs_bucket, plan_launch, resume_job, submit_launch,
)
from clef_tune.yamlio import dump_yaml, load_yaml_file

pytestmark = pytest.mark.hfjobs
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
POLL_SECONDS = 15
ACTIVE = ("SCHEDULING", "RUNNING")


def _wait_for(predicate, timeout_s, what):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(POLL_SECONDS)
    pytest.fail(f"timed out waiting for {what}")


def test_a_job_resumes_after_cancel_and_pushes_the_adapter(tmp_path):
    token = get_token()
    if token is None:
        pytest.skip("needs a Hugging Face token")
    api = HfApi(token=token)
    run = f"hfjobs-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    (tmp_path / "support_tickets.csv").write_bytes((EXAMPLES / "support_tickets.csv").read_bytes())
    example = load_yaml_file(EXAMPLES / "support_tickets.yaml")
    config = tmp_path / "config.yaml"
    config.write_text(dump_yaml({
        **example,
        "run_name": run,
        "data": {**example["data"], "eval_fraction": 0.2},
        "train": {"quantization": "4bit", "epochs": 2, "max_length": 1024, "max_tokens_per_batch": 1024},
        "eval": {"eval_every": 1000},
        "output": {"save_every": 4},
    }), encoding="utf-8")

    plan = plan_launch(config, JobRequest(flavor="l4x1", timeout="1h"), api=api, token=token)
    bucket = jobs_bucket(plan.namespace)
    first = submit_launch(plan, api=api, token=token)
    jobs = [first.id]

    def stage(job_id):
        return job_status(job_id, api=api, token=token).stage

    try:
        def checkpointed():
            current = stage(first.id)
            if current not in ACTIVE:
                pytest.fail(f"the first job ended as {current} before its first checkpoint")
            files = api.list_bucket_tree(bucket, prefix=f"{first.work_folder}/out", recursive=True)
            return any(f.path.endswith("/state.json") and "/checkpoints/step-" in f.path for f in files)

        _wait_for(checkpointed, 45 * 60, "the first checkpoint")
        cancel_job(first.id, api=api, token=token)
        _wait_for(lambda: stage(first.id) == "CANCELED", 5 * 60, "the cancel")

        resumed = resume_job(first.id, api=api, token=token)
        jobs.append(resumed.id)
        _wait_for(lambda: stage(resumed.id) not in ACTIVE, 60 * 60, "the resumed job")
        logs = "\n".join(job_logs(resumed.id, api=api, token=token))
        assert stage(resumed.id) == "COMPLETED", logs[-3000:]
        assert "Resuming from step" in logs
        assert "Using the cached causal-conv1d wheel" in logs
        assert "fast path is not available" not in logs

        files = set(api.list_repo_files(plan.adapter_repo))
        assert {"joint_head.safetensors", "calibration.json", "adapter/adapter_config.json"} <= files
        left = [f.path for f in api.list_bucket_tree(bucket, prefix=first.work_folder, recursive=True)]
        assert not any("/bundle/" in p or "/checkpoints/" in p for p in left), left
    finally:
        for job_id in jobs:
            if stage(job_id) in ACTIVE:
                cancel_job(job_id, api=api, token=token)
        api.delete_repo(plan.adapter_repo, missing_ok=True)
        leftover = [f.path for f in api.list_bucket_tree(bucket, prefix=first.work_folder, recursive=True)
                    if f.type == "file"]
        if leftover:
            api.batch_bucket_files(bucket, delete=leftover)
```

- [ ] **Step 2: Check it is deselected by default**

Run: `uv run pytest -q`
Expected: the full suite PASSES and `tests/test_hfjobs.py` is deselected (`-m 'not gpu and not hfjobs'`).

- [ ] **Step 3: Lint and commit** (`test: opt-in HF Jobs run with cancel, resume and the kernel cache`).

- [ ] **Step 4: STOP and ask the user before anything leaves the machine.** The job installs the branch's commit from GitHub, so the repo must exist and hold that commit.
  - **Ask** the user to create the public repo `hersh-gupta/clef-tune` if it doesn't exist yet. Get their explicit OK to add the `origin` remote and push `plan-3-hf-jobs`.
  - **Confirm** they have at least $1 of HF Jobs credit; the run costs about $0.40.
  - **Do not push without that OK.**

- [ ] **Step 5: Run the live test** (in the background; it takes about 30 minutes):

Run: `uv run pytest tests/test_hfjobs.py -m hfjobs -v -s`
Expected: PASS. The first job's logs show either "Building causal-conv1d" or "Using the cached causal-conv1d wheel". The resumed job shows "Resuming from step N" and the cached wheel. Report the job IDs and the actual cost to the user. If it fails, debug with `clef-tune jobs logs <id>` before changing code.
