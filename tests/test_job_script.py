import shlex
import shutil
import signal
import subprocess
import sys
import time

import pytest

from clef_tune.launch.job_script import job_env, job_script, kernel_step
from clef_tune.launch.source import Source
from clef_tune.runtime import ESTIMATE_ENV, KERNEL_PACKAGES, PRICE_ENV

SOURCE = Source(
    "https://github.com/hersh-gupta/clef-tune", "0123456789abcdef0123456789abcdef01234567"
)
POSIX_BASH = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None, reason="needs a POSIX bash"
)


def _working_bash():
    """A bash that runs, or None. On Windows `bash` can be the WSL launcher with no distribution,
    and Windows looks in System32 before PATH, so run the one found here by its full path."""
    bash = shutil.which("bash")
    try:
        probe = subprocess.run([bash, "-c", "echo ok"], capture_output=True) if bash else None
    except OSError:
        return None
    return bash if probe is not None and probe.stdout.strip() == b"ok" else None


BASH = _working_bash()


def script(**overrides):
    options = {"gpus": 1, "adapter_repo": "me/clef-tune-run", "keep_data": False, "kernels": True}
    return job_script(SOURCE, **(options | overrides))


def test_a_single_gpu_job_installs_the_pinned_source_and_trains():
    text = script()
    assert f"pip install --root-user-action=ignore {shlex.quote(SOURCE.requirement())}" in text
    assert (
        "clef-tune train /work/bundle/config.yaml --push-to-hub me/clef-tune-run "
        "--delete-bundle-on-success &" in text
    )


def test_keep_data_keeps_the_bundle():
    assert "--push-to-hub me/clef-tune-run &" in script(keep_data=True)


def test_multi_gpu_flavors_launch_one_process_per_gpu():
    assert (
        "accelerate launch --multi_gpu --num_processes 4 -m clef_tune train "
        "/work/bundle/config.yaml" in script(gpus=4)
    )


def test_the_kernel_step_builds_against_the_installed_torch_and_checks_the_import():
    step = kernel_step()
    assert "--no-build-isolation" in step and all(p in step for p in KERNEL_PACKAGES)
    assert 'python -c "import causal_conv1d"' in step
    assert "pip uninstall -y causal-conv1d flash-linear-attention fla-core" in step
    assert step in script()
    assert "causal" not in script(kernels=False)


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_the_job_script_is_valid_bash():
    result = subprocess.run([BASH, "-n"], input=script(gpus=2), text=True, capture_output=True)
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
def test_an_unreadable_cached_wheel_falls_back_to_the_slow_path(tmp_path):
    stubs, _ = _stubs(tmp_path, pip_succeeds=False)
    cache = tmp_path / "cache"
    (cache / "causal_conv1d-1.7.0-cp312-cp312-linux_x86_64.whl").mkdir(parents=True)  # cp fails
    result = subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + kernel_step(str(cache))],
        env={"PATH": f"{stubs}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "slow PyTorch path" in result.stdout


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


@POSIX_BASH
def test_a_cancel_stops_training_within_seconds(tmp_path):
    # HF cancels a job by signalling its bash; training must stop, not run on unattended.
    stubs = tmp_path / "bin"
    stubs.mkdir()
    trained = tmp_path / "trained"
    (stubs / "pip").write_text("#!/bin/sh\nexit 0\n")
    (stubs / "clef-tune").write_text(f"#!/bin/sh\necho started\nsleep 3\ntouch {trained}\n")
    for stub in stubs.iterdir():
        stub.chmod(0o755)
    job = subprocess.Popen(
        ["bash", "-c", script(kernels=False)],
        env={"PATH": f"{stubs}:/usr/bin:/bin"},
        stdout=subprocess.PIPE,
        text=True,
    )
    while job.stdout.readline().strip() != "started":
        pass
    job.send_signal(signal.SIGTERM)
    assert job.wait(timeout=10) != 0
    time.sleep(5)
    assert not trained.exists()
