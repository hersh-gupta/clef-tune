"""The bash command and environment of a clef-tune training job on Hugging Face Jobs.
Torch-free."""

from __future__ import annotations

import shlex

from clef_tune.launch.source import Source
from clef_tune.runtime import ESTIMATE_ENV, KERNEL_PINS, PRICE_ENV

WORK_MOUNT = "/work"  # the run's folder in the jobs-artifacts bucket: bundle, output, checkpoints
MODEL_MOUNT = "/model"  # the base model repo, mounted read-only
KERNEL_MOUNT = "/kernels"  # the bucket folder caching the causal-conv1d wheel between jobs
BUNDLE_CONFIG = f"{WORK_MOUNT}/bundle/config.yaml"
JOB_OUTPUT_DIR = f"{WORK_MOUNT}/out"


def kernel_step(cache_dir: str = KERNEL_MOUNT) -> str:
    """Install the fast linear-attention kernels, reusing a cached causal-conv1d wheel. It must
    build against the image's torch, and transformers imports any installed copy, so a build
    that cannot import is removed and training continues on the slow path."""
    conv, fla = KERNEL_PINS["causal-conv1d"], KERNEL_PINS["flash-linear-attention"]
    return f"""mkdir -p /tmp/kernel-wheels
if ls {cache_dir}/causal_conv1d-*.whl >/dev/null 2>&1; then
    echo "Using the cached causal-conv1d wheel"
    cp {cache_dir}/causal_conv1d-*.whl /tmp/kernel-wheels/ || echo "warning: could not read the cached causal-conv1d wheel"
else
    echo "Building causal-conv1d {conv} (about 12 minutes on 8 vCPUs; cached for later jobs)"
    if pip wheel --no-deps --no-build-isolation -w /tmp/kernel-wheels causal-conv1d=={conv}; then
        cp /tmp/kernel-wheels/causal_conv1d-*.whl {cache_dir}/ || echo "warning: could not cache the causal-conv1d wheel"
    fi
fi
if ! (pip install --root-user-action=ignore /tmp/kernel-wheels/causal_conv1d-*.whl flash-linear-attention=={fla} \
        && python -c "import causal_conv1d"); then
    pip uninstall -y causal-conv1d flash-linear-attention fla-core || true
    echo "warning: kernel packages failed to install; linear attention will use the slow PyTorch path"
fi"""  # noqa: E501


def job_script(
    source: Source, *, gpus: int, adapter_repo: str, keep_data: bool, kernels: bool
) -> str:
    requirement = source.requirement()
    if gpus > 1:
        launcher = f"accelerate launch --multi_gpu --num_processes {gpus} -m clef_tune train"
    else:
        launcher = "clef-tune train"
    train = f"{launcher} {BUNDLE_CONFIG} --push-to-hub {shlex.quote(adapter_repo)}"
    if not keep_data:
        train += " --delete-bundle-on-success"
    lines = [
        "set -euo pipefail",
        f'echo "clef-tune job: installing {requirement}"',
        f"pip install --root-user-action=ignore {shlex.quote(requirement)}",
        *([kernel_step()] if kernels else []),
        # A cancel or timeout signals this shell, which as PID 1 would ignore it while training
        # kept writing to /work for about a minute; pass it on so training stops now.
        f"{train} &",
        "train_pid=$!",
        'trap \'echo "clef-tune job: stopping"; kill -TERM "$train_pid" 2>/dev/null\' TERM INT',
        'wait "$train_pid"',
    ]
    return "\n".join(lines)


def job_command(
    source: Source, *, gpus: int, adapter_repo: str, keep_data: bool, kernels: bool
) -> list[str]:
    script = job_script(
        source, gpus=gpus, adapter_repo=adapter_repo, keep_data=keep_data, kernels=kernels
    )
    return ["bash", "-c", script]


def job_env(
    source: Source, *, price_per_hour: float | None, estimated_hours: float | None
) -> dict[str, str]:
    """The job's environment. Never a name starting with NVIDIA_, which HF rejects."""
    env = {
        "PIP_BREAK_SYSTEM_PACKAGES": "1",  # the image's torch is in an externally managed Python
        "PYTHONUNBUFFERED": "1",
        # a GitHub archive has no git metadata for hatch-vcs to read a version from
        "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CLEF_TUNE": source.version,
    }
    if price_per_hour is not None:
        env[PRICE_ENV] = str(price_per_hour)
    if estimated_hours is not None:
        env[ESTIMATE_ENV] = str(estimated_hours)
    return env
