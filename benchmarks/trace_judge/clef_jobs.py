"""Step 3 of the trace-judge pilot: read the held-out test sets with Clef on Hugging Face Jobs.

Training uses clef-tune's own CLI on data/prepared/trace_judge_27b.yaml. This script adds what
the CLI doesn't do:

  estimate          the tokens to read, and clef-tune's estimate of the time and cost
  score             one job: the tuned adapter, then the untrained model, on every test set
                    (nothing starts without --yes; --resume continues a stopped job's folder)
  download          fetch the job's outputs into data/scores/

--arms and --sets narrow a job (`--arms base --sets controlarena` reads one set with the
untrained model). --name keeps jobs apart in data/jobs.json; give `download` the same name.

Long traces are read whole, up to --max-length tokens (see score_job.py for what happens when
one does not fit in memory). The test files go to the private jobs-artifacts bucket and the job
deletes them when it finishes: the Storf bundle may not be passed on.

The job is not labelled as a clef-tune job, so it stays out of the web UI's "Your models" list.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.cfpb.common import machine_env

HERE = Path(__file__).resolve().parent
PREPARED = HERE / "data" / "prepared"  # git-ignored
SCORES = HERE / "data" / "scores"
JOBS = HERE / "data" / "jobs.json"
CFPB_JOB = HERE.parent / "cfpb" / "score_job.py"  # its `base` and `fetch` steps are reused

CONFIG = "trace_judge_27b.yaml"
SETS = ("malt_test", "atbench500", "atbench", "atbench_no_tools", "controlarena")
ARMS = ("tuned", "base")  # tuned first: if the job stops early, the result that matters exists
MAX_LENGTH = 65536
FLAVOR = "rtx-pro-6000"
LABELS = {"benchmark": "trace-judge"}


def token() -> str:
    value = machine_env("HF_TOKEN")
    if not value:
        raise SystemExit("HF_TOKEN is not set (process, Machine or User scope)")
    return value


def tuned_repo() -> str:
    """The private Hub repo the training run pushes to, from the training config."""
    from clef_tune.config import load_config

    repo = load_config(PREPARED / CONFIG).output.push_to_hub
    if not repo:
        raise SystemExit(f"{CONFIG} names no Hub repo for the tuned model")
    return repo


def set_tokens(max_length: int, sets: list[str]) -> dict[str, dict[str, int]]:
    """Per test set: its rows, the tokens read under the limit, the longest trace as read, and
    how many traces are cut."""
    csv.field_size_limit(2**31 - 1)
    out = {}
    for name in sets:
        with (PREPARED / f"{name}.csv").open(encoding="utf-8", newline="") as handle:
            tokens = [int(float(row["tokens"])) for row in csv.DictReader(handle)]
        out[name] = {
            "rows": len(tokens),
            "tokens": sum(min(t, max_length) for t in tokens),
            "longest": min(max(tokens), max_length),
            "cut": sum(t > max_length for t in tokens),
        }
    return out


def estimate(arms: list[str], sets: list[str], max_length: int, flavor: str):
    """clef-tune's estimate for reading each of `sets` once per arm on `flavor`."""
    from clef_tune.config import load_config
    from clef_tune.estimate import (
        estimate_hf_flavors,
        fetch_hf_prices,
        model_spec,
        prediction_workload,
    )

    config = load_config(PREPARED / CONFIG)
    found = set_tokens(max_length, sets)
    tokens = sum(item["tokens"] for item in found.values()) * len(arms)
    longest = max(item["longest"] for item in found.values())
    workload = prediction_workload(config, tokens, longest)
    rows = estimate_hf_flavors(model_spec(config.model.id), workload, fetch_hf_prices())
    row = next((r for r in rows if r.flavor == flavor), None)
    if row is None:
        raise SystemExit(f"unknown hardware {flavor!r}")
    return found, row


def estimate_command(arms: list[str], sets: list[str], max_length: int, flavor: str) -> None:
    from clef_tune.estimate import suggest_timeout

    found, row = estimate(arms, sets, max_length, flavor)
    for name, item in found.items():
        print(
            f"{name:>17}: {item['rows']:>5,} traces, {item['tokens'] / 1e6:6.1f}M tokens, "
            f"{item['cut']} cut at {max_length:,}"
        )
    total = sum(item["tokens"] for item in found.values())
    print(f"{'each arm':>17}: {total / 1e6:6.1f}M tokens; arms: {', '.join(arms)}")
    print(
        f"{flavor}: {row.hours_low:.1f}-{row.hours_high:.1f} h, "
        f"${row.cost_low:.2f}-${row.cost_high:.2f} at ${row.price_per_hour}/h"
    )
    print(
        f"memory for the longest trace: {row.memory_needed_gb:.0f} GB needed, "
        f"{row.memory_available_gb:.0f} GB available"
        + ("" if row.fits else " (too little by this estimate: such traces get cut)")
    )
    print(f"suggested timeout: {suggest_timeout(row, extra_minutes=20)}")


def base_config() -> str:
    from clef_tune.config import load_config
    from clef_tune.yamlio import dump_yaml

    config = load_config(PREPARED / CONFIG)
    data = config.to_portable_dict()
    data["model"]["local_path"] = "/model"
    data["train"]["mode"] = "head_only"  # no LoRA modules: the backbone exactly as released
    data["run_name"] = f"{config.run_name}-untrained"
    data.get("output", {}).pop("push_to_hub", None)  # the untrained model is never published
    return dump_yaml(data)


def job_script(source, arms: list[str], max_length: int, repo: str, sets: list[str]) -> str:
    from clef_tune.launch.job_script import kernel_step

    requirement = source.requirement()
    w = "/work"
    names = ",".join(sets)
    prepare = {
        "tuned": f"python {w}/cfpb_score_job.py fetch {shlex.quote(repo)} /tmp/tuned",
        "base": f"python {w}/cfpb_score_job.py base {w}/base_config.yaml /tmp/base",
    }
    steps = [
        "set -euo pipefail",
        f'echo "scoring job: installing {requirement}"',
        f"pip install --root-user-action=ignore {shlex.quote(requirement)}",
        kernel_step(),
        f"mkdir -p {w}/out",
    ]
    for arm in arms:
        steps += [
            prepare[arm],
            f"python {w}/score_job.py score /tmp/{arm} {arm} {w}/input {w}/out {max_length} "
            f"{names}",
        ]
    steps += [f"rm -rf {w}/input", 'echo "scoring job: done"']
    return "\n".join(steps)


def _jobs() -> dict:
    return json.loads(JOBS.read_text(encoding="utf-8")) if JOBS.exists() else {}


def score_command(
    arms: list[str],
    sets: list[str],
    max_length: int,
    flavor: str,
    timeout: str | None,
    yes: bool,
    resume: bool,
    name: str,
) -> None:
    from huggingface_hub import HfApi, Volume

    from clef_tune.config import load_config
    from clef_tune.estimate import suggest_timeout
    from clef_tune.launch.hf_jobs import (
        jobs_bucket,
        kernel_cache_folder,
        run_folder,
        start_job,
        timeout_hours,
    )
    from clef_tune.launch.job_script import KERNEL_MOUNT, MODEL_MOUNT, WORK_MOUNT, job_env
    from clef_tune.launch.source import pin_commit, resolve_source
    from clef_tune.runtime import BASE_IMAGE

    config = load_config(PREPARED / CONFIG)
    source = pin_commit(resolve_source())  # this checkout's commit, which must be on GitHub
    _, row = estimate(arms, sets, max_length, flavor)
    timeout = timeout or suggest_timeout(row, extra_minutes=20)
    most = timeout_hours(timeout) * row.price_per_hour
    repo = tuned_repo()
    print(
        f"Score {', '.join(arms)} on {', '.join(sets)} with {flavor}"
        + (f" (tuned model: {repo})" if "tuned" in arms else "")
        + f", reading up to {max_length:,} tokens: "
        f"{row.hours_low:.1f}-{row.hours_high:.1f} h, ${row.cost_low:.2f}-${row.cost_high:.2f}; "
        f"timeout {timeout}, at most ${most:.2f}"
    )
    if not yes:
        print("Nothing started. Run again with --yes once approved.")
        return
    hf_token = token()
    api = HfApi(token=hf_token)
    namespace = api.whoami()["name"]
    bucket = jobs_bucket(namespace)
    sync = {"read_only": False, "namespace": namespace, "token": hf_token}
    with tempfile.TemporaryDirectory() as tmp:
        if resume:
            folder = _jobs()[name]["folder"]
            work = Volume(
                type="bucket", source=bucket, mount_path=WORK_MOUNT, path=folder, read_only=False
            )
        else:
            folder = run_folder("trace-judge-score", datetime.now(UTC))
            stage = Path(tmp) / "stage"
            (stage / "input").mkdir(parents=True)
            for item in sets:
                (stage / "input" / f"{item}.csv").write_bytes(
                    (PREPARED / f"{item}.csv").read_bytes()
                )
            (stage / "score_job.py").write_bytes((HERE / "score_job.py").read_bytes())
            (stage / "cfpb_score_job.py").write_bytes(CFPB_JOB.read_bytes())
            (stage / "base_config.yaml").write_text(base_config(), encoding="utf-8")
            work = api.sync_job_volume(stage, WORK_MOUNT, remote_name=folder, **sync)
        empty = Path(tmp) / "kernels"
        empty.mkdir()
        volumes = [
            work,
            Volume(
                type="model",
                source=config.model.id,
                mount_path=MODEL_MOUNT,
                revision=config.model.resolved_revision(),
            ),
            api.sync_job_volume(empty, KERNEL_MOUNT, remote_name=kernel_cache_folder(), **sync),
        ]
    info = start_job(
        api,
        image=BASE_IMAGE,
        command=["bash", "-c", job_script(source, arms, max_length, repo, sets)],
        env=job_env(source, price_per_hour=row.price_per_hour, estimated_hours=None),
        secrets={"HF_TOKEN": hf_token},
        flavor=flavor,
        timeout=timeout,
        labels=LABELS,
        volumes=volumes,
        token=hf_token,
    )
    jobs = _jobs()
    jobs[name] = {
        "id": info.id,
        "url": info.url,
        "flavor": flavor,
        "price_per_hour": row.price_per_hour,
        "bucket": bucket,
        "folder": folder,
        "arms": arms,
        "sets": sets,
        "repo": repo if "tuned" in arms else None,
        "max_length": max_length,
        "source": source.ref,
    }
    JOBS.parent.mkdir(parents=True, exist_ok=True)
    JOBS.write_text(json.dumps(jobs, indent=2), encoding="utf-8")
    print(f"Job started: {info.id} {info.url}")


def download_command(name: str) -> None:
    from huggingface_hub import HfApi

    job = _jobs()[name]
    api = HfApi(token=token())
    SCORES.mkdir(parents=True, exist_ok=True)
    names = [
        f"{arm}-{item}{suffix}"
        for arm in job["arms"]
        for item in job.get("sets", SETS)
        for suffix in (".jsonl", "-timing.json")
    ]
    api.download_bucket_files(
        job["bucket"],
        [(f"{job['folder']}/out/{item}", SCORES / item) for item in names],
        raise_on_missing_files=False,
    )
    missing = [item for item in names if not (SCORES / item).exists()]
    print(f"{len(names) - len(missing)} of {len(names)} files in {SCORES}")
    if missing:
        print("not in the bucket yet:", ", ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["estimate", "score", "download"])
    parser.add_argument("--arms", default=",".join(ARMS), help="tuned, base or both")
    parser.add_argument("--sets", default=",".join(SETS), help="test sets to read")
    parser.add_argument("--name", default="score", help="the job's name in data/jobs.json")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--flavor", default=FLAVOR)
    parser.add_argument("--timeout", default=None)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    arms = [arm for arm in ARMS if arm in args.arms.split(",")]
    if not arms:
        raise SystemExit("--arms must name tuned, base or both")
    sets = [item for item in SETS if item in args.sets.split(",")]
    if not sets or set(args.sets.split(",")) - set(SETS):
        raise SystemExit(f"--sets must name some of: {', '.join(SETS)}")
    if args.command == "estimate":
        estimate_command(arms, sets, args.max_length, args.flavor)
    elif args.command == "score":
        score_command(
            arms, sets, args.max_length, args.flavor, args.timeout, args.yes, args.resume, args.name
        )
    else:
        download_command(args.name)


if __name__ == "__main__":
    main()
