"""Step 4: the Clef contenders on Hugging Face Jobs.

Training uses clef-tune's own CLI (see benchmarks/README.md):
  clef-tune validate / estimate / jobs launch  benchmarks/cfpb/data/prepared/cfpb_<model>.yaml

This script adds what the CLI doesn't do, scoring the held-out test set:
  check-test        clef-tune's checks on test.csv, with the same config
  score MODEL       one job per model: score the untrained model and the tuned adapter on
                    test.csv with `clef-tune eval` and timed `clef-tune predict` code paths
  download MODEL    fetch a finished scoring job's outputs; write results/answers/clef-*.csv

MODEL is "flash" or "27b". Scoring jobs are not labelled as clef-tune jobs, so they stay out
of the web UI's "Your models" list.
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

from benchmarks.cfpb.common import ANSWERS, DATA, HERE, PREPARED, machine_env

SOURCE_REF = "af1a9a4708f25c8949ca9ec6a9e9b974d44cf3bb"  # main on GitHub; what training installed
MODELS = {
    "flash": ("cfpb_flash.yaml", "hvgupta1/clef-flash-cfpb-complaints", "clef-flash"),
    "27b": ("cfpb_27b.yaml", "hvgupta1/clef-cfpb-complaints", "clef-27b"),
}
QUESTIONS = ("issue", "product", "outcome")
OUT = DATA / "clef"
JOBS = DATA / "jobs.json"


def token() -> str:
    value = machine_env("HF_TOKEN")
    if not value:
        raise SystemExit("HF_TOKEN is not set (process, Machine or User scope)")
    return value


def check_test_command() -> None:
    """`clef-tune validate` on test.csv: the same config with test.csv in train's place."""
    import subprocess

    from clef_tune.config import load_config

    config = load_config(PREPARED / "cfpb_flash.yaml")
    data = config.to_dict()
    data["data"]["train"] = "test.csv"
    from clef_tune.yamlio import dump_yaml

    check = PREPARED / "check_test.yaml"
    check.write_text(dump_yaml(data), encoding="utf-8")
    try:
        result = subprocess.run(["clef-tune", "validate", str(check)], check=False)
    finally:
        check.unlink()
    if result.returncode:
        raise SystemExit(result.returncode)


def _base_config(name: str) -> str:
    from clef_tune.config import load_config

    config = load_config(PREPARED / name)
    data = config.to_portable_dict()
    data["model"]["local_path"] = "/model"
    data["train"]["mode"] = "head_only"  # no LoRA modules: the backbone exactly as released
    data["run_name"] = f"{config.run_name}-untrained"
    from clef_tune.yamlio import dump_yaml

    return dump_yaml(data)


def job_script(source, repo: str) -> str:
    from clef_tune.launch.job_script import kernel_step

    requirement = source.requirement()
    w = "/work"
    test = f"{w}/input/test.csv"
    steps = [
        "set -euo pipefail",
        f'echo "scoring job: installing {requirement}"',
        f"pip install --root-user-action=ignore {shlex.quote(requirement)}",
        kernel_step(),
        f"mkdir -p {w}/out",
        f"python {w}/score_job.py base {w}/base_config.yaml /tmp/base",
        f"clef-tune eval /tmp/base {test} --output {w}/out/base-eval.json",
        f"python {w}/score_job.py predict /tmp/base {test} {w}/out/base-predictions.csv "
        f"{w}/out/base-timing.json",
        f"python {w}/score_job.py fetch {shlex.quote(repo)} /tmp/tuned",
        f"clef-tune eval /tmp/tuned {test} --output {w}/out/tuned-eval.json",
        f"python {w}/score_job.py predict /tmp/tuned {test} {w}/out/tuned-predictions.csv "
        f"{w}/out/tuned-timing.json",
        'echo "scoring job: done"',
    ]
    return "\n".join(steps)


def _jobs() -> dict:
    return json.loads(JOBS.read_text(encoding="utf-8")) if JOBS.exists() else {}


def score_command(model: str, flavor: str, timeout: str, yes: bool) -> None:
    from huggingface_hub import HfApi, Volume

    from clef_tune.config import load_config
    from clef_tune.estimate import fetch_hf_prices
    from clef_tune.launch.hf_jobs import (
        jobs_bucket,
        kernel_cache_folder,
        run_folder,
        start_job,
    )
    from clef_tune.launch.job_script import KERNEL_MOUNT, MODEL_MOUNT, WORK_MOUNT, job_env
    from clef_tune.launch.source import pin_commit, resolve_source
    from clef_tune.runtime import BASE_IMAGE

    config_name, repo, _ = MODELS[model]
    config = load_config(PREPARED / config_name)
    source = pin_commit(resolve_source(SOURCE_REF))
    price = fetch_hf_prices().get(flavor)
    print(f"Score {model} (untrained and {repo}) on {flavor} at ${price}/h, timeout {timeout}")
    if not yes:
        print("Nothing started. Run again with --yes once approved.")
        return
    hf_token = token()
    api = HfApi(token=hf_token)
    namespace = api.whoami()["name"]
    folder = run_folder(f"benchmark-score-{model}", datetime.now(UTC))
    sync = {"read_only": False, "namespace": namespace, "token": hf_token}
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "stage"
        (stage / "input").mkdir(parents=True)
        (stage / "input" / "test.csv").write_bytes((PREPARED / "test.csv").read_bytes())
        (stage / "score_job.py").write_bytes((HERE / "score_job.py").read_bytes())
        (stage / "base_config.yaml").write_text(_base_config(config_name), encoding="utf-8")
        empty = Path(tmp) / "kernels"
        empty.mkdir()
        volumes = [
            api.sync_job_volume(stage, WORK_MOUNT, remote_name=folder, **sync),
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
        command=["bash", "-c", job_script(source, repo)],
        env=job_env(source, price_per_hour=price, estimated_hours=None),
        secrets={"HF_TOKEN": hf_token},
        flavor=flavor,
        timeout=timeout,
        labels={"benchmark": "cfpb", "model": model},
        volumes=volumes,
        token=hf_token,
    )
    jobs = _jobs()
    jobs[f"score-{model}"] = {
        "id": info.id,
        "url": info.url,
        "flavor": flavor,
        "price_per_hour": price,
        "bucket": jobs_bucket(namespace),
        "folder": folder,
    }
    JOBS.parent.mkdir(parents=True, exist_ok=True)
    JOBS.write_text(json.dumps(jobs, indent=2), encoding="utf-8")
    print(f"Job started: {info.id} {info.url}")


def download_command(model: str) -> None:
    from huggingface_hub import HfApi

    job = _jobs()[f"score-{model}"]
    api = HfApi(token=token())
    target = OUT / model
    target.mkdir(parents=True, exist_ok=True)
    names = [
        f"{arm}-{kind}"
        for arm in ("base", "tuned")
        for kind in ("eval.json", "predictions.csv", "timing.json")
    ]
    api.download_bucket_files(
        job["bucket"],
        [(f"{job['folder']}/out/{name}", target / name) for name in names],
        raise_on_missing_files=False,
    )
    missing = [n for n in names if not (target / n).exists()]
    if missing:
        raise SystemExit(f"not in the bucket yet: {', '.join(missing)}")
    for arm in ("base", "tuned"):
        export_answers(model, arm, target)
    print("Outputs in", target)


def export_answers(model: str, arm: str, folder: Path) -> Path:
    """The scorer's shape: one row per complaint with each answer, its confidence and the truth."""
    with (PREPARED / "test.csv").open(encoding="utf-8", newline="") as rows:
        truth = {r["complaint_id"]: r for r in csv.DictReader(rows)}
    with (folder / f"{arm}-predictions.csv").open(encoding="utf-8", newline="") as rows:
        predicted = list(csv.DictReader(rows))
    name = MODELS[model][2] + ("-tuned" if arm == "tuned" else "-untrained")
    ANSWERS.mkdir(parents=True, exist_ok=True)
    path = ANSWERS / f"{name}.csv"
    header = ["complaint_id", "ok"]
    for q in QUESTIONS:
        header += [f"{q}_answer", f"{q}_confidence", f"{q}_true"]
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(header)
        for row in sorted(predicted, key=lambda r: r["complaint_id"]):
            line = [row["complaint_id"], "1"]
            for q in QUESTIONS:
                line += [row[f"{q}_predicted"], row[f"{q}_confidence"]]
                line.append(truth[row["complaint_id"]][q])
            writer.writerow(line)
    timing = json.loads((folder / f"{arm}-timing.json").read_text(encoding="utf-8"))
    job = _jobs()[f"score-{model}"]
    summary = {
        "rows": timing["rows"],
        "seconds": timing["seconds"],
        "gpu": timing["gpu"],
        "flavor": job["flavor"],
        "price_per_hour": job["price_per_hour"],
        "complaints_per_minute": timing["rows"] / timing["seconds"] * 60,
        "usd_per_1000": job["price_per_hour"] * timing["seconds"] / 3600 / timing["rows"] * 1000,
    }
    (folder / f"{arm}-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(name, json.dumps(summary))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-test")
    score = sub.add_parser("score")
    score.add_argument("model", choices=list(MODELS))
    score.add_argument("--flavor", default="rtx-pro-6000")
    score.add_argument("--timeout", default="60m")
    score.add_argument("--yes", action="store_true")
    dl = sub.add_parser("download")
    dl.add_argument("model", choices=list(MODELS))
    args = parser.parse_args()
    if args.command == "check-test":
        check_test_command()
    elif args.command == "score":
        score_command(args.model, args.flavor, args.timeout, args.yes)
    else:
        download_command(args.model)


if __name__ == "__main__":
    main()
