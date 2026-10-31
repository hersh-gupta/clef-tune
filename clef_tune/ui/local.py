"""Train on this machine from the UI (never on a Space). Training runs as `clef-tune train` in
a subprocess: Accelerate's state is one per process, and the UI must not import torch. The
processes live here, not in the Session."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from clef_tune.bundle import write_bundle
from clef_tune.config import load_config
from clef_tune.launch.hf_jobs import METRICS_LOG, JobStatus
from clef_tune.ui.runs import ResultsView, RunState, results_view, run_state
from clef_tune.ui.session import Session
from clef_tune.yamlio import load_yaml_file

FINAL_DIR = "final"  # where clef_tune.train writes the adapter (train.py imports torch)
LOG_FILE = "train.log"
LOCAL_ID = "local"  # the run id the pages use for this machine's run


class LocalRunError(RuntimeError):
    """A local run cannot start; the message says why."""


@dataclass
class LocalRun:
    folder: Path
    config_path: Path
    output_dir: Path
    log_path: Path
    process: subprocess.Popen | None = None
    canceled: bool = False
    started: float = 0.0
    finished: float | None = None
    log_file: IO[str] | None = field(default=None, repr=False)

    def stage(self) -> str:
        if self.process is None:
            return "CANCELED" if self.canceled else "ERROR"
        code = self.process.poll()
        if code is None:
            return "RUNNING"
        if self.log_file is not None:  # Windows cannot delete a folder with an open file
            self.log_file.close()
            self.log_file = None
        if self.finished is None:
            self.finished = time.monotonic()
        if self.canceled:
            return "CANCELED"
        return "COMPLETED" if code == 0 else "ERROR"

    def hours(self) -> float:
        end = self.finished if self.finished is not None else time.monotonic()
        return max(end - self.started, 0.0) / 3600


RUNS: dict[Path, LocalRun] = {}


def train_command(config_path: Path) -> list[str]:
    return [sys.executable, "-m", "clef_tune", "train", str(config_path)]


def default_folder(session: Session) -> str:
    name = session.draft.run_name if session.draft else "run"
    return str(Path.cwd() / "runs" / name)


def start_local(
    session: Session, folder: str, *, command: Callable[[Path], list[str]] = train_command
) -> LocalRun:
    """Write the bundle into folder and start training it. Starting a stopped run again keeps
    its bundle, so training resumes from its newest checkpoint."""
    report = session.report
    if report is None or not report.ok:
        raise LocalRunError("Fix the validation errors first")
    path = Path(folder).expanduser().resolve()
    run = RUNS.get(path)
    if run is not None and run.stage() == "RUNNING":
        raise LocalRunError(f"{path} is already training")
    if run is None:
        if path.exists() and any(path.iterdir()):
            raise LocalRunError(f"{path} already exists; choose another folder")
        config_path = write_bundle(report, session.config(), path)
        output_dir = load_config(config_path).output_dir()
        run = LocalRun(path, config_path, output_dir, path / LOG_FILE)
        RUNS[path] = run
    log = open(run.log_path, "a", encoding="utf-8")  # noqa: SIM115 - closed when it exits
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    run.process = subprocess.Popen(
        command(run.config_path),
        cwd=path,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
    )
    run.log_file, run.canceled = log, False
    run.started, run.finished = time.monotonic(), None
    session.local_folder = str(path)
    return run


def stop_local(folder: str) -> None:
    run = RUNS.get(Path(folder).expanduser().resolve())
    if run is not None and run.process is not None and run.process.poll() is None:
        run.canceled = True
        run.process.terminate()


def _events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:  # a line still being written
            continue
        if isinstance(record, dict):
            events.append(record)
    return events


def local_state(
    session: Session, *, model: str = "the model", noun: str = "example"
) -> tuple[RunState | None, ResultsView | None]:
    """This session's local run, as a run state, and its results once it finished."""
    run = RUNS.get(Path(session.local_folder).resolve()) if session.local_folder else None
    if run is None:
        return None, None
    stage = run.stage()
    status = JobStatus(
        id=LOCAL_ID,
        url="",
        run=session.draft.run_name if session.draft else "local",
        flavor=None,
        stage=stage,
        message=None if stage != "ERROR" else f"see {run.log_path}",
        hours=run.hours(),
        cost=None,
        progress=_events(run.output_dir / METRICS_LOG),
    )
    state = run_state(status, model=model, noun=noun)
    state.stages[0].detail = "This computer's GPU"
    state.can_resume = stage in ("ERROR", "CANCELED")
    adapter = run.output_dir / FINAL_DIR
    metrics_path = adapter / "metrics.json"
    if stage != "COMPLETED" or not metrics_path.exists():
        return state, None
    config_path = adapter / "train_config.yaml"
    config = load_yaml_file(config_path) if config_path.exists() else {}
    results = results_view(
        json.loads(metrics_path.read_text(encoding="utf-8")),
        config,
        None,
        name=status.run or "local",
        local_path=str(adapter),
    )
    return state, results
