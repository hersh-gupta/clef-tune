"""Fake services for the video: the web UI runs unchanged, but every call that would reach Hugging
Face is answered from the real Clef-Flash benchmark run (its estimate, its training log and its
results), so the screens show that run's numbers. Nothing touches the network.

The training log replays at the pace the recorder sets with `advance(fraction)`, so about 50
minutes of training can be shown, sped up, in a few seconds.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from clef_tune.hub import RunFiles
from clef_tune.launch.hf_jobs import JobStatus
from clef_tune.yamlio import load_yaml_file
from tests.ui.fakes import RECORDED_RUN, SHA, fake_services

JOB = "job-cfpb"
RUN = "clef-flash-cfpb-complaints"
REPO = f"you/{RUN}"


class RealRun:
    """The real run's files in `folder`: metrics.jsonl (the training log), metrics.json and
    config.yaml (from its Hub repo), estimate.json (what `clef-tune jobs launch` showed). Falls
    back to the test suite's recorded run while the real one isn't downloaded yet, for drafts."""

    def __init__(self, folder: Path) -> None:
        real = (folder / "metrics.jsonl").exists()
        source = folder if real else RECORDED_RUN
        lines = (source / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        self.events = [json.loads(line) for line in lines if line.strip()]
        self.metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
        config = folder / "config.yaml"
        self.config = load_yaml_file(config) if config.exists() else None
        estimate = folder / "estimate.json"
        self.estimate = (
            json.loads(estimate.read_text(encoding="utf-8"))
            if estimate.exists()
            else {
                "flavor": "rtx-pro-6000",
                "price_per_hour": 2.75,
                "hours_low": 0.5,
                "hours_high": 1.0,
                "cost_low": 1.30,
                "cost_high": 2.78,
                "max_cost": 5.50,
                "timeout": "2h",
            }
        )
        self.real = real
        self.last_step = max((e.get("step", 0) for e in self.events), default=0)
        self.fraction = 0.0  # how far through training the replay is
        self.started = datetime.now(UTC)

    def advance(self, fraction: float) -> None:
        self.fraction = max(0.0, min(1.0, fraction))

    def visible_events(self) -> list[dict]:
        if self.fraction >= 1.0:
            return self.events
        step = self.fraction * self.last_step
        return [e for e in self.events if e["event"] == "baseline" or e.get("step", 0) <= step]

    def status(self) -> JobStatus:
        done = self.fraction >= 1.0
        hours = self.metrics.get("run", {}).get("hours") or 0.8
        price = self.estimate["price_per_hour"]
        shown = hours if done else hours * self.fraction
        return JobStatus(
            id=JOB,
            url=f"https://huggingface.co/jobs/you/{JOB}",
            run=RUN,
            flavor=self.estimate["flavor"],
            stage="COMPLETED" if done else "RUNNING",
            message=None,
            hours=shown,
            cost=shown * price,
            progress=self.visible_events(),
            adapter_repo=REPO,
            price_per_hour=price,
            timeout=self.estimate["timeout"],
            started_at=self.started - timedelta(hours=shown),
            created_at=self.started - timedelta(hours=shown),
        )


def video_services(run: RealRun, example_config: Path):
    from clef_tune.config import load_config

    def plan(config_path, request, **kwargs):
        config = load_config(config_path)
        e = run.estimate
        return SimpleNamespace(
            config=config,
            source=SimpleNamespace(repo_url="https://github.com/hersh-gupta/clef-tune", ref=SHA),
            namespace="you",
            adapter_repo=request.adapter_repo or f"you/{config.run_name}",
            flavor=e["flavor"],
            gpus=1,
            timeout=e["timeout"],
            price_per_hour=e["price_per_hour"],
            estimate=SimpleNamespace(
                hours_low=e["hours_low"],
                hours_high=e["hours_high"],
                cost_low=e["cost_low"],
                cost_high=e["cost_high"],
            ),
            max_cost=e["max_cost"],
            kernels=True,
            kernels_cached=True,
            keep_data=request.keep_data,
        )

    def info(job_id):
        return SimpleNamespace(
            id=JOB,
            url=f"https://huggingface.co/jobs/you/{JOB}",
            status=SimpleNamespace(stage=run.status().stage, message=None),
            labels={"tool": "clef-tune", "run": RUN, "timeout": run.estimate["timeout"]},
            flavor=run.estimate["flavor"],
            created_at=run.started,
        )

    config = run.config or load_yaml_file(example_config)
    from clef_tune.prompt import counter_for_config

    services, _ = fake_services(
        counter=lambda config: counter_for_config(config, token=False),  # real token counts
        whoami=lambda token: "you",
        plan_launch=plan,
        submit_launch=lambda plan, **k: SimpleNamespace(
            id=JOB, url=f"https://huggingface.co/jobs/you/{JOB}", plan=plan
        ),
        list_jobs=lambda **k: [info(JOB)],
        job_status=lambda job_id, **k: run.status(),
        pull_metrics=lambda repo_id, **k: run.metrics,
        pull_run_files=lambda repo_id, **k: RunFiles(run.metrics, config, None),
    )
    return services
