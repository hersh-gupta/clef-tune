import sys
import time

import pytest

from clef_tune.ui import handlers, local
from clef_tune.ui.session import Session
from tests.ui.fakes import fake_services

FAKE_TRAIN = r"""
import json, pathlib, sys
from clef_tune.config import load_config
out = load_config(sys.argv[1]).output_dir(); out.mkdir(parents=True, exist_ok=True)
with open(out / "metrics.jsonl", "w") as f:
    f.write(json.dumps({"event": "train", "step": 5, "loss": 0.5, "lr": 1e-4, "tokens_per_sec": 9.0,
                        "eta_s": 1.0, "projected_cost": None, "total_steps": 10}) + "\n")
print("<i>trained</i>")
"""


def fake_command(config_path):
    return [sys.executable, "-c", FAKE_TRAIN, str(config_path)]


def sleeper(config_path):
    return [sys.executable, "-c", "import time; time.sleep(60)"]


def wait(run, timeout=30):
    deadline = time.monotonic() + timeout
    while run.process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)


@pytest.fixture
def session():
    s = Session()
    handlers.sample(s)
    handlers.refresh(s, fake_services()[0])
    yield s
    s.close()


def test_a_local_run_trains_in_a_subprocess_and_is_monitored(session, tmp_path):
    run = local.start_local(session, str(tmp_path / "run1"), command=fake_command)
    wait(run)
    state, results = local.local_state(session)
    assert run.stage() == "COMPLETED" and state.status == "done"
    assert any(stage.detail == "Step 5 of 10" for stage in state.stages)
    assert state.stages[0].detail == "This computer's GPU" and results is None
    assert (tmp_path / "run1" / "config.yaml").exists()


def test_an_existing_folder_is_not_overwritten(session, tmp_path):
    (tmp_path / "taken").mkdir()
    (tmp_path / "taken" / "notes.txt").write_text("mine")
    with pytest.raises(local.LocalRunError, match="already exists"):
        local.start_local(session, str(tmp_path / "taken"), command=fake_command)


def test_stopping_marks_the_run_canceled(session, tmp_path):
    run = local.start_local(session, str(tmp_path / "slow"), command=sleeper)
    local.stop_local(str(tmp_path / "slow"))
    wait(run)
    state, _ = local.local_state(session)
    assert run.stage() == "CANCELED" and state.status == "stopped" and state.can_resume


def test_a_run_needs_validated_data(tmp_path):
    s = Session()
    handlers.sample(s)
    with pytest.raises(local.LocalRunError, match="validation"):
        local.start_local(s, str(tmp_path / "early"), command=fake_command)
    s.close()
