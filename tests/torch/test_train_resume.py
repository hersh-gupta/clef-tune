import json
import math

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import load_file  # noqa: E402

from clef_tune import train as train_module  # noqa: E402
from clef_tune.adapter import read_adapter  # noqa: E402
from clef_tune.bundle import write_bundle  # noqa: E402
from clef_tune.checkpoints import ConfigMismatchError  # noqa: E402
from clef_tune.config import load_config  # noqa: E402
from clef_tune.train import TrainingError, train  # noqa: E402
from clef_tune.validation import validate  # noqa: E402

pytestmark = pytest.mark.torch


class Stop(Exception):
    pass


# 800-token batches hold 1-2 tiny records, so an epoch has 8-16 batches (4-8 optimizer steps at
# grad accumulation 2), and the step-3 checkpoint always lands mid-epoch.
OVERRIDES = dict(
    train={"epochs": 2, "grad_accum_steps": 2, "max_tokens_per_batch": 800},
    eval={"eval_every": 2},
    output={"save_every": 3},
)


def adapter_tensors(result):
    tensors = load_file(result.adapter_dir / "joint_head.safetensors")
    tensors.update(load_file(result.adapter_dir / "adapter" / "adapter_model.safetensors"))
    return tensors


def events(config_path):
    lines = (config_path.parent / "out" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def logged_by_step(config_path, event, key):
    """The last logged value per step: a resumed run re-logs the steps after its checkpoint."""
    return {e["step"]: e[key] for e in events(config_path) if e["event"] == event}


def test_interrupted_run_resumes_to_the_same_result(tiny_project):
    full_config = tiny_project("full", **OVERRIDES)
    full = train(full_config)
    config = tiny_project("resumed", **OVERRIDES)

    def stop_at_four(step):
        if step == 4:
            raise Stop

    with pytest.raises(Stop):
        train(config, on_step=stop_at_four)
    resumed = train(config)
    assert full.resumed_from is None and resumed.resumed_from == 3
    expected, got = adapter_tensors(full), adapter_tensors(resumed)
    assert expected.keys() == got.keys()
    assert all(torch.allclose(expected[k], got[k], atol=1e-6) for k in expected)
    for key in ("exact_match", "mean_log_loss"):
        assert resumed.metrics["final"][key] == pytest.approx(full.metrics["final"][key], abs=1e-6)
    # The best eval can predate the checkpoint, so also compare what the steps after it logged:
    # every eval, and the last train event (in both runs the mean loss of steps 11-16).
    expected_evals = logged_by_step(full_config, "eval", "mean_log_loss")
    got_evals = logged_by_step(config, "eval", "mean_log_loss")
    assert expected_evals.keys() == got_evals.keys()
    assert all(got_evals[s] == pytest.approx(expected_evals[s], abs=1e-6) for s in expected_evals)
    expected_loss = logged_by_step(full_config, "train", "loss")
    got_loss, last = logged_by_step(config, "train", "loss"), max(expected_loss)
    assert got_loss[last] == pytest.approx(expected_loss[last], abs=1e-6)


def test_changed_config_refuses_to_resume_until_restart(tiny_project):
    config = tiny_project(train={"epochs": 2}, output={"save_every": 1})

    def stop(step):
        raise Stop

    with pytest.raises(Stop):
        train(config, on_step=stop)
    config.write_text(
        config.read_text(encoding="utf-8").replace("0.003", "0.002"), encoding="utf-8"
    )
    with pytest.raises(ConfigMismatchError, match="--restart"):
        train(config)
    assert train(config, restart=True).resumed_from is None


def test_bundle_is_deleted_only_on_success_and_only_if_it_is_a_bundle(tiny_project, tmp_path):
    source = load_config(tiny_project(train={"epochs": 1}))
    bundle_config = write_bundle(validate(source, None), source, tmp_path / "bundle")
    train(bundle_config, delete_bundle_on_success=True)
    remaining = {p.name for p in (tmp_path / "bundle").iterdir()}
    assert remaining == {"out"}  # data and config gone; the run's output stays
    with pytest.raises(TrainingError, match="is not a bundle"):
        train(tiny_project("plain", train={"epochs": 1}), delete_bundle_on_success=True)


def test_nan_loss_stops_and_names_the_last_good_checkpoint(tiny_project, monkeypatch):
    real, calls = train_module.batch_loss, []

    def poisoned(*args, **kwargs):
        calls.append(1)
        loss = real(*args, **kwargs)
        return loss * float("nan") if len(calls) == 5 else loss

    monkeypatch.setattr(train_module, "batch_loss", poisoned)
    config = tiny_project(output={"save_every": 2})
    with pytest.raises(TrainingError, match="non-finite at step 5.*step 4"):
        train(config)
    assert (config.parent / "out" / "checkpoints" / "step-4").exists()


def test_out_of_memory_is_reraised_with_advice(tiny_project, monkeypatch):
    def oom(*args, **kwargs):
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(train_module, "batch_loss", oom)
    with pytest.raises(TrainingError) as error:
        train(tiny_project())
    for hint in ("out of memory", "max_tokens_per_batch", "max_length", "quantization: 4bit"):
        assert hint in str(error.value)


def test_short_final_accumulation_group_keeps_the_schedule(tiny_project, monkeypatch):
    # Tiny records are 403-404 tokens, so a 2500-token budget packs 6 per batch: 3 batches per
    # epoch, and each epoch ends with a one-batch accumulation group.
    real, counts = train_module.token_budget_batches, []

    def counting(*args, **kwargs):
        batches = real(*args, **kwargs)
        counts.append(len(batches))
        return batches

    monkeypatch.setattr(train_module, "token_budget_batches", counting)
    config = tiny_project(train={"epochs": 2, "grad_accum_steps": 2, "max_tokens_per_batch": 2500})
    run = train(config).metrics["run"]
    assert counts[0] % 2 == 1
    assert run["steps"] == 2 * math.ceil(counts[0] / 2)
    last = [e for e in events(config) if e["event"] == "train"][-1]
    assert last["step"] == run["steps"]
    assert last["lr"] == pytest.approx(0.0, abs=1e-9)  # the cosine schedule ran to its end


def test_head_only_rerun_leaves_no_stale_lora_dir(tiny_project):
    first = train(tiny_project(train={"epochs": 1}))
    assert (first.adapter_dir / "adapter").exists()
    config = tiny_project(train={"mode": "head_only", "epochs": 1})
    final = train(config, restart=True).adapter_dir
    assert not (final / "adapter").exists()
    assert read_adapter(final).has_lora is False
    # A run that does not resume starts a new metrics log.
    assert [e["event"] for e in events(config)].count("baseline") == 1


def test_resume_after_an_early_stop_finishes_without_training_on(tiny_project):
    config = tiny_project(
        train={"lora_lr": 1e-12, "head_lr": 1e-12, "epochs": 2},
        eval={"eval_every": 1, "early_stopping_patience": 2},
        output={"save_every": 1},
    )

    def stop_at_three(step):  # the early stop's step, after its checkpoint
        if step == 3:
            raise Stop

    with pytest.raises(Stop):
        train(config, on_step=stop_at_three)
    result = train(config)
    run = result.metrics["run"]
    assert result.resumed_from == 3 and run["stopped_early"] and run["steps"] == 3
    logged = events(config)
    assert [e["event"] for e in logged].count("baseline") == 1  # a resume appends to the log
    assert [e["step"] for e in logged if e["event"] == "eval"] == [1, 2, 3]
    assert [e for e in logged if e["event"] == "train"][-1]["step"] == 3  # the stop logs its step


def test_a_directory_whose_config_points_elsewhere_is_not_a_bundle(tiny_project, tmp_path):
    source = load_config(tiny_project(train={"epochs": 1}))
    bundle_config = write_bundle(validate(source, None), source, tmp_path / "bundle")
    folder = bundle_config.parent
    (folder / "other.jsonl").write_text((folder / "train.jsonl").read_text(encoding="utf-8"))
    bundle_config.write_text(
        bundle_config.read_text(encoding="utf-8").replace("train.jsonl", "other.jsonl"),
        encoding="utf-8",
    )
    with pytest.raises(TrainingError, match="is not a bundle"):
        train(bundle_config, delete_bundle_on_success=True)
    assert (folder / "train.jsonl").exists()


def test_resuming_with_a_different_number_of_processes_is_refused(tiny_project):
    config = tiny_project(train={"epochs": 2}, output={"save_every": 1})

    def stop(step):
        raise Stop

    with pytest.raises(Stop):
        train(config, on_step=stop)
    state_file = config.parent / "out" / "checkpoints" / "step-1" / "state.json"
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert state["world_size"] == 1
    state_file.write_text(json.dumps({**state, "world_size": 2}), encoding="utf-8")
    with pytest.raises(TrainingError, match="made with 2 processes; resume with 2 processes"):
        train(config)
    assert train(config, restart=True).resumed_from is None
