import pytest

torch = pytest.importorskip("torch")
from clef_tune.checkpoints import (  # noqa: E402
    ConfigMismatchError,
    TrainState,
    check_resume,
    clear_checkpoints,
    latest_checkpoint,
    load_baseline,
    load_best,
    load_checkpoint,
    save_baseline,
    save_best,
    save_checkpoint,
)
from clef_tune.config import TrainSection  # noqa: E402
from clef_tune.model import load_for_training, trainable_state_dict  # noqa: E402

pytestmark = pytest.mark.torch
CPU = torch.device("cpu")


def make(tiny_release):
    loaded = load_for_training(tiny_release, TrainSection(lora={"r": 4, "alpha": 8}), device=CPU)
    params = [p for p in loaded.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (step + 1))
    sum(p.sum() for p in params).backward()
    optimizer.step()
    scheduler.step()
    return loaded.model, optimizer, scheduler


def state(step, config_hash="abc"):
    return TrainState(step, 0, step, None, None, 0, config_hash, 100 * step, 1.0)


def test_keeps_the_last_two_and_restores_weights_optimizer_scheduler_and_rng(
    tiny_release, tmp_path
):
    model, optimizer, scheduler = make(tiny_release)
    for step in (1, 2, 3):
        save_checkpoint(tmp_path, model, optimizer, scheduler, state(step))
    names = sorted(p.name for p in (tmp_path / "checkpoints").iterdir())
    assert names == ["step-2", "step-3"]
    saved = trainable_state_dict(model)
    expected_draw = torch.rand(3)
    model2, optimizer2, scheduler2 = make(tiny_release)
    restored = load_checkpoint(latest_checkpoint(tmp_path), model2, optimizer2, scheduler2)
    assert restored == state(3)
    assert torch.equal(torch.rand(3), expected_draw)
    assert all(torch.equal(saved[k], v) for k, v in trainable_state_dict(model2).items())
    assert scheduler2.last_epoch == scheduler.last_epoch
    assert optimizer2.state_dict()["state"].keys() == optimizer.state_dict()["state"].keys()


def test_python_and_numpy_rng_are_restored(tiny_release, tmp_path):
    import random

    import numpy as np

    model, optimizer, scheduler = make(tiny_release)
    save_checkpoint(tmp_path, model, optimizer, scheduler, state(1))
    expected = (random.random(), float(np.random.rand()))
    random.random()
    np.random.rand()
    load_checkpoint(latest_checkpoint(tmp_path), model, optimizer, scheduler)
    assert (random.random(), float(np.random.rand())) == expected


def test_resave_of_same_step_and_tmp_dirs_ignored(tiny_release, tmp_path):
    model, optimizer, scheduler = make(tiny_release)
    save_checkpoint(tmp_path, model, optimizer, scheduler, state(1))
    save_checkpoint(tmp_path, model, optimizer, scheduler, state(1))
    (tmp_path / "checkpoints" / "step-9.tmp").mkdir()
    assert latest_checkpoint(tmp_path).name == "step-1"


def test_check_resume_refuses_a_different_config(tiny_release, tmp_path):
    assert check_resume(tmp_path, "abc") is None
    model, optimizer, scheduler = make(tiny_release)
    save_checkpoint(tmp_path, model, optimizer, scheduler, state(1))
    assert check_resume(tmp_path, "abc").name == "step-1"
    with pytest.raises(ConfigMismatchError, match="--restart"):
        check_resume(tmp_path, "xyz")
    clear_checkpoints(tmp_path)
    assert check_resume(tmp_path, "xyz") is None


def test_best_and_baseline_round_trip(tiny_release, tmp_path):
    model, optimizer, scheduler = make(tiny_release)
    save_best(tmp_path, model, {"q": [[1.0, 2.0]]}, {"q": [[0.0, 1.0]]}, state(2))
    best = trainable_state_dict(model)
    model2, _, _ = make(tiny_release)
    assert load_best(tmp_path, model2) == ({"q": [[1.0, 2.0]]}, {"q": [[0.0, 1.0]]})
    assert all(torch.equal(best[k], v) for k, v in trainable_state_dict(model2).items())
    save_baseline(tmp_path, {"exact_match": 0.25})
    assert load_baseline(tmp_path) == {"exact_match": 0.25}
    assert load_baseline(tmp_path / "nowhere") is None
