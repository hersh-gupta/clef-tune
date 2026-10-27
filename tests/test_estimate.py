import math
from types import SimpleNamespace

import pytest

from clef_tune import estimate as est
from clef_tune.validation import TokenStats
from tests.helpers import make_config

CLEF = est.MODELS["Cloudflare/clef"]


def workload(**overrides):
    values = dict(
        train_tokens_per_epoch=8_000_000,
        eval_tokens=0,
        epochs=3,
        eval_passes=0,
        mode="lora",
        quantization="none",
        max_tokens_per_batch=16384,
        lora_r=16,
    )
    values.update(overrides)
    return est.Workload(**values)


def test_training_flops_hand_calculation():
    assert est.training_flops(CLEF, workload()) == pytest.approx(6 * 24.4e9 * 8e6 * 3)
    head_only = est.training_flops(CLEF, workload(mode="head_only"))
    assert head_only == pytest.approx(2 * 24.4e9 * 8e6 * 3)
    with_eval = est.training_flops(CLEF, workload(eval_tokens=1_000_000, eval_passes=4))
    assert with_eval - est.training_flops(CLEF, workload()) == pytest.approx(2 * 24.4e9 * 4e6)


def test_hours_on_one_h200():
    row = est.estimate_hardware(CLEF, workload(), est.GPUS["H200"], 1, 5.0, "h200")
    compute_hours = 6 * 24.4e9 * 8e6 * 3 / (989e12 * 0.35) / 3600
    assert row.hours == pytest.approx(compute_hours + 20 / 60)
    assert row.hours_low == pytest.approx(row.hours * 0.7)
    assert row.hours_high == pytest.approx(row.hours * 1.5)
    assert row.cost_point == pytest.approx(row.hours * 5.0)


def test_ddp_and_qlora_throughput():
    single = est.throughput(est.GPUS["H200"], 1, "none")
    assert est.throughput(est.GPUS["H200"], 2, "none") == pytest.approx(2 * 0.9 * single)
    assert est.throughput(est.GPUS["H200"], 1, "4bit") == pytest.approx(0.7 * single)


def test_memory_fit_clef_bf16_vs_4bit():
    need = est.memory_needed_gb(CLEF, workload(), "none")
    assert 80 < need < 95
    a100 = est.estimate_hardware(CLEF, workload(), est.GPUS["A100-80GB"], 1, 2.5, "a100-large")
    assert not a100.fits and a100.fits_with_4bit
    assert a100.note == "fits only with 4-bit (QLoRA)"
    h200 = est.estimate_hardware(CLEF, workload(), est.GPUS["H200"], 1, 5.0, "h200")
    assert h200.fits and h200.note is None
    assert est.memory_needed_gb(CLEF, workload(), "4bit") < 48


def test_no_gradient_checkpointing_needs_far_more_memory():
    off = est.memory_needed_gb(CLEF, workload(gradient_checkpointing=False), "none")
    assert off > 3 * est.memory_needed_gb(CLEF, workload(), "none")


def test_t4_is_not_supported():
    flash = est.MODELS["Cloudflare/clef-flash"]
    row = est.estimate_hardware(flash, workload(quantization="4bit"), est.GPUS["T4"], 1, 0.4, "t4")
    assert not row.fits and row.note == "no bf16 support"


def test_hf_flavors_sorted_and_recommendation():
    rows = est.estimate_hf_flavors(CLEF, workload(), prices={"h200": 5.0})
    assert rows[0].fits
    assert all(r.fits for r in rows[: sum(r.fits for r in rows)])
    best = est.recommend(rows)
    fitting_costs = [r.cost_point for r in rows if r.fits]
    assert best.cost_point == min(fitting_costs)
    assert est.recommend([r for r in rows if not r.fits]) is None


def test_suggest_timeout_rounds_up_to_half_hours():
    row = est.estimate_hardware(CLEF, workload(), est.GPUS["H200"], 1, 5.0, "h200")
    minutes = math.ceil(row.hours_high * 1.25 * 60 / 30) * 30
    expected = f"{minutes // 60}h" if minutes % 60 == 0 else f"{minutes / 60:g}h"
    assert est.suggest_timeout(row) == expected
    fixed = est.EstimateRow("x", "H200", 1, None, 1, 141, True, True, None, hours=4.0)
    assert est.suggest_timeout(fixed) == "7.5h"
    assert fixed.cost_point is None
    short = est.EstimateRow("x", "H200", 1, None, 1, 141, True, True, None, hours=0.2)
    assert est.suggest_timeout(short) == "30m"


def test_fetch_prices(monkeypatch):
    import huggingface_hub

    hardware = [
        SimpleNamespace(name="h200", unit_cost_usd=0.0833, unit_label="minute"),
        SimpleNamespace(name="weird", unit_cost_usd=None, unit_label="minute"),
    ]
    monkeypatch.setattr(huggingface_hub, "list_jobs_hardware", lambda: hardware)
    assert est.fetch_hf_prices() == {"h200": pytest.approx(4.998)}

    def offline():
        raise OSError("no network")

    monkeypatch.setattr(huggingface_hub, "list_jobs_hardware", offline)
    assert est.fetch_hf_prices() == {}


def test_workload_from_config(tmp_path):
    cfg = make_config(tmp_path, data={"train": "t.csv"}, eval={"eval_every": 2})
    tokens = TokenStats(train_lengths=[1000] * 200, eval_lengths=[1000] * 20, truncated_rows=0)
    w = est.workload_from(cfg, tokens)
    # 200k tokens / (16384 * 4 per step) -> 4 steps/epoch, 12 total -> 2 + 12 // 2 = 8 passes
    assert (w.train_tokens_per_epoch, w.eval_tokens, w.epochs) == (200_000, 20_000, 3)
    assert w.eval_passes == 8
    assert (w.mode, w.quantization, w.lora_r) == ("lora", "none", 16)


def test_model_spec_lookup():
    assert est.model_spec("Cloudflare/clef-flash").layers == 32
    assert est.model_spec("someone/else") is None
