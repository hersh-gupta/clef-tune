import dataclasses
import math
import re
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
    rtx = est.estimate_hardware(CLEF, workload(), est.GPUS["RTX-PRO-6000"], 1, 2.75, "rtx")
    assert rtx.fits and rtx.note is None
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
    monkeypatch.setattr(huggingface_hub, "list_jobs_hardware", lambda token=None: hardware)
    assert est.fetch_hf_prices() == {"h200": pytest.approx(4.998)}

    def offline(token=None):
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


def test_memory_counts_the_real_joint_head():
    assert CLEF.head_params == 128_056_324
    assert est.MODELS["Cloudflare/clef-flash"].head_params == 121_762_820
    smaller = dataclasses.replace(CLEF, head_params=CLEF.head_params - 1e8)
    delta = est.memory_needed_gb(CLEF, workload(), "none") - est.memory_needed_gb(
        smaller, workload(), "none"
    )
    assert delta == pytest.approx(1e8 * 16 / 1e9 * est.MEMORY_HEADROOM)


def test_suggest_timeout_adds_setup_minutes():
    row = est.EstimateRow(
        flavor="l4x1",
        gpu="L4",
        count=1,
        price_per_hour=0.8,
        memory_needed_gb=14,
        memory_available_gb=24,
        fits=True,
        fits_with_4bit=True,
        note=None,
        hours=0.2,
    )
    assert est.suggest_timeout(row) == "30m"  # 0.3 h high x 1.25 = 22.5 min -> 30m
    assert est.suggest_timeout(row, extra_minutes=15) == "1h"  # (18 + 15) x 1.25 = 41.25 -> 60


HOPPER_FLAVORS = ["h200", "h200x2", "h200x4", "h200x8"]


def test_only_h100_and_h200_count_as_hopper():
    assert {gpu.name for gpu in est.GPUS.values() if gpu.hopper} == {"H100", "H200"}


@pytest.mark.parametrize("gpu", ["H100", "H200"])
def test_hopper_gpus_cannot_do_lora_training_with_the_fast_kernels(gpu):
    row = est.estimate_hardware(CLEF, workload(), est.GPUS[gpu], 1, 5.0, gpu)
    assert not row.fits and not row.fits_with_4bit  # 4-bit doesn't help, so it mustn't be offered
    assert row.note == est.HOPPER_NOTE


def test_hopper_flavors_are_never_recommended_for_lora_training_however_cheap():
    prices = dict.fromkeys(HOPPER_FLAVORS, 0.01)
    rows = est.estimate_hf_flavors(CLEF, workload(), prices)
    hopper = [row for row in rows if row.flavor in HOPPER_FLAVORS]
    assert len(hopper) == len(HOPPER_FLAVORS) and not any(row.fits for row in hopper)
    assert est.recommend(rows).gpu not in ("H100", "H200")


@pytest.mark.parametrize("changes", [{"mode": "head_only"}, {"kernels": False}])
def test_hopper_still_fits_what_never_runs_the_failing_kernel(changes):
    # Head-only training never backpropagates through attention; no kernels, no such kernel.
    row = est.estimate_hardware(CLEF, workload(**changes), est.GPUS["H200"], 1, 5.0, "h200")
    assert row.fits and row.note is None


def test_a_run_too_big_for_every_usable_gpu_is_told_how_to_fix_it():
    big = workload(max_tokens_per_batch=32768)
    assert est.recommend(est.estimate_hf_flavors(CLEF, big, {})) is None
    message = est.no_hardware_message(CLEF, big)
    assert "about 106 GB" in message and "96 GB" in message
    assert "train.max_tokens_per_batch" in message
    tokens = int(re.search(r"to ([0-9,]+) or less", message).group(1).replace(",", ""))
    assert tokens % 1000 == 0
    lowered = workload(max_tokens_per_batch=tokens)
    assert est.recommend(est.estimate_hf_flavors(CLEF, lowered, {})) is not None


def test_when_a_smaller_batch_cannot_help_the_message_says_what_can():
    message = est.no_hardware_message(CLEF, workload(lora_r=1024))
    assert "train.max_tokens_per_batch" not in message and "train.lora.r" in message


def test_without_the_kernels_the_largest_gpu_counts_again():
    message = est.no_hardware_message(CLEF, workload(max_tokens_per_batch=300_000, kernels=False))
    assert "141 GB" in message


def test_one_long_prompt_sets_the_memory_even_when_the_batch_is_smaller():
    # A prompt is never split across batches, so the longest one is a batch on its own.
    long = est.memory_needed_gb(CLEF, workload(longest_prompt=65536), "none")
    assert long == est.memory_needed_gb(CLEF, workload(max_tokens_per_batch=65536), "none")
    short = est.memory_needed_gb(CLEF, workload(longest_prompt=1000), "none")
    assert short == est.memory_needed_gb(CLEF, workload(), "none")


def test_the_workload_carries_the_longest_prompt(tmp_path):
    cfg = make_config(tmp_path, data={"train": "t.csv"}, train={"max_length": 40_000})
    tokens = TokenStats(train_lengths=[1000, 40_000], eval_lengths=[1000], truncated_rows=0)
    assert est.workload_from(cfg, tokens).largest_batch == 40_000
    assert est.prediction_workload(cfg, 1000, longest_prompt=50_000).largest_batch == 50_000
    assert est.prediction_workload(cfg, 1000).largest_batch == cfg.train.max_tokens_per_batch


def test_a_prompt_too_long_for_every_usable_gpu_is_told_to_shorten_it():
    big = workload(longest_prompt=65536)
    assert est.recommend(est.estimate_hf_flavors(CLEF, big, {})) is None
    message = est.no_hardware_message(CLEF, big)
    assert "65,536 tokens" in message and "train.max_length" in message
    assert "train.max_tokens_per_batch" not in message
    tokens = int(re.search(r"to ([0-9,]+) or less", message).group(1).replace(",", ""))
    shorter = workload(longest_prompt=tokens)
    assert est.recommend(est.estimate_hf_flavors(CLEF, shorter, {})) is not None
