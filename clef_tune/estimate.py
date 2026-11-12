"""Napkin-math estimates of run time, cost and memory fit, shown to users as ranges.

FLOPs: LoRA with gradient checkpointing ~ 6 x params x tokens (forward, recompute, backward
through frozen weights); head-only ~ 2 x params x tokens (forward only); eval ~ 2 x params x
tokens. The trainer replaces these with measured throughput after 20 steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from clef_tune.config import Config
    from clef_tune.validation import TokenStats

UTILIZATION = 0.35
DDP_EFFICIENCY_PER_DOUBLING = 0.9
QLORA_SLOWDOWN = 0.7
RANGE_LOW = 0.7
RANGE_HIGH = 1.5
MEMORY_HEADROOM = 1.15
CUDA_CONTEXT_GB = 1.5
WEIGHT_BYTES_4BIT = 0.55
TIMEOUT_MARGIN = 1.25
HOPPER_NOTE = "H100 and H200 can't do this training"


@dataclass(frozen=True)
class GPUSpec:
    name: str
    bf16_tflops: float  # approximate dense BF16 tensor throughput from datasheets
    memory_gb: float
    supports_bf16: bool = True
    hopper: bool = False


GPUS: dict[str, GPUSpec] = {
    gpu.name: gpu
    for gpu in [
        GPUSpec("T4", 65, 16, supports_bf16=False),
        GPUSpec("A10G", 70, 24),
        GPUSpec("L4", 121, 24),
        GPUSpec("L40S", 362, 48),
        GPUSpec("A100-80GB", 312, 80),
        GPUSpec("H100", 989, 80, hopper=True),
        GPUSpec("H200", 989, 141, hopper=True),
        GPUSpec("B200", 2250, 180),
        GPUSpec("RTX-PRO-6000", 500, 96),
        GPUSpec("RTX-4090", 165, 24),
        GPUSpec("RTX-5090", 209, 32),
    ]
}


@dataclass(frozen=True)
class Flavor:
    name: str
    gpu: str
    count: int
    price_per_hour: float  # fallback when live prices are unavailable


HF_FLAVORS: dict[str, Flavor] = {
    flavor.name: flavor
    for flavor in [
        Flavor("t4-small", "T4", 1, 0.40),
        Flavor("t4-medium", "T4", 1, 0.60),
        Flavor("a10g-small", "A10G", 1, 1.00),
        Flavor("a10g-large", "A10G", 1, 1.50),
        Flavor("a10g-largex2", "A10G", 2, 3.00),
        Flavor("a10g-largex4", "A10G", 4, 5.00),
        Flavor("a100-large", "A100-80GB", 1, 2.50),
        Flavor("a100x4", "A100-80GB", 4, 10.00),
        Flavor("a100x8", "A100-80GB", 8, 20.00),
        Flavor("h200", "H200", 1, 5.00),
        Flavor("h200x2", "H200", 2, 10.00),
        Flavor("h200x4", "H200", 4, 20.00),
        Flavor("h200x8", "H200", 8, 40.00),
        Flavor("rtx-pro-6000", "RTX-PRO-6000", 1, 2.75),
        Flavor("rtx-pro-6000x2", "RTX-PRO-6000", 2, 5.50),
        Flavor("rtx-pro-6000x4", "RTX-PRO-6000", 4, 11.00),
        Flavor("rtx-pro-6000x8", "RTX-PRO-6000", 8, 22.00),
        Flavor("l4x1", "L4", 1, 0.80),
        Flavor("l4x4", "L4", 4, 3.80),
        Flavor("l40sx1", "L40S", 1, 1.80),
        Flavor("l40sx4", "L40S", 4, 8.30),
        Flavor("l40sx8", "L40S", 8, 23.50),
    ]
}


@dataclass(frozen=True)
class ModelSpec:
    id: str
    flop_params: float  # matmul params per text token (no vision tower, embeddings or lm_head)
    total_params: float
    embedding_params: float  # input embeddings + lm_head; kept in bf16 under 4-bit
    hidden: int
    layers: int
    intermediate: int
    overhead_minutes: int  # start, install, load, save, push
    head_params: float  # joint schema head, counted from the release's joint_head_config.json


# Derived from config.json and safetensors totals: vocab 248320; vision tower ~0.43B params.
# head_params were measured by building JointSchemaHead from each release's head config.
MODELS: dict[str, ModelSpec] = {
    "Cloudflare/clef": ModelSpec(
        "Cloudflare/clef", 24.4e9, 27.36e9, 2.54e9, 5120, 64, 17408, 20, 128_056_324
    ),
    "Cloudflare/clef-flash": ModelSpec(
        "Cloudflare/clef-flash", 6.95e9, 9.41e9, 2.03e9, 4096, 32, 12288, 10, 121_762_820
    ),
}


def model_spec(model_id: str) -> ModelSpec | None:
    return MODELS.get(model_id)


@dataclass(frozen=True)
class Workload:
    train_tokens_per_epoch: int
    eval_tokens: int
    epochs: int
    eval_passes: int
    mode: Literal["lora", "head_only"]
    quantization: Literal["none", "4bit"]
    max_tokens_per_batch: int
    lora_r: int
    gradient_checkpointing: bool = True
    kernels: bool = True  # the fast attention kernels; see cannot_train
    longest_prompt: int = 0  # a prompt is never split, so one this long is a batch on its own

    @property
    def largest_batch(self) -> int:
        return max(self.max_tokens_per_batch, self.longest_prompt)


def optimizer_steps(config: Config, tokens: TokenStats) -> int:
    """How many optimizer steps a run takes over all its epochs."""
    train = config.train
    tokens_per_step = train.max_tokens_per_batch * train.grad_accum_steps
    return max(1, math.ceil(tokens.train_tokens / tokens_per_step)) * train.epochs


def workload_from(config: Config, tokens: TokenStats, *, kernels: bool = True) -> Workload:
    train = config.train
    total_steps = optimizer_steps(config, tokens)
    eval_passes = 2 + total_steps // config.eval.eval_every  # baseline + periodic + final
    return Workload(
        train_tokens_per_epoch=tokens.train_tokens,
        eval_tokens=tokens.eval_tokens,
        epochs=train.epochs,
        eval_passes=eval_passes,
        mode=train.mode,
        quantization=train.quantization,
        max_tokens_per_batch=train.max_tokens_per_batch,
        lora_r=train.lora.r,
        gradient_checkpointing=train.gradient_checkpointing,
        kernels=kernels,
        longest_prompt=tokens.max,
    )


def prediction_workload(config: Config, tokens: int, longest_prompt: int = 0) -> Workload:
    """One forward pass over `tokens` prompt tokens: training_flops then counts only the
    evaluation term, and the memory is a frozen (head-only) model's."""
    train = config.train
    return Workload(
        train_tokens_per_epoch=0,
        eval_tokens=tokens,
        epochs=1,
        eval_passes=1,
        mode="head_only",
        quantization=train.quantization,
        max_tokens_per_batch=train.max_tokens_per_batch,
        lora_r=train.lora.r,
        gradient_checkpointing=train.gradient_checkpointing,
        longest_prompt=longest_prompt,
    )


def training_flops(model: ModelSpec, workload: Workload) -> float:
    per_token = 6 if workload.mode == "lora" else 2
    train = per_token * model.flop_params * workload.train_tokens_per_epoch * workload.epochs
    evaluation = 2 * model.flop_params * workload.eval_tokens * workload.eval_passes
    return train + evaluation


def throughput(gpu: GPUSpec, count: int, quantization: str) -> float:
    flops = gpu.bf16_tflops * 1e12 * UTILIZATION * count
    flops *= DDP_EFFICIENCY_PER_DOUBLING ** math.log2(count)
    return flops * (QLORA_SLOWDOWN if quantization == "4bit" else 1.0)


def memory_needed_gb(model: ModelSpec, workload: Workload, quantization: str) -> float:
    """Per-GPU memory; DDP replicates the whole model on every GPU."""
    if quantization == "4bit":
        quantized = model.total_params - model.embedding_params
        weights = quantized * WEIGHT_BYTES_4BIT + model.embedding_params * 2
    else:
        weights = model.total_params * 2
    lora_params = 0.0
    if workload.mode == "lora":
        per_layer = 3 * (model.hidden + model.intermediate) + 8 * model.hidden
        lora_params = workload.lora_r * model.layers * per_layer
    trainable = (lora_params + model.head_params) * 16  # fp32 weights + grads + two Adam moments
    tokens = workload.largest_batch
    layer_working_set = tokens * (18 * model.hidden + 4 * model.intermediate) * 2
    if workload.mode == "head_only":
        activations = layer_working_set
    elif workload.gradient_checkpointing:
        activations = tokens * model.hidden * model.layers * 2 + layer_working_set
    else:
        activations = layer_working_set * model.layers
    return (weights + trainable + activations) / 1e9 * MEMORY_HEADROOM + CUDA_CONTEXT_GB


def cannot_train(gpu: GPUSpec, mode: str, kernels: bool) -> bool:
    """flash-linear-attention 0.6.0 refuses to backpropagate through gated attention on Hopper
    GPUs with Triton 3.4 to 3.7.0, which gives wrong gradients there, and torch 2.11 pins Triton
    3.6.0. Head-only training and predictions never backpropagate through attention, and without
    the kernels the slow path runs instead."""
    return gpu.hopper and kernels and mode == "lora"


@dataclass(frozen=True)
class EstimateRow:
    flavor: str
    gpu: str
    count: int
    price_per_hour: float | None
    memory_needed_gb: float
    memory_available_gb: float
    fits: bool
    fits_with_4bit: bool
    note: str | None
    hours: float

    @property
    def hours_low(self) -> float:
        return self.hours * RANGE_LOW

    @property
    def hours_high(self) -> float:
        return self.hours * RANGE_HIGH

    def _cost(self, hours: float) -> float | None:
        return None if self.price_per_hour is None else hours * self.price_per_hour

    @property
    def cost_point(self) -> float | None:
        return self._cost(self.hours)

    @property
    def cost_low(self) -> float | None:
        return self._cost(self.hours_low)

    @property
    def cost_high(self) -> float | None:
        return self._cost(self.hours_high)


def estimate_hardware(
    model: ModelSpec,
    workload: Workload,
    gpu: GPUSpec,
    count: int,
    price: float | None,
    name: str,
) -> EstimateRow:
    need = memory_needed_gb(model, workload, workload.quantization)
    need_4bit = memory_needed_gb(model, workload, "4bit")
    usable = gpu.supports_bf16 and not cannot_train(gpu, workload.mode, workload.kernels)
    fits = usable and need <= gpu.memory_gb
    fits_4bit = usable and need_4bit <= gpu.memory_gb
    note = None
    if not gpu.supports_bf16:
        note = "no bf16 support"
    elif not usable:
        note = HOPPER_NOTE
    elif not fits and fits_4bit and workload.quantization == "none":
        note = "fits only with 4-bit (QLoRA)"
    elif not fits:
        note = f"needs ~{need:.0f} GB, has {gpu.memory_gb:.0f} GB"
    compute_seconds = training_flops(model, workload) / throughput(
        gpu, count, workload.quantization
    )
    hours = compute_seconds / 3600 + model.overhead_minutes / 60
    return EstimateRow(
        flavor=name,
        gpu=gpu.name,
        count=count,
        price_per_hour=price,
        memory_needed_gb=need,
        memory_available_gb=gpu.memory_gb,
        fits=fits,
        fits_with_4bit=fits_4bit,
        note=note,
        hours=hours,
    )


def estimate_hf_flavors(
    model: ModelSpec, workload: Workload, prices: dict[str, float] | None = None
) -> list[EstimateRow]:
    prices = prices or {}
    rows = [
        estimate_hardware(
            model,
            workload,
            GPUS[flavor.gpu],
            flavor.count,
            prices.get(flavor.name, flavor.price_per_hour),
            flavor.name,
        )
        for flavor in HF_FLAVORS.values()
    ]
    return sorted(
        rows,
        key=lambda r: (not r.fits, r.cost_point if r.cost_point is not None else math.inf),
    )


def recommend(rows: list[EstimateRow]) -> EstimateRow | None:
    fitting = [row for row in rows if row.fits and row.cost_point is not None]
    return min(fitting, key=lambda row: row.cost_point) if fitting else None


def _largest_batch(model: ModelSpec, workload: Workload, memory_gb: float) -> int:
    """The largest max_tokens_per_batch that fits, rounded down to a thousand. Memory grows in a
    straight line with the batch."""
    empty, one = (
        memory_needed_gb(
            model,
            replace(workload, max_tokens_per_batch=t, longest_prompt=0),
            workload.quantization,
        )
        for t in (0, 1)
    )
    return int((memory_gb - empty) / (one - empty)) // 1000 * 1000


def no_hardware_message(model: ModelSpec, workload: Workload) -> str:
    """Why no HF Jobs flavor can take this workload, and what to change."""
    largest = max(
        GPUS[flavor.gpu].memory_gb
        for flavor in HF_FLAVORS.values()
        if GPUS[flavor.gpu].supports_bf16
        and not cannot_train(GPUS[flavor.gpu], workload.mode, workload.kernels)
    )
    need = memory_needed_gb(model, workload, workload.quantization)
    problem = (
        "This run is too big for any GPU on Hugging Face that can train it: it needs about "
        f"{need:.0f} GB of memory, and the largest has {largest:.0f} GB."
    )
    tokens = _largest_batch(model, workload, largest)
    if tokens > 0 and workload.longest_prompt > workload.max_tokens_per_batch:
        return (
            f"{problem} The longest prompt alone is {workload.longest_prompt:,} tokens. "
            f"Lower the prompt length (train.max_length) to {tokens:,} or less."
        )
    if tokens > 0:
        return (
            f"{problem} Lower the tokens per batch (train.max_tokens_per_batch) "
            f"to {tokens:,} or less."
        )
    return (
        f"{problem} Lower the LoRA rank (train.lora.r), "
        "or load the model in 4-bit (train.quantization)."
    )


def estimate_prediction(
    model: ModelSpec, config: Config, tokens: int, prices: dict[str, float] | None = None
) -> EstimateRow | None:
    """The cheapest fitting hardware for predicting `tokens` prompt tokens, or None."""
    return recommend(estimate_hf_flavors(model, prediction_workload(config, tokens), prices))


def suggest_timeout(row: EstimateRow, extra_minutes: float = 0) -> str:
    """HF Jobs timeout: (the high estimate + setup minutes) x 1.25, rounded up to 30 minutes."""
    minutes = math.ceil((row.hours_high * 60 + extra_minutes) * TIMEOUT_MARGIN / 30) * 30
    if minutes < 60:
        return f"{minutes}m"
    return f"{minutes // 60}h" if minutes % 60 == 0 else f"{minutes / 60:g}h"


def fetch_hf_prices(token: bool | str | None = None) -> dict[str, float]:
    """Live $/hour per HF Jobs flavor; empty when the Hub can't be reached. `token` is passed
    to the Hub as it is (False sends none)."""
    try:
        from huggingface_hub import list_jobs_hardware

        hardware = list_jobs_hardware(token=token)
    except Exception:  # best effort: offline, rate-limited, or API changes fall back to defaults
        return {}
    per_unit_to_hour = {"second": 3600, "minute": 60, "hour": 1}
    prices: dict[str, float] = {}
    for item in hardware:
        cost = getattr(item, "unit_cost_usd", None)
        unit = getattr(item, "unit_label", "minute")
        if cost is None or unit not in per_unit_to_hour:
            continue
        prices[item.name] = cost * per_unit_to_hour[unit]
    return prices
