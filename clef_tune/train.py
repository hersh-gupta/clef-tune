"""The training run: validate, load Clef, train with Accelerate (resuming from the latest
checkpoint), keep the best eval, calibrate from its cached logits and write the adapter."""

from __future__ import annotations

import json
import math
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import (
    InitProcessGroupKwargs,
    broadcast_object_list,
    gather_object,
    set_seed,
)
from transformers import get_cosine_schedule_with_warmup

from clef_tune.adapter import save_adapter
from clef_tune.batching import shard_batches, token_budget_batches
from clef_tune.calibration import MIN_ROWS, calibrate, temperatures
from clef_tune.checkpoints import (
    TrainState,
    best_complete,
    check_resume,
    clear_checkpoints,
    latest_checkpoint,
    load_baseline,
    load_best,
    load_checkpoint,
    read_state,
    save_baseline,
    save_best,
    save_checkpoint,
)
from clef_tune.clef_code import model_dir
from clef_tune.config import Config, TrainSection, config_hash, load_config
from clef_tune.encoding import TrainItem, collate, prepare_items
from clef_tune.estimate import memory_needed_gb, model_spec, workload_from
from clef_tune.evaluation import EvalResult, predict_logits, summarize_eval
from clef_tune.hub import HubError, ensure_repo, push_adapter
from clef_tune.loss import batch_loss
from clef_tune.metrics import MetricsReport, is_better
from clef_tune.model import LoadedClef, default_device, load_for_training
from clef_tune.prompt import counter_for_config
from clef_tune.runtime import ESTIMATE_ENV, PRICE_ENV, env_float, kernels_available
from clef_tune.validation import ValidationReport, validate
from clef_tune.weights import class_weights

LOG_EVERY = 10
ETA_AFTER_STEPS = 20
MAX_GRAD_NORM = 1.0
METRICS_LOG = "metrics.jsonl"
FINAL_DIR = "final"
OOM_ADVICE = (
    "lower train.max_tokens_per_batch or train.max_length, set train.quantization: 4bit, "
    "or use a larger GPU flavor"
)
BUNDLE_FILES = ("config.yaml", "train.jsonl", "eval.jsonl")
BUNDLE_IMAGES = "images"
# NCCL's default 10 minutes is too short for the other ranks to wait while rank 0 pushes.
DIST_TIMEOUT = timedelta(hours=1)


class TrainingError(RuntimeError):
    """Training could not start or did not finish; the message says why."""


@dataclass
class TrainResult:
    adapter_dir: Path
    metrics: dict
    resumed_from: int | None
    hub_url: str | None
    main_process: bool = True  # only the main process reports the result


class _EventLog:
    """One JSON object per line in metrics.jsonl, a short line on stdout, and the numbers to any
    Accelerate trackers. A resumed run appends to the log; any other run starts a new one. Only
    the main process writes. Each event opens and closes the file: a bucket mount uploads a file
    only when it is closed, so `jobs status` can read the log while the job runs."""

    def __init__(self, path: Path, accelerator: Accelerator, trackers: bool, append: bool) -> None:
        self._accelerator = accelerator
        self._trackers = trackers
        self._path = path if accelerator.is_main_process else None
        if self._path is not None and not append:
            open(self._path, "w", encoding="utf-8").close()

    def __enter__(self) -> _EventLog:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def __call__(self, event: str, **fields: Any) -> None:
        if self._path is None:
            return
        record = {"event": event, **fields}
        with open(self._path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(_describe(record), flush=True)
        if self._trackers:
            values = {
                f"{event}/{key}": value
                for key, value in fields.items()
                if key != "step" and isinstance(value, int | float)
            }
            if values:
                self._accelerator.log(values, step=fields.get("step"))


def _duration(seconds: float) -> str:
    minutes, secs = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"


def _describe(record: dict[str, Any]) -> str:
    event = record["event"]
    if event == "train":
        parts = [
            f"step {record['step']}",
            f"loss {record['loss']:.4f}",
            f"lr {record['lr']:.2e}",
            f"{record['tokens_per_sec']:,.0f} tokens/s",
        ]
        if record["eta_s"] is not None:
            parts.append(f"ETA {_duration(record['eta_s'])}")
        if record["projected_cost"] is not None:
            parts.append(f"projected cost ${record['projected_cost']:.2f}")
        return " | ".join(parts)
    if event in ("baseline", "eval"):
        where = "baseline" if event == "baseline" else f"eval at step {record['step']}"
        return (
            f"{where}: exact match {record['exact_match']:.3f} | "
            f"mean macro-F1 {record['mean_macro_f1']:.3f} | "
            f"mean log-loss {record['mean_log_loss']:.4f}"
        )
    if event == "done":
        return f"adapter written to {record['adapter']}"
    return json.dumps(record)


def _headline(metrics: MetricsReport) -> dict[str, float]:
    return {
        "exact_match": metrics.exact_match,
        "mean_macro_f1": metrics.mean_macro_f1,
        "mean_log_loss": metrics.mean_log_loss,
    }


def _data_files(config: Config) -> list[Path]:
    files = [config.resolve_path(config.data.train)]
    if config.data.eval:
        files.append(config.resolve_path(config.data.eval))
    return files


def _check_memory(config: Config, report: ValidationReport, device: torch.device) -> None:
    spec = model_spec(config.model.id)
    if spec is None or report.tokens is None:
        return
    workload = workload_from(config, report.tokens)
    needed = memory_needed_gb(spec, workload, config.train.quantization)
    available = torch.cuda.get_device_properties(device).total_memory / 1e9
    if needed > available:
        raise TrainingError(
            f"{config.model.id} needs about {needed:.1f} GB of GPU memory with this config, "
            f"but {torch.cuda.get_device_name(device)} has {available:.1f} GB; {OOM_ADVICE}"
        )


def _check_bundle(config: Config, config_path: Path) -> None:
    """A bundle is a directory of config.yaml, train.jsonl and eval.jsonl, where the config reads
    exactly those two data files; anything else is not ours to delete."""
    bundle_dir = config.base_dir
    is_bundle = (
        config_path.name == BUNDLE_FILES[0]
        and all((bundle_dir / name).is_file() for name in BUNDLE_FILES)
        and config.data.eval is not None
        and config.resolve_path(config.data.train) == bundle_dir / BUNDLE_FILES[1]
        and config.resolve_path(config.data.eval) == bundle_dir / BUNDLE_FILES[2]
    )
    if not is_bundle:
        raise TrainingError(
            f"--delete-bundle-on-success needs a bundle ({', '.join(BUNDLE_FILES)}); "
            f"{bundle_dir} is not a bundle"
        )


def _delete_bundle(bundle_dir: Path) -> None:
    """Delete the bundle's data and config; the directory goes too if nothing else is in it (the
    default output dir lives inside the bundle)."""
    for name in BUNDLE_FILES:
        (bundle_dir / name).unlink(missing_ok=True)
    if (bundle_dir / BUNDLE_IMAGES).is_dir():
        shutil.rmtree(bundle_dir / BUNDLE_IMAGES)
    if not any(bundle_dir.iterdir()):
        bundle_dir.rmdir()


def _non_finite_loss(output_dir: Path, step: int) -> TrainingError:
    latest = latest_checkpoint(output_dir)
    where = (
        f"the last good checkpoint is step {latest.name.removeprefix('step-')} in {latest.parent}"
        if latest is not None
        else "no checkpoint was saved yet"
    )
    return TrainingError(f"loss became non-finite at step {step}; {where}")


def _eta(elapsed: float, steps_done: int, steps_left: int) -> float | None:
    """Seconds left at this run's pace, once it has run ETA_AFTER_STEPS steps."""
    if steps_done < ETA_AFTER_STEPS:
        return None
    return elapsed / steps_done * steps_left


def _projected_cost(train_seconds: float, eta_s: float | None) -> float | None:
    price = env_float(PRICE_ENV)
    if eta_s is None or price is None:
        return None
    return (train_seconds + eta_s) / 3600 * price


def _run_summary(
    config: Config, state: TrainState, stopped_early: bool, hours: float
) -> dict[str, Any]:
    price = env_float(PRICE_ENV)
    estimated_hours = env_float(ESTIMATE_ENV)
    return {
        "steps": state.step,
        "best_step": state.best_step,
        "stopped_early": stopped_early,
        "select_metric": config.eval.select_metric,
        "class_weighting": config.train.class_weighting,
        "hours": hours,
        "estimated_hours": estimated_hours,
        "price_per_hour": price,
        "cost": hours * price if price is not None else None,
        "estimated_cost": (
            estimated_hours * price if price is not None and estimated_hours is not None else None
        ),
    }


def _optimizer(model: Any, train: TrainSection) -> torch.optim.AdamW:
    head = [p for p in model.head.parameters() if p.requires_grad]
    head_ids = {id(p) for p in head}
    lora = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": lora, "lr": train.lora_lr}, {"params": head, "lr": train.head_lr}]
    return torch.optim.AdamW(
        [group for group in groups if group["params"]], weight_decay=train.weight_decay
    )


def _evaluate(
    accelerator: Accelerator,
    loaded: LoadedClef,
    model: Any,
    items: list[TrainItem],
    train: TrainSection,
    device: torch.device,
) -> tuple[EvalResult, list[dict[str, list[float]]]]:
    """Each rank scores its shard of the eval items; the gathered logits come back in item order."""
    shard = list(range(accelerator.process_index, len(items), accelerator.num_processes))
    logits = predict_logits(
        model,
        [items[i] for i in shard],
        loaded.clef_code,
        loaded.processor,
        max_length=train.max_length,
        max_tokens_per_batch=train.max_tokens_per_batch,
        device=device,
    )
    pairs = gather_object(list(zip(shard, logits, strict=True)))
    per_item = [row for _, row in sorted(pairs, key=lambda pair: pair[0])]
    return summarize_eval(items, per_item), per_item


def train(
    config_path: str | Path,
    *,
    restart: bool = False,
    push_to_hub: str | None = None,
    delete_bundle_on_success: bool = False,
    on_step: Callable[[int], None] | None = None,
    device: torch.device | None = None,
) -> TrainResult:
    try:
        return _train(
            config_path,
            restart=restart,
            push_to_hub=push_to_hub,
            delete_bundle_on_success=delete_bundle_on_success,
            on_step=on_step,
            device=device,
        )
    except torch.OutOfMemoryError as error:
        raise TrainingError(f"training ran out of memory: {error}; {OOM_ADVICE}") from error


def _on_main(accelerator: Accelerator, action: Callable[[], Any]) -> tuple[Any, str | None]:
    """Run a Hub action on the main process only, so N ranks don't make N calls (a 429 on one
    would stop them all); every rank learns the result or the HubError text."""
    outcome: list[Any] = [None, None]  # [result, error text]
    if accelerator.is_main_process:
        try:
            outcome[0] = action()
        except HubError as error:
            outcome[1] = str(error)
    broadcast_object_list(outcome)
    return outcome[0], outcome[1]


def _push(accelerator: Accelerator, adapter_dir: Path, repo: str) -> str | None:
    """Push from the main process; every rank learns the outcome, so a failure stops them all
    before any cleanup."""
    url, error = _on_main(accelerator, lambda: push_adapter(adapter_dir, repo))
    if error is not None:
        raise TrainingError(
            f"training finished and the adapter is in {adapter_dir}, "
            f"but pushing to {repo} failed: {error}"
        )
    return url


def _tokens_per_sec(accelerator: Accelerator, tokens: int, seconds: float) -> float:
    """Throughput across every process (a collective: call it on all ranks at the same step)."""
    total = accelerator.reduce(
        torch.tensor(float(tokens), device=accelerator.device), reduction="sum"
    )
    return total.item() / max(seconds, 1e-9)


def _train(
    config_path: str | Path,
    *,
    restart: bool,
    push_to_hub: str | None,
    delete_bundle_on_success: bool,
    on_step: Callable[[int], None] | None,
    device: torch.device | None,
) -> TrainResult:
    started = time.monotonic()
    config = load_config(config_path)
    if delete_bundle_on_success:
        _check_bundle(config, Path(config_path))
    report = validate(config, counter_for_config(config))
    if not report.ok:
        raise TrainingError(
            "validation failed:\n" + "\n".join(str(issue) for issue in report.errors)
        )
    repo = push_to_hub or config.output.push_to_hub
    train_cfg, eval_cfg = config.train, config.eval
    output_dir = config.output_dir()
    run_hash = config_hash(config, _data_files(config))

    device = torch.device(device) if device is not None else default_device()
    accelerator = Accelerator(
        # This torch may be a CUDA build; without cpu=True Accelerate moves a CPU run to the GPU.
        cpu=device.type == "cpu",
        mixed_precision="bf16" if device.type == "cuda" else "no",
        gradient_accumulation_steps=train_cfg.grad_accum_steps,
        log_with=config.output.report_to or None,
        project_dir=output_dir,
        # The model is loaded straight onto its device; 4-bit weights must not be moved again.
        device_placement=False,
        # The schedule counts this rank's optimizer steps: one scheduler step per optimizer step.
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=DIST_TIMEOUT)],
    )
    if accelerator.num_processes > 1:
        device = accelerator.device  # one GPU per DDP rank
    main = accelerator.is_main_process
    if repo:  # find out about missing write access now, not after the GPU hours
        _, error = _on_main(accelerator, lambda: ensure_repo(repo))
        if error is not None:
            raise TrainingError(error)

    if restart and main:
        clear_checkpoints(output_dir)
    accelerator.wait_for_everyone()
    resume_checkpoint = check_resume(output_dir, run_hash)
    if resume_checkpoint is not None:
        # Each rank's batches and the step count depend on the number of processes.
        saved = read_state(resume_checkpoint).world_size
        if saved != accelerator.num_processes:
            raise TrainingError(
                f"checkpoints in {resume_checkpoint.parent} were made with {saved} processes; "
                f"resume with {saved} processes, or run with --restart"
            )

    if device.type == "cuda":
        _check_memory(config, report, device)
        if not kernels_available():
            accelerator.print(
                "warning: flash-linear-attention and causal-conv1d are not installed; "
                "linear attention will use the slow PyTorch path"
            )

    set_seed(train_cfg.seed)
    loaded = load_for_training(model_dir(config), train_cfg, device=device)
    clef_code, processor = loaded.clef_code, loaded.processor
    train_items = prepare_items(clef_code, processor, report.train, train_cfg.max_length)
    eval_items = prepare_items(clef_code, processor, report.eval, train_cfg.max_length)
    weights = class_weights(report.train) if train_cfg.class_weighting == "balanced" else {}
    lengths = [item.length for item in train_items]

    def epoch_batches(epoch: int) -> list[list[int]]:
        batches = token_budget_batches(
            lengths, train_cfg.max_tokens_per_batch, train_cfg.seed, epoch
        )
        return shard_batches(batches, accelerator.process_index, accelerator.num_processes)

    accum = train_cfg.grad_accum_steps
    total_steps = train_cfg.epochs * math.ceil(len(epoch_batches(0)) / accum)
    optimizer = _optimizer(loaded.model, train_cfg)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, round(train_cfg.warmup_ratio * total_steps), total_steps
    )
    model, optimizer, scheduler = accelerator.prepare(loaded.model, optimizer, scheduler)
    unwrapped = accelerator.unwrap_model(model)
    trainable = [p for p in unwrapped.parameters() if p.requires_grad]

    state = TrainState(
        step=0,
        epoch=0,
        batch_in_epoch=0,
        best_metric=None,
        best_step=None,
        bad_evals=0,
        config_hash=run_hash,
        tokens_seen=0,
        train_seconds=0.0,
        world_size=accelerator.num_processes,
    )
    resumed_from: int | None = None
    best_lost = False
    if resume_checkpoint is not None:
        # Weights, optimizer, scheduler, RNG and data position; nothing below draws randomness
        # before the loop, so the run continues exactly where the checkpoint left off.
        state = load_checkpoint(resume_checkpoint, unwrapped, optimizer, scheduler)
        resumed_from = state.step
        accelerator.print(f"Resuming from step {resumed_from}")
        if state.best_step is not None and not best_complete(output_dir):
            # On a bucket mount a stop can cut the best weights' upload short: select again.
            accelerator.print(
                f"warning: the best checkpoint (step {state.best_step}) is incomplete; "
                "the best step is chosen again from the evaluations after this resume"
            )
            state.best_metric, state.best_step, best_lost = None, None, True

    if main:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    if config.output.report_to:
        accelerator.init_trackers(config.run_name)

    with _EventLog(
        output_dir / METRICS_LOG,
        accelerator,
        bool(config.output.report_to),
        append=resumed_from is not None,
    ) as log:

        def evaluate_and_select() -> None:
            result, per_item = _evaluate(
                accelerator, loaded, unwrapped, eval_items, train_cfg, device
            )
            value = result.metrics.value(eval_cfg.select_metric)
            if is_better(eval_cfg.select_metric, value, state.best_metric):
                state.best_metric, state.best_step, state.bad_evals = value, state.step, 0
                if main:
                    save_best(output_dir, unwrapped, per_item, result.targets, state)
                accelerator.wait_for_everyone()
            else:
                state.bad_evals += 1
            log("eval", step=state.step, **_headline(result.metrics))

        baseline = load_baseline(output_dir) if resumed_from is not None else None
        if baseline is None:
            result, _ = _evaluate(accelerator, loaded, unwrapped, eval_items, train_cfg, device)
            baseline = result.metrics.to_dict()
            if main:
                save_baseline(output_dir, baseline)
            log("baseline", step=state.step, **_headline(result.metrics))

        base_seconds = state.train_seconds
        loop_started = time.monotonic()
        start_step, start_epoch, skip = state.step, state.epoch, state.batch_in_epoch
        tokens_this_run = 0
        loss_sum, loss_count = 0.0, 0
        # A checkpoint is saved after its step's eval, so a run resumed from an eval step has
        # already evaluated it, and may already have run out of patience there. Without its
        # best weights it evaluates again, so there is a best to load at the end.
        resumed_at_eval = (
            resumed_from is not None and state.step % eval_cfg.eval_every == 0 and not best_lost
        )
        last_eval_step: int | None = state.step if resumed_at_eval else None
        patience = eval_cfg.early_stopping_patience
        stopped_early = patience is not None and state.bad_evals >= patience

        def log_train(now: float, steps_left: int) -> None:
            eta_s = _eta(now - loop_started, state.step - start_step, steps_left)
            log(
                "train",
                step=state.step,
                loss=loss_sum / loss_count,
                lr=scheduler.get_last_lr()[0],
                tokens_per_sec=_tokens_per_sec(accelerator, tokens_this_run, now - loop_started),
                eta_s=eta_s,
                projected_cost=_projected_cost(state.train_seconds, eta_s),
            )

        model.train()
        for epoch in range(start_epoch, train_cfg.epochs):
            if stopped_early:
                break
            batches = epoch_batches(epoch)
            first = skip if epoch == start_epoch else 0
            state.epoch, state.batch_in_epoch = epoch, first
            for position in range(first, len(batches)):
                # An optimizer step closes each group of `accum` micro-batches and the epoch's
                # last (possibly shorter) group: ceil(len(batches) / accum) steps per epoch.
                # accumulate() syncs when its counter reaches `accum`, so set the counter to
                # sync exactly at each group's end, independent of earlier epochs or a resume.
                group_start = position - position % accum
                group_size = min(accum, len(batches) - group_start)
                accelerator.step = accum - 1 if position == group_start + group_size - 1 else 0
                with accelerator.accumulate(model):
                    items = [train_items[i] for i in batches[position]]
                    batch = collate(items, clef_code, processor, train_cfg.max_length, device)
                    loss = batch_loss(
                        model(batch),
                        items,
                        class_weights=weights,
                        question_weights=train_cfg.question_weights,
                        label_smoothing=train_cfg.label_smoothing,
                    )
                    loss_value = loss.item()
                    if not math.isfinite(loss_value):
                        raise _non_finite_loss(output_dir, state.step + 1)
                    # Accelerate divides by `accum`; average over the group's micro-batches.
                    accelerator.backward(loss * (accum / group_size))
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(trainable, MAX_GRAD_NORM)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                padded_tokens = batch["input_ids"].numel()
                tokens_this_run += padded_tokens
                state.tokens_seen += padded_tokens
                state.batch_in_epoch = position + 1
                loss_sum, loss_count = loss_sum + loss_value, loss_count + 1
                if not accelerator.sync_gradients:
                    continue

                state.step += 1
                now = time.monotonic()
                state.train_seconds = base_seconds + now - started
                if state.step % LOG_EVERY == 0 or state.step == total_steps:
                    log_train(now, total_steps - state.step)
                    loss_sum, loss_count = 0.0, 0
                if state.step % eval_cfg.eval_every == 0:
                    evaluate_and_select()
                    last_eval_step = state.step
                    stopped_early = patience is not None and state.bad_evals >= patience
                    if stopped_early and loss_count:  # the run ends here: log its last step
                        log_train(now, 0)
                        loss_sum, loss_count = 0.0, 0
                if state.step % config.output.save_every == 0:
                    if main:
                        save_checkpoint(output_dir, unwrapped, optimizer, scheduler, state)
                    accelerator.wait_for_everyone()
                if on_step is not None:
                    on_step(state.step)
                if stopped_early:
                    break

        if last_eval_step != state.step:
            evaluate_and_select()
        per_item, _ = load_best(output_dir, unwrapped)
        best = summarize_eval(eval_items, per_item)
        calibration = calibrate(best.logits, best.targets) if eval_cfg.calibrate else {}
        for qid, item in calibration.items():
            if not item.fitted:
                accelerator.print(
                    f"warning: {qid} has {item.n} labeled eval rows (fewer than {MIN_ROWS}); "
                    "its calibration temperature stays 1"
                )

        hours = (base_seconds + time.monotonic() - started) / 3600
        metrics = {
            "baseline": baseline,
            "final": best.metrics.to_dict(),
            "calibration": {qid: asdict(item) for qid, item in calibration.items()},
            "data": {"train_rows": len(report.train), "eval_rows": len(report.eval)},
            "run": _run_summary(config, state, stopped_early, hours),
        }
        final_dir = output_dir / FINAL_DIR
        if main:
            if final_dir.exists():
                shutil.rmtree(final_dir)  # a head-only rerun must not keep an old LoRA adapter
            save_adapter(
                final_dir,
                unwrapped,
                loaded.head_config,
                config,
                calibration=temperatures(calibration),
                metrics=metrics,
            )
            log("done", adapter=str(final_dir))
        accelerator.wait_for_everyone()

    hub_url = _push(accelerator, final_dir, repo) if repo else None

    if main:
        clear_checkpoints(output_dir)
        if delete_bundle_on_success:
            _delete_bundle(config.base_dir)
    accelerator.end_training()  # finishes trackers and destroys the process group
    return TrainResult(final_dir, metrics, resumed_from, hub_url, main_process=main)
