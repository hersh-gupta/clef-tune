# clef-tune Roadmap

**Spec:** `docs/superpowers/specs/2026-10-26-clef-finetuning-harness-design.md` (revision 2)

The spec covers four subsystems that depend on each other in sequence. Each gets its own plan,
and each plan ships working, tested software on its own. A plan is written in full just before
it starts, so it can build on facts verified by the plans before it.

| # | Plan | Delivers | Depends on |
|---|---|---|---|
| 1 | **Core** (`2026-10-26-clef-tune-plan-1-core.md`) | Package scaffold, schema, labels, config, CSV/JSONL loading, prompt layout and token counting, validation report, metrics, calibration, estimates, bundle writer, `clef-tune validate` / `estimate`, example dataset, CI | — |
| 2 | **Training engine** (`2026-10-27-clef-tune-plan-2-training.md`) | `clef_code.py`, `model.py`, `train.py`, `adapter.py`, `predictor.py`, `hub.py`, `runtime.py`; `clef-tune train` / `eval` / `predict` / `export`; tiny-model integration tests; tokenization-parity test; Dockerfile; opt-in `gpu` test | 1 |
| 3 | **HF Jobs** | `launch/source.py`, `launch/hf_jobs.py`; `clef-tune jobs launch/list/status/logs/cancel/resume`; mocked-API tests; opt-in `hfjobs` test | 1, 2 |
| 4 | **Web UI** | `ui/` (theme, CSS, handlers, layout), Space mode, `deploy/space/`, Playwright screenshots, iterative refinement with the user until sign-off | 1, 2 (results view), 3 (Launch step) |

## Interfaces Plan 1 provides to later plans

These names are fixed by Plan 1. Later plans consume them as written.

- `clef_tune.schema`: `Question` (`id`, `type`, `instructions`, `criteria`, `options()`,
  `option_ids()`, `to_clef()`), `Schema` (`questions`, `from_dict`, `load`, `save`, `to_dict`),
  `SchemaError`, `option_key()`
- `clef_tune.labels`: `Label` (`hard`, `soft`, `distribution(option_ids)`, `top()`, `to_json()`),
  `parse_label(question, raw) -> (Label | None, warning | None)`, `LabelError`
- `clef_tune.config`: `Config` (`run_name`, `model`, `data`, `train`, `eval`, `output`,
  `get_schema()`, `resolve_path()`, `output_dir()`, `with_preset()`, `with_base_dir()`,
  `to_dict()`, `to_yaml()`, `save()`), `load_config(path)`, `config_hash(config, files)`,
  `file_digest(path)`, `PINNED_REVISIONS`, `PRESETS`
- `clef_tune.data`: `Record` (`id`, `state`, `questions`, `labels`, `images`, `source_row`,
  `to_json()`), `Issue`, `LoadResult`, `DataError`, `read_rows()`, `load_file(config, path)`,
  `split_records(records, eval_fraction, seed)`
- `clef_tune.prompt`: `prompt_pieces(state, questions, n_images)`, `Piece`, `render()`,
  `RecordTokenCounter`, `TokenCount`, `TokenCounter` protocol, `counter_for_config(config)`,
  `get_token_counter(model_id, revision)`, `image_token_count()`, `media_token_count()`
- `clef_tune.validation`: `validate(config, counter) -> ValidationReport` (`issues`, `errors`,
  `warnings`, `ok`, `train`, `eval`, `questions: dict[str, QuestionStats]`, `tokens: TokenStats`,
  `columns`)
- `clef_tune.metrics`: `Prediction`, `summarize(records) -> MetricsReport` (`questions`,
  `exact_match`, `mean_macro_f1`, `mean_log_loss`, `value(metric)`, `to_dict()`),
  `is_better(metric, new, best)`
- `clef_tune.calibration`: `softmax()`, `fit_temperature()`, `calibrate() -> dict[str,
  QuestionCalibration]`, `temperatures()`, `expected_calibration_error()`,
  `temper_probabilities()`
- `clef_tune.estimate`: `MODELS`, `GPUS`, `HF_FLAVORS`, `model_spec()`, `Workload`,
  `workload_from(config, tokens)`, `estimate_hardware()`, `estimate_hf_flavors()`, `recommend()`,
  `suggest_timeout()`, `fetch_hf_prices()`, `EstimateRow`
- `clef_tune.bundle`: `write_bundle(report, config, out_dir) -> Path`,
  `zip_bundle(bundle_dir, zip_path) -> Path`
- `clef_tune.cli`: Typer `app`; later plans add commands to it.

## Interfaces Plan 2 provides to later plans

- `clef_tune.runtime`: `BASE_IMAGE`, `KERNEL_PACKAGES`, `PRICE_ENV`, `ESTIMATE_ENV`, `kernels_available()`,
  `env_float()`
- `clef_tune.hub`: `ensure_repo()`, `push_adapter()`, `pull_adapter()`, `HubError`
- `clef_tune.train`: `train(config_path, *, restart, push_to_hub, delete_bundle_on_success, on_step,
  device) -> TrainResult(adapter_dir, metrics, resumed_from, hub_url)`, `TrainingError`; output
  layout `metrics.jsonl`, `checkpoints/`, `final/`; `metrics.jsonl` events `baseline`, `train`
  (`step`, `loss`, `lr`, `tokens_per_sec`, `eta_s`, `projected_cost`), `eval`, `done`
- `clef_tune.adapter`: `read_adapter() -> AdapterInfo`, `save_adapter()`, `export_merged()`; the
  `metrics.json` shape (`baseline`, `final`, `calibration`, `data`, `run`) the results view renders
- `clef_tune.predictor`: `Predictor` (`load`, `logits`, `probabilities`, `predict`, `systemone`),
  `predict_file()`
- `clef_tune.checkpoints`: `ConfigMismatchError`

## Verified constraints and spec amendments

Checked on 2026-10-26 while writing Plan 1. These override the spec where they differ.

1. **huggingface_hub < 2.** transformers 5.10.x (Cloudflare's tested line) requires
   `huggingface_hub>=1.5,<2.0`. The base dependency is `huggingface_hub>=1.33,<2`. Hub 1.33 has
   `run_job`, `run_uv_job`, `inspect_job`, `fetch_job_logs`, `cancel_job`, `list_jobs(labels=…)`,
   `wait_for_job`, `list_jobs_hardware`, `sync_job_volume`, `Volume` and `sync_bucket`, but **not
   `rerun_job`**. *Amends spec §8.3:* `clef-tune jobs resume JOB_ID` rebuilds the job from
   `inspect_job` (`docker_image`, `command`, `arguments`, `environment`, `flavor`, `labels`,
   `volumes`) and calls `run_job` again with a fresh `HF_TOKEN` secret. The training run then
   auto-resumes from its checkpoint.
2. **Kernel packages are not a pyproject extra.** `uv lock` resolves every extra, and
   `causal-conv1d` ships an sdist that needs torch at build time. *Amends spec §3/§12:* the
   Dockerfile and the HF Jobs command install `flash-linear-attention` and `causal-conv1d`
   explicitly (failure only warns). README instructions say the same.
3. **Pinned model revisions:** `Cloudflare/clef` → `a30162b99d68a481d8bd75d38df069a0cf83df40`,
   `Cloudflare/clef-flash` → `f0b7d3cd65296abfe459e6c228b618f5d8b26d2f`.
4. **Resolved stack** (`uv lock`, Python 3.12): torch 2.11.0, transformers 5.10.4, peft 0.21.2,
   accelerate 1.15.0, bitsandbytes 0.50.2, huggingface-hub 1.33.0, gradio 6.30.0, pydantic
   2.14.1, typer 0.27.3.
5. **HF Jobs prices:** `list_jobs_hardware()` returns `JobHardwareInfo(name, unit_cost_usd,
   unit_label="minute", …)`; $/hour = `unit_cost_usd × 60`.
6. **Tokenizer:** `tokenizers.Tokenizer.from_file(hf_hub_download(model, "tokenizer.json",
   revision))` counts `<|im_start|>` as one token with `add_special_tokens=False`. Plan 2's parity
   test confirms it matches `encode_record` exactly.
7. **Module layout refinements** (no behavior change): validation lives in `validation.py`
   (the spec says `data.validate()`). The prompt layout and token counting live in `prompt.py`.
   `labels.py`, `yamlio.py` and `bundle.py` are focused helper modules. CSV reading uses the
   standard `csv` module, so no cell is ever coerced to NaN. As a result, `pandas` leaves the base
   dependencies; Plan 4 gets it through gradio.
8. **Plan 2 CI note:** install CPU-only torch on Linux CI (PyTorch CPU index) to avoid
   downloading CUDA wheels for the tiny-model tests.
9. **torchvision is required.** `AutoProcessor` for Clef loads `Qwen3VLVideoProcessor`, which needs
   torchvision; the `train` extra adds `torchvision>=0.26,<0.27` (0.26 pairs with torch 2.11).
10. **CUDA torch on Windows** (user decision, 2026-10-26). PyPI ships only CPU torch for Windows, so
    `[tool.uv.sources]` sends `torch`/`torchvision` on `sys_platform == 'win32'` to the explicit
    `https://download.pytorch.org/whl/cu130` index. `uv pip install` from git ignores uv sources, so
    the README tells those users to pass `--torch-backend`.
11. **The joint head is larger than the spec says.** `JointSchemaHead(**joint_head_config.json)` has
    128,056,324 parameters (Clef) and 121,762,820 (Clef-Flash), not "~30–40M" (spec §2). The release
    head file is bf16. `estimate.ModelSpec.head_params` replaces the 40M constant.
12. **Head optimizer stays fp32 AdamW** (user decision, 2026-10-27). With the real head, the §7 memory
    model puts Clef-Flash QLoRA at ~14.5 GB for 1024-token batches.
13. **The opt-in `gpu` test targets GPUs with ≥16 GB** (user decision, 2026-10-27). *Amends spec
    §11.5*, which said it fits a 12 GB GPU.
14. **`hub.py` moves to Plan 2.** `train --push-to-hub` and `Predictor.load(repo)` need it; Plan 3
    consumes it.
15. **Base image:** `pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel` (Ubuntu 24.04, Python 3.12, CUDA
    13.0.2). PyPI's Linux torch 2.11 bundles CUDA 13.0, so kernel builds match. The Dockerfile and
    Plan 3's HF Jobs use it via `runtime.BASE_IMAGE`.
16. **Clef code with transformers 5.10.4 / peft 0.21.2** (verified by running it on a tiny random
    model in the release layout):
    - importing `joint_schema_model.py` needs `sys.modules` registration before `exec_module`
    - LoRA on the 12 language-model Linear kinds excludes the vision tower and `lm_head`
    - gradient checkpointing with `use_reentrant=False`, enabled before `get_peft_model`,
      backpropagates through `ClefModel`
    - `merge_and_unload` matches the adapter's logits, and the merged keys equal the release index keys
    - CPU training is deterministic
    - Plan 1's token counter matches `encode_record` on 33 varied records with 0 mismatches
17. **Run-time inputs and small interface additions:**
    - `CLEF_TUNE_PRICE_PER_HOUR` and `CLEF_TUNE_ESTIMATED_HOURS` (set by Plan 3's launcher) feed the
      projected cost and the card's estimated-vs-actual line.
    - `train_config.yaml` carries clef-tune provenance as header comments, so it still loads as a
      `Config`.
    - `data.load_file(..., for_prediction=True)` loads prediction files without label columns.
