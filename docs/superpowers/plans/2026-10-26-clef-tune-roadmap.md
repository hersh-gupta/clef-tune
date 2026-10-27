# clef-tune Roadmap

**Spec:** `docs/superpowers/specs/2026-10-26-clef-finetuning-harness-design.md` (revision 2)

The spec covers four subsystems that depend on each other in sequence. Each gets its own plan,
and each plan ships working, tested software on its own. A plan is written in full just before
it starts, so it can build on facts verified by the plans before it.

| # | Plan | Delivers | Depends on |
|---|---|---|---|
| 1 | **Core** (`2026-10-26-clef-tune-plan-1-core.md`) | Package scaffold, schema, labels, config, CSV/JSONL loading, prompt layout and token counting, validation report, metrics, calibration, estimates, bundle writer, `clef-tune validate` / `estimate`, example dataset, CI | — |
| 2 | **Training engine** | `clef_code.py`, `model.py`, `train.py`, `adapter.py`, `predictor.py`; `clef-tune train` / `eval` / `predict` / `export`; tiny-model integration tests; tokenization-parity test; Dockerfile; opt-in `gpu` test | 1 |
| 3 | **HF Jobs** | `hub.py`, `launch/source.py`, `launch/hf_jobs.py`; `clef-tune jobs launch/list/status/logs/cancel/resume`; mocked-API tests; opt-in `hfjobs` test | 1, 2 |
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
