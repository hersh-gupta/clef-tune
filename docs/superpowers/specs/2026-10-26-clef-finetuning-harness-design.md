# clef-tune: Fine-Tuning Harness for Clef — Design

**Date:** 2026-10-26
**Status:** Draft for review

## 1. Purpose

A general, reusable tool for fine-tuning Cloudflare's Clef decision models
([Cloudflare/clef](https://huggingface.co/Cloudflare/clef), 27B; and
[Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash), 9.4B) on a user's own
labeled decision data.

- **Input:** CSV or JSONL.
- **Output:** a lightweight adapter (LoRA + fine-tuned joint head), with optional export to a
  merged, release-format checkpoint servable by SGLang.
- **Frontends:** a CLI core plus a polished Gradio web UI (deployable to Hugging Face Spaces) for
  previewing data, building the schema/config, estimating cost, and launching training.
- **Infra:** infra-agnostic (any machine that can run `accelerate launch`), with first-class
  launching on **Hugging Face Jobs**.
- **Guiding principle:** keep it simple. YAGNI on everything not listed here.

### Success criteria

1. A user with a labeled CSV can go from upload → validated schema → launched HF Job → adapter in
   a Hub repo entirely from the UI, without editing YAML.
2. The same `config.yaml` trains identically via `clef-tune train` on any GPU box.
3. Every run reports baseline (untuned) vs. fine-tuned metrics on the eval set.
4. A merged export loads with Cloudflare's own `load_release_model()` and `sglang serve`.
5. Before spending money, the user sees an estimated run time and cost per hardware option.
6. The UI looks polished and is refined iteratively with the user until signed off (§9).

## 2. Background: how Clef works (constraints on the design)

- Clef is **not generative**. It is a Qwen3.x multimodal backbone plus a small **joint schema
  head** (~30–40M params; config in `joint_head_config.json`) that outputs **one logit per allowed
  option per question**. Softmax per question gives probabilities.
- Questions are typed: `noul` (true/false), `choice` (named options), `score` (ordered options,
  indexed from 0).
- The release ships `joint_schema_model.py`, which provides `encode_record`, `collate_records`,
  `JointSchemaHead`, `ClefModel`, `load_release_model`, and `systemone`. `ClefModel.forward`
  already supports PEFT-wrapped backbones (`get_base_model()`).
- The head reads the backbone's **output embedding weight** (lexical prior), so `lm_head` must stay
  unquantized.
- Clef and Clef-Flash share identical `joint_schema_model.py`; one harness supports both.
- Fine-tuning objective is therefore **per-question cross-entropy over option logits**, not
  token-level SFT.

**Decision:** the harness does **not** reimplement Clef's encoding or model. `clef_code.py` imports
`joint_schema_model.py` from the (pinned-revision) model snapshot and uses it directly.

## 3. Architecture

Python package `clef_tune`. Modules marked *(no torch)* must not import torch, transformers, or
peft — the UI, Spaces deployment, and `validate`/`estimate` depend only on these.

```
clef_tune/
  schema.py      # Question/Schema dataclasses, YAML load/save, validation          (no torch)
  data.py        # CSV/JSONL → normalized records; column mapping; auto-mapping;
                 # validation report; bundle writer                                  (no torch)
  config.py      # TrainConfig (pydantic) + YAML round-trip                          (no torch)
  metrics.py     # per-question & per-record metrics                                 (no torch)
  estimate.py    # time/cost/memory-fit estimates per hardware flavor                (no torch)
  hub.py         # push/pull bundles, adapters, checkpoints to/from Hub repos        (no torch)
  launch/
    hf_jobs.py   # submit / status / logs / cancel via huggingface_hub Jobs API      (no torch)
  clef_code.py   # locate snapshot (pinned revision) and import joint_schema_model.py
  model.py       # load backbone (bf16 or 4-bit), apply LoRA, load/unfreeze head
  train.py       # Accelerate training loop, eval, checkpoint/resume, live ETA, push
  adapter.py     # save/load adapter dir; merge → release-format checkpoint
  predict.py     # batch inference → JSONL/CSV with per-option probabilities
  cli.py         # Typer CLI
  ui/
    app.py       # Gradio layout (Blocks)
    handlers.py  # event handlers as plain functions (unit-testable)
    theme.py     # custom Gradio theme
    style.css    # custom CSS
app.py           # Hugging Face Spaces entry point → clef_tune.ui.app
```

### Dependency extras

- base: `pydantic`, `pyyaml`, `huggingface_hub`, `typer`, `tokenizers`, `pandas`
- `[ui]`: `gradio`
- `[train]`: `torch`, `transformers`, `peft`, `accelerate`, `bitsandbytes`, `safetensors`,
  `pillow`
- `[dev]`: `pytest`, `ruff`

Training deps are pinned near Cloudflare's tested versions (torch 2.11, transformers 5.10.x).
Python ≥3.11; development and Docker use 3.12.

### CLI

```
clef-tune validate  config.yaml                 # data + schema checks, token stats
clef-tune estimate  config.yaml [--price 2.5]   # time/cost/fit table
clef-tune train     config.yaml [--resume PATH|hub] [--push-to-hub REPO]
clef-tune eval      ADAPTER data.csv
clef-tune predict   ADAPTER data.csv -o preds.jsonl|preds.csv
clef-tune export    ADAPTER --merge -o merged/
clef-tune ui        [--share] [--port N]
clef-tune jobs launch config.yaml [--flavor h200] [--timeout 6h]
clef-tune jobs status|logs|cancel JOB_ID
```

`train` runs under plain `python` (single GPU) or `accelerate launch -m clef_tune train …`
(multi-GPU DDP/FSDP). No infra-specific code paths exist in training.

## 4. Data model

### 4.1 Schema

Mirrors Clef's native question format:

```yaml
questions:
  department:
    type: choice
    instructions: Which team should handle the message?
    criteria: {billing: "Payments or invoices", technical: "Bugs or outages"}
  urgency:
    type: score
    criteria: ["Can wait", "This week", "Today"]
  outage:
    type: noul
    instructions: Is a service down?
```

Rules: `choice` and `score` require non-empty `criteria`; `noul` criteria are optional
descriptions for `true`/`false`; `instructions` is optional (Clef falls back to the question ID).

### 4.2 Column mapping (CSV and flat JSONL)

```yaml
data:
  train: data/train.csv
  eval: data/eval.csv        # optional; else seeded random split by eval_fraction
  eval_fraction: 0.1
  id_column: ticket_id       # optional; else row index
  state_columns: [subject, body, tier]   # >1 → JSON object; exactly 1 → raw string value
  label_columns: {department: dept}      # unmapped questions default to same-named column
  image_columns: [receipt]               # optional; paths relative to data file, or URLs
```

### 4.3 Label parsing

| Type | Accepted values |
|---|---|
| `choice` | option ID; whitespace trimmed; case-only mismatch → warning + auto-fix |
| `score` | integer index `0..n-1`, or exact criteria text |
| `noul` | `true/false`, `yes/no`, `1/0`, `t/f` (case-insensitive) |
| empty | question masked for that row |
| soft (JSONL only) | `{"billing": 0.7, "technical": 0.3}` — must cover only valid options, sum to 1 ±1e-3 |

Rows with zero labels are dropped with a warning.

### 4.4 Native JSONL

Rows of `{"id", "state", "questions"?, "labels", "images"?}`. A row's `questions` overrides the
global schema, enabling mixed multi-task datasets. A JSONL row is treated as native if it has a
`labels` key; otherwise it is treated as a flat row and the column mapping applies.

### 4.5 Normalized record and bundle

Internal record: `{id, state, questions, labels, images}` (Clef record + `labels`).
A **bundle** is the portable training input: `train.jsonl`, `eval.jsonl`, `config.yaml` (schema
embedded, data paths rewritten to the bundle), and `images/` if present. Bundles are what the UI
downloads, what gets pushed to Hub for HF Jobs, and what any infra consumes.

### 4.6 Validation report

Computed by `data.validate()`, shared by CLI and UI. Token counts use the real Clef tokenizer
(downloaded alone; a few MB) and Clef's own `encode_record` text layout replicated without torch —
see §11 for the parity test. Image tokens are estimated from image dimensions with the Qwen
vision patch/merge formula read from `processor_config.json`.

- **Errors (block training):** unknown labels (with row numbers), duplicate IDs, a question with
  fewer than 2 observed classes in train, schema alone exceeding `max_length`, missing image
  files, invalid soft labels.
- **Warnings:** dropped rows, case-fixed labels, class imbalance (minority class < 5%), rows that
  will be truncated, eval set missing a class.
- **Stats:** row counts, per-question label distribution, missing-label counts, token length
  p50/p95/max.

### 4.7 Auto-mapping (UI)

On upload: a column whose non-empty values all parse as `noul` values (§4.3) → proposed `noul`;
otherwise a column with ≤ 50 distinct values and < 50% unique values → proposed `choice` with the
distinct values as options; everything else → proposed state column. The user confirms/edits.

**Out of scope for v1:** video inputs.

## 5. Training

### 5.1 Model setup

- Backbone loads via `Qwen3_5ForConditionalGeneration` in bf16, or 4-bit NF4 (`quantization:
  4bit`, QLoRA). `lm_head` and embeddings are never quantized.
- Vision tower always frozen.
- **`train_mode: lora`** (default): PEFT LoRA on all linear layers of the language model
  (excluding vision tower and `lm_head`) + full joint head trainable.
- **`train_mode: head_only`**: frozen backbone, head trainable.
- Head trained with fp32 master weights under bf16 autocast. Separate LRs: `lora_lr` (default
  1e-4), `head_lr` (default 5e-5).
- Gradient checkpointing on by default.

### 5.2 Loss

Per-question cross-entropy over that question's option logits (soft-target CE for soft labels),
skipping masked questions, averaged over labeled questions in the batch. Optional
`question_weights` (per question ID) and `label_smoothing`.

### 5.3 Loop

- Length-sorted, token-budget batches (`max_tokens_per_batch`) with shuffling of batch order; plus
  `grad_accum_steps`.
- AdamW, cosine schedule, 3% warmup, grad clip 1.0, seeded.
- `max_length` default 4096 (Clef supports up to 16384).
- Hugging Face Accelerate handles single-GPU / DDP / FSDP.

### 5.4 Evaluation

- **Baseline:** untuned model evaluated on eval set before training.
- Per question:
  - `choice`: accuracy, macro-F1, log-loss
  - `score`: accuracy, MAE of expected score
  - `noul`: accuracy, F1 (positive = true), log-loss
- Per record: exact match across all labeled questions.
- Runs every `eval_every` steps and at end. Best checkpoint chosen by `select_metric`; options
  `mean_macro_f1` (default: mean over questions of macro-F1, where `score` uses macro-F1 over
  levels and `noul` uses binary F1), `exact_match`, `mean_log_loss`. Optional
  `early_stopping_patience`.

### 5.5 Logging

Always stdout + `out/metrics.jsonl` (one JSON object per log event: `step`, `loss`, `lr`,
`tokens_per_sec`, `eta_s`, `projected_cost`, eval metrics). Optional `report_to: [tensorboard,
wandb]` via Accelerate trackers. UI views consume `metrics.jsonl`-format lines.

### 5.6 Checkpoints and resume

- Every `save_every` steps: `out/checkpoints/step-N/` with LoRA, head, optimizer, scheduler, RNG
  states, step, and data-order position. Keep last 2.
- With `push_to_hub`, the latest checkpoint is uploaded to the `checkpoints` branch of the adapter
  repo.
- `--resume PATH` or `--resume hub` restores exactly.

### 5.7 Error handling

- **OOM:** caught, re-raised with actionable advice (lower `max_tokens_per_batch` / `max_length`,
  enable 4-bit, larger flavor); last checkpoint retained.
- **NaN/inf loss:** stop, keep last good checkpoint, report step.
- **Pre-flight (before GPU spend):** validation passes, memory-fit check passes for the detected
  GPU, Hub write access verified when pushing.

## 6. Outputs

### 6.1 Adapter (directory or Hub model repo)

```
adapter/                     # PEFT LoRA weights + adapter_config.json (absent for head_only)
joint_head.safetensors
joint_head_config.json
schema.yaml
train_config.yaml            # includes base model id + pinned revision
metrics.json                 # baseline vs final, per question
README.md                    # auto card: base model, data stats, before/after table, est vs actual cost
```

### 6.2 Commands

- `predict`: per row, per question: argmax + all option probabilities, in SystemOne answer shape
  (reusing `systemone_answer` from Clef's code). JSONL or CSV (flattened columns).
- `eval`: metrics report on a labeled file (same metrics as §5.4).
- `export --merge`: loads backbone unquantized in bf16, merges LoRA, writes the **exact release
  layout** (sharded safetensors, index, config, new `joint_head.safetensors`,
  `joint_head_config.json`, Clef's `joint_schema_model.py`, tokenizer/processor files, LICENSE)
  so `load_release_model()` and `sglang serve` work unchanged.

## 7. Cost and time estimation (`estimate.py`)

- **Tokens:** exact, from validation: Σ(truncated lengths) × epochs for training, plus eval
  tokens × number of eval passes (including baseline).
- **Compute:**
  - `lora` ≈ 6 × backbone_params × train_tokens (forward + checkpoint recompute + backward through
    frozen weights)
  - `head_only` ≈ 2 × backbone_params × train_tokens
  - eval ≈ 2 × params × eval_tokens
- **Throughput:** peak bf16 dense FLOPs of the GPU × utilization (default 0.35) × GPU count ×
  multi-GPU efficiency (0.9 per doubling); × 0.7 if 4-bit.
- **Overhead:** fixed 10 min (Flash) / 20 min (Clef) for start, load, save, push.
- **Cost:** time × $/hr; HF Jobs prices from `list_jobs_hardware()` live (static fallback table);
  `--price` for other infra.
- **Memory fit:** weights (2 B/param bf16, ~0.55 B/param 4-bit) + LoRA params/grads/Adam states +
  head fp32 + activations ≈ `max_tokens_per_batch` × hidden × layers × constant (with
  checkpointing) + 15% headroom. Flags flavors that don't fit; marks bf16 vs QLoRA-only.
- **Presentation:** range of −30% / +50% around the point estimate, labeled "estimate". UI
  pre-selects the cheapest fitting flavor and auto-fills HF Jobs timeout = upper bound × 1.25,
  rounded up to 30 min.
- **Live correction:** after 20 optimizer steps, the trainer logs measured tokens/sec, ETA, and
  projected total cost; repeated at each log interval.
- GPU spec table (peak FLOPs, memory) lives in `estimate.py` for all HF Jobs flavors plus common
  rental GPUs (H100, H200, B200, A100, L40S, A10G, L4, RTX 4090/5090).

## 8. Hugging Face Jobs integration (`launch/hf_jobs.py`)

Launch sequence:

1. Validate locally; abort on errors.
2. Push the bundle to a private Hub dataset repo `<namespace>/clef-tune-data-<run-name>`.
3. `run_uv_job` with:
   - dependency `clef-tune[train]` (pinned to the running version)
   - `flavor` and `timeout` from estimate or user override
   - `volumes=[Volume(type="model", source=<base model>, mount_path="/model")]`
   - `secrets={"HF_TOKEN": token}`
   - the bundle dataset repo mounted read-only as a second Volume at `/bundle`
   - command `clef-tune train /bundle/config.yaml --push-to-hub <namespace>/<adapter-repo>`
   - `labels={"tool": "clef-tune", "run": <run-name>}`
4. Return job ID + URL; CLI/UI then poll `inspect_job`, stream `fetch_job_logs`, offer
   `cancel_job`.

Resilience: periodic checkpoint pushes (§5.6); a re-launch with `--resume hub` continues after a
timeout.

**Auth:** CLI uses the locally logged-in token. On Spaces, "Sign in with Hugging Face" via Gradio
OAuth (`hf_oauth: true`, `hf_oauth_scopes` set to the repo-write and Jobs scopes named in the
current Spaces OAuth docs, checked at implementation time). If no OAuth scope grants Jobs access,
the HF Jobs tab requires a pasted token; repo-only actions still use OAuth. Tokens are held in session state only, never persisted.

## 9. Web UI

### 9.1 Function

A five-step flow, each step a panel, navigable freely once reached:

1. **Data** — drag-and-drop CSV/JSONL (or zip with images); "load example" button; preview table
   appears immediately (paginated, column type chips, row/column counts).
2. **Schema** — auto-mapped proposal (§4.7); one card per question: type selector, instructions,
   option editor showing label counts per option; state-column picker; **prompt preview** showing
   a sample row rendered exactly as Clef sees it, with state / question / option spans
   highlighted.
3. **Validate** — summary stat tiles; errors and warnings list with row references; per-question
   label distribution bars; token-length histogram with the `max_length` cut-off marked.
   Re-runs automatically on schema/config edits.
4. **Configure** — model picker (Clef / Clef-Flash); presets *Quick test*, *Balanced*,
   *Thorough*, with an *Advanced* accordion for every config field; live estimate table (§7).
5. **Launch** — three tabs: *Download bundle* (zip incl. `config.yaml`), *HF Jobs* (sign-in,
   flavor, timeout, repo names, cost confirmation), *Train locally* (shown only if CUDA is
   detected). A **job monitor** card shows stage badge, progress bar, ETA, cost so far, loss and
   eval-metric charts, and a log tail.

The UI imports only no-torch modules. Event handlers live in `handlers.py` as plain functions.

### 9.2 Visual requirements

- Custom Gradio theme (`theme.py`) + CSS (`style.css`); not the stock look.
- One neutral palette with a single accent color; full light and dark mode support.
- Clean typography (Inter or similar via Google Fonts), consistent spacing scale, generous
  whitespace, rounded cards with subtle borders.
- Left vertical stepper showing each step's status (not started / done / has warnings / has
  errors).
- Charts follow one consistent style in both themes.
- Usable at laptop widths down to 1024px; degrades gracefully on narrow screens.
- Empty, loading, and error states are designed, not default.

### 9.3 Iterative refinement

UI polish is an explicit phase. After the functional, themed UI exists:

1. Capture screenshots (Playwright) of every step, in light and dark, at 1440px and 1024px, using
   the example dataset.
2. Present them to the user for feedback.
3. Revise and repeat until the user signs off.

## 10. Configuration reference (`config.yaml`)

```yaml
run_name: support-tickets-v1
model:
  id: Cloudflare/clef            # or Cloudflare/clef-flash
  revision: <pinned commit>      # default pinned in code at implementation time
  local_path: null               # e.g. /model when mounted on HF Jobs
data: { ... }                    # §4.2
schema: { questions: { ... } }   # §4.1 (or schema_file: schema.yaml)
train:
  mode: lora                     # lora | head_only
  quantization: none             # none | 4bit
  lora: { r: 16, alpha: 32, dropout: 0.05 }
  lora_lr: 1.0e-4
  head_lr: 5.0e-5
  epochs: 3
  max_length: 4096
  max_tokens_per_batch: 16384
  grad_accum_steps: 4
  warmup_ratio: 0.03
  weight_decay: 0.0
  label_smoothing: 0.0
  question_weights: {}
  gradient_checkpointing: true
  seed: 42
eval:
  eval_every: 200
  select_metric: mean_macro_f1
  early_stopping_patience: null
output:
  dir: out/support-tickets-v1
  save_every: 200
  push_to_hub: null              # e.g. me/clef-support-tickets
  report_to: []
```

Presets set `epochs`, `lora.r`, `max_length`, and `mode`:

- *Quick test*: head_only, 1 epoch, max_length 2048
- *Balanced*: lora r16, 3 epochs
- *Thorough*: lora r32, 5 epochs, early stopping

## 11. Testing

1. **Unit (no torch, CI):** schema, data loading, label parsing, auto-mapping, validation
   report, metrics vs hand-computed values, config round-trip, estimate math vs hand calculation,
   memory-fit, HF Jobs launcher against a mocked `HfApi` (asserting exact `run_uv_job` arguments
   and bundle push), UI handlers.
2. **Tokenization parity (CI):** the no-torch token counter matches `len(encode_record(...).input_ids)`
   from Clef's code on varied records.
3. **Tiny-model integration (torch CPU, CI, ~1 min):** a fixture builds a tiny random Qwen3.5
   backbone (2 layers, hidden 64, tiny vision config) + matching `JointSchemaHead`, saved in the
   exact release layout, with a pinned copy of `joint_schema_model.py`. Tests:
   - LoRA training overfits 16 rows (loss decreases, accuracy reaches 1.0)
   - `head_only` mode trains
   - checkpoint + resume equals an uninterrupted run
   - adapter save/load round-trip
   - merged export loaded with `load_release_model()` gives logits allclose to the adapter
   - `predict` output matches SystemOne answer shape
4. **UI smoke (CI):** Gradio app builds without launching.
5. **Opt-in real-model (pytest markers, manual):**
   - `gpu`: Clef-Flash QLoRA, 50 example rows, short sequences (fits a 12 GB GPU)
   - `hfjobs`: same, submitted to a cheap HF Jobs flavor (~$1)

## 12. Scaffolding and deployment

- `uv` project; `pyproject.toml` with extras; `clef-tune` console script (Typer).
- ruff + pytest; GitHub Actions workflow runs lint + test tiers 1–4.
- `examples/support_tickets.csv` (~200 synthetic rows; department / urgency / outage) +
  `examples/support_tickets.yaml`.
- `Dockerfile` (CUDA base, Python 3.12, `.[train]`).
- `app.py` + Space README front matter (`sdk: gradio`, `hf_oauth: true`) for Spaces deployment.
- `README.md` with three quickstarts: UI, CLI on any GPU box, HF Jobs.

## 13. Out of scope (v1)

Video inputs; full-parameter fine-tuning; vision-tower tuning; launchers other than HF Jobs (other
infra uses the CLI directly); hyperparameter search; serving (export targets SGLang instead).
