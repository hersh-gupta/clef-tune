# clef-tune: Fine-Tuning Harness for Clef — Design

**Date:** 2026-10-26
**Status:** Draft for review (revision 2)
**Repository:** https://github.com/hersh-gupta/clef-tune (public)

## 1. Purpose

A general, reusable tool for fine-tuning Cloudflare's Clef decision models
([Cloudflare/clef](https://huggingface.co/Cloudflare/clef), 27B; and
[Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash), 9.4B) on a user's own
labeled decision data.

- **Input:** CSV or JSONL.
- **Output:** a lightweight adapter (LoRA + fine-tuned joint head + per-question calibration), with
  optional export to a merged, release-format checkpoint servable by SGLang.
- **Frontends:** a CLI core plus a polished Gradio web UI, deployable as a public, multi-user
  Hugging Face Space, for previewing data, building the schema/config, estimating cost, launching
  training, and viewing results.
- **Infra:** infra-agnostic (any machine that can run `accelerate launch`), with first-class
  launching on **Hugging Face Jobs**.
- **Distribution:** installed from GitHub only (no PyPI).
- **Guiding principle:** keep it simple. YAGNI on everything not listed here.

### Success criteria

1. A user with a labeled CSV can go from upload → validated schema → launched HF Job → adapter in
   a Hub repo → results view entirely from the UI, without editing YAML.
2. The same `config.yaml` trains identically via `clef-tune train` on any GPU box.
3. Every run reports baseline (untuned) vs. fine-tuned metrics, confusion matrices, and
   calibration error on the eval set.
4. A merged export loads with Cloudflare's own `load_release_model()` and `sglang serve`.
5. Before spending money, the user sees an estimated run time and cost per hardware option.
6. The UI looks polished and is refined iteratively with the user until signed off (§9.4).

## 2. Background: how Clef works (constraints on the design)

- Clef is **not generative**. It is a Qwen3.x multimodal backbone plus a small **joint schema
  head** (~30–40M params; config in `joint_head_config.json`) that outputs **one logit per allowed
  option per question**. Softmax per question gives probabilities.
- Questions are typed: `noul` (true/false), `choice` (named options), `score` (ordered options,
  indexed from 0).
- The release ships `joint_schema_model.py`, which provides `encode_record`, `collate_records`,
  `JointSchemaHead`, `ClefModel`, `load_release_model`, `systemone_answer`, and `systemone`.
  `ClefModel.forward` already supports PEFT-wrapped backbones (`get_base_model()`).
- The head reads the backbone's **output embedding weight** (lexical prior), so `lm_head` must stay
  unquantized.
- Clef's backbone is **dense** (no mixture-of-experts): 64 layers, hidden size 5120, a hybrid of
  linear-attention and full-attention layers (3:1). The linear-attention layers need the
  `flash-linear-attention` and `causal-conv1d` kernels for practical training speed; without them
  transformers falls back to a slow pure-PyTorch path.
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
  schema.py        # Question/Schema dataclasses, YAML load/save, validation         (no torch)
  data.py          # CSV/JSONL → normalized records; column mapping; auto-mapping;
                   # stratified split; validation report; bundle writer               (no torch)
  config.py        # TrainConfig (pydantic) + YAML round-trip + config hash          (no torch)
  metrics.py       # per-question & per-record metrics, confusion matrices           (no torch)
  calibration.py   # per-question temperature fitting, ECE, apply temperatures       (no torch)
  estimate.py      # time/cost/memory-fit estimates per hardware flavor              (no torch)
  hub.py           # push/pull adapters to/from Hub model repos                      (no torch)
  launch/
    source.py      # resolve GitHub install source + exact commit of running code    (no torch)
    hf_jobs.py     # launch / list / status / logs / cancel / resume HF Jobs         (no torch)
  clef_code.py     # locate snapshot (pinned revision) and import joint_schema_model.py
  model.py         # load backbone (bf16 or 4-bit), apply LoRA, load/unfreeze head
  train.py         # Accelerate loop, eval, calibration, checkpoint/auto-resume, live ETA
  adapter.py       # save/load adapter dir; merge → release-format checkpoint
  predictor.py     # Predictor API + batch file inference
  cli.py           # Typer CLI
  ui/
    app.py         # Gradio layout (Blocks)
    handlers.py    # event handlers as plain functions (unit-testable)
    theme.py       # custom Gradio theme
    style.css      # custom CSS
app.py             # Hugging Face Spaces entry point → clef_tune.ui.app
```

### Dependency extras

- base: `pydantic`, `pyyaml`, `huggingface_hub`, `typer`, `tokenizers`, `pandas`, `numpy`
- `[ui]`: `gradio`
- `[train]`: `torch`, `transformers`, `peft`, `accelerate`, `bitsandbytes`, `safetensors`,
  `pillow`
- `[kernels]`: `flash-linear-attention`, `causal-conv1d` (Linux + CUDA; installed in Docker and
  HF Jobs; kept separate so `[train]` stays installable on Windows/macOS)
- `[dev]`: `pytest`, `ruff`, `playwright`

Training deps are pinned near Cloudflare's tested versions (torch 2.11, transformers 5.10.x).
Python ≥3.11; development and Docker use 3.12. Versioning via `hatch-vcs` (version and commit
derived from git).

### CLI

```
clef-tune validate  config.yaml                  # data + schema checks, token stats
clef-tune estimate  config.yaml [--price 2.5]    # time/cost/fit table
clef-tune train     config.yaml [--restart] [--push-to-hub REPO] [--delete-bundle-on-success]
clef-tune eval      ADAPTER data.csv
clef-tune predict   ADAPTER data.csv -o preds.jsonl|preds.csv [--no-calibration]
clef-tune export    ADAPTER --merge -o merged/
clef-tune ui        [--share] [--port N]
clef-tune jobs launch config.yaml [--flavor h200] [--timeout 6h] [--keep-data] [--source-ref REF]
clef-tune jobs list
clef-tune jobs status|logs|cancel|resume JOB_ID
```

`train` runs under plain `python` (single GPU) or `accelerate launch -m clef_tune train …`
(multi-GPU DDP). No infra-specific code paths exist in training.

### Python API

```python
from clef_tune import Predictor

p = Predictor.load("hersh-gupta/clef-support-tickets")  # Hub repo or local dir; device, quantization optional
p.predict([{"state": {...}}])        # questions default to the adapter's schema → SystemOne answers
p.systemone(request)                 # same contract as Clef's systemone(), calibrated
p.model, p.processor                 # raw ClefModel + processor (uncalibrated logits)
```

Calibration is applied by default; `calibrated=False` disables it.

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
  eval: data/eval.csv        # optional; else stratified seeded split by eval_fraction
  eval_fraction: 0.1
  id_column: ticket_id       # optional; else row index
  state_columns: [subject, body, tier]   # >1 → JSON object; exactly 1 → raw string value
  label_columns: {department: dept}      # unmapped questions default to same-named column
  image_columns: [receipt]               # optional; paths relative to data file, or URLs
```

**Split:** when no eval file is given, the split is stratified on the first schema question (in
schema order) that has labels; rows without that label are split randomly; classes with a single
row go to train. Seeded by `train.seed`.

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
`labels` key; otherwise it is treated as a flat row and the column mapping applies. Per-question
statistics (class weights, calibration, confusion matrices) are keyed by question ID.

### 4.5 Normalized record and bundle

Internal record: `{id, state, questions, labels, images}` (Clef record + `labels`).
A **bundle** is the portable training input: `train.jsonl`, `eval.jsonl`, `config.yaml` (schema
embedded, data paths rewritten to the bundle), and `images/` if present. Bundles are what the UI
downloads, what gets uploaded for HF Jobs, and what any infra consumes.

### 4.6 Validation report

Computed by `data.validate()`, shared by CLI and UI. Token counts use the real Clef tokenizer
(downloaded alone; a few MB) and Clef's own `encode_record` text layout replicated without torch —
see §11 for the parity test. Image tokens are estimated from image dimensions with the Qwen
vision patch/merge formula read from `processor_config.json`.

- **Errors (block training):** unknown labels (with row numbers), duplicate IDs, a question with
  fewer than 2 observed classes in train, schema alone exceeding `max_length`, missing image
  files, invalid soft labels.
- **Warnings:** dropped rows, case-fixed labels, class imbalance (minority class < 5%), rows that
  will be truncated, eval set missing a class, fewer than 20 labeled eval rows for a question
  (calibration will be skipped for it).
- **Stats:** row counts, per-question label distribution, missing-label counts, token length
  p50/p95/max.

### 4.7 Auto-mapping (UI)

On upload: a column whose non-empty values all parse as `noul` values (§4.3) → proposed `noul`;
otherwise a column with ≤ 50 distinct values and < 50% unique values → proposed `choice` with the
distinct values as options; everything else → proposed state column. The user confirms/edits.

## 5. Training

### 5.1 Model setup

- Backbone loads via `Qwen3_5ForConditionalGeneration` in bf16, or 4-bit NF4 (`quantization:
  4bit`, QLoRA). `lm_head` and embeddings are never quantized.
- Vision tower always frozen.
- **`mode: lora`** (default): PEFT LoRA on all linear layers of the language model (excluding
  vision tower and `lm_head`) + full joint head trainable.
- **`mode: head_only`**: frozen backbone, head trainable.
- Head trained with fp32 master weights under bf16 autocast. Separate LRs: `lora_lr` (default
  1e-4), `head_lr` (default 5e-5).
- Gradient checkpointing on by default.

### 5.2 Loss

- Per-question cross-entropy over that question's option logits (soft-target CE for soft labels),
  skipping masked questions.
- **Class weighting** (`class_weighting: none | balanced`, default `none`): with `balanced`, each
  option's weight is `N_q / (K_q × n_{q,o})` (labeled train rows for the question, options observed
  in train, count of that option), clipped to [0.1, 10]. A labeled question's loss is
  `w(target) × CE`; soft labels use the expected weight `Σ_o p_o w_o`. The batch loss is
  `Σ w_i CE_i / Σ w_i`.
- Optional `question_weights` (per question ID, multiplies the above) and `label_smoothing`.
- Note (surfaced in the adapter card and a UI tooltip): balanced weighting shifts predicted
  probabilities toward rare classes; temperature calibration rescales confidence but does not undo
  that shift.

### 5.3 Loop

- Length-sorted, token-budget batches (`max_tokens_per_batch`) with shuffling of batch order; plus
  `grad_accum_steps`.
- AdamW, cosine schedule, 3% warmup, grad clip 1.0, seeded.
- `max_length` default 4096 (Clef supports up to 16384).
- Hugging Face Accelerate handles single-GPU and multi-GPU DDP (each GPU holds a full model copy).

### 5.4 Evaluation

- **Baseline:** untuned model evaluated on eval set before training.
- Per question:
  - `choice`: accuracy, macro-F1, log-loss
  - `score`: accuracy, MAE of expected score
  - `noul`: accuracy, F1 (positive = true), log-loss
  - **confusion matrix** (rows = true, columns = argmax prediction, counts)
- Per record: exact match across all labeled questions.
- Runs every `eval_every` steps and at end. Best checkpoint chosen by `select_metric`; options
  `mean_macro_f1` (default: mean over questions of macro-F1, where `score` uses macro-F1 over
  levels and `noul` uses binary F1), `exact_match`, `mean_log_loss`. Optional
  `early_stopping_patience`.

### 5.5 Calibration (`calibration.py`, `calibrate: true` by default)

- After training, using the best checkpoint's cached eval logits (no extra forward passes), fit one
  temperature `T_q` per question ID by minimizing eval NLL (soft CE for soft labels) over
  `T ∈ [0.05, 20]` with golden-section search on `log T`.
- Questions with fewer than 20 labeled eval rows get `T = 1` and a warning.
- Report top-label **ECE** (15 equal-width bins) and log-loss before and after calibration, per
  question.
- Saved as `calibration.json` (`{question_id: T}`); applied as `softmax(logits / T_q)` by
  `Predictor` and `predict` unless disabled. Unknown question IDs at inference use `T = 1`.
- The adapter card notes that temperatures are fit on the same eval set used for checkpoint
  selection, so post-calibration ECE is slightly optimistic.

### 5.6 Logging

Always stdout + `out/metrics.jsonl` (one JSON object per log event: `step`, `loss`, `lr`,
`tokens_per_sec`, `eta_s`, `projected_cost`, eval metrics). Optional `report_to: [tensorboard,
wandb]` via Accelerate trackers. UI views consume `metrics.jsonl`-format lines.

### 5.7 Checkpoints and auto-resume

- Every `save_every` steps: `output.dir/checkpoints/step-N/` with LoRA, head, optimizer,
  scheduler, RNG states, step, data-order position, and the **config hash** (hash of the
  training-relevant config fields and bundle file digests). Keep last 2.
- `train` **auto-resumes** from the latest checkpoint in `output.dir` when its config hash matches,
  printing a clear "Resuming from step N" message. A mismatched hash is an error that names
  `--restart` (discard checkpoints) or a new `output.dir`.
- Checkpoints are deleted after a successful run once the final adapter is written (and pushed, if
  `push_to_hub`).

### 5.8 Error handling

- **OOM:** caught, re-raised with actionable advice (lower `max_tokens_per_batch` / `max_length`,
  enable 4-bit, larger flavor); last checkpoint retained.
- **NaN/inf loss:** stop, keep last good checkpoint, report step.
- **Pre-flight (before GPU spend):** validation passes; memory-fit check passes for the detected
  GPU; Hub write access verified when pushing; warning if `[kernels]` packages are missing on a
  CUDA machine (with the expected slowdown).

## 6. Outputs

### 6.1 Adapter (directory or Hub model repo)

```
adapter/                     # PEFT LoRA weights + adapter_config.json (absent for head_only)
joint_head.safetensors
joint_head_config.json
calibration.json             # per-question temperatures
schema.yaml
train_config.yaml            # includes base model id + pinned revision + clef-tune commit
metrics.json                 # baseline vs final per question, confusion matrices, ECE before/after
README.md                    # auto card (see below)
```

**Adapter card** (`README.md`): base model and revision, clef-tune commit, data stats, baseline
vs. fine-tuned metrics table, confusion matrices (markdown tables; questions with more than 10
options show the 10 most frequent classes plus "other"), calibration ECE before/after with the
caveat from §5.5, class-weighting note if used, and estimated vs. actual run time and cost.
Adapter repos are created **private** by default.

### 6.2 Commands

- `predict`: per row, per question: argmax + all option probabilities, in SystemOne answer shape
  (reusing `systemone_answer` from Clef's code), calibrated by default. JSONL or CSV (flattened
  columns).
- `eval`: metrics report on a labeled file (same metrics as §5.4, including confusion matrices and
  ECE).
- `export --merge`: loads backbone unquantized in bf16, merges LoRA, writes the **exact release
  layout** (sharded safetensors, index, config, new `joint_head.safetensors`,
  `joint_head_config.json`, Clef's `joint_schema_model.py`, tokenizer/processor files, LICENSE)
  so `load_release_model()` and `sglang serve` work unchanged. It also writes `calibration.json`
  and a README section explaining that SGLang serves **uncalibrated** probabilities and how a
  client applies calibration to returned probabilities:
  `p_i^(1/T) / Σ_j p_j^(1/T)`.

## 7. Cost and time estimation (`estimate.py`)

- **Tokens:** exact, from validation: Σ(truncated lengths) × epochs for training, plus eval
  tokens × number of eval passes (including baseline).
- **Compute:**
  - `lora` ≈ 6 × backbone_params × train_tokens (forward + checkpoint recompute + backward through
    frozen weights)
  - `head_only` ≈ 2 × backbone_params × train_tokens
  - eval ≈ 2 × params × eval_tokens
- **Throughput:** peak bf16 dense FLOPs of the GPU × utilization (default 0.35) × GPU count ×
  DDP efficiency (0.9 per doubling); × 0.7 if 4-bit.
- **Overhead:** fixed 10 min (Flash) / 20 min (Clef) for start, install, load, save, push.
- **Cost:** time × $/hr; HF Jobs prices from `list_jobs_hardware()` live (static fallback table);
  `--price` for other infra.
- **Memory fit (per GPU; DDP replicates the model):** weights (2 B/param bf16, ~0.55 B/param 4-bit)
  + LoRA params/grads/Adam states + head fp32 + activations ≈ `max_tokens_per_batch` × hidden ×
  layers × constant (with checkpointing) + 15% headroom. Flags flavors that don't fit; marks bf16
  vs QLoRA-only.
- **Presentation:** range of −30% / +50% around the point estimate, labeled "estimate". UI
  pre-selects the cheapest fitting flavor and auto-fills HF Jobs timeout = upper bound × 1.25,
  rounded up to 30 min.
- **Live correction:** after 20 optimizer steps, the trainer logs measured tokens/sec, ETA, and
  projected total cost; repeated at each log interval.
- GPU spec table (peak FLOPs, memory) lives in `estimate.py` for all HF Jobs flavors plus common
  rental GPUs (H100, H200, B200, A100, L40S, A10G, L4, RTX 4090/5090).

## 8. Hugging Face Jobs integration

### 8.1 Install source (`launch/source.py`)

Jobs install clef-tune from GitHub at the **exact commit** the launcher is running:

- Source repo: `https://github.com/hersh-gupta/clef-tune`, overridable with the
  `CLEF_TUNE_SOURCE` environment variable (for forks).
- Commit resolution, first match wins:
  1. `--source-ref REF` (explicit override).
  2. PEP 610 `direct_url.json` `vcs_info.commit_id` (package installed from git — the Space case).
  3. The commit embedded by `hatch-vcs` (editable/dev install). A dirty working tree is refused
     with a message to commit + push or pass `--source-ref`.
- Before submitting, the commit's existence on GitHub is checked via the GitHub REST API
  (unauthenticated); a 404 aborts with "push your commits first"; a network failure only warns.

### 8.2 Launch sequence (`launch/hf_jobs.py`)

1. Validate locally; abort on errors.
2. Write a staging directory containing `bundle/` and an empty `out/`, and upload it with
   `sync_job_volume(staging_dir, "/work", read_only=False)` — a folder in the user's private
   `jobs-artifacts` storage bucket, mounted read-write.
3. `run_job` with:
   - `image`: a pinned PyTorch CUDA *devel* image matching the pinned torch version (so kernel
     packages install from wheels or compile)
   - `command`: install `clef-tune[train] @ git+<source>@<commit>`, then attempt
     `clef-tune[kernels]` (failure only warns; training proceeds slower), then
     `clef-tune train /work/bundle/config.yaml --push-to-hub <adapter-repo>` with
     `output.dir=/work/out`, plus `--delete-bundle-on-success` unless `--keep-data` was given.
     (Local runs never delete input data; the flag is off by default.)
   - `flavor` and `timeout` from the estimate or user override
   - `volumes`: the `/work` volume, plus `Volume(type="model", source=<base model>,
     mount_path="/model")`. If `Volume` cannot pin a revision, the job verifies the mounted
     snapshot's commit and falls back to `snapshot_download` at the pinned revision on mismatch.
   - `secrets={"HF_TOKEN": token}`
   - `labels={"tool": "clef-tune", "run": <run-name>}`
4. Return job ID + URL. CLI/UI then poll `inspect_job`, stream `fetch_job_logs`, and offer
   `cancel_job`.

### 8.3 After the run

- **Success:** the job pushes the adapter to a private model repo
  (`<namespace>/clef-tune-<run-name>` unless named), deletes `/work/out/checkpoints`, and deletes
  `/work/bundle` when run with `--delete-bundle-on-success`. Deletion happens inside the job via the mounted
  read-write volume, so it works even if nobody is watching.
- **Failure or timeout:** bundle and checkpoints are kept. `clef-tune jobs resume JOB_ID` calls
  `rerun_job`; the rerun auto-resumes from the latest checkpoint (§5.7).
- `clef-tune jobs list` shows the user's jobs filtered by `labels={"tool": "clef-tune"}`.

### 8.4 Auth

CLI uses the locally logged-in token. On Spaces, "Sign in with Hugging Face" via Gradio OAuth
(`hf_oauth: true`, `hf_oauth_scopes` set to the repo-write, storage-bucket, and Jobs scopes named
in the current Spaces OAuth docs, checked at implementation time). If OAuth cannot grant Jobs or
bucket access, the HF Jobs tab requires a pasted token; repo-only actions still use OAuth. Tokens
are held in session state only, never persisted, and the app never uses a server-side token for
user actions.

## 9. Web UI

### 9.1 Layout

- **Header:** product mark and name, model picker (Clef / Clef-Flash), light/dark toggle, sign-in
  button.
- **Left sidebar:** vertical stepper — 1 Data, 2 Schema, 3 Validate, 4 Configure, 5 Launch — with a
  status icon per step (not started / current / done / warnings / errors), and below it a compact
  live estimate (cost range and time range) once data is loaded.
- **Main panel:** the active step. Steps are navigable freely once reached; a "Next →" button
  advances.

### 9.2 Steps

1. **Data** — drag-and-drop CSV/JSONL (or zip with images); "load example" button; preview table
   appears immediately (paginated, column type chips, row/column counts).
2. **Schema** — auto-mapped proposal (§4.7); one card per question: type selector, instructions,
   option editor showing label counts per option; state-column picker; **prompt preview** showing
   a sample row rendered exactly as Clef sees it, with state / question / option spans
   highlighted.
3. **Validate** — summary stat tiles; errors and warnings list with row references; per-question
   label distribution bars; token-length histogram with the `max_length` cut-off marked.
   Re-runs automatically on schema/config edits.
4. **Configure** — presets *Quick test*, *Balanced*, *Thorough*, with an *Advanced* accordion for
   every config field (including class weighting and calibration); live estimate table (§7).
5. **Launch** — tabs: *Download bundle* (zip incl. `config.yaml`), *HF Jobs* (sign-in, flavor,
   timeout, adapter repo name, "Delete training data after a successful run" checkbox (default
   on), cost confirmation), *Train locally* (see §9.3). Below the tabs:
   - **Your runs:** the signed-in user's clef-tune jobs (`jobs list`), so a run can be reattached
     after the tab was closed.
   - **Job monitor:** stage badge, progress bar, ETA, cost so far, loss and eval-metric charts, log
     tail, cancel and resume buttons.
   - **Results** (when a run completes, loaded from the adapter repo's `metrics.json`): baseline
     vs. fine-tuned metrics table, per-question confusion-matrix heatmaps, calibration ECE
     before/after, link to the adapter repo, and a copyable `Predictor` snippet.

The UI imports only no-torch modules. Event handlers live in `handlers.py` as plain functions.

### 9.3 Public Space mode

Detected via the `SPACE_ID` environment variable:

- *Train locally* is hidden even if the Space has a GPU, so visitors can't use the owner's
  hardware. Outside Spaces it is shown only when CUDA is detected.
- Each session gets its own temp directory, deleted when the session ends.
- Upload size is capped (`CLEF_TUNE_MAX_UPLOAD_MB`, default 200).
- Server-side tokens (e.g. a Space `HF_TOKEN` secret) are never used for user actions.

### 9.4 Visual requirements

- Custom Gradio theme (`theme.py`) + CSS (`style.css`); not the stock look.
- One neutral palette with a single accent color; full light and dark mode support.
- Clean typography (Inter or similar via Google Fonts), consistent spacing scale, generous
  whitespace, rounded cards with subtle borders.
- Charts (distributions, histogram, training curves, confusion-matrix heatmaps) follow one
  consistent style in both themes.
- Usable at laptop widths down to 1024px; degrades gracefully on narrow screens.
- Empty, loading, and error states are designed, not default.

### 9.5 Iterative refinement

UI polish is an explicit phase. After the functional, themed UI exists:

1. Capture screenshots (Playwright) of every step, plus the results view, in light and dark, at
   1440px and 1024px, using the example dataset and a recorded example run.
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
  class_weighting: none          # none | balanced
  question_weights: {}
  gradient_checkpointing: true
  seed: 42
eval:
  eval_every: 200
  select_metric: mean_macro_f1   # mean_macro_f1 | exact_match | mean_log_loss
  early_stopping_patience: null
  calibrate: true
output:
  dir: out/support-tickets-v1
  save_every: 200
  push_to_hub: null              # e.g. hersh-gupta/clef-support-tickets
  report_to: []
```

Presets set `mode`, `epochs`, `lora.r`, `max_length`, and early stopping:

- *Quick test*: head_only, 1 epoch, max_length 2048
- *Balanced*: lora r16, 3 epochs
- *Thorough*: lora r32, 5 epochs, early stopping (patience 3 evals)

## 11. Testing

1. **Unit (no torch, CI):**
   - schema, data loading, label parsing, auto-mapping, stratified split, validation report
   - metrics and confusion matrices vs hand-computed values
   - class-weight computation (incl. clipping and soft labels)
   - calibration: temperature fitting recovers a known temperature on synthetic logits; ECE vs a
     hand-computed example; small-eval fallback to `T = 1`
   - config round-trip and config-hash stability
   - estimate math vs hand calculation; memory fit
   - source resolution: PEP 610 commit, hatch-vcs commit, dirty-tree refusal, `--source-ref`
     override, GitHub 404 handling (mocked HTTP)
   - HF Jobs launcher against a mocked `HfApi`: exact `sync_job_volume` / `run_job` arguments,
     `--keep-data` propagation, `jobs list` label filter, `resume` → `rerun_job`
   - UI handlers, including Space-mode behavior (local training hidden, upload cap, server token
     ignored)
2. **Tokenization parity (CI):** the no-torch token counter matches `len(encode_record(...).input_ids)`
   from Clef's code on varied records.
3. **Tiny-model integration (torch CPU, CI, ~1–2 min):** a fixture builds a tiny random Qwen3.5
   backbone (2 layers, hidden 64, tiny vision config) + matching `JointSchemaHead`, saved in the
   exact release layout, with a pinned copy of `joint_schema_model.py`. Tests:
   - LoRA training overfits 16 rows (loss decreases, accuracy reaches 1.0)
   - `head_only` mode trains; `class_weighting: balanced` trains
   - interrupted run + auto-resume equals an uninterrupted run; config-hash mismatch errors
   - success path deletes checkpoints (and bundle unless keep-data)
   - adapter save/load round-trip, including `calibration.json`
   - merged export loaded with `load_release_model()` gives logits allclose to the adapter
   - `Predictor` output matches SystemOne answer shape; calibrated vs uncalibrated probabilities
     differ exactly by the stored temperatures
4. **UI smoke (CI):** Gradio app builds without launching, in normal and Space mode.
5. **Opt-in real-model (pytest markers, manual):**
   - `gpu`: Clef-Flash QLoRA, 50 example rows, short sequences (fits a 12 GB GPU)
   - `hfjobs`: same, submitted to a cheap HF Jobs flavor (~$1), including resume after a forced
     timeout

## 12. Scaffolding, distribution, and deployment

- `uv` project; `pyproject.toml` with extras and `hatch-vcs`; `clef-tune` console script (Typer);
  `Project-URL` pointing to the GitHub repo.
- Install (README): `uv pip install "clef-tune[ui] @ git+https://github.com/hersh-gupta/clef-tune"`
  (add `train`, `kernels` extras on GPU machines).
- ruff + pytest; GitHub Actions workflow on the GitHub repo runs lint + test tiers 1–4.
- `examples/support_tickets.csv` (~200 synthetic rows; department / urgency / outage) +
  `examples/support_tickets.yaml`.
- `Dockerfile` (same pinned PyTorch CUDA devel base as HF Jobs, Python 3.12, `.[train,kernels]`).
- `deploy/space/`: `app.py`, Space `README.md` front matter (`sdk: gradio`, `hf_oauth: true`,
  `hf_oauth_scopes`), and `requirements.txt` pinning `clef-tune[ui] @ git+…@<commit>`; README
  documents deploying it with `hf upload`.
- `README.md` with three quickstarts: UI, CLI on any GPU box, HF Jobs.

## 13. Out of scope (v1)

Video inputs; full-parameter fine-tuning; vision-tower tuning; FSDP / model sharding across GPUs;
launchers other than HF Jobs (other infra uses the CLI directly); PyPI publishing;
hyperparameter search; serving (export targets SGLang instead).
