# clef-tune Plan 2: Training Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune Clef and Clef-Flash on a validated config: LoRA or head-only training with checkpoints and auto-resume, evaluation and calibration, a portable adapter, a `Predictor` API, merged release-format export, Hub push/pull, and the `train` / `eval` / `predict` / `export` commands.

**Architecture:** Torch-free helpers (class weights, batching, adapter card, Hub I/O, runtime constants) sit beside torch modules that wrap Cloudflare's own `joint_schema_model.py`, imported from the pinned model snapshot: `clef_code` → `model` → `encoding`/`loss`/`evaluation` → `checkpoints` → `train` → `adapter`/`predictor`. Tests run on a tiny random Qwen3.5 model saved in the exact release layout, so the CPU suite exercises the same code paths as a real run.

**Tech Stack:** torch 2.11, torchvision 0.26, transformers 5.10.x (`Qwen3_5ForConditionalGeneration`), peft 0.21, accelerate 1.15, bitsandbytes 0.50, safetensors; everything from Plan 1.

**Spec:** `docs/superpowers/specs/2026-10-26-clef-finetuning-harness-design.md` (revision 2). Read `docs/superpowers/plans/2026-10-26-clef-tune-roadmap.md` too: its "Verified constraints and spec amendments" (items 1–17) override the spec where they differ. Items 9–17 were verified on 2026-10-27 for this plan by running prototypes against the pinned Clef code.

## Global Constraints

- Every Plan 1 global constraint still holds: Python `>=3.11` (dev 3.12), ruff line length 100, `uv run ruff format . && uv run ruff check --fix .` before every commit, spreadsheet row numbers for CSV and line numbers for JSONL.
- Every commit message ends with:

  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01BydXoAAPdrHBxWFt1mckNw
  ```

- `train` extra: `torch>=2.11,<2.12`, `torchvision>=0.26,<0.27`, `transformers>=5.10.2,<5.11`, `peft>=0.21`, `accelerate>=1.10`, `bitsandbytes>=0.45`, `safetensors>=0.5`. On Windows, torch and torchvision come from `https://download.pytorch.org/whl/cu130` (user decision, roadmap item 10).
- Never reimplement Clef: encoding, collation, the model, the head and SystemOne answers come from the imported `joint_schema_model` module (`encode_record`, `collate_records`, `ClefModel`, `JointSchemaHead`, `systemone_answer`, `load_release_model`).
- These modules must not import torch, transformers or peft at import time: everything Plan 1 lists, plus `runtime`, `clef_code`, `weights`, `batching`, `card`, `hub`, `adapter` (torch only inside functions) and `cli` (training commands import lazily).
- Training numerics (spec §5): AdamW with separate `lora_lr` / `head_lr`, `weight_decay` from config, cosine schedule with `warmup_ratio` warmup, gradient clipping at 1.0, seeded. Head parameters are fp32 master weights. On CUDA the forward pass runs under bf16 autocast with the backbone in bf16. On CPU everything is fp32.
- `lm_head` and embeddings are never quantized. The vision tower is always frozen. LoRA targets exactly `LORA_TARGET_MODULES` (Task 5).
- Class weights clip to `[0.1, 10]`. Calibration needs 20 labeled eval rows per question. Keep the last 2 step checkpoints. Adapter repos are private.
- The opt-in GPU test targets cards with ≥16 GB (user decision 2026-10-27; roadmap item 13).

## Review Focus

1. **Records with images** (local files or URLs) must train and predict. Media is encoded per batch and never held for the whole dataset. Pinned by `test_image_records_encode_media_lazily` (Task 7) and `test_image_rows_predict` (Task 14).
2. **Resuming mid-epoch with gradient accumulation** must reproduce the uninterrupted run exactly (data order, RNG, optimizer, scheduler). Pinned by `test_interrupted_run_resumes_to_the_same_result` (Task 12; grad accumulation 2, checkpoint mid-epoch).
3. **A failed Hub push after training** must leave the adapter and the checkpoints on disk and say what happened. Nothing may be deleted before the push succeeds. Pinned by `test_push_failure_keeps_adapter_and_checkpoints` (Task 13).
4. **Prediction files without label columns** (the normal case for `predict`) must load every row, with no "label column not found" errors and no dropped rows. Pinned by `test_prediction_input_needs_no_label_columns` (Task 14).
5. **Multi-task JSONL with per-row questions**, including questions with no labeled eval rows, must evaluate and calibrate per question ID without crashing. Pinned by `test_summarize_eval_handles_per_row_and_unlabeled_questions` (Task 8).

---

## File Structure

```
pyproject.toml                       # Task 1: torchvision, cu130 Windows source; Task 4: ruff exclude
.github/workflows/ci.yml             # Task 1: torch job with CPU torch
Dockerfile, .dockerignore            # Task 3
clef_tune/runtime.py                 # Task 3: BASE_IMAGE, kernels check, env-provided price/estimate (no torch)
clef_tune/estimate.py                # Task 2: per-model head size
clef_tune/clef_code.py               # Task 4: locate the model snapshot, import joint_schema_model.py (no torch)
clef_tune/model.py                   # Task 5: backbone/head loading, LoRA, 4-bit, trainable state
clef_tune/weights.py                 # Task 6: balanced class weights (no torch)
clef_tune/batching.py                # Task 6: token-budget batches, DDP sharding (no torch)
clef_tune/encoding.py                # Task 7: TrainItem, lazy media encoding, collation
clef_tune/loss.py                    # Task 7: weighted soft-target cross-entropy
clef_tune/evaluation.py              # Task 8: eval logits, metrics with optional temperatures
clef_tune/card.py                    # Task 9: adapter README card (no torch)
clef_tune/adapter.py                 # Task 9: save/read adapter; Task 15: export_merged
clef_tune/checkpoints.py             # Task 10: step checkpoints, best state, resume check
clef_tune/train.py                   # Tasks 11-13: the training run
clef_tune/hub.py                     # Task 13: private repo, push, pull (no torch)
clef_tune/predictor.py               # Task 14: Predictor API, predict_file
clef_tune/data.py                    # Task 14: load_file(..., for_prediction=True)
clef_tune/cli.py                     # Task 16: train / eval / predict / export
README.md                            # Task 16
tests/conftest.py                    # Task 1: skip tests/torch when the training stack is missing
tests/torch/                         # Tasks 1, 4-17: torch tests (module-level `pytestmark = pytest.mark.torch`)
tests/torch/fixtures/                # Task 4: vendored Clef files (pinned revision, Apache-2.0)
tests/torch/tiny_clef.py             # Task 4: tiny release-layout builder and tiny project writer
```

---

### Task 1: Training stack, torch test harness and CI job

**Files:**
- Modify: `pyproject.toml`, `uv.lock`, `.github/workflows/ci.yml`
- Create: `tests/conftest.py`, `tests/torch/__init__.py`, `tests/torch/test_stack.py`

**Interfaces:**
- Consumes: nothing.
- Produces: the `train` extra with torchvision; `uv sync --extra dev --extra train` installs `torch 2.11.0+cu130` on Windows; `tests/torch/` is collected only when torch, torchvision, transformers and peft import; a CI job `torch` runs `tests/torch` on CPU torch.

- [ ] **Step 1: Write the failing test** `tests/torch/test_stack.py` (and an empty `tests/torch/__init__.py`):

```python
import sys

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch


def test_training_stack_versions():
    import peft
    import torchvision
    import transformers
    from transformers import Qwen3_5ForConditionalGeneration  # noqa: F401

    assert torch.__version__.startswith("2.11.")
    assert torchvision.__version__.startswith("0.26.")
    assert transformers.__version__.startswith("5.10.")
    assert int(peft.__version__.split(".")[1]) >= 21


@pytest.mark.skipif(sys.platform != "win32", reason="the cu130 source applies to Windows only")
def test_windows_gets_the_cuda_build():
    assert torch.__version__.endswith("+cu130")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --no-sync pytest tests/torch/test_stack.py -v`
Expected: on Windows, `test_windows_gets_the_cuda_build` FAILS (`2.11.0+cpu`). If the venv lacks torchvision (a fresh `uv sync` removes it), `test_training_stack_versions` fails with `ModuleNotFoundError: torchvision`.

- [ ] **Step 3: Update `pyproject.toml`**

Add `"torchvision>=0.26,<0.27"` to the `train` extra and append:

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cu130", marker = "sys_platform == 'win32'" }]
torchvision = [{ index = "pytorch-cu130", marker = "sys_platform == 'win32'" }]

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true
```

Run: `uv lock && uv sync --extra dev --extra train`
Expected: the lock gains `torch 2.11.0+cu130` and `torchvision 0.26.0+cu130` (Windows markers) next to the PyPI `2.11.0` / `0.26.0`. The sync downloads about 2.5 GB on Windows.

- [ ] **Step 4: Write `tests/conftest.py`**

```python
import importlib.util

_TRAINING_STACK = ("torch", "torchvision", "transformers", "peft")

# The base install has no torch: skip the torch tests instead of failing to import them.
collect_ignore = [] if all(importlib.util.find_spec(m) for m in _TRAINING_STACK) else ["torch"]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -v`
Expected: all Plan 1 tests plus the 2 new tests PASS. On non-Windows, the cu130 test is SKIPPED.

- [ ] **Step 6: Add the CI `torch` job** to `.github/workflows/ci.yml` (the existing `test` job stays as is and skips `tests/torch` because it installs no torch):

```yaml
  torch:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: astral-sh/setup-uv@v6
      - run: uv venv --python 3.12
      # Locked versions of everything except torch: PyPI's Linux torch pulls ~3 GB of CUDA wheels.
      - run: >
          uv export --frozen --extra dev --extra train --no-emit-project --no-hashes --no-header
          --no-annotate | grep -vE '^(torch|torchvision|triton|nvidia-[A-Za-z0-9._-]+)==' > ci-req.txt
      - run: uv pip install -r ci-req.txt
      - run: uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
      - run: uv pip install --no-deps -e .
      - run: uv run --no-sync pytest tests/torch -m "not gpu and not hfjobs"
```

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff format . && uv run ruff check --fix .
git add pyproject.toml uv.lock .github/workflows/ci.yml tests/conftest.py tests/torch
git commit -F - <<'EOF'
build: add the training stack, cu130 torch on Windows and a CPU-torch CI job

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BydXoAAPdrHBxWFt1mckNw
EOF
```

---

### Task 2: Real head sizes in the estimates

**Files:**
- Modify: `clef_tune/estimate.py`, `tests/test_estimate.py`

**Interfaces:**
- Consumes: Plan 1 `estimate`.
- Produces: `ModelSpec.head_params: float` (new last field). `MODELS["Cloudflare/clef"].head_params == 128_056_324` and `MODELS["Cloudflare/clef-flash"].head_params == 121_762_820` (counted from `JointSchemaHead(**joint_head_config.json)`). `HEAD_PARAMS` is removed. `memory_needed_gb` uses `model.head_params`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_estimate.py`):

```python
def test_memory_counts_the_real_joint_head():
    assert CLEF.head_params == 128_056_324
    assert est.MODELS["Cloudflare/clef-flash"].head_params == 121_762_820
    smaller = dataclasses.replace(CLEF, head_params=CLEF.head_params - 1e8)
    delta = est.memory_needed_gb(CLEF, workload(), "none") - est.memory_needed_gb(
        smaller, workload(), "none"
    )
    assert delta == pytest.approx(1e8 * 16 / 1e9 * est.MEMORY_HEADROOM)
```

(add `import dataclasses` to the imports).

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_estimate.py -v`
Expected: FAIL with `AttributeError: 'ModelSpec' object has no attribute 'head_params'`.

- [ ] **Step 3: Implement.** Add the `head_params` field to `ModelSpec` and to both `MODELS` entries. Replace `HEAD_PARAMS` in `memory_needed_gb` with `model.head_params`, delete the constant, and update the module comment to say the head counts were measured from the release head configs.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_estimate.py -v`
Expected: all PASS. Existing bounds still hold: Clef bf16 needs about 87.3 GB and Clef 4-bit about 45.9 GB.

- [ ] **Step 5: Lint and commit** (`fix: count the real 122-128M parameter joint head in memory estimates`).

---

### Task 3: Runtime constants and Dockerfile

**Files:**
- Create: `clef_tune/runtime.py`, `Dockerfile`, `.dockerignore`, `tests/test_runtime.py`

**Interfaces:**
- Consumes: nothing.
- Produces (no torch):
  - `BASE_IMAGE = "pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel"` (Ubuntu 24.04, Python 3.12, CUDA 13.0.2; matches the CUDA 13.0 that PyPI's Linux torch 2.11 bundles). Plan 3 runs HF Jobs on the same image.
  - `KERNEL_PACKAGES = ("flash-linear-attention", "causal-conv1d")`, `KERNEL_MODULES = ("fla", "causal_conv1d")`
  - `PRICE_ENV = "CLEF_TUNE_PRICE_PER_HOUR"`, `ESTIMATE_ENV = "CLEF_TUNE_ESTIMATED_HOURS"`. Plan 3's launcher sets both; anyone may set them by hand.
  - `kernels_available() -> bool` (`importlib.util.find_spec` for each of `KERNEL_MODULES`)
  - `env_float(name: str) -> float | None`: a positive finite float, or `None` when unset or invalid

- [ ] **Step 1: Write the failing tests** `tests/test_runtime.py`:

```python
from pathlib import Path

import pytest

from clef_tune import runtime

ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_uses_the_pinned_base_image():
    first = (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()[0]
    assert first == f"FROM {runtime.BASE_IMAGE}"


def test_kernels_available_checks_every_module(monkeypatch):
    found = {"fla": object(), "causal_conv1d": None}
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda name: found.get(name))
    assert runtime.kernels_available() is False
    found["causal_conv1d"] = object()
    assert runtime.kernels_available() is True


@pytest.mark.parametrize(
    "value, expected", [(None, None), ("", None), ("abc", None), ("-1", None), ("nan", None), ("2.5", 2.5)]
)
def test_env_float(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(runtime.PRICE_ENV, raising=False)
    else:
        monkeypatch.setenv(runtime.PRICE_ENV, value)
    assert runtime.env_float(runtime.PRICE_ENV) == expected
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_runtime.py -v`
Expected: FAIL with `ImportError: cannot import name 'runtime'`.

- [ ] **Step 3: Implement `clef_tune/runtime.py`**, and write the `Dockerfile`:

```dockerfile
FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
WORKDIR /opt/clef-tune
COPY . .
RUN pip install ".[train]"
# Kernels speed up Clef's linear-attention layers; without them training still works, slower.
RUN pip install flash-linear-attention causal-conv1d \
    || echo "WARNING: kernel packages failed to install; linear attention will use the slow PyTorch path"
ENTRYPOINT ["clef-tune"]
```

`.dockerignore`: `.venv`, `out`, `.superpowers`, `**/__pycache__`, `.pytest_cache`, `.ruff_cache`.

- [ ] **Step 4: Run the tests to verify they pass** (`uv run pytest tests/test_runtime.py -v`, all PASS).

- [ ] **Step 5: Lint and commit** (`feat: add runtime constants, kernel check and Dockerfile`).

---

### Task 4: Tiny Clef fixture and Clef code import

**Files:**
- Create: `clef_tune/clef_code.py`, `tests/test_clef_code.py`, `tests/torch/conftest.py`, `tests/torch/tiny_clef.py`, `tests/torch/test_clef_code.py`
- Create (vendored, byte-for-byte from `Cloudflare/clef@a30162b99d68a481d8bd75d38df069a0cf83df40`, Apache-2.0): `tests/torch/fixtures/joint_schema_model.py`, `tokenizer_config.json`, `processor_config.json`, `chat_template.jinja`, `LICENSE`, plus a `tests/torch/fixtures/README.md` naming the source repo, revision and license
- Modify: `pyproject.toml` (`extend-exclude = ["docs", "tests/torch/fixtures"]`, so ruff never rewrites the vendored file)

**Interfaces:**
- Consumes: `config.Config`, `config.PINNED_REVISIONS`.
- Produces:
  - `clef_code.CLEF_CODE_FILE = "joint_schema_model.py"`, `clef_code.ClefCodeError(RuntimeError)`
  - `clef_code.model_dir(config: Config) -> Path`: if `model.local_path` is set, it is resolved with `config.resolve_path`; it must be a directory containing `config.json`, else raise `ClefCodeError` naming `local_path`. Otherwise return `Path(snapshot_download(model.id, revision=model.resolved_revision()))`. Import `snapshot_download` at module level so tests can monkeypatch it.
  - `clef_code.load_clef_code(model_path: str | Path) -> ModuleType`: import `<model_path>/joint_schema_model.py` as module `clef_joint_schema_model_<sha256 of the file, first 12 hex>`. Register it in `sys.modules` **before** `exec_module`, because its dataclasses look themselves up there. Return the already-registered module on later calls. Raise `ClefCodeError` if the file is missing.
  - `tests/torch/tiny_clef.py`: `build_tiny_release(out_dir: Path, seed: int = 0) -> Path`, `TINY_SCHEMA`, `write_tiny_project(project_dir: Path, release_dir: Path, overrides: dict | None = None) -> Path` (returns the config path)
  - `tests/torch/conftest.py`: session fixture `tiny_release -> Path`; fixture `tiny_project(tmp_path, tiny_release)` returning a function `make(name="run", **overrides) -> Path`

- [ ] **Step 1: Copy the vendored files.** Download them with `huggingface_hub.hf_hub_download("Cloudflare/clef", name, revision=...)` and copy the bytes unchanged. Task 17 adds a network test that pins them to the revision.

- [ ] **Step 2: Write the failing tests**

`tests/test_clef_code.py` (no torch):

```python
import pytest

from clef_tune import clef_code
from clef_tune.clef_code import ClefCodeError, model_dir
from clef_tune.config import PINNED_REVISIONS
from tests.helpers import make_config


def test_model_dir_uses_local_path(tmp_path):
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "config.json").write_text("{}", encoding="utf-8")
    cfg = make_config(tmp_path, data={"train": "t.csv"}, model={"local_path": "weights"})
    assert model_dir(cfg) == (tmp_path / "weights").resolve()


def test_model_dir_rejects_a_missing_local_path(tmp_path):
    cfg = make_config(tmp_path, data={"train": "t.csv"}, model={"local_path": "nope"})
    with pytest.raises(ClefCodeError, match="local_path"):
        model_dir(cfg)


def test_model_dir_downloads_the_pinned_revision(tmp_path, monkeypatch):
    calls = []

    def fake_snapshot_download(repo_id, revision=None, **kwargs):
        calls.append((repo_id, revision))
        return str(tmp_path)

    monkeypatch.setattr(clef_code, "snapshot_download", fake_snapshot_download)
    cfg = make_config(tmp_path, data={"train": "t.csv"}, model={"id": "Cloudflare/clef-flash"})
    assert model_dir(cfg) == tmp_path
    assert calls == [("Cloudflare/clef-flash", PINNED_REVISIONS["Cloudflare/clef-flash"])]
```

`tests/torch/test_clef_code.py`:

```python
import sys

import pytest

torch = pytest.importorskip("torch")
from clef_tune.clef_code import ClefCodeError, load_clef_code  # noqa: E402

pytestmark = pytest.mark.torch


def test_load_clef_code_registers_the_module_once(tiny_release):
    first = load_clef_code(tiny_release)
    assert load_clef_code(tiny_release) is first
    assert first.__name__.startswith("clef_joint_schema_model_")
    assert sys.modules[first.__name__] is first
    assert hasattr(first, "encode_record") and hasattr(first, "ClefModel")


def test_load_clef_code_without_the_file(tmp_path):
    with pytest.raises(ClefCodeError, match="joint_schema_model.py"):
        load_clef_code(tmp_path)


def test_tiny_release_loads_with_clefs_own_loader(tiny_release):
    jsm = load_clef_code(tiny_release)
    model, processor = jsm.load_release_model(tiny_release, device="cpu", dtype=torch.float32)
    request = {"model": "tiny", "state": "the site is down", "questions": {"outage": {"type": "noul"}}}
    answer = jsm.systemone(model, processor, request)["answers"]["outage"]
    assert answer["type"] == "noul" and 0.0 <= answer["noul"] <= 1.0
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/test_clef_code.py tests/torch/test_clef_code.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clef_tune.clef_code'` (the torch file also errors on the missing `tiny_release` fixture).

- [ ] **Step 4: Write `tests/torch/tiny_clef.py`.** These values were verified with transformers 5.10.4. The tiny model loads through Clef's `load_release_model`, runs text and image records, merges LoRA, and its saved tensor keys match the release index.

```python
"""A tiny random Clef in the exact release layout: sharded safetensors + index, joint head,
Clef's own joint_schema_model.py, tokenizer and processor files, LICENSE."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>",
                  "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"]
TINY_HEAD = {"hidden_size": 64, "width": 32, "routing_layers": 1, "layers": 1, "heads": 2,
             "feedforward": 64}
TINY_SCHEMA = {"questions": {
    "outage": {"type": "noul", "instructions": "Is a service down?"},
    "department": {"type": "choice", "criteria": {"billing": "Money", "technical": "Bugs"}},
}}
_CORPUS = ["Read the complete state and schema. Decide every field jointly.", "SCHEMA FIELDS",
           "OPTION", "END FIELD", '{"option_id":"true","description":"The proposition is true"}',
           "JOINT SCHEMA DECISIONS:", "<think>", "the site is down fine invoice server matters"]


def build_tiny_release(out_dir: Path, seed: int = 0) -> Path:
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    from clef_tune.clef_code import load_clef_code

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=SPECIAL_TOKENS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator(_CORPUS * 20, trainer)
    tokenizer.save(str(out_dir / "tokenizer.json"))
    for name in ("tokenizer_config.json", "processor_config.json", "chat_template.jinja",
                 "LICENSE", "joint_schema_model.py"):
        shutil.copyfile(FIXTURES / name, out_dir / name)
    token = tokenizer.token_to_id
    config = Qwen3_5Config(
        text_config=dict(
            vocab_size=tokenizer.get_vocab_size(), hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, layer_types=["linear_attention", "full_attention"],
            num_attention_heads=2, num_key_value_heads=1, head_dim=32,
            linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16,
            linear_value_head_dim=16, linear_conv_kernel_dim=4, max_position_embeddings=4096,
            tie_word_embeddings=False, eos_token_id=token("<|im_end|>"),
            rope_parameters={"mrope_interleaved": True, "mrope_section": [2, 2, 0],
                             "partial_rotary_factor": 0.25, "rope_theta": 10000000,
                             "rope_type": "default"},
        ),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
                           out_hidden_size=64, patch_size=16, spatial_merge_size=2,
                           temporal_patch_size=2, num_position_embeddings=64,
                           deepstack_visual_indexes=[]),
        image_token_id=token("<|image_pad|>"), video_token_id=token("<|video_pad|>"),
        vision_start_token_id=token("<|vision_start|>"),
        vision_end_token_id=token("<|vision_end|>"), tie_word_embeddings=False,
    )
    torch.manual_seed(seed)
    Qwen3_5ForConditionalGeneration(config).save_pretrained(out_dir, max_shard_size="200KB")
    head = load_clef_code(out_dir).JointSchemaHead(**TINY_HEAD)
    save_file({k: v.contiguous() for k, v in head.state_dict().items()},
              out_dir / "joint_head.safetensors")
    (out_dir / "joint_head_config.json").write_text(json.dumps(TINY_HEAD, indent=2))
    return out_dir
```

`write_tiny_project(project_dir, release_dir, overrides)` writes `train.csv` and `eval.csv` (16 rows each; columns `id,subject,body,outage,department`). The ids are `t0..t15` and `e0..e15`. Row `i` has subject `ticket {i}` and body `the site is {"down" if i % 2 == 0 else "fine"} and the {"invoice" if (i // 2) % 2 == 0 else "server"} matters`. `outage` is `yes` when the body says "down", and `department` is `billing` when it says "invoice". It then writes `config.yaml` with these base sections, deep-merging `overrides` section by section:

```yaml
run_name: tiny
model: {id: tiny/clef, local_path: <absolute release_dir>}
data: {train: train.csv, eval: eval.csv, id_column: id, state_columns: [subject, body]}
schema: <TINY_SCHEMA>
train: {mode: lora, lora: {r: 8, alpha: 16, dropout: 0.05}, lora_lr: 3.0e-3, head_lr: 3.0e-3,
        epochs: 24, max_length: 1024, max_tokens_per_batch: 1600, grad_accum_steps: 1, seed: 0}
eval: {eval_every: 24, select_metric: exact_match, calibrate: true}
output: {dir: out, save_every: 1000}
```

This recipe was verified: LoRA r=8 at learning rate 3e-3 with a cosine schedule reaches 100% exact match on these 16 rows by step 56 of 96 (loss 0.003), about 12 s on CPU. Each tiny record is about 380 tokens, so a 1600-token budget packs 4 per batch, giving 4 batches per epoch.

`tests/torch/conftest.py`:
- `tiny_release` is a session-scoped fixture that calls `build_tiny_release(tmp_path_factory.mktemp("tiny-release"))`.
- `tiny_project(tmp_path, tiny_release)` returns `make(name="run", **overrides)`, which calls `write_tiny_project(tmp_path / name, tiny_release, overrides)`.
- An autouse fixture sets `CLEF_TUNE_DEVICE=cpu` (via `monkeypatch.setenv`) for every test not marked `gpu`. The tiny-model recipes were verified on CPU, and a CUDA build of torch (Windows after Task 1) would otherwise pick the GPU.

- [ ] **Step 5: Implement `clef_tune/clef_code.py`**, then run: `uv run pytest tests/test_clef_code.py tests/torch/test_clef_code.py -v`
Expected: all PASS.

- [ ] **Step 6: Lint and commit** (`feat: import Clef's code from the model snapshot; tiny release-layout test model`).

---

### Task 5: Model loading, LoRA and trainable state

**Files:**
- Create: `clef_tune/model.py`, `tests/torch/test_model.py`

**Interfaces:**
- Consumes: `clef_code.load_clef_code`, `config.TrainSection`.
- Produces:
  - `LORA_TARGET_MODULES = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)|mlp\.(gate|up|down)_proj)"`. These are all 12 Linear kinds in Qwen3.5 decoder layers; the vision tower uses other names.
  - `DEVICE_ENV = "CLEF_TUNE_DEVICE"`; `default_device() -> torch.device`: `torch.device(os.environ[DEVICE_ENV])` when set (e.g. `cpu`, `cuda:1`), else cuda if available, else cpu
  - `compute_dtype(device) -> torch.dtype` (bf16 on cuda, fp32 on cpu)
  - `quantization_config(quantization: str) -> BitsAndBytesConfig | None`: `"4bit"` gives `load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["lm_head"]`
  - `load_backbone(model_path, *, quantization: str, device, dtype) -> Qwen3_5ForConditionalGeneration` (`device_map={"": str(device)}`, `config.use_cache = False`, every parameter frozen)
  - `load_head(model_path, clef_code, *, device, dtype) -> tuple[JointSchemaHead, dict]` (reads `joint_head_config.json`, loads `joint_head.safetensors` with `strict=True`, moves to device/dtype)
  - `@dataclass LoadedClef(model, processor, clef_code, model_path: Path, head_config: dict)`
  - `load_for_training(model_path, train: TrainSection, *, device) -> LoadedClef`
    - Backbone in `compute_dtype(device)`.
    - `lora` mode: if `train.gradient_checkpointing`, call `gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})` **before** `get_peft_model(backbone, LoraConfig(r=lora.r, lora_alpha=lora.alpha, lora_dropout=lora.dropout, target_modules=LORA_TARGET_MODULES))`.
    - `head_only` mode: no PEFT, no checkpointing.
    - Head: fp32 with `requires_grad=True`; model is `clef_code.ClefModel(backbone, head)`; processor from `AutoProcessor.from_pretrained(model_path)`.
  - `trainable_state_dict(model) -> dict[str, torch.Tensor]` (name → detached CPU clone, for every `requires_grad` parameter); `load_trainable_state(model, state) -> None` (raises `KeyError` on a missing or extra name)

- [ ] **Step 1: Write the failing tests** `tests/torch/test_model.py`:

```python
import re

import pytest

torch = pytest.importorskip("torch")
from clef_tune.config import TrainSection  # noqa: E402
from clef_tune.model import (  # noqa: E402
    compute_dtype,
    default_device,
    load_for_training,
    load_trainable_state,
    quantization_config,
    trainable_state_dict,
)

pytestmark = pytest.mark.torch
CPU = torch.device("cpu")
KINDS = {
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b",
    "linear_attn.in_proj_a", "linear_attn.out_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
}


def test_lora_targets_only_language_model_linears(tiny_release):
    loaded = load_for_training(tiny_release, TrainSection(lora={"r": 4, "alpha": 8}), device=CPU)
    names = [n for n, p in loaded.model.named_parameters() if p.requires_grad and "lora_" in n]
    assert {re.sub(r".*layers\.\d+\.", "", n).split(".lora_")[0] for n in names} == KINDS
    assert not any("visual" in n or "lm_head" in n for n in names)
    text_model = loaded.model.language_model.get_base_model().model.language_model
    assert text_model.gradient_checkpointing


def test_head_only_freezes_the_backbone_and_trains_an_fp32_head(tiny_release):
    loaded = load_for_training(tiny_release, TrainSection(mode="head_only"), device=CPU)
    trainable = [n for n, p in loaded.model.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith("head.") for n in trainable)
    assert all(p.dtype == torch.float32 for p in loaded.model.head.parameters())
    assert not hasattr(loaded.model.language_model, "get_base_model")


def test_quantization_config():
    assert quantization_config("none") is None
    q = quantization_config("4bit")
    assert q.load_in_4bit and q.bnb_4bit_quant_type == "nf4" and q.bnb_4bit_use_double_quant
    assert q.bnb_4bit_compute_dtype == torch.bfloat16 and q.llm_int8_skip_modules == ["lm_head"]


def test_compute_dtype():
    assert compute_dtype(CPU) == torch.float32
    assert compute_dtype(torch.device("cuda")) == torch.bfloat16


def test_device_env_overrides_the_default(monkeypatch):
    monkeypatch.setenv("CLEF_TUNE_DEVICE", "cpu")
    assert default_device() == CPU


def test_trainable_state_round_trip(tiny_release):
    loaded = load_for_training(tiny_release, TrainSection(lora={"r": 4, "alpha": 8}), device=CPU)
    saved = trainable_state_dict(loaded.model)
    with torch.no_grad():
        for p in loaded.model.parameters():
            if p.requires_grad:
                p.add_(1.0)
    load_trainable_state(loaded.model, saved)
    for name, p in loaded.model.named_parameters():
        if p.requires_grad:
            assert torch.equal(saved[name], p.detach().cpu())
    with pytest.raises(KeyError):
        load_trainable_state(loaded.model, {**saved, "bogus": torch.zeros(1)})
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/torch/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clef_tune.model'`.

- [ ] **Step 3: Implement `clef_tune/model.py`** as specified in Interfaces.

- [ ] **Step 4: Run the tests to verify they pass** (all PASS).

- [ ] **Step 5: Lint and commit** (`feat: load Clef for training with LoRA, head-only and 4-bit options`).

---

### Task 6: Class weights and token-budget batching (no torch)

**Files:**
- Create: `clef_tune/weights.py`, `clef_tune/batching.py`, `tests/test_weights.py`, `tests/test_batching.py`

**Interfaces:**
- Consumes: `data.Record`, `labels.Label`.
- Produces:
  - `weights.WEIGHT_CLIP = (0.1, 10.0)`
  - `weights.class_weights(records: list[Record]) -> dict[str, dict[str, float]]` (spec §5.2):
    - per question, `w_o = N_q / (K_q × n_{q,o})`, clipped to `WEIGHT_CLIP`
    - `N_q` = labeled train rows for the question; `K_q` = options observed; `n_{q,o}` = rows whose `label.top()` is `o`
  - `weights.target_weight(weights_q: dict[str, float], option_ids: list[str], target: list[float]) -> float`: `Σ_o target_o × w_o`, where an option with no weight (never observed) counts as 1.0
  - `batching.token_budget_batches(lengths: list[int], max_tokens: int, seed: int, epoch: int) -> list[list[int]]`
    - Sort indices by `(length, index)` and pack consecutive ones greedily while `len(batch) × max length in batch ≤ max_tokens`; this is the padded cost, since collation pads to the longest. An over-budget item forms its own batch.
    - Shuffle the batch order with `random.Random(seed * 1_000_003 + epoch)`. Same batches every epoch, different order.
  - `batching.shard_batches(batches, rank: int, world_size: int) -> list[list[int]]`: pad the list to a multiple of `world_size` by repeating batches from the start, then return `padded[rank::world_size]`. Every DDP rank then runs the same number of steps and none hangs.

- [ ] **Step 1: Write the failing tests**

`tests/test_weights.py`:

```python
import pytest

from clef_tune.data import Record
from clef_tune.labels import Label
from clef_tune.weights import class_weights, target_weight


def rec(i, dept):
    return Record(str(i), "s", {}, {"department": Label(hard=dept)}, [], i + 2)


def test_balanced_weights_hand_computed():
    records = [rec(i, "billing") for i in range(8)] + [rec(i, "technical") for i in range(8, 10)]
    assert class_weights(records) == {
        "department": {"billing": pytest.approx(0.625), "technical": pytest.approx(2.5)}
    }


def test_weights_clip_at_both_ends():
    records = [rec(i, "billing") for i in range(1000)] + [rec(1000, "technical")]
    assert class_weights(records)["department"]["technical"] == 10.0
    many = [rec(i, "big") for i in range(1000)] + [rec(1000 + k, f"c{k}") for k in range(10)]
    assert class_weights(many)["department"]["big"] == 0.1  # 1010 / (11 * 1000) < 0.1


def test_soft_labels_count_by_top_option_and_use_the_expected_weight():
    soft = Record("s", "s", {}, {"department": Label(soft={"billing": 0.6, "technical": 0.4})}, [], 2)
    assert class_weights([soft, rec(1, "technical")])["department"] == {
        "billing": 1.0, "technical": 1.0
    }
    weights = {"billing": 0.625, "technical": 2.5}
    assert target_weight(weights, ["billing", "technical"], [0.5, 0.5]) == pytest.approx(1.5625)
    assert target_weight(weights, ["NA", "billing", "technical"], [1.0, 0.0, 0.0]) == 1.0
```

`tests/test_batching.py`:

```python
from clef_tune.batching import shard_batches, token_budget_batches


def test_batches_cover_every_index_once_within_the_padded_budget():
    lengths = [10, 300, 50, 50, 120, 7, 300, 45]
    batches = token_budget_batches(lengths, max_tokens=400, seed=1, epoch=0)
    assert sorted(i for batch in batches for i in batch) == list(range(8))
    for batch in batches:
        assert len(batch) == 1 or len(batch) * max(lengths[i] for i in batch) <= 400


def test_an_oversized_item_gets_its_own_batch():
    batches = token_budget_batches([900, 10, 12], max_tokens=400, seed=0, epoch=0)
    assert [0] in batches


def test_batch_order_is_seeded_per_epoch():
    lengths = list(range(1, 41))
    first = token_budget_batches(lengths, 60, seed=3, epoch=0)
    assert first == token_budget_batches(lengths, 60, seed=3, epoch=0)
    second = token_budget_batches(lengths, 60, seed=3, epoch=1)
    assert first != second and sorted(first) == sorted(second)


def test_shard_batches_pads_ranks_to_equal_length():
    batches = [[0], [1], [2], [3], [4]]
    assert [shard_batches(batches, r, 2) for r in range(2)] == [[[0], [2], [4]], [[1], [3], [0]]]
    assert shard_batches(batches, 0, 1) == batches
```

- [ ] **Step 2: Run them to verify they fail** (`uv run pytest tests/test_weights.py tests/test_batching.py -v`; Expected: `ModuleNotFoundError`).

- [ ] **Step 3: Implement both modules** as specified.

- [ ] **Step 4: Run the tests to verify they pass** (all PASS).

- [ ] **Step 5: Lint and commit** (`feat: add balanced class weights and token-budget batching`).

---

### Task 7: Training items and the loss

**Files:**
- Create: `clef_tune/encoding.py`, `clef_tune/loss.py`, `tests/torch/test_encoding.py`, `tests/torch/test_loss.py`

**Interfaces:**
- Consumes: Clef's `encode_record`, `collate_records` (via the module from `load_clef_code`); `prompt.media_token_count`, `prompt.DEFAULT_URL_IMAGE_SIZE`; `data.Record`, `data.URL_PREFIXES`; `weights.target_weight`.
- Produces:
  - `@dataclass TrainItem(record: Record | None, length: int, question_ids: list[str], option_ids: dict[str, list[str]], targets: dict[str, list[float]], encoded: EncodedRecord | None)`
    - `targets` follow the encoded option order (`label.distribution(option_ids)`); masked questions are absent
    - `encoded` is `None` for records with images
  - `encoding.record_payload(record, *, with_images: bool) -> dict`: Clef's record dict `{"id", "state", "questions": {qid: q.to_clef()}}`, plus `"images"` as loaded PIL images when `with_images`. Local paths open with `PIL.Image.open(path).convert("RGB")`; URLs load with `transformers.image_utils.load_image(url)`.
  - `encoding.prepare_items(clef_code, processor, records, max_length) -> list[TrainItem]`: encodes text with `encode_record(processor.tokenizer, payload, max_length=max_length)` without images.
    - Text-only records keep the result.
    - Records with images keep only `length = text length + media_token_count(sizes)`. Sizes come from PIL for files and `DEFAULT_URL_IMAGE_SIZE` for URLs. Media is never kept in memory.
  - `encoding.materialize(item, clef_code, processor, max_length) -> EncodedRecord`: returns the cached encoding, or encodes the record with images through `processor`
  - `encoding.collate(items, clef_code, processor, max_length, device) -> dict` (`collate_records` over materialized items with `processor.tokenizer.pad_token_id`)
  - `loss.batch_loss(logits: list[list[Tensor]], items: list[TrainItem], *, class_weights: dict[str, dict[str, float]], question_weights: dict[str, float], label_smoothing: float) -> Tensor`
    - Per record `i` and labeled question `q`, with K options: smoothed target `t̃ = (1−ε)·t + ε/K`, `CE = −Σ t̃ · log_softmax(logits.float())`.
    - Weight `w = target_weight(class_weights.get(q, {}), option_ids, t) × question_weights.get(q, 1.0)`; result `Σ w·CE / Σ w`.
    - A batch with no labeled question returns `sum(l.sum() for l in all logits) * 0.0` (a zero that still backpropagates).

- [ ] **Step 1: Write the failing tests**

`tests/torch/test_loss.py`:

```python
import math

import pytest

torch = pytest.importorskip("torch")
from clef_tune.encoding import TrainItem  # noqa: E402
from clef_tune.loss import batch_loss  # noqa: E402

pytestmark = pytest.mark.torch
NOUL = ["true", "false"]


def item(targets, option_ids):
    return TrainItem(None, 1, list(option_ids), option_ids, targets, None)


def loss(logits, items, **kwargs):
    options = dict(class_weights={}, question_weights={}, label_smoothing=0.0) | kwargs
    return batch_loss(logits, items, **options).item()


def test_hard_label_cross_entropy():
    value = loss([[torch.tensor([2.0, 0.0])]], [item({"q": [1.0, 0.0]}, {"q": NOUL})])
    assert value == pytest.approx(math.log(1 + math.exp(-2)))


def test_soft_target_and_masked_question():
    logits = [[torch.zeros(3), torch.tensor([5.0, -5.0])]]
    it = item({"d": [0.5, 0.5, 0.0]}, {"d": ["a", "b", "c"], "o": NOUL})
    assert loss(logits, [it]) == pytest.approx(math.log(3))


def test_weighted_mean_uses_class_and_question_weights():
    logits = [[torch.zeros(2)], [torch.tensor([math.log(3), 0.0])]]
    items = [item({"q": [1.0, 0.0]}, {"q": NOUL}), item({"q": [0.0, 1.0]}, {"q": NOUL})]
    value = loss(logits, items, class_weights={"q": {"true": 3.0, "false": 1.0}},
                 question_weights={"q": 2.0})
    assert value == pytest.approx((6 * math.log(2) + 2 * math.log(4)) / 8)


def test_label_smoothing():
    value = loss([[torch.tensor([math.log(3), 0.0])]], [item({"q": [1.0, 0.0]}, {"q": NOUL})],
                 label_smoothing=0.2)
    assert value == pytest.approx(-(0.9 * math.log(0.75) + 0.1 * math.log(0.25)))


def test_batch_without_labels_is_a_zero_that_backpropagates():
    logits = [[torch.zeros(2, requires_grad=True)]]
    value = batch_loss(logits, [item({}, {"q": NOUL})], class_weights={}, question_weights={},
                       label_smoothing=0.0)
    assert value.item() == 0.0
    value.backward()
```

`tests/torch/test_encoding.py` (records come from `validate(load_config(tiny_project()), None).train`; `jsm` and `processor` from `load_clef_code(tiny_release)` and `AutoProcessor.from_pretrained(tiny_release)`):

```python
def test_items_follow_clefs_question_and_option_order(tiny_release, tiny_project):
    jsm, processor, records = setup(tiny_release, tiny_project)
    item = prepare_items(jsm, processor, records[:1], max_length=1024)[0]
    assert item.question_ids == ["outage", "department"]
    assert item.option_ids == {"outage": ["true", "false"], "department": ["billing", "technical"]}
    assert item.targets == {"outage": [1.0, 0.0], "department": [1.0, 0.0]}  # row t0: down, invoice
    assert item.length == len(item.encoded.input_ids)


def test_long_state_is_truncated_to_max_length(tiny_release, tiny_project):
    jsm, processor, records = setup(tiny_release, tiny_project)
    long = dataclasses.replace(records[0], state="word " * 2000)
    assert prepare_items(jsm, processor, [long], max_length=512)[0].length == 512


def test_image_records_encode_media_lazily(tiny_release, tiny_project, tmp_path):
    jsm, processor, records = setup(tiny_release, tiny_project)
    Image.new("RGB", (64, 64)).save(tmp_path / "a.png")
    with_image = dataclasses.replace(records[0], images=[str(tmp_path / "a.png")])
    (item,) = prepare_items(jsm, processor, [with_image], max_length=1024)
    (plain,) = prepare_items(jsm, processor, records[:1], max_length=1024)
    assert item.encoded is None and item.length == plain.length + 67  # 64 pads + 2 markers + newline
    encoded = materialize(item, jsm, processor, 1024)
    assert len(encoded.input_ids) == item.length and encoded.media["pixel_values"].shape[0] > 0
```

- [ ] **Step 2: Run them to verify they fail** (`ModuleNotFoundError: clef_tune.encoding`).

- [ ] **Step 3: Implement `encoding.py` and `loss.py`** as specified.

- [ ] **Step 4: Run the tests to verify they pass** (all PASS).

- [ ] **Step 5: Lint and commit** (`feat: encode training items with lazy media and add the weighted loss`).

---

### Task 8: Evaluation

**Files:**
- Create: `clef_tune/evaluation.py`, `tests/torch/test_evaluation.py`

**Interfaces:**
- Consumes: `encoding.collate`, `batching.token_budget_batches`, `metrics.Prediction`, `metrics.summarize`, `calibration.softmax`.
- Produces:
  - `predict_logits(model, items, clef_code, processor, *, max_length, max_tokens_per_batch, device) -> list[dict[str, list[float]]]`: per item, `question_id → logits`, in item order. Uses `model.eval()`, `torch.inference_mode()`, token-budget batches (`seed=0, epoch=0`), and bf16 autocast on CUDA.
  - `@dataclass EvalResult(metrics: MetricsReport, logits: dict[str, list[list[float]]], targets: dict[str, list[list[float]]])`. `logits` and `targets` hold labeled rows only, keyed by question ID, and are what `calibration.calibrate` consumes.
  - `summarize_eval(items, logits, temperatures: dict[str, float] | None = None) -> EvalResult`: one `Prediction(qid, record.questions[qid].type, option_ids, softmax(logits, T_q), target)` per labeled question (`T_q` defaults to 1.0), grouped per record for `summarize`. Questions with no labeled rows appear nowhere.

- [ ] **Step 1: Write the failing tests** `tests/torch/test_evaluation.py`:

```python
def test_predict_logits_match_clefs_forward(tiny_release, tiny_project):
    loaded = load_for_training(tiny_release, TrainSection(mode="head_only"), device=CPU)
    records = validate(load_config(tiny_project()), None).eval[:6]
    items = prepare_items(loaded.clef_code, loaded.processor, records, max_length=1024)
    got = predict_logits(loaded.model, items, loaded.clef_code, loaded.processor,
                         max_length=1024, max_tokens_per_batch=800, device=CPU)
    batch = collate(items, loaded.clef_code, loaded.processor, 1024, CPU)
    with torch.no_grad():
        direct = loaded.model(batch)
    for row, record_logits, item in zip(got, direct, items):
        for qid, tensor in zip(item.question_ids, record_logits):
            assert row[qid] == pytest.approx(tensor.tolist(), abs=1e-5)


def test_summarize_eval_handles_per_row_and_unlabeled_questions():
    q1 = Question.from_dict("q1", {"type": "noul"})
    q2 = Question.from_dict("q2", {"type": "choice", "criteria": {"a": 1, "b": 2}})
    q3 = Question.from_dict("q3", {"type": "noul"})
    a = Record("a", "s", {"q1": q1, "q2": q2}, {"q1": Label(hard="true")}, [], 2)
    b = Record("b", "s", {"q3": q3}, {"q3": Label(hard="false")}, [], 3)
    items = [
        TrainItem(a, 1, ["q1", "q2"], {"q1": NOUL, "q2": ["a", "b"]}, {"q1": [1.0, 0.0]}, None),
        TrainItem(b, 1, ["q3"], {"q3": NOUL}, {"q3": [0.0, 1.0]}, None),
    ]
    logits = [{"q1": [2.0, 0.0], "q2": [0.0, 0.0]}, {"q3": [0.0, 1.0]}]
    result = summarize_eval(items, logits, temperatures={"q1": 2.0})
    assert set(result.metrics.questions) == set(result.logits) == {"q1", "q3"}
    assert result.targets == {"q1": [[1.0, 0.0]], "q3": [[0.0, 1.0]]}
    expected = math.log(1 + math.exp(-1.0))  # softmax([2, 0] / 2)
    assert result.metrics.questions["q1"].log_loss == pytest.approx(expected)
    assert calibrate(result.logits, result.targets)["q1"].fitted is False
```

(Imports: `math`, `pytest`, `torch`, `TrainSection`, `load_config`, `validate`, `Question`, `Record`, `Label`, `calibrate`, `TrainItem`, `prepare_items`, `collate`, `load_for_training`, `predict_logits`, `summarize_eval`; `CPU = torch.device("cpu")`, `NOUL = ["true", "false"]`.)

- [ ] **Step 2: Run them to verify they fail** (`ModuleNotFoundError: clef_tune.evaluation`).

- [ ] **Step 3: Implement `clef_tune/evaluation.py`.**

- [ ] **Step 4: Run the tests to verify they pass** (all PASS).

- [ ] **Step 5: Lint and commit** (`feat: evaluate Clef logits into per-question metrics`).

---

### Task 9: Adapter card and adapter files

**Files:**
- Create: `clef_tune/card.py`, `clef_tune/adapter.py`, `tests/test_card.py`, `tests/torch/test_adapter.py`

**Interfaces:**
- Consumes: `config.Config`, `schema.Schema`, `yamlio.dump_yaml`, `clef_tune.__version__`; `model.LoadedClef`.
- Produces:
  - The **metrics dict** that `train` writes as `metrics.json` and the card renders:
    - `"baseline"` and `"final"`: `MetricsReport.to_dict()`
    - `"calibration"`: `{qid: dataclasses.asdict(QuestionCalibration)}`
    - `"data"`: `{"train_rows": int, "eval_rows": int}`
    - `"run"`: `{"steps", "best_step", "stopped_early": bool, "select_metric", "class_weighting", "hours", "estimated_hours", "price_per_hour", "cost", "estimated_cost"}` (the last four may be `None`)
  - `card.TOP_CLASSES = 10`; `card.confusion_markdown(labels: list[str], matrix: list[list[int]], top: int = TOP_CLASSES) -> str`: keeps the `top` labels with the most true rows (ties in label order) and merges the rest into `other` for both rows and columns. The header row is `| true \ predicted | <labels...> |`.
  - `card.render_card(metrics: dict, config: Config, version: str) -> str`, a Markdown card containing:
    - `Base model: {id} @ {resolved revision}` and `clef-tune {version}`
    - train/eval row counts
    - a table `| question | metric | baseline | fine-tuned |` with 3-decimal values: choice shows accuracy, macro-F1, log-loss; score shows accuracy, MAE; noul shows accuracy, F1, log-loss; plus an `| all | exact match | … |` row
    - one confusion matrix per question
    - calibration ECE before/after per question, with the sentence "Temperatures were fit on the same eval set used to choose the checkpoint, so post-calibration ECE is slightly optimistic."
    - when `class_weighting == "balanced"`, the note "Balanced class weighting shifts predicted probabilities toward rare classes; temperature calibration rescales confidence but does not undo that shift."
    - `Estimated time: {estimated_hours:.1f} h` (only when known) and `Actual time: {hours:.1f} h`, each followed by ` (${cost:.2f})` when a price is known
  - `adapter` constants: `LORA_DIR = "adapter"`, `HEAD_FILE = "joint_head.safetensors"`, `HEAD_CONFIG_FILE = "joint_head_config.json"`, `CALIBRATION_FILE = "calibration.json"`, `SCHEMA_FILE = "schema.yaml"`, `CONFIG_FILE = "train_config.yaml"`, `METRICS_FILE = "metrics.json"`, `CARD_FILE = "README.md"`
  - `adapter.save_adapter(out_dir, model, head_config: dict, config: Config, *, calibration: dict[str, float], metrics: dict) -> Path`
    - PEFT weights via `model.language_model.save_pretrained(out_dir / LORA_DIR)`, only when the backbone is a `PeftModel`
    - the head's fp32 state as safetensors, plus the head config
    - `calibration.json` as `{qid: T}`
    - `schema.yaml` when the config has a schema
    - `train_config.yaml`: the config with `model.revision` set to the resolved revision, preceded by the comment lines `# clef-tune {version}` and `# base model: {id} @ {revision}` (comments keep it loadable by `load_config`)
    - `metrics.json` (indent 2) and `README.md` from `render_card`
    - Imports torch, peft and safetensors inside the function only.
  - `@dataclass AdapterInfo(path: Path, config: Config, schema: Schema | None, calibration: dict[str, float], head_config: dict, has_lora: bool)`; `adapter.read_adapter(path) -> AdapterInfo` (no torch; `has_lora` = `(path / LORA_DIR / "adapter_config.json").exists()`)

- [ ] **Step 1: Write the failing tests**

`tests/test_card.py` defines a module-level `METRICS` dict in the shape above: questions `department` (choice) and `outage` (noul); baseline department accuracy 0.5 and final 0.9; calibration for both; `run` with `hours=1.0`, `estimated_hours=0.8`, `price_per_hour=2.5`.

```python
def test_card_reports_model_metrics_and_caveats(tmp_path):
    cfg = make_config(tmp_path, data={"train": "t.csv"}, train={"class_weighting": "balanced"})
    text = render_card(METRICS, cfg, "1.2.3")
    assert f"Base model: Cloudflare/clef @ {PINNED_REVISIONS['Cloudflare/clef']}" in text
    assert "clef-tune 1.2.3" in text
    assert "| department | accuracy | 0.500 | 0.900 |" in text
    assert "| all | exact match |" in text
    assert "slightly optimistic" in text
    assert "Balanced class weighting shifts predicted probabilities" in text
    assert "Estimated time: 0.8 h" in text and "Actual time: 1.0 h" in text


def test_card_omits_the_weighting_note_without_balanced_weighting(tmp_path):
    cfg = make_config(tmp_path, data={"train": "t.csv"})
    assert "Balanced class weighting" not in render_card(METRICS, cfg, "1.2.3")


def test_confusion_markdown_keeps_the_top_classes_and_merges_the_rest():
    labels = [f"c{i:02d}" for i in range(12)]
    matrix = [[(12 - i) if i == j else 0 for j in range(12)] for i in range(12)]
    lines = confusion_markdown(labels, matrix).splitlines()
    assert lines[0] == "| true \\ predicted | " + " | ".join(labels[:10] + ["other"]) + " |"
    assert lines[-1].startswith("| other | ") and lines[-1].endswith(" | 3 |")  # 2 + 1 on the diagonal
```

`tests/torch/test_adapter.py`:

```python
def test_save_and_read_adapter_round_trip(tiny_release, tiny_project, tmp_path):
    cfg = load_config(tiny_project())
    loaded = load_for_training(tiny_release, cfg.train, device=CPU)
    out = save_adapter(tmp_path / "final", loaded.model, loaded.head_config, cfg,
                       calibration={"outage": 1.5}, metrics=METRICS)
    assert sorted(p.name for p in out.iterdir()) == sorted([
        "README.md", "adapter", "calibration.json", "joint_head.safetensors",
        "joint_head_config.json", "metrics.json", "schema.yaml", "train_config.yaml"])
    assert (out / "adapter" / "adapter_model.safetensors").exists()
    assert (out / "train_config.yaml").read_text(encoding="utf-8").startswith("# clef-tune ")
    info = read_adapter(out)
    assert info.has_lora and info.calibration == {"outage": 1.5}
    assert info.schema == cfg.get_schema() and info.head_config == TINY_HEAD
    assert info.config.model.revision == cfg.model.resolved_revision()


def test_head_only_adapter_has_no_lora_dir(tiny_release, tiny_project, tmp_path):
    cfg = load_config(tiny_project(train={"mode": "head_only"}))
    loaded = load_for_training(tiny_release, cfg.train, device=CPU)
    out = save_adapter(tmp_path / "final", loaded.model, loaded.head_config, cfg,
                       calibration={}, metrics=METRICS)
    assert not (out / "adapter").exists() and not read_adapter(out).has_lora
```

(`METRICS` is imported from `tests/test_card.py`.)

- [ ] **Step 2: Run them to verify they fail** (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `card.py` and `adapter.py`** (`save_adapter`, `read_adapter`; Task 15 adds `export_merged`).

- [ ] **Step 4: Run the tests to verify they pass** (all PASS).

- [ ] **Step 5: Lint and commit** (`feat: write portable adapters with a metrics card`).

---

### Task 10: Checkpoints

**Files:**
- Create: `clef_tune/checkpoints.py`, `tests/torch/test_checkpoints.py`

**Interfaces:**
- Consumes: `model.trainable_state_dict`, `model.load_trainable_state`.
- Produces (everything lives under `output_dir / "checkpoints"`):
  - `KEEP_CHECKPOINTS = 2`
  - `@dataclass TrainState(step: int, epoch: int, batch_in_epoch: int, best_metric: float | None, best_step: int | None, bad_evals: int, config_hash: str, tokens_seen: int, train_seconds: float)`
  - `class ConfigMismatchError(RuntimeError)`
  - `save_checkpoint(output_dir, model, optimizer, scheduler, state) -> Path`
    - writes `step-{N}/` containing `trainable.safetensors`, `optimizer.pt`, `scheduler.pt`, `rng.pt` (Python, NumPy, torch CPU and, when present, CUDA RNG states) and `state.json`
    - writes to `step-{N}.tmp/` then renames, so a crash never leaves a half-written checkpoint
    - prunes to the newest `KEEP_CHECKPOINTS` by step number
  - `latest_checkpoint(output_dir) -> Path | None`
  - `load_checkpoint(path, model, optimizer, scheduler) -> TrainState` (restores the RNG states too; `rng.pt` loads with `weights_only=False` because it holds NumPy state)
  - `check_resume(output_dir, config_hash) -> Path | None`: the latest checkpoint, or `None`. On a hash mismatch, raises `ConfigMismatchError("checkpoints in <dir> were made with a different config or data; run with --restart to discard them, or set a new output.dir")`.
  - `save_best(output_dir, model, eval_logits: dict, eval_targets: dict, state) -> None` → `best/` (trainable state + `eval.json` + `state.json`); `load_best(output_dir, model) -> tuple[dict, dict]` (loads the weights into `model`, returns logits and targets)
  - `save_baseline(output_dir, metrics: dict)` / `load_baseline(output_dir) -> dict | None` → `baseline.json`
  - `clear_checkpoints(output_dir) -> None` (deletes `checkpoints/`)

- [ ] **Step 1: Write the failing tests** `tests/torch/test_checkpoints.py`:

```python
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


def test_keeps_the_last_two_and_restores_weights_optimizer_scheduler_and_rng(tiny_release, tmp_path):
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
```

- [ ] **Step 2: Run them to verify they fail** (`ModuleNotFoundError: clef_tune.checkpoints`).

- [ ] **Step 3: Implement `clef_tune/checkpoints.py`.**

- [ ] **Step 4: Run the tests to verify they pass** (all PASS).

- [ ] **Step 5: Lint and commit** (`feat: add atomic step checkpoints, best state and resume checks`).

---

### Task 11: The training run

**Files:**
- Create: `clef_tune/train.py`, `tests/torch/test_train.py`

**Interfaces:**
- Consumes: Plan 1 `load_config`, `validate`, `counter_for_config`, `config_hash`, `estimate.model_spec` / `workload_from` / `memory_needed_gb`, `calibration.calibrate` / `temperatures`, `metrics.is_better`; Tasks 3–10.
- Produces:
  - `class TrainingError(RuntimeError)`
  - `@dataclass TrainResult(adapter_dir: Path, metrics: dict, resumed_from: int | None, hub_url: str | None)`
  - `LOG_EVERY = 10`, `ETA_AFTER_STEPS = 20`
  - `train(config_path, *, restart: bool = False, push_to_hub: str | None = None, delete_bundle_on_success: bool = False, on_step: Callable[[int], None] | None = None, device: torch.device | None = None) -> TrainResult`
  - Output layout under `config.output_dir()`: `metrics.jsonl`, `checkpoints/` (Task 10) and `final/` (the adapter; `TrainResult.adapter_dir`).

The run, in order:

1. Load the config and run `validate(config, counter_for_config(config))`. On errors, raise `TrainingError("validation failed:\n" + one issue per line)` before loading any model.
2. `hash = config_hash(config, [train file, eval file if set])`. With `restart`, call `clear_checkpoints`. Then `check_resume(...)`.
3. Pre-flight on CUDA:
   - If `model_spec(model.id)` is known and `memory_needed_gb(spec, workload_from(config, report.tokens), quantization)` exceeds the device's total memory, raise `TrainingError`. Its advice is the same list as the OOM advice in Task 12.
   - If `kernels_available()` is false, print a warning that linear attention will use the slow PyTorch path.
4. `loaded = load_for_training(model_dir(config), config.train, device=device or default_device())`.
   - Items come from `prepare_items` over `report.train` and `report.eval`.
   - `class_weights(report.train)` when `class_weighting == "balanced"`, else `{}`.
5. Optimizer: AdamW with two parameter groups (LoRA parameters at `lora_lr`, head at `head_lr`) and `weight_decay`.
   - Total optimizer steps = `epochs × ceil(batches per epoch per rank / grad_accum_steps)`.
   - Scheduler: `transformers.get_cosine_schedule_with_warmup(optimizer, round(warmup_ratio × total), total)`.
   - Wrap with `accelerate.Accelerator(mixed_precision="bf16" if cuda else "no", gradient_accumulation_steps=grad_accum_steps, log_with=report_to or None, project_dir=output_dir)` and `prepare(model, optimizer, scheduler)`.
6. Baseline: when not resuming, evaluate with `predict_logits` + `summarize_eval` on this rank's shard of the eval items and gather with `accelerator.gather_object`. Save with `save_baseline` and log `{"event": "baseline", ...}`. When resuming, `load_baseline`.
7. For each epoch (starting at the restored epoch):
   - `batches = shard_batches(token_budget_batches(lengths, max_tokens_per_batch, seed, epoch), rank, world)`, skipping the first `batch_in_epoch` when resuming.
   - Per batch, under `accelerator.accumulate(model)`: `batch_loss` → backward → when gradients sync, clip to 1.0, then `optimizer.step()`, `scheduler.step()`, `zero_grad()`.
   - A non-finite loss raises `TrainingError` (Task 12 adds the message).
   - After each optimizer step: `step += 1`. Every `LOG_EVERY` steps and on the last step, log `{"event": "train", "step", "loss", "lr", "tokens_per_sec", "eta_s", "projected_cost"}`.
     - `tokens_per_sec` counts padded tokens trained since this process started.
     - `eta_s` and `projected_cost` are `None` until `ETA_AFTER_STEPS` steps.
     - `projected_cost = (train_seconds + eta_s) / 3600 × env_float(PRICE_ENV)`, or `None` without a price.
   - Every `eval_every` steps: evaluate and log `{"event": "eval", "step", "exact_match", "mean_macro_f1", "mean_log_loss"}`.
     - If `is_better(select_metric, value, best_metric)`: `save_best` and reset `bad_evals`. Otherwise `bad_evals += 1`.
     - Stop early when `early_stopping_patience` is set and `bad_evals ≥ early_stopping_patience`.
   - Every `save_every` steps: `save_checkpoint` (main process only; all ranks wait). `TrainState.batch_in_epoch` counts the micro-batches this rank has consumed in the current epoch.
   - Last, call `on_step(step)` if given. The step is complete by then, including its eval and checkpoint.
8. After the loop, evaluate again unless the last step was just evaluated (same best logic). `load_best` into the model.
   - `calibration = temperatures(calibrate(logits, targets))` when `eval.calibrate`, else `{}`.
   - Print a warning per question that keeps `T = 1` for having fewer than 20 rows.
9. Build the metrics dict (Task 9 shape): `final` = the best eval's `MetricsReport.to_dict()`.
   - `run.hours` is wall-clock hours of this process plus `train_seconds` restored from the checkpoint.
   - `estimated_hours = env_float(ESTIMATE_ENV)`; `cost` and `estimated_cost` come from `env_float(PRICE_ENV)`.
   - On the main process, `save_adapter(output_dir / "final", ...)` with the unwrapped model, then log `{"event": "done", "adapter": str(path)}`.
10. On success only, after the adapter is written: `clear_checkpoints(output_dir)`.

`metrics.jsonl` is opened in append mode, so a resumed run keeps the earlier lines.

- [ ] **Step 1: Write the failing tests** `tests/torch/test_train.py`:

```python
import json

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import load_file  # noqa: E402

from clef_tune import train as train_module  # noqa: E402
from clef_tune.train import TrainingError, train  # noqa: E402

pytestmark = pytest.mark.torch


def events(config_path):
    lines = (config_path.parent / "out" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_lora_overfits_sixteen_rows(tiny_project):
    config = tiny_project()
    result = train(config)
    final, baseline = result.metrics["final"], result.metrics["baseline"]
    assert final["exact_match"] == 1.0
    assert final["mean_log_loss"] < baseline["mean_log_loss"]
    assert result.adapter_dir == config.parent / "out" / "final"
    assert json.loads((result.adapter_dir / "calibration.json").read_text()) == {
        "outage": 1.0, "department": 1.0}  # 16 eval rows < 20: T stays 1
    kinds = [e["event"] for e in events(config)]
    assert kinds[0] == "baseline" and kinds[-1] == "done" and "eval" in kinds
    train_events = [e for e in events(config) if e["event"] == "train"]
    assert {"step", "loss", "lr", "tokens_per_sec", "eta_s", "projected_cost"} <= set(train_events[-1])
    assert train_events[-1]["step"] == result.metrics["run"]["steps"] >= 24 * 4
    assert train_events[-1]["eta_s"] is not None
    assert not (config.parent / "out" / "checkpoints").exists()


def test_head_only_trains_only_the_head(tiny_project, tiny_release):
    before = load_file(tiny_release / "joint_head.safetensors")
    result = train(tiny_project(train={"mode": "head_only", "epochs": 2}))
    after = load_file(result.adapter_dir / "joint_head.safetensors")
    assert any(not torch.allclose(before[k].float(), after[k]) for k in before)
    assert not (result.adapter_dir / "adapter").exists()


def test_balanced_class_weighting_trains(tiny_project):
    result = train(tiny_project(train={"class_weighting": "balanced", "epochs": 2}))
    assert result.metrics["run"]["class_weighting"] == "balanced"
    assert "Balanced class weighting" in (result.adapter_dir / "README.md").read_text(encoding="utf-8")


def test_early_stopping_stops_when_the_metric_stalls(tiny_project):
    config = tiny_project(train={"lora_lr": 1e-12, "head_lr": 1e-12, "epochs": 2},
                          eval={"eval_every": 1, "early_stopping_patience": 2})
    run = train(config).metrics["run"]
    assert run["stopped_early"] and run["steps"] == 3 and run["best_step"] == 1


def test_validation_errors_stop_before_the_model_loads(tiny_project, monkeypatch):
    config = tiny_project()
    csv = config.parent / "train.csv"
    csv.write_text(csv.read_text(encoding="utf-8").replace(",billing", ",bogus", 1), encoding="utf-8")
    monkeypatch.setattr(train_module, "load_for_training",
                        lambda *a, **k: pytest.fail("model loaded despite validation errors"))
    with pytest.raises(TrainingError, match="validation failed"):
        train(config)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/torch/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clef_tune.train'`.

- [ ] **Step 3: Implement `clef_tune/train.py`** following the sequence above.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/torch/test_train.py -v`
Expected: all PASS in about a minute on CPU. If `test_lora_overfits_sixteen_rows` ends below 1.0 exact match, raise the fixture's `epochs` to 32 and ledger a ruling. The shuffled batch order differs from the verification prototype, which ran fixed-order batches.

- [ ] **Step 5: Lint and commit** (`feat: train Clef with evaluation, best-checkpoint selection and calibration`).

---

### Task 12: Resume, restart, cleanup and failure handling

**Files:**
- Modify: `clef_tune/train.py`
- Test: `tests/torch/test_train_resume.py`

**Interfaces:**
- Consumes: Task 11 `train`, Task 10 checkpoints, `bundle.write_bundle`.
- Produces, in `train` behavior:
  - **Resume:** prints `Resuming from step N` and sets `TrainResult.resumed_from`.
  - **NaN/inf loss:** `TrainingError("loss became non-finite at step {S}; the last good checkpoint is step {N} in {dir}")`, or `"...; no checkpoint was saved yet"` when none exists. `S` is the optimizer step being computed (completed steps + 1).
  - **`torch.OutOfMemoryError`:** re-raised as `TrainingError` containing the original message and the advice "lower train.max_tokens_per_batch or train.max_length, set train.quantization: 4bit, or use a larger GPU flavor". Checkpoints are kept.
  - **`delete_bundle_on_success`:** validated in pre-flight. The config's directory must contain `config.yaml`, `train.jsonl` and `eval.jsonl`, else `TrainingError("--delete-bundle-on-success needs a bundle (config.yaml, train.jsonl, eval.jsonl); <dir> is not a bundle")`. After success and cleanup, deletes exactly those three files plus `images/`, then removes the directory if it is empty.

- [ ] **Step 1: Write the failing tests** `tests/torch/test_train_resume.py`:

```python
class Stop(Exception):
    pass


# 800-token batches hold 1-2 tiny records, so an epoch has 8-16 batches (4-8 optimizer steps at
# grad accumulation 2), and the step-3 checkpoint always lands mid-epoch.
OVERRIDES = dict(train={"epochs": 2, "grad_accum_steps": 2, "max_tokens_per_batch": 800},
                 eval={"eval_every": 2}, output={"save_every": 3})


def adapter_tensors(result):
    tensors = load_file(result.adapter_dir / "joint_head.safetensors")
    tensors.update(load_file(result.adapter_dir / "adapter" / "adapter_model.safetensors"))
    return tensors


def test_interrupted_run_resumes_to_the_same_result(tiny_project):
    full = train(tiny_project("full", **OVERRIDES))
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


def test_changed_config_refuses_to_resume_until_restart(tiny_project):
    config = tiny_project(train={"epochs": 2}, output={"save_every": 1})

    def stop(step):
        raise Stop

    with pytest.raises(Stop):
        train(config, on_step=stop)
    config.write_text(config.read_text(encoding="utf-8").replace("0.003", "0.002"), encoding="utf-8")
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
```

(Imports: `pytest`, `torch` via importorskip, `load_file`, `load_config`, `validate`, `write_bundle`, `ConfigMismatchError`, `train_module`, `TrainingError`, `train`. Add `pytestmark = pytest.mark.torch`.)

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/torch/test_train_resume.py -v`
Expected: the resume test fails on `resumed_from` (Task 11 always starts fresh, or resumes without restoring state). The bundle, NaN and OOM tests fail on the missing behavior or messages.

- [ ] **Step 3: Implement the behaviors** in `train.py`. Restore `TrainState` (epoch, `batch_in_epoch`, best, `bad_evals`, `tokens_seen`, `train_seconds`) and the RNG states before the epoch loop.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/torch/test_train.py tests/torch/test_train_resume.py -v`
Expected: all PASS.

- [ ] **Step 5: Lint and commit** (`feat: auto-resume training, clean up on success and explain failures`).

---

### Task 13: Hub push and pull

**Files:**
- Create: `clef_tune/hub.py`, `tests/test_hub.py`
- Modify: `clef_tune/train.py`, `tests/torch/test_train.py`

**Interfaces:**
- Consumes: `huggingface_hub.HfApi`, `huggingface_hub.snapshot_download`.
- Produces (no torch):
  - `class HubError(RuntimeError)`
  - `ensure_repo(repo_id, *, token=None, private=True, api=None) -> str`: `api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True, token=token)`. Any exception becomes `HubError(f"cannot create or write to {repo_id}: {exc} (check that your token has write access)")`.
  - `push_adapter(adapter_dir, repo_id, *, token=None, api=None) -> str`: `ensure_repo` then `api.upload_folder(folder_path=str(adapter_dir), repo_id=repo_id, repo_type="model", commit_message="Upload clef-tune adapter", token=token)`; returns the commit URL as a string. Failures raise `HubError`.
  - `pull_adapter(repo_id, *, revision=None, token=None) -> Path` (`snapshot_download(repo_id, revision=revision, token=token)`; module-level import for monkeypatching)
  - `api=None` means `HfApi()`
  - `train` changes:
    - `push_to_hub or config.output.push_to_hub` names the repo.
    - Pre-flight calls `ensure_repo` before the model loads; a `HubError` becomes `TrainingError`.
    - After `save_adapter`, call `push_adapter` and store the URL in `TrainResult.hub_url`.
    - A push failure raises `TrainingError(f"training finished and the adapter is in {final}, but pushing to {repo} failed: {error}")` **before** `clear_checkpoints` and bundle deletion.

- [ ] **Step 1: Write the failing tests**

`tests/test_hub.py`:

```python
class FakeApi:
    def __init__(self, fail=None):
        self.calls, self.fail = [], fail

    def create_repo(self, repo_id, **kwargs):
        self.calls.append(("create_repo", repo_id, kwargs))
        if self.fail:
            raise self.fail
        return f"https://huggingface.co/{repo_id}"

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", kwargs))
        return "https://huggingface.co/me/run/commit/abc"


def test_ensure_repo_creates_a_private_model_repo():
    api = FakeApi()
    ensure_repo("me/run", api=api)
    assert api.calls == [("create_repo", "me/run",
                          {"repo_type": "model", "private": True, "exist_ok": True, "token": None})]


def test_ensure_repo_explains_permission_errors():
    with pytest.raises(HubError, match="write access"):
        ensure_repo("me/run", api=FakeApi(fail=RuntimeError("403 Forbidden")))


def test_push_adapter_uploads_the_folder(tmp_path):
    api = FakeApi()
    assert push_adapter(tmp_path, "me/run", api=api) == "https://huggingface.co/me/run/commit/abc"
    kwargs = api.calls[-1][1]
    assert kwargs["folder_path"] == str(tmp_path) and kwargs["repo_id"] == "me/run"
    assert kwargs["repo_type"] == "model"


def test_pull_adapter(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(hub, "snapshot_download",
                        lambda repo_id, **kw: calls.append((repo_id, kw)) or str(tmp_path))
    assert pull_adapter("me/run", revision="v1") == tmp_path
    assert calls == [("me/run", {"revision": "v1", "token": None})]
```

Append to `tests/torch/test_train.py`:

```python
def test_push_to_hub_after_training(tiny_project, monkeypatch):
    calls = []
    monkeypatch.setattr(train_module, "ensure_repo", lambda repo, **k: calls.append(("ensure", repo)))
    monkeypatch.setattr(train_module, "push_adapter",
                        lambda path, repo, **k: calls.append(("push", repo)) or "https://hub/commit")
    result = train(tiny_project(train={"epochs": 1}), push_to_hub="me/run")
    assert calls == [("ensure", "me/run"), ("push", "me/run")]
    assert result.hub_url == "https://hub/commit"


def test_push_failure_keeps_adapter_and_checkpoints(tiny_project, monkeypatch):
    def fail(path, repo, **kwargs):
        raise HubError("503 Service Unavailable")

    monkeypatch.setattr(train_module, "ensure_repo", lambda repo, **k: None)
    monkeypatch.setattr(train_module, "push_adapter", fail)
    config = tiny_project(train={"epochs": 1}, output={"save_every": 2})
    with pytest.raises(TrainingError, match="adapter is in .* but pushing to me/run failed"):
        train(config, push_to_hub="me/run")
    assert (config.parent / "out" / "final" / "joint_head.safetensors").exists()
    assert (config.parent / "out" / "checkpoints").exists()


def test_hub_access_is_checked_before_the_model_loads(tiny_project, monkeypatch):
    def denied(repo, **kwargs):
        raise HubError("cannot create or write to me/run")

    monkeypatch.setattr(train_module, "ensure_repo", denied)
    monkeypatch.setattr(train_module, "load_for_training",
                        lambda *a, **k: pytest.fail("model loaded before the hub check"))
    with pytest.raises(TrainingError, match="me/run"):
        train(tiny_project(), push_to_hub="me/run")
```

- [ ] **Step 2: Run them to verify they fail** (`ModuleNotFoundError: clef_tune.hub`; the train tests fail on the missing `ensure_repo` attribute).

- [ ] **Step 3: Implement `hub.py`** and wire it into `train.py` as specified.

- [ ] **Step 4: Run the tests to verify they pass** (`uv run pytest tests/test_hub.py tests/torch/test_train.py -v`, all PASS).

- [ ] **Step 5: Lint and commit** (`feat: push adapters to private Hub repos and pull them back`).

---

### Task 14: Predictor and prediction files

**Files:**
- Create: `clef_tune/predictor.py`, `tests/torch/test_predictor.py`
- Modify: `clef_tune/data.py`, `tests/test_data.py`

**Interfaces:**
- Consumes: `adapter.read_adapter`, `hub.pull_adapter`, `clef_code.model_dir` / `load_clef_code`, `model.load_backbone` / `load_head` / `default_device` / `compute_dtype`, `encoding.prepare_items` / `collate`, `evaluation.predict_logits`, `calibration.softmax`, `data.load_file`.
- Produces:
  - `data.load_file(config, path_value, *, for_prediction: bool = False)` and `normalize_rows(..., for_prediction: bool = False)`. With `for_prediction`, missing label columns are not errors and rows with no labels are kept (labels present are still parsed). Default behavior is unchanged.
  - `class Predictor` with attributes `model`, `processor`, `clef_code`, `schema: Schema | None`, `temperatures: dict[str, float]`, `info: AdapterInfo`, `max_length: int`:
    - `Predictor.load(path_or_repo, *, device=None, quantization: str | None = None, revision=None, token=None) -> Predictor`
      - An existing local dir is used as-is; anything else goes through `pull_adapter`.
      - The base model comes from `model_dir(info.config)`. `quantization` defaults to `info.config.train.quantization`.
      - The backbone loads in `compute_dtype(device)`. If `info.has_lora`, wrap it with `PeftModel.from_pretrained(backbone, path / "adapter")`.
      - The head loads from the adapter dir in the same dtype; `max_length = info.config.train.max_length`.
    - `logits(rows) -> list[dict[str, list[float]]]`: each row is `{"id"?, "state", "questions"?, "images"?}`; questions default to the adapter schema (a `ValueError` when there is neither)
    - `probabilities(rows, *, calibrated=True) -> list[dict[str, dict[str, float]]]`: `softmax(logits, T_q)` with `T_q = temperatures.get(qid, 1.0)` when calibrated
    - `predict(rows, *, calibrated=True) -> list[dict]`: `[{"id": row id or row index as str, "answers": {qid: clef_code.systemone_answer(question.to_clef(), probabilities)}}]`
    - `systemone(request, *, calibrated=True) -> dict`: the same validation and response shape as Clef's `systemone` (`model`, `answers`, `usage.input_tokens`), with calibration applied
  - `predict_file(predictor, data_path, out_path, *, calibrated=True) -> int` (rows written)
    - Rows load with `load_file(predictor.info.config, str(Path(data_path).resolve()), for_prediction=True)`.
    - A `.jsonl` output holds one `{"id", "answers"}` per line.
    - A `.csv` output has columns `id`, then per question `<qid>` (the argmax option ID) followed by `<qid>.p.<option>` for each option, in Clef's option order.
    - Any other suffix raises `ValueError`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_data.py`:

```python
def test_prediction_input_needs_no_label_columns(tmp_path):
    (tmp_path / "p.csv").write_text("id,subject,body\n1,a,b\n2,c,d\n", encoding="utf-8")
    cfg = make_config(tmp_path, data={"train": "t.csv", "id_column": "id"})
    result = load_file(cfg, "p.csv", for_prediction=True)
    assert [r.id for r in result.records] == ["1", "2"]
    assert all(r.labels == {} for r in result.records)
    assert result.records[0].state == {"subject": "a", "body": "b"}
    assert [i for i in result.issues if i.level == "error"] == []


def test_prediction_input_still_parses_labels_that_are_present(tmp_path):
    (tmp_path / "p.csv").write_text("id,body,outage\n1,a,yes\n2,b,\n", encoding="utf-8")
    cfg = make_config(tmp_path, data={"train": "t.csv", "id_column": "id"})
    first, second = load_file(cfg, "p.csv", for_prediction=True).records
    assert first.labels == {"outage": Label(hard="true")} and second.labels == {}
```

Add a session-scoped `adapter` fixture to `tests/torch/conftest.py`. It calls `write_tiny_project(tmp_path_factory.mktemp("adapter"), tiny_release, {"train": {"epochs": 2}})`, trains once, overwrites `calibration.json` with `{"outage": 2.0}`, and returns the adapter dir. Task 15 reuses it. `tests/torch/test_predictor.py`:

```python
ROWS = [{"id": "a", "state": {"subject": "x", "body": "the site is down and the invoice matters"}}]


def test_predict_returns_systemone_answers(adapter):
    out = Predictor.load(adapter, device="cpu").predict(ROWS)
    assert out[0]["id"] == "a" and set(out[0]["answers"]) == {"outage", "department"}
    assert out[0]["answers"]["outage"]["type"] == "noul"
    assert set(out[0]["answers"]["department"]["probabilities"]) == {"billing", "technical"}


def test_calibration_applies_the_stored_temperatures(adapter):
    predictor = Predictor.load(adapter, device="cpu")
    raw = predictor.probabilities(ROWS, calibrated=False)[0]
    calibrated = predictor.probabilities(ROWS)[0]
    tempered = temper_probabilities(list(raw["outage"].values()), 2.0)
    assert list(calibrated["outage"].values()) == pytest.approx(tempered, abs=1e-6)
    assert calibrated["department"] == pytest.approx(raw["department"], abs=1e-6)


def test_systemone_keeps_clefs_contract(adapter):
    response = Predictor.load(adapter, device="cpu").systemone(
        {"model": "m", "state": "s", "questions": {"spam": {"type": "noul"}}})
    assert response["model"] == "m" and set(response["answers"]) == {"spam"}
    assert response["usage"]["input_tokens"] > 0 and response["usage"]["output_tokens"] == 0


def test_predict_file_writes_jsonl_and_csv(adapter, tmp_path):
    (tmp_path / "in.csv").write_text("id,subject,body\n1,a,down\n2,b,fine\n3,c,x\n", encoding="utf-8")
    predictor = Predictor.load(adapter, device="cpu")
    assert predict_file(predictor, tmp_path / "in.csv", tmp_path / "out.jsonl") == 3
    assert len((tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    predict_file(predictor, tmp_path / "in.csv", tmp_path / "out.csv")
    header = (tmp_path / "out.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header[:5] == ["id", "outage", "outage.p.true", "outage.p.false", "department"]


def test_image_rows_predict(adapter, tmp_path):
    Image.new("RGB", (64, 64), (200, 10, 10)).save(tmp_path / "i.png")
    out = Predictor.load(adapter, device="cpu").predict(
        [{"id": "i", "state": "s", "images": [str(tmp_path / "i.png")]}])
    assert set(out[0]["answers"]) == {"outage", "department"}
```

- [ ] **Step 2: Run them to verify they fail** (`TypeError: load_file() got an unexpected keyword argument 'for_prediction'`; `ModuleNotFoundError: clef_tune.predictor`).

- [ ] **Step 3: Implement** the `for_prediction` path in `data.py`, then `predictor.py`.

- [ ] **Step 4: Run the tests to verify they pass** (`uv run pytest tests/test_data.py tests/torch/test_predictor.py -v`, all PASS).

- [ ] **Step 5: Lint and commit** (`feat: add the calibrated Predictor API and prediction files`).

---

### Task 15: Merged release-format export

**Files:**
- Modify: `clef_tune/adapter.py`
- Test: `tests/torch/test_export.py`

**Interfaces:**
- Consumes: `read_adapter`, `model_dir`, `load_backbone`, `peft.PeftModel`, `Predictor.logits`.
- Produces:
  - `RELEASE_COPY_FILES = ("joint_schema_model.py", "tokenizer.json", "tokenizer_config.json", "processor_config.json", "chat_template.jinja", "LICENSE")`
  - `export_merged(adapter, out_dir, *, dtype=None, device="cpu", max_shard_size="5GB") -> Path` (`dtype=None` means bf16)
    - Loads the base backbone unquantized; with LoRA, `PeftModel.from_pretrained(...).merge_and_unload()`.
    - `save_pretrained(out_dir, max_shard_size=max_shard_size)`. Verified: the merged keys equal the release index keys.
    - Writes the head in `dtype` as `joint_head.safetensors`, plus `joint_head_config.json`, and copies `RELEASE_COPY_FILES` that exist in the base snapshot.
    - Writes `calibration.json` and a `README.md` stating that SGLang serves **uncalibrated** probabilities, and how a client calibrates them with `p_i^(1/T) / Σ_j p_j^(1/T)`, using the per-question temperatures in `calibration.json`.

- [ ] **Step 1: Write the failing test** `tests/torch/test_export.py` (uses the session `adapter` fixture from Task 14, `ROWS` from `tests.torch.test_predictor` and `TINY_SCHEMA` from `tests.torch.tiny_clef`):

```python
def test_merged_export_loads_with_clefs_loader_and_matches_the_adapter(adapter, tmp_path):
    out = export_merged(adapter, tmp_path / "merged", dtype=torch.float32, max_shard_size="200KB")
    names = {p.name for p in out.iterdir()}
    assert {"config.json", "model.safetensors.index.json", "joint_head.safetensors",
            "joint_head_config.json", "calibration.json", "README.md", *RELEASE_COPY_FILES} <= names
    jsm = load_clef_code(out)
    model, processor = jsm.load_release_model(out, device="cpu", dtype=torch.float32)
    request = {"model": "m", "state": ROWS[0]["state"], "questions": TINY_SCHEMA["questions"]}
    encoded = jsm.encode_record(processor.tokenizer, request, max_length=1024)
    with torch.no_grad():
        merged = model(jsm.collate_records([encoded], processor.tokenizer.pad_token_id, torch.device("cpu")))[0]
    expected = Predictor.load(adapter, device="cpu").logits(ROWS)[0]
    for question, logits in zip(encoded.questions, merged):
        assert logits.tolist() == pytest.approx(expected[question.question_id], abs=1e-4)
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "uncalibrated" in readme and "p_i^(1/T)" in readme
```

- [ ] **Step 2: Run it to verify it fails** (`ImportError: cannot import name 'export_merged'`).

- [ ] **Step 3: Implement `export_merged`.**

- [ ] **Step 4: Run the test to verify it passes** (PASS).

- [ ] **Step 5: Lint and commit** (`feat: export merged release-format checkpoints for SGLang`).

---

### Task 16: CLI commands and README

**Files:**
- Modify: `clef_tune/cli.py`, `tests/test_cli.py`, `tests/test_boundaries.py`, `README.md`
- Test: `tests/torch/test_cli_end_to_end.py`

**Interfaces:**
- Consumes: `train.train` / `TrainingError`, `checkpoints.ConfigMismatchError`, `predictor.Predictor` / `predict_file`, `adapter.read_adapter` / `export_merged`, `data.load_file`, `evaluation`, `calibration`, `card.confusion_markdown`. Each command imports these **inside** its function.
- Produces commands. Exit codes: 0 ok; 1 for a training, validation or Hub failure; 2 for an invalid config or bad arguments.
  - `train CONFIG [--restart] [--push-to-hub REPO] [--delete-bundle-on-success]`
    - Loads the config with `_load` (exit 2 when invalid), then calls `train(...)`.
    - Prints `Adapter written: <dir>` and, if pushed, `Pushed: <url>`.
    - `TrainingError` and `ConfigMismatchError` print `[red]Training failed:[/] <message>` and exit 1.
  - `eval ADAPTER DATA [--output FILE] [--no-calibration]`
    - Loads `DATA` with `load_file(adapter config, path)`, runs `Predictor.logits` and `summarize_eval`, using the stored temperatures unless `--no-calibration`.
    - Prints a per-question metrics table, an `Exact match: x.xxx` line, a confusion matrix per question (`confusion_markdown`), and per-question ECE before and after the stored temperatures (`expected_calibration_error`).
    - `--output` writes `MetricsReport.to_dict()` plus a `"calibration": {qid: {"ece_before", "ece_after", "temperature"}}` key as JSON.
    - Fails with exit 1 if the file has no labeled rows.
  - `predict ADAPTER DATA -o OUT [--no-calibration]`: `predict_file`, then `Wrote N predictions to OUT`
  - `export ADAPTER --merge -o OUT`: without `--merge`, print "only --merge export is supported" and exit 2
- `tests/test_boundaries.py`: add `clef_tune.runtime`, `clef_tune.clef_code`, `clef_tune.weights`, `clef_tune.batching`, `clef_tune.card`, `clef_tune.hub` and `clef_tune.adapter` to `NO_TORCH_MODULES`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py` (no torch; a fake `clef_tune.train` module goes into `sys.modules`):

```python
def fake_training(monkeypatch, tmp_path, error=None):
    calls = []

    class FakeError(RuntimeError):
        pass

    def train(config, **kwargs):
        calls.append((config, kwargs))
        if error:
            raise FakeError(error)
        return SimpleNamespace(adapter_dir=tmp_path / "final", hub_url=None, resumed_from=None)

    module = SimpleNamespace(train=train, TrainingError=FakeError)
    monkeypatch.setitem(sys.modules, "clef_tune.train", module)
    monkeypatch.setitem(sys.modules, "clef_tune.checkpoints",
                        SimpleNamespace(ConfigMismatchError=type("ConfigMismatchError", (RuntimeError,), {})))
    return calls


def test_train_command_passes_its_options(monkeypatch, tmp_path):
    calls = fake_training(monkeypatch, tmp_path)
    result = runner.invoke(cli.app, ["train", str(EXAMPLE), "--restart", "--push-to-hub", "me/r",
                                     "--delete-bundle-on-success"])
    assert result.exit_code == 0, result.output
    assert calls == [(EXAMPLE, {"restart": True, "push_to_hub": "me/r",
                                "delete_bundle_on_success": True})]
    assert "Adapter written" in result.output


def test_train_command_reports_failures_with_exit_1(monkeypatch, tmp_path):
    fake_training(monkeypatch, tmp_path, error="loss became non-finite at step 5")
    result = runner.invoke(cli.app, ["train", str(EXAMPLE)])
    assert result.exit_code == 1 and "non-finite at step 5" in result.output


def test_export_requires_merge(tmp_path):
    result = runner.invoke(cli.app, ["export", str(tmp_path), "-o", str(tmp_path / "m")])
    assert result.exit_code == 2 and "only --merge" in result.output
```

`tests/torch/test_cli_end_to_end.py`:

```python
def test_train_predict_eval_export_from_the_command_line(tiny_project, tmp_path):
    config = tiny_project(train={"epochs": 2})
    assert runner.invoke(cli.app, ["train", str(config)]).exit_code == 0
    adapter = config.parent / "out" / "final"
    result = runner.invoke(cli.app, ["predict", str(adapter), str(config.parent / "eval.csv"),
                                     "-o", str(tmp_path / "p.jsonl")])
    assert result.exit_code == 0 and "Wrote 16 predictions" in result.output
    result = runner.invoke(cli.app, ["eval", str(adapter), str(config.parent / "eval.csv"),
                                     "--output", str(tmp_path / "m.json")])
    assert result.exit_code == 0 and "exact match" in result.output.lower()
    assert json.loads((tmp_path / "m.json").read_text())["exact_match"] >= 0.0
    result = runner.invoke(cli.app, ["export", str(adapter), "--merge", "-o", str(tmp_path / "merged")])
    assert result.exit_code == 0 and (tmp_path / "merged" / "joint_head.safetensors").exists()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_cli.py tests/torch/test_cli_end_to_end.py tests/test_boundaries.py -v`
Expected: the new CLI tests FAIL with "No such command 'train'"; the boundary test FAILS until the new modules exist torch-free.

- [ ] **Step 3: Implement the commands**, keeping `import clef_tune.cli` torch-free.

- [ ] **Step 4: Update `README.md`**:
  - Status: training, evaluation, prediction and export are done; HF Jobs and the UI are next.
  - **Train on a GPU box:** `uv sync --extra train` (on Windows this installs CUDA 13.0 torch). Users installing from git with `uv pip` add `--torch-backend cu130`, or use their CUDA's backend.
  - **Kernels on Linux:** `pip install flash-linear-attention causal-conv1d` is optional but much faster.
  - Commands: `clef-tune validate cfg.yaml --bundle out/bundle`, `clef-tune train out/bundle/config.yaml`, `accelerate launch -m clef_tune train cfg.yaml` for multi-GPU DDP, `predict`, `eval`, `export --merge`.
  - The Docker image (`docker build -t clef-tune .`).
  - A Python `Predictor` example matching spec §3.
  - Calibration: what `calibration.json` holds, and the client formula for merged exports.

- [ ] **Step 5: Run the full suite** (`uv run ruff format . && uv run ruff check --fix . && uv run pytest -v`; all PASS) **and commit** (`feat: add train, eval, predict and export commands`).

---

### Task 17: Parity, vendored-file and opt-in GPU tests

**Files:**
- Create: `tests/torch/test_parity.py`, `tests/torch/test_fixtures_match_clef.py`, `tests/torch/test_gpu.py`

**Interfaces:**
- Consumes: `prompt.RecordTokenCounter` / `get_token_counter`, `clef_code.load_clef_code`, `train`, `Predictor`, `examples/support_tickets.csv`.
- Produces: tests only.

- [ ] **Step 1: Write the tests**

`tests/torch/test_parity.py` (`pytestmark = [pytest.mark.torch, pytest.mark.network]`). Verified on 2026-10-27: 33 records, 0 mismatches.

```python
STATES = ["hello world", "", "  leading and trailing  ", "Ünïcödé — “quotes” and emoji 🚀🔥",
          "中文测试，日本語のテスト", "line1\nline2\r\n\ttab",
          {"subject": "Card declined", "body": "Help!", "n": 3, "x": None, "nested": {"a": [1, 2.5, True]}},
          "x" * 3000, '{"already": "json"}', "<|im_start|> literal special", " sep "]
QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which team?",
                   "criteria": {"billing": "Money", "Technical": "Bugs 🐛", "NA": None}},
    "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
    "outage": {"type": "noul"},
    "spam": {"type": "noul", "instructions": "", "criteria": {"true": "Unsolicited", "false": "Legit"}},
}


@pytest.mark.parametrize("subset", [list(QUESTIONS), ["outage"], ["urgency", "department"]])
def test_token_counter_matches_clefs_encode_record(subset):
    revision = PINNED_REVISIONS["Cloudflare/clef"]
    tokenizer = AutoTokenizer.from_pretrained("Cloudflare/clef", revision=revision)
    jsm = load_clef_code(Path(__file__).parent / "fixtures")
    questions = Schema.from_dict({"questions": {q: QUESTIONS[q] for q in subset}}).questions
    counter = RecordTokenCounter(get_token_counter("Cloudflare/clef", revision), max_length=1024)
    for state in STATES:
        record = {"id": "r", "state": state, "questions": {q: v.to_clef() for q, v in questions.items()}}
        expected = len(jsm.encode_record(tokenizer, record, max_length=1024).input_ids)
        assert counter.count_records([(state, questions, 0)])[0].total == expected, repr(state)[:40]
```

`tests/torch/test_fixtures_match_clef.py` (`network`): for each vendored file, `hf_hub_download("Cloudflare/clef", name, revision=PINNED_REVISIONS["Cloudflare/clef"])` gives the same bytes as `tests/torch/fixtures/<name>`.

`tests/torch/test_gpu.py` (`pytestmark = [pytest.mark.torch, pytest.mark.gpu]`):
- Skip unless `torch.cuda.is_available()` and `torch.cuda.get_device_properties(0).total_memory >= 16e9`.
- The test writes a project from the first 50 rows of `examples/support_tickets.csv` with `examples/support_tickets.yaml`'s schema and data mapping.
- Config: `model: {id: Cloudflare/clef-flash}`, `eval_fraction: 0.2`, `train: {quantization: 4bit, mode: lora, epochs: 1, max_length: 1024, max_tokens_per_batch: 1024}`, `eval: {eval_every: 1000}`.
- Run `train` (downloads Clef-Flash, about 19 GB), then assert the adapter exists and every `final` metric is finite.
- Then `Predictor.load(adapter).predict(3 rows)` returns answers for all three questions.

- [ ] **Step 2: Run them**

Run: `uv run pytest tests/torch/test_parity.py tests/torch/test_fixtures_match_clef.py -v`
Expected: all PASS (network). `uv run pytest tests/torch/test_gpu.py -m gpu -v` SKIPS on machines below 16 GB, including the development laptop (12 GB); it runs on a ≥16 GB GPU.

- [ ] **Step 3: Full suite, lint and commit**

Run: `uv run ruff format . && uv run ruff check --fix . && uv run pytest -v`
Expected: all PASS; the GPU test is deselected by the default `-m 'not gpu and not hfjobs'`.

```bash
git add tests/torch
git commit -F - <<'EOF'
test: pin tokenization parity, vendored Clef files and an opt-in GPU run

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BydXoAAPdrHBxWFt1mckNw
EOF
```

---

## Spec coverage (Plan 2 portion)

| Spec section | Covered by |
|---|---|
| §2 import Clef's code from the pinned snapshot | Task 4 |
| §3 modules `clef_code`, `model`, `train`, `adapter`, `predictor`; CLI `train`/`eval`/`predict`/`export`; Python API | Tasks 4, 5, 11–16 |
| §3 extras (`train` + torchvision; kernels installed explicitly per roadmap item 2) | Tasks 1, 3 |
| §5.1 bf16 or 4-bit backbone, `lm_head`/embeddings unquantized, vision frozen, LoRA on language-model linears, head-only, fp32 head under bf16 autocast, separate LRs, gradient checkpointing | Tasks 5, 11 |
| §5.2 soft-target CE, balanced class weights with clipping, question weights, label smoothing | Tasks 6, 7 |
| §5.3 length-sorted token-budget batches, shuffled order, grad accumulation, AdamW, cosine, warmup, clip 1.0, Accelerate DDP | Tasks 6, 11 |
| §5.4 baseline + periodic + final eval, best checkpoint by `select_metric`, early stopping | Tasks 8, 11 |
| §5.5 calibration from the best checkpoint's cached eval logits | Tasks 8, 11 |
| §5.6 stdout + `metrics.jsonl`, trackers via Accelerate | Task 11 |
| §5.7 checkpoints every `save_every`, keep 2, config hash, auto-resume, mismatch error, cleanup on success | Tasks 10, 12 |
| §5.8 OOM advice, NaN stop, pre-flight (validation, memory fit, Hub access, kernels warning) | Tasks 11–13 |
| §6.1 adapter layout and card | Task 9 |
| §6.2 `predict` (SystemOne shape, calibrated, JSONL/CSV), `eval`, `export --merge` | Tasks 14–16 |
| §7 live ETA and projected cost after 20 steps | Task 11 |
| §8.3 `--delete-bundle-on-success`, private adapter repos | Tasks 12, 13 |
| §11.2 tokenization parity; §11.3 tiny-model integration; §11.5 `gpu` test | Tasks 4–17 |
| §12 Dockerfile; README | Tasks 3, 16 |
| §8 HF Jobs launcher, §9 UI | Plans 3, 4 |
