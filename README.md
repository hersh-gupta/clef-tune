# clef-tune

Fine-tune Cloudflare's [Clef](https://huggingface.co/Cloudflare/clef) decision models
(Clef 27B and Clef-Flash 9.4B) on your own labeled CSV or JSONL data.

> Status: data validation, estimates, training, evaluation, prediction and merged export are
> done. Hugging Face Jobs launching and the web UI are next (see `docs/superpowers/`).

## Install

```bash
uv pip install "clef-tune @ git+https://github.com/hersh-gupta/clef-tune"
```

## Quickstart

1. Describe your data and questions in a config. See `examples/support_tickets.yaml`:
   each CSV row supplies the state (`state_columns`) and one label column per question.
2. Validate. This catches bad labels, duplicate IDs, single-class questions and over-long
   rows before you spend money:

   ```bash
   clef-tune validate examples/support_tickets.yaml
   ```

3. Estimate time and cost on Hugging Face Jobs hardware, or on your own GPU:

   ```bash
   clef-tune estimate examples/support_tickets.yaml
   clef-tune estimate examples/support_tickets.yaml --gpu H100 --price 2.49
   ```

4. Write a portable bundle (normalized data plus a self-contained config) to train anywhere:

   ```bash
   clef-tune validate examples/support_tickets.yaml --bundle out/bundle
   ```

## Train on a GPU box

```bash
uv sync --extra train        # on Windows this installs CUDA 13.0 torch
```

If you install from git with `uv pip`, add `--torch-backend cu130` (or the backend for your
CUDA version):

```bash
uv pip install "clef-tune[train] @ git+https://github.com/hersh-gupta/clef-tune" --torch-backend cu130
```

On Linux, `pip install flash-linear-attention causal-conv1d` is optional but much faster:
without these kernels Clef's linear-attention layers fall back to a slow PyTorch path.

```bash
clef-tune validate cfg.yaml --bundle out/bundle      # normalized data + self-contained config
clef-tune train out/bundle/config.yaml               # resumes automatically from checkpoints
accelerate launch -m clef_tune train cfg.yaml        # multi-GPU DDP
clef-tune predict out/<run_name>/final new.csv -o preds.jsonl  # or preds.csv; --no-calibration
clef-tune eval out/<run_name>/final labeled.csv --output metrics.json
clef-tune export out/<run_name>/final --merge -o merged          # Cloudflare's release layout
```

`train` accepts `--restart` (ignore checkpoints), `--push-to-hub REPO` and
`--delete-bundle-on-success`. The adapter is written to `<output.dir>/final`; `output.dir`
defaults to `out/<run_name>` next to the config, so a bundle's run lands in
`out/bundle/out/<run_name>/final`. `eval` prints per-question metrics, exact match,
confusion matrices and ECE before and after calibration.
Exit codes: 0 ok, 1 training/data/Hub failure, 2 invalid config or arguments.

### Docker

```bash
docker build -t clef-tune .
docker run --gpus all -v "$PWD:/work" -w /work clef-tune train cfg.yaml
```

## Using an adapter from Python

```python
from clef_tune import Predictor

p = Predictor.load("out/<run_name>/final")  # local dir or Hub repo
p.predict([{"state": {"subject": "Site down", "body": "..."}}])  # SystemOne-shaped answers
p.systemone(request)  # same contract as Clef's systemone(), calibrated
p.model, p.processor  # raw ClefModel + processor (uncalibrated logits)
```

Questions default to the adapter's schema; calibration is applied unless `calibrated=False`.
`Predictor.load` also takes `device`, `quantization` and `base_model_path`; without
`base_model_path` it uses the base model directory recorded at training time, or downloads
the pinned base model when that directory does not exist on this machine.

## Calibration

Training fits one temperature per question on the eval set and stores them in the adapter's
`calibration.json` (`{question_id: T}`; a question not listed uses T = 1). `Predictor` and
`predict` apply `softmax(logits / T)` by default. SGLang serving of a merged export returns
**uncalibrated** probabilities; a client calibrates the served probabilities of each question's
options with

    p_i^(1/T) / sum_j p_j^(1/T)

Temperatures are fit on the same eval set used for checkpoint selection, so post-calibration
ECE is slightly optimistic.

## Data format

- **CSV/TSV**: one row per example. State comes from `data.state_columns` (several columns
  become a JSON object; one column is passed as plain text). Labels come from one column per
  question (`data.label_columns` maps question IDs to column names, defaulting to the same
  name). `choice` labels are option IDs, `score` labels are 0-based levels or the level text,
  and `noul` labels are true/false, yes/no, 1/0 or t/f. Empty cells mean "not labeled".
- **JSONL**: flat rows work like CSV. Rows with a `labels` object use Clef's native format,
  `{"id", "state", "questions"?, "labels", "images"?}`, and may carry their own questions.
  Labels may be soft: `{"billing": 0.7, "technical": 0.3}`.

Messages refer to spreadsheet rows for CSV (the header is row 1) and to line numbers for JSONL.

## Development

```bash
uv sync --extra dev
uv run pytest
```
