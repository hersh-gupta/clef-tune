# clef-tune

Fine-tune Cloudflare's [Clef](https://huggingface.co/Cloudflare/clef) decision models
(Clef 27B and Clef-Flash 9.4B) on your own labeled CSV or JSONL data.

> Status: the core is done: data validation, token statistics and cost/time estimates.
> Training, Hugging Face Jobs launching and the web UI are next (see `docs/superpowers/`).

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
