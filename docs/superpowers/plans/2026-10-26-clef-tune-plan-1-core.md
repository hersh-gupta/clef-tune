# clef-tune Plan 1: Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the torch-free core of clef-tune: load CSV/JSONL into Clef records, validate them with the real Clef tokenizer, estimate cost and time, compute metrics and calibration, write portable bundles, and expose `clef-tune validate` / `clef-tune estimate`.

**Architecture:** A pure-Python package (`clef_tune`) of small modules with one responsibility each. Data flows config → loader → normalized `Record`s → validation report → estimates/bundle. Nothing in this plan imports torch; a test enforces that boundary so the UI and a CPU Hugging Face Space can use all of it.

**Tech Stack:** Python 3.12 via `uv`, hatchling + hatch-vcs, pydantic 2, PyYAML, tokenizers, huggingface_hub 1.x, Typer + Rich, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-26-clef-finetuning-harness-design.md` (revision 2). Read `docs/superpowers/plans/2026-10-26-clef-tune-roadmap.md` too; its "Verified constraints and spec amendments" override the spec where they differ.

## Global Constraints

- Python `>=3.11`; development uses 3.12 (`.python-version` contains `3.12`).
- Distribution name `clef-tune`, import package `clef_tune`, console script `clef-tune`, repo `https://github.com/hersh-gupta/clef-tune`.
- Base dependency `huggingface_hub>=1.33,<2` (transformers 5.10.x requires hub < 2). The `train` extra pins `torch>=2.11,<2.12` and `transformers>=5.10.2,<5.11`.
- No module in this plan may import `torch`, `transformers` or `peft` (Task 11 adds a test that enforces this).
- Pinned model revisions: `Cloudflare/clef` → `a30162b99d68a481d8bd75d38df069a0cf83df40`; `Cloudflare/clef-flash` → `f0b7d3cd65296abfe459e6c228b618f5d8b26d2f`.
- Clef's prompt layout must match `joint_schema_model.py` `encode_record` byte for byte (Task 6 pins it with a golden string).
- Row numbers in user-facing messages: CSV/TSV use spreadsheet rows (the header is row 1, the first data row is row 2); JSONL uses 1-based line numbers.
- Code style: ruff, line length 100. Run `uv run ruff format . && uv run ruff check --fix .` before every commit.
- Commit messages carry no Claude attribution: no `Co-Authored-By` or session trailer.

## Review Focus

1. **Spreadsheet cells like `NA`, `N/A`, `None`, `null`** used as real option names or text must stay literal strings, never become missing values. Pinned by `test_na_like_values_stay_literal` (Task 5).
2. **YAML criteria keys `yes`, `no`, `on`, `off`** must stay strings; PyYAML's default YAML 1.1 rules would turn them into booleans. Pinned by `test_yaml_yes_no_keys_stay_strings` (Task 2).
3. **CSVs saved by Excel** (UTF-8 with BOM, or Windows-1252) must load with the right header names and characters. Pinned by `test_bom_header_and_cp1252_fallback` (Task 5).
4. **JSONL with blank lines or one malformed line** must load every valid line and report the bad line's number. Pinned by `test_jsonl_native_blank_and_malformed_lines` (Task 5).
5. **Relative paths** (data files, `schema_file`, image paths) resolve against the config or data file's directory, not the current working directory, including Windows-style backslashes. Pinned by `test_paths_resolve_against_config_dir` (Task 4) and `test_image_paths_resolve_relative_to_data_file` (Task 5).

---

## File Structure

```
pyproject.toml                    # Task 1: package metadata, extras, ruff, pytest
.python-version                   # Task 1
.gitignore                        # Task 1
.github/workflows/ci.yml          # Task 1: lint + tests on push/PR
README.md                         # Task 1 stub; Task 11 quickstart
clef_tune/__init__.py             # Task 1: __version__
clef_tune/__main__.py             # Task 11: python -m clef_tune
clef_tune/yamlio.py               # Task 2: YAML 1.2-boolean loader/dumper
clef_tune/schema.py               # Task 2: Question, Schema
clef_tune/labels.py               # Task 3: Label, parse_label
clef_tune/config.py               # Task 4: Config models, presets, paths, hashing
clef_tune/data.py                 # Task 5: file reading, normalization, split
clef_tune/prompt.py               # Task 6: Clef prompt layout, token counting, image tokens
clef_tune/validation.py           # Task 7: ValidationReport
clef_tune/metrics.py              # Task 8: per-question metrics, confusion matrices
clef_tune/calibration.py          # Task 9: temperature scaling, ECE
clef_tune/estimate.py             # Task 10: time/cost/memory estimates
clef_tune/bundle.py               # Task 11: portable training bundle
clef_tune/cli.py                  # Task 11: validate / estimate commands
scripts/make_example.py           # Task 11: deterministic example generator
examples/support_tickets.csv      # Task 11 (generated)
examples/support_tickets.yaml     # Task 11
tests/__init__.py                 # Task 1
tests/helpers.py                  # Task 5: CharCounter, SCHEMA, write_csv, make_config, ticket_rows
tests/test_*.py                   # one per module
```

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `.github/workflows/ci.yml`, `README.md`, `clef_tune/__init__.py`, `tests/__init__.py`
- Test: `tests/test_package.py`

**Interfaces:**
- Consumes: nothing.
- Produces: installable package `clef_tune` with `clef_tune.__version__: str`; `uv sync --extra dev` + `uv run pytest` workflow; pytest markers `torch`, `gpu`, `hfjobs`, `network`.

- [ ] **Step 1: Write the failing test**

`tests/__init__.py` is an empty file. `tests/test_package.py`:

```python
import clef_tune


def test_version_is_a_non_empty_string():
    assert isinstance(clef_tune.__version__, str)
    assert clef_tune.__version__
```

- [ ] **Step 2: Write `pyproject.toml`, `.python-version`, `.gitignore`**

`pyproject.toml`:

```toml
[build-system]
requires = ["hatchling>=1.25", "hatch-vcs>=0.4"]
build-backend = "hatchling.build"

[project]
name = "clef-tune"
dynamic = ["version"]
description = "Fine-tuning harness for Cloudflare's Clef decision models"
readme = "README.md"
license = "Apache-2.0"
requires-python = ">=3.11"
dependencies = [
  "huggingface_hub>=1.33,<2",
  "numpy>=2.0",
  "pillow>=10.0",
  "pydantic>=2.11",
  "pyyaml>=6.0",
  "rich>=13.0",
  "tokenizers>=0.22",
  "typer>=0.12",
]

[project.optional-dependencies]
ui = ["gradio>=6.0"]
train = [
  "torch>=2.11,<2.12",
  "transformers>=5.10.2,<5.11",
  "peft>=0.21",
  "accelerate>=1.10",
  "bitsandbytes>=0.45",
  "safetensors>=0.5",
]
dev = ["pytest>=8.0", "ruff>=0.6", "playwright>=1.45"]

[project.scripts]
clef-tune = "clef_tune.cli:app"

[project.urls]
Homepage = "https://github.com/hersh-gupta/clef-tune"
Source = "https://github.com/hersh-gupta/clef-tune"

[tool.hatch.version]
source = "vcs"
fallback-version = "0.0.0"

[tool.hatch.build.hooks.vcs]
version-file = "clef_tune/_version.py"

[tool.hatch.build.targets.wheel]
packages = ["clef_tune"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-m 'not gpu and not hfjobs'"
markers = [
  "torch: needs torch (tiny-model tests)",
  "gpu: needs a CUDA GPU and real model weights (opt-in)",
  "hfjobs: submits a real Hugging Face Job (opt-in, costs money)",
  "network: downloads from the Hugging Face Hub",
]
```

`.python-version`:

```
3.12
```

`.gitignore`:

```
.venv/
__pycache__/
*.egg-info/
dist/
build/
.pytest_cache/
.ruff_cache/
clef_tune/_version.py
out/
```

- [ ] **Step 3: Write `clef_tune/__init__.py` and a README stub**

`clef_tune/__init__.py`:

```python
"""clef-tune: fine-tuning harness for Cloudflare's Clef decision models."""

try:
    from clef_tune._version import __version__
except ImportError:  # source checkout that was never built or installed
    __version__ = "0.0.0"
```

`README.md`:

```markdown
# clef-tune

Fine-tune Cloudflare's [Clef](https://huggingface.co/Cloudflare/clef) decision models on your own
labeled CSV or JSONL data.

Status: under construction. See `docs/superpowers/` for the design and plans.
```

- [ ] **Step 4: Install and run the test**

Run: `uv sync --extra dev && uv run pytest tests/test_package.py -v`
Expected: `uv` creates `.venv` with Python 3.12, writes `uv.lock`, and the test PASSES.

- [ ] **Step 5: Add CI**

`.github/workflows/ci.yml`:

```yaml
name: ci
on:
  push:
  pull_request:
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0 # hatch-vcs derives the version from git history
      - uses: astral-sh/setup-uv@v6
      - run: uv sync --locked --extra dev
      - run: uv run ruff format --check .
      - run: uv run ruff check .
      - run: uv run pytest
```

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format . && uv run ruff check --fix .
git add pyproject.toml uv.lock .python-version .gitignore .github README.md clef_tune tests
git commit -F - <<'EOF'
chore: scaffold clef-tune package
EOF
```

---

### Task 2: YAML helpers and question schema

**Files:**
- Create: `clef_tune/yamlio.py`, `clef_tune/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `yamlio.load_yaml_text(text: str) -> Any`, `yamlio.load_yaml_file(path) -> Any`, `yamlio.dump_yaml(data) -> str`
  - `schema.QUESTION_TYPES`, `schema.NOUL_DEFAULT_CRITERIA`, `schema.SchemaError(ValueError)`, `schema.option_key(key) -> str`
  - `schema.Question(id: str, type: str, instructions: str | None, criteria: dict | list | None)` with `from_dict(question_id, data)`, `options() -> list[tuple[str, Any]]`, `option_ids() -> list[str]`, `to_clef() -> dict`
  - `schema.Schema(questions: dict[str, Question])` with `from_dict(data)`, `load(path)`, `save(path)`, `to_dict()`

- [ ] **Step 1: Write the failing tests**

`tests/test_schema.py`:

```python
import pytest

from clef_tune.schema import Question, Schema, SchemaError
from clef_tune.yamlio import load_yaml_text


def test_choice_options_are_sorted_like_clef():
    q = Question.from_dict(
        "dept", {"type": "choice", "criteria": {"technical": "Bugs", "billing": "Money"}}
    )
    assert q.options() == [("billing", "Money"), ("technical", "Bugs")]


def test_score_options_are_indices():
    q = Question.from_dict("urgency", {"type": "score", "criteria": ["Low", "High"]})
    assert q.option_ids() == ["0", "1"]
    assert q.options() == [("0", "Low"), ("1", "High")]


def test_noul_options_default_and_override():
    q = Question.from_dict("outage", {"type": "noul", "criteria": {"true": "Down"}})
    assert q.options() == [
        ("true", "Down"),
        ("false", "The proposition is false or the answer is no."),
    ]


@pytest.mark.parametrize(
    "data, message",
    [
        ({"type": "bogus"}, "type must be one of"),
        ({"type": "choice"}, "non-empty criteria mapping"),
        ({"type": "score", "criteria": []}, "non-empty criteria list"),
        ({"type": "noul", "criteria": {"maybe": "?"}}, "must be true/false"),
        ({"type": "noul", "extra": 1}, "unknown keys"),
        ({"type": "noul", "instructions": 5}, "instructions must be text"),
        ("noul", "must be a mapping"),
    ],
)
def test_invalid_questions(data, message):
    with pytest.raises(SchemaError, match=message):
        Question.from_dict("q", data)


def test_empty_schema_rejected():
    with pytest.raises(SchemaError, match="non-empty 'questions'"):
        Schema.from_dict({"questions": {}})


def test_yaml_yes_no_keys_stay_strings():
    data = load_yaml_text(
        "questions:\n"
        "  decision:\n"
        "    type: choice\n"
        "    criteria: {yes: Approve, no: Reject, on: Hold, off: Drop}\n"
    )
    question = Schema.from_dict(data).questions["decision"]
    assert question.option_ids() == ["no", "off", "on", "yes"]


def test_yaml_bool_and_int_keys_normalize():
    # Not `1:` next to `true:`: Python treats True == 1, so those two keys would collide.
    data = load_yaml_text("questions:\n  q:\n    type: choice\n    criteria: {true: T, 2: two}\n")
    assert Schema.from_dict(data).questions["q"].option_ids() == ["2", "true"]


def test_to_clef_round_trips_through_from_dict():
    q = Question.from_dict(
        "d", {"type": "choice", "instructions": "Which?", "criteria": {"a": "A", "b": None}}
    )
    assert Question.from_dict("d", q.to_clef()) == q


def test_save_and_load_preserve_order(tmp_path):
    schema = Schema.from_dict(
        {
            "questions": {
                "d": {"type": "choice", "criteria": {"yes": "Y", "no": "N"}},
                "u": {"type": "score", "criteria": ["a", "b"]},
            }
        }
    )
    path = tmp_path / "schema.yaml"
    schema.save(path)
    loaded = Schema.load(path)
    assert loaded == schema
    assert list(loaded.questions) == ["d", "u"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clef_tune.schema'`.

- [ ] **Step 3: Implement `clef_tune/yamlio.py`**

```python
"""YAML helpers using YAML 1.2 booleans: only true/false are booleans, so keys like
`yes`, `no`, `on` and `off` stay strings."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

_BOOL_TAG = "tag:yaml.org,2002:bool"


class _Loader(yaml.SafeLoader):
    pass


_Loader.yaml_implicit_resolvers = {
    first: [(tag, regexp) for tag, regexp in resolvers if tag != _BOOL_TAG]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_Loader.add_implicit_resolver(
    _BOOL_TAG,
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)


def load_yaml_text(text: str) -> Any:
    return yaml.load(text, Loader=_Loader)  # _Loader is a SafeLoader subclass


def load_yaml_file(path: str | Path) -> Any:
    return load_yaml_text(Path(path).read_text(encoding="utf-8-sig"))


def dump_yaml(data: Any) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
```

- [ ] **Step 4: Implement `clef_tune/schema.py`**

```python
"""Clef's typed questions (noul / choice / score) and the schema that groups them."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clef_tune.yamlio import dump_yaml, load_yaml_file

QUESTION_TYPES = ("noul", "choice", "score")
NOUL_DEFAULT_CRITERIA = {
    "true": "The proposition is true or the answer is yes.",
    "false": "The proposition is false or the answer is no.",
}


class SchemaError(ValueError):
    """A schema or question definition is invalid."""


def option_key(key: Any) -> str:
    """Normalize a criteria key the way the user meant it (YAML/JSON may give bools or ints)."""
    if key is True:
        return "true"
    if key is False:
        return "false"
    return str(key)


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    instructions: str | None = None
    criteria: Any = None  # dict[str, Any] for choice/noul, list[Any] for score

    @classmethod
    def from_dict(cls, question_id: Any, data: Any) -> Question:
        qid = str(question_id).strip()
        if not qid:
            raise SchemaError("question IDs must be non-empty")
        if not isinstance(data, dict):
            raise SchemaError(f"{qid}: question must be a mapping with a 'type'")
        unknown = set(data) - {"type", "instructions", "criteria"}
        if unknown:
            raise SchemaError(f"{qid}: unknown keys {sorted(unknown)}")
        qtype = data.get("type")
        if qtype not in QUESTION_TYPES:
            raise SchemaError(f"{qid}: type must be one of noul, choice, score (got {qtype!r})")
        instructions = data.get("instructions")
        if instructions is not None and not isinstance(instructions, str):
            raise SchemaError(f"{qid}: instructions must be text")
        raw = data.get("criteria")
        if qtype == "choice":
            if not isinstance(raw, dict) or not raw:
                raise SchemaError(f"{qid}: choice questions need a non-empty criteria mapping")
            criteria: Any = {option_key(k): v for k, v in raw.items()}
            if len(criteria) != len(raw):
                raise SchemaError(f"{qid}: criteria keys collide after normalization")
        elif qtype == "score":
            if not isinstance(raw, list) or not raw:
                raise SchemaError(f"{qid}: score questions need a non-empty criteria list")
            criteria = list(raw)
        elif raw is None:
            criteria = None
        elif not isinstance(raw, dict):
            raise SchemaError(f"{qid}: noul criteria must map true/false to descriptions")
        else:
            criteria = {option_key(k): v for k, v in raw.items()}
            bad = set(criteria) - {"true", "false"}
            if bad:
                raise SchemaError(f"{qid}: noul criteria keys must be true/false (got {sorted(bad)})")
        return cls(id=qid, type=qtype, instructions=instructions or None, criteria=criteria)

    def options(self) -> list[tuple[str, Any]]:
        """Allowed (option_id, description) pairs in Clef's order (mirrors question_options)."""
        if self.type == "noul":
            criteria = dict(NOUL_DEFAULT_CRITERIA)
            criteria.update(self.criteria or {})
            return [(key, criteria[key]) for key in ("true", "false")]
        if self.type == "choice":
            return sorted((str(key), value) for key, value in self.criteria.items())
        return [(str(index), value) for index, value in enumerate(self.criteria)]

    def option_ids(self) -> list[str]:
        return [option_id for option_id, _ in self.options()]

    def to_clef(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type}
        if self.instructions:
            out["instructions"] = self.instructions
        if self.criteria is not None:
            out["criteria"] = (
                dict(self.criteria) if isinstance(self.criteria, dict) else list(self.criteria)
            )
        return out


@dataclass(frozen=True)
class Schema:
    questions: dict[str, Question]  # insertion order is the prompt order

    @classmethod
    def from_dict(cls, data: Any) -> Schema:
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("questions"), dict)
            or not data["questions"]
        ):
            raise SchemaError("schema must have a non-empty 'questions' mapping")
        questions: dict[str, Question] = {}
        for question_id, question_data in data["questions"].items():
            question = Question.from_dict(question_id, question_data)
            if question.id in questions:
                raise SchemaError(f"duplicate question ID {question.id!r}")
            questions[question.id] = question
        return cls(questions)

    @classmethod
    def load(cls, path: str | Path) -> Schema:
        return cls.from_dict(load_yaml_file(path))

    def to_dict(self) -> dict[str, Any]:
        return {"questions": {qid: q.to_clef() for qid, q in self.questions.items()}}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(dump_yaml(self.to_dict()), encoding="utf-8")
```

Note: `sorted(...)` over `(key, value)` tuples orders by key only, because keys are unique. That matches Clef's `question_options`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_schema.py -v`
Expected: all PASS.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format . && uv run ruff check --fix .
git add clef_tune/yamlio.py clef_tune/schema.py tests/test_schema.py
git commit -F - <<'EOF'
feat: add question schema with YAML 1.2 boolean handling
EOF
```

---

### Task 3: Label parsing

**Files:**
- Create: `clef_tune/labels.py`
- Test: `tests/test_labels.py`

**Interfaces:**
- Consumes: `schema.Question`, `schema.option_key`.
- Produces:
  - `labels.Label(hard: str | None = None, soft: dict[str, float] | None = None)` with `distribution(option_ids) -> list[float]`, `top() -> str`, `to_json() -> str | dict[str, float]`
  - `labels.LabelError(ValueError)`, `labels.is_missing(raw) -> bool`
  - `labels.parse_label(question, raw) -> tuple[Label | None, str | None]`: `(None, None)` means the value is missing (question masked); the second item is a warning such as a case fix; raises `LabelError` for invalid values.

- [ ] **Step 1: Write the failing tests**

`tests/test_labels.py`:

```python
import math

import pytest

from clef_tune.labels import Label, LabelError, parse_label
from clef_tune.schema import Question

CHOICE = Question.from_dict(
    "dept", {"type": "choice", "criteria": {"billing": "", "technical": "", "NA": ""}}
)
SCORE = Question.from_dict("urgency", {"type": "score", "criteria": ["Low", "Medium", "High"]})
NOUL = Question.from_dict("outage", {"type": "noul"})


@pytest.mark.parametrize("raw", [None, "", "   ", math.nan])
def test_missing_values_mask_the_question(raw):
    assert parse_label(CHOICE, raw) == (None, None)


@pytest.mark.parametrize(
    "raw, expected",
    [("billing", "billing"), ("  technical ", "technical"), ("NA", "NA")],
)
def test_choice_exact(raw, expected):
    assert parse_label(CHOICE, raw) == (Label(hard=expected), None)


def test_choice_case_fix_warns():
    label, warning = parse_label(CHOICE, "Billing")
    assert label == Label(hard="billing")
    assert "matched option 'billing' by case" in warning


def test_choice_unknown_value():
    with pytest.raises(LabelError, match="'biling' is not an option"):
        parse_label(CHOICE, "biling")


@pytest.mark.parametrize(
    "raw, expected",
    [("2", "2"), (" 0 ", "0"), ("1.0", "1"), (2, "2"), (1.0, "1"), ("High", "2")],
)
def test_score_index_or_text(raw, expected):
    assert parse_label(SCORE, raw)[0] == Label(hard=expected)


def test_score_text_case_fix_warns():
    label, warning = parse_label(SCORE, "medium")
    assert label == Label(hard="1") and "by case" in warning


@pytest.mark.parametrize("raw", ["3", -1, True, "urgent", 1.5])
def test_score_invalid(raw):
    with pytest.raises(LabelError, match="not a score level"):
        parse_label(SCORE, raw)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("true", "true"), ("YES", "true"), ("1", "true"), ("t", "true"), (True, "true"),
        (1, "true"), ("1.0", "true"), ("false", "false"), ("No", "false"), ("0", "false"),
        ("F", "false"), (False, "false"), (0, "false"),
    ],
)
def test_noul_values(raw, expected):
    assert parse_label(NOUL, raw) == (Label(hard=expected), None)


def test_noul_invalid():
    with pytest.raises(LabelError, match="not true/false"):
        parse_label(NOUL, "maybe")


def test_soft_label():
    label, _ = parse_label(CHOICE, {"billing": 0.7, "technical": 0.3})
    assert label.soft == {"billing": 0.7, "technical": 0.3}
    assert label.distribution(CHOICE.option_ids()) == [0.0, 0.7, 0.3]
    assert label.top() == "billing"
    assert label.to_json() == {"billing": 0.7, "technical": 0.3}


def test_soft_label_score_int_keys():
    label, _ = parse_label(SCORE, {0: 0.5, 2: 0.5})
    assert label.soft == {"0": 0.5, "2": 0.5}


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"billing": 0.5}, "sum to 0.5000"),
        ({"bogus": 1.0}, "'bogus' is not an option"),
        ({"billing": -0.2, "technical": 1.2}, "non-negative number"),
        ({"billing": math.nan, "technical": 1.0}, "non-negative number"),
        ({"billing": "x"}, "non-negative number"),
    ],
)
def test_soft_label_invalid(raw, message):
    with pytest.raises(LabelError, match=message):
        parse_label(CHOICE, raw)


def test_hard_label_distribution_and_json():
    label = Label(hard="technical")
    assert label.distribution(CHOICE.option_ids()) == [0.0, 0.0, 1.0]
    assert label.to_json() == "technical"
```

Note: `CHOICE.option_ids()` is `["NA", "billing", "technical"]`, because uppercase sorts before lowercase.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clef_tune.labels'`.

- [ ] **Step 3: Implement `clef_tune/labels.py`**

```python
"""Parse raw label values (CSV cells, JSON values) into option IDs or soft distributions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from clef_tune.schema import Question, option_key

NOUL_TRUE = {"true", "yes", "1", "t"}
NOUL_FALSE = {"false", "no", "0", "f"}
SOFT_SUM_TOLERANCE = 1e-3


class LabelError(ValueError):
    """A label value is present but cannot be interpreted for its question."""


@dataclass(frozen=True)
class Label:
    hard: str | None = None
    soft: dict[str, float] | None = None

    def distribution(self, option_ids: list[str]) -> list[float]:
        if self.soft is not None:
            return [float(self.soft.get(option, 0.0)) for option in option_ids]
        return [1.0 if option == self.hard else 0.0 for option in option_ids]

    def top(self) -> str:
        """The single most likely option: the hard label, or the soft argmax."""
        if self.hard is not None:
            return self.hard
        return max(self.soft, key=self.soft.__getitem__)

    def to_json(self) -> str | dict[str, float]:
        return self.hard if self.hard is not None else dict(self.soft)


def is_missing(raw: Any) -> bool:
    if raw is None:
        return True
    if isinstance(raw, float) and math.isnan(raw):
        return True
    return isinstance(raw, str) and raw.strip() == ""


def parse_label(question: Question, raw: Any) -> tuple[Label | None, str | None]:
    """Return ``(label, warning)``. ``label`` is None when the value is missing.

    Raises LabelError when the value is present but invalid for the question.
    """
    if is_missing(raw):
        return None, None
    if isinstance(raw, dict):
        return _parse_soft(question, raw), None
    if question.type == "choice":
        return _parse_choice(question, raw)
    if question.type == "score":
        return _parse_score(question, raw)
    return _parse_noul(raw), None


def _parse_choice(question: Question, raw: Any) -> tuple[Label, str | None]:
    value = raw.strip() if isinstance(raw, str) else option_key(raw)
    options = question.option_ids()
    if value in options:
        return Label(hard=value), None
    folded = [option for option in options if option.casefold() == value.casefold()]
    if len(folded) == 1:
        return Label(hard=folded[0]), f"'{value}' matched option '{folded[0]}' by case"
    raise LabelError(f"'{value}' is not an option (options: {', '.join(options)})")


def _parse_score(question: Question, raw: Any) -> tuple[Label, str | None]:
    levels = len(question.criteria)
    index: int | None = None
    warning = None
    if isinstance(raw, bool):
        index = None
    elif isinstance(raw, int | float):
        if float(raw).is_integer():
            index = int(raw)
    else:
        text = str(raw).strip()
        try:
            number: float | None = float(text)
        except ValueError:
            number = None
        if number is not None and number.is_integer():
            index = int(number)
        elif number is None:
            texts = [str(criterion).strip() for criterion in question.criteria]
            if text in texts:
                index = texts.index(text)
            else:
                folded = [i for i, t in enumerate(texts) if t.casefold() == text.casefold()]
                if len(folded) == 1:
                    index = folded[0]
                    warning = f"'{text}' matched level '{texts[index]}' by case"
    if index is None or not 0 <= index < levels:
        raise LabelError(
            f"{raw!r} is not a score level (expected 0..{levels - 1} or a criteria text)"
        )
    return Label(hard=str(index)), warning


def _parse_noul(raw: Any) -> Label:
    if isinstance(raw, bool):
        return Label(hard="true" if raw else "false")
    if isinstance(raw, int | float) and raw in (0, 1):
        return Label(hard="true" if raw == 1 else "false")
    text = str(raw).strip().casefold()
    try:  # spreadsheets sometimes write 1.0 / 0.0
        number = float(text)
        if number in (0.0, 1.0):
            text = "1" if number == 1.0 else "0"
    except ValueError:
        pass
    if text in NOUL_TRUE:
        return Label(hard="true")
    if text in NOUL_FALSE:
        return Label(hard="false")
    raise LabelError(f"{raw!r} is not true/false (use true/false, yes/no, 1/0 or t/f)")


def _parse_soft(question: Question, raw: dict[Any, Any]) -> Label:
    options = set(question.option_ids())
    distribution: dict[str, float] = {}
    for key, value in raw.items():
        option = option_key(key)
        if option not in options:
            raise LabelError(f"soft label key '{option}' is not an option")
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or value < 0
        ):
            raise LabelError(f"soft label for '{option}' must be a non-negative number")
        distribution[option] = float(value)
    total = sum(distribution.values())
    if abs(total - 1.0) > SOFT_SUM_TOLERANCE:
        raise LabelError(f"soft label probabilities sum to {total:.4f}, not 1")
    return Label(soft=distribution)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_labels.py -v`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check --fix .
git add clef_tune/labels.py tests/test_labels.py
git commit -F - <<'EOF'
feat: parse choice/score/noul and soft labels
EOF
```

---

### Task 4: Training configuration

**Files:**
- Create: `clef_tune/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `schema.Schema`, `yamlio.load_yaml_file`, `yamlio.dump_yaml`.
- Produces:
  - `config.PINNED_REVISIONS: dict[str, str]`, `config.PRESETS: dict[str, dict[str, Any]]` (keys `quick`, `balanced`, `thorough`)
  - Models `ModelConfig(id, revision, local_path)` with `resolved_revision() -> str`; `DataConfig(train, eval, eval_fraction, id_column, state_columns, label_columns, image_columns)`; `LoraConfig(r, alpha, dropout)`; `TrainSection(...)`; `EvalSection(eval_every, select_metric, early_stopping_patience, calibrate)`; `OutputSection(dir, save_every, push_to_hub, report_to)`
  - `Config(run_name, model, data, schema_def [alias "schema"], schema_file, train, eval, output)` with `base_dir`, `with_base_dir(path) -> Config`, `resolve_path(value: str) -> Path`, `get_schema() -> Schema | None`, `output_dir() -> Path`, `to_dict()`, `to_yaml()`, `save(path)`, `with_preset(name) -> Config`
  - `load_config(path) -> Config`, `file_digest(path) -> str`, `config_hash(config, files=()) -> str` (16 hex chars)

- [ ] **Step 1: Write the failing tests**

`tests/test_config.py`:

```python
import pytest
from pydantic import ValidationError

from clef_tune.config import PINNED_REVISIONS, Config, config_hash, load_config

MINIMAL = """\
run_name: demo
data:
  train: data/train.csv
schema:
  questions:
    outage: {type: noul}
"""


def write(directory, text, name="config.yaml"):
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    assert cfg.train.mode == "lora"
    assert cfg.train.lora.r == 16 and cfg.train.lora.alpha == 32
    assert cfg.train.max_length == 4096
    assert cfg.train.class_weighting == "none"
    assert cfg.eval.select_metric == "mean_macro_f1" and cfg.eval.calibrate is True
    assert cfg.model.resolved_revision() == PINNED_REVISIONS["Cloudflare/clef"]


def test_unknown_model_revision_defaults_to_main(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL + "model: {id: me/custom-clef}\n"))
    assert cfg.model.resolved_revision() == "main"


def test_paths_resolve_against_config_dir(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    cfg = load_config(write(project, MINIMAL.replace("data/train.csv", "data\\train.csv")))
    monkeypatch.chdir(tmp_path)  # the working directory must not matter
    assert cfg.resolve_path(cfg.data.train) == (project / "data" / "train.csv").resolve()
    assert cfg.output_dir() == (project / "out" / "demo").resolve()


def test_typo_rejected(tmp_path):
    with pytest.raises(ValidationError, match="epoch"):
        load_config(write(tmp_path, MINIMAL + "train:\n  epoch: 3\n"))


@pytest.mark.parametrize("name", ["bad name", "", "-leading-dash", "x" * 65])
def test_run_name_validated(tmp_path, name):
    with pytest.raises(ValidationError):
        Config.model_validate({"run_name": name, "data": {"train": "t.csv"}})


def test_schema_and_schema_file_conflict(tmp_path):
    with pytest.raises(ValidationError, match="not both"):
        load_config(write(tmp_path, MINIMAL + "schema_file: s.yaml\n"))


def test_schema_file_loads_relative_to_config(tmp_path):
    write(tmp_path, "questions:\n  outage: {type: noul}\n", "schemas/s.yaml")
    cfg = load_config(
        write(tmp_path, "run_name: demo\ndata: {train: t.csv}\nschema_file: schemas/s.yaml\n")
    )
    assert list(cfg.get_schema().questions) == ["outage"]


def test_no_schema_is_allowed(tmp_path):
    cfg = load_config(write(tmp_path, "run_name: demo\ndata: {train: t.jsonl}\n"))
    assert cfg.get_schema() is None


def test_invalid_schema_fails_fast(tmp_path):
    with pytest.raises(ValidationError, match="type must be one of"):
        load_config(write(tmp_path, MINIMAL.replace("noul", "bogus")))


def test_presets(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    quick = cfg.with_preset("quick")
    assert (quick.train.mode, quick.train.epochs, quick.train.max_length) == ("head_only", 1, 2048)
    thorough = cfg.with_preset("thorough")
    assert (thorough.train.lora.r, thorough.train.epochs) == (32, 5)
    assert thorough.eval.early_stopping_patience == 3
    assert thorough.base_dir == cfg.base_dir
    assert cfg.train.epochs == 3  # the original is unchanged


def test_yaml_round_trip(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    again = load_config(write(tmp_path, cfg.to_yaml(), "again.yaml"))
    assert again.to_dict() == cfg.to_dict()
    assert "schema" in cfg.to_dict() and "schema_def" not in cfg.to_dict()


def test_hash_ignores_run_name_and_output_but_not_training(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    base = config_hash(cfg)
    renamed = Config.model_validate(
        {**cfg.to_dict(), "run_name": "other", "output": {"dir": "elsewhere"}}
    )
    assert config_hash(renamed) == base
    changed = Config.model_validate({**cfg.to_dict(), "train": {"lora_lr": 3e-4}})
    assert config_hash(changed) != base
    assert len(base) == 16


def test_hash_includes_file_contents(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    data_file = tmp_path / "train.jsonl"
    data_file.write_text("a", encoding="utf-8")
    first = config_hash(cfg, [data_file])
    data_file.write_text("b", encoding="utf-8")
    assert config_hash(cfg, [data_file]) != first
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clef_tune.config'`.

- [ ] **Step 3: Implement `clef_tune/config.py`**

```python
"""Training configuration (config.yaml): models, presets, path resolution and hashing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from clef_tune.schema import Schema
from clef_tune.yamlio import dump_yaml, load_yaml_file

PINNED_REVISIONS = {
    "Cloudflare/clef": "a30162b99d68a481d8bd75d38df069a0cf83df40",
    "Cloudflare/clef-flash": "f0b7d3cd65296abfe459e6c228b618f5d8b26d2f",
}

PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "train.mode": "head_only",
        "train.epochs": 1,
        "train.max_length": 2048,
        "eval.early_stopping_patience": None,
    },
    "balanced": {
        "train.mode": "lora",
        "train.lora.r": 16,
        "train.lora.alpha": 32,
        "train.epochs": 3,
        "train.max_length": 4096,
        "eval.early_stopping_patience": None,
    },
    "thorough": {
        "train.mode": "lora",
        "train.lora.r": 32,
        "train.lora.alpha": 64,
        "train.epochs": 5,
        "train.max_length": 4096,
        "eval.early_stopping_patience": 3,
    },
}


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(_Section):
    id: str = "Cloudflare/clef"
    revision: str | None = None
    local_path: str | None = None

    def resolved_revision(self) -> str:
        return self.revision or PINNED_REVISIONS.get(self.id, "main")


class DataConfig(_Section):
    train: str
    eval: str | None = None
    eval_fraction: float = Field(0.1, gt=0, lt=1)
    id_column: str | None = None
    state_columns: list[str] | None = None  # None: every column not used for id/labels/images
    label_columns: dict[str, str] = Field(default_factory=dict)
    image_columns: list[str] = Field(default_factory=list)


class LoraConfig(_Section):
    r: int = Field(16, ge=1)
    alpha: int = Field(32, ge=1)
    dropout: float = Field(0.05, ge=0, lt=1)


class TrainSection(_Section):
    mode: Literal["lora", "head_only"] = "lora"
    quantization: Literal["none", "4bit"] = "none"
    lora: LoraConfig = Field(default_factory=LoraConfig)
    lora_lr: float = Field(1e-4, gt=0)
    head_lr: float = Field(5e-5, gt=0)
    epochs: int = Field(3, ge=1)
    max_length: int = Field(4096, ge=256, le=16384)
    max_tokens_per_batch: int = Field(16384, ge=256)
    grad_accum_steps: int = Field(4, ge=1)
    warmup_ratio: float = Field(0.03, ge=0, lt=1)
    weight_decay: float = Field(0.0, ge=0)
    label_smoothing: float = Field(0.0, ge=0, lt=1)
    class_weighting: Literal["none", "balanced"] = "none"
    question_weights: dict[str, float] = Field(default_factory=dict)
    gradient_checkpointing: bool = True
    seed: int = 42


class EvalSection(_Section):
    eval_every: int = Field(200, ge=1)
    select_metric: Literal["mean_macro_f1", "exact_match", "mean_log_loss"] = "mean_macro_f1"
    early_stopping_patience: int | None = Field(None, ge=1)
    calibrate: bool = True


class OutputSection(_Section):
    dir: str | None = None  # default: out/<run_name> next to the config
    save_every: int = Field(200, ge=1)
    push_to_hub: str | None = None
    report_to: list[Literal["tensorboard", "wandb"]] = Field(default_factory=list)


class Config(_Section):
    model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

    run_name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig
    schema_def: dict[str, Any] | None = Field(default=None, alias="schema")
    schema_file: str | None = None
    train: TrainSection = Field(default_factory=TrainSection)
    eval: EvalSection = Field(default_factory=EvalSection)
    output: OutputSection = Field(default_factory=OutputSection)
    _base_dir: Path = PrivateAttr(default_factory=Path.cwd)

    @model_validator(mode="after")
    def _one_schema_source(self) -> Config:
        if self.schema_def is not None and self.schema_file is not None:
            raise ValueError("set either 'schema' or 'schema_file', not both")
        if self.schema_def is not None:
            Schema.from_dict(self.schema_def)  # fail fast with a clear message
        return self

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def with_base_dir(self, base_dir: str | Path) -> Config:
        self._base_dir = Path(base_dir).resolve()
        return self

    def resolve_path(self, value: str) -> Path:
        """Resolve a config path against the config's directory (accepts Windows separators)."""
        path = Path(value.replace("\\", "/"))
        return path if path.is_absolute() else self._base_dir / path

    def get_schema(self) -> Schema | None:
        if self.schema_def is not None:
            return Schema.from_dict(self.schema_def)
        if self.schema_file is not None:
            return Schema.load(self.resolve_path(self.schema_file))
        return None

    def output_dir(self) -> Path:
        return self.resolve_path(self.output.dir or f"out/{self.run_name}")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)

    def to_yaml(self) -> str:
        return dump_yaml(self.to_dict())

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_yaml(), encoding="utf-8")

    def with_preset(self, name: str) -> Config:
        if name not in PRESETS:
            raise ValueError(f"unknown preset {name!r} (choose from {', '.join(PRESETS)})")
        data = self.to_dict()
        for dotted, value in PRESETS[name].items():
            *parents, leaf = dotted.split(".")
            section = data
            for parent in parents:
                section = section.setdefault(parent, {})
            section[leaf] = value
        return Config.model_validate(data).with_base_dir(self._base_dir)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    data = load_yaml_file(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: config must be a YAML mapping")
    return Config.model_validate(data).with_base_dir(path.parent)


def file_digest(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def config_hash(config: Config, files: Iterable[str | Path] = ()) -> str:
    """Hash of everything that changes training results (not run name, output, or weight paths)."""
    payload = config.to_dict()
    payload.pop("run_name", None)
    payload.pop("output", None)
    payload.pop("schema_file", None)
    payload["model"]["revision"] = config.model.resolved_revision()
    payload["model"].pop("local_path", None)
    schema = config.get_schema()
    payload["schema"] = schema.to_dict() if schema else None
    payload["files"] = sorted(file_digest(path) for path in files)
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check --fix .
git add clef_tune/config.py tests/test_config.py
git commit -F - <<'EOF'
feat: add training config with presets, path resolution and hashing
EOF
```

---

### Task 5: Data loading, normalization and split

**Files:**
- Create: `clef_tune/data.py`, `tests/helpers.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `config.Config`, `config.DataConfig`, `labels.Label`, `labels.LabelError`, `labels.is_missing`, `labels.parse_label`, `schema.Question`, `schema.Schema`, `schema.SchemaError`.
- Produces:
  - `data.DataError(ValueError)`: the file can't be read at all (missing, empty, unsupported type, duplicate headers).
  - `data.Issue(level: "error" | "warning", message: str, rows: list[int] = [], question: str | None = None)`; `str(issue)` renders `"<question>: <message> (rows 2, 5)"`.
  - `data.Record(id, state, questions: dict[str, Question], labels: dict[str, Label], images: list[str], source_row: int)` with `to_json() -> dict`.
  - `data.LoadResult(records: list[Record], issues: list[Issue], columns: list[str])`.
  - `data.read_rows(path) -> tuple[list[tuple[int, dict]], list[str], list[Issue]]`.
  - `data.normalize_rows(rows, columns, data_cfg, schema, data_dir) -> LoadResult`.
  - `data.load_file(config, path_value: str) -> LoadResult`.
  - `data.split_records(records, eval_fraction, seed) -> tuple[list[Record], list[Record]]`.
  - `tests/helpers.py`: `SCHEMA`, `CharCounter`, `write_csv(path, rows)`, `make_config(base_dir, schema=SCHEMA, **sections)`, `ticket_rows(n, start=1)`.

- [ ] **Step 1: Write the test helpers**

`tests/helpers.py`:

```python
"""Shared test helpers. Import explicitly: `from tests.helpers import ...`."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from clef_tune.config import Config

SCHEMA = {
    "questions": {
        "department": {
            "type": "choice",
            "criteria": {"billing": "Money", "technical": "Bugs", "NA": "Not applicable"},
        },
        "urgency": {"type": "score", "criteria": ["Low", "Medium", "High"]},
        "outage": {"type": "noul"},
    }
}


class CharCounter:
    """Token counter stand-in: one token per character."""

    def count(self, text: str) -> int:
        return len(text)

    def count_many(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def make_config(base_dir: Path, schema: dict | None = SCHEMA, **sections: Any) -> Config:
    data: dict[str, Any] = {"run_name": "test", **sections}
    if schema is not None:
        data["schema"] = schema
    return Config.model_validate(data).with_base_dir(base_dir)


def ticket_rows(n: int, start: int = 1) -> list[dict[str, str]]:
    """Labeled rows: departments alternate, urgency cycles 0-2, every 4th row is an outage."""
    departments = ["billing", "technical"]
    return [
        {
            "id": str(i),
            "subject": f"subject {i}",
            "body": f"body text {i}",
            "department": departments[i % 2],
            "urgency": str(i % 3),
            "outage": "yes" if i % 4 == 0 else "no",
        }
        for i in range(start, start + n)
    ]
```

- [ ] **Step 2: Write the failing tests**

`tests/test_data.py`:

```python
import json

from clef_tune.data import Record, load_file, split_records
from clef_tune.labels import Label
from tests.helpers import make_config, ticket_rows, write_csv

HEADER = "id,subject,body,department,urgency,outage\n"


def load_text(tmp_path, text, name="train.csv", encoding="utf-8", **data):
    (tmp_path / name).write_bytes(text.encode(encoding))
    cfg = make_config(tmp_path, data={"train": name, "id_column": "id", **data})
    return load_file(cfg, name)


def test_flat_csv_basic(tmp_path):
    result = load_text(
        tmp_path,
        HEADER + "1,Card declined,Help,billing,2,yes\n2,Site down,Now,technical,High,no\n",
        state_columns=["subject", "body"],
    )
    assert [i for i in result.issues if i.level == "error"] == []
    first, second = result.records
    assert first.id == "1" and first.source_row == 2
    assert first.state == {"subject": "Card declined", "body": "Help"}
    assert first.labels == {
        "department": Label(hard="billing"),
        "urgency": Label(hard="2"),
        "outage": Label(hard="true"),
    }
    assert second.labels["urgency"] == Label(hard="2")
    assert second.labels["outage"] == Label(hard="false")
    assert list(first.questions) == ["department", "urgency", "outage"]


def test_single_state_column_is_raw_string(tmp_path):
    result = load_text(tmp_path, HEADER + "1,Hi,There,billing,0,no\n", state_columns=["body"])
    assert result.records[0].state == "There"


def test_default_state_columns_are_the_unused_columns(tmp_path):
    result = load_text(tmp_path, HEADER + "1,Hi,There,billing,0,no\n")
    assert result.records[0].state == {"subject": "Hi", "body": "There"}


def test_na_like_values_stay_literal(tmp_path):
    result = load_text(tmp_path, HEADER + "1,N/A,None,NA,0,no\n2,null,NaN,billing,1,no\n")
    first, second = result.records
    assert first.labels["department"] == Label(hard="NA")
    assert first.state == {"subject": "N/A", "body": "None"}
    assert second.state == {"subject": "null", "body": "NaN"}


def test_empty_label_masks_question_and_unlabeled_rows_are_dropped(tmp_path):
    result = load_text(tmp_path, HEADER + "1,a,b,billing,,\n2,c,d,,,\n")
    assert len(result.records) == 1
    assert set(result.records[0].labels) == {"department"}
    dropped = [i for i in result.issues if "no labels" in i.message]
    assert dropped and dropped[0].rows == [3] and dropped[0].level == "warning"


def test_unknown_label_values_are_grouped_with_rows(tmp_path):
    result = load_text(tmp_path, HEADER + "1,a,b,biling,0,no\n2,c,d,biling,0,no\n")
    errors = [i for i in result.issues if i.level == "error"]
    assert len(errors) == 1
    assert errors[0].question == "department" and errors[0].rows == [2, 3]
    assert "'biling' is not an option" in str(errors[0])
    assert "(rows 2, 3)" in str(errors[0])


def test_case_fix_warning(tmp_path):
    result = load_text(tmp_path, HEADER + "1,a,b,Billing,0,no\n")
    assert result.records[0].labels["department"] == Label(hard="billing")
    assert any(i.level == "warning" and "by case" in i.message for i in result.issues)


def test_missing_label_column_suggests_close_match(tmp_path):
    text = "id,subject,dept_label,urgency,outage\n1,a,billing,0,no\n"
    result = load_text(tmp_path, text, label_columns={"department": "dept_labl"})
    errors = [str(i) for i in result.issues if i.level == "error"]
    assert any("label column 'dept_labl' not found" in e and "dept_label" in e for e in errors)


def test_header_whitespace_is_stripped(tmp_path):
    text = "id, subject ,body,department ,urgency,outage\n1,a,b,billing,0,no\n"
    result = load_text(tmp_path, text)
    assert result.records[0].labels["department"] == Label(hard="billing")
    assert "subject" in result.columns


def test_wrong_field_count_reports_row(tmp_path):
    result = load_text(tmp_path, HEADER + "1,a,b,billing,0,no\n2,a,billing\n")
    errors = [i for i in result.issues if i.level == "error"]
    assert errors[0].rows == [3] and "fields" in errors[0].message


def test_bom_header_and_cp1252_fallback(tmp_path):
    bom = load_text(tmp_path, "\ufeff" + HEADER + "1,a,b,billing,0,no\n")
    assert bom.columns[0] == "id"
    assert bom.records[0].id == "1"
    legacy = load_text(tmp_path, HEADER + "1,café,b,billing,0,no\n", encoding="cp1252")
    assert legacy.records[0].state["subject"] == "café"
    assert any("Windows-1252" in i.message for i in legacy.issues)


def test_jsonl_native_blank_and_malformed_lines(tmp_path):
    lines = [
        json.dumps({"id": "a", "state": {"text": "hi"}, "labels": {"outage": "yes"}}),
        "",
        "{not json",
        json.dumps(
            {
                "id": "b",
                "state": "free text",
                "questions": {"spam": {"type": "noul", "instructions": "Spam?"}},
                "labels": {"spam": False},
            }
        ),
        "",
    ]
    (tmp_path / "train.jsonl").write_text("\n".join(lines), encoding="utf-8")
    result = load_file(make_config(tmp_path, data={"train": "train.jsonl"}), "train.jsonl")
    assert [r.id for r in result.records] == ["a", "b"]
    assert result.records[0].labels == {"outage": Label(hard="true")}
    assert list(result.records[1].questions) == ["spam"]
    assert result.records[1].source_row == 4
    errors = [i for i in result.issues if i.level == "error"]
    assert len(errors) == 1 and errors[0].rows == [3] and "invalid JSON" in errors[0].message


def test_jsonl_soft_labels_and_unknown_question(tmp_path):
    lines = [
        json.dumps({"state": "x", "labels": {"department": {"billing": 0.6, "technical": 0.4}}}),
        json.dumps({"state": "y", "labels": {"bogus": "yes", "outage": "no"}}),
    ]
    (tmp_path / "train.jsonl").write_text("\n".join(lines), encoding="utf-8")
    result = load_file(make_config(tmp_path, data={"train": "train.jsonl"}), "train.jsonl")
    assert result.records[0].labels["department"].soft == {"billing": 0.6, "technical": 0.4}
    assert result.records[0].id == "row-1"
    errors = [i for i in result.issues if i.level == "error"]
    assert errors[0].question == "bogus" and errors[0].rows == [2]


def test_flat_rows_without_schema_error(tmp_path):
    write_csv(tmp_path / "train.csv", ticket_rows(2))
    cfg = make_config(tmp_path, schema=None, data={"train": "train.csv"})
    result = load_file(cfg, "train.csv")
    assert result.records == []
    assert any("need a schema" in i.message for i in result.issues)


def test_image_paths_resolve_relative_to_data_file(tmp_path):
    text = "id,body,receipt,department,urgency,outage\n1,b,imgs\\a.png,billing,0,no\n"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.csv").write_text(text, encoding="utf-8")
    cfg = make_config(
        tmp_path,
        data={"train": "data/train.csv", "id_column": "id", "image_columns": ["receipt"]},
    )
    record = load_file(cfg, "data/train.csv").records[0]
    assert record.images == [str((data_dir / "imgs" / "a.png").resolve())]
    assert record.state == "b"


def test_record_to_json(tmp_path):
    record = load_text(tmp_path, HEADER + "1,a,b,billing,2,yes\n").records[0]
    data = record.to_json()
    assert data["labels"] == {"department": "billing", "urgency": "2", "outage": "true"}
    assert data["questions"]["urgency"] == {"type": "score", "criteria": ["Low", "Medium", "High"]}
    assert "images" not in data


def _records(departments):
    return [
        Record(str(i), "s", {}, {"department": Label(hard=d)}, [], i + 2)
        for i, d in enumerate(departments)
    ]


def test_split_is_stratified_and_deterministic():
    records = _records(["billing"] * 80 + ["technical"] * 20)
    train, evals = split_records(records, 0.1, seed=42)
    assert len(evals) == 10
    assert sum(r.labels["department"].hard == "technical" for r in evals) == 2
    assert [r.id for r in split_records(records, 0.1, seed=42)[1]] == [r.id for r in evals]
    assert [r.id for r in split_records(records, 0.1, seed=7)[1]] != [r.id for r in evals]
    assert len(train) + len(evals) == 100
    assert [r.source_row for r in train] == sorted(r.source_row for r in train)


def test_split_keeps_singletons_in_train_but_always_makes_an_eval_row():
    records = _records(["billing", "technical", "NA"])
    train, evals = split_records(records, 0.1, seed=1)
    assert len(evals) == 1 and len(train) == 2
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clef_tune.data'`.

- [ ] **Step 4: Implement `clef_tune/data.py`**

```python
"""Load CSV/TSV/JSONL files into normalized records, and split train/eval.

Row numbers in messages are spreadsheet rows for CSV/TSV (the header is row 1) and line
numbers for JSONL.
"""

from __future__ import annotations

import csv
import difflib
import io
import json
import random
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from clef_tune.config import Config, DataConfig
from clef_tune.labels import Label, LabelError, is_missing, parse_label
from clef_tune.schema import Question, Schema, SchemaError

TABLE_DELIMITERS = {".csv": ",", ".tsv": "\t"}
JSONL_SUFFIXES = {".jsonl", ".ndjson"}
MAX_LISTED_ROWS = 20
URL_PREFIXES = ("http://", "https://")

Row = tuple[int, dict[str, Any]]
Note = Callable[[str, str | None, str, int], None]


class DataError(ValueError):
    """A data file cannot be read at all."""


@dataclass
class Issue:
    level: Literal["error", "warning"]
    message: str
    rows: list[int] = field(default_factory=list)
    question: str | None = None

    def __str__(self) -> str:
        where = ""
        if self.rows:
            shown = ", ".join(str(row) for row in self.rows[:MAX_LISTED_ROWS])
            extra = len(self.rows) - MAX_LISTED_ROWS
            more = f" +{extra} more" if extra > 0 else ""
            where = f" (rows {shown}{more})"
        prefix = f"{self.question}: " if self.question else ""
        return f"{prefix}{self.message}{where}"


@dataclass
class Record:
    id: str
    state: Any
    questions: dict[str, Question]
    labels: dict[str, Label]
    images: list[str] = field(default_factory=list)
    source_row: int = 0

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "state": self.state,
            "questions": {qid: q.to_clef() for qid, q in self.questions.items()},
            "labels": {qid: label.to_json() for qid, label in self.labels.items()},
        }
        if self.images:
            out["images"] = list(self.images)
        return out


@dataclass
class LoadResult:
    records: list[Record]
    issues: list[Issue]
    columns: list[str]


def read_rows(path: Path) -> tuple[list[Row], list[str], list[Issue]]:
    """Read a CSV/TSV/JSONL file into numbered row dicts. Returns (rows, columns, issues)."""
    if not path.exists():
        raise DataError(f"{path}: file not found")
    suffix = path.suffix.lower()
    if suffix in TABLE_DELIMITERS:
        return _read_table(path, TABLE_DELIMITERS[suffix])
    if suffix in JSONL_SUFFIXES:
        return _read_jsonl(path)
    raise DataError(f"{path.name}: unsupported file type {suffix!r} (use .csv, .tsv or .jsonl)")


def _decode(path: Path) -> tuple[str, list[Issue]]:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig"), []
    except UnicodeDecodeError:
        message = f"{path.name} is not UTF-8; read it as Windows-1252 (re-save as UTF-8 to be sure)"
        return raw.decode("cp1252", errors="replace"), [Issue("warning", message)]


def _read_table(path: Path, delimiter: str) -> tuple[list[Row], list[str], list[Issue]]:
    text, issues = _decode(path)
    records = list(csv.reader(io.StringIO(text, newline=""), delimiter=delimiter))
    if not records or not any(cell.strip() for cell in records[0]):
        raise DataError(f"{path.name}: no header row")
    header = [cell.strip() for cell in records[0]]
    duplicates = sorted({name for name in header if header.count(name) > 1})
    if duplicates:
        raise DataError(f"{path.name}: duplicate column names {duplicates}")
    if "" in header:
        raise DataError(f"{path.name}: column {header.index('') + 1} has no name")
    rows: list[Row] = []
    bad_width: list[int] = []
    for number, values in enumerate(records[1:], start=2):
        if not any(value.strip() for value in values):
            continue  # blank line
        if len(values) != len(header):
            bad_width.append(number)
            continue
        rows.append((number, dict(zip(header, values, strict=True))))
    if bad_width:
        issues.append(Issue("error", f"rows must have {len(header)} fields", bad_width))
    return rows, header, issues


def _read_jsonl(path: Path) -> tuple[list[Row], list[str], list[Issue]]:
    text, issues = _decode(path)
    rows: list[Row] = []
    columns: dict[str, None] = {}
    invalid: list[int] = []
    not_objects: list[int] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            invalid.append(number)
            continue
        if not isinstance(value, dict):
            not_objects.append(number)
            continue
        rows.append((number, value))
        columns.update(dict.fromkeys(value))
    if invalid:
        issues.append(Issue("error", "invalid JSON", invalid))
    if not_objects:
        issues.append(Issue("error", "each line must be a JSON object", not_objects))
    return rows, list(columns), issues


@dataclass
class _FlatLayout:
    state_columns: list[str]
    label_columns: dict[str, str]  # question id -> existing column


def _flat_layout(
    columns: list[str], cfg: DataConfig, schema: Schema, issues: list[Issue]
) -> _FlatLayout:
    present = set(columns)

    def missing(column: str, what: str, question: str | None = None) -> None:
        hint = difflib.get_close_matches(column, columns, n=1)
        suffix = f" (did you mean '{hint[0]}'?)" if hint else ""
        issues.append(Issue("error", f"{what} '{column}' not found{suffix}", question=question))

    for qid in sorted(set(cfg.label_columns) - set(schema.questions)):
        issues.append(Issue("error", f"label_columns refers to unknown question '{qid}'"))
    label_columns: dict[str, str] = {}
    for qid in schema.questions:
        column = cfg.label_columns.get(qid, qid)
        if column in present:
            label_columns[qid] = column
        else:
            missing(column, "label column", qid)
    if cfg.id_column and cfg.id_column not in present:
        missing(cfg.id_column, "id column")
    for column in cfg.image_columns:
        if column not in present:
            missing(column, "image column")
    if cfg.state_columns is not None:
        for column in cfg.state_columns:
            if column not in present:
                missing(column, "state column")
        state_columns = [c for c in cfg.state_columns if c in present]
    else:
        used = set(label_columns.values()) | set(cfg.image_columns) | {cfg.id_column}
        state_columns = [c for c in columns if c not in used]
    if not state_columns:
        issues.append(Issue("error", "no state columns: set data.state_columns"))
    return _FlatLayout(state_columns, label_columns)


def _resolve_image(value: str, data_dir: Path) -> str:
    if value.startswith(URL_PREFIXES):
        return value
    path = Path(value.replace("\\", "/"))
    return str(path if path.is_absolute() else (data_dir / path).resolve())


def _labels_for(number: int, pairs: Iterable[tuple[Question, Any]], note: Note) -> dict[str, Label]:
    labels: dict[str, Label] = {}
    for question, raw in pairs:
        try:
            label, warning = parse_label(question, raw)
        except LabelError as exc:
            note("error", question.id, str(exc), number)
            continue
        if warning:
            note("warning", question.id, warning, number)
        if label is not None:
            labels[question.id] = label
    return labels


def _flat_record(
    number: int,
    row: dict[str, Any],
    cfg: DataConfig,
    schema: Schema,
    layout: _FlatLayout,
    data_dir: Path,
    note: Note,
) -> Record | None:
    if cfg.id_column:
        raw_id = row.get(cfg.id_column)
        record_id = "" if is_missing(raw_id) else str(raw_id).strip()
        if not record_id:
            note("error", None, f"empty id in column '{cfg.id_column}'", number)
            return None
    else:
        record_id = f"row-{number}"
    values = {column: row.get(column) for column in layout.state_columns}
    state = values[layout.state_columns[0]] if len(layout.state_columns) == 1 else values
    pairs = ((schema.questions[qid], row.get(col)) for qid, col in layout.label_columns.items())
    labels = _labels_for(number, pairs, note)
    images = [
        _resolve_image(str(row[column]).strip(), data_dir)
        for column in cfg.image_columns
        if column in row and not is_missing(row[column])
    ]
    return Record(record_id, state, schema.questions, labels, images, number)


def _native_record(
    number: int,
    row: dict[str, Any],
    schema: Schema | None,
    data_dir: Path,
    note: Note,
    question_cache: dict[str, dict[str, Question]],
) -> Record | None:
    if "state" not in row:
        note("error", None, "native rows need a 'state'", number)
        return None
    if row.get("questions") is not None:
        key = json.dumps(row["questions"], sort_keys=True, default=str)
        if key not in question_cache:
            try:
                question_cache[key] = Schema.from_dict({"questions": row["questions"]}).questions
            except SchemaError as exc:
                note("error", None, f"invalid questions: {exc}", number)
                return None
        questions = question_cache[key]
    elif schema is not None:
        questions = schema.questions
    else:
        note("error", None, "row has no 'questions' and the config has no schema", number)
        return None
    pairs = []
    for qid, raw in row["labels"].items():
        question = questions.get(str(qid))
        if question is None:
            note("error", str(qid), "label for a question the row does not ask", number)
            continue
        pairs.append((question, raw))
    labels = _labels_for(number, pairs, note)
    images = [_resolve_image(str(value).strip(), data_dir) for value in row.get("images") or []]
    record_id = f"row-{number}" if is_missing(row.get("id")) else str(row["id"]).strip()
    return Record(record_id, row["state"], questions, labels, images, number)


def normalize_rows(
    rows: list[Row],
    columns: list[str],
    cfg: DataConfig,
    schema: Schema | None,
    data_dir: Path,
) -> LoadResult:
    issues: list[Issue] = []
    grouped: dict[tuple[str, str | None, str], list[int]] = defaultdict(list)

    def note(level: str, question: str | None, message: str, number: int) -> None:
        grouped[(level, question, message)].append(number)

    layout = None
    if schema is not None and any(not isinstance(r.get("labels"), dict) for _, r in rows):
        layout = _flat_layout(columns, cfg, schema, issues)
    question_cache: dict[str, dict[str, Question]] = {}
    records: list[Record] = []
    unlabeled: list[int] = []
    for number, row in rows:
        if isinstance(row.get("labels"), dict):
            record = _native_record(number, row, schema, data_dir, note, question_cache)
        elif schema is None or layout is None:
            message = "flat rows need a schema (add 'schema' or 'schema_file' to the config)"
            note("error", None, message, number)
            record = None
        else:
            record = _flat_record(number, row, cfg, schema, layout, data_dir, note)
        if record is None:
            continue
        if not record.labels:
            unlabeled.append(number)
            continue
        records.append(record)
    for (level, question, message), numbers in grouped.items():
        issues.append(Issue(level, message, numbers, question))  # type: ignore[arg-type]
    if unlabeled:
        message = f"{len(unlabeled)} rows have no labels and were dropped"
        issues.append(Issue("warning", message, unlabeled))
    return LoadResult(records, issues, columns)


def load_file(config: Config, path_value: str) -> LoadResult:
    path = config.resolve_path(path_value)
    rows, columns, read_issues = read_rows(path)
    result = normalize_rows(rows, columns, config.data, config.get_schema(), path.parent)
    result.issues = read_issues + result.issues
    return result


def _stratify_question(records: list[Record]) -> str | None:
    labeled = {qid for record in records for qid in record.labels}
    ordered = dict.fromkeys(qid for r in records for qid in [*r.questions, *r.labels])
    return next((qid for qid in ordered if qid in labeled), None)


def split_records(
    records: list[Record], eval_fraction: float, seed: int
) -> tuple[list[Record], list[Record]]:
    """Stratified, seeded split on the first labeled question; singleton classes stay in train."""
    key_question = _stratify_question(records)
    groups: dict[str | None, list[Record]] = defaultdict(list)
    for record in records:
        label = record.labels.get(key_question) if key_question else None
        groups[label.top() if label else None].append(record)
    rng = random.Random(seed)
    chosen: set[int] = set()
    for key in sorted(groups, key=lambda k: (k is None, str(k))):
        members = groups[key][:]
        rng.shuffle(members)
        count = round(len(members) * eval_fraction) if len(members) > 1 else 0
        chosen.update(id(member) for member in members[:count])
    if not chosen and len(records) > 1:
        largest = max(groups.values(), key=len)
        chosen.add(id(rng.choice(largest)))
    train = [record for record in records if id(record) not in chosen]
    evals = [record for record in records if id(record) in chosen]
    return train, evals
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_data.py -v`
Expected: all PASS.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format . && uv run ruff check --fix .
git add clef_tune/data.py tests/helpers.py tests/test_data.py
git commit -F - <<'EOF'
feat: load CSV/JSONL into Clef records with grouped issues and stratified split
EOF
```

---

### Task 6: Clef prompt layout and token counting

**Files:**
- Create: `clef_tune/prompt.py`
- Test: `tests/test_prompt.py`

**Interfaces:**
- Consumes: `schema.Question`, `config.Config` (for `counter_for_config`), `tests.helpers.CharCounter`.
- Produces:
  - Constants `SYSTEM_PROMPT`, `PREFIX`, `SUFFIX`, `IMAGE_PLACEHOLDER`, `DEFAULT_URL_IMAGE_SIZE = (1024, 1024)`
  - `render(value) -> str`; `Piece(kind, text, question_id=None)`; `schema_pieces(questions) -> list[Piece]`; `prompt_pieces(state, questions, n_images=0) -> list[Piece]` (the UI's prompt preview uses `kind` to highlight `state` / `question` / `option` spans)
  - `TokenCounter` protocol (`count(text) -> int`, `count_many(texts) -> list[int]`); `HFTokenCounter(tokenizer_path)`; `get_token_counter(model_id, revision) -> HFTokenCounter` (cached); `counter_for_config(config) -> HFTokenCounter`
  - `TokenCount(fixed, state, max_length)` with `schema_too_long`, `truncated`, `total`; `RecordTokenCounter(counter, max_length)` with `count_records(items: list[tuple[state, questions, media_tokens]]) -> list[TokenCount]`
  - `image_token_count(width, height) -> int`; `media_token_count(sizes: list[tuple[int, int]]) -> int`

- [ ] **Step 1: Write the failing tests**

`tests/test_prompt.py`:

```python
import pytest

from clef_tune.config import PINNED_REVISIONS
from clef_tune.prompt import (
    PREFIX,
    SUFFIX,
    RecordTokenCounter,
    get_token_counter,
    image_token_count,
    media_token_count,
    prompt_pieces,
    render,
    schema_pieces,
)
from clef_tune.schema import Question
from tests.helpers import CharCounter

URGENT = {"urgent": Question.from_dict("urgent", {"type": "noul", "instructions": "Is it urgent?"})}


def test_prompt_matches_clef_encode_record_layout():
    text = "".join(piece.text for piece in prompt_pieces("hi", URGENT))
    expected = (
        PREFIX
        + "hi"
        + "\n\nSCHEMA FIELDS:\n"
        + "\nFIELD 1\nID: urgent\nTYPE: noul\nINSTRUCTION: "
        + "Is it urgent?"
        + "\nALLOWED OPTIONS:\n"
        + 'OPTION 1: {"description":"The proposition is true or the answer is yes.",'
        + '"option_id":"true"}\n'
        + 'OPTION 2: {"description":"The proposition is false or the answer is no.",'
        + '"option_id":"false"}\n'
        + "END FIELD\n"
        + SUFFIX
    )
    assert text == expected


def test_prefix_and_suffix_are_clefs():
    assert PREFIX.startswith("<|im_start|>system\nRead the complete state and schema.")
    assert PREFIX.endswith("<|im_end|>\n<|im_start|>user\nSTATE:\n")
    assert SUFFIX == (
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
    )


def test_piece_kinds_mark_spans_for_the_preview():
    kinds = [piece.kind for piece in prompt_pieces({"a": 1}, URGENT, n_images=2)]
    assert kinds[:3] == ["prefix", "media", "state"]
    assert kinds.count("question") == 1 and kinds.count("option") == 2
    assert kinds[-1] == "suffix"


def test_missing_instructions_fall_back_to_question_id():
    question = Question.from_dict("refund", {"type": "noul"})
    pieces = schema_pieces({"refund": question})
    assert [p.text for p in pieces if p.kind == "question"] == ["refund"]


def test_option_without_description_omits_it():
    question = Question.from_dict("d", {"type": "choice", "criteria": {"x": None}})
    assert [p.text for p in schema_pieces({"d": question}) if p.kind == "option"] == [
        '{"option_id":"x"}'
    ]


def test_render_is_compact_sorted_json():
    assert render({"b": 1, "a": "é"}) == '{"a":"é","b":1}'
    assert render("plain") == "plain"


def test_token_counts_and_truncation():
    counter = RecordTokenCounter(CharCounter(), max_length=1000)
    fixed = len("".join(p.text for p in prompt_pieces("", URGENT)))
    short, long = counter.count_records([("hi", URGENT, 0), ("x" * 2000, URGENT, 0)])
    assert short.fixed == fixed and short.state == 2
    assert short.total == fixed + 2 and not short.truncated
    assert long.truncated and long.total == 1000


def test_media_tokens_add_to_fixed():
    counter = RecordTokenCounter(CharCounter(), max_length=10_000)
    (plain,) = counter.count_records([("hi", URGENT, 0)])
    (with_image,) = counter.count_records([("hi", URGENT, 1027)])
    assert with_image.fixed - plain.fixed == 1027


def test_schema_too_long():
    counter = RecordTokenCounter(CharCounter(), max_length=256)
    (count,) = counter.count_records([("hi", URGENT, 0)])
    assert count.schema_too_long and not count.truncated and count.total == count.fixed


def test_image_token_count_matches_qwen_smart_resize():
    assert image_token_count(1024, 1024) == 1024
    assert image_token_count(100, 100) == 64  # upscaled to the 65536-pixel minimum: 256x256
    assert image_token_count(8000, 6000) == 16170  # downscaled to the 16777216-pixel maximum


def test_media_token_count_adds_vision_markers_and_newline():
    assert media_token_count([]) == 0
    assert media_token_count([(1024, 1024), (100, 100)]) == (1024 + 2) + (64 + 2) + 1


@pytest.mark.network
def test_real_clef_tokenizer():
    counter = get_token_counter("Cloudflare/clef", PINNED_REVISIONS["Cloudflare/clef"])
    assert counter.count("<|im_start|>") == 1
    assert counter.count_many(["hello world", ""]) == [2, 0]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_prompt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clef_tune.prompt'`.

- [ ] **Step 3: Implement `clef_tune/prompt.py`**

```python
"""Clef's exact prompt layout (mirrors encode_record in joint_schema_model.py), token
counting with the real Clef tokenizer, and Qwen vision token estimates for images.

encode_record tokenizes each piece separately and concatenates the ids, so a record's
token count is the sum of the pieces' counts, not the count of the joined string.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from clef_tune.schema import Question

if TYPE_CHECKING:
    from clef_tune.config import Config

SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
PREFIX = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
SUFFIX = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

# Qwen2VLImageProcessor settings from Clef's processor_config.json.
PATCH_SIZE = 16
MERGE_SIZE = 2
MIN_PIXELS = 65536
MAX_PIXELS = 16777216
DEFAULT_URL_IMAGE_SIZE = (1024, 1024)  # assumed for images we can't open (URLs)
COUNT_BATCH = 2048

PieceKind = Literal["prefix", "media", "state", "schema", "question", "option", "suffix"]


def render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True)
class Piece:
    kind: PieceKind
    text: str
    question_id: str | None = None


def schema_pieces(questions: dict[str, Question]) -> list[Piece]:
    pieces = [Piece("schema", "\n\nSCHEMA FIELDS:\n")]
    for index, (qid, question) in enumerate(questions.items()):
        header = f"\nFIELD {index + 1}\nID: {qid}\nTYPE: {question.type}\nINSTRUCTION: "
        pieces.append(Piece("schema", header, qid))
        instructions = question.instructions if question.instructions else str(qid)
        pieces.append(Piece("question", render(instructions), qid))
        pieces.append(Piece("schema", "\nALLOWED OPTIONS:\n", qid))
        for option_index, (option_id, description) in enumerate(question.options()):
            pieces.append(Piece("schema", f"OPTION {option_index + 1}: ", qid))
            semantics: dict[str, Any] = {"option_id": option_id}
            if description is not None:
                semantics["description"] = description
            pieces.append(Piece("option", render(semantics), qid))
            pieces.append(Piece("schema", "\n", qid))
        pieces.append(Piece("schema", "END FIELD\n", qid))
    return pieces


def prompt_pieces(state: Any, questions: dict[str, Question], n_images: int = 0) -> list[Piece]:
    pieces = [Piece("prefix", PREFIX)]
    if n_images:
        pieces.append(Piece("media", IMAGE_PLACEHOLDER * n_images + "\n"))
    pieces.append(Piece("state", render(state)))
    pieces.extend(schema_pieces(questions))
    pieces.append(Piece("suffix", SUFFIX))
    return pieces


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

    def count_many(self, texts: list[str]) -> list[int]: ...


class HFTokenCounter:
    """Counts tokens with a `tokenizer.json` (the same tokenizer Clef's processor uses)."""

    def __init__(self, tokenizer_path: str | Path) -> None:
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def count_many(self, texts: list[str]) -> list[int]:
        encodings = self._tokenizer.encode_batch(list(texts), add_special_tokens=False)
        return [len(encoding.ids) for encoding in encodings]


@lru_cache(maxsize=4)
def get_token_counter(model_id: str, revision: str) -> HFTokenCounter:
    from huggingface_hub import hf_hub_download

    return HFTokenCounter(hf_hub_download(model_id, "tokenizer.json", revision=revision))


@lru_cache(maxsize=4)
def _counter_from_file(path: str) -> HFTokenCounter:
    return HFTokenCounter(path)


def counter_for_config(config: Config) -> HFTokenCounter:
    if config.model.local_path:
        local = config.resolve_path(config.model.local_path) / "tokenizer.json"
        if local.exists():
            return _counter_from_file(str(local))
    return get_token_counter(config.model.id, config.model.resolved_revision())


@dataclass(frozen=True)
class TokenCount:
    fixed: int  # prefix + media + schema + suffix
    state: int  # state tokens before truncation
    max_length: int

    @property
    def schema_too_long(self) -> bool:
        return self.fixed > self.max_length

    @property
    def truncated(self) -> bool:
        return not self.schema_too_long and self.fixed + self.state > self.max_length

    @property
    def total(self) -> int:
        if self.schema_too_long:
            return self.fixed
        return min(self.fixed + self.state, self.max_length)


class RecordTokenCounter:
    """Counts full Clef prompts, caching schema counts (shared by most records)."""

    def __init__(self, counter: TokenCounter, max_length: int) -> None:
        self.counter = counter
        self.max_length = max_length
        self._frame = counter.count(PREFIX) + counter.count(SUFFIX)
        self._schema_cache: dict[str, int] = {}

    def schema_tokens(self, questions: dict[str, Question]) -> int:
        key = json.dumps([[qid, q.to_clef()] for qid, q in questions.items()], default=str)
        if key not in self._schema_cache:
            self._schema_cache[key] = sum(
                self.counter.count(piece.text) for piece in schema_pieces(questions)
            )
        return self._schema_cache[key]

    def count_records(
        self, items: list[tuple[Any, dict[str, Question], int]]
    ) -> list[TokenCount]:
        """items: (state, questions, media_tokens) per record."""
        states: list[int] = []
        for start in range(0, len(items), COUNT_BATCH):
            chunk = items[start : start + COUNT_BATCH]
            states.extend(self.counter.count_many([render(state) for state, _, _ in chunk]))
        return [
            TokenCount(
                fixed=self._frame + media + self.schema_tokens(questions),
                state=n_state,
                max_length=self.max_length,
            )
            for (_, questions, media), n_state in zip(items, states, strict=True)
        ]


def image_token_count(
    width: int, height: int, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> int:
    """Vision tokens for one image after Qwen's smart_resize (patch 16, merge 2)."""
    factor = PATCH_SIZE * MERGE_SIZE
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return (h_bar // PATCH_SIZE) * (w_bar // PATCH_SIZE) // (MERGE_SIZE * MERGE_SIZE)


def media_token_count(sizes: list[tuple[int, int]]) -> int:
    """Tokens for a record's images: each image's pads + vision_start/end, plus one newline."""
    if not sizes:
        return 0
    return sum(image_token_count(width, height) + 2 for width, height in sizes) + 1
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_prompt.py -v`
Expected: all PASS. `test_real_clef_tokenizer` downloads `tokenizer.json` (about 20 MB) the first time.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check --fix .
git add clef_tune/prompt.py tests/test_prompt.py
git commit -F - <<'EOF'
feat: mirror Clef's prompt layout and count tokens with the real tokenizer
EOF
```

---

### Task 7: Validation report

**Files:**
- Create: `clef_tune/validation.py`
- Test: `tests/test_validation.py`

**Interfaces:**
- Consumes: `config.Config`; `data.DataError`, `data.Issue`, `data.Record`, `data.load_file`, `data.split_records`; `prompt.DEFAULT_URL_IMAGE_SIZE`, `prompt.RecordTokenCounter`, `prompt.TokenCounter`, `prompt.media_token_count`; `schema.Question`, `schema.SchemaError`.
- Produces:
  - `validation.IMBALANCE_SHARE = 0.05`, `validation.MIN_CALIBRATION_ROWS = 20`
  - `QuestionStats(question_id, type, option_ids, train_counts: dict[str, int], eval_counts: dict[str, int], train_missing: int, eval_labeled: int)`
  - `TokenStats(train_lengths: list[int], eval_lengths: list[int], truncated_rows: int)` with `p50`, `p95`, `max`, `train_tokens`, `eval_tokens`
  - `ValidationReport(issues, train, eval, questions, tokens, columns)` with `errors`, `warnings`, `ok`
  - `validate(config, counter: TokenCounter | None) -> ValidationReport`

- [ ] **Step 1: Write the failing tests**

`tests/test_validation.py`:

```python
from PIL import Image

from clef_tune.validation import validate
from tests.helpers import CharCounter, make_config, ticket_rows, write_csv


def config_for(tmp_path, rows, **data):
    write_csv(tmp_path / "train.csv", rows)
    return make_config(tmp_path, data={"train": "train.csv", "id_column": "id", **data})


def messages(issues):
    return [str(issue) for issue in issues]


def test_valid_dataset(tmp_path):
    report = validate(config_for(tmp_path, ticket_rows(40)), CharCounter())
    assert report.ok, messages(report.errors)
    assert len(report.train) == 36 and len(report.eval) == 4
    dept = report.questions["department"]
    assert dept.train_counts == {"NA": 0, "billing": 18, "technical": 18}
    assert dept.eval_labeled == 4
    assert report.tokens is not None and report.tokens.max >= report.tokens.p95 >= report.tokens.p50
    assert report.tokens.train_tokens == sum(report.tokens.train_lengths)
    assert "subject" in report.columns
    assert any("calibration will be skipped" in m for m in messages(report.warnings))


def test_single_class_question_is_an_error(tmp_path):
    rows = ticket_rows(20)
    for row in rows:
        row["outage"] = "no"
    report = validate(config_for(tmp_path, rows), CharCounter())
    assert any(m.startswith("outage: needs at least 2 classes") for m in messages(report.errors))


def test_rare_class_warning(tmp_path):
    rows = ticket_rows(60)
    for row in rows:
        row["outage"] = "no"
    rows[0]["outage"] = "yes"
    rows[1]["outage"] = "yes"  # 2 of 60 training rows (3%)
    write_csv(tmp_path / "eval.csv", ticket_rows(10, start=100))
    report = validate(config_for(tmp_path, rows, eval="eval.csv"), CharCounter())
    assert any("rare classes" in m and "class_weighting" in m for m in messages(report.warnings))


def test_duplicate_ids_are_an_error(tmp_path):
    rows = ticket_rows(20)
    rows[5]["id"] = rows[4]["id"]
    report = validate(config_for(tmp_path, rows), CharCounter())
    assert any("duplicate ID" in m for m in messages(report.errors))


def test_eval_file_is_used_and_missing_class_warns(tmp_path):
    write_csv(tmp_path / "eval.csv", [r for r in ticket_rows(10, start=100) if r["department"] == "billing"])
    report = validate(config_for(tmp_path, ticket_rows(40), eval="eval.csv"), CharCounter())
    assert len(report.train) == 40 and len(report.eval) == 5
    assert any(
        m.startswith("department: eval set has no examples of: technical")
        for m in messages(report.warnings)
    )


def test_schema_too_long_is_an_error(tmp_path):
    cfg = config_for(tmp_path, ticket_rows(20))
    cfg.train.max_length = 256
    report = validate(cfg, CharCounter())
    assert any("max_length is 256" in m for m in messages(report.errors))


def test_truncation_warning(tmp_path):
    rows = ticket_rows(20)
    rows[3]["body"] = "x" * 5000
    report = validate(config_for(tmp_path, rows), CharCounter())
    assert report.tokens.truncated_rows == 1
    assert any("will be truncated" in m for m in messages(report.warnings))


def test_images_missing_and_present(tmp_path):
    rows = ticket_rows(20)
    for row in rows:
        row["receipt"] = ""
    Image.new("RGB", (1024, 1024)).save(tmp_path / "ok.png")
    rows[0]["receipt"] = "ok.png"
    rows[1]["receipt"] = "missing.png"
    rows[2]["receipt"] = "https://example.com/r.png"
    report = validate(config_for(tmp_path, rows, image_columns=["receipt"]), CharCounter())
    assert any("image file not found" in m and "rows 3" in m for m in messages(report.errors))
    assert any("not checked" in m for m in messages(report.warnings))


def test_without_counter_tokens_are_skipped(tmp_path):
    report = validate(config_for(tmp_path, ticket_rows(20)), None)
    assert report.tokens is None
    assert any("tokenizer unavailable" in m for m in messages(report.warnings))


def test_missing_data_file_and_bad_schema_file(tmp_path):
    cfg = make_config(tmp_path, data={"train": "nope.csv"})
    assert any("file not found" in m for m in messages(validate(cfg, None).errors))
    (tmp_path / "s.yaml").write_text("questions: {q: {type: bogus}}\n", encoding="utf-8")
    cfg = make_config(tmp_path, schema=None, data={"train": "x.csv"}, schema_file="s.yaml")
    assert any(m.startswith("schema:") for m in messages(validate(cfg, None).errors))


def test_too_few_rows(tmp_path):
    report = validate(config_for(tmp_path, ticket_rows(1)), CharCounter())
    assert any("eval set is empty" in m for m in messages(report.errors))
```

Note: in `test_images_missing_and_present`, `missing.png` is on spreadsheet row 3 (the header is row 1).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_validation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clef_tune.validation'`.

- [ ] **Step 3: Implement `clef_tune/validation.py`**

```python
"""Validate a config and its data. Errors block training; warnings inform; the stats feed
the UI and the cost estimates."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from clef_tune.config import Config
from clef_tune.data import URL_PREFIXES, DataError, Issue, Record, load_file, split_records
from clef_tune.prompt import (
    DEFAULT_URL_IMAGE_SIZE,
    RecordTokenCounter,
    TokenCounter,
    media_token_count,
)
from clef_tune.schema import Question, SchemaError

IMBALANCE_SHARE = 0.05
MIN_CALIBRATION_ROWS = 20


@dataclass
class QuestionStats:
    question_id: str
    type: str
    option_ids: list[str]
    train_counts: dict[str, int]
    eval_counts: dict[str, int]
    train_missing: int
    eval_labeled: int


def _percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


@dataclass
class TokenStats:
    train_lengths: list[int]
    eval_lengths: list[int]
    truncated_rows: int

    @property
    def p50(self) -> int:
        return _percentile(self.train_lengths + self.eval_lengths, 0.50)

    @property
    def p95(self) -> int:
        return _percentile(self.train_lengths + self.eval_lengths, 0.95)

    @property
    def max(self) -> int:
        return max(self.train_lengths + self.eval_lengths, default=0)

    @property
    def train_tokens(self) -> int:
        return sum(self.train_lengths)

    @property
    def eval_tokens(self) -> int:
        return sum(self.eval_lengths)


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)
    train: list[Record] = field(default_factory=list)
    eval: list[Record] = field(default_factory=list)
    questions: dict[str, QuestionStats] = field(default_factory=dict)
    tokens: TokenStats | None = None
    columns: list[str] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors


def validate(config: Config, counter: TokenCounter | None) -> ValidationReport:
    report = ValidationReport()
    try:
        config.get_schema()
    except (SchemaError, OSError) as exc:
        report.issues.append(Issue("error", f"schema: {exc}"))
        return report
    try:
        train_result = load_file(config, config.data.train)
        eval_result = load_file(config, config.data.eval) if config.data.eval else None
    except DataError as exc:
        report.issues.append(Issue("error", str(exc)))
        return report
    report.columns = train_result.columns
    report.issues.extend(train_result.issues)
    if eval_result is not None:
        report.issues.extend(
            Issue(i.level, f"eval file: {i.message}", i.rows, i.question)
            for i in eval_result.issues
        )
        report.train, report.eval = train_result.records, eval_result.records
    else:
        report.train, report.eval = split_records(
            train_result.records, config.data.eval_fraction, config.train.seed
        )
    if not report.train:
        report.issues.append(Issue("error", "no labeled training rows"))
    if not report.eval:
        message = "eval set is empty (need at least 2 labeled rows, or set data.eval)"
        report.issues.append(Issue("error", message))
    _check_ids(report)
    _question_stats(report)
    _check_images(report)
    if counter is None:
        message = "tokenizer unavailable; token statistics and truncation checks skipped"
        report.issues.append(Issue("warning", message))
    else:
        _token_stats(report, config, counter)
    return report


def _check_ids(report: ValidationReport) -> None:
    records = report.train + report.eval
    counts = Counter(record.id for record in records)
    duplicates = [record_id for record_id, n in counts.items() if n > 1]
    if duplicates:
        rows = sorted(r.source_row for r in records if counts[r.id] > 1)
        examples = ", ".join(duplicates[:5])
        message = f"{len(duplicates)} duplicate ID(s) (e.g. {examples})"
        report.issues.append(Issue("error", message, rows))


def _question_stats(report: ValidationReport) -> None:
    records = report.train + report.eval
    questions: dict[str, Question] = {}
    option_ids: dict[str, dict[str, None]] = {}
    for record in records:
        for qid, question in record.questions.items():
            questions.setdefault(qid, question)
            option_ids.setdefault(qid, {}).update(dict.fromkeys(question.option_ids()))
    for qid, question in questions.items():
        options = list(option_ids[qid])
        train = Counter(r.labels[qid].top() for r in report.train if qid in r.labels)
        evals = Counter(r.labels[qid].top() for r in report.eval if qid in r.labels)
        stats = QuestionStats(
            question_id=qid,
            type=question.type,
            option_ids=options,
            train_counts={option: train.get(option, 0) for option in options},
            eval_counts={option: evals.get(option, 0) for option in options},
            train_missing=sum(1 for r in report.train if qid in r.questions and qid not in r.labels),
            eval_labeled=sum(evals.values()),
        )
        report.questions[qid] = stats
        observed = [option for option in options if train.get(option)]
        if len(observed) < 2:
            message = f"needs at least 2 classes in training data (found {len(observed)})"
            report.issues.append(Issue("error", message, question=qid))
            continue
        total = sum(train.values())
        rare = [option for option in observed if train[option] / total < IMBALANCE_SHARE]
        if rare:
            message = (
                f"rare classes (<5% of labels): {', '.join(rare)}; "
                "consider class_weighting: balanced"
            )
            report.issues.append(Issue("warning", message, question=qid))
        missing_eval = [option for option in observed if not evals.get(option)]
        if missing_eval and report.eval:
            message = f"eval set has no examples of: {', '.join(missing_eval)}"
            report.issues.append(Issue("warning", message, question=qid))
        if stats.eval_labeled < MIN_CALIBRATION_ROWS:
            message = (
                f"only {stats.eval_labeled} labeled eval rows; "
                "calibration will be skipped for this question"
            )
            report.issues.append(Issue("warning", message, question=qid))


def _check_images(report: ValidationReport) -> None:
    missing: list[int] = []
    remote: list[int] = []
    for record in report.train + report.eval:
        for image in record.images:
            if image.startswith(URL_PREFIXES):
                remote.append(record.source_row)
            elif not Path(image).exists():
                missing.append(record.source_row)
    if missing:
        report.issues.append(Issue("error", "image file not found", missing))
    if remote:
        message = "image URLs are not checked; assumed 1024x1024 for token estimates"
        report.issues.append(Issue("warning", message, remote))


def _media_tokens(record: Record) -> int:
    from PIL import Image

    sizes: list[tuple[int, int]] = []
    for image in record.images:
        size = DEFAULT_URL_IMAGE_SIZE
        if not image.startswith(URL_PREFIXES) and Path(image).exists():
            try:
                with Image.open(image) as opened:
                    size = opened.size
            except OSError:
                pass
        sizes.append(size)
    return media_token_count(sizes)


def _token_stats(report: ValidationReport, config: Config, counter: TokenCounter) -> None:
    max_length = config.train.max_length
    counting = RecordTokenCounter(counter, max_length)
    records = report.train + report.eval
    counts = counting.count_records([(r.state, r.questions, _media_tokens(r)) for r in records])
    too_long = [r.source_row for r, c in zip(records, counts, strict=True) if c.schema_too_long]
    if too_long:
        worst = max(c.fixed for c in counts)
        message = (
            f"schema needs up to {worst} tokens before the state; max_length is {max_length}"
        )
        report.issues.append(Issue("error", message, too_long))
    truncated = [r.source_row for r, c in zip(records, counts, strict=True) if c.truncated]
    if truncated:
        message = (
            f"{len(truncated)} rows exceed max_length ({max_length}); "
            "their state will be truncated"
        )
        report.issues.append(Issue("warning", message, truncated))
    split = len(report.train)
    report.tokens = TokenStats(
        train_lengths=[c.total for c in counts[:split]],
        eval_lengths=[c.total for c in counts[split:]],
        truncated_rows=len(truncated),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_validation.py -v`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check --fix .
git add clef_tune/validation.py tests/test_validation.py
git commit -F - <<'EOF'
feat: add validation report with label, image and token checks
EOF
```

---

### Task 8: Metrics

**Files:**
- Create: `clef_tune/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing from the project (pure Python).
- Produces:
  - `Prediction(question_id, question_type, option_ids: list[str], probs: list[float], target: list[float])` with `predicted`, `actual`, `correct`, `log_loss()`
  - `ConfusionMatrix(labels: list[str], matrix: list[list[int]])` (rows = true, columns = predicted)
  - `QuestionMetrics(question_id, question_type, n, accuracy, f1, log_loss, mae, confusion)`: `f1` is macro-F1 for choice/score and binary F1 (positive `true`) for noul; `mae` is only set for score questions
  - `MetricsReport(questions, exact_match, n_records)` with `mean_macro_f1`, `mean_log_loss`, `value(metric)`, `to_dict()`
  - `confusion_matrix(preds)`, `question_metrics(preds)`, `summarize(records: list[list[Prediction]]) -> MetricsReport`, `HIGHER_IS_BETTER`, `is_better(metric, new, best) -> bool`

- [ ] **Step 1: Write the failing tests**

`tests/test_metrics.py`:

```python
import math

import pytest

from clef_tune.metrics import Prediction, is_better, question_metrics, summarize


def pred(actual, predicted, options=("a", "b", "c"), qid="q", qtype="choice", conf=0.7):
    other = (1 - conf) / (len(options) - 1)
    probs = [conf if option == predicted else other for option in options]
    target = [1.0 if option == actual else 0.0 for option in options]
    return Prediction(qid, qtype, list(options), probs, target)


def test_choice_accuracy_macro_f1_and_confusion():
    pairs = [("a", "a"), ("a", "a"), ("a", "b"), ("b", "b"), ("b", "a"), ("c", "c")]
    metrics = question_metrics([pred(actual, predicted) for actual, predicted in pairs])
    assert metrics.n == 6
    assert metrics.accuracy == pytest.approx(4 / 6)
    assert metrics.f1 == pytest.approx((2 / 3 + 1 / 2 + 1) / 3)
    assert metrics.confusion.labels == ["a", "b", "c"]
    assert metrics.confusion.matrix == [[2, 1, 0], [1, 1, 0], [0, 0, 1]]
    assert metrics.mae is None


def test_log_loss():
    metrics = question_metrics([pred("a", "a"), pred("a", "b")])
    assert metrics.log_loss == pytest.approx((-math.log(0.7) - math.log(0.15)) / 2)


def test_noul_uses_binary_f1_on_true():
    options = ("true", "false")
    pairs = [("true", "true"), ("true", "false"), ("false", "true"), ("false", "false")]
    preds = [pred(a, p, options, qtype="noul") for a, p in pairs]
    assert question_metrics(preds).f1 == pytest.approx(0.5)


def test_noul_f1_zero_without_true_positives():
    preds = [pred("true", "false", ("true", "false"), qtype="noul")]
    assert question_metrics(preds).f1 == 0.0


def test_score_mae_uses_expected_level():
    p = Prediction("u", "score", ["0", "1", "2"], [0.2, 0.3, 0.5], [0.0, 0.0, 1.0])
    assert question_metrics([p]).mae == pytest.approx(0.7)


def test_soft_target_actual_is_argmax():
    p = Prediction("q", "choice", ["a", "b"], [0.4, 0.6], [0.3, 0.7])
    assert p.actual == "b" and p.correct


def test_summarize_exact_match_and_aggregates():
    records = [
        [pred("a", "a"), pred("true", "true", ("true", "false"), qid="n", qtype="noul")],
        [pred("a", "b"), pred("true", "true", ("true", "false"), qid="n", qtype="noul")],
        [],
    ]
    report = summarize(records)
    assert report.n_records == 2
    assert report.exact_match == pytest.approx(0.5)
    assert set(report.questions) == {"q", "n"}
    assert report.mean_macro_f1 == pytest.approx(
        (report.questions["q"].f1 + report.questions["n"].f1) / 2
    )
    assert report.value("exact_match") == report.exact_match
    data = report.to_dict()
    assert data["questions"]["q"]["confusion"]["labels"] == ["a", "b", "c"]
    assert data["mean_macro_f1"] == report.mean_macro_f1


def test_is_better():
    assert is_better("mean_macro_f1", 0.8, None)
    assert is_better("mean_macro_f1", 0.8, 0.7)
    assert not is_better("exact_match", 0.6, 0.7)
    assert is_better("mean_log_loss", 0.3, 0.4)
    assert not is_better("mean_log_loss", 0.5, 0.4)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clef_tune.metrics'`.

- [ ] **Step 3: Implement `clef_tune/metrics.py`**

```python
"""Evaluation metrics: per-question accuracy / F1 / log-loss / MAE, confusion matrices, and
per-record exact match."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Any, Literal

EPS = 1e-12
SelectMetric = Literal["mean_macro_f1", "exact_match", "mean_log_loss"]
HIGHER_IS_BETTER: dict[str, bool] = {
    "mean_macro_f1": True,
    "exact_match": True,
    "mean_log_loss": False,
}


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


@dataclass(frozen=True)
class Prediction:
    question_id: str
    question_type: str
    option_ids: list[str]
    probs: list[float]
    target: list[float]  # one-hot for hard labels, a distribution for soft labels

    @property
    def predicted(self) -> str:
        return self.option_ids[_argmax(self.probs)]

    @property
    def actual(self) -> str:
        return self.option_ids[_argmax(self.target)]

    @property
    def correct(self) -> bool:
        return self.predicted == self.actual

    def log_loss(self) -> float:
        return -sum(
            t * math.log(max(p, EPS)) for p, t in zip(self.probs, self.target, strict=True) if t > 0
        )


@dataclass
class ConfusionMatrix:
    labels: list[str]
    matrix: list[list[int]]  # matrix[true][predicted]


@dataclass
class QuestionMetrics:
    question_id: str
    question_type: str
    n: int
    accuracy: float
    f1: float  # macro-F1 for choice/score; binary F1 with positive "true" for noul
    log_loss: float
    mae: float | None
    confusion: ConfusionMatrix


@dataclass
class MetricsReport:
    questions: dict[str, QuestionMetrics]
    exact_match: float
    n_records: int

    @property
    def mean_macro_f1(self) -> float:
        return fmean(q.f1 for q in self.questions.values()) if self.questions else 0.0

    @property
    def mean_log_loss(self) -> float:
        return fmean(q.log_loss for q in self.questions.values()) if self.questions else 0.0

    def value(self, metric: SelectMetric) -> float:
        return float(getattr(self, metric))

    def to_dict(self) -> dict[str, Any]:
        return {
            "questions": {qid: asdict(metrics) for qid, metrics in self.questions.items()},
            "exact_match": self.exact_match,
            "mean_macro_f1": self.mean_macro_f1,
            "mean_log_loss": self.mean_log_loss,
            "n_records": self.n_records,
        }


def is_better(metric: str, new: float, best: float | None) -> bool:
    if best is None:
        return True
    return new > best if HIGHER_IS_BETTER[metric] else new < best


def confusion_matrix(preds: list[Prediction]) -> ConfusionMatrix:
    labels = list(dict.fromkeys(option for p in preds for option in p.option_ids))
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for p in preds:
        matrix[index[p.actual]][index[p.predicted]] += 1
    return ConfusionMatrix(labels, matrix)


def _f1(label: str, preds: list[Prediction]) -> float:
    tp = sum(1 for p in preds if p.predicted == label and p.actual == label)
    fp = sum(1 for p in preds if p.predicted == label and p.actual != label)
    fn = sum(1 for p in preds if p.predicted != label and p.actual == label)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def _expected_level(values: list[float]) -> float:
    return sum(index * value for index, value in enumerate(values))


def question_metrics(preds: list[Prediction]) -> QuestionMetrics:
    first = preds[0]
    if first.question_type == "noul":
        f1 = _f1("true", preds)
    else:
        classes = sorted({p.actual for p in preds} | {p.predicted for p in preds})
        f1 = fmean(_f1(label, preds) for label in classes)
    mae = None
    if first.question_type == "score":
        mae = fmean(abs(_expected_level(p.probs) - _expected_level(p.target)) for p in preds)
    return QuestionMetrics(
        question_id=first.question_id,
        question_type=first.question_type,
        n=len(preds),
        accuracy=fmean(1.0 if p.correct else 0.0 for p in preds),
        f1=f1,
        log_loss=fmean(p.log_loss() for p in preds),
        mae=mae,
        confusion=confusion_matrix(preds),
    )


def summarize(records: list[list[Prediction]]) -> MetricsReport:
    """records: the labeled-question predictions of each eval record."""
    by_question: dict[str, list[Prediction]] = defaultdict(list)
    for preds in records:
        for p in preds:
            by_question[p.question_id].append(p)
    labeled = [preds for preds in records if preds]
    exact = fmean(1.0 if all(p.correct for p in preds) else 0.0 for preds in labeled) if labeled else 0.0
    return MetricsReport(
        questions={qid: question_metrics(preds) for qid, preds in by_question.items()},
        exact_match=exact,
        n_records=len(labeled),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check --fix .
git add clef_tune/metrics.py tests/test_metrics.py
git commit -F - <<'EOF'
feat: add per-question metrics and confusion matrices
EOF
```

---

### Task 9: Calibration

**Files:**
- Create: `clef_tune/calibration.py`
- Test: `tests/test_calibration.py`

**Interfaces:**
- Consumes: nothing from the project (pure Python).
- Produces:
  - Constants `T_MIN = 0.05`, `T_MAX = 20.0`, `MIN_ROWS = 20`, `ECE_BINS = 15`
  - `softmax(logits, temperature=1.0) -> list[float]`; `nll(logits_rows, targets, temperature) -> float`
  - `fit_temperature(logits_rows, targets) -> float`
  - `expected_calibration_error(probs_rows, targets, n_bins=15) -> float`
  - `temper_probabilities(probs, temperature) -> list[float]` (the client-side formula for SGLang users)
  - `QuestionCalibration(temperature, n, ece_before, ece_after, log_loss_before, log_loss_after, fitted)`
  - `calibrate(logits_by_question: dict[str, list[list[float]]], targets_by_question, min_rows=20) -> dict[str, QuestionCalibration]`; `temperatures(calibration) -> dict[str, float]`

- [ ] **Step 1: Write the failing tests**

`tests/test_calibration.py`:

```python
import random

import pytest

from clef_tune.calibration import (
    calibrate,
    expected_calibration_error,
    fit_temperature,
    softmax,
    temper_probabilities,
    temperatures,
)


def synthetic(n, scale, seed=0):
    """Labels drawn from softmax(z); the model reports scale * z (overconfident when scale > 1)."""
    rng = random.Random(seed)
    logits, targets = [], []
    for _ in range(n):
        true = [rng.gauss(0, 1.5) for _ in range(3)]
        probs = softmax(true)
        label = rng.choices(range(3), weights=probs)[0]
        logits.append([scale * z for z in true])
        targets.append([1.0 if i == label else 0.0 for i in range(3)])
    return logits, targets


def test_fit_temperature_recovers_overconfidence():
    logits, targets = synthetic(4000, scale=3.0)
    assert fit_temperature(logits, targets) == pytest.approx(3.0, rel=0.1)


def test_fit_temperature_near_one_when_calibrated():
    logits, targets = synthetic(4000, scale=1.0, seed=1)
    assert fit_temperature(logits, targets) == pytest.approx(1.0, rel=0.1)


def test_ece_hand_example():
    probs = [[0.9, 0.1], [0.6, 0.4], [0.8, 0.2]]
    targets = [[1, 0], [0, 1], [1, 0]]
    assert expected_calibration_error(probs, targets, n_bins=10) == pytest.approx(0.3)


def test_temper_probabilities_matches_softmax_with_temperature():
    logits = [2.0, -1.0, 0.5]
    assert temper_probabilities(softmax(logits), 2.5) == pytest.approx(softmax(logits, 2.5))


def test_calibrate_improves_ece_and_skips_small_questions():
    logits, targets = synthetic(2000, scale=3.0, seed=2)
    result = calibrate({"big": logits, "small": logits[:5]}, {"big": targets, "small": targets[:5]})
    big, small = result["big"], result["small"]
    assert big.fitted and big.n == 2000
    assert big.ece_after < big.ece_before
    assert big.log_loss_after < big.log_loss_before
    assert not small.fitted and small.temperature == 1.0
    assert small.ece_after == small.ece_before
    assert temperatures(result) == {"big": big.temperature, "small": 1.0}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_calibration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clef_tune.calibration'`.

- [ ] **Step 3: Implement `clef_tune/calibration.py`**

```python
"""Per-question temperature scaling and top-label expected calibration error (ECE).

Rows within one question may have different option counts (multi-task JSONL), so
everything works on lists of rows rather than fixed-width arrays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

T_MIN = 0.05
T_MAX = 20.0
MIN_ROWS = 20
ECE_BINS = 15
_INV_GOLDEN = (math.sqrt(5) - 1) / 2


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    scaled = [z / temperature for z in logits]
    top = max(scaled)
    exps = [math.exp(z - top) for z in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def nll(logits_rows: list[list[float]], targets: list[list[float]], temperature: float) -> float:
    total = 0.0
    for logits, target in zip(logits_rows, targets, strict=True):
        scaled = [z / temperature for z in logits]
        top = max(scaled)
        log_norm = top + math.log(sum(math.exp(z - top) for z in scaled))
        total -= sum(t * (z - log_norm) for z, t in zip(scaled, target, strict=True) if t > 0)
    return total / len(logits_rows)


def fit_temperature(
    logits_rows: list[list[float]],
    targets: list[list[float]],
    t_min: float = T_MIN,
    t_max: float = T_MAX,
    tol: float = 1e-4,
) -> float:
    """Golden-section search on log T. NLL is convex in 1/T, hence unimodal in log T."""
    lo, hi = math.log(t_min), math.log(t_max)

    def loss(u: float) -> float:
        return nll(logits_rows, targets, math.exp(u))

    c = hi - _INV_GOLDEN * (hi - lo)
    d = lo + _INV_GOLDEN * (hi - lo)
    fc, fd = loss(c), loss(d)
    while hi - lo > tol:
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - _INV_GOLDEN * (hi - lo)
            fc = loss(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + _INV_GOLDEN * (hi - lo)
            fd = loss(d)
    return math.exp((lo + hi) / 2)


def expected_calibration_error(
    probs_rows: list[list[float]], targets: list[list[float]], n_bins: int = ECE_BINS
) -> float:
    bins = [[0, 0.0, 0.0] for _ in range(n_bins)]  # count, sum of confidence, sum of correct
    for probs, target in zip(probs_rows, targets, strict=True):
        predicted = max(range(len(probs)), key=probs.__getitem__)
        actual = max(range(len(target)), key=target.__getitem__)
        confidence = probs[predicted]
        index = min(int(confidence * n_bins), n_bins - 1)
        bins[index][0] += 1
        bins[index][1] += confidence
        bins[index][2] += 1.0 if predicted == actual else 0.0
    n = len(probs_rows)
    return sum(abs(conf - correct) / n for count, conf, correct in bins if count)


def temper_probabilities(probs: list[float], temperature: float) -> list[float]:
    """Calibrate already-softmaxed probabilities: p_i^(1/T) / sum_j p_j^(1/T)."""
    powered = [max(p, 1e-300) ** (1 / temperature) for p in probs]
    total = sum(powered)
    return [value / total for value in powered]


@dataclass
class QuestionCalibration:
    temperature: float
    n: int
    ece_before: float
    ece_after: float
    log_loss_before: float
    log_loss_after: float
    fitted: bool


def calibrate(
    logits_by_question: dict[str, list[list[float]]],
    targets_by_question: dict[str, list[list[float]]],
    min_rows: int = MIN_ROWS,
) -> dict[str, QuestionCalibration]:
    result: dict[str, QuestionCalibration] = {}
    for qid, rows in logits_by_question.items():
        targets = targets_by_question[qid]
        fitted = len(rows) >= min_rows
        temperature = fit_temperature(rows, targets) if fitted else 1.0
        before = [softmax(z) for z in rows]
        after = [softmax(z, temperature) for z in rows]
        result[qid] = QuestionCalibration(
            temperature=temperature,
            n=len(rows),
            ece_before=expected_calibration_error(before, targets),
            ece_after=expected_calibration_error(after, targets),
            log_loss_before=nll(rows, targets, 1.0),
            log_loss_after=nll(rows, targets, temperature),
            fitted=fitted,
        )
    return result


def temperatures(calibration: dict[str, QuestionCalibration]) -> dict[str, float]:
    return {qid: item.temperature for qid, item in calibration.items()}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_calibration.py -v`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check --fix .
git add clef_tune/calibration.py tests/test_calibration.py
git commit -F - <<'EOF'
feat: add per-question temperature calibration and ECE
EOF
```

---

### Task 10: Cost, time and memory estimates

**Files:**
- Create: `clef_tune/estimate.py`
- Test: `tests/test_estimate.py`

**Interfaces:**
- Consumes: `config.Config` and `validation.TokenStats` (types only, for `workload_from`).
- Produces:
  - Constants `UTILIZATION = 0.35`, `DDP_EFFICIENCY_PER_DOUBLING = 0.9`, `QLORA_SLOWDOWN = 0.7`, `RANGE_LOW = 0.7`, `RANGE_HIGH = 1.5`, `MEMORY_HEADROOM = 1.15`, `CUDA_CONTEXT_GB = 1.5`, `TIMEOUT_MARGIN = 1.25`, `HEAD_PARAMS = 40e6`
  - `GPUSpec(name, bf16_tflops, memory_gb, supports_bf16=True)`, `GPUS: dict[str, GPUSpec]`
  - `Flavor(name, gpu, count, price_per_hour)`, `HF_FLAVORS: dict[str, Flavor]`
  - `ModelSpec(id, flop_params, total_params, embedding_params, hidden, layers, intermediate, overhead_minutes)`, `MODELS`, `model_spec(model_id) -> ModelSpec | None`
  - `Workload(train_tokens_per_epoch, eval_tokens, epochs, eval_passes, mode, quantization, max_tokens_per_batch, lora_r, gradient_checkpointing=True)`, `workload_from(config, tokens) -> Workload`
  - `training_flops(model, workload)`, `throughput(gpu, count, quantization)`, `memory_needed_gb(model, workload, quantization)`
  - `EstimateRow(flavor, gpu, count, price_per_hour, memory_needed_gb, memory_available_gb, fits, fits_with_4bit, note, hours)` with `hours_low`, `hours_high`, `cost_point`, `cost_low`, `cost_high`
  - `estimate_hardware(model, workload, gpu, count, price, name) -> EstimateRow`, `estimate_hf_flavors(model, workload, prices=None) -> list[EstimateRow]` (fitting rows first, cheapest first), `recommend(rows) -> EstimateRow | None`, `suggest_timeout(row) -> str`, `fetch_hf_prices() -> dict[str, float]`

- [ ] **Step 1: Write the failing tests**

`tests/test_estimate.py`:

```python
import math
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
    h200 = est.estimate_hardware(CLEF, workload(), est.GPUS["H200"], 1, 5.0, "h200")
    assert h200.fits and h200.note is None
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
    monkeypatch.setattr(huggingface_hub, "list_jobs_hardware", lambda: hardware)
    assert est.fetch_hf_prices() == {"h200": pytest.approx(4.998)}

    def offline():
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_estimate.py -v`
Expected: FAIL with `ImportError` for `clef_tune.estimate`.

- [ ] **Step 3: Implement `clef_tune/estimate.py`**

```python
"""Napkin-math estimates of run time, cost and memory fit, shown to users as ranges.

FLOPs: LoRA with gradient checkpointing ~ 6 x params x tokens (forward, recompute, backward
through frozen weights); head-only ~ 2 x params x tokens (forward only); eval ~ 2 x params x
tokens. The trainer replaces these with measured throughput after 20 steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
HEAD_PARAMS = 40e6  # joint schema head, upper bound


@dataclass(frozen=True)
class GPUSpec:
    name: str
    bf16_tflops: float  # approximate dense BF16 tensor throughput from datasheets
    memory_gb: float
    supports_bf16: bool = True


GPUS: dict[str, GPUSpec] = {
    gpu.name: gpu
    for gpu in [
        GPUSpec("T4", 65, 16, supports_bf16=False),
        GPUSpec("A10G", 70, 24),
        GPUSpec("L4", 121, 24),
        GPUSpec("L40S", 362, 48),
        GPUSpec("A100-80GB", 312, 80),
        GPUSpec("H100", 989, 80),
        GPUSpec("H200", 989, 141),
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


# Derived from config.json and safetensors totals: vocab 248320; vision tower ~0.43B params.
MODELS: dict[str, ModelSpec] = {
    "Cloudflare/clef": ModelSpec(
        "Cloudflare/clef", 24.4e9, 27.36e9, 2.54e9, 5120, 64, 17408, 20
    ),
    "Cloudflare/clef-flash": ModelSpec(
        "Cloudflare/clef-flash", 6.95e9, 9.41e9, 2.03e9, 4096, 32, 12288, 10
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


def workload_from(config: Config, tokens: TokenStats) -> Workload:
    train = config.train
    tokens_per_step = train.max_tokens_per_batch * train.grad_accum_steps
    steps_per_epoch = max(1, math.ceil(tokens.train_tokens / tokens_per_step))
    total_steps = steps_per_epoch * train.epochs
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
    trainable = (lora_params + HEAD_PARAMS) * 16  # fp32 weights + grads + two Adam moments
    tokens = workload.max_tokens_per_batch
    layer_working_set = tokens * (18 * model.hidden + 4 * model.intermediate) * 2
    if workload.mode == "head_only":
        activations = layer_working_set
    elif workload.gradient_checkpointing:
        activations = tokens * model.hidden * model.layers * 2 + layer_working_set
    else:
        activations = layer_working_set * model.layers
    return (weights + trainable + activations) / 1e9 * MEMORY_HEADROOM + CUDA_CONTEXT_GB


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
    fits = gpu.supports_bf16 and need <= gpu.memory_gb
    fits_4bit = gpu.supports_bf16 and need_4bit <= gpu.memory_gb
    note = None
    if not gpu.supports_bf16:
        note = "no bf16 support"
    elif not fits and fits_4bit and workload.quantization == "none":
        note = "fits only with 4-bit (QLoRA)"
    elif not fits:
        note = f"needs ~{need:.0f} GB, has {gpu.memory_gb:.0f} GB"
    compute_seconds = training_flops(model, workload) / throughput(gpu, count, workload.quantization)
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


def suggest_timeout(row: EstimateRow) -> str:
    """HF Jobs timeout: the high estimate x 1.25, rounded up to 30 minutes."""
    minutes = math.ceil(row.hours_high * TIMEOUT_MARGIN * 60 / 30) * 30
    if minutes < 60:
        return f"{minutes}m"
    return f"{minutes // 60}h" if minutes % 60 == 0 else f"{minutes / 60:g}h"


def fetch_hf_prices() -> dict[str, float]:
    """Live $/hour per HF Jobs flavor; empty when the Hub can't be reached."""
    try:
        from huggingface_hub import list_jobs_hardware

        hardware = list_jobs_hardware()
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_estimate.py -v`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check --fix .
git add clef_tune/estimate.py tests/test_estimate.py
git commit -F - <<'EOF'
feat: estimate run time, cost and memory fit per GPU and HF Jobs flavor
EOF
```

---

### Task 11: Bundle writer, CLI, example dataset and README

**Files:**
- Create: `clef_tune/bundle.py`, `clef_tune/cli.py`, `clef_tune/__main__.py`, `scripts/make_example.py`, `examples/support_tickets.yaml`, `examples/support_tickets.csv` (generated)
- Modify: `README.md`
- Test: `tests/test_bundle.py`, `tests/test_cli.py`, `tests/test_boundaries.py`

**Interfaces:**
- Consumes: `config.Config`, `config.load_config`; `validation.validate`, `validation.ValidationReport`; `prompt.counter_for_config`; everything public in `estimate` listed in Task 10; `data.URL_PREFIXES`.
- Produces:
  - `bundle.write_bundle(report, config, out_dir) -> Path` (returns the bundle's `config.yaml`), `bundle.zip_bundle(bundle_dir, zip_path) -> Path`
  - `cli.app` (Typer) with commands `validate CONFIG [--bundle DIR]` and `estimate CONFIG [--gpu NAME --gpus N --price USD]`, plus `--version`; exit code 0 = ok, 1 = validation errors / no estimate possible, 2 = invalid config
  - `python -m clef_tune` runs `cli.app`
  - `examples/support_tickets.{csv,yaml}`: 200 synthetic tickets with `department` (choice), `urgency` (score), `outage` (noul)

- [ ] **Step 1: Write the example generator and run it**

`scripts/make_example.py`:

```python
"""Generate examples/support_tickets.csv: 200 synthetic support tickets (deterministic)."""

from __future__ import annotations

import csv
import random
from pathlib import Path

TEMPLATES = {
    "billing": [
        ("Charged twice for {plan}", "I was billed two times this month for the {plan} plan."),
        ("Invoice {n} looks wrong", "The total on invoice {n} doesn't match our contract."),
        ("Card declined", "My card keeps getting declined when I renew {plan}."),
        ("Refund request", "We cancelled {plan} last week but were still charged."),
    ],
    "technical": [
        ("API returning 500 errors", "Since {time} our calls to the API return HTTP 500."),
        ("Dashboard won't load", "The dashboard spins forever after login since {time}."),
        ("Webhook delays", "Webhooks are arriving {n} seconds late."),
        ("SDK crash on startup", "The Python SDK crashes on import after upgrading."),
    ],
    "account": [
        ("Can't reset password", "The password reset email never arrives."),
        ("Add a teammate", "How do I invite a new teammate to our {plan} workspace?"),
        ("Change account owner", "Our admin left; please move ownership to me."),
        ("Close my account", "Please delete our account and all data."),
    ],
}
OUTAGE_SUBJECTS = {"API returning 500 errors", "Dashboard won't load"}
PLANS = ["Starter", "Pro", "Enterprise"]
TIMES = ["this morning", "an hour ago", "yesterday", "10:00 UTC"]
TIERS = ["free", "pro", "enterprise"]
URGENT_WORDS = ["Please fix ASAP.", "This is urgent.", "This is blocking our launch."]
OUT = Path(__file__).resolve().parents[1] / "examples" / "support_tickets.csv"


def main() -> None:
    rng = random.Random(7)
    rows = []
    for i in range(1, 201):
        department = rng.choices(list(TEMPLATES), weights=[0.4, 0.4, 0.2])[0]
        subject, body = rng.choice(TEMPLATES[department])
        fill = {"plan": rng.choice(PLANS), "n": rng.randint(2, 9999), "time": rng.choice(TIMES)}
        tier = rng.choice(TIERS)
        outage = subject in OUTAGE_SUBJECTS and rng.random() < 0.85
        urgent_words = rng.random() < 0.3
        text = body.format(**fill)
        if urgent_words:
            text = f"{text} {rng.choice(URGENT_WORDS)}"
        if outage or (urgent_words and tier == "enterprise"):
            urgency = 2
        elif urgent_words or department == "technical":
            urgency = 1
        else:
            urgency = 0
        rows.append(
            {
                "ticket_id": f"T{i:04d}",
                "subject": subject.format(**fill),
                "body": text,
                "customer_tier": tier,
                "department": department,
                "urgency": urgency,
                "outage": "yes" if outage else "no",
            }
        )
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
```

`examples/support_tickets.yaml`:

```yaml
run_name: support-tickets
model:
  id: Cloudflare/clef-flash
data:
  train: support_tickets.csv
  eval_fraction: 0.15
  id_column: ticket_id
  state_columns: [subject, body, customer_tier]
schema:
  questions:
    department:
      type: choice
      instructions: Which team should handle this ticket?
      criteria:
        billing: Payments, invoices and refunds
        technical: Bugs, errors, outages and integrations
        account: Logins, teammates, ownership and closing accounts
    urgency:
      type: score
      instructions: How quickly does this ticket need a response?
      criteria: [Can wait, This week, Today]
    outage:
      type: noul
      instructions: Does the ticket report a service outage?
```

Run: `uv run python scripts/make_example.py`
Expected: `examples/support_tickets.csv` exists with a header and 200 data rows.

- [ ] **Step 2: Write the failing tests**

`tests/test_bundle.py`:

```python
import json
import zipfile

import pytest
from PIL import Image

from clef_tune.bundle import write_bundle, zip_bundle
from clef_tune.config import load_config
from clef_tune.validation import validate
from tests.helpers import CharCounter, make_config, ticket_rows, write_csv


def test_bundle_round_trip_preserves_split_and_labels(tmp_path):
    rows = ticket_rows(40)
    for row in rows:
        row["receipt"] = ""
    Image.new("RGB", (64, 64)).save(tmp_path / "r.png")
    rows[0]["receipt"] = "r.png"
    rows[1]["receipt"] = "r.png"
    write_csv(tmp_path / "src" / "train.csv", rows)
    (tmp_path / "src" / "r.png").write_bytes((tmp_path / "r.png").read_bytes())
    cfg = make_config(
        tmp_path / "src",
        data={"train": "train.csv", "id_column": "id", "image_columns": ["receipt"]},
    )
    report = validate(cfg, CharCounter())
    assert report.ok

    config_path = write_bundle(report, cfg, tmp_path / "bundle")
    bundled = load_config(config_path)
    assert bundled.data.train == "train.jsonl" and bundled.data.eval == "eval.jsonl"
    assert bundled.get_schema() == cfg.get_schema()
    again = validate(bundled, CharCounter())
    assert again.ok
    assert [r.id for r in again.train] == [r.id for r in report.train]
    assert [r.id for r in again.eval] == [r.id for r in report.eval]
    assert [r.labels for r in again.train] == [r.labels for r in report.train]

    first = json.loads((tmp_path / "bundle" / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "questions" not in first  # rows that use the config schema don't repeat it
    images = list((tmp_path / "bundle" / "images").iterdir())
    assert len(images) == 1  # the same file is copied once
    with_image = [r for r in again.train + again.eval if r.images]
    assert all(r.images == [str(images[0].resolve())] for r in with_image)

    archive = zip_bundle(tmp_path / "bundle", tmp_path / "bundle.zip")
    assert "config.yaml" in zipfile.ZipFile(archive).namelist()


def test_bundle_refuses_reports_with_errors(tmp_path):
    rows = ticket_rows(20)
    rows[0]["department"] = "biling"
    write_csv(tmp_path / "train.csv", rows)
    cfg = make_config(tmp_path, data={"train": "train.csv", "id_column": "id"})
    report = validate(cfg, CharCounter())
    with pytest.raises(ValueError, match="validation has errors"):
        write_bundle(report, cfg, tmp_path / "bundle")
```

`tests/test_cli.py`:

```python
from pathlib import Path

import pytest
from typer.testing import CliRunner

from clef_tune import cli
from tests.helpers import CharCounter, ticket_rows, write_csv

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "support_tickets.yaml"
runner = CliRunner()


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(cli, "counter_for_config", lambda config: CharCounter())
    monkeypatch.setattr(cli, "fetch_hf_prices", lambda: {})


def test_version():
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0 and "clef-tune" in result.output


def test_validate_example_passes_and_writes_bundle(tmp_path):
    result = runner.invoke(cli.app, ["validate", str(EXAMPLE), "--bundle", str(tmp_path / "b")])
    assert result.exit_code == 0, result.output
    assert "Validation passed" in result.output
    assert (tmp_path / "b" / "config.yaml").exists()


def test_validate_reports_errors_with_rows(tmp_path):
    rows = ticket_rows(20)
    rows[1]["department"] = "biling"
    write_csv(tmp_path / "train.csv", rows)
    config = tmp_path / "config.yaml"
    config.write_text(
        "run_name: t\n"
        "data: {train: train.csv, id_column: id}\n"
        "schema:\n"
        "  questions:\n"
        "    department: {type: choice, criteria: {billing: x, technical: y}}\n"
        "    urgency: {type: score, criteria: [a, b, c]}\n"
        "    outage: {type: noul}\n",
        encoding="utf-8",
    )
    result = runner.invoke(cli.app, ["validate", str(config)])
    assert result.exit_code == 1
    assert "'biling' is not an option" in result.output and "rows 3" in result.output


def test_invalid_config_exits_2(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("run_name: t\ndata: {train: x.csv}\ntrain: {epoch: 3}\n", encoding="utf-8")
    result = runner.invoke(cli.app, ["validate", str(config)])
    assert result.exit_code == 2 and "Invalid config" in result.output


def test_estimate_example_lists_hf_flavors():
    result = runner.invoke(cli.app, ["estimate", str(EXAMPLE)])
    assert result.exit_code == 0, result.output
    assert "l40sx1" in result.output and "Recommended" in result.output


def test_estimate_custom_gpu():
    result = runner.invoke(cli.app, ["estimate", str(EXAMPLE), "--gpu", "H100", "--price", "2.5"])
    assert result.exit_code == 0, result.output
    assert "1x H100" in result.output and "$" in result.output


def test_estimate_rejects_unknown_gpu():
    result = runner.invoke(cli.app, ["estimate", str(EXAMPLE), "--gpu", "TPU"])
    assert result.exit_code != 0
```

`tests/test_boundaries.py`:

```python
import subprocess
import sys

NO_TORCH_MODULES = [
    "clef_tune.cli",
    "clef_tune.bundle",
    "clef_tune.validation",
    "clef_tune.estimate",
    "clef_tune.calibration",
    "clef_tune.metrics",
]


def test_core_modules_never_import_torch():
    imports = "; ".join(f"import {module}" for module in NO_TORCH_MODULES)
    check = "bad = [m for m in ('torch', 'transformers', 'peft') if m in sys.modules]"
    code = f"import sys; {imports}; {check}; assert not bad, bad"
    subprocess.run([sys.executable, "-c", code], check=True)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_bundle.py tests/test_cli.py tests/test_boundaries.py -v`
Expected: FAIL with `ModuleNotFoundError` for `clef_tune.bundle` and `clef_tune.cli`.

- [ ] **Step 4: Implement `clef_tune/bundle.py`**

```python
"""Write a portable training bundle: normalized train/eval JSONL, copied images, and a
self-contained config.yaml. Bundles are what the UI downloads, what HF Jobs mounts, and what
any GPU machine trains from."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from clef_tune.config import Config
from clef_tune.data import URL_PREFIXES, Record
from clef_tune.validation import ValidationReport


def write_bundle(report: ValidationReport, config: Config, out_dir: str | Path) -> Path:
    if not report.ok:
        raise ValueError("cannot write a bundle while validation has errors")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = config.get_schema()
    copied: dict[str, str] = {}

    def row(record: Record) -> dict:
        data = record.to_json()
        if schema is not None and record.questions == schema.questions:
            del data["questions"]
        if record.images:
            data["images"] = [_bundle_image(path, out_dir / "images", copied) for path in record.images]
        return data

    for name, records in (("train.jsonl", report.train), ("eval.jsonl", report.eval)):
        with open(out_dir / name, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(row(record), ensure_ascii=False) + "\n")

    bundled = config.to_dict()
    bundled["data"] = {"train": "train.jsonl", "eval": "eval.jsonl"}
    bundled.pop("schema_file", None)
    if schema is not None:
        bundled["schema"] = schema.to_dict()
    bundled.get("output", {}).pop("dir", None)
    config_path = out_dir / "config.yaml"
    Config.model_validate(bundled).save(config_path)
    return config_path


def _bundle_image(path: str, images_dir: Path, copied: dict[str, str]) -> str:
    if path.startswith(URL_PREFIXES):
        return path
    if path not in copied:
        source = Path(path)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        target = images_dir / f"{digest}{source.suffix.lower()}"
        images_dir.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        copied[path] = f"images/{target.name}"
    return copied[path]


def zip_bundle(bundle_dir: str | Path, zip_path: str | Path) -> Path:
    base = str(Path(zip_path).with_suffix(""))
    return Path(shutil.make_archive(base, "zip", root_dir=bundle_dir))
```

- [ ] **Step 5: Implement `clef_tune/cli.py` and `clef_tune/__main__.py`**

`clef_tune/cli.py`:

```python
"""clef-tune command line."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from clef_tune import __version__
from clef_tune.bundle import write_bundle
from clef_tune.config import Config, load_config
from clef_tune.estimate import (
    GPUS,
    EstimateRow,
    estimate_hardware,
    estimate_hf_flavors,
    fetch_hf_prices,
    model_spec,
    recommend,
    suggest_timeout,
    workload_from,
)
from clef_tune.prompt import TokenCounter, counter_for_config
from clef_tune.validation import ValidationReport
from clef_tune.validation import validate as run_validation

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Fine-tune Cloudflare's Clef decision models.",
)
console = Console(soft_wrap=True)

ConfigArg = Annotated[
    Path, typer.Argument(exists=True, dir_okay=False, help="Path to config.yaml.")
]


def _show_version(value: bool) -> None:
    if value:
        console.print(f"clef-tune {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_show_version, is_eager=True, help="Show the version."),
    ] = False,
) -> None:
    """Fine-tune Cloudflare's Clef decision models."""


def _load(path: Path) -> Config:
    try:
        return load_config(path)
    except (ValueError, OSError) as exc:
        console.print(f"[red]Invalid config:[/] {escape(str(exc))}")
        raise typer.Exit(2) from exc


def _counter(config: Config) -> TokenCounter | None:
    try:
        return counter_for_config(config)
    except Exception as exc:  # offline or missing tokenizer: validation continues without it
        console.print(f"[yellow]Could not load the Clef tokenizer:[/] {escape(str(exc))}")
        return None


def _validate(path: Path) -> tuple[Config, ValidationReport]:
    config = _load(path)
    return config, run_validation(config, _counter(config))


def print_report(report: ValidationReport) -> None:
    console.print(f"[bold]{len(report.train)}[/] training rows, [bold]{len(report.eval)}[/] eval rows")
    table = Table("question", "type", "train labels", "eval labels", "train classes")
    for stats in report.questions.values():
        classes = ", ".join(f"{option}: {n}" for option, n in stats.train_counts.items())
        table.add_row(
            escape(stats.question_id),
            stats.type,
            str(sum(stats.train_counts.values())),
            str(stats.eval_labeled),
            escape(classes),
        )
    console.print(table)
    if report.tokens is not None:
        t = report.tokens
        console.print(
            f"Tokens per row: p50 {t.p50}, p95 {t.p95}, max {t.max}; "
            f"{t.truncated_rows} rows truncated"
        )
    for issue in report.errors:
        console.print(f"[red]error[/] {escape(str(issue))}")
    for issue in report.warnings:
        console.print(f"[yellow]warning[/] {escape(str(issue))}")
    if report.ok:
        console.print("[green]Validation passed.[/]")
    else:
        console.print(f"[red]Validation failed with {len(report.errors)} error(s).[/]")


@app.command()
def validate(
    config: ConfigArg,
    bundle: Annotated[
        Path | None,
        typer.Option("--bundle", help="Also write a portable training bundle to this directory."),
    ] = None,
) -> None:
    """Check data and schema; report label and token statistics."""
    cfg, report = _validate(config)
    print_report(report)
    if not report.ok:
        raise typer.Exit(1)
    if bundle is not None:
        console.print(f"Bundle written: {write_bundle(report, cfg, bundle)}")


def _money(value: float | None) -> str:
    return "-" if value is None else f"${value:,.2f}"


def _print_estimates(rows: list[EstimateRow]) -> None:
    table = Table("hardware", "fits", "memory (GB)", "time", "cost")
    for row in rows:
        if row.fits:
            fits = "yes"
        elif row.fits_with_4bit and row.note == "fits only with 4-bit (QLoRA)":
            fits = "4-bit only"
        else:
            fits = f"no ({row.note})"
        cost = "-" if row.cost_low is None else f"{_money(row.cost_low)}-{_money(row.cost_high)}"
        table.add_row(
            row.flavor,
            fits,
            f"{row.memory_needed_gb:.0f} / {row.memory_available_gb:.0f}",
            f"{row.hours_low:.1f}-{row.hours_high:.1f} h",
            cost,
        )
    console.print(table)


@app.command()
def estimate(
    config: ConfigArg,
    gpu: Annotated[
        str | None, typer.Option(help=f"Your GPU type: {', '.join(GPUS)}.")
    ] = None,
    gpus: Annotated[int, typer.Option(min=1, help="Number of GPUs (DDP).")] = 1,
    price: Annotated[
        float | None, typer.Option(help="Your $/hour for --gpu (other infra).")
    ] = None,
) -> None:
    """Estimate run time, cost and memory fit per hardware option."""
    if gpu is not None and gpu not in GPUS:
        raise typer.BadParameter(f"choose one of {', '.join(GPUS)}", param_hint="--gpu")
    cfg, report = _validate(config)
    if not report.ok:
        console.print("[yellow]Validation has errors; estimating anyway.[/]")
    if report.tokens is None:
        console.print("[red]Token statistics are unavailable, so there is nothing to estimate.[/]")
        raise typer.Exit(1)
    spec = model_spec(cfg.model.id)
    if spec is None:
        console.print(
            f"[red]No size data for {escape(cfg.model.id)}; estimates support "
            "Cloudflare/clef and Cloudflare/clef-flash.[/]"
        )
        raise typer.Exit(1)
    workload = workload_from(cfg, report.tokens)
    if gpu is not None:
        rows = [estimate_hardware(spec, workload, GPUS[gpu], gpus, price, f"{gpus}x {gpu}")]
    else:
        rows = estimate_hf_flavors(spec, workload, fetch_hf_prices())
    _print_estimates(rows)
    best = recommend(rows)
    if best is not None:
        console.print(
            f"Recommended: [bold]{best.flavor}[/] (HF Jobs timeout {suggest_timeout(best)})"
        )
    console.print(
        "[dim]Estimates are rough ranges (-30%/+50%). Training reports a measured ETA "
        "after 20 steps.[/]"
    )
```

`clef_tune/__main__.py`:

```python
from clef_tune.cli import app

app()
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_bundle.py tests/test_cli.py tests/test_boundaries.py -v`
Expected: all PASS. If `test_estimate_example_lists_hf_flavors` fails on `l40sx1`, print `result.output`: with the example data, Clef-Flash bf16 needs about 34 GB, so `l40sx1` (48 GB) should be listed and fit.

- [ ] **Step 7: Try the real CLI against the real tokenizer**

Run: `uv run clef-tune validate examples/support_tickets.yaml && uv run clef-tune estimate examples/support_tickets.yaml`
Expected: `Validation passed.` (calibration warnings are fine), token stats of a few hundred tokens per row, and an estimate table with a "Recommended:" line. This downloads the Clef-Flash tokenizer once.

- [ ] **Step 8: Update the README**

Replace `README.md` with:

````markdown
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
````

- [ ] **Step 9: Run the full suite, lint, and commit**

Run: `uv run ruff format . && uv run ruff check --fix . && uv run pytest -v`
Expected: all tests PASS (network-marked tests download the tokenizer once).

```bash
git add clef_tune/bundle.py clef_tune/cli.py clef_tune/__main__.py scripts examples README.md tests/test_bundle.py tests/test_cli.py tests/test_boundaries.py
git commit -F - <<'EOF'
feat: add bundle writer, validate/estimate CLI and example dataset
EOF
```

---

## Spec coverage (Plan 1 portion)

| Spec section | Covered by |
|---|---|
| §3 no-torch modules: schema, data, config, metrics, calibration, estimate | Tasks 2-10; boundary test in Task 11 |
| §3 CLI `validate`, `estimate` | Task 11 |
| §4.1 schema rules | Task 2 |
| §4.2 column mapping, stratified split | Tasks 4, 5 |
| §4.3 label parsing incl. soft labels | Task 3 |
| §4.4 native JSONL with per-row questions | Task 5 |
| §4.5 normalized record and bundle | Tasks 5, 11 |
| §4.6 validation errors, warnings, stats; image tokens | Tasks 6, 7 |
| §4.7 auto-mapping | Plan 4 (UI), which consumes `read_rows` and `Schema` |
| §5.4 metrics and confusion matrices; §5.5 calibration math | Tasks 8, 9 (wired into training in Plan 2) |
| §7 estimates, memory fit, ranges, timeout | Task 10 |
| §10 config reference and presets | Task 4 |
| §11 tier 1 unit tests | every task |
| §12 uv project, hatch-vcs, CI, example data, README | Tasks 1, 11 |
| §5-6, §8, §9 (training, outputs, HF Jobs, UI) | Plans 2-4 |
