"""Event handlers for the Data, Schema, Validate and Configure steps. Plain functions over a
Session that return small views of HTML and lists; ui/app.py maps them onto components."""

from __future__ import annotations

import json
import math
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from clef_tune.automap import propose_column
from clef_tune.bundle import write_bundle, zip_bundle
from clef_tune.card import CALIBRATION_CAVEAT, WEIGHTING_NOTE
from clef_tune.config import PRESETS, Config
from clef_tune.data import Issue
from clef_tune.estimate import estimate_hf_flavors, model_spec, recommend, workload_from
from clef_tune.labels import LabelError, parse_label
from clef_tune.prompt import prompt_pieces
from clef_tune.schema import Question, Schema, SchemaError
from clef_tune.ui import charts, fragments
from clef_tune.ui.services import Services
from clef_tune.ui.session import (
    OptionDraft,
    QuestionDraft,
    Session,
    UploadError,
    load_example,
    load_upload,
)
from clef_tune.ui.settings import Settings
from clef_tune.validation import ValidationReport, validate

STEPS = ("Data", "Validate", "Configure", "Launch")
PREVIEW_PAGE = 50
CELL_CHARS = 200
UNKNOWN_SHOWN = 8
SAMPLE_VALUES = 3
SAMPLE_CHARS = 80
# What a column can be for (flat files): the label shown, and the role it sets.
ROLE_CHOICES = (
    ("State: Clef reads it", "state"),
    ("Question: Clef answers it", "question"),
    ("ID", "id"),
    ("Image", "image"),
    ("Skip", "skip"),
)
ROLES = tuple(role for _, role in ROLE_CHOICES)
MODELS = {"Cloudflare/clef-flash": "Clef-Flash · 9B", "Cloudflare/clef": "Clef · 27B"}
PRESET_LABELS = {"quick": "Quick test", "balanced": "Balanced", "thorough": "Thorough"}
PRESET_HELP = {
    "quick": "Trains only the decision head for one epoch on prompts up to 2048 tokens. "
    "The cheapest way to check everything works end to end.",
    "balanced": "Trains LoRA adapters (rank 16) and the head for three epochs. "
    "A good first real run.",
    "thorough": "Trains LoRA adapters (rank 32) and the head for up to five epochs, "
    "stopping early once the eval score stops improving.",
}
StepState = Literal["todo", "done", "warning", "error"]
_MISSING = object()


# ---------------------------------------------------------------------------- errors


def error_text(exc: Exception) -> str:
    """A one-line message for a config or schema error, naming the field."""
    if isinstance(exc, ValidationError):
        parts = []
        for error in exc.errors():
            location = ".".join(str(part) for part in error["loc"])
            message = str(error["msg"]).removeprefix("Value error, ")
            parts.append(f"{location}: {message}" if location else message)
        return "; ".join(parts)
    return str(exc)


# ---------------------------------------------------------------------------- data step


@dataclass
class DataView:
    summary: str  # the file bar, or an empty state before any data
    headers: list[str]
    rows: list[list[str]]
    page: int
    page_label: str
    error: str | None


def _cell(value: Any) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= CELL_CHARS else text[:CELL_CHARS] + "…"


def column_roles(session: Session) -> dict[str, str]:
    """What each column is for: one of ROLES (native JSONL fields get their own names)."""
    table, draft = session.table, session.draft
    if table is None or draft is None:
        return {}
    if draft.native:
        native = {"id": "id", "state": "state", "labels": "question", "images": "image"}
        return {column: native.get(column, "skip") for column in table.columns}
    questions = {q.column for q in draft.questions}
    roles = {}
    for column in table.columns:
        if column == draft.id_column:
            roles[column] = "id"
        elif column in questions:
            roles[column] = "question"
        elif column in draft.image_columns:
            roles[column] = "image"
        elif column in draft.state_columns:
            roles[column] = "state"
        else:
            roles[column] = "skip"
    return roles


@dataclass
class ColumnRow:
    name: str
    samples: str  # a few distinct values, shortened
    role: str


def _samples(session: Session, column: str) -> str:
    values: list[str] = []
    for value in session.table.values(column) if column != "labels" else []:
        text = _cell(value).strip()
        if text and text not in values:
            values.append(text)
        if len(values) == SAMPLE_VALUES:
            break
    if column == "labels" and session.draft.native:
        values = [q.id for q in session.draft.questions]
    joined = ", ".join(values)
    return joined if len(joined) <= SAMPLE_CHARS else joined[: SAMPLE_CHARS - 1] + "…"


def column_rows(session: Session) -> list[ColumnRow]:
    """One row per column, in file order, for the column roles list."""
    roles = column_roles(session)
    return [ColumnRow(name, _samples(session, name), role) for name, role in roles.items()]


def set_role(session: Session, column: str, role: str) -> str | None:
    """Make a column the state, a question, the ID, an image or nothing. A column has one role,
    so a label column never ends up in the state Clef reads."""
    draft, table = session.draft, session.table
    if role not in ROLES:
        return f"Unknown role {role!r}; choose one of {', '.join(ROLES)}"
    if draft.native:
        return "In this JSONL file each row brings its own state and labels"
    if draft.id_column == column:
        draft.id_column = None
    draft.questions = [q for q in draft.questions if q.column != column or role == "question"]
    in_order = lambda keep: [c for c in table.columns if c in keep]  # noqa: E731
    state = set(draft.state_columns) - {column}
    images = set(draft.image_columns) - {column}
    if role == "state":
        state.add(column)
    elif role == "image":
        images.add(column)
    elif role == "id":
        draft.id_column = column
    draft.state_columns, draft.image_columns = in_order(state), in_order(images)
    if role == "question" and not any(q.column == column for q in draft.questions):
        return add_question(session, column)
    return None


def data_view(session: Session, page: int = 0, error: str | None = None) -> DataView:
    table = session.table
    if table is None:
        empty = fragments.empty_state(
            "No data yet",
            "Upload a CSV, TSV or JSONL file (or a zip with your images), "
            "or load the example to look around.",
        )
        return DataView(empty, [], [], 0, "", error)
    total = len(table.rows)
    pages = max(1, math.ceil(total / PREVIEW_PAGE))
    page = min(max(page, 0), pages - 1)
    start = page * PREVIEW_PAGE
    chunk = table.rows[start : start + PREVIEW_PAGE]
    rows = [[_cell(row.get(column)) for column in table.columns] for row in chunk]
    label = f"Rows {start + 1}–{start + len(chunk)} of {total:,}" if total else "No rows"
    columns = len(table.columns)
    summary = (
        '<div class="ct-filebar"><span class="ct-file-icon" aria-hidden="true">▤</span>'
        f'<span class="ct-file-name">{fragments.esc(table.path.name)}</span>'
        f'<span class="ct-file-meta">{total:,} rows · {columns} column'
        f"{'s' if columns != 1 else ''}</span></div>"
    )
    return DataView(summary, list(table.columns), rows, page, label, error)


def upload(session: Session, path: str | None, settings: Settings) -> DataView:
    if not path:
        return data_view(session)
    try:
        load_upload(session, Path(path), settings)
    except (UploadError, OSError) as exc:
        return data_view(session, error=f"Could not use {Path(path).name}: {exc}")
    return data_view(session)


def example(session: Session) -> DataView:
    load_example(session)
    return data_view(session)


# ---------------------------------------------------------------------------- schema step


@dataclass
class QuestionCard:
    id: str
    column: str
    type: str
    instructions: str
    options: list[list[Any]]  # [option id, description, label count]
    unknown: str  # a callout listing values that are not options, or ""


@dataclass
class SchemaView:
    columns: list[str]
    id_column: str | None
    state_columns: list[str]
    image_columns: list[str]
    free_columns: list[str]  # columns that can become a question
    cards: list[QuestionCard]
    preview: str
    error: str | None


def option_counts(
    session: Session, question: QuestionDraft
) -> tuple[dict[str, int], dict[str, int]]:
    """Labels per option, and the values that are not an option (with their counts)."""
    counts = {option.id: 0 for option in question.options}
    try:
        parsed = Question.from_dict(question.id, question.to_clef())
    except SchemaError:
        return counts, {}
    if question.type == "score":
        id_for = {str(i): option.id for i, option in enumerate(question.options)}
    else:
        id_for = {option.id: option.id for option in question.options}
    unknown: Counter[str] = Counter()
    for raw in session.table.values(question.column) if session.table else []:
        try:
            label, _ = parse_label(parsed, raw)
        except LabelError:
            unknown[str(raw).strip()] += 1
            continue
        if label is not None:
            key = id_for.get(label.top(), label.top())
            counts[key] = counts.get(key, 0) + 1
    return counts, dict(unknown)


def _unknown_callout(unknown: dict[str, int]) -> str:
    if not unknown:
        return ""
    ordered = sorted(unknown.items(), key=lambda item: (-item[1], item[0]))
    shown = ", ".join(f"{value} ({count})" for value, count in ordered[:UNKNOWN_SHOWN])
    more = f" and {len(ordered) - UNKNOWN_SHOWN} more" if len(ordered) > UNKNOWN_SHOWN else ""
    rows = sum(unknown.values())
    return fragments.callout(
        "warning",
        f"{rows:,} rows have a value that is not an option: {shown}{more}. "
        "Add them as options, or fix them in your data.",
    )


def _draft_schema(session: Session) -> Schema:
    questions = {q.id: q.to_clef() for q in session.draft.questions}
    return Schema.from_dict({"questions": questions})


def _row_state(session: Session, row: dict[str, Any]) -> tuple[Any, int]:
    draft = session.draft
    if draft.native:
        images = row.get("images") or []
        return row.get("state"), len(images) if isinstance(images, list) else 1
    values = {column: row.get(column) for column in draft.state_columns}
    state = values[draft.state_columns[0]] if len(draft.state_columns) == 1 else values
    images = sum(1 for column in draft.image_columns if str(row.get(column) or "").strip())
    return state, images


def _preview(session: Session, row_index: int) -> str:
    rows = session.table.rows
    if not rows:
        return fragments.empty_state("No rows to preview", "The file has a header but no rows.")
    try:
        schema = _draft_schema(session)
    except SchemaError as exc:
        return fragments.callout("error", f"Fix the questions to see the prompt: {exc}")
    row = rows[min(max(row_index, 0), len(rows) - 1)]
    state, images = _row_state(session, row)
    return fragments.prompt_preview(prompt_pieces(state, schema.questions, images))


def schema_view(session: Session, preview_row: int = 0) -> SchemaView:
    table, draft = session.table, session.draft
    if table is None or draft is None:
        empty = fragments.empty_state("No data yet", "Load data on the Data step first.")
        return SchemaView([], None, [], [], [], [], empty, None)
    cards = []
    for question in draft.questions:
        counts, unknown = option_counts(session, question)
        options = [[o.id, o.description, counts.get(o.id, 0)] for o in question.options]
        cards.append(
            QuestionCard(
                question.id,
                question.column,
                question.type,
                question.instructions,
                options,
                _unknown_callout(unknown),
            )
        )
    used = {q.column for q in draft.questions} | {draft.id_column}
    free = [column for column in table.columns if column not in used]
    if draft.native:
        free = [qid for qid in _native_label_keys(session) if qid not in used]
    error = None
    try:
        session.config()
    except ValueError as exc:
        error = error_text(exc)
    return SchemaView(
        list(table.columns),
        draft.id_column,
        list(draft.state_columns),
        list(draft.image_columns),
        free,
        cards,
        _preview(session, preview_row),
        error,
    )


def _native_label_keys(session: Session) -> list[str]:
    keys: dict[str, None] = {}
    for row in session.table.rows:
        if isinstance(row.get("labels"), dict):
            keys.update(dict.fromkeys(str(key) for key in row["labels"]))
    return list(keys)


def add_question(session: Session, column: str) -> str | None:
    draft, table = session.draft, session.table
    if any(q.column == column or q.id == column for q in draft.questions):
        return f"{column} is already a question"
    proposal = None if draft.native else propose_column(table.rows, column)
    if proposal is not None:
        question = QuestionDraft(
            column, column, proposal.type, options=[OptionDraft(o) for o in proposal.options]
        )
    else:
        options = [OptionDraft(value) for value in table.observed(column)]
        question = QuestionDraft(column, column, "choice", options=options)
    draft.questions.append(question)
    draft.state_columns = [c for c in draft.state_columns if c != column]
    draft.image_columns = [c for c in draft.image_columns if c != column]
    return None


def remove_question(session: Session, qid: str) -> None:
    session.draft.questions = [q for q in session.draft.questions if q.id != qid]


def _options_from_rows(rows: list[list[Any]]) -> tuple[list[OptionDraft], str | None]:
    options: list[OptionDraft] = []
    for row in rows:
        if not row or row[0] is None or not str(row[0]).strip():
            continue
        description = "" if len(row) < 2 or row[1] is None else str(row[1])
        options.append(OptionDraft(str(row[0]).strip(), description))
    ids = Counter(option.id for option in options)
    repeated = [oid for oid, count in ids.items() if count > 1]
    if repeated:
        return options, f"Option IDs must be unique ({', '.join(repeated)} appears twice)"
    return options, None


def update_question(
    session: Session,
    qid: str,
    *,
    new_id: str | None = None,
    type: str | None = None,
    instructions: str | None = None,
    options: list[list[Any]] | None = None,
) -> str | None:
    draft = session.draft
    question = draft.question(qid)
    if question is None:
        return f"There is no question {qid}"
    if new_id is not None:
        new_id = new_id.strip()
        if not new_id:
            return "A question needs an ID"
        if new_id != qid and draft.question(new_id) is not None:
            return f"Another question already uses the ID {new_id}"
        question.id = new_id
    if type is not None and type != question.type:
        question.set_type(type, session.table.observed(question.column))
    if instructions is not None:
        question.instructions = instructions
    if options is not None:
        parsed, error = _options_from_rows(options)
        if error:
            return error
        question.options = parsed
    return None


# ---------------------------------------------------------------------------- validate step


def refresh(session: Session, services: Services) -> None:
    """Validate the current draft and estimate its cost. Runs after every edit."""
    session.estimates = []
    if session.table is None or session.draft is None:
        session.report = None
        return
    try:
        config = session.config()
    except ValueError as exc:
        session.report = ValidationReport(issues=[Issue("error", f"Setup: {error_text(exc)}")])
        return
    counter, counter_error = None, None
    try:
        counter = services.counter(config)
    except Exception as exc:  # offline or no tokenizer: validate without token statistics
        counter_error = str(exc) or type(exc).__name__
    report = validate(config, counter, counter_error=counter_error)
    session.report = report
    spec = model_spec(config.model.id)
    if report.tokens is not None and spec is not None:
        workload = workload_from(config, report.tokens)
        session.estimates = estimate_hf_flavors(spec, workload, services.prices())


@dataclass
class ValidateView:
    tiles: str
    issues: str
    distributions: str
    histogram: str
    ok: bool


def validate_view(session: Session) -> ValidateView:
    report = session.report
    if report is None:
        empty = fragments.empty_state(
            "Not checked yet", "Load data and set up your questions; the checks run on their own."
        )
        return ValidateView("", empty, "", "", False)
    tiles = [
        ("Train rows", f"{len(report.train):,}"),
        ("Eval rows", f"{len(report.eval):,}"),
        ("Questions", str(len(report.questions))),
        ("Errors", str(len(report.errors))),
        ("Warnings", str(len(report.warnings))),
    ]
    histogram = ""
    tokens = report.tokens
    if tokens is not None:
        tiles += [
            ("Median prompt", f"{tokens.p50:,} tokens"),
            ("95th percentile", f"{tokens.p95:,} tokens"),
            ("Longest prompt", f"{tokens.max:,} tokens"),
        ]
        try:
            cutoff = session.config().train.max_length
        except ValueError:
            cutoff = None
        histogram = charts.histogram(tokens.train_lengths + tokens.eval_lengths, cutoff=cutoff)
    sections = [
        f'<section class="ct-dist"><h4>{fragments.esc(qid)}</h4>'
        f"{charts.bar_chart(stats.train_counts, missing=stats.train_missing)}</section>"
        for qid, stats in report.questions.items()
    ]
    distributions = f'<div class="ct-dists">{"".join(sections)}</div>' if sections else ""
    return ValidateView(
        fragments.stat_tiles(tiles),
        fragments.issue_list(report.issues),
        distributions,
        histogram,
        report.ok,
    )


# ---------------------------------------------------------------------------- configure step


@dataclass(frozen=True)
class FieldSpec:
    path: str
    label: str
    kind: Literal["int", "float", "bool", "choice", "text", "optional_int"]
    choices: tuple[str, ...] = ()
    help: str = ""


ADVANCED_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        "train.mode",
        "What to train",
        "choice",
        ("lora", "head_only"),
        "lora trains small adapters in the language model plus the decision head; head_only "
        "trains just the head, which is faster and cheaper but learns less.",
    ),
    FieldSpec(
        "train.quantization",
        "Quantization",
        "choice",
        ("none", "4bit"),
        "4bit loads the model in 4-bit (QLoRA) so it fits smaller GPUs, at some cost in speed.",
    ),
    FieldSpec("train.lora.r", "LoRA rank", "int", help="Adapter size; higher learns more."),
    FieldSpec(
        "train.lora.alpha", "LoRA alpha", "int", help="Scales adapter updates; usually 2 × rank."
    ),
    FieldSpec("train.lora.dropout", "LoRA dropout", "float", help="Dropout on adapter inputs."),
    FieldSpec(
        "train.lora_lr", "LoRA learning rate", "float", help="Step size for the LoRA adapters."
    ),
    FieldSpec("train.head_lr", "Head learning rate", "float", help="Step size for the head."),
    FieldSpec("train.epochs", "Epochs", "int", help="Passes over the training rows."),
    FieldSpec(
        "train.max_length",
        "Max prompt length",
        "int",
        help="Prompts longer than this (256–16384 tokens) have their state cut to fit.",
    ),
    FieldSpec(
        "train.max_tokens_per_batch",
        "Tokens per batch",
        "int",
        help="Rows are packed into batches of up to this many tokens; lower it if a run runs "
        "out of memory.",
    ),
    FieldSpec(
        "train.grad_accum_steps",
        "Gradient accumulation",
        "int",
        help="Batches added up for each optimizer step.",
    ),
    FieldSpec(
        "train.warmup_ratio",
        "Warmup share",
        "float",
        help="Share of steps spent ramping up the learning rate.",
    ),
    FieldSpec("train.weight_decay", "Weight decay", "float", help="AdamW weight decay."),
    FieldSpec(
        "train.label_smoothing",
        "Label smoothing",
        "float",
        help="Softens the targets; 0 turns it off.",
    ),
    FieldSpec(
        "train.class_weighting", "Class weighting", "choice", ("none", "balanced"), WEIGHTING_NOTE
    ),
    FieldSpec(
        "train.gradient_checkpointing",
        "Gradient checkpointing",
        "bool",
        help="Saves memory by recomputing activations. Turn it off only on large GPUs.",
    ),
    FieldSpec("train.seed", "Seed", "int", help="Makes the split and the training repeatable."),
    FieldSpec(
        "data.eval_fraction",
        "Eval share",
        "float",
        help="Share of labeled rows held out to evaluate the model.",
    ),
    FieldSpec(
        "eval.eval_every", "Evaluate every", "int", help="Optimizer steps between evaluations."
    ),
    FieldSpec(
        "eval.select_metric",
        "Keep the checkpoint with the best",
        "choice",
        ("mean_macro_f1", "exact_match", "mean_log_loss"),
        "The eval score that picks which checkpoint becomes the adapter.",
    ),
    FieldSpec(
        "eval.early_stopping_patience",
        "Early stopping patience",
        "optional_int",
        help="Stop after this many evaluations without improvement. Leave empty to never stop "
        "early.",
    ),
    FieldSpec(
        "eval.calibrate",
        "Calibrate probabilities",
        "bool",
        help="Fits one temperature per question so probabilities match how often answers are "
        f"right. {CALIBRATION_CAVEAT}",
    ),
    FieldSpec(
        "output.save_every",
        "Checkpoint every",
        "int",
        help="Optimizer steps between checkpoints. A stopped run resumes from the last one.",
    ),
)
# Config fields the Configure step leaves to other steps or to the launcher.
EXCLUDED_FIELDS = frozenset(
    {
        "run_name",  # its own textbox on the Configure step
        "model.id",  # the header's model picker
        "model.revision",  # pinned in code
        "model.local_path",  # set by the launcher on HF Jobs
        "schema",  # the Schema step
        "schema_file",
        "data.train",  # the uploaded file
        "data.eval",
        "data.id_column",
        "data.state_columns",
        "data.label_columns",
        "data.image_columns",
        "output.dir",  # set by the launcher
        "output.push_to_hub",  # the Launch step's adapter repo
        "output.report_to",  # jobs get no tracker credentials
    }
)
_FIELDS = {spec.path: spec for spec in ADVANCED_FIELDS}


@dataclass
class ConfigureView:
    preset: str
    values: dict[str, Any]
    question_weights: list[list[Any]]
    estimates: str
    error: str | None


def _get(config: Config, path: str) -> Any:
    value: Any = config
    for part in path.split("."):
        value = getattr(value, part)
    return value


def _coerce(spec: FieldSpec, value: Any) -> Any:
    if spec.kind in ("int", "optional_int"):
        if value is None or (isinstance(value, str) and not value.strip()):
            if spec.kind == "optional_int":
                return None
            raise ValueError(f"{spec.label} needs a value")
        number = float(value)
        if not number.is_integer():
            raise ValueError(f"{spec.label} must be a whole number")
        return int(number)
    if spec.kind == "float":
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"{spec.label} needs a value")
        return float(value)
    if spec.kind == "bool":
        return bool(value)
    return value


def _try_override(session: Session, path: str, value: Any) -> str | None:
    overrides = session.draft.overrides
    previous = overrides.get(path, _MISSING)
    overrides[path] = value
    try:
        session.config()
    except ValueError as exc:
        if previous is _MISSING:
            overrides.pop(path)
        else:
            overrides[path] = previous
        return error_text(exc)
    return None


def set_field(session: Session, path: str, value: Any) -> str | None:
    spec = _FIELDS.get(path)
    if spec is None:
        return f"{path} cannot be changed here"
    try:
        value = _coerce(spec, value)
    except (TypeError, ValueError) as exc:
        return str(exc)
    return _try_override(session, path, value)


def set_question_weights(session: Session, rows: list[list[Any]]) -> str | None:
    weights: dict[str, float] = {}
    for row in rows:
        if not row or row[0] is None or len(row) < 2 or row[1] in (None, ""):
            continue
        try:
            weights[str(row[0])] = float(row[1])
        except (TypeError, ValueError):
            return f"The weight for {row[0]} must be a number"
    return _try_override(session, "train.question_weights", weights)


def set_preset(session: Session, preset: str) -> None:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}")
    draft = session.draft
    draft.preset = preset
    for path in PRESETS[preset]:
        draft.overrides.pop(path, None)


def configure_view(session: Session) -> ConfigureView:
    draft = session.draft
    if draft is None:
        return ConfigureView("balanced", {}, [], estimate_view(session), None)
    try:
        config = session.config()
    except ValueError as exc:
        return ConfigureView(draft.preset, {}, [], estimate_view(session), error_text(exc))
    values = {spec.path: _get(config, spec.path) for spec in ADVANCED_FIELDS}
    weights = [[q.id, config.train.question_weights.get(q.id, 1.0)] for q in draft.questions]
    return ConfigureView(draft.preset, values, weights, estimate_view(session), None)


def estimate_view(session: Session) -> str:
    recommended = recommend(session.estimates)
    return fragments.estimate_table(
        session.estimates, recommended.flavor if recommended is not None else None
    )


def set_model(session: Session, model_id: str) -> None:
    if model_id not in MODELS:
        raise ValueError(f"unknown model {model_id!r}")
    if session.draft is not None:
        session.draft.model_id = model_id


def set_run_name(session: Session, name: str) -> str | None:
    draft = session.draft
    previous = draft.run_name
    draft.run_name = name.strip()
    try:
        session.config()
    except ValueError:
        draft.run_name = previous
        return (
            "Use up to 64 letters, digits, dots, dashes or underscores, "
            "starting with a letter or digit"
        )
    return None


# ---------------------------------------------------------------------------- sidebar, bundle


def step_states(session: Session) -> list[StepState]:
    """Data, Validate, Configure, Launch: each todo, done, warning or error."""
    states: list[StepState] = ["todo"] * len(STEPS)
    if session.table is not None and session.draft is not None:
        try:
            session.config()
            states[0] = "done"
        except ValueError:
            states[0] = "error"
    report = session.report
    if report is not None:
        states[1] = "error" if report.errors else "warning" if report.warnings else "done"
    if session.reached > 2 and states[0] == "done":
        states[2] = "done"
    if session.job_id or session.local_folder:
        states[3] = "done"
    return states


def sidebar_estimate(session: Session) -> str:
    row = recommend(session.estimates)
    if row is None:
        body = (
            "Load data to see what a run costs."
            if session.table is None
            else "Fix the errors on the Validate step to see what a run costs."
        )
        return fragments.empty_state("No estimate yet", body)
    cost = "—" if row.cost_low is None else f"${row.cost_low:,.2f}–${row.cost_high:,.2f}"
    return (
        '<div class="ct-side-estimate"><div class="ct-side-label">Estimated cost</div>'
        f'<div class="ct-side-cost">{cost}</div>'
        f'<div class="ct-side-time">{row.hours_low:.1f}–{row.hours_high:.1f} h on '
        f"{fragments.esc(row.flavor)}</div></div>"
    )


def bundle(session: Session) -> Path:
    """Write the training bundle and zip it for download."""
    report = session.report
    if report is None or not report.ok:
        raise ValueError("Fix the validation errors before downloading the bundle")
    config = session.config()
    out = session.workdir() / "bundle"
    shutil.rmtree(out, ignore_errors=True)
    write_bundle(report, config, out)
    return zip_bundle(out, session.workdir() / f"{config.run_name}-bundle.zip")
