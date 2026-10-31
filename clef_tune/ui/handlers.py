"""What the Data, What to predict and Settings steps change: plain functions over a Session.
They return an error message (or None) and never build HTML; ui/views.py turns a Session into
what the pages show."""

from __future__ import annotations

import json
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
from clef_tune.estimate import estimate_hf_flavors, model_spec, workload_from
from clef_tune.labels import LabelError, parse_label
from clef_tune.schema import Question, SchemaError
from clef_tune.ui.services import Services
from clef_tune.ui.session import (
    OptionDraft,
    QuestionDraft,
    Session,
    UploadError,
    clear_data,
    load_sample,
    load_upload,
)
from clef_tune.ui.settings import Settings
from clef_tune.validation import ValidationReport, validate

CELL_CHARS = 200
SAMPLE_VALUES = 3
SAMPLE_CHARS = 80
# What a column can be for (flat files): the role it sets, its name, and what it means.
ROLE_CHOICES = (
    ("state", "Model reads it", "Text or numbers the model reads for each row."),
    (
        "question",
        "Expected output",
        "What the model learns to predict. Each one becomes a question.",
    ),
    ("image", "Image", "A picture the model looks at, given as a file name or link."),
    ("id", "Row ID", "Names each row, so predictions can be matched back to it."),
    ("skip", "Ignore", "Left out completely."),
)
ROLES = tuple(role for role, _, _ in ROLE_CHOICES)
MODELS = {"Cloudflare/clef-flash": "Clef-Flash", "Cloudflare/clef": "Clef"}
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


def upload(session: Session, path: str | Path, settings: Settings) -> str | None:
    """Use an uploaded file; return why it cannot be used, or None."""
    try:
        load_upload(session, Path(path), settings)
    except (UploadError, OSError) as exc:
        return f"Could not use {Path(path).name}: {exc}"
    return None


def sample(session: Session, name: str = "support_tickets") -> None:
    load_sample(session, name)


def remove_data(session: Session) -> None:
    clear_data(session)


def set_noun(session: Session, noun: str) -> None:
    """What one row is; an empty word means "example"."""
    if session.draft is not None:
        session.draft.row_noun = noun.strip()[:40] or None


# ---------------------------------------------------------------------------- schema step


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


def describe_option(session: Session, qid: str, option_id: str, description: str) -> str | None:
    question = session.draft.question(qid) if session.draft else None
    if question is None:
        return f"There is no question {qid}"
    option = next((o for o in question.options if o.id == option_id), None)
    if option is None:
        return f"{qid} has no answer {option_id}"
    option.description = description.strip()
    return None


def move_option(session: Session, qid: str, option_id: str, offset: int) -> None:
    """Move a scale's answer up or down (its order is the scale's order)."""
    question = session.draft.question(qid) if session.draft else None
    if question is None:
        return
    ids = [o.id for o in question.options]
    if option_id not in ids:
        return
    k = ids.index(option_id)
    j = min(max(k + offset, 0), len(ids) - 1)
    question.options.insert(j, question.options.pop(k))


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
        "run_name",  # the Settings step's name row
        "model.id",  # the Settings step's model row
        "model.revision",  # pinned in code
        "model.local_path",  # set by the launcher on HF Jobs
        "schema",  # the What to predict step
        "schema_file",
        "data.train",  # the uploaded file
        "data.eval",
        "data.id_column",
        "data.state_columns",
        "data.label_columns",
        "data.image_columns",
        "data.row_noun",  # the Data step's "Each row is one ..."
        "output.dir",  # set by the launcher
        "output.push_to_hub",  # the Train step's adapter repo
        "output.report_to",  # jobs get no tracker credentials
    }
)
_FIELDS = {spec.path: spec for spec in ADVANCED_FIELDS}


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


HOLDOUT_PERCENT = (5, 50)  # the share of rows the Settings step lets people set aside to test


def set_holdout(session: Session, percent: str) -> str | None:
    """Set aside this percent of the rows to test the model; return why not, or None."""
    low, high = HOLDOUT_PERCENT
    refusal = f"Choose a share from {low}% to {high}%"
    try:
        value = float(percent.strip().rstrip("%"))
    except ValueError:
        return refusal
    if not low <= value <= high:
        return refusal
    return _try_override(session, "data.eval_fraction", round(value / 100, 4))


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
