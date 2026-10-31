"""What each page shows, as plain data for the templates: the step bar, the Data, What to
predict and Settings steps, and setting up a prediction. Nothing here builds HTML."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from clef_tune.card import CALIBRATION_CAVEAT, WEIGHTING_NOTE
from clef_tune.config import Config
from clef_tune.estimate import (
    EstimateRow,
    estimate_hf_flavors,
    model_spec,
    optimizer_steps,
    recommend,
    suggest_timeout,
    workload_from,
)
from clef_tune.predictions import InputCheck, PredictionSummary
from clef_tune.prompt import prompt_pieces
from clef_tune.schema import Schema, SchemaError
from clef_tune.ui import handlers
from clef_tune.ui import wording as w
from clef_tune.ui.runs import cap_cost, hardware
from clef_tune.ui.services import Services
from clef_tune.ui.session import SAMPLES, Session
from clef_tune.ui.settings import Settings

PREVIEW_ROWS = 4
ROWS_PAGE = 50
StepState = Literal["done", "now", "todo", "locked"]
CheckLevel = Literal["ok", "warn", "err", "info"]

STEPS = (
    ("data", "Your data", "/data"),
    ("questions", "What to predict", "/questions"),
    ("settings", "Settings", "/settings"),
    ("train", "Train", "/train"),
)
PRESETS = {
    "quick": ("Quick test", "A fast, light run to check everything works."),
    "balanced": ("Balanced", "A good first real model. Three passes over your examples."),
    "thorough": ("Thorough", "Trains longer and more deeply. Most useful with larger datasets."),
}
MODELS = {
    "Cloudflare/clef-flash": ("Clef-Flash", "Fast and affordable · 9B"),
    "Cloudflare/clef": ("Clef", "Most capable · 27B · about 3× the cost"),
}
KINDS = (
    ("choice", "Pick one", "of named answers"),
    ("score", "Scale", "answers in order"),
    ("noul", "Yes or no", "true or false"),
)
HOLDOUT_PATH = "data.eval_fraction"  # set in the Settings step's "We'll set aside" sentence


@dataclass(frozen=True)
class Guide:
    """How Advanced settings explain one field: its name, a line under it, the hover tip, and
    the usual range shown beside the box (empty: show the allowed values instead)."""

    label: str
    help: str
    tip: str
    usual: str = ""


# In the order the Advanced settings show them: the ones people are most likely to touch first.
FIELD_GUIDE = {
    "train.epochs": Guide(
        "Passes over your data",
        "How many times training reads every example.",
        "Each pass reads all of your training examples once. More passes can learn more from a "
        "small dataset, but too many and the model starts memorising examples instead of the "
        "pattern; the test results show if that happens.",
        "Usually 1–5",
    ),
    "train.mode": Guide(
        "What gets trained",
        "Adapters change how the model reads; the answer layer turns that into predictions.",
        "lora trains small adapters inside the language model plus the answer layer (the "
        "decision head). head_only trains just the answer layer: faster and cheaper, but it "
        "learns less.",
    ),
    "train.lora.r": Guide(
        "Adapter size",
        "Bigger adapters can learn more, and need more data.",
        "The LoRA rank: how much room each adapter has to learn. Larger adapters can pick up "
        "finer detail but train more slowly and can memorise a small dataset. Change LoRA alpha "
        "with it.",
        "Usually 8–64",
    ),
    "train.max_length": Guide(
        "Longest example",
        "Longer examples are cut to fit.",
        "The most tokens (word pieces) the model reads for one example, questions included. "
        "Longer examples have their text cut to fit. Raising it uses more memory and time.",
        "Usually 1,024–8,192",
    ),
    "train.quantization": Guide(
        "Memory saving",
        "Stores the model in 4 bits to fit smaller GPUs, a little slower.",
        "4bit loads the model in 4-bit precision (QLoRA) so it fits on a smaller, cheaper GPU. "
        "Training is somewhat slower, and results are usually close to full precision.",
    ),
    "eval.calibrate": Guide(
        "Calibrate confidence",
        "Adjusts confidence so 80% sure means right about 80% of the time.",
        "After training, fits one temperature per question so the model's confidence matches "
        f"how often it is right. {CALIBRATION_CAVEAT}",
    ),
    "eval.early_stopping_patience": Guide(
        "Stop early",
        "Stops after this many checks without improvement. Empty never stops early.",
        "The model is checked on the test examples every few steps (Evaluate every). If its "
        "score doesn't improve for this many checks in a row, training stops and keeps the "
        "best checkpoint, which saves time once more training stops helping.",
        "Usually 2–5, or empty",
    ),
    "train.lora.alpha": Guide(
        "LoRA alpha",
        "How strongly the adapters' changes count.",
        "Alpha divided by the adapter size scales every adapter update. The usual rule is "
        "alpha = 2 × adapter size; change the two together.",
        "Usually 2 × adapter size",
    ),
    "train.lora.dropout": Guide(
        "LoRA dropout",
        "Randomly drops some adapter inputs while training, against memorising.",
        "The share of adapter inputs ignored at random on each step. A little dropout helps a "
        "small dataset generalise; 0 turns it off.",
        "Usually 0–0.1",
    ),
    "train.lora_lr": Guide(
        "LoRA learning rate",
        "How big a step the adapters take on each update.",
        "Too high and training becomes unstable (the loss jumps around or rises); too low and "
        "it learns slowly. Change it by factors of 2 or 3, not small amounts.",
        "Usually 5e-5 to 3e-4",
    ),
    "train.head_lr": Guide(
        "Head learning rate",
        "How big a step the answer layer takes on each update.",
        "The step size for the answer layer (the decision head), which turns what the model "
        "reads into answer probabilities. Too high makes training unstable; too low learns "
        "slowly.",
        "Usually 1e-5 to 1e-4",
    ),
    "train.max_tokens_per_batch": Guide(
        "Tokens per batch",
        "Examples are packed into batches up to this size; lower it if a run runs out of memory.",
        "How many tokens go through the GPU at once. Bigger batches train faster but use more "
        "memory. If a run stops with an out-of-memory error, halve it.",
        "Usually 4,096–32,768",
    ),
    "train.grad_accum_steps": Guide(
        "Gradient accumulation",
        "Batches added up before each update.",
        "Adds up this many batches before the model takes a step, which acts like a bigger "
        "batch without using more memory. Higher values mean fewer, smoother updates.",
        "Usually 1–8",
    ),
    "train.warmup_ratio": Guide(
        "Warmup share",
        "Share of training spent slowly raising the learning rate.",
        "Starts with a small learning rate and raises it over this share of the steps, so the "
        "first updates are gentle. 0.03 means the first 3% of steps.",
        "Usually 0–0.1",
    ),
    "train.weight_decay": Guide(
        "Weight decay",
        "Gently pulls the adapters' weights toward zero.",
        "AdamW weight decay: a mild brake against memorising. 0 turns it off.",
        "Usually 0–0.1",
    ),
    "train.label_smoothing": Guide(
        "Label smoothing",
        "Softens the expected outputs so the model is less overconfident.",
        "Trains toward “mostly this answer” rather than “certainly this answer”. Can help when "
        "some expected outputs are wrong or borderline. 0 turns it off.",
        "Usually 0–0.1",
    ),
    "train.class_weighting": Guide(
        "Class weighting",
        "Gives rare answers more weight while training.",
        "balanced makes every answer count equally, so rare answers aren't drowned out by "
        f"common ones; none counts every example the same. {WEIGHTING_NOTE}",
    ),
    "train.gradient_checkpointing": Guide(
        "Gradient checkpointing",
        "Saves memory by redoing some work. Turn it off only on large GPUs.",
        "Recomputes parts of the model during training instead of storing them, which uses "
        "much less memory and makes each step a little slower.",
    ),
    "train.seed": Guide(
        "Seed",
        "Makes the test split and the training repeatable.",
        "The same seed with the same data and settings picks the same test examples and gives "
        "very similar results. Change it to see how much results vary from run to run.",
        "Any whole number",
    ),
    "eval.eval_every": Guide(
        "Evaluate every",
        "Training steps between checks on the test examples.",
        "Each check scores the model on the test examples, which takes a little time. The "
        "checks drive Stop early and pick the best checkpoint; there is always one at the end.",
    ),
    "eval.select_metric": Guide(
        "Keep the checkpoint with the best",
        "Which test score picks the checkpoint that becomes your model.",
        "mean_macro_f1 balances precision and recall, with rare answers counting as much as "
        "common ones. exact_match is the share of examples with every question right. "
        "mean_log_loss rewards confident right answers and punishes confident wrong ones.",
    ),
    "output.save_every": Guide(
        "Checkpoint every",
        "Training steps between saved checkpoints. A stopped run resumes from the last one.",
        "Saving more often means a stopped run loses less work when resumed, at the cost of a "
        "little time and storage.",
    ),
}
STEP_FIELDS = ("eval.eval_every", "output.save_every")  # ranged against this run's step count


def _number(value: float) -> str:
    return f"{int(value):,}" if float(value).is_integer() else f"{value:g}"


def _field_info(path: str) -> Any:
    model: Any = Config
    parts = path.split(".")
    for part in parts[:-1]:
        model = model.model_fields[part].annotation
    return model.model_fields[parts[-1]]


def allowed(path: str) -> str:
    """The values the config accepts for a field, in words ("256 to 16,384")."""
    info = _field_info(path)
    low = high = None
    low_open = high_open = False
    for rule in info.metadata:
        if getattr(rule, "ge", None) is not None:
            low = rule.ge
        if getattr(rule, "gt", None) is not None:
            low, low_open = rule.gt, True
        if getattr(rule, "le", None) is not None:
            high = rule.le
        if getattr(rule, "lt", None) is not None:
            high, high_open = rule.lt, True
    if low is not None and high is not None:
        if low_open and high_open:
            text = f"between {_number(low)} and {_number(high)}"
        elif high_open:
            text = f"{_number(low)} to just under {_number(high)}"
        else:
            text = f"{_number(low)} to {_number(high)}"
    elif low is not None:
        text = f"more than {_number(low)}" if low_open else f"{_number(low)} or more"
    elif high is not None:
        text = f"up to {_number(high)}"
    else:
        text = "any number"
    if info.default is None and not info.is_required():
        text += ", or empty"
    return text


_RANGE_ERRORS = ("greater than", "less than", "convert", "whole number", "needs a value")


def field_error_text(path: str, message: str) -> str:
    """A refused Advanced setting in plain words: for a number out of range or mistyped, the
    values it accepts; any other reason as it is."""
    spec = next((f for f in handlers.ADVANCED_FIELDS if f.path == path), None)
    if spec is None or spec.kind not in ("int", "float", "optional_int"):
        return message
    if not any(phrase in message for phrase in _RANGE_ERRORS):
        return message
    kind = "a number" if spec.kind == "float" else "a whole number"
    return f"Use {kind}: {allowed(path)}."


def _shown(value: Any) -> str:
    if value is None:
        return "empty"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, (int, float)):
        return _number(value)
    return str(value)


def _defaults() -> Config:
    return Config.model_validate({"run_name": "x", "data": {"train": "x"}})


# ---------------------------------------------------------------------------- the step bar


@dataclass
class StepItem:
    number: str
    label: str
    url: str
    state: StepState


def furthest_step(session: Session) -> int:
    """The last step a person may open: data first, then questions, settings and training."""
    if session.table is None or session.draft is None:
        return 0
    if not session.draft.questions:
        return 1
    try:
        session.config()
    except ValueError:
        return 1
    report = session.report
    return 3 if report is not None and report.ok else 2


def step_bar(session: Session, current: str) -> list[StepItem]:
    furthest = furthest_step(session)
    index = [key for key, _, _ in STEPS].index(current)
    items = []
    for k, (_, label, url) in enumerate(STEPS):
        if k == index:
            state: StepState = "now"
        elif k > furthest:
            state = "locked"
        else:
            state = "done" if k < index else "todo"
        items.append(StepItem(f"{k + 1:02d}", label, url, state))
    return items


# ---------------------------------------------------------------------------- data


@dataclass
class ColumnRow:
    name: str
    samples: str
    role: str
    role_name: str
    flags: list[str] = field(default_factory=list)


@dataclass
class CheckLine:
    level: CheckLevel
    text: str
    detail: str = ""


@dataclass
class DataPage:
    has_data: bool
    samples: list[tuple[str, str]]
    max_mb: int
    error: str | None
    file_name: str = ""
    rows: int = 0
    columns: int = 0
    noun: str = ""
    noun_guessed: bool = False
    headers: list[tuple[str, str]] = field(default_factory=list)  # (column, role)
    preview: list[list[str]] = field(default_factory=list)
    column_rows: list[ColumnRow] = field(default_factory=list)
    checks: list[CheckLine] = field(default_factory=list)
    to_look_at: int = 0
    blocking: bool = False
    summary: str = ""
    native: bool = False


ROLE_NAMES = {role: name for role, name, _ in handlers.ROLE_CHOICES}


def _question_flags(session: Session) -> dict[str, list[str]]:
    """Notes to show under a question's column: empty rows and its validation warnings."""
    report, draft = session.report, session.draft
    flags: dict[str, list[str]] = {}
    if report is None or draft is None:
        return flags
    columns = {q.id: q.column for q in draft.questions}
    for qid, stats in report.questions.items():
        empty = stats.train_missing
        if empty:
            flags.setdefault(columns.get(qid, qid), []).append(
                f"Empty in {empty:,} row{'s' if empty != 1 else ''}. "
                "Those rows still teach the other questions."
            )
    for issue in report.issues:
        if issue.question and issue.level == "warning":
            flags.setdefault(columns.get(issue.question, issue.question), []).append(
                _plain_issue(issue.message)
            )
    return flags


def _plain_issue(message: str) -> str:
    """Some validation messages in plain words; the rest as they are."""
    rare = re.match(r"rare classes \(<5% of labels\): (.+?);", message)
    if rare:
        return f"Few examples of {rare.group(1)}. The model may not learn those well."
    missing = re.match(r"eval set has no examples of: (.+)", message)
    if missing:
        return f"None of the test rows has {missing.group(1)}, so it can't be measured."
    return message[:1].upper() + message[1:]


def data_checks(session: Session) -> tuple[list[CheckLine], bool]:
    """The summary under the columns, and whether anything blocks Continue."""
    report, table, draft = session.report, session.table, session.draft
    lines: list[CheckLine] = []
    if table is None or draft is None:
        return lines, True
    read_errors = [issue for issue in table.issues if issue.level == "error"]
    if not read_errors:
        lines.append(
            CheckLine("ok", f"All {len(table.rows):,} rows read, with {len(table.columns)} columns")
        )
    if not draft.questions:
        lines.append(
            CheckLine(
                "err",
                "Choose at least one expected output",
                "Mark the column that holds what the model should predict.",
            )
        )
    if report is None:
        return lines, not draft.questions or bool(read_errors)
    messages = [issue.message for issue in report.issues]
    if draft.id_column and not any("duplicate ID" in m for m in messages):
        lines.append(CheckLine("ok", "Every row has its own ID"))
    unavailable = next((m for m in messages if m.startswith("tokenizer unavailable")), None)
    if unavailable:
        lines.append(
            CheckLine("info", "Couldn't check how long the rows are", _plain_issue(unavailable))
        )
    elif report.tokens is not None:
        if report.tokens.truncated_rows:
            n = report.tokens.truncated_rows
            lines.append(
                CheckLine(
                    "warn",
                    f"{n:,} row{'s are' if n != 1 else ' is'} too long to read in full",
                    "The end of each is cut off. Advanced settings can raise the limit.",
                )
            )
        else:
            lines.append(CheckLine("ok", "Every row is short enough for the model to read in full"))
    for issue in report.issues:
        if issue.question and issue.level == "warning":
            continue  # shown under its column
        if (
            issue.message.startswith("tokenizer unavailable")
            or "exceed max_length" in issue.message
        ):
            continue
        level: CheckLevel = "err" if issue.level == "error" else "warn"
        rows = f" (rows {', '.join(str(r) for r in issue.rows[:5])})" if issue.rows else ""
        lines.append(CheckLine(level, _plain_issue(issue.message) + rows))
    return lines, bool(report.errors) or not draft.questions


def data_page(session: Session, settings: Settings, error: str | None = None) -> DataPage:
    samples = list(SAMPLES.items())
    page = DataPage(session.table is not None, samples, settings.max_upload_mb, error)
    table, draft = session.table, session.draft
    if table is None or draft is None:
        return page
    roles = handlers.column_roles(session)
    flags = _question_flags(session)
    guess = w.guess_noun(table.path.name)
    page.file_name = table.path.name
    page.rows, page.columns = len(table.rows), len(table.columns)
    page.noun = draft.row_noun or ""
    page.noun_guessed = bool(draft.row_noun) and draft.row_noun == guess
    page.headers = [(column, roles.get(column, "skip")) for column in table.columns]
    page.preview = [
        [handlers._cell(row.get(column)) for column in table.columns]
        for row in table.rows[:PREVIEW_ROWS]
    ]
    page.column_rows = [
        ColumnRow(row.name, row.samples, row.role, ROLE_NAMES[row.role], flags.get(row.name, []))
        for row in handlers.column_rows(session)
    ]
    page.checks, page.blocking = data_checks(session)
    page.to_look_at = sum(1 for line in page.checks if line.level == "warn") + sum(
        len(row.flags) for row in page.column_rows
    )
    outputs = sum(1 for role in roles.values() if role == "question")
    reads = sum(1 for role in roles.values() if role in ("state", "image"))
    page.summary = (
        f"{outputs} expected output{'s' if outputs != 1 else ''} · "
        f"{reads} column{'s' if reads != 1 else ''} read"
    )
    page.native = draft.native
    return page


@dataclass
class RowsPage:
    file_name: str
    headers: list[str]
    rows: list[list[str]]
    page: int
    pages: int
    label: str


def rows_page(session: Session, page: int) -> RowsPage | None:
    table = session.table
    if table is None:
        return None
    total = len(table.rows)
    pages = max(1, -(-total // ROWS_PAGE))
    page = min(max(page, 0), pages - 1)
    start = page * ROWS_PAGE
    chunk = table.rows[start : start + ROWS_PAGE]
    rows = [[handlers._cell(row.get(column)) for column in table.columns] for row in chunk]
    label = f"Rows {start + 1:,}–{start + len(chunk):,} of {total:,}" if total else "No rows"
    return RowsPage(table.path.name, list(table.columns), rows, page, pages, label)


# ---------------------------------------------------------------------------- what to predict


@dataclass
class AnswerRow:
    id: str
    description: str
    count: int
    bar: float  # share of the most common answer, 0-1


@dataclass
class Editor:
    id: str
    column: str
    kind: str
    ask: str
    answers: list[AnswerRow]
    hint: str
    flags: list[str]
    counted: str  # "3 answers in your data"


@dataclass
class QuestionItem:
    number: str
    id: str
    ask: str


@dataclass
class QuestionsPage:
    items: list[QuestionItem]
    selected: Editor | None
    reads: list[str]
    ignores: list[str]
    noun: str
    rows: int
    error: str | None
    blocking: bool


def _editor(session: Session, qid: str) -> Editor | None:
    question = session.draft.question(qid)
    if question is None:
        return None
    counts, unknown = handlers.option_counts(session, question)
    top = max(counts.values(), default=0) or 1
    answers = [
        AnswerRow(o.id, o.description, counts.get(o.id, 0), counts.get(o.id, 0) / top)
        for o in question.options
    ]
    flags = list(_question_flags(session).get(question.column, []))
    if unknown:
        shown = ", ".join(
            f"{value} ({count})"
            for value, count in sorted(unknown.items(), key=lambda item: (-item[1], item[0]))[:6]
        )
        rows = sum(unknown.values())
        flags.append(
            f"{rows:,} row{'s have' if rows != 1 else ' has'} an answer that isn't in this list: "
            f"{shown}. Fix them in your file, or they'll be left out."
        )
    hint = {
        "score": "Lowest first. Use the arrows to reorder the scale.",
        "noul": "Descriptions are optional. The question alone is often enough for yes or no.",
    }.get(question.type, "")
    n = len(question.options)
    return Editor(
        question.id,
        question.column,
        question.type,
        question.instructions,
        answers,
        hint,
        flags,
        f"{n} answer{'s' if n != 1 else ''} in your data",
    )


def questions_page(session: Session, selected: str | None = None) -> QuestionsPage:
    draft, table = session.draft, session.table
    if draft is None or table is None:
        return QuestionsPage([], None, [], [], w.DEFAULT_NOUN, 0, None, True)
    items = [
        QuestionItem(f"{k:02d}", q.id, q.instructions.strip() or q.id)
        for k, q in enumerate(draft.questions, start=1)
    ]
    qid = selected if selected and draft.question(selected) else (items[0].id if items else None)
    roles = handlers.column_roles(session)
    error = None
    try:
        session.config()
    except ValueError as exc:
        error = handlers.error_text(exc)
    report = session.report
    return QuestionsPage(
        items,
        _editor(session, qid) if qid else None,
        [c for c, r in roles.items() if r in ("state", "image")],
        [c for c, r in roles.items() if r == "skip"],
        w.noun_or_default(draft.row_noun),
        len(table.rows),
        error,
        error is not None or not items or (report is not None and not report.ok),
    )


def prompt_preview(session: Session, row: int = 0) -> list[tuple[str, str]]:
    """The exact prompt the model reads for one row, as (kind, text) pieces."""
    table, draft = session.table, session.draft
    if table is None or draft is None or not table.rows:
        return []
    try:
        schema = Schema.from_dict({"questions": {q.id: q.to_clef() for q in draft.questions}})
    except SchemaError as exc:
        return [("error", f"Fix the questions to see the prompt: {exc}")]
    record = table.rows[min(max(row, 0), len(table.rows) - 1)]
    if draft.native:
        state, images = record.get("state"), len(record.get("images") or [])
    else:
        values = {column: record.get(column) for column in draft.state_columns}
        state = values[draft.state_columns[0]] if len(draft.state_columns) == 1 else values
        images = sum(1 for c in draft.image_columns if str(record.get(c) or "").strip())
    return [(piece.kind, piece.text) for piece in prompt_pieces(state, schema.questions, images)]


# ---------------------------------------------------------------------------- settings


@dataclass
class Choice:
    key: str
    name: str
    description: str
    extra: str = ""
    recommended: bool = False


@dataclass
class EstimateView:
    time: str
    time_short: str
    cost: str
    hardware: str
    cap_time: str
    cap_cost: str | None
    test_rows: int
    total_rows: int


@dataclass
class AdvancedRow:
    path: str
    label: str
    code: str
    help: str
    kind: str
    value: Any
    choices: tuple[str, ...]
    tip: str = ""
    range: str = ""  # shown beside the box: the usual values, or the allowed ones


@dataclass
class HoldoutView:
    percent: str  # the share set aside to test, as a percent
    test_rows: int
    total_rows: int


@dataclass
class FlavorRow:
    flavor: str
    hardware: str
    time: str
    cost: str
    fits: bool
    note: str
    recommended: bool


@dataclass
class SettingsPage:
    name: str
    repo: str
    model: Choice
    models: list[Choice]
    preset: Choice
    presets: list[Choice]
    estimate: EstimateView | None
    estimate_note: str
    advanced: list[AdvancedRow]
    flavors: list[FlavorRow]
    flavor: str | None
    noun: str
    error: str | None
    holdout: HoldoutView | None = None


def _row_for(session: Session, rows: list[EstimateRow]) -> EstimateRow | None:
    if session.flavor:
        return next((r for r in rows if r.flavor == session.flavor and r.fits), None)
    return recommend(rows)


def _estimate(row: EstimateRow, test_rows: int, total: int) -> EstimateView:
    cap = cap_cost(row.price_per_hour, suggest_timeout(row))
    return EstimateView(
        time=w.minutes_range(row.hours_low, row.hours_high),
        time_short=w.minutes_range(row.hours_low, row.hours_high, short=True),
        cost=w.money_range(row.cost_low, row.cost_high),
        hardware=hardware(row.flavor),
        cap_time=w.timeout_text(suggest_timeout(row)),
        cap_cost=w.money(cap) if cap is not None else None,
        test_rows=test_rows,
        total_rows=total,
    )


def settings_page(session: Session, services: Services, namespace: str | None) -> SettingsPage:
    draft = session.draft
    noun = w.noun_or_default(draft.row_noun if draft else None)
    model_key = draft.model_id if draft else "Cloudflare/clef-flash"
    models = [Choice(key, name, text) for key, (name, text) in MODELS.items()]
    preset_key = draft.preset if draft else "balanced"
    error = None
    config = None
    try:
        config = session.config()
    except ValueError as exc:
        error = handlers.error_text(exc)
    report = session.report
    spec = model_spec(model_key)
    tokens = report.tokens if report is not None else None
    prices = services.prices() if tokens is not None and spec is not None else {}
    presets = []
    for key, (name, text) in PRESETS.items():
        extra = ""
        if config is not None and tokens is not None and spec is not None:
            rows = estimate_hf_flavors(spec, workload_from(config.with_preset(key), tokens), prices)
            row = _row_for(session, rows)
            if row is not None:
                extra = (
                    f"About {w.minutes_range(row.hours_low, row.hours_high, short=True)} · "
                    f"{w.money_range(row.cost_low, row.cost_high)}"
                )
        presets.append(Choice(key, name, text, extra, key == "balanced"))
    estimate, note, flavors = None, "", []
    rows = session.estimates
    if rows:
        chosen = _row_for(session, rows)
        best = recommend(rows)
        test = len(report.eval) if report else 0
        total = test + (len(report.train) if report else 0)
        if chosen is not None:
            estimate = _estimate(chosen, test, total)
        flavors = [
            FlavorRow(
                r.flavor,
                hardware(r.flavor),
                w.minutes_range(r.hours_low, r.hours_high, short=True),
                w.money_range(r.cost_low, r.cost_high),
                r.fits,
                r.note or "",
                best is not None and r.flavor == best.flavor,
            )
            for r in rows
        ]
    elif report is not None and not report.ok:
        note = "Fix the problems on the first two steps to see the time and cost."
    else:
        note = "The time and cost appear once the rows have been counted."
    steps = optimizer_steps(config, tokens) if config is not None and tokens else None
    defaults = _defaults()
    advanced = []
    for spec_field in handlers.ADVANCED_FIELDS:
        if spec_field.path == HOLDOUT_PATH:
            continue
        guide = FIELD_GUIDE.get(spec_field.path, Guide(spec_field.label, spec_field.help, ""))
        value = handlers._get(config, spec_field.path) if config is not None else None
        tip, shown_range = guide.tip, ""
        if spec_field.kind in ("int", "float", "optional_int"):
            limits = allowed(spec_field.path)
            tip = f"{tip} Allowed: {limits}.".strip()
            if spec_field.path in STEP_FIELDS and steps:
                shown_range = f"This run has about {steps:,} steps"
            else:
                shown_range = guide.usual or limits[:1].upper() + limits[1:]
        default = _shown(handlers._get(defaults, spec_field.path))
        tip = f"{tip} Default: {default}.".strip()
        advanced.append(
            AdvancedRow(
                spec_field.path,
                guide.label,
                spec_field.path,
                guide.help,
                spec_field.kind,
                value,
                spec_field.choices,
                tip,
                shown_range,
            )
        )
    order = list(FIELD_GUIDE)
    advanced.sort(key=lambda row: order.index(row.path) if row.path in order else len(order))
    holdout = None
    if config is not None and report is not None and (report.eval or report.train):
        holdout = HoldoutView(
            _number(round(config.data.eval_fraction * 100, 1)),
            len(report.eval),
            len(report.eval) + len(report.train),
        )
    name = draft.run_name if draft else ""
    return SettingsPage(
        name=name,
        repo=f"{namespace}/{name}" if namespace else name,
        model=next(c for c in models if c.key == model_key),
        models=models,
        preset=next(c for c in presets if c.key == preset_key),
        presets=presets,
        estimate=estimate,
        estimate_note=note,
        advanced=advanced,
        flavors=flavors,
        flavor=session.flavor,
        noun=noun,
        error=error,
        holdout=holdout,
    )


# ---------------------------------------------------------------------------- predicting


@dataclass(frozen=True)
class Download:
    key: str  # the file suffix, without its dot
    name: str
    description: str
    media_type: str


# The formats finished predictions download as, in the order the Download menu lists them.
DOWNLOADS = (
    Download(
        "xlsx",
        "Excel",
        "A workbook, with confidence as numbers",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ),
    Download("csv", "CSV", "Opens in Excel, Google Sheets or Numbers", "text/csv"),
    Download(
        "tsv", "TSV", "Tab-separated text, for tools that prefer tabs", "text/tab-separated-values"
    ),
    Download(
        "jsonl",
        "JSON Lines",
        "One record per line, for code and data tools",
        "application/x-ndjson",
    ),
    Download("json", "JSON", "All the records in one list", "application/json"),
)


@dataclass
class PredictionDraft:
    """A prediction being set up in this session, from one trained model."""

    job_id: str  # the training run whose model predicts
    adapter_repo: str
    name: str
    config: Config  # the model's training config
    file_name: str = ""
    path: Path | None = None
    check: InputCheck | None = None
    rename: dict[str, str] = field(default_factory=dict)
    threshold: float = 0.7
    plan: Any = None
    error: str | None = None
    calibrated: bool = False  # training fitted its confidence, so "70% sure" means what it says


@dataclass
class PredictPage:
    job_id: str
    name: str
    noun: str
    has_file: bool
    file_name: str
    rows: int
    checks: list[CheckLine]
    missing: list[str]
    columns: list[str]
    threshold: int
    can_check_cost: bool
    error: str | None
    calibrated: bool = False


def predict_page(draft: PredictionDraft) -> PredictPage:
    noun = w.noun_or_default(draft.config.data.row_noun)
    check = draft.check
    checks: list[CheckLine] = []
    missing: list[str] = []
    if check is not None:
        missing = check.missing
        reads = list(draft.config.data.state_columns or []) + list(draft.config.data.image_columns)
        if missing:
            checks.append(
                CheckLine(
                    "err",
                    f"Couldn't find {', '.join(missing)}",
                    "Pick the column in your file that holds each one.",
                )
            )
        elif reads:
            checks.append(
                CheckLine("ok", f"Found {_join(reads)}", "The columns the model learned to read.")
            )
        if check.numbered_ids:
            checks.append(CheckLine("info", "No ID column, so rows are numbered from 1"))
        if not missing:
            if check.truncated:
                checks.append(
                    CheckLine("warn", f"{check.truncated:,} rows are too long to read in full")
                )
            else:
                checks.append(CheckLine("ok", "Every row is short enough to read in full"))
            if check.expected:
                checks.append(
                    CheckLine(
                        "info",
                        f"It has expected outputs for {_join(check.expected)}",
                        "The results will say how often the model matched them.",
                    )
                )
            else:
                checks.append(
                    CheckLine(
                        "info",
                        "No expected outputs in this file",
                        "That's fine. If a file has them, the results also show how often the "
                        "model matches them.",
                    )
                )
        for issue in check.issues:
            if issue.level == "error":
                checks.append(CheckLine("err", _plain_issue(issue.message)))
    return PredictPage(
        job_id=draft.job_id,
        name=draft.name,
        noun=noun,
        has_file=check is not None,
        file_name=draft.file_name,
        rows=check.rows if check else 0,
        checks=checks,
        missing=missing,
        columns=check.columns if check else [],
        threshold=round(draft.threshold * 100),
        can_check_cost=check is not None and check.ok,
        error=draft.error,
        calibrated=draft.calibrated,
    )


def _join(names: list[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


@dataclass
class PredictionEstimate:
    time: str
    cost: str
    cap_cost: str | None
    hardware: str


def prediction_estimate(plan: Any) -> PredictionEstimate:
    estimate = plan.estimate
    return PredictionEstimate(
        time=w.minutes_range(estimate.hours_low, estimate.hours_high) if estimate else "",
        cost=w.money_range(estimate.cost_low, estimate.cost_high) if estimate else "",
        cap_cost=w.money(plan.max_cost) if plan.max_cost is not None else None,
        hardware=hardware(plan.flavor),
    )


@dataclass
class ShareBar:
    ask: str
    parts: list[tuple[str, int]]  # (answer, percent)


@dataclass
class PredictionResult:
    rows: int
    noun: str
    review: int
    threshold: int
    header: list[str]
    preview: list[list[str]]
    added: set[str]  # the columns the prediction added
    low: set[tuple[int, str]]  # (row, column) of confidences under the threshold
    shares: list[ShareBar]
    matches: list[str]
    title_columns: int = 2  # how many of the person's own columns name a row on a phone


def prediction_result(
    summary: PredictionSummary, config: Config, *, threshold: float
) -> PredictionResult:
    noun = w.noun_or_default(config.data.row_noun)
    schema = config.get_schema()
    asks = {qid: (q.instructions or qid) for qid, q in (schema.questions.items() if schema else [])}
    header = list(summary.preview[0]) if summary.preview else []
    added = {
        name
        for name in header
        if name == "review"
        or name.endswith(("_confidence", "_predicted"))
        or (name in asks and f"{name}_confidence" in header)
    }
    preview, low = [], set()
    for k, row in enumerate(summary.preview):
        preview.append([row.get(name, "") for name in header])
        for name in header:
            if name.endswith("_confidence") and row.get(name):
                try:
                    if float(row[name]) < threshold:
                        low.add((k, name))
                except ValueError:
                    pass
    names: dict[str, dict[str, str]] = {}
    for qid, question in schema.questions.items() if schema else []:
        ids = question.option_ids()
        if question.type == "score":
            descriptions = [str(text) for _, text in question.options()]
            names[qid] = dict(zip(ids, w.answer_names("score", ids, descriptions), strict=True))
    shares = [
        ShareBar(
            asks.get(qid, qid),
            [
                (names.get(qid, {}).get(answer, answer), round(share * 100))
                for answer, share in parts
            ],
        )
        for qid, parts in summary.shares.items()
    ]
    matches = [
        f"{asks.get(qid, qid)}: matched {matched:,} of {compared:,}"
        for qid, (matched, compared) in summary.matches.items()
    ]
    return PredictionResult(
        rows=summary.rows,
        noun=noun,
        review=summary.review,
        threshold=round(threshold * 100),
        header=header,
        preview=preview,
        added=added,
        low=low,
        shares=shares,
        matches=matches,
    )
