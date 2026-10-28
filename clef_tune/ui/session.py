"""One browser session's state: its private folder, the uploaded table, and the draft config the
person edits step by step. Holds no tokens, locks or processes: Gradio deep-copies a fresh
Session for every browser session."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import time
import uuid
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import clef_tune
from clef_tune.automap import Proposal, propose
from clef_tune.config import Config, load_config
from clef_tune.data import DataError, Issue, read_rows
from clef_tune.estimate import EstimateRow
from clef_tune.labels import is_missing
from clef_tune.ui.settings import Settings
from clef_tune.validation import ValidationReport

SESSIONS_ROOT = Path(tempfile.gettempdir()) / "clef-tune-ui"
STALE_HOURS = 24
UNZIP_FACTOR = 5  # a zip may expand to this many times the upload cap
DEFAULT_MODEL = "Cloudflare/clef-flash"
DATA_SUFFIXES = (".csv", ".tsv", ".jsonl", ".ndjson")
NOUL_IDS = ("true", "false")
# Fields the example (or any loaded config) may set beyond the schema and the data columns.
_TUNABLE_PREFIXES = ("train.", "eval.", "output.save_every", "data.eval_fraction")


class UploadError(ValueError):
    """An upload cannot be used; the message says why."""


@dataclass
class OptionDraft:
    id: str
    description: str = ""


def _is_int(text: str) -> bool:
    return bool(re.fullmatch(r"-?\d+", text.strip()))


@dataclass
class QuestionDraft:
    id: str
    column: str  # the label column (flat files) or the labels key (native JSONL)
    type: str
    instructions: str = ""
    options: list[OptionDraft] = field(default_factory=list)

    def set_type(self, new_type: str, observed: list[str]) -> None:
        if new_type == "noul":
            described = {option.id: option.description for option in self.options}
            self.options = [OptionDraft(oid, described.get(oid, "")) for oid in NOUL_IDS]
        elif new_type == "choice":
            if not self.options or [option.id for option in self.options] == list(NOUL_IDS):
                self.options = [OptionDraft(value) for value in observed]
        elif new_type == "score":
            if self.options and all(_is_int(option.id) for option in self.options):
                self.options = sorted(self.options, key=lambda option: int(option.id))
        else:
            raise ValueError(f"unknown question type {new_type!r}")
        self.type = new_type

    def to_clef(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type}
        if self.instructions.strip():
            out["instructions"] = self.instructions.strip()
        if self.type == "choice":
            out["criteria"] = {o.id: o.description.strip() or None for o in self.options}
        elif self.type == "score":
            out["criteria"] = [o.description.strip() or o.id for o in self.options]
        else:
            described = {o.id: o.description.strip() for o in self.options if o.description}
            if described:
                out["criteria"] = described
        return out


def _set_path(data: dict[str, Any], dotted: str, value: Any) -> None:
    *parents, leaf = dotted.split(".")
    for parent in parents:
        data = data.setdefault(parent, {})
    data[leaf] = value


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict) and value and key != "question_weights":
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


@dataclass
class Draft:
    run_name: str
    model_id: str = DEFAULT_MODEL
    id_column: str | None = None
    state_columns: list[str] = field(default_factory=list)
    image_columns: list[str] = field(default_factory=list)
    questions: list[QuestionDraft] = field(default_factory=list)
    native: bool = False
    preset: str = "balanced"
    overrides: dict[str, Any] = field(default_factory=dict)  # dotted path -> value

    def question(self, qid: str) -> QuestionDraft | None:
        return next((q for q in self.questions if q.id == qid), None)

    def to_config(self, base_dir: Path, data_path: str) -> Config:
        """The config this draft describes. Raises ValueError (pydantic's ValidationError and
        SchemaError are ValueErrors) with a message that names the bad field."""
        data: dict[str, Any] = {"train": data_path}
        if not self.native:
            if self.id_column:
                data["id_column"] = self.id_column
            data["state_columns"] = list(self.state_columns)
            labels = {q.id: q.column for q in self.questions if q.column != q.id}
            if labels:
                data["label_columns"] = labels
            if self.image_columns:
                data["image_columns"] = list(self.image_columns)
        raw = {
            "run_name": self.run_name,
            "model": {"id": self.model_id},
            "data": data,
            "schema": {"questions": {q.id: q.to_clef() for q in self.questions}},
        }
        config = Config.model_validate(raw).with_base_dir(base_dir).with_preset(self.preset)
        if not self.overrides:
            return config
        values = config.to_dict()
        for dotted, value in self.overrides.items():
            _set_path(values, dotted, value)
        return Config.model_validate(values).with_base_dir(base_dir)

    def fingerprint(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass
class Table:
    path: Path
    rows: list[dict[str, Any]]
    columns: list[str]
    issues: list[Issue]

    def values(self, column: str) -> list[Any]:
        """A column's cells; for native JSONL rows, a question's labels."""
        return [
            row["labels"].get(column) if isinstance(row.get("labels"), dict) else row.get(column)
            for row in self.rows
        ]

    def observed(self, column: str) -> list[str]:
        """Distinct values, most common first, then by text."""
        counts = Counter(
            str(value).strip()
            for value in self.values(column)
            if not is_missing(value) and not isinstance(value, dict) and str(value).strip()
        )
        return [value for value, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


@dataclass
class Session:
    root: Path | None = None
    table: Table | None = None
    draft: Draft | None = None
    report: ValidationReport | None = None
    estimates: list[EstimateRow] = field(default_factory=list)
    reached: int = 0  # the furthest step index the person may open
    job_id: str | None = None  # the run the monitor shows
    local_folder: str | None = None
    pending: Any = None  # a reviewed launch waiting for confirmation (ui.runs.PendingLaunch)
    last_view: Any = None  # the monitor's last good view, kept through a failed poll
    results: dict[str, dict] = field(default_factory=dict)  # job id -> metrics.json

    def workdir(self) -> Path:
        if self.root is None:
            self.root = SESSIONS_ROOT / uuid.uuid4().hex
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def close(self) -> None:
        if self.root is not None:
            shutil.rmtree(self.root, ignore_errors=True)

    def data_path(self) -> str:
        if self.table is None:
            raise ValueError("load a data file first")
        return self.table.path.relative_to(self.workdir()).as_posix()

    def config(self) -> Config:
        if self.draft is None:
            raise ValueError("load a data file first")
        return self.draft.to_config(self.workdir(), self.data_path())

    def write_config(self) -> Path:
        path = self.workdir() / "config.yaml"
        self.config().save(path)
        return path


def slug(text: str) -> str:
    """A valid run_name from any text (such as a file name)."""
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    value = re.sub(r"^[^A-Za-z0-9]+", "", value)
    return value[:64].rstrip("-") or "run"


def _skipped(name: str) -> bool:
    return name.startswith("__MACOSX/") or Path(name).name.startswith(".")


def extract_zip(archive: Path, dest: Path, max_bytes: int) -> Path:
    """Unpack a zip into dest after checking every member stays inside it and the total size
    stays under max_bytes; return the one data file inside."""
    try:
        bundle = zipfile.ZipFile(archive)
    except zipfile.BadZipFile as exc:
        raise UploadError(f"{archive.name} is not a valid zip file") from exc
    with bundle:
        members = [m for m in bundle.infolist() if not _skipped(m.filename)]
        root = dest.resolve()
        targets = []
        for member in members:
            target = (root / member.filename).resolve()
            if target != root and root not in target.parents:
                raise UploadError(f"the zip has a file outside its folder: {member.filename}")
            targets.append(target)
        if sum(member.file_size for member in members) > max_bytes:
            raise UploadError(f"the zip expands to more than {max_bytes / 2**20:.0f} MB")
        data = [
            target
            for member, target in zip(members, targets, strict=True)
            if not member.is_dir() and target.suffix.lower() in DATA_SUFFIXES
        ]
        if len(data) != 1:
            raise UploadError(
                f"the zip must contain exactly one .csv, .tsv or .jsonl file (found {len(data)})"
            )
        for member, target in zip(members, targets, strict=True):
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, open(target, "wb") as out:
                shutil.copyfileobj(source, out)
    return data[0]


def _read_table(path: Path) -> Table:
    try:
        rows, columns, issues = read_rows(path)
    except DataError as exc:
        raise UploadError(str(exc)) from exc
    return Table(path, [row for _, row in rows], columns, issues)


def _start(session: Session, table: Table, draft: Draft) -> None:
    session.table, session.draft = table, draft
    session.report, session.estimates = None, []
    session.pending = session.last_view = session.job_id = None
    session.results = {}
    session.reached = 1


def draft_from_proposal(proposal: Proposal, name: str) -> Draft:
    questions = [
        QuestionDraft(q.id, q.column, q.type, options=[OptionDraft(o) for o in q.options])
        for q in proposal.questions
    ]
    return Draft(
        run_name=slug(name),
        id_column=proposal.id_column,
        state_columns=list(proposal.state_columns),
        questions=questions,
        native=proposal.native,
    )


def _option_drafts(question: Any) -> list[OptionDraft]:
    if question.type == "score":
        return [OptionDraft(str(i), str(text)) for i, text in enumerate(question.criteria)]
    criteria = question.criteria or {}
    if question.type == "noul":
        return [OptionDraft(oid, str(criteria.get(oid) or "")) for oid in NOUL_IDS]
    return [OptionDraft(str(k), "" if v is None else str(v)) for k, v in criteria.items()]


def draft_from_config(config: Config) -> Draft:
    schema = config.get_schema()
    questions = [
        QuestionDraft(
            qid,
            config.data.label_columns.get(qid, qid),
            question.type,
            question.instructions or "",
            _option_drafts(question),
        )
        for qid, question in (schema.questions.items() if schema else [])
    ]
    defaults = _flatten(Config.model_validate({"run_name": "x", "data": {"train": "x"}}).to_dict())
    overrides = {
        path: value
        for path, value in _flatten(config.to_dict()).items()
        if path.startswith(_TUNABLE_PREFIXES) and defaults.get(path) != value
    }
    return Draft(
        run_name=config.run_name,
        model_id=config.model.id,
        id_column=config.data.id_column,
        state_columns=list(config.data.state_columns or []),
        image_columns=list(config.data.image_columns),
        questions=questions,
        overrides=overrides,
    )


def example_files() -> tuple[Path, Path]:
    """The support-tickets example: shipped in the wheel, or read from a checkout."""
    packaged = Path(str(resources.files("clef_tune") / "_examples"))
    folder = packaged if packaged.is_dir() else Path(clef_tune.__file__).parents[1] / "examples"
    return folder / "support_tickets.csv", folder / "support_tickets.yaml"


def load_upload(session: Session, upload: Path, settings: Settings) -> None:
    """Use an uploaded CSV/TSV/JSONL file (or a zip holding one, with its images) and propose a
    schema for it. Any UploadError leaves the session as it was."""
    upload = Path(upload)
    size = upload.stat().st_size
    if size > settings.max_upload_bytes:
        raise UploadError(
            f"the file is {size / 2**20:.1f} MB; the limit is {settings.max_upload_mb} MB"
        )
    staging = session.workdir() / f"upload-{uuid.uuid4().hex[:8]}"
    try:
        if upload.suffix.lower() == ".zip":
            data_file = extract_zip(upload, staging, settings.max_upload_bytes * UNZIP_FACTOR)
        else:
            staging.mkdir(parents=True)
            data_file = staging / upload.name
            shutil.copyfile(upload, data_file)
        table = _read_table(data_file)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    data_dir = session.workdir() / "data"
    shutil.rmtree(data_dir, ignore_errors=True)
    staging.rename(data_dir)
    table.path = data_dir / data_file.resolve().relative_to(staging.resolve())
    _start(session, table, draft_from_proposal(propose(table.rows, table.columns), upload.stem))


def load_example(session: Session) -> None:
    csv_path, yaml_path = example_files()
    data_dir = session.workdir() / "data"
    shutil.rmtree(data_dir, ignore_errors=True)
    data_dir.mkdir(parents=True)
    target = data_dir / csv_path.name
    shutil.copyfile(csv_path, target)
    _start(session, _read_table(target), draft_from_config(load_config(yaml_path)))


def sweep_stale(
    root: Path = SESSIONS_ROOT, *, max_age_hours: float = STALE_HOURS, now: float | None = None
) -> int:
    """Delete session folders untouched for max_age_hours: Gradio skips the cleanup callback
    for sessions it evicts, and a restart forgets every session."""
    if not root.is_dir():
        return 0
    now = time.time() if now is None else now
    removed = 0
    for child in root.iterdir():
        if child.is_dir() and now - child.stat().st_mtime > max_age_hours * 3600:
            shutil.rmtree(child, ignore_errors=True)
            removed += 1
    return removed
