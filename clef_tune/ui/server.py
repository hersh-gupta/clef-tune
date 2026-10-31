"""The web UI: a small FastAPI app that serves clef-tune's pages. Pages are Jinja templates
filled from ui/views.py and ui/runs.py; forms change the session through ui/handlers.py.
Background form posts get back <template data-swap> blocks (see static/clef-tune.js), so only
the parts of a page that changed are replaced.

Sessions live in memory, keyed by a signed cookie. On a Hugging Face Space, people sign in
with Hugging Face (attach_huggingface_oauth) and every Hub call uses their own token; the
server's token is never used for a visitor. Elsewhere the machine's own login is used."""

from __future__ import annotations

import dataclasses
import os
import secrets
import shutil
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from clef_tune.config import Config
from clef_tune.predictions import check_input, read_table, summarize, write_input_csv, write_table
from clef_tune.ui import handlers, local, runs, views
from clef_tune.ui import wording as w
from clef_tune.ui.services import Services
from clef_tune.ui.session import STALE_HOURS, Session, UploadError, extract_zip, sweep_stale
from clef_tune.ui.settings import Settings, SignInRequired, user_token

HERE = Path(__file__).parent
SESSION_KEY = "clef_tune_sid"
FROM_SCRIPT = "clef-tune"  # the X-Requested-With header clef-tune.js sends
CHUNK = 1 << 20
LOSS_WIDTH, LOSS_HEIGHT = 600, 90
LATEST_SHOWN = 2


class SessionStore:
    """In-memory sessions, each with its own lock; idle ones are closed after a day."""

    def __init__(self, max_idle_hours: float = STALE_HOURS) -> None:
        self.max_idle = max_idle_hours * 3600
        self.sessions: dict[str, Session] = {}
        self.locks: dict[str, threading.Lock] = {}
        self.seen: dict[str, float] = {}
        self.guard = threading.Lock()

    def get(self, sid: str) -> tuple[Session, threading.Lock]:
        with self.guard:
            now = time.monotonic()
            for old in [s for s, t in self.seen.items() if now - t > self.max_idle and s != sid]:
                self.sessions.pop(old).close()
                self.locks.pop(old, None)
                self.seen.pop(old, None)
            if sid not in self.sessions:
                self.sessions[sid] = Session()
                self.locks[sid] = threading.Lock()
            self.seen[sid] = now
            return self.sessions[sid], self.locks[sid]

    def drop(self, sid: str) -> None:
        """Close a session and delete its files."""
        with self.guard:
            session = self.sessions.pop(sid, None)
            self.locks.pop(sid, None)
            self.seen.pop(sid, None)
        if session is not None:
            session.close()


@dataclass
class Ctx:
    """What one request knows: its session (and lock), and who is signed in."""

    session: Session
    lock: threading.Lock
    token: str | None
    user: str | None


def _login_url(target: str = "/") -> str:
    return "/oauth/huggingface/login?" + urlencode({"_target_url": target})


def _day(value: datetime | None) -> str:
    return f"{value:%b} {value.day}" if value else ""


def _swaps(*parts: tuple[str, str], animate: tuple[str, ...] = ()) -> HTMLResponse:
    body = "".join(
        f'<template data-swap="{selector}"{" data-animate" if selector in animate else ""}>'
        f"{html}</template>"
        for selector, html in parts
    )
    return HTMLResponse(body)


def _loss_points(loss: list[tuple[int, float]]) -> str:
    if len(loss) < 2:
        return ""
    steps = [step for step, _ in loss]
    values = [value for _, value in loss]
    first, last = min(steps), max(steps)
    top = max(values) or 1.0
    return " ".join(
        f"{(step - first) / max(last - first, 1) * LOSS_WIDTH:.1f},"
        f"{LOSS_HEIGHT - value / top * (LOSS_HEIGHT - 5):.1f}"
        for step, value in loss
    )


def _model_name(session: Session) -> str:
    draft = session.draft
    return views.MODELS.get(draft.model_id, ("the model",))[0] if draft else "the model"


def _noun(session: Session) -> str:
    return w.noun_or_default(session.draft.row_noun if session.draft else None)


def pop_flash(session: Session) -> str | None:
    """A message left by a form post for the page it redirects to."""
    message, session.flash = session.flash, None
    return message


def _replace(plan: Any, **changes: Any) -> Any:
    if dataclasses.is_dataclass(plan):
        return dataclasses.replace(plan, **changes)
    for key, value in changes.items():
        setattr(plan, key, value)
    return plan


def create_app(
    settings: Settings | None = None,
    services: Services | None = None,
    *,
    secret: str | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    services = services or Services()
    app = FastAPI(title="clef-tune", docs_url=None, redoc_url=None, openapi_url=None)
    if settings.space:
        from huggingface_hub import attach_huggingface_oauth

        attach_huggingface_oauth(app)  # also adds the (SameSite=None) session middleware
    else:
        app.add_middleware(
            SessionMiddleware,
            secret_key=secret or os.environ.get("CLEF_TUNE_SECRET") or secrets.token_hex(32),
            session_cookie="clef_tune",
            same_site="lax",
            https_only=False,
        )
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.globals.update(
        # Versioned by the file's own time, so browsers pick up a changed file straight away.
        asset=lambda name: f"/static/{name}?v={int((HERE / 'static' / name).stat().st_mtime)}",
        plural=w.plural,
        login_url=_login_url,
    )
    templates.env.filters["day"] = _day
    store = SessionStore()
    names: dict[str, str] = {}
    sweep_stale()

    # ------------------------------------------------------------------------ helpers

    def ctx(request: Request) -> Ctx:
        sid = request.session.get(SESSION_KEY)
        if not sid:
            sid = uuid.uuid4().hex
            request.session[SESSION_KEY] = sid
        session, lock = store.get(sid)
        oauth, user = None, None
        if settings.space:
            from huggingface_hub import parse_huggingface_oauth

            info = parse_huggingface_oauth(request)
            if info is not None:
                oauth = info.access_token
                user = getattr(info.user_info, "preferred_username", None)
        try:
            token = user_token(settings, oauth, session.pasted_token)
        except SignInRequired:
            token = None
        if user is None and token:
            if token not in names:
                try:
                    names[token] = services.whoami(token)
                except Exception:  # offline, or a token without read access
                    names[token] = ""
            user = names[token] or None
        return Ctx(session, lock, token, user)

    def render(request: Request, c: Ctx, name: str, **context: Any) -> HTMLResponse:
        return templates.TemplateResponse(
            request, name, {"user": c.user, "space": settings.space, **context}
        )

    def fragment(request: Request, c: Ctx, name: str, **context: Any) -> str:
        return templates.env.get_template(name).render(
            request=request, user=c.user, space=settings.space, **context
        )

    def go(request: Request, url: str) -> Response:
        if request.headers.get("X-Requested-With") == FROM_SCRIPT:
            return Response(status_code=204, headers={"X-Redirect": url})
        return RedirectResponse(url, status_code=303)

    def refresh(c: Ctx) -> None:
        handlers.refresh(c.session, services)

    def save_upload(c: Ctx, upload: UploadFile, folder: str) -> tuple[Path | None, str | None]:
        """Stream an upload into the session's folder under its own name, within the cap."""
        name = Path(upload.filename or "upload").name or "upload"
        target_dir = c.session.workdir() / folder / uuid.uuid4().hex[:8]
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / name
        size = 0
        with open(target, "wb") as out:
            while chunk := upload.file.read(CHUNK):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    out.close()
                    shutil.rmtree(target_dir, ignore_errors=True)
                    return None, (
                        f"Could not use {name}: it is over the {settings.max_upload_mb} MB limit"
                    )
                out.write(chunk)
        return target, None

    def step_page(request: Request, c: Ctx, step: str) -> Response | None:
        """Send a person back to the furthest step they can open, if this one is not open."""
        index = [key for key, _, _ in views.STEPS].index(step)
        furthest = views.furthest_step(c.session)
        if index > furthest:
            return RedirectResponse(views.STEPS[furthest][2], status_code=303)
        return None

    # ------------------------------------------------------------------------ welcome, models

    @app.get("/", response_class=HTMLResponse)
    def welcome(request: Request) -> Response:
        c = ctx(request)
        latest: list[runs.ModelRow] = []
        total = 0
        if c.token and c.user:
            rows, _ = runs.list_models(services, token=c.token)
            latest, total = rows[:LATEST_SHOWN], len(rows)
        samples = [
            ("support_tickets", "support tickets", "Which team, how urgent, is it an outage"),
            ("claims_triage", "claim triage", "Which reviewer, how risky, pay in full"),
            ("server_logs", "server logs", "How severe, which component, act now"),
        ]
        return render(
            request,
            c,
            "welcome.html",
            returning=bool(latest),
            latest=latest,
            total=total,
            greeting="Hello again",
            samples=samples,
        )

    @app.post("/signout")
    def signout(request: Request) -> Response:
        """Sign out, and forget everything set up in this browser: on a shared Space, the
        next person must not find the last one's data."""
        sid = request.session.pop(SESSION_KEY, None)
        if sid:
            store.drop(sid)
        target = "/oauth/huggingface/logout?_target_url=/" if settings.space else "/"
        return RedirectResponse(target, status_code=303)

    @app.get("/models", response_class=HTMLResponse)
    def models(request: Request) -> Response:
        c = ctx(request)
        local_state, _ = local.local_state(c.session, model=_model_name(c.session))
        labels = {
            "running": "Training",
            "done": "Ready",
            "stopped": "Stopped",
            "failed": "Didn't finish",
        }
        if not c.token:
            return render(
                request,
                c,
                "models.html",
                need_signin=settings.space,
                rows=[],
                error=None,
                local=local_state,
                local_label=labels.get(local_state.status) if local_state else "",
            )
        rows, error = runs.list_models(services, token=c.token)
        return render(
            request,
            c,
            "models.html",
            need_signin=False,
            rows=rows,
            error=error,
            local=local_state,
            local_label=labels.get(local_state.status) if local_state else "",
        )

    @app.get("/models/{job_id}/line", response_class=HTMLResponse)
    def model_line(request: Request, job_id: str) -> Response:
        c = ctx(request)
        if not c.token:
            return HTMLResponse("")
        with c.lock:
            state, _ = runs.job_state(
                c.session, services, job_id, token=c.token, model="the model", noun="example"
            )
            if state is None:
                return HTMLResponse("")
            results = None
            if state.status == "done" and state.kind == "train":
                status = services.job_status(job_id, token=c.token)
                results, _ = runs.job_results(c.session, services, status, token=c.token)
        return _swaps((f"#line-{job_id}", runs.model_detail(state, results)))

    # ------------------------------------------------------------------------ step 1: data

    @app.get("/data", response_class=HTMLResponse)
    def data(request: Request) -> Response:
        c = ctx(request)
        with c.lock:
            error = pop_flash(c.session)
            page = views.data_page(c.session, settings, error)
            steps = views.step_bar(c.session, "data")
        return render(request, c, "data.html", page=page, steps=steps, roles=handlers.ROLE_CHOICES)

    @app.post("/data/upload")
    def data_upload(request: Request, file: UploadFile = File(...)) -> Response:
        c = ctx(request)
        with c.lock:
            path, error = save_upload(c, file, "incoming")
            if path is not None:
                error = handlers.upload(c.session, path, settings)
                shutil.rmtree(path.parent, ignore_errors=True)
                if error is None:
                    refresh(c)
            if error:
                c.session.flash = error
        return go(request, "/data")

    @app.post("/data/sample")
    def data_sample(request: Request, name: str = Form(...)) -> Response:
        c = ctx(request)
        with c.lock:
            try:
                handlers.sample(c.session, name)
            except ValueError as exc:
                c.session.flash = str(exc)
                return go(request, "/data")
            refresh(c)
        return go(request, "/data")

    @app.post("/data/remove")
    def data_remove(request: Request) -> Response:
        c = ctx(request)
        with c.lock:
            handlers.remove_data(c.session)
        return go(request, "/data")

    @app.post("/data/noun")
    def data_noun(request: Request, noun: str = Form("")) -> Response:
        c = ctx(request)
        with c.lock:
            handlers.set_noun(c.session, noun)
            page = views.data_page(c.session, settings)
        mark = "Guessed from the file name" if page.noun_guessed else ""
        return _swaps(("#noun-mark", mark))

    @app.post("/data/role")
    def data_role(request: Request, column: str = Form(...), role: str = Form(...)) -> Response:
        c = ctx(request)
        with c.lock:
            error = handlers.set_role(c.session, column, role)
            refresh(c)
            page = views.data_page(c.session, settings, error)
            steps = views.step_bar(c.session, "data")
        body = fragment(request, c, "_data_body.html", page=page, roles=handlers.ROLE_CHOICES)
        bar = templates.env.get_template("macros.html").module.step_bar(steps)
        return _swaps(("#data-body", body), ("#steps-slot", str(bar)))

    @app.get("/data/rows", response_class=HTMLResponse)
    def data_rows(request: Request, page: int = 0) -> Response:
        c = ctx(request)
        with c.lock:
            rows = views.rows_page(c.session, page)
        if rows is None:
            return RedirectResponse("/data", status_code=303)
        return render(request, c, "rows.html", rows=rows)

    # ------------------------------------------------------------------------ step 2: questions

    @app.get("/questions", response_class=HTMLResponse)
    def questions(request: Request, q: str | None = None) -> Response:
        c = ctx(request)
        with c.lock:
            if (redirect := step_page(request, c, "questions")) is not None:
                return redirect
            page = views.questions_page(c.session, q)
            steps = views.step_bar(c.session, "questions")
        if request.headers.get("X-Requested-With") == FROM_SCRIPT and page.selected:
            # Switching question in the page: only the list, switcher and editor change.
            parts = templates.env.get_template("_qparts.html").module
            editor = fragment(
                request, c, "_editor.html", page=page, kinds=views.KINDS, noun=page.noun
            )
            return _swaps(
                ("#qlist", str(parts.qlist(page))),
                ("#qswitch", str(parts.qswitch(page))),
                ("#editor", editor),
                animate=("#editor",),
            )
        return render(
            request, c, "questions.html", page=page, steps=steps, kinds=views.KINDS, noun=page.noun
        )

    def editor_swap(request: Request, c: Ctx, qid: str, *, whole: bool) -> Response:
        page = views.questions_page(c.session, qid)
        if whole:
            html = fragment(
                request, c, "_editor.html", page=page, kinds=views.KINDS, noun=page.noun
            )
            return _swaps(("#editor", html))
        if page.selected is None:
            return HTMLResponse("")
        flags = "".join(
            f'<p class="flag">{templates.env.filters["e"](f)}</p>' for f in page.selected.flags
        )
        number = [item.id for item in page.items].index(page.selected.id) + 1
        # Numbered per question, so an answer that lands after a switch changes nothing.
        return _swaps((f"#flags-{number}", flags))

    @app.post("/questions/{qid}/ask")
    def question_ask(request: Request, qid: str, ask: str = Form("")) -> Response:
        c = ctx(request)
        with c.lock:
            handlers.update_question(c.session, qid, instructions=ask)
            refresh(c)
            return editor_swap(request, c, qid, whole=False)

    @app.post("/questions/{qid}/type")
    def question_type(request: Request, qid: str, kind: str = Form(...)) -> Response:
        c = ctx(request)
        with c.lock:
            try:
                handlers.update_question(c.session, qid, type=kind)
            except ValueError:
                pass
            refresh(c)
            return editor_swap(request, c, qid, whole=True)

    @app.post("/questions/{qid}/describe")
    def question_describe(
        request: Request, qid: str, option: str = Form(...), description: str = Form("")
    ) -> Response:
        c = ctx(request)
        with c.lock:
            handlers.describe_option(c.session, qid, option, description)
            refresh(c)
            return editor_swap(request, c, qid, whole=False)

    @app.post("/questions/{qid}/move")
    def question_move(
        request: Request, qid: str, option: str = Form(...), offset: int = Form(...)
    ) -> Response:
        c = ctx(request)
        with c.lock:
            handlers.move_option(c.session, qid, option, offset)
            refresh(c)
            return editor_swap(request, c, qid, whole=True)

    @app.get("/questions/prompt", response_class=HTMLResponse)
    def question_prompt(request: Request, row: int = 0) -> Response:
        c = ctx(request)
        with c.lock:
            pieces = views.prompt_preview(c.session, row)
        return _swaps(("#prompt-body", fragment(request, c, "_prompt.html", pieces=pieces)))

    # ------------------------------------------------------------------------ step 3: settings

    def settings_view(
        request: Request,
        c: Ctx,
        field_error: tuple[str, str] | None = None,
        holdout_error: tuple[str, str] | None = None,
    ) -> str:
        """The Settings body; an error is (field path, message) or (typed percent, message)."""
        page = views.settings_page(c.session, services, c.user)
        return fragment(
            request,
            c,
            "_settings_body.html",
            page=page,
            field_error=field_error,
            holdout_error=holdout_error,
            holdout_range=handlers.HOLDOUT_PERCENT,
        )

    @app.get("/settings", response_class=HTMLResponse)
    def settings_page(request: Request) -> Response:
        c = ctx(request)
        with c.lock:
            if (redirect := step_page(request, c, "settings")) is not None:
                return redirect
            if c.session.report is None:
                refresh(c)
            page = views.settings_page(c.session, services, c.user)
            steps = views.step_bar(c.session, "settings")
        return render(
            request,
            c,
            "settings.html",
            page=page,
            steps=steps,
            field_error=None,
            holdout_error=None,
            holdout_range=handlers.HOLDOUT_PERCENT,
        )

    @app.post("/settings/name")
    def settings_name(request: Request, name: str = Form("")) -> Response:
        c = ctx(request)
        with c.lock:
            error = handlers.set_run_name(c.session, name)
            if error:
                return _swaps(("#name-note", templates.env.filters["e"](error)))
            page = views.settings_page(c.session, services, c.user)
        note = (
            f"Saved to your Hugging Face account as <b>{templates.env.filters['e'](page.repo)}</b>"
        )
        return _swaps(("#name-note", note))

    @app.post("/settings/model")
    def settings_model(request: Request, model: str = Form(...)) -> Response:
        c = ctx(request)
        with c.lock:
            if model in views.MODELS:
                handlers.set_model(c.session, model)
                c.session.flavor = None
                refresh(c)
            return _swaps(("#settings-body", settings_view(request, c)))

    @app.post("/settings/preset")
    def settings_preset(request: Request, preset: str = Form(...)) -> Response:
        c = ctx(request)
        with c.lock:
            if preset in views.PRESETS:
                handlers.set_preset(c.session, preset)
                refresh(c)
            return _swaps(("#settings-body", settings_view(request, c)))

    @app.post("/settings/field")
    async def settings_field(request: Request) -> Response:
        form = await request.form()
        path = str(form.get("path", ""))
        values = [str(v) for v in form.getlist("value")]
        value: Any = values[-1] if values else ""
        c = ctx(request)
        spec = next((s for s in handlers.ADVANCED_FIELDS if s.path == path), None)
        if spec is not None and spec.kind == "bool":
            value = value == "1"
        with c.lock:
            error = handlers.set_field(c.session, path, value)
            refresh(c)
            field_error = (path, views.field_error_text(path, error)) if error else None
            return _swaps(("#settings-body", settings_view(request, c, field_error)))

    @app.post("/settings/holdout")
    def settings_holdout(request: Request, percent: str = Form("")) -> Response:
        c = ctx(request)
        with c.lock:
            error = handlers.set_holdout(c.session, percent)
            refresh(c)
            holdout_error = (percent, error) if error else None
            return _swaps(
                ("#settings-body", settings_view(request, c, holdout_error=holdout_error))
            )

    @app.post("/settings/flavor")
    def settings_flavor(request: Request, flavor: str = Form("")) -> Response:
        c = ctx(request)
        with c.lock:
            c.session.flavor = flavor or None
            return _swaps(("#settings-body", settings_view(request, c)))

    # ------------------------------------------------------------------------ step 4: train

    @app.get("/train", response_class=HTMLResponse)
    def train(request: Request) -> Response:
        c = ctx(request)
        with c.lock:
            if (redirect := step_page(request, c, "train")) is not None:
                return redirect
            steps = views.step_bar(c.session, "train")
            error = pop_flash(c.session)
        return render(request, c, "train.html", steps=steps, error=error)

    def review_fragment(request: Request, c: Ctx) -> str:
        session = c.session
        draft = session.draft
        model = _model_name(session)
        preset = views.PRESETS.get(draft.preset, ("",))[0] if draft else ""
        common = {"local_training": settings.local_training, "delete_data": session.delete_data}
        if not c.token:
            page = views.settings_page(session, services, None)
            return fragment(
                request,
                c,
                "_review.html",
                need_signin=True,
                model=model,
                preset=preset,
                settings_estimate=page.estimate,
                **common,
            )
        repo = f"{c.user}/{draft.run_name}" if c.user and draft else None
        ready = runs.review_launch(
            session,
            services,
            token=c.token,
            adapter_repo=repo,
            delete_data=session.delete_data,
            model=model,
            preset=preset,
            flavor=session.flavor,
        )
        return fragment(request, c, "_review.html", need_signin=False, ready=ready, **common)

    @app.get("/train/review", response_class=HTMLResponse)
    def train_review(request: Request) -> Response:
        c = ctx(request)
        with c.lock:
            return _swaps(("#review", review_fragment(request, c)), animate=("#review",))

    @app.post("/train/delete-data")
    async def train_delete_data(request: Request) -> Response:
        form = await request.form()
        c = ctx(request)
        with c.lock:
            c.session.delete_data = "1" in form.getlist("delete")
            return _swaps(("#review", review_fragment(request, c)))

    @app.post("/train/token")
    def train_token(request: Request, token: str = Form("")) -> Response:
        """A token pasted for this browser session, used when sign-in lacks a permission."""
        c = ctx(request)
        with c.lock:
            c.session.pasted_token = token.strip() or None
        return go(request, "/train")

    @app.post("/train/start")
    def train_start(request: Request) -> Response:
        c = ctx(request)
        if not c.token:
            return go(request, _login_url("/train"))
        with c.lock:
            job_id, error = runs.confirm_launch(c.session, services, token=c.token)
            if error:
                c.session.flash = error
                return go(request, "/train")
        return go(request, f"/runs/{job_id}")

    @app.post("/train/local")
    def train_local(request: Request) -> Response:
        c = ctx(request)
        if not settings.local_training:
            return go(request, "/train")
        with c.lock:
            try:
                local.start_local(c.session, local.default_folder(c.session))
            except local.LocalRunError as exc:
                c.session.flash = str(exc)
                return go(request, "/train")
        return go(request, f"/runs/{local.LOCAL_ID}")

    @app.get("/train/bundle")
    def train_bundle(request: Request) -> Response:
        c = ctx(request)
        with c.lock:
            try:
                path = handlers.bundle(c.session)
            except ValueError as exc:
                c.session.flash = str(exc)
                return RedirectResponse("/train", status_code=303)
        return FileResponse(path, filename=path.name, media_type="application/zip")

    # ------------------------------------------------------------------------ runs

    def job_view(c: Ctx, job_id: str) -> tuple[Any, Any, str | None]:
        """(status, state, notice) for a job; status is None when it could not be read."""
        try:
            status = services.job_status(job_id, token=c.token)
        except Exception as exc:  # unreachable, unknown job or no access
            last = c.session.last_states.get(job_id)
            return None, last, f"Could not read this run from Hugging Face ({exc})."
        noun = _noun(c.session) if c.session.job_id == job_id else "example"
        if status.kind == "predict" and c.session.prediction is not None:
            noun = w.noun_or_default(c.session.prediction.config.data.row_noun)
        model = _model_name(c.session) if c.session.job_id == job_id else "the model"
        state = runs.run_state(status, model=model, noun=noun)
        c.session.last_states[job_id] = state
        return status, state, None

    def title_for(state: runs.RunState) -> str:
        if state.status == "running":
            return f"Training {state.name}" if state.kind == "train" else "Predicting"
        if state.status == "stopped":
            return f"{state.name} was stopped"
        return f"{state.name} didn't finish" if state.status == "failed" else state.name

    def predictions_csv(c: Ctx, job_id: str) -> Path:
        path = c.session.workdir() / "predictions" / f"{job_id}.csv"
        if not path.exists():
            services.download_predictions(job_id, path, token=c.token)
        return path

    def prediction_config(c: Ctx, repo: str | None) -> Config:
        draft = c.session.prediction
        if draft is not None and draft.adapter_repo == repo:
            return draft.config
        files = c.session.results.get(f"repo:{repo}")
        if files is None and repo:
            files = services.pull_run_files(repo, token=c.token)
            c.session.results[f"repo:{repo}"] = files
        return (
            Config.model_validate(files.config)
            if files
            else Config.model_validate({"run_name": "x", "data": {"train": "x"}})
        )

    @app.get("/runs/{job_id}", response_class=HTMLResponse)
    def run_page(request: Request, job_id: str) -> Response:
        c = ctx(request)
        common = {"poll_seconds": runs.POLL_SECONDS, "usual": "10 to 20 minutes", "notice": None}
        if job_id == local.LOCAL_ID:
            with c.lock:
                state, results = local.local_state(
                    c.session, model=_model_name(c.session), noun=_noun(c.session)
                )
            if state is None:
                return RedirectResponse("/models", status_code=303)
            if results is not None:
                return render(request, c, "results.html", view=results, job_id=job_id)
            return render(
                request,
                c,
                "run.html",
                state=state,
                title=title_for(state),
                loss_points=_loss_points(state.loss),
                **common,
            )
        if not c.token:
            return render(
                request,
                c,
                "models.html",
                need_signin=True,
                rows=[],
                error=None,
                local=None,
                local_label="",
            )
        with c.lock:
            status, state, notice = job_view(c, job_id)
            if state is None:
                return render(
                    request,
                    c,
                    "models.html",
                    need_signin=False,
                    rows=[],
                    error=notice,
                    local=None,
                    local_label="",
                )
            if status is not None and state.status == "done" and state.kind == "train":
                view, problem = runs.job_results(c.session, services, status, token=c.token)
                if view is not None:
                    return render(request, c, "results.html", view=view, job_id=job_id)
                notice = problem
            if status is not None and state.status == "done" and state.kind == "predict":
                try:
                    config = prediction_config(c, status.adapter_repo)
                    header, rows = read_table(predictions_csv(c, job_id))
                    draft = c.session.prediction
                    threshold = draft.threshold if draft is not None else 0.7
                    result = views.prediction_result(
                        summarize(header, rows, config), config, threshold=threshold
                    )
                except Exception as exc:  # not downloadable yet, or the model was deleted
                    notice = f"The spreadsheet isn't readable yet ({exc}). Try again shortly."
                else:
                    return render(
                        request,
                        c,
                        "prediction_results.html",
                        state=state,
                        result=result,
                        took=w.duration(state.hours or 0),
                        downloads=views.DOWNLOADS,
                    )
        common["notice"] = notice
        return render(
            request,
            c,
            "run.html",
            state=state,
            title=title_for(state),
            loss_points=_loss_points(state.loss),
            **common,
        )

    @app.get("/runs/{job_id}/status", response_class=HTMLResponse)
    def run_status(request: Request, job_id: str) -> Response:
        c = ctx(request)
        with c.lock:
            if job_id == local.LOCAL_ID:
                state, _ = local.local_state(
                    c.session, model=_model_name(c.session), noun=_noun(c.session)
                )
                notice = None
            elif c.token:
                _, state, notice = job_view(c, job_id)
            else:
                state, notice = None, "Sign in again to follow this run."
        if state is None:
            return Response(status_code=204, headers={"X-Stop-Polling": "1"})
        html = fragment(
            request,
            c,
            "_run_status.html",
            state=state,
            notice=notice,
            usual="10 to 20 minutes",
            loss_points=_loss_points(state.loss),
        )
        response = _swaps(("#run-status", html))
        if state.status != "running":
            response.headers["X-Run-Ended"] = state.status
            response.headers["X-Refresh"] = "1"
            response.headers["X-Stop-Polling"] = "1"
        return response

    @app.post("/runs/{job_id}/stop")
    def run_stop(request: Request, job_id: str) -> Response:
        c = ctx(request)
        with c.lock:
            if job_id == local.LOCAL_ID:
                if c.session.local_folder:
                    local.stop_local(c.session.local_folder)
            elif c.token:
                error = runs.stop_run(services, job_id, token=c.token)
                if error:
                    c.session.flash = error
        return go(request, f"/runs/{job_id}")

    @app.get("/runs/{job_id}/resume", response_class=HTMLResponse)
    def run_resume_quote(request: Request, job_id: str) -> Response:
        c = ctx(request)
        if job_id == local.LOCAL_ID:
            html = (
                f'<form action="/runs/{job_id}/resume" method="post"><button class="btn" '
                'type="submit">Try again from where it stopped</button></form>'
            )
            return _swaps(("#resume", html))
        if not c.token:
            return _swaps(("#resume", ""))
        quote = runs.review_resume(services, job_id, token=c.token)
        escape = templates.env.filters["e"]
        if not quote.can_resume:
            return _swaps(("#resume", f'<p class="fine">{escape(quote.error)}</p>'))
        cap = f", at most {quote.cap_cost}" if quote.cap_cost else ""
        html = (
            f'<form action="/runs/{job_id}/resume" method="post" class="start"><button '
            'class="btn" type="submit">Try again from where it stopped</button><small>'
            f"On {escape(quote.hardware)}, for up to {escape(quote.cap_time)}{escape(cap)}."
            "</small></form>"
        )
        return _swaps(("#resume", html))

    @app.post("/runs/{job_id}/resume")
    def run_resume(request: Request, job_id: str) -> Response:
        c = ctx(request)
        with c.lock:
            if job_id == local.LOCAL_ID:
                if c.session.local_folder:
                    try:
                        local.start_local(c.session, c.session.local_folder)
                    except local.LocalRunError as exc:
                        c.session.flash = str(exc)
                return go(request, f"/runs/{job_id}")
            if not c.token:
                return go(request, _login_url(f"/runs/{job_id}"))
            new_id, error = runs.resume_run(services, job_id, token=c.token)
        if error:
            return go(request, f"/runs/{job_id}")
        return go(request, f"/runs/{new_id}")

    @app.post("/runs/{job_id}/delete-files")
    def run_delete_files(request: Request, job_id: str) -> Response:
        c = ctx(request)
        if not c.token:
            return _swaps(("#delete-result", "Sign in first."))
        count, error = runs.delete_files(services, job_id, token=c.token)
        text = error or (
            f"Deleted {count} file{'s' if count != 1 else ''}."
            if count
            else "There were no files left to delete."
        )
        return _swaps(("#delete-result", templates.env.filters["e"](text)), ("#delete-confirm", ""))

    @app.get("/runs/{job_id}/download.{fmt}")
    def run_download(request: Request, job_id: str, fmt: str) -> Response:
        c = ctx(request)
        download = next((d for d in views.DOWNLOADS if d.key == fmt), None)
        if download is None or not c.token:
            return RedirectResponse(f"/runs/{job_id}", status_code=303)
        with c.lock:
            try:
                csv_path = predictions_csv(c, job_id)
            except Exception:
                return RedirectResponse(f"/runs/{job_id}", status_code=303)
            state = c.session.last_states.get(job_id)
            name = f"{state.name if state else job_id}-predictions.{fmt}"
            path = csv_path.with_suffix(f".{fmt}")  # the other formats are made from the CSV
            if not path.exists():
                write_table(*read_table(csv_path), path)
        return FileResponse(path, filename=name, media_type=download.media_type)

    # ------------------------------------------------------------------------ predicting

    def prediction_draft(c: Ctx, job_id: str) -> tuple[views.PredictionDraft | None, str | None]:
        draft = c.session.prediction
        if draft is not None and draft.job_id == job_id:
            return draft, None
        try:
            status = services.job_status(job_id, token=c.token)
        except Exception as exc:
            return None, f"Could not read this model from Hugging Face ({exc})."
        if status.kind != "train" or status.stage != "COMPLETED" or not status.adapter_repo:
            return None, "Only a finished model can predict."
        try:
            files = services.pull_run_files(status.adapter_repo, token=c.token)
            config = Config.model_validate(files.config)
        except Exception as exc:
            return None, f"Could not read the model's settings ({exc})."
        draft = views.PredictionDraft(job_id, status.adapter_repo, status.run or job_id, config)
        c.session.prediction = draft
        return draft, None

    def check(c: Ctx, draft: views.PredictionDraft) -> None:
        try:
            counter = services.counter(draft.config)
        except Exception:  # offline: no token counts, so no estimate
            counter = None
        try:
            draft.check = check_input(draft.config, draft.path, counter, draft.rename)
            draft.error = None
        except (ValueError, OSError) as exc:
            draft.check, draft.error = None, f"Could not use {draft.file_name}: {exc}"
        draft.plan = None

    @app.get("/predict/{job_id}", response_class=HTMLResponse)
    def predict(request: Request, job_id: str) -> Response:
        c = ctx(request)
        if not c.token:
            return render(
                request,
                c,
                "models.html",
                need_signin=True,
                rows=[],
                error=None,
                local=None,
                local_label="",
            )
        with c.lock:
            draft, error = prediction_draft(c, job_id)
            if draft is None:
                return render(
                    request,
                    c,
                    "models.html",
                    need_signin=False,
                    rows=[],
                    error=error,
                    local=None,
                    local_label="",
                )
            page = views.predict_page(draft)
        return render(request, c, "predict.html", page=page, max_mb=settings.max_upload_mb)

    @app.post("/predict/{job_id}/upload")
    def predict_upload(request: Request, job_id: str, file: UploadFile = File(...)) -> Response:
        c = ctx(request)
        if not c.token:
            return go(request, _login_url(f"/predict/{job_id}"))
        with c.lock:
            draft, error = prediction_draft(c, job_id)
            if draft is None:
                return go(request, f"/predict/{job_id}")
            path, error = save_upload(c, file, "predict")
            if error:
                draft.error = error
                return go(request, f"/predict/{job_id}")
            if path.suffix.lower() == ".zip":
                try:
                    path = extract_zip(
                        path, path.parent / "unzipped", settings.max_upload_bytes * 5
                    )
                except (UploadError, zipfile.BadZipFile) as exc:
                    draft.error = f"Could not use {file.filename}: {exc}"
                    return go(request, f"/predict/{job_id}")
            draft.file_name, draft.path, draft.rename = (
                Path(file.filename or path.name).name,
                path,
                {},
            )
            check(c, draft)
        return go(request, f"/predict/{job_id}")

    @app.post("/predict/{job_id}/rename")
    async def predict_rename(request: Request, job_id: str) -> Response:
        form = await request.form()
        c = ctx(request)
        with c.lock:
            draft = c.session.prediction
            if draft is None or draft.job_id != job_id or draft.path is None:
                return go(request, f"/predict/{job_id}")
            for source, target in zip(form.getlist("from"), form.getlist("to"), strict=False):
                if source:
                    draft.rename[str(source)] = str(target)
            check(c, draft)
            page = views.predict_page(draft)
        return _swaps(("#predict-body", fragment(request, c, "_predict_body.html", page=page)))

    @app.post("/predict/{job_id}/threshold")
    def predict_threshold(request: Request, job_id: str, percent: str = Form("")) -> Response:
        c = ctx(request)
        with c.lock:
            draft = c.session.prediction
            if draft is not None and draft.job_id == job_id:
                try:
                    draft.threshold = min(max(float(percent), 1.0), 99.0) / 100
                except ValueError:
                    pass
        return Response(status_code=204)

    @app.get("/predict/{job_id}/estimate", response_class=HTMLResponse)
    def predict_estimate(request: Request, job_id: str) -> Response:
        c = ctx(request)
        if not c.token:
            return _swaps(
                (
                    "#predict-estimate",
                    fragment(request, c, "_predict_estimate.html", need_signin=True),
                )
            )
        with c.lock:
            draft = c.session.prediction
            if draft is None or draft.job_id != job_id or draft.check is None:
                return _swaps(("#predict-estimate", ""))
            staged = write_input_csv(
                draft.check,
                c.session.workdir()
                / "predict"
                / "staged"
                / f"{Path(draft.file_name).stem or 'input'}.csv",
            )
            try:
                draft.plan = services.plan_prediction(
                    draft.adapter_repo,
                    staged,
                    config=draft.config,
                    tokens=draft.check.tokens,
                    rows=draft.check.rows,
                    review_below=draft.threshold,
                    token=c.token,
                    keep_input=not c.session.delete_data,
                    prices=services.prices(),
                )
            except Exception as exc:
                html = fragment(
                    request,
                    c,
                    "_predict_estimate.html",
                    need_signin=False,
                    error=f"Can't predict yet: {exc}",
                    job_id=job_id,
                )
                return _swaps(("#predict-estimate", html))
            html = fragment(
                request,
                c,
                "_predict_estimate.html",
                need_signin=False,
                error=None,
                est=views.prediction_estimate(draft.plan),
                job_id=job_id,
            )
        return _swaps(("#predict-estimate", html), animate=("#predict-estimate",))

    @app.post("/predict/{job_id}/start")
    def predict_start(request: Request, job_id: str) -> Response:
        c = ctx(request)
        if not c.token:
            return go(request, _login_url(f"/predict/{job_id}"))
        with c.lock:
            draft = c.session.prediction
            if draft is None or draft.job_id != job_id or draft.plan is None:
                return go(request, f"/predict/{job_id}")
            plan = _replace(draft.plan, review_below=draft.threshold)
            try:
                job = services.submit_prediction(plan, token=c.token)
            except Exception as exc:
                draft.error = f"Predicting did not start: {exc}"
                return go(request, f"/predict/{job_id}")
            draft.plan = None
        return go(request, f"/runs/{job.id}")

    return app


def launch(
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    settings: Settings | None = None,
    services: Services | None = None,
) -> None:
    """Serve the web UI until stopped."""
    import uvicorn

    uvicorn.run(create_app(settings, services), host=host, port=port or 7860, log_level="info")
