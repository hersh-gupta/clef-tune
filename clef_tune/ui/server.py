"""The web UI: a small FastAPI app that serves clef-tune's pages. Pages are Jinja templates
filled from ui/views.py and ui/runs.py; forms change the session through ui/handlers.py.
Background form posts get back <template data-swap> blocks (see static/clef-tune.js), so only
the parts of a page that changed are replaced.

Sessions live in memory, keyed by a signed cookie. On a Hugging Face Space, people sign in
with Hugging Face (attach_huggingface_oauth) and every Hub call uses their own token; the
server's token is never used for a visitor. Elsewhere the machine's own login is used, and the
app answers only to this computer's own names (a page elsewhere can't rebind a name to it).
Either way, a state-changing request another site's page sent is refused."""

from __future__ import annotations

import dataclasses
import hashlib
import os
import re
import secrets
import shutil
import sys
import threading
import time
import uuid
import zipfile
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import escape
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers
from starlette.datastructures import UploadFile as FormFile
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from clef_tune.config import Config
from clef_tune.predictions import (
    DEFAULT_REVIEW_BELOW,
    check_input,
    read_table,
    summarize,
    write_input_csv,
    write_table,
)
from clef_tune.ui import handlers, local, runs, views
from clef_tune.ui import wording as w
from clef_tune.ui.services import Services, public_services
from clef_tune.ui.session import STALE_HOURS, Session, UploadError, extract_zip, sweep_stale
from clef_tune.ui.settings import Settings, SignInRequired, user_token

HERE = Path(__file__).parent
SESSION_KEY = "clef_tune_sid"
FROM_SCRIPT = "clef-tune"  # the X-Requested-With header clef-tune.js sends
CHUNK = 1 << 20
FORM_ROOM = 64 * 1024  # an upload's form framing, on top of the file itself
LOSS_WIDTH, LOSS_HEIGHT = 600, 90
LATEST_SHOWN = 2
JOB_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")  # a Hugging Face job id
LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "[::1]")
ANY_ADDRESS = ("0.0.0.0", "::", "")  # listening on every network the computer is on
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
SAFE_FETCH_SITES = ("same-origin", "none")  # "none": typed in the address bar, or a bookmark
DEFAULT_PORTS = {"http": 80, "https": 443}
LOCAL_RETURNS = ("/train", "/models")  # where pasting a token on this computer may go back to
EXPIRED = "This page has expired, so that change wasn't saved. Start again from your data."
NETWORK_WARNING = (
    "Warning: clef-tune is serving your whole network, so anyone on your network can open it "
    "and act with this computer's Hugging Face login."
)


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


class AccountNames:
    """Which account each token belongs to, remembered for the process. Only answers are kept,
    so a lookup that failed (offline, or a moment's outage) is tried again next time. At most
    `limit` are kept, keyed by each token's hash, so pasted tokens don't pile up."""

    def __init__(self, limit: int = 256) -> None:
        self.limit = limit
        self.names: OrderedDict[str, str] = OrderedDict()
        self.guard = threading.Lock()

    def __len__(self) -> int:
        return len(self.names)

    def lookup(self, token: str, whoami: Callable[[str], str]) -> str | None:
        key = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self.guard:
            if key in self.names:
                self.names.move_to_end(key)
                return self.names[key]
        try:
            name = whoami(token)
        except Exception:  # offline, or a token without read access
            return None
        if not name:
            return None
        with self.guard:
            self.names[key] = name
            while len(self.names) > self.limit:
                self.names.popitem(last=False)
        return name


def _host_and_port(value: str, default: int | None) -> tuple[str, int | None]:
    host, colon, port = value.rpartition(":")
    if colon and port.isdigit():
        return host.strip("[]").lower(), int(port)
    return value.strip("[]").lower(), default


def same_origin(origin: str, host: str) -> bool:
    """Whether an Origin header names the host and port the request was sent to."""
    parts = urlsplit(origin)
    default = DEFAULT_PORTS.get(parts.scheme)
    if not parts.hostname or default is None:  # "null", or not a web page
        return False
    try:
        port = parts.port or default
    except ValueError:
        return False
    return (parts.hostname.lower(), port) == _host_and_port(host, default)


class SameOriginPosts:
    """Refuses a state-changing request that another site's page sent (cross-site request
    forgery: on a Space the session cookie must be SameSite=None to work in Hugging Face's
    frame, so any site's form would carry it). Browsers say where a request came from in
    Sec-Fetch-Site and Origin; a request with neither is not from a page, and passes."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] in UNSAFE_METHODS:
            headers = Headers(scope=scope)
            site, origin = headers.get("sec-fetch-site"), headers.get("origin")
            if (site is not None and site not in SAFE_FETCH_SITES) or (
                origin is not None and not same_origin(origin, headers.get("host", ""))
            ):
                refused = PlainTextResponse(
                    "This request came from another site, so it was refused.", status_code=403
                )
                await refused(scope, receive, send)
                return
        await self.app(scope, receive, send)


def bound_hosts(host: str) -> list[str]:
    """The names the app answers to on this computer: its loopback names and the address it was
    told to listen on. Listening on every address (0.0.0.0) is choosing the whole network, so
    any name is taken."""
    if host in ANY_ADDRESS:
        return ["*"]
    name = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return [*LOOPBACK_HOSTS, name.lower()]


def checked_job_id(job_id: str) -> str:
    """A run id from the address: a Hugging Face job id, or this computer's run. Anything else
    (a path, markup) is not a run, so it never reaches a file path, a Hub call or a page."""
    if job_id != local.LOCAL_ID and not JOB_ID.fullmatch(job_id):
        raise HTTPException(status_code=404)
    return job_id


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
        f'<template data-swap="{escape(selector)}"{" data-animate" if selector in animate else ""}>'
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


def _upload_name(filename: str | None) -> str:
    """A safe name for an uploaded file: its base name without characters a file system
    refuses, or "upload" when nothing is left (a name such as ".." or an empty one)."""
    base = re.split(r"[/\\]", filename or "")[-1]
    return re.sub(r'[<>:"|?*\x00-\x1f]', "-", base).strip(" .") or "upload"


def _too_big(name: str, max_mb: int) -> str:
    return f"Could not use {name}: it is over the {max_mb} MB limit"


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
    allowed_hosts: Sequence[str] | None = None,
) -> FastAPI:
    """The app. On a Space every visitor signs in; elsewhere the app answers only to
    `allowed_hosts` (by default this computer's loopback names; see bound_hosts)."""
    settings = settings or Settings.from_env()
    services = services or (public_services() if settings.space else Services())
    app = FastAPI(title="clef-tune", docs_url=None, redoc_url=None, openapi_url=None)
    if settings.space:
        from huggingface_hub import attach_huggingface_oauth

        attach_huggingface_oauth(app)  # also adds the (SameSite=None) session middleware
        app.add_middleware(SameOriginPosts)
    else:
        app.add_middleware(
            SessionMiddleware,
            secret_key=secret or os.environ.get("CLEF_TUNE_SECRET") or secrets.token_hex(32),
            session_cookie="clef_tune",
            same_site="lax",
            https_only=False,
        )
        app.add_middleware(SameOriginPosts)
        app.add_middleware(
            TrustedHostMiddleware, allowed_hosts=list(allowed_hosts or LOOPBACK_HOSTS)
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
    names = AccountNames()
    # Every route with a run's id in its address checks the id first (checked_job_id).
    jobs = APIRouter(dependencies=[Depends(checked_job_id)])
    sweep_stale()

    # ------------------------------------------------------------------------ helpers

    def signed_in(request: Request) -> Any:
        """This visitor's Hugging Face sign-in, or None. One that has expired counts as signed
        out and is forgotten, so they're asked to sign in again instead of shown errors."""
        from huggingface_hub import parse_huggingface_oauth

        try:
            info = parse_huggingface_oauth(request)
        except (TypeError, ValueError, OverflowError, OSError):  # a session it can't read
            info = None
        if info is not None and info.access_token_expires_at > datetime.now():
            return info
        request.session.pop("oauth_info", None)
        return None

    def ctx(request: Request) -> Ctx:
        sid = request.session.get(SESSION_KEY)
        if not sid:
            sid = uuid.uuid4().hex
            request.session[SESSION_KEY] = sid
        session, lock = store.get(sid)
        oauth, user = None, None
        if settings.space:
            info = signed_in(request)
            if info is not None:
                oauth = info.access_token
                user = getattr(info.user_info, "preferred_username", None)
        try:
            token = user_token(settings, oauth, session.pasted_token)
        except SignInRequired:
            token = None
        if token and (user is None or token != oauth):  # a pasted token shows its own account
            user = names.lookup(token, services.whoami)
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

    def sign_in(target: str) -> str:
        """Where a signed-out person goes to sign in: Hugging Face's sign-in on a Space. On this
        computer there is none, so back to the page, which says how to log in."""
        return _login_url(target) if settings.space else target

    def lost(request: Request, c: Ctx) -> Response | None:
        """For an in-page edit whose setup is gone (a restart, a day idle, or signing out in
        another tab): back to the first step, saying why. None while the setup is there."""
        if c.session.draft is not None:
            return None
        c.session.flash = EXPIRED
        return go(request, "/data")

    def refresh(c: Ctx) -> None:
        handlers.refresh(c.session, services)

    def too_large(request: Request) -> bool:
        """Whether an upload says it is over the cap, so it can be refused before it is read
        (with room for the form's own framing; save_upload checks the file itself)."""
        try:
            declared = int(request.headers.get("content-length", ""))
        except ValueError:
            return False
        return declared > settings.max_upload_bytes + FORM_ROOM

    def save_upload(c: Ctx, upload: FormFile, folder: str) -> tuple[Path | None, str | None]:
        """Stream an upload into the session's folder under its own name, within the cap."""
        name = _upload_name(upload.filename)
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
                    return None, _too_big(name, settings.max_upload_mb)
                out.write(chunk)
        return target, None

    def step_page(request: Request, c: Ctx, step: str) -> Response | None:
        """Send a person back to the furthest step they can open, if this one is not open."""
        index = [key for key, _, _ in views.STEPS].index(step)
        furthest = views.furthest_step(c.session)
        if index > furthest:
            return RedirectResponse(views.STEPS[furthest][2], status_code=303)
        return None

    def signin_page(request: Request, c: Ctx, error: str | None = None) -> Response:
        """Your models, asking a signed-out person to sign in (or, with an error, saying why a
        run can't be shown)."""
        return render(
            request,
            c,
            "models.html",
            need_signin=error is None,
            rows=[],
            error=error,
            local=None,
            local_label="",
        )

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
        rows, error = runs.list_models(services, token=c.token) if c.token else ([], None)
        return render(
            request,
            c,
            "models.html",
            need_signin=not c.token,
            rows=rows,
            error=error,
            local=local_state,
            local_label=labels.get(local_state.status) if local_state else "",
        )

    @jobs.get("/models/{job_id}/line", response_class=HTMLResponse)
    def model_line(request: Request, job_id: str) -> Response:
        c = ctx(request)
        if not c.token:
            return HTMLResponse("")
        with c.lock:
            line = runs.model_line(c.session, services, job_id, token=c.token)
        return _swaps((f"#line-{job_id}", str(escape(line))))

    # ------------------------------------------------------------------------ step 1: data

    @app.get("/data", response_class=HTMLResponse)
    def data(request: Request) -> Response:
        c = ctx(request)
        with c.lock:
            error = pop_flash(c.session)
            page = views.data_page(c.session, settings, error)
            steps = views.step_bar(c.session, "data")
        return render(request, c, "data.html", page=page, steps=steps, roles=handlers.ROLE_CHOICES)

    def use_data_upload(request: Request, upload: Any) -> Response:
        c = ctx(request)
        with c.lock:
            if not isinstance(upload, FormFile):
                c.session.flash = "Choose a file to upload."
                return go(request, "/data")
            path, error = save_upload(c, upload, "incoming")
            if path is not None:
                error = handlers.upload(c.session, path, settings)
                shutil.rmtree(path.parent, ignore_errors=True)
                if error is None:
                    refresh(c)
            if error:
                c.session.flash = error
        return go(request, "/data")

    def refuse_data_upload(request: Request) -> Response:
        c = ctx(request)
        with c.lock:
            c.session.flash = _too_big("that file", settings.max_upload_mb)
        return go(request, "/data")

    @app.post("/data/upload")
    async def data_upload(request: Request) -> Response:
        """Async only so an oversized upload is refused before its body is read; the work
        runs in the thread pool, off the event loop."""
        if too_large(request):
            return await run_in_threadpool(refuse_data_upload, request)
        async with request.form(max_files=1) as form:
            return await run_in_threadpool(use_data_upload, request, form.get("file"))

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
            if (expired := lost(request, c)) is not None:
                return expired
            handlers.set_noun(c.session, noun)
            page = views.data_page(c.session, settings)
        mark = "Guessed from the file name" if page.noun_guessed else ""
        return _swaps(("#noun-mark", mark))

    @app.post("/data/role")
    def data_role(request: Request, column: str = Form(...), role: str = Form(...)) -> Response:
        c = ctx(request)
        with c.lock:
            if (expired := lost(request, c)) is not None:
                return expired
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
        flags = "".join(f'<p class="flag">{escape(f)}</p>' for f in page.selected.flags)
        number = [item.id for item in page.items].index(page.selected.id) + 1
        # Numbered per question, so an answer that lands after a switch changes nothing.
        return _swaps((f"#flags-{number}", flags))

    # A question's id is its column's name, which may hold a "/", so it travels in the form
    # (qid), never in the address.

    @app.post("/questions/ask")
    def question_ask(request: Request, qid: str = Form(...), ask: str = Form("")) -> Response:
        c = ctx(request)
        with c.lock:
            if (expired := lost(request, c)) is not None:
                return expired
            handlers.update_question(c.session, qid, instructions=ask)
            refresh(c)
            return editor_swap(request, c, qid, whole=False)

    @app.post("/questions/type")
    def question_type(request: Request, qid: str = Form(...), kind: str = Form(...)) -> Response:
        c = ctx(request)
        with c.lock:
            if (expired := lost(request, c)) is not None:
                return expired
            try:
                handlers.update_question(c.session, qid, type=kind)
            except ValueError:
                pass
            refresh(c)
            return editor_swap(request, c, qid, whole=True)

    @app.post("/questions/describe")
    def question_describe(
        request: Request,
        qid: str = Form(...),
        option: str = Form(...),
        description: str = Form(""),
    ) -> Response:
        c = ctx(request)
        with c.lock:
            if (expired := lost(request, c)) is not None:
                return expired
            handlers.describe_option(c.session, qid, option, description)
            refresh(c)
            return editor_swap(request, c, qid, whole=False)

    @app.post("/questions/move")
    def question_move(
        request: Request, qid: str = Form(...), option: str = Form(...), offset: int = Form(...)
    ) -> Response:
        c = ctx(request)
        with c.lock:
            if (expired := lost(request, c)) is not None:
                return expired
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
            if (expired := lost(request, c)) is not None:
                return expired
            error = handlers.set_run_name(c.session, name)
            if error:
                return _swaps(("#name-note", str(escape(error))))
            page = views.settings_page(c.session, services, c.user)
        note = f"Saved to your Hugging Face account as <b>{escape(page.repo)}</b>"
        return _swaps(("#name-note", note))

    @app.post("/settings/model")
    def settings_model(request: Request, model: str = Form(...)) -> Response:
        c = ctx(request)
        with c.lock:
            if (expired := lost(request, c)) is not None:
                return expired
            if model in views.MODELS:
                handlers.set_model(c.session, model)
                c.session.flavor = None
                refresh(c)
            return _swaps(("#settings-body", settings_view(request, c)))

    @app.post("/settings/preset")
    def settings_preset(request: Request, preset: str = Form(...)) -> Response:
        c = ctx(request)
        with c.lock:
            if (expired := lost(request, c)) is not None:
                return expired
            if preset in views.PRESETS:
                handlers.set_preset(c.session, preset)
                refresh(c)
            return _swaps(("#settings-body", settings_view(request, c)))

    @app.post("/settings/field")
    def settings_field(
        request: Request, path: str = Form(""), value: list[str] = Form([])
    ) -> Response:
        c = ctx(request)
        chosen: Any = value[-1] if value else ""  # a checkbox sends "" and then "1" when ticked
        spec = next((s for s in handlers.ADVANCED_FIELDS if s.path == path), None)
        if spec is not None and spec.kind == "bool":
            chosen = chosen == "1"
        with c.lock:
            if (expired := lost(request, c)) is not None:
                return expired
            error = handlers.set_field(c.session, path, chosen)
            refresh(c)
            field_error = (path, views.field_error_text(path, error)) if error else None
            return _swaps(("#settings-body", settings_view(request, c, field_error)))

    @app.post("/settings/holdout")
    def settings_holdout(request: Request, percent: str = Form("")) -> Response:
        c = ctx(request)
        with c.lock:
            if (expired := lost(request, c)) is not None:
                return expired
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
            if (expired := lost(request, c)) is not None:
                return expired
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
        common = {
            "local_training": settings.local_training,
            "delete_data": session.delete_data,
            "pasted": bool(session.pasted_token),
        }
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
    def train_delete_data(request: Request, delete: list[str] = Form([])) -> Response:
        c = ctx(request)
        with c.lock:
            c.session.delete_data = "1" in delete
            return _swaps(("#review", review_fragment(request, c)))

    @app.post("/train/token")
    def train_token(
        request: Request, token: str = Form(""), back: str = Form("/train")
    ) -> Response:
        """A token pasted for this browser session: on a Space when sign-in lacks a permission,
        and on this computer when it has no Hugging Face login."""
        c = ctx(request)
        with c.lock:
            c.session.pasted_token = token.strip() or None
        return go(request, back if back in LOCAL_RETURNS else "/train")

    @app.post("/train/switch")
    def train_switch(request: Request) -> Response:
        """Use another account: forget a pasted token, and on a Space the sign-in too."""
        c = ctx(request)
        with c.lock:
            c.session.pasted_token = None
        target = "/oauth/huggingface/logout?_target_url=/train" if settings.space else "/train"
        return go(request, target)

    @app.post("/train/start")
    def train_start(request: Request) -> Response:
        c = ctx(request)
        if not c.token:
            return go(request, sign_in("/train"))
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

    def predictions_csv(c: Ctx, job_id: str, status: Any = None) -> Path:
        """A prediction job's spreadsheet, downloaded once into this session's folder, with the
        person's own column names put back (the job read the model's)."""
        path = c.session.workdir() / "predictions" / f"{job_id}.csv"
        if path.exists():
            return path
        status = status or services.job_status(job_id, token=c.token)
        services.download_predictions(job_id, path, token=c.token)
        if status.output_names:
            header, rows = read_table(path)
            write_table([status.output_names.get(name, name) for name in header], rows, path)
        return path

    def forget_predictions(c: Ctx, job_id: str) -> None:
        """Delete this session's copies of a job's spreadsheet, in every format."""
        folder = c.session.workdir() / "predictions"
        for path in folder.glob(f"{job_id}.*"):  # an id is only letters, digits, - and _
            path.unlink(missing_ok=True)

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

    def finished_page(
        request: Request, c: Ctx, job_id: str, status: Any, state: runs.RunState
    ) -> tuple[Response | None, str | None]:
        """A completed run's results page, or why it can't be read yet (None, notice). The
        files can lag the job's end (the bucket uploads on close), so the run page waits."""
        if state.kind == "train":
            view, problem = runs.job_results(c.session, services, status, token=c.token)
            if view is None:
                return None, problem
            return render(request, c, "results.html", view=view, job_id=job_id), None
        try:
            config = prediction_config(c, status.adapter_repo)
            header, rows = read_table(predictions_csv(c, job_id, status))
            result = views.prediction_result(
                summarize(header, rows, config),
                config,
                threshold=status.review_below or DEFAULT_REVIEW_BELOW,
            )
        except Exception as exc:  # not downloadable yet, or the model was deleted
            return None, f"The spreadsheet isn't readable yet ({exc}); trying again."
        page = render(
            request,
            c,
            "prediction_results.html",
            state=state,
            result=result,
            took=w.duration(state.hours or 0),
            downloads=views.DOWNLOADS,
        )
        return page, None

    @jobs.get("/runs/{job_id}", response_class=HTMLResponse)
    def run_page(request: Request, job_id: str) -> Response:
        c = ctx(request)
        common = {"poll_seconds": runs.POLL_SECONDS, "notice": None, "retry": False}
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
            return signin_page(request, c)
        with c.lock:
            status, state, notice = job_view(c, job_id)
            if state is None:
                return signin_page(request, c, notice)
            if status is not None and state.status == "done":
                page, notice = finished_page(request, c, job_id, status, state)
                if page is not None:
                    return page
                common["retry"] = notice is not None
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

    @jobs.get("/runs/{job_id}/status", response_class=HTMLResponse)
    def run_status(request: Request, job_id: str) -> Response:
        c = ctx(request)
        waiting = False  # finished, but its results can't be read yet: keep polling
        with c.lock:
            if job_id == local.LOCAL_ID:
                state, _ = local.local_state(
                    c.session, model=_model_name(c.session), noun=_noun(c.session)
                )
                notice = None
            elif c.token:
                status, state, notice = job_view(c, job_id)
                if status is not None and state.status == "done":
                    page, problem = finished_page(request, c, job_id, status, state)
                    if page is None and problem is not None:
                        notice, waiting = problem, True
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
            loss_points=_loss_points(state.loss),
        )
        response = _swaps(("#run-status", html))
        if state.status != "running" and not waiting:
            response.headers["X-Run-Ended"] = state.status
            response.headers["X-Refresh"] = "1"
            response.headers["X-Stop-Polling"] = "1"
        return response

    @jobs.post("/runs/{job_id}/stop")
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

    @jobs.get("/runs/{job_id}/resume", response_class=HTMLResponse)
    def run_resume_quote(request: Request, job_id: str) -> Response:
        c = ctx(request)
        if job_id != local.LOCAL_ID and not c.token:
            return _swaps(("#resume", ""))
        quote = (
            None
            if job_id == local.LOCAL_ID
            else runs.review_resume(services, job_id, token=c.token)
        )
        html = fragment(request, c, "_resume.html", job_id=job_id, quote=quote)
        return _swaps(("#resume", html))

    @jobs.post("/runs/{job_id}/resume")
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
                return go(request, sign_in(f"/runs/{job_id}"))
            new_id, error = runs.resume_run(services, job_id, token=c.token)
        if error:
            return go(request, f"/runs/{job_id}")
        return go(request, f"/runs/{new_id}")

    @jobs.post("/runs/{job_id}/delete-files")
    def run_delete_files(request: Request, job_id: str) -> Response:
        c = ctx(request)
        if not c.token:
            return _swaps(("#delete-result", "Sign in first."))
        count, error = runs.delete_files(services, job_id, token=c.token)
        if error is None:
            with c.lock:
                forget_predictions(c, job_id)
        text = error or (
            f"Deleted {count} file{'s' if count != 1 else ''}."
            if count
            else "There were no files left to delete."
        )
        return _swaps(("#delete-result", str(escape(text))), ("#delete-confirm", ""))

    @jobs.get("/runs/{job_id}/download.{fmt}")
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
        draft = views.PredictionDraft(
            job_id,
            status.adapter_repo,
            status.run or job_id,
            config,
            calibrated=runs.was_calibrated(files.metrics or {}),
        )
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

    @jobs.get("/predict/{job_id}", response_class=HTMLResponse)
    def predict(request: Request, job_id: str) -> Response:
        c = ctx(request)
        if not c.token:
            return signin_page(request, c)
        with c.lock:
            draft, error = prediction_draft(c, job_id)
            if draft is None:
                return signin_page(request, c, error)
            page = views.predict_page(draft)
        return render(request, c, "predict.html", page=page, max_mb=settings.max_upload_mb)

    def use_prediction_upload(request: Request, job_id: str, upload: Any) -> Response:
        c = ctx(request)
        if not c.token:
            return go(request, sign_in(f"/predict/{job_id}"))
        with c.lock:
            draft, error = prediction_draft(c, job_id)
            if draft is None:
                return go(request, f"/predict/{job_id}")
            if not isinstance(upload, FormFile):
                draft.error = "Choose a file to upload."
                return go(request, f"/predict/{job_id}")
            path, error = save_upload(c, upload, "predict")
            if error:
                draft.error = error
                return go(request, f"/predict/{job_id}")
            if path.suffix.lower() == ".zip":
                try:
                    path = extract_zip(
                        path, path.parent / "unzipped", settings.max_upload_bytes * 5
                    )
                except (UploadError, zipfile.BadZipFile) as exc:
                    draft.error = f"Could not use {_upload_name(upload.filename)}: {exc}"
                    return go(request, f"/predict/{job_id}")
            draft.file_name, draft.path, draft.rename = _upload_name(upload.filename), path, {}
            check(c, draft)
        return go(request, f"/predict/{job_id}")

    def refuse_prediction_upload(request: Request, job_id: str) -> Response:
        c = ctx(request)
        if not c.token:
            return go(request, sign_in(f"/predict/{job_id}"))
        with c.lock:
            draft, _ = prediction_draft(c, job_id)
            if draft is not None:
                draft.error = _too_big("that file", settings.max_upload_mb)
        return go(request, f"/predict/{job_id}")

    @jobs.post("/predict/{job_id}/upload")
    async def predict_upload(request: Request, job_id: str) -> Response:
        """Async only so an oversized upload is refused before its body is read; the work
        runs in the thread pool, off the event loop."""
        if too_large(request):
            return await run_in_threadpool(refuse_prediction_upload, request, job_id)
        async with request.form(max_files=1) as form:
            upload = form.get("file")
            return await run_in_threadpool(use_prediction_upload, request, job_id, upload)

    @jobs.post("/predict/{job_id}/rename")
    def predict_rename(
        request: Request,
        job_id: str,
        sources: list[str] = Form([], alias="from"),
        targets: list[str] = Form([], alias="to"),
    ) -> Response:
        c = ctx(request)
        with c.lock:
            draft = c.session.prediction
            if draft is None or draft.job_id != job_id or draft.path is None:
                return go(request, f"/predict/{job_id}")
            for source, target in zip(sources, targets, strict=False):
                if source:
                    draft.rename[source] = target
            check(c, draft)
            page = views.predict_page(draft)
        return _swaps(("#predict-body", fragment(request, c, "_predict_body.html", page=page)))

    @jobs.post("/predict/{job_id}/threshold")
    def predict_threshold(request: Request, job_id: str, percent: str = Form("")) -> Response:
        c = ctx(request)
        with c.lock:
            draft = c.session.prediction
            if draft is not None and draft.job_id == job_id:
                handlers.set_threshold(draft, percent)
        return Response(status_code=204)

    @jobs.get("/predict/{job_id}/estimate", response_class=HTMLResponse)
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
                    output_names=draft.check.output_names(),
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

    @jobs.post("/predict/{job_id}/start")
    def predict_start(request: Request, job_id: str) -> Response:
        c = ctx(request)
        if not c.token:
            return go(request, sign_in(f"/predict/{job_id}"))
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

    app.include_router(jobs)
    return app


def launch(
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    settings: Settings | None = None,
    services: Services | None = None,
) -> None:
    """Serve the web UI until stopped. It answers only to this computer's own names and the
    address it listens on; listening on every address serves the whole network, with a
    warning, since anyone there can then act with this computer's Hugging Face login."""
    import uvicorn

    hosts = bound_hosts(host)
    if "*" in hosts:
        print(NETWORK_WARNING, file=sys.stderr)
    app = create_app(settings, services, allowed_hosts=hosts)
    uvicorn.run(app, host=host, port=port or 7860, log_level="info")
