# clef-tune Plan 4: Web UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A polished Gradio web UI that takes a person from an uploaded CSV/JSONL to a validated schema, a cost estimate, a launched HF Job, live monitoring and a results view, without editing YAML. It runs locally (`clef-tune ui`) and as a public, multi-user Hugging Face Space. The phase ends only when the user signs off on screenshots.

**Architecture:** The UI is split into layers.
- **Event handlers:** plain, torch-free and Gradio-free functions in `clef_tune/ui/`. They read and change a per-browser `Session` and return small view dataclasses of HTML strings and lists.
- **Layout:** only `ui/app.py` and `ui/theme.py` import Gradio. `app.py` maps the view dataclasses onto components.
- **Charts:** hand-written SVG styled by `style.css` classes, so one style covers light and dark themes.
- **Hub and Jobs calls:** everything that talks to the Hub or Jobs goes through an injectable `Services` object. Tests, the browser test and the screenshot script pass fakes that replay a recorded run.
- **Space mode:** set by `SPACE_ID`. It hides local training, gives each session its own temp folder, caps uploads, and never uses a server-side token.

**Tech Stack:** Gradio 6.30 (`gradio[oauth]`), stdlib `html`/`zipfile`/`subprocess`, Playwright 1.64 (dev) for the browser test and screenshots, and everything from Plans 1–3.

**Spec:** `docs/superpowers/specs/2026-10-26-clef-finetuning-harness-design.md` (revision 2), especially §4.7, §8.4 and §9. Also read `docs/superpowers/plans/2026-10-26-clef-tune-roadmap.md`: its "Verified constraints and spec amendments" override the spec, and item 29 onward were checked on 2026-10-28 while writing this plan.

**Branch:** work on `plan-4-ui`, created from `main`.

## Global Constraints

- **Plans 1–3 constraints still hold:**
  - Python `>=3.11` (dev 3.12), ruff line length 100.
  - Run `uv run ruff format . && uv run ruff check --fix .` before every commit.
  - Commit messages carry no Claude attribution: no `Co-Authored-By` or session trailer.
  - No network in unit tests.
- **No torch:** the UI modules and `clef_tune.automap` must not import torch, transformers or peft. The UI modules are every `clef_tune.ui.*` module plus `clef_tune.ui.app`. Each task adds its modules to `tests/test_boundaries.py`.
- **Gradio only in the layout:** only `clef_tune/ui/app.py` and `clef_tune/ui/theme.py` import gradio. A boundary test checks the rest.
- **Escape everything:** every string from data, the Hub or a log is passed through `html.escape` before it goes into HTML or SVG. This covers cell values, column names, option ids, issue messages, run names and log lines. The Space is public.
- **Tokens:**
  - Resolve a token per call with `settings.user_token()` (Task 3).
  - On a Space, never use `huggingface_hub.get_token()`. It would return the Space owner's `HF_TOKEN`. Every Plan 3 function falls back to `get_token()` when given `None`, so a handler never passes `None` there. It raises `SignInRequired` first.
  - Tokens are never stored in `Session`, printed or logged. The pasted token stays in its password textbox and arrives as an event input.
- **Gradio 6 placement:** `theme`, `css_paths`, `js` and `head` are `Blocks.launch()` arguments, not `gr.Blocks()` arguments.
- **Dependency:** the `ui` extra becomes `gradio[oauth]>=6.30,<7`, because the OAuth routes import itsdangerous and authlib. Add nothing else; charts are hand-written SVG.
- **Spending:** no job starts or resumes without a confirmation that shows its maximum cost (timeout × price).
- **Visual system:**
  - **Colour:** one neutral palette (zinc) and one accent (indigo), defined once as `--ct-*` CSS variables for `:root` and `.dark`.
  - **Type:** Inter, with JetBrains Mono for code, both from Google Fonts.
  - **Layout:** spacing on a 4 px scale; rounded cards with 1 px borders.
  - **Width:** usable down to 1024 px and stacked below 900 px.
- **Copy:** sentence case and plain words. Every error says what to do next.

## Review Focus

1. **A Space visitor who has not signed in**, while the Space has an `HF_TOKEN` secret, must not reach the Hub as the owner. Every Hub action must ask them to sign in. Pinned by `test_a_space_never_uses_the_server_token` (Task 3) and `test_space_actions_need_sign_in_and_call_nothing` (Task 6).
2. **Hostile text in data, logs or Hub replies** (`<img onerror=…>`, `<script>`) must show as text. Pinned by:
   - `test_every_fragment_escapes_user_text` (Task 2);
   - `test_prompt_preview_escapes_the_row` (Task 4);
   - `test_log_tail_is_escaped` (Task 6).
3. **Hostile or huge uploads** must be refused with a clear message, and nothing may be written outside the session folder. This covers zips with `../` or absolute member paths, zips that expand past the cap, and files over the cap. Pinned by `test_hostile_zips_are_refused` and `test_an_upload_over_the_cap_is_refused` (Task 3).
4. **A launch confirmed after the settings changed since the review, or confirmed twice,** must not start the old plan or a second job. Pinned by `test_a_stale_review_is_not_launched` and `test_confirming_twice_launches_once` (Task 6).
5. **A Hub error while a run is being watched** (network blip, 5xx, expired token) must keep the last view and keep polling, with a notice, not a traceback. Pinned by `test_a_failed_poll_keeps_the_last_view_and_polling` (Task 6).

---

## Verified facts for this plan (2026-10-28)

These go into the roadmap as items 29–35 in the plan-doc commit.

29. **Gradio 6.30 layout API:**
    - `Blocks.launch()` takes `theme`, `css_paths`, `js`, `head` and `max_file_size`. `gr.Blocks()` takes only `title`, `fill_height`, `fill_width` and `delete_cache`.
    - `gr.render(inputs, triggers)` exists, and components take `key` to keep their value across re-renders.
    - `gr.Timer(value, active=…)`, `gr.DownloadButton`, `gr.Sidebar` and `gr.Walkthrough` exist.
    - Dark mode is the `dark` class on `document.body`. `?__theme=dark` sets it.
    - Importing gradio does not import torch; it does import pandas.
30. **Gradio OAuth outside a Space is a mock.**
    - With any `gr.LoginButton` in the app, Gradio installs fake `/login` routes. At launch it calls `whoami()` with the local token and fails if none.
    - Handlers then receive the token `"mock-oauth-token-for-local-dev"`, which cannot call the Hub.
    - The routes need `itsdangerous`/`authlib` (`gradio[oauth]`).
    - *Amends spec §9.1:* the sign-in button appears only on a Space. Locally the UI uses the person's own `hf auth login` token, or a pasted one.
31. **Gradio session state:**
    - `gr.State(delete_callback=…)` runs about an hour after the browser session closes.
    - When more than `state_session_capacity` sessions push one out (LRU), the callback does not run.
    - So the UI also sweeps session folders older than 24 h at startup.
32. **Spaces OAuth scopes** (from huggingface.co/docs/hub/spaces-oauth):
    - `openid profile` are always included.
    - The optional scopes include `read-repos`, `write-repos`, `manage-repos` (create/delete repos) and `jobs` (run jobs).
    - There is no storage-bucket scope. Whether `manage-repos`/`jobs` cover the jobs-artifacts bucket is unverified.
    - So the HF Jobs tab always offers a pasted token. A 401/403 from the bucket upload already says the token lacks a permission (Plan 3 `SCOPE_ADVICE`).
33. **The wheel ships only `clef_tune/`.** The UI's "load example" needs the example in the wheel, so hatch `force-include` copies `examples/support_tickets.{csv,yaml}` to `clef_tune/_examples/`.
34. **Playwright 1.64.1** is installed with chromium-1208 in `%LOCALAPPDATA%\ms-playwright`. If the browser revision is stale, run `uv run playwright install chromium`.
35. **Spec amendments made by this plan:**
    - **No root `app.py`:** a Space built from the repo root would collide with the GitHub README front matter. The entry point is only `deploy/space/app.py`.
    - **Charts are SVG.**
    - **Auto-mapping also proposes an id column** (named `id` or ending in `_id`, all values unique and present). It ignores empty columns.
    - **The UI's default model is Clef-Flash.**
    - **`train` events carry `total_steps`.**

---

## File Structure

```
clef_tune/automap.py              # Task 1: §4.7 column proposals (no torch)
clef_tune/card.py                 # Task 2: public METRIC_ROWS / WEIGHTING_NOTE, top_confusion()
clef_tune/ui/__init__.py          # Task 2 (empty)
clef_tune/ui/charts.py            # Task 2: SVG bar chart, histogram, line chart, heatmap
clef_tune/ui/fragments.py         # Task 2: escaped HTML fragments (tiles, issues, prompt preview, tables…)
clef_tune/ui/settings.py          # Task 3: Settings (Space mode, upload cap, local training), user_token()
clef_tune/ui/session.py           # Task 3: Session, Draft, uploads, zips, example, sweep
clef_tune/ui/services.py          # Task 4 (counter, prices), Task 6 (Hub and Jobs calls)
clef_tune/ui/handlers.py          # Task 4: Data, Schema, Validate, Configure handlers, step states, bundle
clef_tune/validation.py           # Task 4: validate(..., counter_error=)
clef_tune/train.py                # Task 5: total_steps in train events
clef_tune/launch/hf_jobs.py       # Task 5: JobStatus.adapter_repo, ResumeQuote, quote_resume()
clef_tune/hub.py                  # Task 5: pull_metrics()
clef_tune/cli.py                  # Task 5: jobs resume confirms; Task 8: `ui` command
scripts/make_recorded_run.py      # Task 5: writes tests/ui/fixtures/recorded_run/
clef_tune/ui/runs.py              # Task 6: launch review/confirm, runs list, monitor, cancel, resume, results
clef_tune/ui/local.py             # Task 7: local training subprocesses
clef_tune/ui/theme.py             # Task 8: Gradio theme
clef_tune/ui/style.css            # Task 8: tokens, layout, stepper, cards, charts, light/dark
clef_tune/ui/app.py               # Task 8: build_app(), launch()
deploy/space/{app.py,README.md}   # Task 9
scripts/prepare_space.py          # Task 9: copies deploy/space and pins requirements.txt to a commit
scripts/screenshots.py            # Task 10: Playwright captures
pyproject.toml, uv.lock, .github/workflows/ci.yml, .gitignore, README.md
tests/test_automap.py             # Task 1
tests/ui/__init__.py, tests/ui/fakes.py
tests/ui/test_render.py           # Task 2
tests/ui/test_session.py          # Task 3
tests/ui/test_handlers.py         # Task 4
tests/ui/test_recorded_run.py     # Task 5
tests/ui/test_runs.py             # Task 6
tests/ui/test_local.py            # Task 7
tests/ui/test_app.py              # Task 8 (needs gradio)
tests/ui/test_browser.py          # Task 8 (opt-in marker `browser`)
tests/ui/test_space.py            # Task 9
```

---

### Task 1: Auto-mapping (§4.7)

**Files:**
- Create: `clef_tune/automap.py`, `tests/test_automap.py`
- Modify: `tests/test_boundaries.py` (add `clef_tune.automap`)

**Interfaces:**
- Consumes: `clef_tune.labels.is_missing`, `NOUL_TRUE`, `NOUL_FALSE`; `clef_tune.schema.option_key`.
- Produces, in `clef_tune.automap`:
  - `MAX_CHOICE_OPTIONS = 50`, `MAX_UNIQUE_SHARE = 0.5`.
  - `@dataclass(frozen=True) class QuestionProposal(id: str, column: str, type: Literal["noul", "choice"], options: list[str])`.
  - `@dataclass(frozen=True) class Proposal(id_column: str | None, state_columns: list[str], questions: list[QuestionProposal], ignored: list[str], native: bool)`.
  - `propose(rows: list[dict[str, Any]], columns: list[str]) -> Proposal`.
  - `propose_column(rows: list[dict[str, Any]], column: str) -> QuestionProposal | None`. This is the per-column rule, also used when the person turns a column into a question. It returns `None` for a column with no values.

- [ ] **Step 1: Write the failing tests** `tests/test_automap.py`:

```python
from clef_tune.automap import Proposal, QuestionProposal, propose, propose_column


def rows(n, **columns):
    """columns: name -> function of the row index giving the cell text."""
    return [{name: make(i) for name, make in columns.items()} for i in range(n)]


def test_columns_are_proposed_from_their_values():
    data = rows(
        40,
        ticket_id=lambda i: f"T{i:03d}",
        subject=lambda i: f"subject {i}",
        dept=lambda i: ["billing", "billing", "technical", "account"][i % 4],
        outage=lambda i: ["Yes", "no"][i % 2],
        flag=lambda i: ["1", "0", ""][i % 3],
        notes=lambda i: " ",
    )
    assert propose(data, list(data[0])) == Proposal(
        id_column="ticket_id",
        state_columns=["subject"],
        questions=[
            QuestionProposal("dept", "dept", "choice", ["billing", "account", "technical"]),
            QuestionProposal("outage", "outage", "noul", ["true", "false"]),
            QuestionProposal("flag", "flag", "noul", ["true", "false"]),
        ],
        ignored=["notes"],
        native=False,
    )


def test_too_many_distinct_values_make_a_state_column():
    data = rows(120, code=lambda i: f"c{i % 51}", tier=lambda i: f"t{i % 50}")
    proposal = propose(data, ["code", "tier"])
    assert proposal.state_columns == ["code"]
    assert [q.id for q in proposal.questions] == ["tier"]


def test_mostly_unique_values_make_a_state_column():
    data = rows(6, size=lambda i: ["s", "m", "l", "xl", "s", "m"][i])  # 4 of 6 distinct
    assert propose(data, ["size"]).state_columns == ["size"]


def test_an_id_column_must_be_unique_and_complete():
    assert propose(rows(5, id=lambda i: f"r{i % 4}"), ["id"]).id_column is None
    gap = rows(5, row_id=lambda i: "" if i == 2 else f"r{i}")
    assert propose(gap, ["row_id"]).id_column is None
    assert propose(rows(5, Ticket_ID=lambda i: f"r{i}"), ["Ticket_ID"]).id_column == "Ticket_ID"


def test_a_single_column_can_be_proposed_on_demand():
    data = rows(10, size=lambda i: ["s", "m", "l"][i % 3])
    assert propose_column(data, "size") == QuestionProposal(
        "size", "size", "choice", ["s", "m", "l"]
    )
    assert propose_column(rows(3, empty=lambda i: ""), "empty") is None


def test_native_rows_propose_questions_from_their_labels():
    data = [
        {"id": "a", "state": {"text": "x"}, "labels": {"dept": "billing", "spam": True}},
        {"id": "b", "state": {"text": "y"}, "labels": {"dept": {"billing": 0.4, "technical": 0.6}}},
        {"id": "c", "state": {"text": "z"}, "labels": {"dept": "technical", "spam": "no"}},
    ]
    proposal = propose(data, ["id", "state", "labels"])
    assert proposal.native and proposal.state_columns == [] and proposal.id_column is None
    assert proposal.questions == [
        QuestionProposal("dept", "dept", "choice", ["billing", "technical"]),
        QuestionProposal("spam", "spam", "noul", ["true", "false"]),
    ]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_automap.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clef_tune.automap'`.

- [ ] **Step 3: Implement `clef_tune/automap.py`**
  - **Cell values:** strip each cell. Empty means `is_missing` or blank after stripping. A cell's value is `option_key(value)` for bools, else `str(value).strip()`.
  - **`propose_column` checks in order:**
    1. A column with no values gives `None`.
    2. If every value's casefold is in `NOUL_TRUE | NOUL_FALSE`, propose `noul` with `["true", "false"]`.
    3. If there are at most `MAX_CHOICE_OPTIONS` distinct values and `distinct / non_empty < MAX_UNIQUE_SHARE`, propose `choice`. Options are the distinct values by count, highest first, then by text.
    4. Otherwise `None`. The caller makes it a state column.
  - **`propose` on flat rows,** in column order:
    - The id column is the first column named `id` (case-insensitive) or ending in `_id`, ` id` or `-id`, whose values are all present and unique.
    - Columns without values go to `ignored`.
    - Other columns become a question when `propose_column` returns one, else a state column.
  - **Native rows:** any row with a `dict` `labels` makes the file native.
    - Questions are the label keys in first-seen order.
    - Soft labels (dicts) count each key once.
    - Native label values are always questions: `noul` when every value parses as noul, else `choice` with no distinct-count limit.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_automap.py tests/test_boundaries.py -v`
Expected: all PASS.

- [ ] **Step 5: Lint and commit** (`feat: propose a schema from an uploaded file's columns`).

---

### Task 2: UI dependency, rendering primitives and shared card pieces

**Files:**
- Create: `clef_tune/ui/__init__.py` (empty), `clef_tune/ui/charts.py`, `clef_tune/ui/fragments.py`, `tests/ui/__init__.py` (empty), `tests/ui/test_render.py`
- Modify:
  - `clef_tune/card.py`: rename `_ROWS` → `METRIC_ROWS`, `_WEIGHTING_NOTE` → `WEIGHTING_NOTE`, `_CALIBRATION_CAVEAT` → `CALIBRATION_CAVEAT`; extract `top_confusion()`.
  - `pyproject.toml`: `ui = ["gradio[oauth]>=6.30,<7"]`.
  - `uv.lock`: `uv lock`.
  - `.github/workflows/ci.yml`: the `test` job runs `uv sync --locked --extra dev --extra ui`.
  - `tests/test_boundaries.py`: add `clef_tune.ui`, `clef_tune.ui.charts`, `clef_tune.ui.fragments`, plus a new test, `test_ui_logic_never_imports_gradio`, over a `NO_GRADIO_MODULES` list.
  - `tests/test_card.py`: add a test for `top_confusion`.

**Interfaces:**
- Consumes: `clef_tune.data.Issue`, `MAX_LISTED_ROWS`; `clef_tune.prompt.Piece`; `clef_tune.estimate.EstimateRow`; `card.METRIC_ROWS`.
- Produces:
  - **In `clef_tune.card`:** `top_confusion(labels: list[str], matrix: list[list[int]], top: int = TOP_CLASSES) -> tuple[list[str], list[list[int]]]`. It returns the kept class names (plus `"other"`) and the merged matrix. `confusion_markdown` uses it.
  - **In `clef_tune.ui.charts`,** each function returns one `<svg class="ct-chart …">` string, or `<div class="ct-empty">…</div>` when there is nothing to draw:
    - `bar_chart(counts: dict[str, int], *, missing: int = 0) -> str`: horizontal bars, one per option, with the count at the end; `missing` adds a muted "unlabeled" bar;
    - `histogram(values: list[int], *, cutoff: int | None = None, bins: int = 24) -> str`: a dashed `ct-cutoff` line labelled with the cutoff;
    - `line_chart(series: dict[str, list[tuple[float, float]]], *, reference: dict[str, float] | None = None, y_label: str = "") -> str`: one `<polyline class="ct-line ct-series-{i}">` per series, plus one dashed `ct-reference` line per reference value;
    - `heatmap(labels: list[str], matrix: list[list[int]]) -> str`: rows are true labels and columns predicted ones, after `top_confusion`. Each cell has the class `ct-cell` and an inline `fill-opacity` equal to its share of the row (0.06 minimum for non-zero cells, 0 for zero). The count is printed in each cell.
  - **In `clef_tune.ui.fragments`,** each function returns HTML:
    - `esc(value) -> str`;
    - `stat_tiles(tiles: list[tuple[str, str]]) -> str`;
    - `issue_list(issues: list[Issue]) -> str`: errors before warnings; rows shortened as `Issue.__str__` does; `"No problems found"` when empty;
    - `prompt_preview(pieces: list[Piece]) -> str`: one `<span class="pp-{kind}">` per piece, inside `<pre class="ct-prompt">`;
    - `stage_badge(stage: str) -> str`;
    - `progress(fraction: float | None, label: str) -> str`: clamped to [0, 1]; `None` adds `ct-progress-unknown`;
    - `metrics_table(baseline: dict, final: dict) -> str`: the rows of `card.METRIC_ROWS`, with the fine-tuned value and its change from baseline;
    - `calibration_table(calibration: dict) -> str`;
    - `estimate_table(rows: list[EstimateRow], recommended: str | None) -> str`;
    - `callout(kind: Literal["info", "warning", "error", "success"], text: str) -> str`;
    - `empty_state(title: str, body: str) -> str`;
    - `column_chips(columns: list[str], roles: dict[str, str]) -> str`.
  - **Styling:** CSS classes only, never colours. Task 8's `style.css` styles them.

- [ ] **Step 1: Write the failing tests** `tests/ui/test_render.py`:

```python
import re

from clef_tune.data import Issue
from clef_tune.estimate import EstimateRow
from clef_tune.prompt import Piece
from clef_tune.ui import charts, fragments

EVIL = '<img src=x onerror="alert(1)">'


def row(flavor, fits=True, price=1.0, hours=1.0):
    return EstimateRow(flavor, "L4", 1, price, 20.0, 24.0, fits, fits, None if fits else "needs ~30 GB", hours)


def test_every_fragment_escapes_user_text():
    final = {
        "questions": {EVIL: {"question_type": "choice", "accuracy": 1.0, "f1": 1.0, "log_loss": 0.1}},
        "exact_match": 1.0,
    }
    outputs = [
        fragments.stat_tiles([(EVIL, EVIL)]),
        fragments.issue_list([Issue("error", EVIL, [2], EVIL)]),
        fragments.prompt_preview([Piece("state", EVIL)]),
        fragments.callout("error", EVIL),
        fragments.column_chips([EVIL], {EVIL: "state"}),
        fragments.stage_badge(EVIL),
        fragments.metrics_table({"questions": {}}, final),
        fragments.calibration_table({EVIL: {"temperature": 1.2, "ece_before": 0.1, "ece_after": 0.05}}),
        charts.bar_chart({EVIL: 3}),
        charts.heatmap([EVIL, "b"], [[1, 0], [0, 1]]),
        charts.line_chart({EVIL: [(1, 0.5), (2, 0.4)]}),
    ]
    for html in outputs:
        assert "<img" not in html and "&lt;img" in html


def test_issue_list_puts_errors_first_and_shortens_rows():
    html = fragments.issue_list(
        [Issue("warning", "rare classes", [], "dept"), Issue("error", "bad label", list(range(2, 40)), "dept")]
    )
    assert html.index("bad label") < html.index("rare classes")
    assert "+18 more" in html
    assert "No problems found" in fragments.issue_list([])


def test_prompt_preview_marks_each_kind():
    pieces = [Piece("prefix", "SYS"), Piece("state", "hello"), Piece("question", "Q?", "d"), Piece("option", "{}", "d")]
    html = fragments.prompt_preview(pieces)
    for kind in ("prefix", "state", "question", "option"):
        assert f'class="pp-{kind}"' in html


def test_estimate_table_marks_the_recommended_and_misfits():
    html = fragments.estimate_table([row("l4x1"), row("t4-small", fits=False)], "l4x1")
    assert "ct-recommended" in html and "needs ~30 GB" in html
    assert html.index("l4x1") < html.index("t4-small")


def test_progress_is_clamped_and_can_be_unknown():
    assert "width:100%" in fragments.progress(1.7, "done")
    assert "ct-progress-unknown" in fragments.progress(None, "Starting")


def test_bar_chart_draws_one_bar_per_option_and_unlabeled():
    svg = charts.bar_chart({"a": 3, "b": 0, "c": 7}, missing=2)
    assert len(re.findall(r'class="ct-bar[ "]', svg)) == 4 and "unlabeled" in svg


def test_histogram_marks_the_cutoff_and_handles_no_data():
    svg = charts.histogram([100, 900, 4096, 5000], cutoff=4096)
    assert 'class="ct-cutoff"' in svg and "4096" in svg
    assert "ct-empty" in charts.histogram([])


def test_heatmap_shades_by_row_share_and_merges_beyond_ten():
    svg = charts.heatmap(["a", "b"], [[3, 1], [0, 4]])
    assert svg.count('class="ct-cell"') == 4 and 'fill-opacity="1' in svg
    labels = [f"c{i}" for i in range(12)]
    big = charts.heatmap(labels, [[1 if i == j else 0 for j in range(12)] for i in range(12)])
    assert "other" in big and big.count('class="ct-cell"') == 11 * 11


def test_line_chart_stays_inside_its_view_box():
    svg = charts.line_chart({"loss": [(10, 1.2), (20, 0.6), (30, 0.3)]}, reference={"baseline": 1.0})
    width, height = map(float, re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg).groups())
    points = re.search(r'<polyline[^>]*points="([^"]+)"', svg).group(1).split()
    for point in points:
        x, y = map(float, point.split(","))
        assert 0 <= x <= width and 0 <= y <= height
    assert 'class="ct-reference' in svg
```

Add to `tests/test_card.py`:

```python
def test_top_confusion_merges_the_rarest_classes_into_other():
    labels = ["a", "b", "c"]
    names, rows = top_confusion(labels, [[5, 1, 0], [0, 3, 1], [1, 0, 1]], top=2)
    assert names == ["a", "b", "other"]
    assert rows == [[5, 1, 0], [0, 3, 1], [1, 0, 1]]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/ui/test_render.py tests/test_card.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clef_tune.ui'`, and an ImportError for `top_confusion`.

- [ ] **Step 3: Implement**
  - **Charts:** fixed view boxes, so `width: 100%` scales them: bar chart `480 × (28 × bars + 16)`, histogram `480 × 200`, line chart `480 × 220`, heatmap `(80 + 40n) × (60 + 40n)`.
  - **Labels:** truncated to 24 characters with `…` and the full text in `<title>`, all escaped.
  - **`stage_badge`:** maps `COMPLETED`→`ct-badge-success`, `RUNNING`/`SCHEDULING`→`ct-badge-active`, `ERROR`→`ct-badge-error`, `CANCELED`/`DELETED`→`ct-badge-muted`, and anything else to `ct-badge-muted`.
  - **Dependency:** run `uv lock`, then `uv sync --locked --extra dev --extra train --extra ui`. The lock gains `authlib`, `itsdangerous` and their dependencies, and gradio stays 6.30.0.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/ui/test_render.py tests/test_card.py tests/test_boundaries.py -v`
Expected: all PASS.

- [ ] **Step 5: Lint and commit** (`feat: add the UI's escaped HTML fragments and SVG charts`).

---

### Task 3: Settings, tokens, sessions and the editable draft

**Files:**
- Create: `clef_tune/ui/settings.py`, `clef_tune/ui/session.py`, `tests/ui/test_session.py`
- Modify:
  - `pyproject.toml`: add `[tool.hatch.build.targets.wheel.force-include]` mapping `"examples/support_tickets.csv" = "clef_tune/_examples/support_tickets.csv"`, and the same for `.yaml`.
  - `tests/test_boundaries.py`: add both modules to `NO_TORCH_MODULES` and `NO_GRADIO_MODULES`.

**Interfaces:**
- Consumes:
  - `automap.propose`, `Proposal`;
  - `data.read_rows`, `DataError`, `Issue`;
  - `config.Config`, `load_config`, `PRESETS`;
  - `schema.SchemaError`;
  - `estimate.EstimateRow`;
  - `validation.ValidationReport`.
- Produces, in `clef_tune.ui.settings`:
  - **Constants:** `MAX_UPLOAD_ENV = "CLEF_TUNE_MAX_UPLOAD_MB"`, `DEFAULT_MAX_UPLOAD_MB = 200`, `MOCK_OAUTH_TOKEN = "mock-oauth-token-for-local-dev"`.
  - **`class SignInRequired(RuntimeError)`.**
  - **`Settings`:** `@dataclass(frozen=True) class Settings(space: bool, max_upload_mb: int, local_training: bool)` with:
    - `max_upload_bytes -> int`;
    - `from_env(env: Mapping[str, str] = os.environ, *, cuda: Callable[[], bool] = cuda_available) -> Settings`.

    `space` is `bool(env.get("SPACE_ID"))`. `local_training = not space and cuda()`, and `cuda` is never called on a Space. A cap that is not a positive integer raises a `ValueError` that names the variable.
  - **`cuda_available() -> bool`:** runs `[sys.executable, "-c", "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"]` with a 120 s timeout. Any failure is `False`.
  - **`user_token(settings: Settings, oauth_token: str | None, pasted: str | None, *, local_token: Callable[[], str | None] = get_token) -> str`:** the order is:
    1. a stripped, non-empty pasted token;
    2. on a Space, the OAuth token, and otherwise `SignInRequired("Sign in with Hugging Face, or paste a token")`;
    3. off a Space, `local_token()` (the OAuth token is the mock there and is ignored), and otherwise `SignInRequired("Run `hf auth login`, or paste a token")`.
- Produces, in `clef_tune.ui.session`:
  - **Constants:** `SESSIONS_ROOT = Path(tempfile.gettempdir()) / "clef-tune-ui"`, `STALE_HOURS = 24`, `UNZIP_FACTOR = 5`, `DEFAULT_MODEL = "Cloudflare/clef-flash"`, `DATA_SUFFIXES = (".csv", ".tsv", ".jsonl", ".ndjson")`.
  - **`class UploadError(ValueError)`.**
  - **`OptionDraft`:** `@dataclass class OptionDraft(id: str, description: str = "")`.
  - **`QuestionDraft`:** `@dataclass class QuestionDraft(id: str, column: str, type: str, instructions: str = "", options: list[OptionDraft])` with `set_type(new_type: str, observed: list[str]) -> None`. Rules:
    - `noul` → options `true`, `false`, keeping matching descriptions;
    - `choice` → if the options are empty or the noul pair, they become `observed`, else they stay;
    - `score` → options stay, sorted numerically when every id is an integer.
  - **`Draft`:** `@dataclass class Draft` with these fields:
    - `run_name: str`, `model_id: str = DEFAULT_MODEL`;
    - `id_column: str | None = None`;
    - `state_columns: list[str]`, `image_columns: list[str]`, `questions: list[QuestionDraft]`;
    - `native: bool = False`, `preset: str = "balanced"`;
    - `overrides: dict[str, Any]`: dotted config path → value, applied after the preset.

    Its methods:
    - `to_config(base_dir: Path, data_path: str) -> Config`: raises `ValueError` (pydantic's `ValidationError` and `SchemaError` both are);
    - `fingerprint() -> str`: sha256 of the sorted-key JSON of `asdict(self)`, first 16 hex characters.

    `to_config` maps each question type to criteria like this:
    - `choice`: `{option.id: option.description or None}`;
    - `score`: `[option.description or option.id, …]`;
    - `noul`: `{id: description}` for options that have a description, else no criteria.

    `label_columns` lists only questions whose column differs from their id. Native drafts omit `state_columns` and `label_columns`.
  - **`Table`:** `@dataclass class Table(path: Path, rows: list[dict[str, Any]], columns: list[str], issues: list[Issue])`. It also has `observed(column: str) -> list[str]` (distinct values, most common first) and `values(column: str) -> list[Any]`.
  - **`Session`:** `@dataclass class Session` with fields `root: Path | None = None`, `table: Table | None = None`, `draft: Draft | None = None`, `report: ValidationReport | None = None`, `estimates: list[EstimateRow]`, `reached: int = 0`, `job_id: str | None = None`, `local_folder: str | None = None`, `pending: Any = None`, `last_view: Any = None`, and `results: dict[str, dict]`. Its methods:
    - `workdir() -> Path`: creates `SESSIONS_ROOT / uuid4().hex` on first use;
    - `close() -> None`: idempotent `rmtree`;
    - `write_config() -> Path`: writes `workdir()/config.yaml` from `draft.to_config(workdir(), table path relative to workdir)`.

    Session holds no locks, processes or tokens. Gradio deep-copies its initial value per browser session.
  - **Other functions:**
    - `slug(text: str) -> str`: a valid `run_name`;
    - `load_upload(session: Session, upload: Path, settings: Settings) -> None`;
    - `load_example(session: Session) -> None`;
    - `example_files() -> tuple[Path, Path]`;
    - `extract_zip(archive: Path, dest: Path, max_bytes: int) -> Path`;
    - `draft_from_proposal(proposal: Proposal, name: str) -> Draft`;
    - `draft_from_config(config: Config) -> Draft`;
    - `sweep_stale(root: Path = SESSIONS_ROOT, *, max_age_hours: float = STALE_HOURS, now: float | None = None) -> int`.

- [ ] **Step 1: Write the failing tests** `tests/ui/test_session.py`:

```python
import os
import zipfile

import pytest

from clef_tune.ui.session import (
    OptionDraft,
    QuestionDraft,
    Session,
    UploadError,
    extract_zip,
    load_example,
    load_upload,
    slug,
    sweep_stale,
)
from clef_tune.ui.settings import MOCK_OAUTH_TOKEN, Settings, SignInRequired, user_token
from tests.helpers import ticket_rows, write_csv

LOCAL = Settings(space=False, max_upload_mb=200, local_training=False)
SPACE = Settings(space=True, max_upload_mb=200, local_training=False)


@pytest.fixture
def session():
    s = Session()
    yield s
    s.close()


def test_settings_read_space_mode_and_the_upload_cap():
    def no_cuda():
        pytest.fail("checked for CUDA on a Space")

    env = {"SPACE_ID": "me/clef-tune", "CLEF_TUNE_MAX_UPLOAD_MB": "50"}
    assert Settings.from_env(env, cuda=no_cuda) == Settings(True, 50, False)
    assert Settings.from_env({}, cuda=lambda: True) == Settings(False, 200, True)
    for bad in ("lots", "0"):
        with pytest.raises(ValueError, match="CLEF_TUNE_MAX_UPLOAD_MB"):
            Settings.from_env({"CLEF_TUNE_MAX_UPLOAD_MB": bad}, cuda=lambda: False)


def test_a_space_never_uses_the_server_token():
    with pytest.raises(SignInRequired, match="Sign in"):
        user_token(SPACE, None, " ", local_token=lambda: "hf_server")
    assert user_token(SPACE, "hf_oauth", None, local_token=lambda: "hf_server") == "hf_oauth"
    assert user_token(SPACE, "hf_oauth", " hf_pasted ") == "hf_pasted"


def test_locally_the_mock_sign_in_is_ignored():
    assert user_token(LOCAL, MOCK_OAUTH_TOKEN, None, local_token=lambda: "hf_local") == "hf_local"
    with pytest.raises(SignInRequired, match="hf auth login"):
        user_token(LOCAL, None, None, local_token=lambda: None)


def test_an_upload_becomes_a_draft(session, tmp_path):
    path = write_csv(tmp_path / "tickets.csv", ticket_rows(30))
    load_upload(session, path, LOCAL)
    draft = session.draft
    assert session.table.columns == ["id", "subject", "body", "department", "urgency", "outage"]
    assert (draft.run_name, draft.id_column, draft.state_columns) == ("tickets", "id", ["subject", "body"])
    assert [(q.id, q.type) for q in draft.questions] == [
        ("department", "choice"),
        ("urgency", "choice"),
        ("outage", "noul"),
    ]
    assert session.reached == 1 and session.table.path.is_relative_to(session.workdir())


def test_run_names_are_made_valid():
    assert slug("My Data (v2)") == "My-Data-v2"
    assert slug("___") == "run" and slug("é") == "run"
    assert len(slug("x" * 100)) == 64


def test_load_example_uses_the_example_config(session):
    load_example(session)
    draft = session.draft
    assert [q.id for q in draft.questions] == ["department", "urgency", "outage"]
    assert draft.questions[1].type == "score" and draft.id_column == "ticket_id"
    assert draft.model_id == "Cloudflare/clef-flash"
    assert len(session.table.rows) == 200


def test_a_zip_keeps_its_folders_so_image_paths_resolve(session, tmp_path):
    archive = tmp_path / "upload.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("set/tickets.csv", "id,text,label,img\n1,a,x,img/a.png\n2,b,y,img/b.png\n")
        z.writestr("set/img/a.png", b"png")
        z.writestr("set/img/b.png", b"png")
        z.writestr("__MACOSX/set/._tickets.csv", b"junk")
    load_upload(session, archive, LOCAL)
    assert session.table.path.name == "tickets.csv"
    assert (session.table.path.parent / "img" / "a.png").exists()


@pytest.mark.parametrize("name", ["../evil.csv", "/abs/evil.csv", "a/../../evil.csv"])
def test_hostile_zips_are_refused(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(name, "id,x\n1,2\n")
    dest = tmp_path / "out" / "dest"
    with pytest.raises(UploadError, match="outside"):
        extract_zip(archive, dest, max_bytes=10_000)
    assert not (tmp_path / "evil.csv").exists() and not (tmp_path / "out" / "evil.csv").exists()


def test_zips_that_expand_too_far_or_hold_the_wrong_files_are_refused(tmp_path):
    bomb = tmp_path / "bomb.zip"
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("big.csv", "0" * 50_000)
    with pytest.raises(UploadError, match="expands to more than"):
        extract_zip(bomb, tmp_path / "a", max_bytes=10_000)
    for names in ([], ["a.csv", "b.jsonl"]):
        archive = tmp_path / f"n{len(names)}.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("readme.txt", "hi")
            for name in names:
                z.writestr(name, "id,x\n1,2\n")
        with pytest.raises(UploadError, match="exactly one .csv, .tsv or .jsonl"):
            extract_zip(archive, tmp_path / f"d{len(names)}", max_bytes=10_000)


def test_an_upload_over_the_cap_is_refused(session, tmp_path):
    path = tmp_path / "big.csv"
    path.write_bytes(b"id,x\n" + b"1,2\n" * 400_000)  # 1.6 MB
    with pytest.raises(UploadError, match="limit is 1 MB"):
        load_upload(session, path, Settings(space=True, max_upload_mb=1, local_training=False))
    assert session.table is None


def test_a_file_that_cannot_be_read_is_an_upload_error(session, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    with pytest.raises(UploadError, match="unsupported file type"):
        load_upload(session, path, LOCAL)


def test_sessions_are_isolated_and_closed():
    first, second = Session(), Session()
    assert first.workdir() != second.workdir() and first.workdir().exists()
    first.close()
    first.close()
    assert not first.root.exists() and second.workdir().exists()
    second.close()


def test_stale_session_folders_are_swept(tmp_path):
    old, fresh = tmp_path / "old", tmp_path / "fresh"
    old.mkdir(), fresh.mkdir()
    os.utime(old, (1_000, 1_000))
    assert sweep_stale(tmp_path, max_age_hours=24, now=1_000 + 25 * 3600) == 1
    assert not old.exists() and fresh.exists()


def test_the_draft_builds_a_config(session):
    load_example(session)
    draft = session.draft
    draft.preset, draft.overrides = "thorough", {"train.epochs": 2}
    draft.questions[0].id = "team"  # column stays "department"
    config = draft.to_config(session.workdir(), "data/support_tickets.csv")
    assert config.data.label_columns == {"team": "department"}
    assert (config.train.lora.r, config.train.epochs) == (32, 2)
    schema = config.get_schema()
    assert schema.questions["urgency"].criteria == ["Can wait", "This week", "Today"]


def test_switching_types_keeps_sensible_options():
    question = QuestionDraft("urgency", "urgency", "choice", options=[OptionDraft(i) for i in "201"])
    question.set_type("score", observed=["0", "1", "2"])
    assert [o.id for o in question.options] == ["0", "1", "2"]
    question.set_type("noul", observed=["0", "1", "2"])
    assert [o.id for o in question.options] == ["true", "false"]
    question.set_type("choice", observed=["a", "b"])
    assert [o.id for o in question.options] == ["a", "b"]


def test_the_fingerprint_changes_with_any_edit(session):
    load_example(session)
    before = session.draft.fingerprint()
    session.draft.questions[0].instructions = "Which team?"
    assert session.draft.fingerprint() != before
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/ui/test_session.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `settings.py` and `session.py`.** The non-obvious rules:
  - **`load_upload`:**
    1. Check the size first: over `settings.max_upload_bytes` raises `UploadError(f"the file is {mb:.1f} MB; the limit is {cap} MB")`.
    2. Replace `workdir()/data`.
    3. For a `.zip`, `extract_zip(..., max_bytes=settings.max_upload_bytes * UNZIP_FACTOR)`. Otherwise, copy the file in under its own name.
    4. Read it with `read_rows`. A `DataError` becomes an `UploadError` with the same message.
    5. Then set `table` and `draft = draft_from_proposal(propose(...), name)`, and reset `report`, `estimates`, `pending`, `job_id` and `results`. Set `reached = 1`.

    On any `UploadError`, leave the session as it was before the call.
  - **`extract_zip`:**
    - Skip `__MACOSX/` and names whose last part starts with `.`.
    - Refuse a member whose resolved path is not inside `dest.resolve()`: `UploadError("the zip has a file outside its folder: …")`.
    - Refuse when the sum of `file_size` exceeds `max_bytes`: `UploadError(f"the zip expands to more than {max_bytes / 2**20:.0f} MB")`.
    - Check both before writing anything.
    - Exactly one member must have a suffix in `DATA_SUFFIXES`, else `UploadError("the zip must contain exactly one .csv, .tsv or .jsonl file (found N)")`.
  - **`slug`:** replace each run of characters outside `[A-Za-z0-9._-]` with `-`. Strip leading characters until an alphanumeric, and strip trailing `-`. Cut to 64 characters. Return `"run"` when empty.
  - **`example_files`:** `importlib.resources.files("clef_tune") / "_examples"` when it exists (an installed wheel), else `Path(clef_tune.__file__).parents[1] / "examples"` (a checkout).
  - **`draft_from_config`:** the inverse of `to_config` for the example yaml. `load_example` copies the CSV into `workdir()/data/` and drafts from the yaml.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/ui/test_session.py tests/test_boundaries.py -v`
Expected: all PASS.

- [ ] **Step 5: Check the wheel ships the example.** Run `uv build --wheel -o %TEMP%\clef-wheel`, then list it with `python -c "import zipfile,glob; print([n for n in zipfile.ZipFile(glob.glob(r'%TEMP%\clef-wheel\*.whl')[0]).namelist() if '_examples' in n])"`. Expected: both example files. Delete the folder afterwards.

- [ ] **Step 6: Lint and commit** (`feat: add UI sessions, uploads, the editable draft and token rules`).

---

### Task 4: Step handlers for Data, Schema, Validate and Configure

**Files:**
- Create: `clef_tune/ui/services.py`, `clef_tune/ui/handlers.py`, `tests/ui/fakes.py`, `tests/ui/test_handlers.py`
- Modify:
  - `clef_tune/validation.py`: `validate(config, counter, *, counter_error: str | None = None)`. The tokenizer warning becomes `f"tokenizer unavailable ({counter_error}); …"` when it is given.
  - `clef_tune/launch/hf_jobs.py` and `clef_tune/cli.py`: pass the exception text from their tokenizer fallbacks as `counter_error`. This fixes Plan 3's deferred minor.
  - `tests/test_validation.py`: one test for the reason.
  - `tests/test_boundaries.py`: add the new modules.

**Interfaces:**
- Consumes: Task 2 fragments and charts; Task 3 `Session`, `Draft`, `QuestionDraft`, `load_upload`, `load_example`, `UploadError`, `Settings`; `automap.propose_column`; `prompt.prompt_pieces`; `validation.validate`; `estimate.*`; `bundle.write_bundle`, `zip_bundle`; `labels.parse_label`.
- Produces, in `clef_tune.ui.services`:
  - **`Services`:** `@dataclass(frozen=True) class Services`. Task 4 adds `counter: Callable[[Config], TokenCounter] = counter_for_config` and `prices: Callable[[], dict[str, float]] = cached_prices`.
  - **`cached_prices() -> dict[str, float]`:** `fetch_hf_prices()`, cached per process for 3600 s. An empty result is not cached.
- Produces, in `tests/ui/fakes.py`: `fake_services(**overrides) -> tuple[Services, list[tuple]]`.
  - It records each call as `(name, args, kwargs)`.
  - The counter returns `tests.helpers.CharCounter()` and prices return `{}`.
  - Task 6 extends it.
- Produces, in `clef_tune.ui.handlers`:
  - **Constants:** `STEPS = ("Data", "Schema", "Validate", "Configure", "Launch")`, `PREVIEW_PAGE = 50`, `CELL_CHARS = 200`.
  - **Data step:**
    - `@dataclass class DataView(summary: str, chips: str, headers: list[str], rows: list[list[str]], page: int, page_label: str, error: str | None)`;
    - `upload(session, path: str | None, settings) -> DataView`: an `UploadError` becomes `DataView.error`;
    - `example(session) -> DataView`;
    - `data_view(session, page: int = 0) -> DataView`: pages are clamped, and cells longer than `CELL_CHARS` end in `…`;
    - `column_roles(session) -> dict[str, str]`: each column → `"id" | "state" | "image" | "question" | "ignored"`.
  - **Schema step:**
    - `@dataclass class QuestionCard(id: str, column: str, type: str, instructions: str, options: list[list[Any]], unknown: str)`. `options` rows are `[option id, description, label count]`. `unknown` is a callout listing values that are not options, with counts, or `""`.
    - `@dataclass class SchemaView(columns: list[str], id_column: str | None, state_columns: list[str], image_columns: list[str], free_columns: list[str], cards: list[QuestionCard], preview: str, error: str | None)`.
    - `schema_view(session, preview_row: int = 0) -> SchemaView`.
    - `set_columns(session, *, id_column: str | None, state_columns: list[str], image_columns: list[str]) -> None`.
    - `add_question(session, column: str) -> str | None`: proposes with `propose_column`, falling back to choice with the observed values. It returns an error when the column is already used.
    - `remove_question(session, qid: str) -> None`.
    - `update_question(session, qid: str, *, new_id: str | None = None, type: str | None = None, instructions: str | None = None, options: list[list[Any]] | None = None) -> str | None`: returns an error message for an empty or duplicate id.
  - **Validate step:**
    - `refresh(session, services) -> None`:
      - A draft that does not build gives a one-issue report: `Issue("error", f"schema/config: {exc}")`.
      - Otherwise it gets the counter (an exception becomes `counter_error=str(exc)`), runs `validate`, and computes `estimate_hf_flavors` with `services.prices()` when `report.tokens` is set.
    - `@dataclass class ValidateView(tiles: str, issues: str, distributions: str, histogram: str, ok: bool)`.
    - `validate_view(session) -> ValidateView`.
  - **Configure step:**
    - `@dataclass(frozen=True) class FieldSpec(path: str, label: str, kind: Literal["int", "float", "bool", "choice", "text", "optional_int"], choices: tuple[str, ...] = (), help: str = "")`.
    - `ADVANCED_FIELDS: tuple[FieldSpec, ...]` and `EXCLUDED_FIELDS: frozenset[str]`.
    - `@dataclass class ConfigureView(preset: str, values: dict[str, Any], question_weights: list[list[Any]], estimates: str, error: str | None)`.
    - `set_preset(session, preset: str) -> None`: drops overrides of the preset's own paths.
    - `set_field(session, path: str, value: Any) -> str | None`: rolls back and returns an error when the config would not build.
    - `set_question_weights(session, rows: list[list[Any]]) -> str | None`.
    - `configure_view(session) -> ConfigureView`.
  - **Sidebar and bundle:**
    - `set_model(session, model_id: str) -> None`;
    - `set_run_name(session, name: str) -> str | None`;
    - `step_states(session) -> list[Literal["todo", "done", "warning", "error"]]`;
    - `sidebar_estimate(session) -> str`;
    - `bundle(session) -> Path`.

- [ ] **Step 1: Write the failing tests** `tests/ui/test_handlers.py`:

```python
import zipfile

import pytest

from clef_tune.ui import handlers
from clef_tune.ui.session import Session
from clef_tune.ui.settings import Settings
from tests.helpers import ticket_rows, write_csv
from tests.ui.fakes import fake_services

LOCAL = Settings(space=False, max_upload_mb=200, local_training=False)


@pytest.fixture
def services():
    return fake_services()[0]


@pytest.fixture
def session():
    s = Session()
    handlers.example(s)
    yield s
    s.close()


def test_the_preview_is_paginated_and_cells_are_shortened(tmp_path):
    rows = ticket_rows(120)
    rows[0]["body"] = "x" * 500
    s = Session()
    handlers.upload(s, str(write_csv(tmp_path / "t.csv", rows)), LOCAL)
    first, second = handlers.data_view(s, 0), handlers.data_view(s, 1)
    assert len(second.rows) == 50 and second.page_label == "Rows 51–100 of 120"
    assert len(first.rows[0][2]) == handlers.CELL_CHARS + 1 and first.rows[0][2].endswith("…")
    assert handlers.data_view(s, 99).page == 2
    s.close()


def test_upload_errors_are_shown_not_raised(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("hi")
    s = Session()
    view = handlers.upload(s, str(bad), LOCAL)
    assert "unsupported file type" in view.error and s.table is None
    s.close()


def test_schema_cards_count_labels_per_option(session):
    cards = {card.id: card for card in handlers.schema_view(session).cards}
    department = cards["department"]
    assert sum(row[2] for row in department.options) == 200
    assert department.unknown == ""


def test_values_that_are_not_options_are_listed(session):
    card = handlers.schema_view(session).cards[0]
    kept = [row for row in card.options if row[0] != "account"]
    handlers.update_question(session, "department", options=kept)
    unknown = handlers.schema_view(session).cards[0].unknown
    assert "account" in unknown and "not an option" in unknown


def test_prompt_preview_escapes_the_row(tmp_path):
    rows = ticket_rows(30)
    rows[0]["subject"] = "<script>alert(1)</script>"
    s = Session()
    handlers.upload(s, str(write_csv(tmp_path / "t.csv", rows)), LOCAL)
    preview = handlers.schema_view(s, preview_row=0).preview
    assert "&lt;script&gt;" in preview and "<script>" not in preview and 'class="pp-state"' in preview
    s.close()


def test_refresh_validates_and_estimates(session, services):
    handlers.refresh(session, services)
    view = handlers.validate_view(session)
    assert view.ok and "Train rows" in view.tiles and 'class="ct-cutoff"' in view.histogram
    assert session.estimates and "$" in handlers.sidebar_estimate(session)
    assert handlers.step_states(session)[:3] in (["done", "done", "done"], ["done", "done", "warning"])


def test_a_broken_schema_is_an_error_not_a_crash(session, services):
    handlers.update_question(session, "department", options=[])
    handlers.refresh(session, services)
    assert not handlers.validate_view(session).ok
    assert "criteria" in handlers.validate_view(session).issues
    assert handlers.step_states(session)[1] == "error"


def test_a_tokenizer_failure_says_why(session):
    def offline(config):
        raise OSError("hub unreachable")

    handlers.refresh(session, fake_services(counter=offline)[0])
    assert "hub unreachable" in handlers.validate_view(session).issues


def test_presets_and_advanced_fields(session, services):
    handlers.set_preset(session, "thorough")
    assert handlers.configure_view(session).values["train.epochs"] == 5
    assert handlers.set_field(session, "train.epochs", 2) is None
    assert handlers.configure_view(session).values["train.epochs"] == 2
    handlers.set_preset(session, "quick")
    assert handlers.configure_view(session).values["train.epochs"] == 1
    error = handlers.set_field(session, "train.max_length", 100)
    assert "max_length" in error and handlers.configure_view(session).values["train.max_length"] == 2048


def test_every_config_field_is_editable_or_deliberately_excluded():
    from clef_tune.config import Config

    def leaves(model, prefix=""):
        for name, field in model.model_fields.items():
            key = field.alias or name
            annotation = field.annotation
            if isinstance(annotation, type) and hasattr(annotation, "model_fields"):
                yield from leaves(annotation, f"{prefix}{key}.")
            else:
                yield f"{prefix}{key}"

    editable = {spec.path for spec in handlers.ADVANCED_FIELDS} | {"train.question_weights"}
    assert set(leaves(Config)) == editable | handlers.EXCLUDED_FIELDS
    assert not editable & handlers.EXCLUDED_FIELDS


def test_the_bundle_zip_holds_a_self_contained_config(session, services):
    handlers.refresh(session, services)
    names = zipfile.ZipFile(handlers.bundle(session)).namelist()
    assert {"config.yaml", "train.jsonl", "eval.jsonl"} <= set(names)


def test_no_bundle_while_validation_has_errors(session, services):
    handlers.update_question(session, "department", options=[])
    handlers.refresh(session, services)
    with pytest.raises(ValueError, match="validation"):
        handlers.bundle(session)
```

Add to `tests/test_validation.py`:

```python
def test_the_tokenizer_warning_says_why(tmp_path):
    config = make_config(tmp_path, data={"train": "t.csv", "id_column": "id"})
    write_csv(tmp_path / "t.csv", ticket_rows(60))
    report = validate(config, None, counter_error="hub unreachable")
    assert any("tokenizer unavailable (hub unreachable)" in i.message for i in report.warnings)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/ui/test_handlers.py tests/test_validation.py -v`
Expected: FAIL with `ModuleNotFoundError` (handlers) and `TypeError` (`counter_error`).

- [ ] **Step 3: Implement `services.py` and `handlers.py`.** The decisions the tests do not fix:
  - **`EXCLUDED_FIELDS`:** `model.id` (header picker), `model.revision` and `model.local_path` (pinned or set by the launcher), `run_name` (Configure has its own textbox), `schema`, `schema_file` and `data.*` except `data.eval_fraction` (the Schema step and the upload), `output.dir` (set by the launcher), `output.push_to_hub` (Launch step), and `output.report_to` (jobs get no tracker credentials).
  - **`help`:** every `FieldSpec` gets one plain sentence. `train.class_weighting` uses `card.WEIGHTING_NOTE`; `eval.calibrate` uses `card.CALIBRATION_CAVEAT`.
  - **`step_states`,** one rule per step:
    - **Data:** `done` once a table is loaded.
    - **Schema:** `error` when the draft does not build, `done` when it does.
    - **Validate:** `todo` without a report; `error`, `warning` or `done` from its issues.
    - **Configure:** `done` once `reached > 3` and the config builds.
    - **Launch:** `done` once `session.job_id` or a local run is set.
  - **Option counts:** parse each cell with `parse_label` against the draft question built so far. A `LabelError` counts the raw value as unknown.
  - **`sidebar_estimate`:** the recommended row as `"$0.31–$0.67 · 0.4–0.8 h on l4x1"`. Without data it shows an empty state: "Load data to see an estimate".

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/ui tests/test_validation.py tests/test_boundaries.py tests/test_cli.py tests/test_hf_jobs.py -v`
Expected: all PASS.

- [ ] **Step 5: Lint and commit** (`feat: add the Data, Schema, Validate and Configure step handlers`).

---

### Task 5: What the monitor reads: total steps, adapter repo, resume quote, metrics download

**Files:**
- Create: `scripts/make_recorded_run.py`, `tests/ui/fixtures/recorded_run/metrics.jsonl`, `tests/ui/fixtures/recorded_run/metrics.json`, `tests/ui/test_recorded_run.py`
- Modify:
  - `clef_tune/train.py`: `log_train` adds `total_steps=total_steps`.
  - `clef_tune/launch/hf_jobs.py`: `adapter_repo_of`, `JobStatus.adapter_repo`, `ResumeQuote`, `quote_resume`; `resume_job` reuses `quote_resume`'s checks.
  - `clef_tune/hub.py`: `pull_metrics`.
  - `clef_tune/cli.py`: `jobs resume` prints the quote and asks unless `--yes`.
  - Tests: `tests/torch/test_train.py`, `tests/test_hf_jobs.py`, `tests/test_hub.py`, `tests/test_cli_jobs.py`. The existing `test_cancel_and_resume` passes `--yes`.

**Interfaces:**
- Produces:
  - **`train` events:** each one carries `total_steps: int`, the planned optimizer steps.
  - **`hf_jobs.adapter_repo_of(info: JobInfo) -> str | None`:** the word after `--push-to-hub` in the job's bash script, read with `shlex.split` per line.
  - **`hf_jobs.JobStatus`:** gains `adapter_repo: str | None = None` as its last field.
  - **`hf_jobs.ResumeQuote`:** `@dataclass(frozen=True) class ResumeQuote(job_id: str, flavor: str | None, timeout: str, price_per_hour: float | None)` with `max_cost -> float | None`.
  - **`hf_jobs.quote_resume(job_id: str, *, timeout: str | None = None, api: HfApi | None = None, token: str | None = None) -> ResumeQuote`:** the same refusals as `resume_job` (not clef-tune, wrong stage, another job continues, no timeout, bad timeout), and starts nothing.
  - **`hub.pull_metrics(repo_id: str, *, token: str | None = None, download: Callable[..., str] = hf_hub_download) -> dict`:** downloads `metrics.json`. Any failure raises `HubError(f"cannot read metrics.json from {repo_id}: {exc}")`.
  - **The fixture:** `tests/ui/fixtures/recorded_run/` holds a deterministic run named `support-tickets` on `l4x1` with 120 steps, `eval_every` 40, 170 train rows and 30 eval rows. Its `metrics.json` is built by `metrics.summarize` and `calibration.calibrate` over seeded synthetic logits (department, urgency, outage). So its shape is the trainer's, and only its numbers are made up.

- [ ] **Step 1: Write the failing tests.** Extend `tests/torch/test_train.py`'s first test:

```python
    assert train_events[-1]["total_steps"] == result.metrics["run"]["steps"]
```

Add to `tests/test_hf_jobs.py`, reusing its `FakeApi`, `job(stage, **overrides)` helper, `SOURCE` and `TOKEN`:

```python
def test_job_status_names_the_adapter_repo():
    command = job_command(SOURCE, gpus=1, adapter_repo="me/clef-tune-x", keep_data=False, kernels=True)
    api = FakeApi(job_info=job(command=command))
    assert job_status("job123", api=api, token=TOKEN).adapter_repo == "me/clef-tune-x"
    assert hf_jobs.adapter_repo_of(job()) is None  # "echo train" pushes nowhere


def test_quote_resume_prices_the_timeout_and_starts_nothing():
    info = job(
        "ERROR",
        labels={"tool": "clef-tune", "run": "r", "timeout": "2h"},
        environment={PRICE_ENV: "1.5"},
    )
    api = FakeApi(job_info=info, jobs=[info])
    quote = hf_jobs.quote_resume("job123", api=api, token=TOKEN)
    assert (quote.timeout, quote.price_per_hour, quote.max_cost) == ("2h", 1.5, 3.0)
    assert hf_jobs.quote_resume("job123", timeout="3h", api=api, token=TOKEN).max_cost == 4.5
    assert not [c for c in api.calls if c[0] == "run_job"]


def test_quote_resume_refuses_what_resume_refuses():
    api = FakeApi(job_info=job("RUNNING"))
    with pytest.raises(LaunchError, match="only failed or canceled"):
        hf_jobs.quote_resume("job123", api=api, token=TOKEN)
```

Add to `tests/test_hub.py`:

```python
def test_pull_metrics_reads_metrics_json(tmp_path):
    (tmp_path / "metrics.json").write_text('{"final": {}}')
    seen = []

    def download(repo_id, filename, **kwargs):
        seen.append((repo_id, filename, kwargs.get("token")))
        return str(tmp_path / filename)

    assert pull_metrics("me/a", token="t", download=download) == {"final": {}}
    assert seen == [("me/a", "metrics.json", "t")]


def test_pull_metrics_explains_a_missing_file():
    def download(*args, **kwargs):
        raise FileNotFoundError("404")

    with pytest.raises(HubError, match="cannot read metrics.json from me/a"):
        pull_metrics("me/a", download=download)
```

Add to `tests/test_cli_jobs.py`:

```python
def test_resume_shows_the_cost_and_asks(monkeypatch):
    calls = []
    quote = ResumeQuote("job123", "a10g-large", "2h", 1.5)
    monkeypatch.setattr(cli, "quote_resume", lambda job_id, **k: quote)
    monkeypatch.setattr(
        cli, "resume_job", lambda job_id, **k: calls.append(job_id) or SimpleNamespace(id="job9", url="u")
    )
    refused = runner.invoke(cli.app, ["jobs", "resume", "job123"], input="n\n")
    assert refused.exit_code == 1 and "at most $3.00" in refused.output and calls == []
    assert runner.invoke(cli.app, ["jobs", "resume", "job123", "--yes"]).exit_code == 0
    assert calls == ["job123"]
```

Create `tests/ui/test_recorded_run.py`:

```python
import json
from pathlib import Path

from clef_tune.card import render_card
from clef_tune.config import load_config

RUN = Path(__file__).parent / "fixtures" / "recorded_run"
EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "support_tickets.yaml"
TRAIN_KEYS = {"event", "step", "loss", "lr", "tokens_per_sec", "eta_s", "projected_cost", "total_steps"}


def test_the_recorded_run_has_the_trainers_shapes():
    events = [json.loads(line) for line in (RUN / "metrics.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "baseline" and events[-1]["event"] == "done"
    assert all(set(e) == TRAIN_KEYS for e in events if e["event"] == "train")
    assert [e["step"] for e in events if e["event"] == "eval"] == [40, 80, 120]
    metrics = json.loads((RUN / "metrics.json").read_text())
    assert set(metrics) == {"baseline", "final", "calibration", "data", "run"}
    assert "## Confusion matrices" in render_card(metrics, load_config(EXAMPLE), "0.0.0")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_hf_jobs.py tests/test_hub.py tests/test_cli_jobs.py tests/ui/test_recorded_run.py -v`
Expected: FAIL with ImportErrors (`adapter_repo_of`, `quote_resume`, `pull_metrics`, `ResumeQuote`) and a missing fixture.

- [ ] **Step 3: Implement.**
  - **`quote_resume`:** move `resume_job`'s checks into a private `_resumable(info, timeout, api, token) -> str` that returns the timeout. Both functions call it.
  - **CLI:** prints `Resume {id} on {flavor} with timeout {timeout}: at most {money}`, or `cost unknown` without a price. It then `typer.confirm("Resume this job?", abort=True)` unless `--yes`.
  - **`scripts/make_recorded_run.py`:** seed `random.Random(7)`. Baseline logits are noisy near uniform, final logits mostly correct, and loss falls from about 1.1 to about 0.25 with ±3 % noise. Train events come every 10 steps, with `tokens_per_sec` near 380, `eta_s` counting down, and `projected_cost` near $0.34. Write both files with `indent=2` for the JSON.
  - **Generate the fixture:** run `uv run python scripts/make_recorded_run.py`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_hf_jobs.py tests/test_hub.py tests/test_cli_jobs.py tests/ui/test_recorded_run.py tests/torch/test_train.py -v`
Expected: all PASS. The torch test takes about a minute.

- [ ] **Step 5: Lint and commit** (`feat: give the monitor total steps, the adapter repo, resume quotes and a recorded run`).

---

### Task 6: Launch, runs, monitor and results handlers

**Files:**
- Create: `clef_tune/ui/runs.py`, `tests/ui/test_runs.py`
- Modify: `clef_tune/ui/services.py` (Jobs fields), `tests/ui/fakes.py` (fake Jobs answers replaying the recorded run), `tests/test_boundaries.py`

**Interfaces:**
- Consumes: Task 3 `user_token`, `SignInRequired`, `Session`; Task 4 `handlers.refresh`; Task 5 `JobStatus.adapter_repo`, `ResumeQuote`, `quote_resume`, `pull_metrics`; Plan 3 `JobRequest`, `LaunchPlan`, `LaunchedJob`, `LaunchError`; `hub.HubError`; `huggingface_hub.errors.HfHubHTTPError`.
- Produces, in `clef_tune.ui.services.Services`, these new fields with the real functions as defaults: `whoami: Callable[[str], str]` (token → namespace), `plan_launch`, `submit_launch`, `list_jobs`, `job_status`, `job_logs`, `cancel_job`, `quote_resume`, `resume_job`, `pull_metrics`. Every call passes `token=` explicitly.
- Produces, in `tests/ui/fakes.py`: `fake_services()` gains:
  - `jobs`: three fake `JobInfo`s, `job-a`/`run-a` RUNNING, `job-b`/`run-b` COMPLETED and `job-c`/`run-c` ERROR;
  - `job_status`: replays the recorded run. For RUNNING, the events stop after the step-70 `train` event. Each status's `adapter_repo` is `me/clef-tune-<run>`;
  - `pull_metrics`: the fixture's `metrics.json`;
  - `job_logs`: `["line 1", "<b>line 2</b>"]`;
  - `plan_launch`/`submit_launch`: a `SimpleNamespace` plan and job;
  - `quote_resume`: `ResumeQuote("job-c", "l4x1", "2h", 0.8)`;
  - `resume_job`: `SimpleNamespace(id="job-resumed", url=…)`;
  - `whoami`: `"me"`.

  Keyword overrides replace any `Services` field (for example `job_status=down`). `plan_warning="…"` makes the fake `plan_launch` call `warnings.warn` with that text, and `plan_error=exc` makes it raise.
- Produces, in `clef_tune.ui.runs`:
  - **Constants:** `LOG_TAIL = 200`, `POLL_SECONDS = 10`, `TERMINAL = ("COMPLETED", "CANCELED", "ERROR", "DELETED")`.
  - **`PendingLaunch`:** `@dataclass class PendingLaunch(plan: Any, fingerprint: str)`.
  - **`LaunchReview`:** `@dataclass class LaunchReview(summary: str, confirm_label: str, can_confirm: bool, timeout: str, error: str | None)`.
  - **`account(settings, services, *, oauth_token, pasted) -> str | None`:** the namespace, or `None` when not signed in.
  - **`default_adapter_repo(session, namespace: str) -> str`:** `f"{namespace}/clef-tune-{run_name}"`.
  - **`review_launch(session, settings, services, *, oauth_token, pasted, flavor: str | None, timeout: str, adapter_repo: str, delete_data: bool) -> LaunchReview`:**
    - Writes the config, then calls `services.plan_launch(config_path, JobRequest(flavor=flavor or None, timeout=timeout.strip() or None, keep_data=not delete_data, adapter_repo=adapter_repo.strip() or None), token=…, counter=…, prices=services.prices())`.
    - Records any `warnings` it raises into the summary.
    - Stores `session.pending = PendingLaunch(plan, draft.fingerprint())`.
    - `confirm_label` reads `f"Launch for up to ${plan.max_cost:.2f}"` (or `"Launch"` without a price).
  - **`confirm_launch(session, settings, services, *, oauth_token, pasted) -> str`:**
    - Pops `session.pending` first. With none: `"Nothing to launch; review the launch first."`.
    - With a stale fingerprint: `"The settings changed after the review; review the launch again."`.
    - Otherwise it submits, sets `session.job_id` and returns a success callout with the job link.
  - **`list_runs(settings, services, *, oauth_token, pasted) -> tuple[list[list[str]], str | None]`:** rows `[id, run, flavor, stage, created]`, plus an error message.
  - **`select_run(session, job_id: str) -> None`.**
  - **`ResultsView`:** `@dataclass class ResultsView(table: str, heatmaps: str, calibration: str, repo_link: str, snippet: str)`.
  - **`RunView`:** `@dataclass class RunView(badge: str, progress: str, stats: str, loss_chart: str, eval_chart: str, logs: str, can_cancel: bool, can_resume: bool, polling: bool, results: ResultsView | None, notice: str | None)`.
  - **`run_view_from(*, stage: str, events: list[dict], hours: float | None, cost: float | None, logs: list[str], metrics: dict | None, adapter_repo: str | None, state_columns: list[str] | None) -> RunView`:** shared with Task 7.
  - **`job_view(session, settings, services, *, oauth_token, pasted, with_logs: bool) -> RunView`.**
  - **`cancel_run(session, settings, services, *, oauth_token, pasted) -> str`.**
  - **`review_resume(session, settings, services, *, oauth_token, pasted, timeout: str) -> LaunchReview`.**
  - **`resume_run(session, settings, services, *, oauth_token, pasted, timeout: str) -> str`:** selects the new job.
  - **`results_view(metrics: dict, adapter_repo: str | None, state_columns: list[str] | None) -> ResultsView`.**
  - **`predictor_snippet(repo: str, state_columns: list[str] | None) -> str`.**

- [ ] **Step 1: Write the failing tests** `tests/ui/test_runs.py`:

```python
import pytest

from clef_tune.launch.hf_jobs import LaunchError
from clef_tune.ui import handlers, runs
from clef_tune.ui.session import Session
from clef_tune.ui.settings import Settings
from tests.ui.fakes import fake_services

LOCAL = Settings(space=False, max_upload_mb=200, local_training=False)
SPACE = Settings(space=True, max_upload_mb=200, local_training=False)
SIGNED_IN = {"oauth_token": None, "pasted": "hf_user"}


@pytest.fixture
def session():
    s = Session()
    handlers.example(s)
    yield s
    s.close()


def review(session, services, **overrides):
    values = {"flavor": "l4x1", "timeout": "", "adapter_repo": "", "delete_data": True} | overrides
    return runs.review_launch(session, LOCAL, services, **SIGNED_IN, **values)


def test_space_actions_need_sign_in_and_call_nothing(session, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_server")
    services, calls = fake_services()
    anonymous = {"oauth_token": None, "pasted": ""}
    assert "Sign in" in runs.review_launch(
        session, SPACE, services, **anonymous, flavor=None, timeout="", adapter_repo="", delete_data=True
    ).error
    assert "Sign in" in runs.list_runs(SPACE, services, **anonymous)[1]
    session.job_id = "job-a"
    assert "Sign in" in runs.job_view(session, SPACE, services, **anonymous, with_logs=True).notice
    assert [c for c in calls if c[0] not in ("counter", "prices")] == []


def test_review_shows_the_maximum_cost_and_warnings(session):
    services, calls = fake_services(plan_warning="could not check main on GitHub")
    result = review(session, services)
    assert result.can_confirm and result.confirm_label.startswith("Launch for up to $")
    assert "could not check main on GitHub" in result.summary
    request = next(c for c in calls if c[0] == "plan_launch")[1][1]
    assert (request.flavor, request.timeout, request.keep_data) == ("l4x1", None, False)


def test_a_launch_error_is_shown_and_nothing_is_pending(session):
    services, _ = fake_services(plan_error=LaunchError("Hugging Face Jobs needs pre-paid credits"))
    result = review(session, services)
    assert not result.can_confirm and "pre-paid credits" in result.error and session.pending is None


def test_a_stale_review_is_not_launched(session):
    services, calls = fake_services()
    review(session, services)
    session.draft.questions[0].instructions = "changed"
    message = runs.confirm_launch(session, LOCAL, services, **SIGNED_IN)
    assert "review the launch again" in message
    assert not [c for c in calls if c[0] == "submit_launch"]


def test_confirming_twice_launches_once(session):
    services, calls = fake_services()
    review(session, services)
    assert "job-new" in runs.confirm_launch(session, LOCAL, services, **SIGNED_IN)
    assert "review the launch first" in runs.confirm_launch(session, LOCAL, services, **SIGNED_IN)
    assert len([c for c in calls if c[0] == "submit_launch"]) == 1 and session.job_id == "job-new"


def test_a_running_job_shows_progress_charts_and_cost(session):
    services, _ = fake_services()
    session.job_id = "job-a"
    view = runs.job_view(session, LOCAL, services, **SIGNED_IN, with_logs=False)
    assert view.polling and view.can_cancel and not view.can_resume
    assert "Step 70 of 120" in view.progress and "<polyline" in view.loss_chart
    assert "ct-reference" in view.eval_chart and view.results is None


def test_a_completed_job_shows_results(session):
    services, calls = fake_services()
    session.job_id = "job-b"
    view = runs.job_view(session, LOCAL, services, **SIGNED_IN, with_logs=False)
    assert not view.polling and view.results is not None
    assert view.results.heatmaps.count("<svg") == 3
    assert 'Predictor.load("me/clef-tune-run-b")' in view.results.snippet
    runs.job_view(session, LOCAL, services, **SIGNED_IN, with_logs=False)
    assert len([c for c in calls if c[0] == "pull_metrics"]) == 1  # cached in the session


def test_log_tail_is_escaped(session):
    services, _ = fake_services()
    session.job_id = "job-a"
    logs = runs.job_view(session, LOCAL, services, **SIGNED_IN, with_logs=True).logs
    assert "&lt;b&gt;line 2&lt;/b&gt;" in logs and "<b>" not in logs


def test_a_failed_poll_keeps_the_last_view_and_polling(session):
    services, _ = fake_services()
    session.job_id = "job-a"
    good = runs.job_view(session, LOCAL, services, **SIGNED_IN, with_logs=False)

    def down(job_id, **kwargs):
        raise OSError("connection reset")

    failing, _ = fake_services(job_status=down)
    view = runs.job_view(session, LOCAL, failing, **SIGNED_IN, with_logs=False)
    assert view.polling and "connection reset" in view.notice and view.progress == good.progress


def test_resume_shows_its_cost_before_starting(session):
    services, calls = fake_services()
    session.job_id = "job-c"
    quote = runs.review_resume(session, LOCAL, services, **SIGNED_IN, timeout="")
    assert quote.confirm_label == "Resume for up to $1.60"
    assert not [c for c in calls if c[0] == "resume_job"]
    runs.resume_run(session, LOCAL, services, **SIGNED_IN, timeout="")
    assert session.job_id == "job-resumed"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/ui/test_runs.py -v`
Expected: FAIL with `ImportError` for `clef_tune.ui.runs`.

- [ ] **Step 3: Implement `runs.py`, the new `Services` fields and the fakes.**
  - **Errors:** every handler catches `SignInRequired`, `LaunchError`, `HubError`, `HfHubHTTPError`, `OSError` and `ValueError`. It shows `str(exc)` escaped in a callout; Gradio never shows a traceback.
  - **`job_view`:**
    - On failure it returns `session.last_view` with `notice` set and `polling=True`.
    - On success it stores the new view in `session.last_view`.
    - It fetches logs only when `with_logs` is true, and keeps the last `LOG_TAIL` lines.
  - **`run_view_from`:**
    - **Progress:** `step / total_steps` from the latest `train` event, labelled `f"Step {step} of {total_steps}"`. Before any event it is `None`, labelled "Waiting for the first steps".
    - **Stats:** elapsed time, cost so far, projected cost, tokens/s and ETA ("about 12 min left").
    - **Loss chart:** `{"loss": …}`.
    - **Eval chart:** `{"mean macro-F1": …, "exact match": …}`, with references from the `baseline` event.
    - **Buttons:** `can_cancel` while not terminal; `can_resume` for `ERROR` and `CANCELED`.
  - **Results:** shown when the stage is `COMPLETED` and an adapter repo is known. They are read once through `services.pull_metrics` and cached in `session.results[job_id]`.
  - **Snippet:** shows a `state` built from the draft's state columns, as `"col": "..."` placeholders, or `"..."` without them.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/ui tests/test_boundaries.py -v`
Expected: all PASS.

- [ ] **Step 5: Lint and commit** (`feat: add launch review, runs, monitor and results handlers`).

---

### Task 7: Train locally

**Files:**
- Create: `clef_tune/ui/local.py`, `tests/ui/test_local.py`
- Modify: `tests/test_boundaries.py`

**Interfaces:**
- Consumes: Task 3 `Session`; Task 4 `handlers.refresh`; Task 6 `runs.run_view_from`, `RunView`; `bundle.write_bundle`; `config.load_config`.
- Produces, in `clef_tune.ui.local`:
  - **`class LocalRunError(RuntimeError)`.**
  - **`LocalRun`:** `@dataclass class LocalRun(folder: Path, config_path: Path, output_dir: Path, log_path: Path, process: subprocess.Popen | None = None, canceled: bool = False)` with `stage() -> str`. The stage is `RUNNING` while the process lives, `COMPLETED` on exit 0, `CANCELED` after `stop`, and `ERROR` otherwise.
  - **`train_command(config_path: Path) -> list[str]`:** `[sys.executable, "-m", "clef_tune", "train", str(config_path)]`.
  - **`default_folder(session) -> str`:** `str(Path.cwd() / "runs" / run_name)`.
  - **`start_local(session, folder: str, *, command: Callable[[Path], list[str]] = train_command) -> LocalRun`:**
    - Without a validated `session.report` with no errors, raises `LocalRunError("fix the validation errors first")`.
    - Writes the bundle into `folder`.
    - Starts the process with stdout and stderr appended to `folder/train.log` and `cwd=folder`.
    - Registers the run in a module-level `RUNS: dict[Path, LocalRun]` and sets `session.local_folder`.
    - A folder that exists, is not empty, and is not a registered stopped run raises `LocalRunError("… already exists; choose another folder")`.
    - Restarting a registered, stopped run keeps its bundle, so `train` resumes from its checkpoints.
  - **`stop_local(folder: str) -> None`:** terminates the process and marks the run canceled.
  - **`local_view(session) -> RunView`:** events come from `output_dir/metrics.jsonl`, logs from the tail of `train.log`, metrics from `output_dir/final/metrics.json`, and `adapter_repo` is `None`.
  - **Training state lives here:** the subprocess is required because Accelerate's state is a per-process singleton (Plan 2 note). Session stores only the folder string.

- [ ] **Step 1: Write the failing tests** `tests/ui/test_local.py`:

```python
import sys
import time

import pytest

from clef_tune.ui import handlers, local
from clef_tune.ui.session import Session
from tests.ui.fakes import fake_services

FAKE_TRAIN = r"""
import json, pathlib, sys
from clef_tune.config import load_config
out = load_config(sys.argv[1]).output_dir(); out.mkdir(parents=True, exist_ok=True)
with open(out / "metrics.jsonl", "w") as f:
    f.write(json.dumps({"event": "train", "step": 5, "loss": 0.5, "lr": 1e-4, "tokens_per_sec": 9.0,
                        "eta_s": 1.0, "projected_cost": None, "total_steps": 10}) + "\n")
print("<i>trained</i>")
"""


def fake_command(config_path):
    return [sys.executable, "-c", FAKE_TRAIN, str(config_path)]


def wait(run, timeout=30):
    deadline = time.monotonic() + timeout
    while run.process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)


@pytest.fixture
def session():
    s = Session()
    handlers.example(s)
    handlers.refresh(s, fake_services()[0])
    yield s
    s.close()


def test_a_local_run_trains_in_a_subprocess_and_is_monitored(session, tmp_path):
    run = local.start_local(session, str(tmp_path / "run1"), command=fake_command)
    wait(run)
    view = local.local_view(session)
    assert run.stage() == "COMPLETED" and "Step 5 of 10" in view.progress
    assert "&lt;i&gt;trained" in view.logs and (tmp_path / "run1" / "config.yaml").exists()


def test_an_existing_folder_is_not_overwritten(session, tmp_path):
    (tmp_path / "taken").mkdir()
    (tmp_path / "taken" / "notes.txt").write_text("mine")
    with pytest.raises(local.LocalRunError, match="already exists"):
        local.start_local(session, str(tmp_path / "taken"), command=fake_command)


def test_stopping_marks_the_run_canceled(session, tmp_path):
    sleeper = lambda config: [sys.executable, "-c", "import time; time.sleep(60)"]  # noqa: E731
    run = local.start_local(session, str(tmp_path / "slow"), command=sleeper)
    local.stop_local(str(tmp_path / "slow"))
    wait(run)
    assert run.stage() == "CANCELED" and local.local_view(session).can_resume
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/ui/test_local.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `local.py`.** Keep a reference to the open log file on the `LocalRun` and close it when the process ends. Windows refuses to delete an open file, and the session folder cleanup would fail.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/ui/test_local.py tests/test_boundaries.py -v`
Expected: all PASS.

- [ ] **Step 5: Lint and commit** (`feat: train locally from the UI in a subprocess`).

---

### Task 8: Layout, theme, `clef-tune ui` and the browser test

**Files:**
- Create: `clef_tune/ui/theme.py`, `clef_tune/ui/style.css`, `clef_tune/ui/app.py`, `tests/ui/test_app.py`, `tests/ui/test_browser.py`
- Modify:
  - `clef_tune/cli.py`: an `ui` command.
  - `pyproject.toml`: add the `browser` pytest marker and `addopts = "-m 'not gpu and not hfjobs and not browser'"`.
  - `README.md`: a "Web UI" quickstart, first of the three quickstarts.
  - `tests/test_boundaries.py`: `clef_tune.ui.app` and `clef_tune.ui.theme` get a no-torch check that skips without gradio.

**Interfaces:**
- Consumes: everything in `clef_tune.ui`.
- Produces:
  - **`clef_tune.ui.theme.theme() -> gr.themes.Base`:**
    - zinc neutrals and an indigo primary;
    - `font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]` and `font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"]`;
    - `radius_size=gr.themes.sizes.radius_md`;
    - `.set(...)` for body and block backgrounds, borders and button colours in light and `_dark`.
  - **`clef_tune.ui.app`:**
    - `STYLE = Path(__file__).with_name("style.css")`;
    - `THEME_JS`: on load, applies `localStorage["clef-tune-theme"]` (`"dark"`/`"light"`) to `document.body.classList`;
    - `build_app(settings: Settings | None = None, services: Services | None = None) -> gr.Blocks`;
    - `launch(*, share: bool = False, port: int | None = None, settings: Settings | None = None, services: Services | None = None, prevent_thread_lock: bool = False) -> None`. It calls `sweep_stale()`, then `build_app(...).launch(theme=theme(), css_paths=[STYLE], js=THEME_JS, max_file_size=f"{settings.max_upload_mb}mb", share=share, server_port=port, footer_links=["gradio"], prevent_thread_lock=prevent_thread_lock)`.
  - **`clef-tune ui [--share] [--port N]`:** calls `launch`. Without gradio it exits 1 with `install the ui extra: uv pip install "clef-tune[ui] @ git+https://github.com/hersh-gupta/clef-tune"`.
  - **Element ids,** for tests and screenshots:
    - `ct-header`, `ct-sidebar`, `ct-step-0` … `ct-step-4`, `ct-estimate`;
    - `ct-data`, `ct-schema`, `ct-validate`, `ct-configure`, `ct-launch`;
    - `ct-launch-bundle`, `ct-launch-jobs`, `ct-launch-local`;
    - `ct-runs`, `ct-monitor`, `ct-results`, `ct-theme-toggle`.

**Layout (§9.1–9.2, built in `build_app`):**
- **Header row:**
  - a tuning-fork SVG mark and "clef-tune" with the subtitle "Fine-tune Clef on your own decisions";
  - a model `gr.Radio` with "Clef-Flash · 9B" and "Clef · 27B";
  - a theme toggle `gr.Button` whose `js` toggles `dark` and stores the choice;
  - on a Space only, `gr.LoginButton()` (fact 30).
- **Body row:**
  - **Sidebar:** a `gr.Column(elem_id="ct-sidebar", scale=0, min_width=240)` holds five step `gr.Button`s and the live estimate `gr.HTML`.
    - Each step button's `elem_classes` carries `ct-step`, `is-{state}` and `is-current`.
    - Its label is `"{n}  {name}"`, and an icon comes from CSS by state.
    - It is `interactive` only up to `session.reached`.
  - **Main column:** a `gr.Tabs(elem_id="ct-main-tabs")` with five `gr.Tab`s (ids 0–4). CSS hides the tab headers, and the sidebar and the Back / Next buttons switch tabs with `gr.Tabs(selected=i)`.
- **Data tab:**
  - `gr.File(file_types=[".csv", ".tsv", ".jsonl", ".ndjson", ".zip"])` and a "Load example" button;
  - a summary `gr.HTML`, chips `gr.HTML` and a `gr.Dataframe(interactive=False)`;
  - previous/next page buttons;
  - an empty state before upload.
- **Schema tab:**
  - id `gr.Dropdown`, state `gr.CheckboxGroup` and image `gr.CheckboxGroup`;
  - "Add a question from column" (`gr.Dropdown` + button);
  - the cards under `@gr.render(inputs=[session], triggers=[schema_version.change])`;
  - a prompt preview `gr.HTML` with a row `gr.Slider`.

  Each card has an id `gr.Textbox`, a type `gr.Radio`, instructions `gr.Textbox`, an options `gr.Dataframe(column_count=3, interactive=True)` (count column `static_columns=[2]`), the unknown-values `gr.HTML` and a Remove button. Every component has `key=f"{qid}-{part}"`. Text fields update on `.blur`/`.submit`, not on every keystroke. Structural edits (add, remove, type, id) bump `schema_version` so the cards re-render; text edits do not.
- **Validate tab:** tiles, issues, distributions and histogram `gr.HTML`.
- **Configure tab:**
  - a run name `gr.Textbox` and a preset `gr.Radio` with descriptions;
  - an estimate table `gr.HTML`;
  - `gr.Accordion("Advanced", open=False)`, built by looping over `ADVANCED_FIELDS`: `int`/`optional_int` → `gr.Number(precision=0)`, `float` → `gr.Number`, `bool` → `gr.Checkbox`, `choice` → `gr.Dropdown`, with the `info=` help;
  - a question-weights `gr.Dataframe`;
  - one error `gr.HTML`.
- **Launch tab:** `gr.Tabs` with:
  - **Download bundle:** `gr.DownloadButton`.
  - **HF Jobs:**
    - a token `gr.Textbox(type="password")` labelled "Hugging Face token (optional on a Space; needs job.write and repo.write)";
    - flavor `gr.Dropdown` (fitting flavors, recommended first and selected), timeout `gr.Textbox(placeholder="auto")`, adapter repo `gr.Textbox`;
    - `gr.Checkbox("Delete training data after a successful run", value=True)`;
    - a "Review launch" button, the review `gr.HTML`, and a confirm button made visible only when `can_confirm`. It is wired with `trigger_mode="once"`, `concurrency_limit=1` and `concurrency_id="launch"`.
  - **Train locally:** only when `settings.local_training`.

  Below the tabs come "Your runs" (`gr.Dataframe` + Refresh; `.select` → `select_run`) and the monitor:
  - badge, progress, stats, charts, a log `gr.Accordion` (`with_logs` follows its open state), Cancel, a Resume review and confirm, and a notice;
  - the results: a metrics table, heatmaps, calibration, a repo link, and `gr.Code(language="python")` for the snippet.

  `gr.Timer(runs.POLL_SECONDS, active=False)` ticks `job_view` or `local_view`. It is switched on by selecting or launching, and off when `polling` is false.
- **Session wiring:**
  - `session = gr.State(Session(), delete_callback=Session.close)`.
  - Handlers that take `oauth_token` declare a `gr.OAuthToken | None` parameter in their app-side wrapper and pass `.token`.
  - Every edit chains `.then(refresh_all, trigger_mode="always_last")`, which re-runs `handlers.refresh` and updates the sidebar, Validate and Configure. This is §9.2's "re-runs automatically".
- **CSS (`style.css`):**
  - The `--ct-*` tokens (zinc/indigo) sit on `:root`, and `.dark` overrides them.
  - Charts use `fill: var(--ct-accent)`, with `ct-series-0..3` from the accent and three neutrals.
  - Stepper states use these icons: ✓ done, ! warning, × error, ○ todo; current has an accent ring.
  - Cards, tiles, badges, callouts, empty states and the prompt preview are styled, with `.pp-state`, `.pp-question` and `.pp-option` highlights.
  - Below 900 px, `#ct-sidebar` becomes a horizontal step bar above the main column.
  - **Designed states:** every empty panel shows `fragments.empty_state`, not a blank box.
  - **Loading:** slow events (refresh, review, polling) use `show_progress="minimal"`, and the CSS restyles Gradio's progress overlay to match the theme.
  - **Errors:** errors are callouts.

- [ ] **Step 1: Write the failing tests** `tests/ui/test_app.py`:

```python
import pytest

gr = pytest.importorskip("gradio")

from typer.testing import CliRunner  # noqa: E402

from clef_tune import cli  # noqa: E402
from clef_tune.ui import app  # noqa: E402
from clef_tune.ui.settings import Settings  # noqa: E402
from tests.ui.fakes import fake_services  # noqa: E402


def ids(demo):
    return {getattr(block, "elem_id", None) for block in demo.blocks.values()}


def test_the_app_builds_without_calling_any_service():
    services, calls = fake_services()
    demo = app.build_app(Settings(space=False, max_upload_mb=200, local_training=True), services)
    assert {"ct-sidebar", "ct-launch-jobs", "ct-launch-local", "ct-results"} <= ids(demo)
    assert not demo.expects_oauth and calls == []


def test_space_mode_hides_local_training_and_signs_in_with_hugging_face():
    services, _ = fake_services()
    demo = app.build_app(Settings(space=True, max_upload_mb=200, local_training=True), services)
    assert "ct-launch-local" not in ids(demo) and demo.expects_oauth


def test_launch_passes_theme_css_and_the_upload_cap(monkeypatch):
    seen = {}
    monkeypatch.setattr(gr.Blocks, "launch", lambda self, **kwargs: seen.update(kwargs))
    monkeypatch.setattr(app, "sweep_stale", lambda: 0)
    app.launch(settings=Settings(space=True, max_upload_mb=50, local_training=False), services=fake_services()[0])
    assert seen["max_file_size"] == "50mb" and app.STYLE in seen["css_paths"]
    assert isinstance(seen["theme"], gr.themes.Base) and app.STYLE.exists()


def test_the_ui_command_launches(monkeypatch):
    seen = {}
    monkeypatch.setattr(app, "launch", lambda **kwargs: seen.update(kwargs))
    result = CliRunner().invoke(cli.app, ["ui", "--port", "7861"])
    assert result.exit_code == 0 and seen["port"] == 7861 and seen["share"] is False
```

Create `tests/ui/test_browser.py` with `pytestmark = pytest.mark.browser`. Its fixture:
- calls `app.launch(services=fake_services()[0], settings=Settings(False, 200, False), port=free_port, prevent_thread_lock=True)`;
- opens headless chromium with `playwright.sync_api`.

`test_the_example_goes_from_data_to_results`:
1. Clicks "Load example" and expects "200 rows".
2. Clicks `#ct-step-1` and expects the "department" card.
3. Goes to Validate and expects "Train rows".
4. Goes to Configure and expects "l4x1".
5. Goes to Launch → HF Jobs, types a token, clicks "Review launch" and expects "Launch for up to $".
6. Clicks the `run-b` row under Your runs and expects `#ct-results svg`.
7. Asserts the browser console logged no errors.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/ui/test_app.py -v`
Expected: FAIL with `ImportError` for `clef_tune.ui.app`.

- [ ] **Step 3: Implement `theme.py`, `style.css`, `app.py` and the `ui` command** following the layout above. In app wrappers, keep each wrapper to mapping view fields onto components. Logic goes in the handlers.

- [ ] **Step 4: Run the unit tests to verify they pass**

Run: `uv run pytest tests/ui tests/test_boundaries.py tests/test_cli.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the browser test**

Run: `uv run pytest tests/ui/test_browser.py -m browser -v`
Expected: PASS. If chromium is missing or stale, run `uv run playwright install chromium` and rerun.

- [ ] **Step 6: Try it by hand.** Run `uv run clef-tune ui`, go through the example in the Browser pane, and fix any wiring bug with a test first.

- [ ] **Step 7: Run the full suite**

Run: `uv run pytest`
Expected: all pass, except the existing POSIX-only skips.

- [ ] **Step 8: Lint and commit** (`feat: add the themed Gradio layout and clef-tune ui`).

---

### Task 9: Hugging Face Space deployment

**Files:**
- Create: `deploy/space/app.py`, `deploy/space/README.md`, `scripts/prepare_space.py`, `tests/ui/test_space.py`
- Modify: `README.md` (deploying the Space), `.gitignore` (`build/`)

**Interfaces:**
- Consumes: `clef_tune.ui.app.launch`; `clef_tune.launch.source.pin_commit`, `Source`, `source_repo`.
- Produces:
  - **`deploy/space/app.py`:** `from clef_tune.ui.app import launch` then `launch()`.
  - **`deploy/space/README.md`:** front matter with `title: clef-tune`, `emoji`, `colorFrom: indigo`, `colorTo: gray`, `sdk: gradio`, and `sdk_version` equal to the locked gradio (`6.30.0`). It also sets `python_version: "3.12"`, `app_file: app.py`, `hf_oauth: true`, `hf_oauth_expiration_minutes: 480`, `hf_oauth_scopes: [read-repos, write-repos, manage-repos, jobs]` and `pinned: false`. The body is a short description and the token note from fact 32.
  - **`scripts/prepare_space.py OUT_DIR [--ref REF]`:**
    - `prepare(out_dir: Path, ref: str | None, *, fetch=github_json) -> Path`.
    - It copies `deploy/space/*` into `OUT_DIR` and writes `requirements.txt` with the single line `clef-tune[ui] @ git+{repo}@{sha}`.
    - `sha` comes from `pin_commit(Source(source_repo(), ref or running commit))`, so an unpushed commit is refused.
    - The Space installs from git, so PEP 610 records the commit and jobs launched from the Space install that exact commit (Plan 3 `resolve_source`).

- [ ] **Step 1: Write the failing tests** `tests/ui/test_space.py`:

```python
from importlib.metadata import version
from pathlib import Path

import yaml

from scripts.prepare_space import prepare

SPACE = Path(__file__).resolve().parents[2] / "deploy" / "space"
SHA = "0123456789abcdef0123456789abcdef01234567"


def front_matter():
    text = (SPACE / "README.md").read_text(encoding="utf-8")
    return yaml.safe_load(text.split("---")[1])


def test_the_space_runs_the_tested_gradio_with_oauth():
    meta = front_matter()
    assert meta["sdk"] == "gradio" and meta["sdk_version"] == version("gradio")
    assert meta["hf_oauth"] is True and {"write-repos", "manage-repos", "jobs"} <= set(meta["hf_oauth_scopes"])
    assert "launch()" in (SPACE / "app.py").read_text()


def test_prepare_pins_the_requirement_to_a_pushed_commit(tmp_path):
    out = prepare(tmp_path / "space", "main", fetch=lambda url: {"sha": SHA})
    assert (out / "requirements.txt").read_text().strip() == (
        f"clef-tune[ui] @ git+https://github.com/hersh-gupta/clef-tune@{SHA}"
    )
    assert (out / "app.py").exists() and (out / "README.md").exists()
```

(Add `scripts/__init__.py` if `scripts` is not importable as a package.)

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/ui/test_space.py -v`
Expected: FAIL with a missing module or file.

- [ ] **Step 3: Implement the files.** Then add a README section "Deploy the web UI as a Space":
  1. Push your commits.
  2. Run `python scripts/prepare_space.py build/space`.
  3. Run `hf upload <you>/clef-tune build/space . --repo-type space`.

  The section also says visitors sign in with Hugging Face, and that launching jobs may need a pasted token with job.write and repo.write (fact 32).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/ui/test_space.py -v`
Expected: PASS.

- [ ] **Step 5: Lint and commit** (`feat: add the Hugging Face Space deployment`).

- [ ] **Step 6 (only with the user's explicit go-ahead; it publishes a public Space): live check.** Prepare from the pushed branch head and upload to a Space the user names. Then confirm:
  - the page loads;
  - "Sign in with Hugging Face" works;
  - Your runs lists the user's jobs with the OAuth token.

  Record in the roadmap whether the OAuth token can stage a launch to the bucket (fact 32). Do not launch a paid job without a second, separate go-ahead.

---

### Task 10: Screenshots and refinement with the user (§9.5)

**Files:**
- Create: `scripts/screenshots.py`
- Modify: `.gitignore` (`screenshots/`), and anything under `clef_tune/ui/` the feedback touches

**Interfaces:**
- Consumes: `clef_tune.ui.app.launch`, `tests.ui.fakes.fake_services` (the recorded run), Playwright.
- Produces: `scripts/screenshots.py [--out screenshots] [--widths 1440 1024] [--themes light dark]`.
  - It launches the app with fake services on a free port. For each width and theme it captures, as full-page PNGs named `{width}-{theme}-{nn}-{view}.png`:
    1. Data (empty);
    2. Data (example loaded);
    3. Schema;
    4. Validate;
    5. Configure (advanced closed);
    6. Configure (advanced open);
    7. Launch → Download;
    8. Launch → HF Jobs after review;
    9. Monitor (running run-a);
    10. Results (completed run-b).
  - Dark mode is set with `?__theme=dark`.
  - It writes `index.html`, a contact sheet of all captures grouped by view.

- [ ] **Step 1: Write `scripts/screenshots.py`,** reusing the browser test's launch helper, and run it: `uv run python scripts/screenshots.py`. Expected: 40 PNGs and `screenshots/index.html`.

- [ ] **Step 2: Check the captures yourself first.** Look for:
  - clipped text, overflow at 1024 px, and unreadable dark-mode contrast;
  - default-looking Gradio pieces;
  - empty or error states that look broken.

  Fix them before showing the user.

- [ ] **Step 3: Present round 1.** Publish the contact sheet as a private Artifact, and send the 1440-light set with SendUserFile. Ask for feedback per view.

- [ ] **Step 4: Apply the feedback.** Changes to logic get a failing test first. Re-run `uv run pytest` and the browser test, recapture, and present the next round.

- [ ] **Step 5: Repeat until the user signs off.** Commit after each round (`style: UI refinement round N`).

- [ ] **Step 6: Wrap up.** Update the roadmap with Plan 4's interfaces and any facts learned. Then use superpowers:finishing-a-development-branch.
