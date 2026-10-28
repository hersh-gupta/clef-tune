"""The Gradio layout (spec §9): a header, a step sidebar, four step panels, the Launch step's
tabs, the run monitor and the results. The logic lives in handlers.py, runs.py and local.py;
this module only maps their views onto components and wires the events."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from clef_tune.ui import fragments, handlers, local, runs
from clef_tune.ui.services import Services
from clef_tune.ui.session import DEFAULT_MODEL, Session, sweep_stale
from clef_tune.ui.settings import Settings
from clef_tune.ui.theme import theme

STYLE = Path(__file__).with_name("style.css")
THEME_JS = """() => {
  try {
    const saved = localStorage.getItem("clef-tune-theme");
    const forced = new URLSearchParams(window.location.search).get("__theme");
    if (!forced && saved === "dark") document.body.classList.add("dark");
    if (!forced && saved === "light") document.body.classList.remove("dark");
  } catch (error) {}
}"""
TOGGLE_JS = """() => {
  const dark = document.body.classList.toggle("dark");
  try { localStorage.setItem("clef-tune-theme", dark ? "dark" : "light"); } catch (error) {}
}"""
MARK = (
    '<svg class="ct-mark" viewBox="0 0 36 36" aria-hidden="true"><rect width="36" height="36" '
    'rx="9"/><path d="M12 7v9.5a6 6 0 0 0 12 0V7M18 22.5V30" fill="none" stroke-width="2.6" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
)
BRAND = (
    f'<div class="ct-brand">{MARK}<div><div class="ct-name">clef-tune</div>'
    '<div class="ct-tagline">Fine-tune Clef on your own decisions</div></div></div>'
)
INTROS = (
    (
        "Data and questions",
        "Upload labeled examples as a CSV, TSV or JSONL file, or a zip with the file and its "
        "images. Then say what each column is for: what Clef reads, and what it learns to answer.",
    ),
    (
        "Checks",
        "The checks run again after every change. Errors block training; warnings are worth "
        "a look.",
    ),
    (
        "Training settings",
        "Pick a preset, or open Advanced settings for every option. The estimate updates as "
        "you go.",
    ),
    (
        "Launch",
        "Train on Hugging Face Jobs, download a bundle to train anywhere, or follow a run you "
        "started earlier.",
    ),
)
OPTION_COLUMNS = ["Option", "Description", "Labels"]
QUESTION_TYPES = [("Choice", "choice"), ("Score", "score"), ("Yes / no", "noul")]
TOKEN_INFO = "Needs job.write and repo.write. Used for this browser session only and never stored."
RUN_COLUMNS = ["Job", "Run", "Hardware", "Stage", "Created"]
LAUNCH_STEP = len(handlers.STEPS) - 1


def _html(value: str = "", **kwargs: Any) -> gr.HTML:
    return gr.HTML(value, elem_classes=["ct-html-flush"], **kwargs)


def _callout(kind: str, text: str | None) -> str:
    return fragments.callout(kind, text) if text else ""  # type: ignore[arg-type]


def _intro(step: int) -> str:
    title, text = INTROS[step]
    return (
        f'<div class="ct-intro"><h2>{fragments.esc(title)}</h2><p>{fragments.esc(text)}</p></div>'
    )


def _oauth(token: gr.OAuthToken | None) -> str | None:
    return token.token if token is not None else None


def _field_component(spec: handlers.FieldSpec) -> gr.components.Component:
    if spec.kind in ("int", "optional_int"):
        return gr.Number(label=spec.label, info=spec.help, precision=0)
    if spec.kind == "float":
        return gr.Number(label=spec.label, info=spec.help)
    if spec.kind == "bool":
        return gr.Checkbox(label=spec.label, info=spec.help)
    return gr.Dropdown(list(spec.choices), label=spec.label, info=spec.help)


def build_app(settings: Settings | None = None, services: Services | None = None) -> gr.Blocks:
    settings = settings or Settings.from_env()
    services = services or Services()
    # Never on a Space, whatever the settings say: visitors must not train on its hardware.
    local_training = settings.local_training and not settings.space
    blank = Session()  # what a fresh browser session sees; never written to

    with gr.Blocks(title="clef-tune", fill_width=True) as demo:
        session = gr.State(Session(), delete_callback=Session.close)
        current = gr.State(0)
        columns_version = gr.State(0)  # bumped to re-render the column roles
        selected = gr.State(None)  # the question the editor shows
        data_page = gr.State(0)
        watching = gr.State("job")  # what the monitor follows: "job" or "local"
        logs_open = gr.State(False)

        # ------------------------------------------------------------------ header
        with gr.Row(elem_id="ct-header", equal_height=True):
            _html(BRAND, elem_id="ct-brand")
            model = gr.Radio(
                [(label, model_id) for model_id, label in handlers.MODELS.items()],
                value=DEFAULT_MODEL,
                show_label=False,
                container=False,
                elem_id="ct-model",
            )
            theme_button = gr.Button(
                "Light / dark", size="sm", variant="secondary", elem_id="ct-theme-toggle", scale=0
            )
            if settings.space:
                gr.LoginButton(size="sm", scale=0)

        with gr.Row(elem_id="ct-body"):
            # -------------------------------------------------------------- sidebar
            with gr.Column(elem_id="ct-sidebar", scale=0, min_width=220):
                _html('<div class="ct-sidebar-title">Steps</div>')
                steps = [
                    gr.Button(
                        f"{index + 1}   {name}",
                        elem_id=f"ct-step-{index}",
                        elem_classes=["ct-step", "is-todo"] + (["is-current"] if not index else []),
                        interactive=index in (0, LAUNCH_STEP),
                    )
                    for index, name in enumerate(handlers.STEPS)
                ]
                estimate = _html(handlers.sidebar_estimate(blank), elem_id="ct-estimate")

            with gr.Column(elem_id="ct-main", scale=1):
                panels = []
                navs = []

                def nav_row(step: int) -> None:
                    with gr.Row(elem_classes=["ct-nav"]):
                        back = gr.Button("← Back", variant="secondary", visible=step > 0)
                        forward = gr.Button(
                            f"Next: {handlers.STEPS[step + 1]} →" if step < LAUNCH_STEP else "Next",
                            variant="primary",
                            visible=step < LAUNCH_STEP,
                        )
                    navs.append((step, back, forward))

                # ---------------------------------------------------------- 1 data
                with gr.Column(elem_id="ct-data", elem_classes=["ct-panel"]) as panel:
                    panels.append(panel)
                    _html(_intro(0))
                    with gr.Row(equal_height=True, elem_id="ct-upload-row") as upload_row:
                        upload = gr.File(
                            label="Data file",
                            file_types=[".csv", ".tsv", ".jsonl", ".ndjson", ".zip"],
                            elem_id="ct-upload",
                            height=120,
                            scale=4,
                        )
                        with gr.Column(scale=1, min_width=200, elem_id="ct-example-box"):
                            example_button = gr.Button("Try the example", elem_id="ct-example")
                            _html(
                                '<p class="ct-muted">200 synthetic support tickets with three '
                                "questions: department, urgency and outage.</p>"
                            )
                    data_error = _html()
                    with gr.Row(
                        visible=False, equal_height=True, elem_id="ct-file-row"
                    ) as file_row:
                        file_bar = _html()
                        replace_button = gr.Button("Replace file", size="sm", scale=0)
                    with gr.Accordion(
                        "Rows", open=False, visible=False, elem_id="ct-rows"
                    ) as rows_box:
                        data_table = gr.Dataframe(
                            interactive=False, max_height=420, elem_id="ct-data-table"
                        )
                        with gr.Row(equal_height=True, elem_classes=["ct-pager"]):
                            previous_page = gr.Button("← Previous", size="sm", scale=0)
                            page_label = _html()
                            next_page = gr.Button("Next →", size="sm", scale=0)
                    with gr.Column(visible=False, elem_id="ct-schema") as schema_area:
                        _html(
                            '<h3 class="ct-section-title">Columns</h3><p class="ct-muted">'
                            "State columns are what Clef reads for each row. Each question column "
                            "becomes a question Clef learns to answer from that column's labels."
                            "</p>"
                        )
                        schema_error = _html()
                        _html(
                            '<div class="ct-col-head"><span>Column and sample values</span>'
                            "<span>Role</span></div>"
                        )
                        with gr.Column(elem_id="ct-columns"):
                            gr.render(inputs=[session], triggers=[columns_version.change])(
                                lambda s: column_list(s)
                            )
                        _html('<h3 class="ct-section-title">Questions</h3>')
                        question_pills = gr.Radio(
                            [], show_label=False, container=False, elem_id="ct-pills"
                        )
                        no_questions = _html()
                        with gr.Group(
                            visible=False, elem_id="ct-question", elem_classes=["ct-card"]
                        ) as editor:
                            with gr.Row(equal_height=True):
                                question_id = gr.Textbox(label="Question ID", scale=1)
                                question_type = gr.Radio(QUESTION_TYPES, label="Type", scale=2)
                            question_text = gr.Textbox(
                                label="What Clef is asked",
                                placeholder="Without it, Clef sees only the question ID.",
                            )
                            question_options = gr.Dataframe(
                                headers=OPTION_COLUMNS,
                                datatype=["str", "str", "number"],
                                type="array",
                                interactive=True,
                                static_columns=[2],
                                label="Options (Labels counts the rows with each answer)",
                                elem_id="ct-options",
                            )
                            question_unknown = _html()
                        with gr.Accordion("Prompt preview", open=False, elem_id="ct-preview"):
                            _html(
                                '<div class="ct-legend-row"><span class="is-state">state</span>'
                                '<span class="is-question">question</span>'
                                '<span class="is-option">options</span></div>'
                            )
                            preview_row = gr.Slider(1, 2, value=1, step=1, label="Row")
                            preview = _html()
                    nav_row(0)

                # ---------------------------------------------------------- 2 validate
                with gr.Column(
                    elem_id="ct-validate", elem_classes=["ct-panel"], visible=False
                ) as panel:
                    panels.append(panel)
                    _html(_intro(1))
                    validate_tiles = _html()
                    validate_issues = _html(handlers.validate_view(blank).issues)
                    validate_dists = _html()
                    validate_hist = _html()
                    nav_row(1)

                # ---------------------------------------------------------- 3 configure
                with gr.Column(
                    elem_id="ct-configure", elem_classes=["ct-panel"], visible=False
                ) as panel:
                    panels.append(panel)
                    _html(_intro(2))
                    config_error = _html()
                    with gr.Row():
                        run_name = gr.Textbox(
                            label="Run name", info="Names the job and the adapter repo.", scale=1
                        )
                        preset = gr.Radio(
                            [(label, key) for key, label in handlers.PRESET_LABELS.items()],
                            value="balanced",
                            label="Preset",
                            scale=2,
                        )
                    preset_help = _html()
                    _html(
                        '<h3 class="ct-section-title">Estimated time and cost on Hugging Face '
                        "Jobs</h3>"
                    )
                    estimate_table = _html()
                    with gr.Accordion("Advanced settings", open=False, elem_id="ct-advanced"):
                        fields: list[gr.components.Component] = []
                        specs = handlers.ADVANCED_FIELDS
                        for chunk in (specs[i : i + 3] for i in range(0, len(specs), 3)):
                            with gr.Row():
                                fields.extend(_field_component(spec) for spec in chunk)
                        weights = gr.Dataframe(
                            headers=["Question", "Weight"],
                            datatype=["str", "number"],
                            type="array",
                            interactive=True,
                            static_columns=[0],
                            label="Question weights (each question's share of the loss)",
                        )
                    nav_row(2)

                # ---------------------------------------------------------- 4 launch
                with gr.Column(
                    elem_id="ct-launch", elem_classes=["ct-panel"], visible=False
                ) as panel:
                    panels.append(panel)
                    _html(_intro(3))
                    with gr.Tabs(elem_id="ct-launch-tabs"):
                        with gr.Tab("Hugging Face Jobs", elem_id="ct-launch-jobs"):
                            account_html = _html()
                            token_box = gr.Textbox(
                                label="Hugging Face token",
                                type="password",
                                placeholder="hf_…",
                                info=TOKEN_INFO,
                            )
                            with gr.Row():
                                flavor = gr.Dropdown(
                                    label="Hardware",
                                    info="Hardware that fits this config, cheapest first.",
                                    choices=[],
                                    scale=2,
                                )
                                timeout = gr.Textbox(
                                    label="Time limit",
                                    placeholder="Automatic",
                                    info="Like 90m or 2h. The job stops there, which caps "
                                    "the cost.",
                                    scale=1,
                                )
                                repo = gr.Textbox(
                                    label="Adapter repo",
                                    placeholder="you/clef-tune-<run name>",
                                    info="Created private if it does not exist.",
                                    scale=2,
                                )
                            delete_data = gr.Checkbox(
                                True, label="Delete the training data after a successful run"
                            )
                            review_button = gr.Button("Review launch", elem_id="ct-review")
                            review_html = _html()
                            confirm_button = gr.Button(
                                "Launch", variant="primary", visible=False, elem_id="ct-confirm"
                            )
                            launch_result = _html()
                        with gr.Tab("Download bundle", elem_id="ct-launch-bundle"):
                            _html(
                                fragments.callout(
                                    "info",
                                    "A zip with train.jsonl, eval.jsonl and a self-contained "
                                    "config.yaml. Train it on any GPU machine with "
                                    "clef-tune train config.yaml.",
                                )
                            )
                            bundle_button = gr.Button("Prepare the bundle")
                            bundle_file = gr.DownloadButton(
                                "Download the bundle", variant="primary", visible=False
                            )
                            bundle_message = _html()
                        if local_training:
                            with gr.Tab("Train locally", elem_id="ct-launch-local"):
                                _html(
                                    fragments.callout(
                                        "info",
                                        "Runs clef-tune train on this machine's GPU. Stopping "
                                        "keeps the checkpoints; starting again resumes from "
                                        "them.",
                                    )
                                )
                                folder = gr.Textbox(
                                    label="Folder",
                                    info="The bundle, checkpoints and adapter go here.",
                                )
                                start_button = gr.Button("Start training", variant="primary")
                                local_message = _html()

                    _html('<h3 class="ct-section-title">Your runs</h3>')
                    with gr.Column(elem_id="ct-runs"):
                        with gr.Row(equal_height=True):
                            runs_message = _html()
                            refresh_runs = gr.Button("Refresh", size="sm", scale=0)
                        runs_table = gr.Dataframe(
                            headers=RUN_COLUMNS, interactive=False, max_height=260
                        )

                    with gr.Column(elem_id="ct-monitor", visible=False) as monitor:
                        monitor_badge = _html()
                        monitor_notice = _html()
                        monitor_progress = _html()
                        monitor_stats = _html()
                        with gr.Row():
                            loss_chart = _html()
                            eval_chart = _html()
                        with gr.Accordion("Logs", open=False) as logs_box:
                            monitor_logs = _html()
                        with gr.Row(equal_height=True):
                            cancel_button = gr.Button(
                                "Cancel the run", variant="stop", size="sm", visible=False
                            )
                            resume_timeout = gr.Textbox(
                                label="New time limit",
                                placeholder="Same as before",
                                visible=False,
                                scale=1,
                            )
                            resume_button = gr.Button("Review resume", size="sm", visible=False)
                            resume_confirm = gr.Button(
                                "Resume", variant="primary", size="sm", visible=False
                            )
                        resume_html = _html()
                        action_message = _html()

                    with gr.Column(elem_id="ct-results", visible=False) as results:
                        _html('<h3 class="ct-section-title">Results on the eval set</h3>')
                        results_table = _html()
                        _html('<h3 class="ct-section-title">Confusion matrices</h3>')
                        results_heatmaps = _html()
                        _html('<h3 class="ct-section-title">Calibration</h3>')
                        results_calibration = _html()
                        results_link = _html()
                        snippet = gr.Code(
                            language="python", label="Use the adapter", interactive=False
                        )
                    nav_row(3)

        timer = gr.Timer(runs.POLL_SECONDS, active=False)

        # ------------------------------------------------------------------ views → outputs
        chrome_out = [*steps, estimate]
        data_out = [
            data_error,
            upload_row,
            file_row,
            file_bar,
            rows_box,
            data_table,
            page_label,
            data_page,
            schema_area,
        ]
        pager_out = [data_table, page_label, data_page]
        schema_out = [
            schema_error,
            columns_version,
            question_pills,
            selected,
            no_questions,
            editor,
            question_id,
            question_type,
            question_text,
            question_options,
            question_unknown,
            preview_row,
            preview,
        ]
        validate_out = [validate_tiles, validate_issues, validate_dists, validate_hist]
        config_out = [
            config_error,
            run_name,
            preset,
            preset_help,
            *fields,
            weights,
            estimate_table,
        ]
        refresh_out = [*validate_out, *config_out, flavor, *chrome_out]
        monitor_out = [
            monitor,
            monitor_badge,
            monitor_notice,
            monitor_progress,
            monitor_stats,
            loss_chart,
            eval_chart,
            monitor_logs,
            cancel_button,
            resume_timeout,
            resume_button,
            resume_confirm,
            results,
            results_table,
            results_heatmaps,
            results_calibration,
            results_link,
            snippet,
            timer,
        ]

        def chrome(s: Session, step: int) -> list:
            states = handlers.step_states(s)
            buttons = [
                gr.Button(
                    elem_classes=["ct-step", f"is-{state}"]
                    + (["is-current"] if index == step else []),
                    interactive=index <= s.reached or index == LAUNCH_STEP,
                )
                for index, state in enumerate(states)
            ]
            return [*buttons, handlers.sidebar_estimate(s)]

        def show_page(view: handlers.DataView) -> list:
            table = pd.DataFrame(view.rows, columns=view.headers) if view.headers else None
            label = f'<div class="ct-muted">{fragments.esc(view.page_label)}</div>'
            return [gr.Dataframe(value=table), label, view.page]

        def show_data(view: handlers.DataView) -> list:
            loaded = bool(view.headers)
            return [
                _callout("error", view.error),
                gr.Row(visible=not loaded or bool(view.error)),
                gr.Row(visible=loaded),
                view.summary if loaded else "",
                gr.Accordion(visible=loaded),
                *show_page(view),
                gr.Column(visible=loaded),
            ]

        def show_schema(
            s: Session,
            version: int,
            chosen: str | None,
            *,
            bump: bool,
            error: str | None = None,
            row: int = 0,
        ) -> list:
            view = handlers.schema_view(s, row)
            ids = [card.id for card in view.cards]
            chosen = chosen if chosen in ids else (ids[0] if ids else None)
            card = next((c for c in view.cards if c.id == chosen), None)
            rows = max(len(s.table.rows) if s.table else 1, 2)
            empty = (
                fragments.empty_state(
                    "No questions yet", "Set a column's role to Question to add one."
                )
                if s.table is not None and not ids
                else ""
            )
            options = (
                pd.DataFrame(card.options, columns=OPTION_COLUMNS)
                if card
                else pd.DataFrame(columns=OPTION_COLUMNS)
            )
            return [
                _callout("error", error or view.error),
                version + 1 if bump else gr.skip(),
                gr.Radio(choices=ids, value=chosen),
                chosen,
                empty,
                gr.Group(visible=card is not None),
                gr.Textbox(
                    value=card.id if card else "",
                    info=f"Labels come from the column {card.column}." if card else None,
                ),
                card.type if card else None,
                card.instructions if card else "",
                gr.Dataframe(
                    value=options,
                    static_columns=[0, 2] if card and card.type == "noul" else [2],
                ),
                card.unknown if card else "",
                gr.Slider(minimum=1, maximum=rows, value=min(row + 1, rows), step=1),
                view.preview,
            ]

        def show_validate(s: Session) -> list:
            view = handlers.validate_view(s)
            dists = (
                f'<h3 class="ct-section-title">Labels per question</h3>{view.distributions}'
                if view.distributions
                else ""
            )
            hist = (
                '<h3 class="ct-section-title">Prompt length in tokens</h3>'
                f'<div class="ct-chart-card">{view.histogram}</div>'
                if view.histogram
                else ""
            )
            return [view.tiles, view.issues, dists, hist]

        def show_config(s: Session, error: str | None = None) -> list:
            view = handlers.configure_view(s)
            if s.draft is None:
                return [gr.skip()] * len(config_out[:-1]) + [view.estimates]
            values = [view.values.get(spec.path, gr.skip()) for spec in handlers.ADVANCED_FIELDS]
            help_html = (
                f'<p class="ct-muted">{fragments.esc(handlers.PRESET_HELP[view.preset])}</p>'
            )
            return [
                _callout("error", error or view.error),
                s.draft.run_name,
                view.preset,
                help_html,
                *values,
                pd.DataFrame(view.question_weights, columns=["Question", "Weight"]),
                view.estimates,
            ]

        def show_flavors(s: Session, chosen: str | None) -> gr.Dropdown:
            fitting = [row for row in s.estimates if row.fits]
            choices = []
            for row in fitting:
                cost = (
                    "" if row.cost_low is None else f" · ${row.cost_low:.2f}–${row.cost_high:.2f}"
                )
                choices.append((f"{row.flavor}{cost}", row.flavor))
            names = [row.flavor for row in fitting]
            value = chosen if chosen in names else (names[0] if names else None)
            return gr.Dropdown(choices=choices, value=value)

        def show_run(view: runs.RunView, kind: str = "job") -> list:
            res = view.results
            shown = bool(view.badge or view.notice)
            loss = (
                f'<div class="ct-chart-card"><h4>Training loss</h4>{view.loss_chart}</div>'
                if view.loss_chart
                else ""
            )
            evals = (
                f'<div class="ct-chart-card"><h4>Eval score</h4>{view.eval_chart}</div>'
                if view.eval_chart
                else ""
            )
            link = (
                f'<p class="ct-muted">Adapter: {res.repo_link}</p>' if res and res.repo_link else ""
            )
            return [
                gr.Column(visible=shown),
                view.badge,
                view.notice or "",
                view.progress,
                view.stats,
                loss,
                evals,
                view.logs,
                gr.Button(visible=view.can_cancel),
                gr.Textbox(visible=view.can_resume and kind == "job"),
                gr.Button(
                    "Start again" if kind == "local" else "Review resume",
                    visible=view.can_resume,
                ),
                gr.Button(visible=False),
                gr.Column(visible=res is not None),
                res.table if res else "",
                res.heatmaps if res else "",
                res.calibration if res else "",
                link,
                res.snippet if res else "",
                gr.Timer(active=view.polling),
            ]

        def follow(s: Session, kind: str, pasted: str, logs: bool, token: str | None) -> list:
            if kind == "local":
                return show_run(local.local_view(s), "local")
            view = runs.job_view(
                s, settings, services, oauth_token=token, pasted=pasted, with_logs=logs
            )
            return show_run(view, "job")

        # ------------------------------------------------------------------ navigation
        def go(s: Session, step: int) -> list:
            s.reached = max(s.reached, step) if step != LAUNCH_STEP or s.table else s.reached
            visible = [gr.Column(visible=index == step) for index in range(len(panels))]
            return [*visible, step, *chrome(s, step)]

        nav_out = [*panels, current, *chrome_out]
        for index, button in enumerate(steps):
            button.click(
                lambda s, step=index: go(s, step), [session], nav_out, show_progress="hidden"
            )
        for step, back, forward in navs:
            back.click(lambda s, step=step: go(s, step - 1), [session], nav_out)
            forward.click(lambda s, step=step: go(s, step + 1), [session], nav_out)

        def refresh_all(s: Session, step: int, chosen: str | None) -> list:
            handlers.refresh(s, services)
            return [
                *show_validate(s),
                *show_config(s),
                show_flavors(s, chosen),
                *chrome(s, step),
            ]

        def after(dependency: Any) -> None:
            dependency.then(
                refresh_all,
                [session, current, flavor],
                refresh_out,
                trigger_mode="always_last",
                show_progress="minimal",
            )

        # ------------------------------------------------------------------ data events
        def loaded(s: Session, view: handlers.DataView, version: int, model_id: str) -> list:
            if s.draft is not None:
                handlers.set_model(s, model_id)
            return [*show_data(view), *show_schema(s, version, None, bump=True)]

        def on_upload(s: Session, path: str | None, version: int, model_id: str) -> list:
            return loaded(s, handlers.upload(s, path, settings), version, model_id)

        def on_example(s: Session, version: int, model_id: str) -> list:
            return loaded(s, handlers.example(s), version, model_id)

        loaded_out = [*data_out, *schema_out]
        after(upload.upload(on_upload, [session, upload, columns_version, model], loaded_out))
        after(example_button.click(on_example, [session, columns_version, model], loaded_out))
        replace_button.click(lambda: gr.Row(visible=True), None, upload_row)
        previous_page.click(
            lambda s, page: show_page(handlers.data_view(s, page - 1)),
            [session, data_page],
            pager_out,
        )
        next_page.click(
            lambda s, page: show_page(handlers.data_view(s, page + 1)),
            [session, data_page],
            pager_out,
        )

        def on_model(s: Session, model_id: str) -> None:
            handlers.set_model(s, model_id)

        after(model.input(on_model, [session, model], None))
        theme_button.click(None, js=TOGGLE_JS)

        # ------------------------------------------------------------------ column roles
        def column_list(s: Session) -> None:
            native = s.draft is not None and s.draft.native
            for row in handlers.column_rows(s):
                with gr.Row(
                    equal_height=True, elem_classes=["ct-column-row"], key=f"column-{row.name}"
                ):
                    _html(
                        f'<div class="ct-col"><div class="ct-col-name">'
                        f"{fragments.esc(row.name)}</div>"
                        f'<div class="ct-col-samples">{fragments.esc(row.samples or "—")}</div>'
                        "</div>"
                    )
                    role = gr.Dropdown(
                        list(handlers.ROLE_CHOICES),
                        value=row.role,
                        show_label=False,
                        container=False,
                        interactive=not native,
                        scale=0,
                        min_width=240,
                        key=f"role-{row.name}",
                        preserved_by_key=None,
                    )

                def on_role(
                    s: Session, version: int, chosen: str | None, value: str, column: str = row.name
                ) -> list:
                    error = handlers.set_role(s, column, value)
                    if value == "question" and error is None:
                        chosen = column
                    return show_schema(s, version, chosen, bump=True, error=error)

                after(role.input(on_role, [session, columns_version, selected, role], schema_out))

        # ------------------------------------------------------------------ question editor
        question_inputs = [session, columns_version, selected]
        question_pills.input(
            lambda s, version, chosen, value: show_schema(s, version, value, bump=False),
            [*question_inputs, question_pills],
            schema_out,
        )

        def on_question(s: Session, version: int, chosen: str | None, **change: Any) -> list:
            error = handlers.update_question(s, chosen, **change) if chosen else None
            if error is None and change.get("new_id"):
                chosen = change["new_id"].strip()
            return show_schema(s, version, chosen, bump=False, error=error)

        after(
            question_id.blur(
                lambda s, v, chosen, value: on_question(s, v, chosen, new_id=value),
                [*question_inputs, question_id],
                schema_out,
            )
        )
        after(
            question_type.input(
                lambda s, v, chosen, value: on_question(s, v, chosen, type=value),
                [*question_inputs, question_type],
                schema_out,
            )
        )
        after(
            question_text.blur(
                lambda s, v, chosen, value: on_question(s, v, chosen, instructions=value),
                [*question_inputs, question_text],
                schema_out,
            )
        )
        after(
            question_options.input(
                lambda s, v, chosen, rows: on_question(s, v, chosen, options=rows),
                [*question_inputs, question_options],
                schema_out,
            )
        )
        preview_row.release(
            lambda s, row: handlers.schema_view(s, int(row) - 1).preview,
            [session, preview_row],
            preview,
        )

        # ------------------------------------------------------------------ configure events
        def on_run_name(s: Session, name: str) -> list:
            error = handlers.set_run_name(s, name) if s.draft else None
            return show_config(s, error)

        for event in (run_name.blur, run_name.submit):
            after(event(on_run_name, [session, run_name], config_out))

        def on_preset(s: Session, value: str) -> list:
            if s.draft is not None:
                handlers.set_preset(s, value)
            return show_config(s)

        after(preset.input(on_preset, [session, preset], config_out))
        for spec, component in zip(handlers.ADVANCED_FIELDS, fields, strict=True):

            def on_field(s: Session, value: Any, path: str = spec.path) -> list:
                error = handlers.set_field(s, path, value) if s.draft else None
                return show_config(s, error)

            triggers = (
                [component.input]
                if spec.kind in ("bool", "choice")
                else [component.blur, component.submit]
            )
            for event in triggers:
                after(event(on_field, [session, component], config_out))

        def on_weights(s: Session, rows: list[list[Any]]) -> list:
            error = handlers.set_question_weights(s, rows) if s.draft else None
            return show_config(s, error)

        after(weights.input(on_weights, [session, weights], config_out))

        # ------------------------------------------------------------------ launch events
        def on_account(pasted: str, oauth: gr.OAuthToken | None = None) -> list:
            name = runs.account(settings, services, oauth_token=_oauth(oauth), pasted=pasted)
            if name:
                status = fragments.callout("success", f"Signed in as {name}.")
            elif settings.space:
                status = fragments.callout(
                    "info",
                    "Sign in with Hugging Face (top right), or paste a token, to launch "
                    "and follow runs.",
                )
            else:
                status = fragments.callout(
                    "info", "No Hugging Face login found. Run hf auth login, or paste a token."
                )
            rows, error = runs.list_runs(
                settings, services, oauth_token=_oauth(oauth), pasted=pasted
            )
            if error:
                message = fragments.callout("warning", error)
            elif not rows:
                message = fragments.empty_state(
                    "No runs yet", "Runs you launch on Hugging Face Jobs show up here."
                )
            else:
                message = '<p class="ct-muted">Select a run to follow it.</p>'
            return [status, pd.DataFrame(rows, columns=RUN_COLUMNS), message]

        account_out = [account_html, runs_table, runs_message]
        demo.load(on_account, [token_box], account_out)
        for event in (token_box.blur, token_box.submit, refresh_runs.click):
            event(on_account, [token_box], account_out)

        def on_review(
            s: Session,
            pasted: str,
            hardware: str | None,
            limit: str,
            adapter_repo: str,
            delete: bool,
            oauth: gr.OAuthToken | None = None,
        ) -> list:
            review = runs.review_launch(
                s,
                settings,
                services,
                oauth_token=_oauth(oauth),
                pasted=pasted,
                flavor=hardware,
                timeout=limit,
                adapter_repo=adapter_repo,
                delete_data=delete,
            )
            body = review.summary if review.can_confirm else _callout("error", review.error)
            return [
                body,
                gr.Button(review.confirm_label, visible=review.can_confirm),
                review.timeout if review.can_confirm else gr.skip(),
                "",
            ]

        review_button.click(
            on_review,
            [session, token_box, flavor, timeout, repo, delete_data],
            [review_html, confirm_button, timeout, launch_result],
        )

        def on_confirm(
            s: Session, pasted: str, logs: bool, step: int, oauth: gr.OAuthToken | None = None
        ) -> list:
            before = s.job_id
            message = runs.confirm_launch(
                s, settings, services, oauth_token=_oauth(oauth), pasted=pasted
            )
            started = s.job_id != before and s.job_id is not None
            monitor_view = (
                follow(s, "job", pasted, logs, _oauth(oauth))
                if started
                else [gr.skip()] * len(monitor_out)
            )
            return [message, gr.Button(visible=False), "job", *monitor_view, *chrome(s, step)]

        confirm_button.click(
            on_confirm,
            [session, token_box, logs_open, current],
            [launch_result, confirm_button, watching, *monitor_out, *chrome_out],
            trigger_mode="once",
            concurrency_limit=1,
            concurrency_id="launch",
        ).then(on_account, [token_box], account_out)

        def on_bundle(s: Session) -> list:
            try:
                path = handlers.bundle(s)
            except (ValueError, OSError) as exc:
                return [gr.DownloadButton(visible=False), _callout("error", str(exc))]
            return [
                gr.DownloadButton(value=str(path), visible=True),
                _callout("success", f"{path.name} is ready."),
            ]

        bundle_button.click(on_bundle, [session], [bundle_file, bundle_message])

        if local_training:

            def on_start(s: Session, chosen: str, step: int) -> list:
                target = (chosen or "").strip() or local.default_folder(s)
                try:
                    local.start_local(s, target)
                except (local.LocalRunError, ValueError, OSError) as exc:
                    skipped = [gr.skip()] * (len(monitor_out) + 1)
                    return [_callout("error", str(exc)), *skipped, *chrome(s, step)]
                message = _callout("success", f"Training started in {target}.")
                return [message, "local", *show_run(local.local_view(s), "local"), *chrome(s, step)]

            start_button.click(
                on_start,
                [session, folder, current],
                [local_message, watching, *monitor_out, *chrome_out],
            )

        # ------------------------------------------------------------------ monitor events
        def on_select(
            s: Session,
            pasted: str,
            logs: bool,
            event: gr.SelectData,
            oauth: gr.OAuthToken | None = None,
        ) -> list:
            row = event.row_value or []
            if not row:
                return ["job", *[gr.skip()] * len(monitor_out)]
            runs.select_run(s, str(row[0]))
            return ["job", *follow(s, "job", pasted, logs, _oauth(oauth))]

        runs_table.select(on_select, [session, token_box, logs_open], [watching, *monitor_out])

        def on_tick(
            s: Session, kind: str, pasted: str, logs: bool, oauth: gr.OAuthToken | None = None
        ) -> list:
            return follow(s, kind, pasted, logs, _oauth(oauth))

        poll_inputs = [session, watching, token_box, logs_open]
        timer.tick(on_tick, poll_inputs, monitor_out, show_progress="hidden")
        logs_box.expand(lambda: True, None, logs_open).then(on_tick, poll_inputs, monitor_out)
        logs_box.collapse(lambda: False, None, logs_open)

        def on_cancel(
            s: Session, kind: str, pasted: str, oauth: gr.OAuthToken | None = None
        ) -> str:
            if kind == "local":
                local.stop_local(s.local_folder or "")
                return _callout(
                    "info", "Stopping. The checkpoints are kept; start again to resume."
                )
            return runs.cancel_run(s, settings, services, oauth_token=_oauth(oauth), pasted=pasted)

        cancel_button.click(on_cancel, [session, watching, token_box], action_message).then(
            on_tick, poll_inputs, monitor_out
        )

        def on_resume_review(
            s: Session, kind: str, pasted: str, limit: str, oauth: gr.OAuthToken | None = None
        ) -> list:
            if kind == "local":
                try:
                    local.start_local(s, s.local_folder or "")
                except (local.LocalRunError, ValueError, OSError) as exc:
                    return [_callout("error", str(exc)), gr.Button(visible=False), gr.skip()]
                message = _callout("success", "Training started again from the newest checkpoint.")
                return [message, gr.Button(visible=False), gr.skip()]
            review = runs.review_resume(
                s, settings, services, oauth_token=_oauth(oauth), pasted=pasted, timeout=limit
            )
            body = review.summary if review.can_confirm else _callout("error", review.error)
            return [
                body,
                gr.Button(review.confirm_label, visible=review.can_confirm),
                review.timeout if review.can_confirm else gr.skip(),
            ]

        resume_button.click(
            on_resume_review,
            [session, watching, token_box, resume_timeout],
            [resume_html, resume_confirm, resume_timeout],
        ).then(on_tick, poll_inputs, monitor_out)

        def on_resume(
            s: Session, pasted: str, limit: str, oauth: gr.OAuthToken | None = None
        ) -> list:
            message = runs.resume_run(
                s, settings, services, oauth_token=_oauth(oauth), pasted=pasted, timeout=limit
            )
            return [message, "", gr.Button(visible=False)]

        resume_confirm.click(
            on_resume,
            [session, token_box, resume_timeout],
            [action_message, resume_html, resume_confirm],
            trigger_mode="once",
            concurrency_limit=1,
            concurrency_id="launch",
        ).then(on_tick, poll_inputs, monitor_out).then(on_account, [token_box], account_out)

    return demo


def launch(
    *,
    share: bool = False,
    port: int | None = None,
    settings: Settings | None = None,
    services: Services | None = None,
    prevent_thread_lock: bool = False,
) -> str | None:
    """Serve the UI and return its local URL (once serving, with prevent_thread_lock). Clears
    session folders left by earlier runs first."""
    settings = settings or Settings.from_env()
    sweep_stale()
    served = build_app(settings, services).launch(
        theme=theme(),
        css_paths=[STYLE],
        js=THEME_JS,
        max_file_size=f"{settings.max_upload_mb}mb",
        share=share,
        server_port=port,
        footer_links=["gradio"],
        prevent_thread_lock=prevent_thread_lock,
    )
    return served[1] if served else None
