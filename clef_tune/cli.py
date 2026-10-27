"""clef-tune command line."""

from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from clef_tune import __version__
from clef_tune.bundle import write_bundle
from clef_tune.clef_code import ClefCodeError
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


def _print_errors(report: ValidationReport) -> None:
    for issue in report.errors:
        console.print(f"[red]error[/] {escape(str(issue))}")


def print_report(report: ValidationReport) -> None:
    console.print(
        f"[bold]{len(report.train)}[/] training rows, [bold]{len(report.eval)}[/] eval rows"
    )
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
    _print_errors(report)
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
    gpu: Annotated[str | None, typer.Option(help=f"Your GPU type: {', '.join(GPUS)}.")] = None,
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
        _print_errors(report)
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


def _fail(message: str) -> typer.Exit:
    console.print(f"[red]{message}")
    return typer.Exit(1)


@app.command()
def train(
    config: ConfigArg,
    restart: Annotated[
        bool, typer.Option("--restart", help="Ignore existing checkpoints and start over.")
    ] = False,
    push_to_hub: Annotated[
        str | None, typer.Option("--push-to-hub", help="Push the adapter to this Hub repo.")
    ] = None,
    delete_bundle_on_success: Annotated[
        bool, typer.Option("--delete-bundle-on-success", help="Delete the bundle after success.")
    ] = False,
) -> None:
    """Train an adapter from a config (or a bundle's config.yaml)."""
    _load(config)
    from clef_tune.checkpoints import ConfigMismatchError
    from clef_tune.hub import HubError
    from clef_tune.train import TrainingError
    from clef_tune.train import train as run_training

    try:
        result = run_training(
            config,
            restart=restart,
            push_to_hub=push_to_hub,
            delete_bundle_on_success=delete_bundle_on_success,
        )
    except (TrainingError, ConfigMismatchError, HubError, ClefCodeError, OSError) as exc:
        console.print(f"[red]Training failed:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    console.print(f"Adapter written: {escape(str(result.adapter_dir))}")
    if result.hub_url:
        console.print(f"Pushed: {escape(str(result.hub_url))}")


AdapterArg = Annotated[
    str, typer.Argument(help="Adapter directory (<output.dir>/final) or Hub repo id.")
]
DataArg = Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="CSV or JSONL data.")]
NoCalibrationOpt = Annotated[
    bool, typer.Option("--no-calibration", help="Do not apply the stored temperatures.")
]


def _looks_like_path(value: str) -> bool:
    """Hub repo ids are `name` or `namespace/name`; anything else is meant as a directory."""
    return (
        "\\" in value
        or value.count("/") > 1
        or value.startswith((".", "~", "/"))
        or value.endswith("/")
        or bool(PureWindowsPath(value).drive)
    )


def _load_predictor(adapter: str):
    path = Path(adapter).expanduser()
    if not path.is_dir() and _looks_like_path(adapter):
        raise _fail(f"adapter directory not found: {escape(str(path))}")
    from clef_tune.hub import HubError
    from clef_tune.predictor import Predictor

    source = str(path) if path.is_dir() else adapter
    try:
        return Predictor.load(source)
    except (HubError, ClefCodeError, OSError, ValueError) as exc:
        where = "" if path.is_dir() else f" (no local directory {adapter!r}; tried the Hub)"
        raise _fail(f"Could not load the adapter{escape(where)}: {escape(str(exc))}") from exc


@app.command("eval")
def evaluate(
    adapter: AdapterArg,
    data: DataArg,
    output: Annotated[
        Path | None, typer.Option("--output", help="Write the metrics report as JSON.")
    ] = None,
    no_calibration: NoCalibrationOpt = False,
) -> None:
    """Evaluate an adapter on a labeled file."""
    from clef_tune.calibration import expected_calibration_error, softmax
    from clef_tune.card import confusion_markdown
    from clef_tune.data import load_file
    from clef_tune.encoding import prepare_items
    from clef_tune.evaluation import predict_logits, summarize_eval

    predictor = _load_predictor(adapter)
    try:
        loaded = load_file(predictor.info.config, str(data.resolve()))
    except (ValueError, OSError) as exc:
        raise _fail(f"Evaluation failed: {escape(str(exc))}") from exc
    for issue in loaded.issues:
        if issue.level == "error":
            console.print(f"[red]error[/] {escape(str(issue))}")
    records = [record for record in loaded.records if record.labels]
    if not records:
        raise _fail("No labeled rows in the data file.")
    train_config = predictor.info.config.train
    try:
        items = prepare_items(
            predictor.clef_code, predictor.processor, records, predictor.max_length
        )
        logits = predict_logits(
            predictor.model,
            items,
            predictor.clef_code,
            predictor.processor,
            max_length=predictor.max_length,
            max_tokens_per_batch=train_config.max_tokens_per_batch,
            device=next(predictor.model.parameters()).device,
        )
    except (ValueError, OSError) as exc:  # e.g. an image that cannot be read
        raise _fail(f"Evaluation failed: {escape(str(exc))}") from exc
    temperatures = {} if no_calibration else predictor.temperatures
    result = summarize_eval(items, logits, temperatures)
    metrics = result.metrics

    table = Table("question", "type", "n", "accuracy", "F1", "log loss", "MAE")
    for qid, q in metrics.questions.items():
        table.add_row(
            escape(qid),
            q.question_type,
            str(q.n),
            f"{q.accuracy:.3f}",
            f"{q.f1:.3f}",
            f"{q.log_loss:.3f}",
            "-" if q.mae is None else f"{q.mae:.3f}",
        )
    console.print(table)
    console.print(f"Exact match: {metrics.exact_match:.3f}")
    for qid, q in metrics.questions.items():
        console.print(f"\n[bold]{escape(qid)}[/] confusion matrix")
        console.print(escape(confusion_markdown(q.confusion.labels, q.confusion.matrix)))

    calibration: dict[str, dict[str, float]] = {}
    for qid, rows in result.logits.items():
        temperature = temperatures.get(qid, 1.0)
        targets = result.targets[qid]
        calibration[qid] = {
            "ece_before": expected_calibration_error([softmax(row) for row in rows], targets),
            "ece_after": expected_calibration_error(
                [softmax(row, temperature) for row in rows], targets
            ),
            "temperature": temperature,
        }
    ece_table = Table("question", "ECE before", "ECE after", "temperature")
    for qid, values in calibration.items():
        ece_table.add_row(
            escape(qid),
            f"{values['ece_before']:.3f}",
            f"{values['ece_after']:.3f}",
            f"{values['temperature']:.3f}",
        )
    console.print(ece_table)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        report = {**metrics.to_dict(), "calibration": calibration}
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        console.print(f"Metrics written: {escape(str(output))}")


@app.command()
def predict(
    adapter: AdapterArg,
    data: DataArg,
    output: Annotated[Path, typer.Option("-o", "--output", help="Output .jsonl or .csv file.")],
    no_calibration: NoCalibrationOpt = False,
) -> None:
    """Predict answers for a data file."""
    from clef_tune.hub import HubError

    predictor = _load_predictor(adapter)
    from clef_tune.predictor import predict_file

    try:
        count = predict_file(predictor, data, output, calibrated=not no_calibration)
    except (ValueError, OSError, ClefCodeError, HubError) as exc:
        raise _fail(escape(str(exc))) from exc
    console.print(f"Wrote {count} predictions to {escape(str(output))}")


@app.command()
def export(
    adapter: Annotated[Path, typer.Argument(help="Adapter directory.")],
    merge: Annotated[
        bool, typer.Option("--merge", help="Merge the adapter into a full checkpoint.")
    ] = False,
    output: Annotated[Path, typer.Option("-o", "--output", help="Output directory.")] = Path(
        "merged"
    ),
) -> None:
    """Export the adapter as a merged checkpoint (Cloudflare's release layout)."""
    if not merge:
        console.print("[red]only --merge export is supported[/]")
        raise typer.Exit(2)
    from clef_tune.adapter import export_merged
    from clef_tune.hub import HubError

    try:
        path = export_merged(adapter, output)
    except (OSError, ValueError, ClefCodeError, HubError) as exc:
        raise _fail(f"Export failed: {escape(str(exc))}") from exc
    console.print(f"Merged checkpoint written: {escape(str(path))}")
