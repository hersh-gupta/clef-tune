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
