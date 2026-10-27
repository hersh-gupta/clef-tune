from pathlib import Path

import pytest
from typer.testing import CliRunner

from clef_tune import cli
from tests.helpers import CharCounter, ticket_rows, write_csv

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "support_tickets.yaml"
runner = CliRunner()


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(cli, "counter_for_config", lambda config: CharCounter())
    monkeypatch.setattr(cli, "fetch_hf_prices", lambda: {})


def test_version():
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0 and "clef-tune" in result.output


def test_validate_example_passes_and_writes_bundle(tmp_path):
    result = runner.invoke(cli.app, ["validate", str(EXAMPLE), "--bundle", str(tmp_path / "b")])
    assert result.exit_code == 0, result.output
    assert "Validation passed" in result.output
    assert (tmp_path / "b" / "config.yaml").exists()


def test_validate_reports_errors_with_rows(tmp_path):
    rows = ticket_rows(20)
    rows[1]["department"] = "biling"
    write_csv(tmp_path / "train.csv", rows)
    config = tmp_path / "config.yaml"
    config.write_text(
        "run_name: t\n"
        "data: {train: train.csv, id_column: id}\n"
        "schema:\n"
        "  questions:\n"
        "    department: {type: choice, criteria: {billing: x, technical: y}}\n"
        "    urgency: {type: score, criteria: [a, b, c]}\n"
        "    outage: {type: noul}\n",
        encoding="utf-8",
    )
    result = runner.invoke(cli.app, ["validate", str(config)])
    assert result.exit_code == 1
    assert "'biling' is not an option" in result.output and "rows 3" in result.output


def test_invalid_config_exits_2(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("run_name: t\ndata: {train: x.csv}\ntrain: {epoch: 3}\n", encoding="utf-8")
    result = runner.invoke(cli.app, ["validate", str(config)])
    assert result.exit_code == 2 and "Invalid config" in result.output


def test_estimate_example_lists_hf_flavors():
    result = runner.invoke(cli.app, ["estimate", str(EXAMPLE)])
    assert result.exit_code == 0, result.output
    assert "l40sx1" in result.output and "Recommended" in result.output


def test_estimate_custom_gpu():
    result = runner.invoke(cli.app, ["estimate", str(EXAMPLE), "--gpu", "H100", "--price", "2.5"])
    assert result.exit_code == 0, result.output
    assert "1x H100" in result.output and "$" in result.output


def test_estimate_rejects_unknown_gpu():
    result = runner.invoke(cli.app, ["estimate", str(EXAMPLE), "--gpu", "TPU"])
    assert result.exit_code != 0


def test_yaml_syntax_error_in_config_exits_2(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("run_name: r\ndata: {train: d.csv\n", encoding="utf-8")
    result = runner.invoke(cli.app, ["validate", str(config)])
    assert result.exit_code == 2 and "Invalid config" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_estimate_shows_the_validation_errors(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "run_name: t\ndata: {train: missing.csv}\nschema: {questions: {q: {type: noul}}}\n",
        encoding="utf-8",
    )
    result = runner.invoke(cli.app, ["estimate", str(config), "--gpu", "H100"])
    assert result.exit_code == 1
    assert "file not found" in result.output
