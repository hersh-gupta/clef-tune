import sys
from pathlib import Path
from types import SimpleNamespace

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


def fake_training(monkeypatch, tmp_path, error=None):
    calls = []

    class FakeError(RuntimeError):
        pass

    def train(config, **kwargs):
        calls.append((config, kwargs))
        if error:
            raise FakeError(error)
        return SimpleNamespace(adapter_dir=tmp_path / "final", hub_url=None, resumed_from=None)

    module = SimpleNamespace(train=train, TrainingError=FakeError)
    monkeypatch.setitem(sys.modules, "clef_tune.train", module)
    monkeypatch.setitem(
        sys.modules,
        "clef_tune.checkpoints",
        SimpleNamespace(ConfigMismatchError=type("ConfigMismatchError", (RuntimeError,), {})),
    )
    return calls


def test_train_command_passes_its_options(monkeypatch, tmp_path):
    calls = fake_training(monkeypatch, tmp_path)
    result = runner.invoke(
        cli.app,
        ["train", str(EXAMPLE), "--restart", "--push-to-hub", "me/r", "--delete-bundle-on-success"],
    )
    assert result.exit_code == 0, result.output
    assert calls == [
        (
            EXAMPLE,
            {"restart": True, "push_to_hub": "me/r", "delete_bundle_on_success": True},
        )
    ]
    assert "Adapter written" in result.output


def test_train_command_reports_failures_with_exit_1(monkeypatch, tmp_path):
    fake_training(monkeypatch, tmp_path, error="loss became non-finite at step 5")
    result = runner.invoke(cli.app, ["train", str(EXAMPLE)])
    assert result.exit_code == 1 and "non-finite at step 5" in result.output


def test_export_requires_merge(tmp_path):
    result = runner.invoke(cli.app, ["export", str(tmp_path), "-o", str(tmp_path / "m")])
    assert result.exit_code == 2 and "only --merge" in result.output
