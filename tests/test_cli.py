import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from clef_tune import cli
from clef_tune.clef_code import ClefCodeError
from tests.helpers import CharCounter, make_config, ticket_rows, write_csv

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


def fake_training(monkeypatch, tmp_path, error=None, main_process=True):
    calls = []

    class FakeError(RuntimeError):
        pass

    def train(config, **kwargs):
        calls.append((config, kwargs))
        if isinstance(error, BaseException):
            raise error
        if error:
            raise FakeError(error)
        return SimpleNamespace(
            adapter_dir=tmp_path / "final",
            hub_url=None,
            resumed_from=None,
            main_process=main_process,
        )

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


@pytest.mark.parametrize(
    "error", [ClefCodeError("model.local_path 'w' is not a model directory"), OSError("disk full")]
)
def test_train_command_reports_load_and_os_errors_with_exit_1(monkeypatch, tmp_path, error):
    fake_training(monkeypatch, tmp_path, error=error)
    result = runner.invoke(cli.app, ["train", str(EXAMPLE)])
    assert result.exit_code == 1 and str(error) in result.output
    assert isinstance(result.exception, SystemExit)


def fake_predictor(monkeypatch, load=None, predict_file=None):
    """Stand in for clef_tune.predictor (a torch module) in the CLI's lazy import."""

    def default_load(path, **kwargs):
        pytest.fail(f"tried to load {path!r}")

    module = SimpleNamespace(
        Predictor=SimpleNamespace(load=load or default_load), predict_file=predict_file
    )
    monkeypatch.setitem(sys.modules, "clef_tune.predictor", module)


@pytest.mark.parametrize(
    "adapter", ["./out/final", "out/tiny/final", r"out\final", r"C:\nowhere\final", "~/nowhere"]
)
def test_a_missing_local_adapter_directory_is_not_looked_up_on_the_hub(
    monkeypatch, tmp_path, adapter
):
    fake_predictor(monkeypatch)
    data = write_csv(tmp_path / "in.csv", ticket_rows(2))
    for command in (
        ["predict", adapter, str(data), "-o", str(tmp_path / "p.jsonl")],
        ["eval", adapter, str(data)],
    ):
        result = runner.invoke(cli.app, command)
        assert result.exit_code == 1, result.output
        assert "adapter directory not found" in result.output


def test_adapter_load_errors_exit_1_without_a_traceback(monkeypatch, tmp_path):
    def load(path, **kwargs):
        raise ClefCodeError("model.local_path 'weights' is not a model directory")

    fake_predictor(monkeypatch, load=load)
    data = write_csv(tmp_path / "in.csv", ticket_rows(2))
    result = runner.invoke(cli.app, ["predict", str(tmp_path), str(data), "-o", "p.jsonl"])
    assert result.exit_code == 1 and "not a model directory" in result.output
    assert isinstance(result.exception, SystemExit)


def test_predict_reports_file_errors_with_exit_1(monkeypatch, tmp_path):
    def predict_file(predictor, data, output, **kwargs):
        raise FileNotFoundError("images/missing.png")

    fake_predictor(monkeypatch, load=lambda path, **kwargs: object(), predict_file=predict_file)
    data = write_csv(tmp_path / "in.csv", ticket_rows(2))
    result = runner.invoke(cli.app, ["predict", str(tmp_path), str(data), "-o", "p.jsonl"])
    assert result.exit_code == 1 and "images/missing.png" in result.output
    assert isinstance(result.exception, SystemExit)


def test_eval_reports_unreadable_data_with_exit_1(monkeypatch, tmp_path):
    config = make_config(tmp_path, data={"train": "t.csv"})
    predictor = SimpleNamespace(info=SimpleNamespace(config=config))
    fake_predictor(monkeypatch, load=lambda path, **kwargs: predictor)
    (tmp_path / "labeled.txt").write_text("not a table", encoding="utf-8")
    result = runner.invoke(cli.app, ["eval", str(tmp_path), str(tmp_path / "labeled.txt")])
    assert result.exit_code == 1 and "unsupported file type" in result.output
    assert isinstance(result.exception, SystemExit)


def test_export_reports_a_missing_base_model_with_exit_1(monkeypatch, tmp_path):
    from clef_tune import adapter

    def export_merged(path, out, **kwargs):
        raise ClefCodeError("model.local_path '/model' is not a model directory")

    monkeypatch.setattr(adapter, "export_merged", export_merged)
    result = runner.invoke(cli.app, ["export", str(tmp_path), "--merge", "-o", "m"])
    assert result.exit_code == 1 and "not a model directory" in result.output
    assert isinstance(result.exception, SystemExit)


def test_train_command_prints_only_on_the_main_process(monkeypatch, tmp_path):
    fake_training(monkeypatch, tmp_path, main_process=False)
    result = runner.invoke(cli.app, ["train", str(EXAMPLE)])
    assert result.exit_code == 0 and "Adapter written" not in result.output


def test_predict_passes_the_review_threshold_probabilities_and_base_model(monkeypatch, tmp_path):
    loaded, called = {}, {}

    def load(path, **kwargs):
        loaded.update(kwargs, path=path)
        return "predictor"

    def predict_file(predictor, data, output, **kwargs):
        called.update(kwargs, output=output)
        return 2

    fake_predictor(monkeypatch, load=load, predict_file=predict_file)
    data = write_csv(tmp_path / "in.csv", ticket_rows(2))
    model = tmp_path / "model"
    model.mkdir()
    result = runner.invoke(
        cli.app,
        [
            "predict",
            "me/repo",
            str(data),
            "-o",
            str(tmp_path / "p.xlsx"),
            "--review-below",
            "0.8",
            "--probabilities",
            "--base-model",
            str(model),
        ],
    )
    assert result.exit_code == 0, result.output
    assert loaded["base_model_path"] == model and loaded["path"] == "me/repo"
    assert called["review_below"] == 0.8 and called["probabilities"] is True
    assert "Wrote 2 predictions" in result.output


def test_the_ui_command_serves_the_web_app(monkeypatch):
    from clef_tune.ui import server

    served = {}
    monkeypatch.setattr(server, "launch", lambda **kwargs: served.update(kwargs))
    result = runner.invoke(cli.app, ["ui", "--port", "7861", "--host", "0.0.0.0"])
    assert result.exit_code == 0, result.output
    assert served == {"host": "0.0.0.0", "port": 7861}


def test_the_ui_help_warns_what_serving_the_network_means():
    result = runner.invoke(cli.app, ["ui", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0 and "Hugging Face login" in result.output
