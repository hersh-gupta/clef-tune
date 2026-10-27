import json

from typer.testing import CliRunner

from clef_tune import cli

runner = CliRunner()


def test_train_predict_eval_export_from_the_command_line(tiny_project, tmp_path):
    config = tiny_project(train={"epochs": 2})
    assert runner.invoke(cli.app, ["train", str(config)]).exit_code == 0
    adapter = config.parent / "out" / "final"
    result = runner.invoke(
        cli.app,
        ["predict", str(adapter), str(config.parent / "eval.csv"), "-o", str(tmp_path / "p.jsonl")],
    )
    assert result.exit_code == 0 and "Wrote 16 predictions" in result.output
    result = runner.invoke(
        cli.app,
        [
            "eval",
            str(adapter),
            str(config.parent / "eval.csv"),
            "--output",
            str(tmp_path / "m.json"),
        ],
    )
    assert result.exit_code == 0 and "exact match" in result.output.lower()
    report = json.loads((tmp_path / "m.json").read_text())
    assert report["exact_match"] >= 0.0 and "calibration" in report
    result = runner.invoke(
        cli.app, ["export", str(adapter), "--merge", "-o", str(tmp_path / "merged")]
    )
    assert result.exit_code == 0 and (tmp_path / "merged" / "joint_head.safetensors").exists()
