import pytest
from pydantic import ValidationError

from clef_tune.config import PINNED_REVISIONS, Config, config_hash, load_config

MINIMAL = """\
run_name: demo
data:
  train: data/train.csv
schema:
  questions:
    outage: {type: noul}
"""


def write(directory, text, name="config.yaml"):
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    assert cfg.train.mode == "lora"
    assert cfg.train.lora.r == 16 and cfg.train.lora.alpha == 32
    assert cfg.train.max_length == 4096
    assert cfg.train.class_weighting == "none"
    assert cfg.eval.select_metric == "mean_macro_f1" and cfg.eval.calibrate is True
    assert cfg.model.resolved_revision() == PINNED_REVISIONS["Cloudflare/clef"]


def test_unknown_model_revision_defaults_to_main(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL + "model: {id: me/custom-clef}\n"))
    assert cfg.model.resolved_revision() == "main"


def test_paths_resolve_against_config_dir(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    cfg = load_config(write(project, MINIMAL.replace("data/train.csv", "data\\train.csv")))
    monkeypatch.chdir(tmp_path)  # the working directory must not matter
    assert cfg.resolve_path(cfg.data.train) == (project / "data" / "train.csv").resolve()
    assert cfg.output_dir() == (project / "out" / "demo").resolve()


def test_typo_rejected(tmp_path):
    with pytest.raises(ValidationError, match="epoch"):
        load_config(write(tmp_path, MINIMAL + "train:\n  epoch: 3\n"))


@pytest.mark.parametrize("name", ["bad name", "", "-leading-dash", "x" * 65])
def test_run_name_validated(tmp_path, name):
    with pytest.raises(ValidationError):
        Config.model_validate({"run_name": name, "data": {"train": "t.csv"}})


def test_schema_and_schema_file_conflict(tmp_path):
    with pytest.raises(ValidationError, match="not both"):
        load_config(write(tmp_path, MINIMAL + "schema_file: s.yaml\n"))


def test_schema_file_loads_relative_to_config(tmp_path):
    write(tmp_path, "questions:\n  outage: {type: noul}\n", "schemas/s.yaml")
    cfg = load_config(
        write(tmp_path, "run_name: demo\ndata: {train: t.csv}\nschema_file: schemas/s.yaml\n")
    )
    assert list(cfg.get_schema().questions) == ["outage"]


def test_no_schema_is_allowed(tmp_path):
    cfg = load_config(write(tmp_path, "run_name: demo\ndata: {train: t.jsonl}\n"))
    assert cfg.get_schema() is None


def test_invalid_schema_fails_fast(tmp_path):
    with pytest.raises(ValidationError, match="type must be one of"):
        load_config(write(tmp_path, MINIMAL.replace("noul", "bogus")))


def test_presets(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    quick = cfg.with_preset("quick")
    assert (quick.train.mode, quick.train.epochs, quick.train.max_length) == ("head_only", 1, 2048)
    thorough = cfg.with_preset("thorough")
    assert (thorough.train.lora.r, thorough.train.epochs) == (32, 5)
    assert thorough.eval.early_stopping_patience == 3
    assert thorough.base_dir == cfg.base_dir
    assert cfg.train.epochs == 3  # the original is unchanged


def test_yaml_round_trip(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    again = load_config(write(tmp_path, cfg.to_yaml(), "again.yaml"))
    assert again.to_dict() == cfg.to_dict()
    assert "schema" in cfg.to_dict() and "schema_def" not in cfg.to_dict()


def test_hash_ignores_run_name_and_output_but_not_training(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    base = config_hash(cfg)
    renamed = Config.model_validate(
        {**cfg.to_dict(), "run_name": "other", "output": {"dir": "elsewhere"}}
    )
    assert config_hash(renamed) == base
    changed = Config.model_validate({**cfg.to_dict(), "train": {"lora_lr": 3e-4}})
    assert config_hash(changed) != base
    assert len(base) == 16


def test_hash_includes_file_contents(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    data_file = tmp_path / "train.jsonl"
    data_file.write_text("a", encoding="utf-8")
    first = config_hash(cfg, [data_file])
    data_file.write_text("b", encoding="utf-8")
    assert config_hash(cfg, [data_file]) != first
