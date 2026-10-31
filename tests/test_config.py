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


@pytest.mark.parametrize("absolute", ["/model", r"C:\models\clef", r"\\server\share\clef"])
def test_portable_dict_keeps_absolute_local_paths_as_written(tmp_path, absolute):
    cfg = load_config(write(tmp_path, MINIMAL + f"model: {{local_path: '{absolute}'}}\n"))
    assert cfg.to_portable_dict()["model"]["local_path"] == absolute


def test_portable_dict_inlines_the_schema_file_and_absolutizes_local_path(tmp_path):
    write(tmp_path, "questions:\n  outage: {type: noul}\n", "schemas/s.yaml")
    text = "run_name: demo\ndata: {train: t.csv}\nschema_file: schemas/s.yaml\n"
    cfg = load_config(write(tmp_path, text + "model: {local_path: weights}\n"))
    portable = cfg.to_portable_dict()
    assert "schema_file" not in portable
    assert portable["schema"] == cfg.get_schema().to_dict()
    assert portable["model"]["local_path"] == str((tmp_path / "weights").resolve())


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_question_weights_must_be_finite_and_not_negative(weight):
    data = {"run_name": "x", "data": {"train": "t.csv"}}
    with pytest.raises(ValidationError, match="question_weights.*department"):
        Config.model_validate({**data, "train": {"question_weights": {"department": weight}}})
    zero = Config.model_validate({**data, "train": {"question_weights": {"department": 0.0}}})
    assert zero.train.question_weights == {"department": 0.0}


def test_row_noun_is_optional_and_trimmed(tmp_path):
    assert load_config(write(tmp_path, MINIMAL)).data.row_noun is None
    cfg = Config.model_validate(
        {"run_name": "demo", "data": {"train": "t.csv", "row_noun": "  claim "}}
    )
    assert cfg.data.row_noun == "claim"
    blank = Config.model_validate({"run_name": "demo", "data": {"train": "t.csv", "row_noun": " "}})
    assert blank.data.row_noun is None
    with pytest.raises(ValidationError):
        Config.model_validate(
            {"run_name": "demo", "data": {"train": "t.csv", "row_noun": "x" * 41}}
        )


def test_hash_ignores_the_row_noun(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    named = Config.model_validate(
        {**cfg.to_dict(), "data": {**cfg.to_dict()["data"], "row_noun": "ticket"}}
    )
    assert config_hash(named) == config_hash(cfg)
