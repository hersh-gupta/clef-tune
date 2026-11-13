"""The trace-judge scoring job's own script, run on the tiny model."""

import json

import pytest

torch = pytest.importorskip("torch")
from benchmarks.trace_judge import score_job  # noqa: E402

pytestmark = pytest.mark.torch


def run(monkeypatch, adapter, tmp_path, rows=5, limit="512"):
    monkeypatch.setattr(score_job, "MODEL_MOUNT", None)  # the adapter's own base model
    monkeypatch.setattr(score_job, "SLICE", 2)
    inputs = tmp_path / "input"
    inputs.mkdir(exist_ok=True)
    body = "".join(f"t{i},s,the site is down {'again ' * i}\n" for i in range(rows))
    (inputs / "malt_test.csv").write_text("id,subject,body\n" + body, encoding="utf-8")
    score_job.score(str(adapter), "tuned", str(inputs), str(tmp_path / "out"), limit)
    lines = (tmp_path / "out" / "tuned-malt_test.jsonl").read_text(encoding="utf-8").splitlines()
    timing = json.loads((tmp_path / "out" / "tuned-malt_test-timing.json").read_text())
    return [json.loads(line) for line in lines], timing


def test_every_trace_gets_a_line_with_logits_and_the_limit_it_was_read_under(
    monkeypatch, adapter, tmp_path
):
    lines, timing = run(monkeypatch, adapter, tmp_path)
    assert [line["id"] for line in lines] == [f"t{i}" for i in range(5)]
    assert all(line["limit"] == 512 and len(line["logits"]["outage"]) == 2 for line in lines)
    assert timing["rows"] == 5 and timing["tokens"] == sum(line["tokens"] for line in lines)
    assert timing["options"]["outage"] == ["true", "false"] and timing["max_length"] == 512
    assert timing["temperatures"]["outage"] == 2.0


def test_a_rerun_reads_only_what_is_missing(monkeypatch, adapter, tmp_path):
    first, _ = run(monkeypatch, adapter, tmp_path, rows=3)
    again, timing = run(monkeypatch, adapter, tmp_path, rows=5)
    assert again[:3] == first and [line["id"] for line in again[3:]] == ["t3", "t4"]
    assert timing["rows"] == 5


def test_a_trace_that_runs_out_of_memory_is_read_under_half_the_limit(
    monkeypatch, adapter, tmp_path
):
    import clef_tune.evaluation as evaluation

    real = evaluation.predict_logits

    def predict_logits(model, items, *args, max_length, **kwargs):
        if len(items) > 1 or max_length > 512:  # no batch fits, and no trace above 512 tokens
            raise torch.OutOfMemoryError("CUDA out of memory")
        return real(model, items, *args, max_length=max_length, **kwargs)

    monkeypatch.setattr(evaluation, "predict_logits", predict_logits)
    monkeypatch.setattr(score_job, "MIN_LENGTH", 512)
    lines, timing = run(monkeypatch, adapter, tmp_path, rows=3, limit="1024")
    assert [line["id"] for line in lines] == ["t0", "t1", "t2"]
    assert all(line["limit"] == 512 for line in lines) and timing["rows"] == 3


def test_only_the_named_test_sets_are_read(monkeypatch, adapter, tmp_path):
    monkeypatch.setattr(score_job, "MODEL_MOUNT", None)
    inputs = tmp_path / "input"
    inputs.mkdir()
    for name in ("malt_test", "atbench"):
        (inputs / f"{name}.csv").write_text(
            "id,subject,body\nt0,s,the site is down\n", encoding="utf-8"
        )
    out = tmp_path / "out"
    score_job.score(str(adapter), "base", str(inputs), str(out), "512", "atbench")
    assert sorted(p.name for p in out.iterdir()) == [
        "base-atbench-timing.json",
        "base-atbench.jsonl",
    ]
    with pytest.raises(SystemExit):
        score_job.score(str(adapter), "base", str(inputs), str(out), "512", "no_such_set")
    with pytest.raises(SystemExit):  # asked for by name, but its file was not uploaded
        score_job.score(str(adapter), "base", str(inputs), str(out), "512", "atbench,controlarena")


def test_a_trace_that_fits_under_no_limit_stops_the_job(monkeypatch, adapter, tmp_path):
    import clef_tune.evaluation as evaluation

    def predict_logits(*args, **kwargs):
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(evaluation, "predict_logits", predict_logits)
    with pytest.raises(torch.OutOfMemoryError):
        run(monkeypatch, adapter, tmp_path, rows=1, limit="8192")
