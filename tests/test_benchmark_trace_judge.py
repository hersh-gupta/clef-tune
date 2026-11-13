"""The trace-judge pilot's scorer and job script (benchmarks/trace_judge/): the ranking
metrics must be right, and a job must run the tuned model before the untrained one."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from benchmarks.trace_judge import clef_jobs, prepare, score
from clef_tune.launch.source import Source


def pairs_auroc(scores, labels):
    """AUROC by its definition: the chance a "yes" outranks a "no", ties counting half."""
    yes, no = scores[labels], scores[~labels]
    wins = (yes[:, None] > no[None, :]).sum() + 0.5 * (yes[:, None] == no[None, :]).sum()
    return wins / (len(yes) * len(no))


@pytest.mark.parametrize("seed", range(5))
def test_auroc_matches_its_definition_with_ties(seed):
    rng = np.random.default_rng(seed)
    labels = rng.random(300) < 0.4
    scores = np.round(rng.normal(labels * 0.8, 1.0), 1)  # rounding makes many ties
    assert score.auroc(scores, labels) == pytest.approx(pairs_auroc(scores, labels))


def test_a_perfect_and_a_blind_judge():
    labels = np.array([True] * 50 + [False] * 50)
    perfect = labels.astype(float)
    assert score.auroc(perfect, labels) == 1.0
    assert score.partial_auroc(perfect, labels) == 1.0
    assert score.caught_at(perfect, labels, 0.01) == 1.0
    blind = np.zeros(100)  # one score for every trace
    assert score.auroc(blind, labels) == 0.5
    assert score.partial_auroc(blind, labels) == pytest.approx(0.1)  # chance, after dividing by 0.2
    assert score.caught_at(blind, labels, 0.05) == 0.0


def test_partial_auroc_counts_only_the_low_false_alarm_end():
    # Half the "yes" traces outrank every "no"; the rest rank below every "no".
    labels = np.array([True] * 10 + [False] * 100 + [True] * 10)
    scores = np.array([2.0] * 10 + [1.0] * 100 + [0.0] * 10)
    assert score.auroc(scores, labels) == pytest.approx(0.5)
    false_alarms, caught = score.roc(scores, labels)
    assert list(zip(false_alarms, caught, strict=True)) == [(0, 0), (0, 0.5), (1, 0.5), (1, 1)]
    assert score.partial_auroc(scores, labels) == pytest.approx(0.5)
    assert score.caught_at(scores, labels, 0.01) == 0.5


def test_caught_at_uses_a_real_threshold():
    labels = np.array([True, True, False, False, False, False])
    scores = np.array([0.9, 0.3, 0.8, 0.2, 0.1, 0.0])
    # Catching the second "yes" flags 1 of 4 "no" traces, which is more than 5%.
    assert score.caught_at(scores, labels, 0.05) == 0.5
    assert score.caught_at(scores, labels, 0.25) == 1.0


def test_a_split_with_too_few_of_a_class_is_not_scored():
    labels = np.array([True] * 9 + [False] * 30)
    assert score.stats(np.arange(39.0), labels) is None


def write_set(monkeypatch, tmp_path, rows, lines, limit=64):
    prepared, scores = tmp_path / "prepared", tmp_path / "scores"
    prepared.mkdir()
    scores.mkdir()
    header = ["id", "trace", *score.QUESTIONS, "source", "group", "detail", "tokens", "fits"]
    text = [",".join(header)] + [",".join(str(row.get(key, "")) for key in header) for row in rows]
    (prepared / "controlarena.csv").write_text("\n".join(text) + "\n", encoding="utf-8")
    (scores / "tuned-controlarena.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8"
    )
    timing = {
        "rows": len(lines),
        "tokens": 1000,
        "seconds": 36.0,
        "gpu": "test",
        "max_length": limit,
        "options": {"hidden_goal": ["true", "false"]},
        "temperatures": {},
    }
    (scores / "tuned-controlarena-timing.json").write_text(json.dumps(timing), encoding="utf-8")
    jobs = {  # named as clef_jobs.py --name writes them; only the last one read this set, tuned
        "score-first-run": {"price_per_hour": 9.0, "arms": ["tuned"], "sets": ["atbench"]},
        "score-untrained": {"price_per_hour": 9.0, "arms": ["base"], "sets": ["controlarena"]},
        "score-blackbox": {"price_per_hour": 2.0, "arms": ["tuned"]},
    }
    (tmp_path / "jobs.json").write_text(json.dumps(jobs))
    monkeypatch.setattr(score, "PREPARED", prepared)
    monkeypatch.setattr(score, "SCORES", scores)
    monkeypatch.setattr(score, "JOBS", tmp_path / "jobs.json")


def test_the_report_splits_by_group_and_length_and_prices_the_reading(monkeypatch, tmp_path):
    rows, lines = [], []
    for index in range(80):
        yes = index % 2 == 0
        group = "bash" if index < 40 else "shade"
        tokens = 10 if index % 4 < 2 else 40_000  # half the traces are longer than training saw
        rows.append(
            {
                "id": f"t{index}",
                "trace": "x",
                "hidden_goal": "true" if yes else "false",
                "source": "controlarena",
                "group": group,
                "detail": "gpt-5",
                "tokens": tokens,
                "fits": "true",
            }
        )
        # bash is judged perfectly; shade is judged backwards
        right = yes if group == "bash" else not yes
        logits = [2.0, 0.0] if right else [0.0, 2.0]
        lines.append({"id": f"t{index}", "tokens": tokens, "limit": 65536, "logits": {
            "hidden_goal": logits}})  # fmt: skip
    write_set(monkeypatch, tmp_path, rows, lines)
    report = score.build()["controlarena"]
    found = report["tuned"]["hidden_goal"]
    assert found["all"]["auroc"] == 0.5 and (found["all"]["yes"], found["all"]["no"]) == (40, 40)
    assert found["by_group"]["bash"]["auroc"] == 1.0 and found["by_group"]["shade"]["auroc"] == 0.0
    assert found["group_mean_auroc"] == 0.5
    assert set(found["by_length"]) == {"up to 16,384 tokens", "longer, read whole"}
    assert "by_detail" not in found  # one agent model: nothing to split
    read = report["tuned"]["read"]
    assert read["seconds_per_1000"] == 450.0 and read["usd_per_1000"] == 0.25
    assert report["length"]["hidden_goal"]["all"]["auroc"] == 0.5  # the shortcut baseline
    score.show({"controlarena": report})  # the table prints without error


def test_only_the_traces_clef_read_are_scored(monkeypatch, tmp_path):
    rows = [
        {"id": f"t{i}", "trace": "x", "hidden_goal": "true" if i % 2 else "false",
         "source": "controlarena", "group": "bash", "detail": "gpt-5", "tokens": 10 + i}
        for i in range(60)
    ]  # fmt: skip
    lines = [
        {"id": f"t{i}", "tokens": 10, "limit": 64, "logits": {"hidden_goal": [float(i % 2), 0.0]}}
        for i in range(40)  # the job stopped after 40 traces
    ]
    write_set(monkeypatch, tmp_path, rows, lines)
    report = score.build()["controlarena"]
    for arm in ("tuned", "length"):
        counted = report[arm]["hidden_goal"]["all"]
        assert counted["yes"] + counted["no"] == 40
    assert report["tuned"]["read"]["traces"] == 40 and report["tuned"]["read"]["of"] == 60


def test_the_job_reads_with_the_tuned_model_first_and_deletes_its_input():
    source = Source("https://github.com/hersh-gupta/clef-tune", "a" * 40)
    every = list(clef_jobs.SETS)
    script = clef_jobs.job_script(source, ["tuned", "base"], 65536, "someone/judge", every)
    steps = script.splitlines()
    tuned = next(i for i, s in enumerate(steps) if "score /tmp/tuned tuned" in s)
    base = next(i for i, s in enumerate(steps) if "score /tmp/base base" in s)
    assert tuned < base and steps[tuned].endswith(" 65536 " + ",".join(every))
    assert any("fetch someone/judge /tmp/tuned" in step for step in steps)
    assert steps[-2] == "rm -rf /work/input"
    assert "a" * 40 in steps[2]
    only = clef_jobs.job_script(source, ["base"], 16384, "someone/judge", ["controlarena"])
    assert "/tmp/tuned" not in only and "someone/judge" not in only
    assert "score /tmp/base base /work/input /work/out 16384 controlarena\n" in only


def test_the_sandbagging_test_families_do_not_depend_on_the_random_numbers():
    rows = [
        {"id": f"{family}-{label}-{i}", "group": family, "sandbagged": label, "tokens": 100 + i}
        for family in (*prepare.MALT_TEST_FAMILIES, "wordle", "pico_ctf")
        for label in ("true", "false")
        for i in range(4)
    ]
    for seed in range(3):
        train, test, info = prepare.malt_split(rows, random.Random(seed))
        assert {r["group"] for r in test} == set(prepare.MALT_TEST_FAMILIES)
        assert not {r["group"] for r in train} & set(prepare.MALT_TEST_FAMILIES)
        assert info["test_families"] == sorted(prepare.MALT_TEST_FAMILIES)
