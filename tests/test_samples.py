from pathlib import Path

import pytest

from clef_tune.config import load_config
from clef_tune.validation import validate
from tests.helpers import CharCounter

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SAMPLES = ["support_tickets", "claims_triage", "server_logs"]


@pytest.mark.parametrize("name", SAMPLES)
def test_each_sample_validates_with_one_question_of_each_type(name):
    config = load_config(EXAMPLES / f"{name}.yaml")
    report = validate(config, CharCounter())
    assert report.ok, [str(issue) for issue in report.errors]
    assert sorted(q.type for q in config.get_schema().questions.values()) == [
        "choice",
        "noul",
        "score",
    ]
    assert len(report.train) + len(report.eval) == 200
    assert config.data.row_noun


@pytest.mark.parametrize("name", SAMPLES)
def test_every_answer_is_common_enough_to_learn(name):
    config = load_config(EXAMPLES / f"{name}.yaml")
    report = validate(config, CharCounter())
    for qid, stats in report.questions.items():
        total = sum(stats.train_counts.values())
        for option, count in stats.train_counts.items():
            assert count / total >= 0.05, f"{name}: {qid}={option} is {count}/{total}"


@pytest.mark.parametrize("name", ["claims_triage", "server_logs"])
def test_the_new_samples_are_reproducible(name, tmp_path):
    from scripts.make_samples import write_sample

    out = write_sample(name, tmp_path)
    assert out.read_bytes() == (EXAMPLES / f"{name}.csv").read_bytes()
