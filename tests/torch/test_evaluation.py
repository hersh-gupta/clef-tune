import math

import pytest

torch = pytest.importorskip("torch")

from clef_tune.calibration import calibrate  # noqa: E402
from clef_tune.config import TrainSection, load_config  # noqa: E402
from clef_tune.data import Record  # noqa: E402
from clef_tune.encoding import TrainItem, collate, prepare_items  # noqa: E402
from clef_tune.evaluation import predict_logits, summarize_eval  # noqa: E402
from clef_tune.labels import Label  # noqa: E402
from clef_tune.model import load_for_training  # noqa: E402
from clef_tune.schema import Question  # noqa: E402
from clef_tune.validation import validate  # noqa: E402

pytestmark = pytest.mark.torch

CPU = torch.device("cpu")
NOUL = ["true", "false"]


def test_predict_logits_match_clefs_forward(tiny_release, tiny_project):
    loaded = load_for_training(tiny_release, TrainSection(mode="head_only"), device=CPU)
    records = validate(load_config(tiny_project()), None).eval[:6]
    items = prepare_items(loaded.clef_code, loaded.processor, records, max_length=1024)
    got = predict_logits(
        loaded.model,
        items,
        loaded.clef_code,
        loaded.processor,
        max_length=1024,
        max_tokens_per_batch=800,
        device=CPU,
    )
    batch = collate(items, loaded.clef_code, loaded.processor, 1024, CPU)
    with torch.no_grad():
        direct = loaded.model(batch)
    for row, record_logits, item in zip(got, direct, items, strict=True):
        for qid, tensor in zip(item.question_ids, record_logits, strict=True):
            assert row[qid] == pytest.approx(tensor.tolist(), abs=1e-5)


def test_summarize_eval_handles_per_row_and_unlabeled_questions():
    q1 = Question.from_dict("q1", {"type": "noul"})
    q2 = Question.from_dict("q2", {"type": "choice", "criteria": {"a": 1, "b": 2}})
    q3 = Question.from_dict("q3", {"type": "noul"})
    a = Record("a", "s", {"q1": q1, "q2": q2}, {"q1": Label(hard="true")}, [], 2)
    b = Record("b", "s", {"q3": q3}, {"q3": Label(hard="false")}, [], 3)
    items = [
        TrainItem(a, 1, ["q1", "q2"], {"q1": NOUL, "q2": ["a", "b"]}, {"q1": [1.0, 0.0]}, None),
        TrainItem(b, 1, ["q3"], {"q3": NOUL}, {"q3": [0.0, 1.0]}, None),
    ]
    logits = [{"q1": [2.0, 0.0], "q2": [0.0, 0.0]}, {"q3": [0.0, 1.0]}]
    result = summarize_eval(items, logits, temperatures={"q1": 2.0})
    assert set(result.metrics.questions) == set(result.logits) == {"q1", "q3"}
    assert result.targets == {"q1": [[1.0, 0.0]], "q3": [[0.0, 1.0]]}
    expected = math.log(1 + math.exp(-1.0))  # softmax([2, 0] / 2)
    assert result.metrics.questions["q1"].log_loss == pytest.approx(expected)
    assert calibrate(result.logits, result.targets)["q1"].fitted is False
