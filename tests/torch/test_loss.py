import math

import pytest

torch = pytest.importorskip("torch")
from clef_tune.encoding import TrainItem  # noqa: E402
from clef_tune.loss import batch_loss  # noqa: E402

pytestmark = pytest.mark.torch
NOUL = ["true", "false"]


def item(targets, option_ids):
    return TrainItem(None, 1, list(option_ids), option_ids, targets, None)


def loss(logits, items, **kwargs):
    options = dict(class_weights={}, question_weights={}, label_smoothing=0.0) | kwargs
    return batch_loss(logits, items, **options).item()


def test_hard_label_cross_entropy():
    value = loss([[torch.tensor([2.0, 0.0])]], [item({"q": [1.0, 0.0]}, {"q": NOUL})])
    assert value == pytest.approx(math.log(1 + math.exp(-2)))


def test_soft_target_and_masked_question():
    logits = [[torch.zeros(3), torch.tensor([5.0, -5.0])]]
    it = item({"d": [0.5, 0.5, 0.0]}, {"d": ["a", "b", "c"], "o": NOUL})
    assert loss(logits, [it]) == pytest.approx(math.log(3))


def test_weighted_mean_uses_class_and_question_weights():
    logits = [[torch.zeros(2)], [torch.tensor([math.log(3), 0.0])]]
    items = [item({"q": [1.0, 0.0]}, {"q": NOUL}), item({"q": [0.0, 1.0]}, {"q": NOUL})]
    value = loss(
        logits,
        items,
        class_weights={"q": {"true": 3.0, "false": 1.0}},
        question_weights={"q": 2.0},
    )
    assert value == pytest.approx((6 * math.log(2) + 2 * math.log(4)) / 8)


def test_label_smoothing():
    value = loss(
        [[torch.tensor([math.log(3), 0.0])]],
        [item({"q": [1.0, 0.0]}, {"q": NOUL})],
        label_smoothing=0.2,
    )
    assert value == pytest.approx(-(0.9 * math.log(0.75) + 0.1 * math.log(0.25)))


def test_batch_without_labels_is_a_zero_that_backpropagates():
    logits = [[torch.zeros(2, requires_grad=True)]]
    value = batch_loss(
        logits, [item({}, {"q": NOUL})], class_weights={}, question_weights={}, label_smoothing=0.0
    )
    assert value.item() == 0.0
    value.backward()
