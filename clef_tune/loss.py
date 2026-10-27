"""Weighted soft-target cross-entropy over Clef's per-question option logits."""

from __future__ import annotations

import torch
from torch import Tensor

from clef_tune.encoding import TrainItem
from clef_tune.weights import target_weight


def batch_loss(
    logits: list[list[Tensor]],
    items: list[TrainItem],
    *,
    class_weights: dict[str, dict[str, float]],
    question_weights: dict[str, float],
    label_smoothing: float,
) -> Tensor:
    numerator = None
    denominator = 0.0
    for record_logits, item in zip(logits, items, strict=True):
        for question, question_logits in zip(item.question_ids, record_logits, strict=True):
            target = item.targets.get(question)
            if target is None:
                continue
            options = item.option_ids[question]
            weight = target_weight(class_weights.get(question, {}), options, target)
            weight *= question_weights.get(question, 1.0)
            probs = torch.tensor(target, dtype=torch.float32, device=question_logits.device)
            probs = (1 - label_smoothing) * probs + label_smoothing / len(options)
            ce = -(probs * torch.log_softmax(question_logits.float(), dim=-1)).sum()
            numerator = weight * ce if numerator is None else numerator + weight * ce
            denominator += weight
    if numerator is None:
        return sum(logit.sum() for record in logits for logit in record) * 0.0
    return numerator / denominator
