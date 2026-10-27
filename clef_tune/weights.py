"""Balanced per-question class weights for the loss (no torch)."""

from __future__ import annotations

from collections import Counter

from clef_tune.data import Record

WEIGHT_CLIP = (0.1, 10.0)


def class_weights(records: list[Record]) -> dict[str, dict[str, float]]:
    """w_o = N_q / (K_q * n_qo), clipped; options never observed get no entry."""
    counts: dict[str, Counter[str]] = {}
    for record in records:
        for question, label in record.labels.items():
            counts.setdefault(question, Counter())[label.top()] += 1
    lo, hi = WEIGHT_CLIP
    weights: dict[str, dict[str, float]] = {}
    for question, counter in counts.items():
        total = sum(counter.values())
        kinds = len(counter)
        weights[question] = {
            option: min(hi, max(lo, total / (kinds * n))) for option, n in counter.items()
        }
    return weights


def target_weight(weights_q: dict[str, float], option_ids: list[str], target: list[float]) -> float:
    """Expected class weight under the target distribution; unseen options count as 1.0."""
    return sum(t * weights_q.get(option, 1.0) for option, t in zip(option_ids, target, strict=True))
