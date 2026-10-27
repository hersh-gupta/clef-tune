"""Run Clef over items and turn logits plus targets into metrics."""

from __future__ import annotations

from dataclasses import dataclass

from clef_tune.batching import token_budget_batches
from clef_tune.calibration import softmax
from clef_tune.encoding import TrainItem, collate
from clef_tune.metrics import MetricsReport, Prediction, summarize


@dataclass
class EvalResult:
    metrics: MetricsReport
    logits: dict[str, list[list[float]]]
    targets: dict[str, list[list[float]]]


def predict_logits(
    model,
    items: list[TrainItem],
    clef_code,
    processor,
    *,
    max_length: int,
    max_tokens_per_batch: int,
    device,
) -> list[dict[str, list[float]]]:
    """Per item, question_id -> logits, in item order."""
    import torch

    results: list[dict[str, list[float]]] = [{} for _ in items]
    batches = token_budget_batches([item.length for item in items], max_tokens_per_batch, 0, 0)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for indices in batches:
                batch_items = [items[i] for i in indices]
                batch = collate(batch_items, clef_code, processor, max_length, device)
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    outputs = model(batch)
                for index, item, record_logits in zip(indices, batch_items, outputs, strict=True):
                    for qid, tensor in zip(item.question_ids, record_logits, strict=True):
                        results[index][qid] = tensor.float().tolist()
    finally:
        model.train(was_training)
    return results


def summarize_eval(
    items: list[TrainItem],
    logits: list[dict[str, list[float]]],
    temperatures: dict[str, float] | None = None,
) -> EvalResult:
    temperatures = temperatures or {}
    records: list[list[Prediction]] = []
    logits_by_question: dict[str, list[list[float]]] = {}
    targets_by_question: dict[str, list[list[float]]] = {}
    for item, item_logits in zip(items, logits, strict=True):
        predictions = []
        for qid in item.question_ids:
            target = item.targets.get(qid)
            if target is None:
                continue
            row = item_logits[qid]
            question_type = item.record.questions[qid].type
            probs = softmax(row, temperatures.get(qid, 1.0))
            predictions.append(Prediction(qid, question_type, item.option_ids[qid], probs, target))
            logits_by_question.setdefault(qid, []).append(row)
            targets_by_question.setdefault(qid, []).append(target)
        if predictions:
            records.append(predictions)
    return EvalResult(summarize(records), logits_by_question, targets_by_question)
