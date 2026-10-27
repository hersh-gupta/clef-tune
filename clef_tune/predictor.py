"""Predict with a trained adapter: SystemOne-shaped answers with calibration applied."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor

from clef_tune.adapter import AdapterInfo, read_adapter
from clef_tune.calibration import softmax
from clef_tune.clef_code import load_clef_code, model_dir
from clef_tune.data import Record, load_file
from clef_tune.encoding import TrainItem, prepare_items
from clef_tune.evaluation import predict_logits
from clef_tune.hub import pull_adapter
from clef_tune.model import compute_dtype, default_device, load_backbone, load_head
from clef_tune.schema import QUESTION_TYPES, Schema

Rows = Iterable[dict[str, Any]]
Probabilities = list[dict[str, dict[str, float]]]


class Predictor:
    def __init__(
        self,
        model: Any,
        processor: Any,
        clef_code: Any,
        info: AdapterInfo,
        device: torch.device,
    ) -> None:
        self.model = model
        self.processor = processor
        self.clef_code = clef_code
        self.info = info
        self.device = device
        self.schema: Schema | None = info.schema
        self.temperatures: dict[str, float] = dict(info.calibration)
        self.max_length: int = info.config.train.max_length

    @classmethod
    def load(
        cls,
        path_or_repo: str | Path,
        *,
        device: str | torch.device | None = None,
        quantization: str | None = None,
        revision: str | None = None,
        token: str | None = None,
    ) -> Predictor:
        local = Path(path_or_repo)
        if local.is_dir():
            path = local
        else:
            path = pull_adapter(str(path_or_repo), revision=revision, token=token)
        info = read_adapter(path)
        target = torch.device(device) if device is not None else default_device()
        dtype = compute_dtype(target)
        base = model_dir(info.config)
        clef_code = load_clef_code(base)
        backbone = load_backbone(
            base,
            quantization=quantization or info.config.train.quantization,
            device=target,
            dtype=dtype,
        )
        if info.has_lora:
            from peft import PeftModel

            backbone = PeftModel.from_pretrained(backbone, info.path / "adapter")
        head, _ = load_head(info.path, clef_code, device=target, dtype=dtype)
        model = clef_code.ClefModel(backbone, head).eval()
        processor = AutoProcessor.from_pretrained(base)
        return cls(model, processor, clef_code, info, target)

    def _records(self, rows: Rows) -> list[Record]:
        records = []
        for index, row in enumerate(rows):
            if row.get("questions") is not None:
                questions = Schema.from_dict({"questions": row["questions"]}).questions
            elif self.schema is not None:
                questions = self.schema.questions
            else:
                raise ValueError(
                    f"row {index} has no 'questions' and the adapter has no schema to default to"
                )
            raw_images = row.get("images") or []
            images = [raw_images] if isinstance(raw_images, str) else list(raw_images)
            record_id = str(row["id"]) if row.get("id") is not None else str(index)
            records.append(Record(record_id, row["state"], questions, {}, images))
        return records

    def _run(self, records: list[Record]) -> tuple[list[TrainItem], list[dict[str, list[float]]]]:
        train = self.info.config.train
        items = prepare_items(self.clef_code, self.processor, records, self.max_length)
        logits = predict_logits(
            self.model,
            items,
            self.clef_code,
            self.processor,
            max_length=self.max_length,
            max_tokens_per_batch=train.max_tokens_per_batch,
            device=self.device,
        )
        return items, logits

    def _probabilities(
        self, records: list[Record], logits: list[dict[str, list[float]]], calibrated: bool
    ) -> Probabilities:
        out: Probabilities = []
        for record, per_question in zip(records, logits, strict=True):
            row = {}
            for qid, values in per_question.items():
                temperature = self.temperatures.get(qid, 1.0) if calibrated else 1.0
                probs = softmax(values, temperature)
                row[qid] = dict(zip(record.questions[qid].option_ids(), probs, strict=True))
            out.append(row)
        return out

    def _answers(self, records: list[Record], probabilities: Probabilities) -> list[dict]:
        return [
            {
                "id": record.id,
                "answers": {
                    qid: self.clef_code.systemone_answer(record.questions[qid].to_clef(), probs)
                    for qid, probs in per_question.items()
                },
            }
            for record, per_question in zip(records, probabilities, strict=True)
        ]

    def logits(self, rows: Rows) -> list[dict[str, list[float]]]:
        return self._run(self._records(rows))[1]

    def probabilities(self, rows: Rows, *, calibrated: bool = True) -> Probabilities:
        records = self._records(rows)
        return self._probabilities(records, self._run(records)[1], calibrated)

    def predict(self, rows: Rows, *, calibrated: bool = True) -> list[dict]:
        records = self._records(rows)
        return self._answers(
            records, self._probabilities(records, self._run(records)[1], calibrated)
        )

    def systemone(self, request: dict[str, Any], *, calibrated: bool = True) -> dict[str, Any]:
        """Same request and response contract as Clef's own ``systemone``."""
        questions = request.get("questions")
        if not isinstance(request.get("model"), str) or "state" not in request:
            raise ValueError("model and state are required")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("at least one question is required")
        for question_id, question in questions.items():
            if question.get("type") not in QUESTION_TYPES:
                raise ValueError(f"{question_id}: type must be noul, choice, or score")
            if question["type"] != "noul" and not question.get("criteria"):
                raise ValueError(f"{question_id}: criteria must not be empty")
        row = {key: request[key] for key in ("state", "questions", "images") if key in request}
        records = self._records([row])
        items, logits = self._run(records)
        probabilities = self._probabilities(records, logits, calibrated)
        return {
            "model": request["model"],
            "answers": self._answers(records, probabilities)[0]["answers"],
            "usage": {"input_tokens": items[0].length, "output_tokens": 0},
        }


def predict_file(
    predictor: Predictor, data_path: str | Path, out_path: str | Path, *, calibrated: bool = True
) -> int:
    out = Path(out_path)
    suffix = out.suffix.lower()
    if suffix not in {".jsonl", ".csv"}:
        raise ValueError(f"unsupported output type {suffix!r} (use .jsonl or .csv)")
    result = load_file(predictor.info.config, str(Path(data_path).resolve()), for_prediction=True)
    errors = [issue for issue in result.issues if issue.level == "error"]
    if errors:
        raise ValueError("cannot predict:\n" + "\n".join(str(issue) for issue in errors))
    records = result.records
    _, logits = predictor._run(records)
    probabilities = predictor._probabilities(records, logits, calibrated)
    out.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".jsonl":
        with out.open("w", encoding="utf-8", newline="\n") as handle:
            for prediction in predictor._answers(records, probabilities):
                handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        return len(records)
    options: dict[str, list[str]] = {}
    for record in records:
        for qid, question in record.questions.items():
            options.setdefault(qid, question.option_ids())
    header = ["id"]
    for qid, option_ids in options.items():
        header += [qid, *(f"{qid}.p.{option}" for option in option_ids)]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for record, per_question in zip(records, probabilities, strict=True):
            line: list[Any] = [record.id]
            for qid, option_ids in options.items():
                probs = per_question.get(qid)
                if probs is None:
                    line += [""] * (1 + len(option_ids))
                else:
                    line.append(max(option_ids, key=lambda option: probs[option]))
                    line += [probs[option] for option in option_ids]
            writer.writerow(line)
    return len(records)
