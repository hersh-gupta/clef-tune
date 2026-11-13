"""Runs inside the trace-judge scoring job on Hugging Face Jobs, next to an installed clef-tune.

  score ADAPTER ARM INPUT_DIR OUT_DIR MAX_LENGTH [SETS]
      read the test files in INPUT_DIR with the adapter (every one, or the comma-separated
      SETS, each of which must be there) and write, per file,
        OUT_DIR/ARM-<set>.jsonl        one line per trace: its id, the tokens read, the length
                                       limit it was read under, and each question's raw logits
        OUT_DIR/ARM-<set>-timing.json  rows, tokens and seconds spent predicting, the answer
                                       options and the adapter's calibration temperatures

The untrained model and the tuned adapter are prepared by benchmarks/cfpb/score_job.py (`base`
and `fetch`), which the job also carries.

Traces are read whole up to MAX_LENGTH tokens, which can be far above the length the adapter
trained with. In the first run every trace was read under the full 65,536 tokens on a 96 GB
GPU. On less memory, a batch that runs out is read again one trace at a time, and a single
trace that still does not fit is read under half the limit (its end is cut), and half again,
until it fits. Every line records the limit its trace was read under.

A file's lines are appended as they are made and a rerun skips the ids already written, so a
job that stops can be started again on the same folder.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

MODEL_MOUNT = "/model"
SETS = ("malt_test", "atbench500", "atbench", "atbench_no_tools", "controlarena")
SLICE = 256  # traces tokenized, predicted and written at a time
MIN_LENGTH = 4096  # a trace that does not fit under this limit stops the job


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as handle:
        return {json.loads(line)["id"] for line in handle if line.strip()}


def read_slice(predictor, records, max_length: int, budget: int) -> tuple[list[dict], float]:
    """Each record's line, and the seconds spent predicting (not tokenizing)."""
    import torch

    from clef_tune.encoding import prepare_items
    from clef_tune.evaluation import predict_logits

    def run(batch, limit: int):
        items = prepare_items(predictor.clef_code, predictor.processor, batch, limit)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        logits = predict_logits(
            predictor.model,
            items,
            predictor.clef_code,
            predictor.processor,
            max_length=limit,
            max_tokens_per_batch=budget,
            device=predictor.device,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        lines = [
            {"id": item.record.id, "tokens": item.length, "limit": limit, "logits": row}
            for item, row in zip(items, logits, strict=True)
        ]
        return lines, seconds

    try:
        return run(records, max_length)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"out of memory on a batch; reading its {len(records)} traces one at a time")
    lines, seconds = [], 0.0
    for record in records:
        limit = max_length
        while True:
            try:
                one, spent = run([record], limit)
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if limit // 2 < MIN_LENGTH:
                    raise
                limit //= 2
                print(f"out of memory on {record.id}; reading it under {limit:,} tokens")
        lines += one
        seconds += spent
    return lines, seconds


def score(
    adapter_dir: str, arm: str, input_dir: str, out_dir: str, max_length: str, sets: str = ""
) -> None:
    """`sets` is a comma-separated list of test sets to read; empty means every one present."""
    import torch

    from clef_tune.data import load_file
    from clef_tune.predictor import Predictor

    limit = int(max_length)
    predictor = Predictor.load(adapter_dir, base_model_path=MODEL_MOUNT, max_length=limit)
    config = predictor.info.config
    budget = config.train.max_tokens_per_batch
    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wanted = [name for name in sets.split(",") if name] or list(SETS)
    unknown = sorted(set(wanted) - set(SETS))
    if unknown:
        raise SystemExit(f"unknown test set: {', '.join(unknown)}")
    absent = [name for name in wanted if not (Path(input_dir) / f"{name}.csv").exists()]
    if sets and absent:  # a set asked for by name: stop now, not after hours on the others
        raise SystemExit(f"no file in {input_dir} for: {', '.join(absent)}")
    for name in wanted:
        source = Path(input_dir) / f"{name}.csv"
        if not source.exists():
            continue
        result = load_file(config, str(source.resolve()), for_prediction=True)
        errors = [issue for issue in result.issues if issue.level == "error"]
        if errors:
            raise SystemExit(f"cannot read {source}:\n" + "\n".join(str(e) for e in errors))
        lines_path = out / f"{arm}-{name}.jsonl"
        timing_path = out / f"{arm}-{name}-timing.json"
        done = done_ids(lines_path)
        todo = [record for record in result.records if record.id not in done]
        timing = {"rows": 0, "tokens": 0, "seconds": 0.0}
        if done and timing_path.exists():
            timing = json.loads(timing_path.read_text(encoding="utf-8"))
        questions = result.records[0].questions if result.records else {}
        timing |= {
            "gpu": gpu,
            "max_length": limit,
            "options": {qid: question.option_ids() for qid, question in questions.items()},
            "temperatures": predictor.temperatures,
        }
        print(f"{arm} {name}: {len(todo)} traces to read, {len(done)} already read", flush=True)
        for start in range(0, len(todo), SLICE):
            lines, seconds = read_slice(predictor, todo[start : start + SLICE], limit, budget)
            # Open, append and close each time: the bucket mount uploads a file when it closes.
            with lines_path.open("a", encoding="utf-8", newline="\n") as handle:
                for line in lines:
                    handle.write(json.dumps(line) + "\n")
            timing["rows"] += len(lines)
            timing["tokens"] += sum(line["tokens"] for line in lines)
            timing["seconds"] += seconds
            timing_path.write_text(json.dumps(timing), encoding="utf-8")
            rate = timing["tokens"] / timing["seconds"] if timing["seconds"] else 0.0
            print(
                f"{arm} {name}: {min(start + SLICE, len(todo))} of {len(todo)} | "
                f"{rate:,.0f} tokens/s",
                flush=True,
            )
        if not todo:
            timing_path.write_text(json.dumps(timing), encoding="utf-8")


def main() -> None:
    command, *args = sys.argv[1:]
    {"score": score}[command](*args)


if __name__ == "__main__":
    main()
