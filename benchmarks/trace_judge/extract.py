"""Step 1 of the trace-judge pilot: pull traces out of the two parquet sources.

  METR's MALT           data/raw/malt/data/public-*.parquet   -> data/interim/malt.jsonl
  the Storf bundle      data/raw/storf_bundle/*.parquet       -> data/interim/storf_synthetic.jsonl
                                                                 data/interim/controlarena.jsonl

Every trace is written as plain text with the same tags prepare.py uses for the other sets:
[AGENT INSTRUCTIONS], [USER], [AGENT], [AGENT ACTION], [ENVIRONMENT].

MALT: only Claude 3.7 Sonnet runs are kept, because every run that was told to underperform
comes from that model; normal runs from other models would let a judge learn the model instead
of the behaviour. The judge must not see the answer, so three things are left out of each run:
  - the system message (in prompted runs it holds the instruction to underperform);
  - the agent's private reasoning (`completion_type == "reasoning_completion"`);
  - the scaffold's notices ("So far in this attempt at the task, you have used ...", and
    "No valid function call found ..."): their wording differs between the prompted and the
    normal batches.

Needs pyarrow, which the project doesn't install:
  uv run --no-project --with pyarrow python -I benchmarks/trace_judge/extract.py
"""

from __future__ import annotations

import glob
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"  # git-ignored
INTERIM = HERE / "data" / "interim"

MALT_MODEL = "claude-3-7-sonnet-20250219"
MALT_UNDERPERFORMING = {"sabotage", "match_weaker_model", "partial_problem_solving"}
MALT_NOTICES = (
    re.compile(r"^\s*So far in this attempt at the task, you have used"),
    re.compile(r"^\s*No valid function call found in the last response"),
)


def text(value) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def call_line(name, arguments) -> str:
    return f"{name}({text(arguments)})"


def malt_trace(messages: list[dict], dropped: Counter) -> str:
    parts: list[str] = []
    for message in messages:
        role = message["role"]
        content = text(message.get("content")).strip()
        kind = (message.get("metadata") or {}).get("completion_type")
        if role == "system":
            dropped["system message"] += 1
            continue
        if kind == "reasoning_completion":
            dropped["private reasoning"] += 1
            continue
        if any(pattern.match(content) for pattern in MALT_NOTICES):
            dropped["scaffold notice"] += 1
            continue
        if role == "user":
            parts.append(f"[USER]\n{content}")
        elif role == "assistant":
            if content:
                parts.append(f"[AGENT]\n{content}")
            call = message.get("function_call")
            if call:
                parts.append(
                    f"[AGENT ACTION]\n{call_line(call.get('name'), call.get('arguments'))}"
                )
        elif role == "function":
            parts.append(f"[ENVIRONMENT]\n{content}")
        else:
            parts.append(f"[{str(role).upper()}]\n{content}")
    return "\n\n".join(parts)


def extract_malt() -> None:
    out = INTERIM / "malt.jsonl"
    kept: Counter = Counter()
    dropped: Counter = Counter()
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for path in sorted(glob.glob(str(RAW / "malt" / "data" / "public-*.parquet"))):
            batches = pq.ParquetFile(path).iter_batches(
                batch_size=8, columns=["samples", "metadata"]
            )
            for batch in batches:
                samples = batch.column("samples")
                for index, meta in enumerate(batch.column("metadata").to_pylist()):
                    labels = set(meta["labels"])
                    if meta["model"] != MALT_MODEL or len(labels) != 1:
                        continue
                    label = next(iter(labels))
                    if label != "normal" and label not in MALT_UNDERPERFORMING:
                        continue
                    run = samples[index]
                    last = run[len(run) - 1].as_py()  # the last generation's input is the whole run
                    messages = list(last["input"]) + list((last["output"] or [[]])[0] or [])
                    record = {
                        "id": f"malt-{meta['run_id']}",
                        "trace": malt_trace(messages, dropped),
                        "source": "malt",
                        "group": meta["task_id"].split("/")[0],
                        "detail": label,
                        "task_id": meta["task_id"],
                        "run_source": meta["run_source"],
                        "manually_reviewed": meta["manually_reviewed"],
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    kept[label] += 1
            print(Path(path).name, dict(kept), flush=True)
    print("MALT kept:", dict(kept), "| left out of traces:", dict(dropped))


def storf_trace(messages: list[dict]) -> str:
    parts: list[str] = []
    for message in messages:
        role = message.get("role")
        content = text(message.get("content")).strip()
        if role == "system":
            if content:
                parts.append(f"[AGENT INSTRUCTIONS]\n{content}")
        elif role == "user":
            parts.append(f"[USER]\n{content}")
        elif role == "assistant":
            if content:
                parts.append(f"[AGENT]\n{content}")
            calls = [
                call_line(
                    call.get("name") or call.get("function"),
                    call.get("arguments", call.get("args")),
                )
                for call in message.get("tool_calls") or []
            ]
            if calls:
                parts.append("[AGENT ACTION]\n" + "\n".join(calls))
        elif role == "tool":
            parts.append(f"[ENVIRONMENT]\n{content}")
        else:
            parts.append(f"[{str(role).upper()}]\n{content}")
    return "\n\n".join(parts)


CONTROL_ARENA_ID = re.compile(
    r"^(agentdojo|bash|iac|rogue_eval|shade)_(.+?)_(positive|negative)\d*_"
)


def extract_storf() -> None:
    counts: Counter = Counter()
    jobs = (
        ("storf_synthetic.jsonl", ("gemini_synthetic", "unified_dataset")),
        ("controlarena.jsonl", ("control_arena",)),
    )
    for out_name, names in jobs:
        with (INTERIM / out_name).open("w", encoding="utf-8", newline="\n") as handle:
            for name in names:
                source = pq.ParquetFile(RAW / "storf_bundle" / f"{name}.parquet")
                for batch in source.iter_batches(
                    batch_size=64, columns=["id", "messages", "ground_truth"]
                ):
                    for row in batch.to_pylist():
                        group, detail = name, ""
                        if name == "control_arena":
                            match = CONTROL_ARENA_ID.match(row["id"])
                            if not match:
                                raise SystemExit(f"unrecognised ControlArena id: {row['id']}")
                            group, detail = match.group(1), match.group(2)
                        record = {
                            "id": f"storf-{row['id']}",
                            "trace": storf_trace(json.loads(row["messages"])),
                            "source": "controlarena"
                            if name == "control_arena"
                            else f"storf-{name}",
                            "group": group,
                            "detail": detail,
                            "hidden_goal": bool(row["ground_truth"]),
                        }
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        counts[(record["source"], group, record["hidden_goal"])] += 1
    for key in sorted(counts):
        print("Storf", key, counts[key])


def main() -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    extract_storf()
    extract_malt()


if __name__ == "__main__":
    main()
