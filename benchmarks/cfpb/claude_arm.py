"""Step 3: the frontier-LLM contenders, Claude Opus 5.5 with instructions only ("zero") and with a
fixed set of worked examples ("few").

  estimate  count input tokens with the API's free token counter and price a run
  run       answer every complaint in a split (resumable: rows already answered are skipped)
  export    write results/answers/claude-<variant>.csv for the scorer

Both variants see the same question wording and choice descriptions as Clef (from the YAML) and
only the complaint text. Answers are constrained to the choices by structured outputs. There is
no refusal fallback, so every answer is Opus 5.5's own; a refusal or unreadable reply counts as
wrong. The API key comes from ANTHROPIC_API_KEY (this process, else the Windows Machine or User
scope) and is never printed or written.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.cfpb.common import ANSWERS, DATA, PREPARED, SEED, machine_env

MODEL = "claude-opus-5-5"
EFFORT = "medium"  # the model's default, set explicitly
MAX_TOKENS = 8000
CONCURRENCY = 8
FEW_SHOT = 30
PRICES = {  # USD per million tokens, Claude Opus 5.5 list prices on 2026-11-03
    "input": 4.00,
    "output": 20.00,
    "cache_write": 5.00,  # 1.25x input, 5-minute cache
    "cache_read": 0.20,
    "batch_discount": 0.5,
    "date": "2026-11-03",
}
OUT = DATA / "claude"
QUESTIONS = ("issue", "product", "outcome")


def load_schema() -> dict:
    from clef_tune.config import load_config

    config = load_config(PREPARED / "cfpb_flash.yaml")
    return config.schema_def["questions"]


def read_rows(split: str) -> list[dict]:
    with (PREPARED / f"{split}.csv").open(encoding="utf-8", newline="") as rows:
        return list(csv.DictReader(rows))


def instructions(questions: dict) -> str:
    lines = [
        "You will read one customer complaint about a bank and answer three questions about it.",
        "For each question, choose exactly one of its options, and say how confident you are that "
        "your choice is right, from 0 to 100.",
    ]
    for qid, question in questions.items():
        lines += ["", f'Question "{qid}": {question["instructions"]}']
        lines += [f"- {option}: {text}" for option, text in question["criteria"].items()]
    return "\n".join(lines)


def pick_examples(train: list[dict], questions: dict, n: int = FEW_SHOT) -> list[dict]:
    """A seeded random draw from the training rows, redrawn until every option of every question
    appears at least once. Never hand-picked."""
    rng = random.Random(SEED)
    ordered = sorted(train, key=lambda r: r["complaint_id"])
    for _ in range(10_000):
        draw = rng.sample(ordered, n)
        if all({r[q] for r in draw} >= set(questions[q]["criteria"]) for q in QUESTIONS):
            return draw
    raise RuntimeError("no draw covered every option")


def answer_json(row: dict) -> str:
    return json.dumps({q: row[q] for q in QUESTIONS})


def system_prompt(questions: dict, examples: list[dict] | None) -> str:
    text = instructions(questions)
    if not examples:
        return text
    parts = [text, "", f"Here are {len(examples)} past complaints with their correct answers."]
    for k, row in enumerate(examples, 1):
        parts += ["", f"Example {k}", "Complaint:", row["complaint"]]
        parts.append("Answers: " + answer_json(row))
    return "\n".join(parts)


def output_schema(questions: dict) -> dict:
    properties = {}
    for qid, question in questions.items():
        properties[qid] = {"type": "string", "enum": list(question["criteria"])}
        properties[f"{qid}_confidence"] = {"type": "integer"}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def user_message(row: dict) -> list[dict]:
    return [{"role": "user", "content": "Complaint:\n" + row["complaint"]}]


def client():
    import anthropic

    key = machine_env("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY is not set (process, Machine or User scope)")
    return anthropic.Anthropic(api_key=key, max_retries=6)


def setup(variant: str) -> tuple[dict, str, dict]:
    questions = load_schema()
    examples = pick_examples(read_rows("train"), questions) if variant == "few" else None
    return questions, system_prompt(questions, examples), output_schema(questions)


def system_blocks(system: str, variant: str) -> list[dict]:
    block: dict = {"type": "text", "text": system}
    if variant == "few":
        block["cache_control"] = {"type": "ephemeral"}
    return [block]


# ------------------------------------------------------------------------------------- estimate
def estimate_command(split: str, limit: int | None, output_guess: tuple[int, int]) -> None:
    api = client()
    rows = read_rows(split)[:limit] if limit else read_rows(split)
    user_tokens = []
    lock = threading.Lock()

    def count(row: dict) -> None:
        result = api.messages.count_tokens(model=MODEL, system="x", messages=user_message(row))
        with lock:
            user_tokens.append(result.input_tokens)

    with ThreadPoolExecutor(CONCURRENCY) as pool:
        list(pool.map(count, rows))
    base = api.messages.count_tokens(
        model=MODEL, system="x", messages=[{"role": "user", "content": "x"}]
    ).input_tokens
    total_user = sum(user_tokens) - base * len(rows) + len(rows)  # each count added "x" twice
    low, high = output_guess
    for variant in ("zero", "few"):
        questions, system, _ = setup(variant)
        sys_tokens = api.messages.count_tokens(
            model=MODEL, system=system, messages=[{"role": "user", "content": "x"}]
        ).input_tokens
        n = len(rows)
        if variant == "few":
            input_cost = (
                sys_tokens * PRICES["cache_write"] * 2  # a write per 5-minute window, roughly
                + sys_tokens * (n - 2) * PRICES["cache_read"]
                + total_user * PRICES["input"]
            ) / 1e6
        else:
            input_cost = (sys_tokens * n + total_user) * PRICES["input"] / 1e6
        out_low, out_high = (n * x * PRICES["output"] / 1e6 for x in (low, high))
        print(
            f"{variant}: {n} rows, system {sys_tokens} tokens, complaints {total_user} tokens; "
            f"input ${input_cost:.2f}, output ${out_low:.2f}-${out_high:.2f} "
            f"(at {low}-{high} output tokens per answer, thinking included); "
            f"total ${input_cost + out_low:.2f}-${input_cost + out_high:.2f}"
        )


# ------------------------------------------------------------------------------------------ run
def ask(api, row: dict, system: list[dict], schema: dict) -> dict:
    started = time.perf_counter()
    response = api.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=user_message(row),
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": schema}},
    )
    seconds = time.perf_counter() - started
    usage = response.usage
    record = {
        "complaint_id": row["complaint_id"],
        "stop_reason": response.stop_reason,
        "seconds": round(seconds, 3),
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_creation_input_tokens": usage.cache_creation_input_tokens or 0,
            "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
        },
        "answers": None,
        "request_id": response._request_id,
    }
    if response.stop_reason == "end_turn":
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            record["answers"] = json.loads(text)
        except json.JSONDecodeError:
            record["error"] = "unreadable reply"
    elif response.stop_reason == "refusal":
        details = response.stop_details
        record["error"] = f"refusal ({getattr(details, 'category', None)})"
    else:
        record["error"] = f"stopped: {response.stop_reason}"
    return record


def run_command(variant: str, split: str, limit: int | None) -> None:
    api = client()
    _, system, schema = setup(variant)
    blocks = system_blocks(system, variant)
    rows = read_rows(split)[:limit] if limit else read_rows(split)
    path = OUT / variant / f"{split}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = (
        {json.loads(line)["complaint_id"] for line in path.open(encoding="utf-8")}
        if (path.exists())
        else set()
    )
    todo = [r for r in rows if r["complaint_id"] not in done]
    if variant == "few" and todo:  # write the cache once before the parallel requests
        first, todo = todo[0], todo[1:]
        _append(path, ask(api, first, blocks, schema))
    lock = threading.Lock()
    started = time.perf_counter()
    with ThreadPoolExecutor(CONCURRENCY) as pool:
        futures = [pool.submit(ask, api, row, blocks, schema) for row in todo]
        for k, future in enumerate(as_completed(futures), 1):
            record = future.result()
            with lock:
                _append(path, record)
            if k % 25 == 0:
                print(f"{variant}/{split}: {k}/{len(todo)}", flush=True)
    wall = time.perf_counter() - started
    timing = OUT / variant / f"{split}-timing.json"
    if todo:
        timing.write_text(
            json.dumps({"rows": len(todo), "wall_seconds": wall, "concurrency": CONCURRENCY}),
            encoding="utf-8",
        )
    print(f"{variant}/{split}: {len(todo)} answered in {wall:.0f} s -> {path}")


def _append(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------------------- export
def records(variant: str, split: str) -> list[dict]:
    path = OUT / variant / f"{split}.jsonl"
    latest: dict[str, dict] = {}
    for line in path.open(encoding="utf-8"):
        record = json.loads(line)
        latest[record["complaint_id"]] = record
    return list(latest.values())


def usage_summary(items: list[dict]) -> dict:
    keys = (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    )
    totals = {k: sum(r["usage"][k] for r in items) for k in keys}
    cost = (
        totals["input_tokens"] * PRICES["input"]
        + totals["output_tokens"] * PRICES["output"]
        + totals["cache_creation_input_tokens"] * PRICES["cache_write"]
        + totals["cache_read_input_tokens"] * PRICES["cache_read"]
    ) / 1e6
    seconds = sorted(r["seconds"] for r in items)
    return {
        **totals,
        "rows": len(items),
        "usd": cost,
        "usd_per_1000": cost / len(items) * 1000,
        "batch_usd_per_1000": cost / len(items) * 1000 * PRICES["batch_discount"],
        "median_seconds": seconds[len(seconds) // 2],
        "failed": sum(1 for r in items if r.get("answers") is None),
        "prices": PRICES,
    }


def export_command(variant: str, split: str) -> None:
    items = records(variant, split)
    truth = {r["complaint_id"]: r for r in read_rows(split)}
    ANSWERS.mkdir(parents=True, exist_ok=True)
    path = (
        ANSWERS / f"claude-{variant}.csv"
        if split == "test"
        else (OUT / variant / f"{split}-answers.csv")
    )
    header = ["complaint_id", "ok"]
    for q in QUESTIONS:
        header += [f"{q}_answer", f"{q}_confidence", f"{q}_true"]
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(header)
        for record in sorted(items, key=lambda r: r["complaint_id"]):
            answers = record.get("answers") or {}
            line = [record["complaint_id"], "1" if answers else "0"]
            for q in QUESTIONS:
                confidence = answers.get(f"{q}_confidence")
                if confidence is not None:
                    confidence = min(max(int(confidence), 0), 100) / 100
                line += [answers.get(q, ""), "" if confidence is None else confidence]
                line.append(truth[record["complaint_id"]][q])
            writer.writerow(line)
    summary = usage_summary(items)
    timing = OUT / variant / f"{split}-timing.json"
    if timing.exists():
        t = json.loads(timing.read_text(encoding="utf-8"))
        summary["complaints_per_minute"] = t["rows"] / t["wall_seconds"] * 60
        summary["concurrency"] = t["concurrency"]
    (OUT / variant / f"{split}-summary.json").write_text(json.dumps(summary, indent=2), "utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "prices"}, indent=2))
    print("->", path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    est = sub.add_parser("estimate")
    est.add_argument("--split", default="test")
    est.add_argument("--limit", type=int)
    est.add_argument("--output-tokens", type=int, nargs=2, default=(150, 1500))
    for name in ("run", "export"):
        p = sub.add_parser(name)
        p.add_argument("variant", choices=["zero", "few"])
        p.add_argument("--split", default="test", choices=["validation", "test"])
        if name == "run":
            p.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.command == "estimate":
        estimate_command(args.split, args.limit, tuple(args.output_tokens))
    elif args.command == "run":
        run_command(args.variant, args.split, args.limit)
    else:
        export_command(args.variant, args.split)


if __name__ == "__main__":
    main()
