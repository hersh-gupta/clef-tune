"""Step 8: put the benchmark on the Hugging Face Hub, privately.

  dataset   create hvgupta1/cfpb-bank-complaints-benchmark (private) with the three CSVs, the
            clef-tune configs and a data card
  cards     add a "Benchmark" section to each tuned model's card, linking the dataset

Nothing here makes a repo public; that is the user's call, made on the Hub.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.cfpb.common import BANK, PREPARED, RESULTS, machine_env

DATASET = "hvgupta1/cfpb-bank-complaints-benchmark"
GITHUB = "https://github.com/hersh-gupta/clef-tune/tree/main/benchmarks"
MODELS = {
    "hvgupta1/clef-flash-cfpb-complaints": ("clef-flash-tuned", "clef-flash-untrained"),
    "hvgupta1/clef-cfpb-complaints": ("clef-27b-tuned", "clef-27b-untrained"),
}
SECTION = "<!-- clef-tune benchmark -->"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def data_card(split: dict) -> str:
    rows, dates, labels = split["rows"], split["dates"], split["labels"]

    def counts(part: str, question: str) -> str:
        return ", ".join(f"{k}: {v}" for k, v in labels[part][question].items())

    return f"""---
license: cc0-1.0
language: [en]
pretty_name: CFPB bank complaints (clef-tune benchmark)
task_categories: [text-classification]
tags: [cfpb, complaints, banking, customer-support, benchmark, clef-tune]
size_categories: [1K<n<10K]
---

# CFPB bank complaints: a decision benchmark

Real consumer complaints about one large US bank, each with three decisions to predict: what the
complaint is mainly about, which product it concerns, and how the bank closed it. Built for the
[clef-tune benchmark]({GITHUB}), which compares fine-tuned Clef models with a prompted frontier
LLM.

## Source and licence

- **Text and labels:** the US Consumer Financial Protection Bureau's
  [Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/),
  from its [narratives archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/)
  (monthly files, downloaded 2026-11-03). Labels were checked against the CFPB's search API: all
  matched.
- **Licence:** CC0 (US government work; the API reports `license: CC0`). The CFPB removes
  personal details before publishing; masking such as `XXXX` is kept as published.
- The CFPB stopped publishing complaint text on 14 August 2026.

## What's in it

| File | Rows | Received |
|---|---|---|
| `train.csv` | {rows["train"]:,} | {dates["train"][0]} to {dates["train"][1]} |
| `validation.csv` | {rows["validation"]:,} | {dates["validation"][0]} to {dates["validation"][1]} |
| `test.csv` | {rows["test"]:,} | {dates["test"][0]} to {dates["test"][1]} |

Columns: `complaint_id` (the CFPB's), `date_received`, `complaint` (the consumer's own words),
`issue`, `product`, `outcome`. `cfpb_flash.yaml` and `cfpb_27b.yaml` are the clef-tune configs
used to train on it.

**Scope:** complaints about Bank of America ("{BANK}" in the CFPB data)
on checking or savings accounts, credit cards and money transfers, with published text and a final company response. Credit reports, debt collection,
loans, mortgages and prepaid cards are left out. {split["dropped_too_long"]["train_months"]}
complaints longer than {split["max_chars"]:,} characters were dropped from the training months
(none from the test month).

**Split by date:** train and validation are a seeded random split of complaints received from
January 2025 to May 2026. June 2026 is left out as a buffer around current Claude models' June
2026 training-data cutoff. The test set is every qualifying complaint received in July 2026; none
from August had published text.

## The three questions

- `issue`: the CFPB issue the consumer picked, folded to the 7 most common in the training months
  plus `other` (CFPB's "Closing an account" and "Closing your account" are merged).
  Test: {counts("test", "issue")}.
- `product`: Test: {counts("test", "product")}.
- `outcome`: the bank's recorded response: "Closed with explanation" -> `explanation only`,
  "Closed with non-monetary relief" -> `fix without money`, "Closed with monetary relief" ->
  `money back`. Test: {counts("test", "outcome")}.

## Caveats

- `issue` and `product` are chosen by the consumer on the CFPB form, so they are noisy.
- `outcome` is what the bank recorded, not a judgement of what was right.
- One bank, and a one-month test set of {rows["test"]} complaints.

## Rebuild it

`benchmarks/cfpb/fetch.py` and `prepare.py` in [clef-tune]({GITHUB}) rebuild these files from
the CFPB's archive.
"""


def dataset_command() -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=machine_env("HF_TOKEN"))
    split = json.loads((PREPARED / "split.json").read_text(encoding="utf-8"))
    api.create_repo(DATASET, repo_type="dataset", private=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for name in ("train.csv", "validation.csv", "test.csv", "cfpb_flash.yaml", "cfpb_27b.yaml"):
            (folder / name).write_bytes((PREPARED / name).read_bytes())
        (folder / "README.md").write_text(data_card(split), encoding="utf-8")
        api.upload_folder(
            repo_id=DATASET,
            repo_type="dataset",
            folder_path=folder,
            commit_message="Add the CFPB bank-complaint benchmark data",
        )
    print(f"https://huggingface.co/datasets/{DATASET} (private)")


def section(tuned: str, untrained: str, metrics: dict) -> str:
    c = metrics["contenders"]
    rows = [
        ("This model", c[tuned]),
        ("The same model out of the box", c[untrained]),
        ("Claude Opus 5.5, with 30 examples", c["claude-few"]),
        ("Claude Opus 5.5, instructions only", c["claude-zero"]),
    ]
    lines = [
        SECTION,
        "## Benchmark",
        "",
        f"Trained on [{DATASET}](https://huggingface.co/datasets/{DATASET}): "
        "real complaints to a large US bank from the CFPB. Tested on "
        f"{metrics['test_rows']} complaints received in July 2026, never used in training.",
        "",
        "| | All three right | Issue | Product | Outcome |",
        "|---|---|---|---|---|",
    ]
    for name, m in rows:
        q = m["questions"]
        low, high = (100 * v for v in m["all_three_right_95"])
        lines.append(
            f"| {name} | {_pct(m['all_three_right'])} ({low:.1f}–{high:.1f}) "
            f"| {_pct(q['issue']['accuracy'])} | {_pct(q['product']['accuracy'])} "
            f"| {_pct(q['outcome']['accuracy'])} |"
        )
    lines += [
        "",
        f"Method, costs, speed and limits: [benchmarks/README.md]({GITHUB}/README.md).",
        "",
    ]
    return "\n".join(lines)


def cards_command() -> None:
    from huggingface_hub import HfApi, hf_hub_download

    token = machine_env("HF_TOKEN")
    api = HfApi(token=token)
    metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    for repo, (tuned, untrained) in MODELS.items():
        if tuned not in metrics["contenders"]:
            print(f"skip {repo}: no results yet")
            continue
        card = Path(hf_hub_download(repo, "README.md", token=token)).read_text(encoding="utf-8")
        card = card.split(SECTION)[0].rstrip() + "\n\n" + section(tuned, untrained, metrics)
        card = card.replace("tags:\n", f"datasets:\n- {DATASET}\ntags:\n", 1)
        api.upload_file(
            path_or_fileobj=card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo,
            commit_message="Add the CFPB benchmark results and link the dataset",
        )
        print(f"https://huggingface.co/{repo} card updated")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("dataset")
    sub.add_parser("cards")
    args = parser.parse_args()
    if args.command == "dataset":
        dataset_command()
    else:
        cards_command()


if __name__ == "__main__":
    main()
