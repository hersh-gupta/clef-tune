"""Step 2: turn the downloaded CFPB archive into the benchmark's files.

  pool    read every archive file, keep this bank's complaints with a narrative in scope, and
          check them against the CFPB search API's labels  -> data/prepared/pool.jsonl
  split   choose the issue choices, split by date, sample, and write train/validation/test CSVs,
          the two clef-tune YAML configs and the dataset card  -> data/prepared/

Run with `python -I` semantics in mind: the archive is data only and nothing in it is executed.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.cfpb.common import (
    BANK,
    OUTCOMES,
    PREPARED,
    PRODUCTS,
    RAW,
    SEED,
    TEST_ROWS,
    TEST_WINDOW,
    TRAIN_ROWS,
    TRAIN_WINDOW,
    VALIDATION_ROWS,
)

POOL = PREPARED / "pool.jsonl"
MAX_CHARS = 12_000  # about 3,000 tokens: leaves room for the questions within max_length 4096
MIN_TEST_ROWS = 350  # the user chose the 378 July complaints (August's text was never published)
ISSUE_CHOICES = 7  # the most frequent issues in the training months; the rest become "other"
OTHER = "other"

# The CFPB's issue names -> a short choice id and a plain one-line description. Only the issues
# that make the top ISSUE_CHOICES are used; split() stops if one is missing here.
ISSUES = {
    "Managing an account": (
        "managing an account",
        "Problems running an existing account: deposits, withdrawals, holds, access, statements",
    ),
    "Problem with a purchase shown on your statement": (
        "disputed purchase",
        "A charge on a statement the customer disputes or doesn't recognise",
    ),
    "Closing an account": (
        "closing an account",
        "The bank closed the account, or the customer couldn't close it",
    ),
    "Closing your account": (  # the credit-card form's name for the same issue
        "closing an account",
        "The bank closed the account, or the customer couldn't close it",
    ),
    "Problem with a lender or other company charging your account": (
        "unwanted charge",
        "Another company charged or debited the account without permission",
    ),
    "Fraud or scam": (
        "fraud or scam",
        "Money lost to fraud or a scam, including payments to scammers",
    ),
    "Opening an account": (
        "opening an account",
        "Trouble opening an account, or an account opened without consent",
    ),
    "Other features, terms, or problems": (
        "card features or terms",
        "Other credit card problems: features, rewards, terms, limits",
    ),
    "Fees or interest": (
        "fees or interest",
        "Fees or interest the customer thinks are wrong or unfair",
    ),
    "Getting a credit card": (
        "getting a card",
        "Applying for a card, or a card opened without permission",
    ),
    "Trouble using your card": (
        "trouble using a card",
        "A card that was declined, blocked or didn't work",
    ),
    "Problem when making payments": (
        "making payments",
        "Payments to the card that were lost, late or misapplied",
    ),
    "Problem with a company's investigation into an existing problem": (
        "the bank's investigation",
        "Unhappy with how the bank investigated a dispute or problem",
    ),
    "Advertising and marketing, including promotional offers": (
        "offers and marketing",
        "Promotions, bonuses or marketing that weren't honoured",
    ),
    "Unexpected or other fees": (
        "unexpected fees",
        "Fees for transfers or services the customer didn't expect",
    ),
    "Other transaction problem": (
        "transfer problem",
        "A money transfer that went wrong for another reason",
    ),
    "Money was not available when promised": (
        "money not available",
        "Transferred or deposited money that didn't arrive when promised",
    ),
}
PRODUCT_TEXT = {
    "checking or savings account": "A checking or savings account",
    "credit card": "A credit card",
    "money transfer": "A money transfer, payment app or wire",
}
OUTCOME_TEXT = {
    "explanation only": "The bank explained its position and changed nothing",
    "fix without money": "The bank fixed something for the customer but paid no money",
    "money back": "The bank refunded, credited or paid the customer money",
}


def _in(window: tuple[date, date], day: date) -> bool:
    return window[0] <= day <= window[1]


def archive_rows(path: Path):
    """Every row of the CSV inside one archive file, read straight from the zip."""
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if not member.filename.lower().endswith(".csv"):
                continue
            with archive.open(member) as raw:
                yield from csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))


def keep(row: dict) -> dict | None:
    """The fields we need, if the row is in scope; None otherwise."""
    if row.get("Company") != BANK or row.get("Product") not in PRODUCTS:
        return None
    response = row.get("Company response to consumer", "")
    narrative = (row.get("Consumer complaint narrative") or "").strip()
    if response not in OUTCOMES or not narrative:
        return None
    try:
        received = date.fromisoformat(row["Date received"][:10])
    except (KeyError, ValueError):
        return None
    if not (_in(TRAIN_WINDOW, received) or _in(TEST_WINDOW, received)):
        return None
    return {
        "complaint_id": row["Complaint ID"],
        "date_received": received.isoformat(),
        "cfpb_product": row["Product"],
        "cfpb_issue": row["Issue"],
        "cfpb_response": response,
        "narrative": narrative,
    }


def api_labels() -> dict[str, dict]:
    labels: dict[str, dict] = {}
    for path in sorted((RAW / "api").glob("labels-*.json")):
        for item in json.loads(path.read_text(encoding="utf-8")):
            labels[str(item["complaint_id"])] = item
    return labels


def pool_command() -> None:
    files = sorted(RAW.glob("CCDB_Export_*/*.zip"))
    if not files:
        raise SystemExit("No archive files; run fetch.py download first.")
    found: dict[str, dict] = {}
    for path in files:
        before = len(found)
        for row in archive_rows(path):
            item = keep(row)
            if item is not None:
                found[item["complaint_id"]] = item  # files can overlap; one copy per complaint
        print(f"{path.name}: {len(found) - before} complaints in scope")
    labels = api_labels()
    differ = Counter()
    for item in found.values():
        api = labels.get(item["complaint_id"])
        if api is None:
            differ["not in the API's results"] += 1
            continue
        for ours, theirs in (
            ("cfpb_product", "product"),
            ("cfpb_issue", "issue"),
            ("cfpb_response", "company_response"),
        ):
            if item[ours] != api.get(theirs):
                differ[f"{theirs} differs"] += 1
    PREPARED.mkdir(parents=True, exist_ok=True)
    with POOL.open("w", encoding="utf-8") as out:
        for item in sorted(found.values(), key=lambda i: (i["date_received"], i["complaint_id"])):
            out.write(json.dumps(item) + "\n")
    print(f"{len(found)} complaints in the pool -> {POOL}")
    print("Checked against the search API:", dict(differ) or "all labels match")


def load_pool() -> list[dict]:
    with POOL.open(encoding="utf-8") as lines:
        return [json.loads(line) for line in lines]


def summary_command() -> None:
    pool = load_pool()
    train = [i for i in pool if _in(TRAIN_WINDOW, date.fromisoformat(i["date_received"]))]
    test = [i for i in pool if _in(TEST_WINDOW, date.fromisoformat(i["date_received"]))]
    for name, items in (("training months", train), ("test months", test)):
        long = sum(1 for i in items if len(i["narrative"]) > MAX_CHARS)
        print(f"\n{name}: {len(items)} complaints, {long} longer than {MAX_CHARS} characters")
        for field in ("cfpb_product", "cfpb_response", "cfpb_issue"):
            print(f"  {field}:")
            for value, count in Counter(i[field] for i in items).most_common(12):
                print(f"    {count:5d}  {value}")


COLUMNS = ["complaint_id", "date_received", "complaint", "issue", "product", "outcome"]
QUESTION_TEXT = {
    "issue": "What is this complaint mainly about?",
    "product": "Which product is the complaint about?",
    "outcome": "How did the bank close this complaint?",
}
MODELS = {  # config file -> (model id, run name, Hub repo)
    "cfpb_flash.yaml": (
        "Cloudflare/clef-flash",
        "clef-flash-cfpb-complaints",
        "hvgupta1/clef-flash-cfpb-complaints",
    ),
    "cfpb_27b.yaml": ("Cloudflare/clef", "clef-cfpb-complaints", "hvgupta1/clef-cfpb-complaints"),
}


def issue_choices(train_pool: list[dict]) -> list[str]:
    """The ISSUE_CHOICES most frequent issue ids in the training months, most frequent first."""
    unknown = sorted({i["cfpb_issue"] for i in train_pool} - set(ISSUES))
    counts = Counter(ISSUES[i["cfpb_issue"]][0] for i in train_pool if i["cfpb_issue"] in ISSUES)
    top = [issue for issue, _ in counts.most_common(ISSUE_CHOICES)]
    rest = Counter(i["cfpb_issue"] for i in train_pool if i["cfpb_issue"] in unknown)
    if rest and rest.most_common(1)[0][1] >= counts[top[-1]]:
        raise SystemExit(f"Add a description for {rest.most_common(1)[0][0]!r} to ISSUES")
    return top


def to_row(item: dict, issues: list[str]) -> dict:
    issue = ISSUES.get(item["cfpb_issue"], (OTHER, ""))[0]
    return {
        "complaint_id": item["complaint_id"],
        "date_received": item["date_received"],
        "complaint": item["narrative"],
        "issue": issue if issue in issues else OTHER,
        "product": PRODUCTS[item["cfpb_product"]],
        "outcome": OUTCOMES[item["cfpb_response"]],
    }


def split(pool: list[dict]) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Train and validation from the training months, test from the test months; complaints too
    long for every contender to read are dropped from all three. Seeded, so it is repeatable."""
    days = [(i, date.fromisoformat(i["date_received"])) for i in pool]
    train_pool = [i for i, d in days if _in(TRAIN_WINDOW, d)]
    test_pool = [i for i, d in days if _in(TEST_WINDOW, d)]
    dropped = {
        "train_months": sum(1 for i in train_pool if len(i["narrative"]) > MAX_CHARS),
        "test_months": sum(1 for i in test_pool if len(i["narrative"]) > MAX_CHARS),
    }
    train_pool = [i for i in train_pool if len(i["narrative"]) <= MAX_CHARS]
    test_pool = [i for i in test_pool if len(i["narrative"]) <= MAX_CHARS]
    issues = issue_choices(train_pool)
    rng = random.Random(SEED)
    shuffled = sorted(train_pool, key=lambda i: i["complaint_id"])
    rng.shuffle(shuffled)
    validation = shuffled[:VALIDATION_ROWS]
    train = shuffled[VALIDATION_ROWS : VALIDATION_ROWS + TRAIN_ROWS]
    test = sorted(test_pool, key=lambda i: i["complaint_id"])
    if len(test) > TEST_ROWS:
        test = random.Random(SEED + 1).sample(test, TEST_ROWS)
    if len(test) < MIN_TEST_ROWS:
        raise SystemExit(f"Only {len(test)} test complaints qualify; ask before widening anything")
    info = {
        "pool": {"train_months": len(train_pool), "test_months": len(test_pool)},
        "dropped_too_long": dropped,
        "max_chars": MAX_CHARS,
        "issues": issues,
    }
    rows = [[to_row(i, issues) for i in part] for part in (train, validation, test)]
    return rows[0], rows[1], rows[2], info


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["complaint_id"]))


def schema(issues: list[str]) -> dict:
    descriptions = {choice: text for choice, text in ISSUES.values()}
    issue_criteria = {choice: descriptions[choice] for choice in issues}
    issue_criteria[OTHER] = "Anything else"
    return {
        "questions": {
            "issue": {
                "type": "choice",
                "instructions": QUESTION_TEXT["issue"],
                "criteria": issue_criteria,
            },
            "product": {
                "type": "choice",
                "instructions": QUESTION_TEXT["product"],
                "criteria": dict(PRODUCT_TEXT),
            },
            "outcome": {
                "type": "choice",
                "instructions": QUESTION_TEXT["outcome"],
                "criteria": dict(OUTCOME_TEXT),
            },
        }
    }


def write_configs(issues: list[str]) -> None:
    from clef_tune.config import Config

    for name, (model_id, run_name, repo) in MODELS.items():
        config = Config.model_validate(
            {
                "run_name": run_name,
                "model": {"id": model_id},
                "data": {
                    "train": "train.csv",
                    "eval": "validation.csv",
                    "id_column": "complaint_id",
                    "state_columns": ["complaint"],
                    "row_noun": "complaint",
                },
                "schema": schema(issues),
                "output": {"push_to_hub": repo},
            }
        ).with_preset("balanced")
        config.save(PREPARED / name)


def distribution(rows: list[dict], question: str) -> dict[str, int]:
    return dict(Counter(r[question] for r in rows).most_common())


def split_command() -> None:
    train, validation, test, info = split(load_pool())
    for name, rows in (("train", train), ("validation", validation), ("test", test)):
        write_csv(PREPARED / f"{name}.csv", rows)
    write_configs(info["issues"])
    parts = {"train": train, "validation": validation, "test": test}
    info["rows"] = {name: len(rows) for name, rows in parts.items()}
    info["dates"] = {
        name: [min(r["date_received"] for r in rows), max(r["date_received"] for r in rows)]
        for name, rows in parts.items()
    }
    info["labels"] = {
        name: {q: distribution(rows, q) for q in QUESTION_TEXT} for name, rows in parts.items()
    }
    (PREPARED / "split.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps({k: info[k] for k in ("rows", "dates", "dropped_too_long", "issues")}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pool", help="read the archive into data/prepared/pool.jsonl")
    sub.add_parser("summary", help="show counts in the pool")
    sub.add_parser("split", help="write the CSVs, the configs and split.json")
    args = parser.parse_args()
    if args.command == "pool":
        pool_command()
    elif args.command == "summary":
        summary_command()
    else:
        split_command()


if __name__ == "__main__":
    main()
