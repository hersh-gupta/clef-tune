"""Generate the UI's two extra samples, examples/claims_triage.csv and
examples/server_logs.csv: 200 synthetic rows each, deterministic, with the answers following
from the text so a model can learn them. (support_tickets.csv comes from make_example.py.)"""

from __future__ import annotations

import csv
import random
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
ROWS = 200

CLAIMS = {
    "vehicle": [
        "Rear-ended at a traffic light on {road}; bumper and trunk damaged.",
        "Hail damage to the roof and windshield after the storm on {day}.",
        "Scraped a pillar in a parking garage; door panel dented.",
        "Car stolen from outside the office on {day}.",
    ],
    "property": [
        "Burst pipe flooded the kitchen floor on {day}.",
        "Storm took off part of the roof; water damage in the attic.",
        "Break-in through the back door; laptop and TV taken.",
        "Kitchen fire damaged the cabinets and the ceiling.",
    ],
    "injury": [
        "Slipped on a wet floor in a shop and hurt a wrist.",
        "Twisted an ankle on a broken step outside the rented flat.",
        "Neck pain after a low-speed collision on {road}.",
        "Cut a hand on broken glass during the break-in.",
    ],
}
WARNING_SIGNS = [
    "Paid in cash, receipt lost.",
    "No witnesses and no police report.",
    "The policy started three weeks before the incident.",
    "Asks for payment to a new bank account.",
]
ROADS = ["Main Street", "the ring road", "Harbour Road", "Elm Avenue"]
DAYS = ["Monday", "Friday night", "the weekend", "Tuesday morning"]

LOGS = {
    "api": [
        "GET /v1/orders 200 in {ms} ms",
        "GET /v1/orders slow: {ms} ms (budget 800 ms)",
        "POST /v1/orders 500 Internal Server Error (request {rid})",
        "API gateway: all upstreams unhealthy, returning 503",
    ],
    "database": [
        "Checkpoint complete: wrote {n} buffers",
        "Replication lag {s}s on replica db-2",
        "Deadlock detected on table orders; transaction rolled back",
        "Disk full on db-1: could not extend file",
    ],
    "auth": [
        "User {user} signed in",
        "{n} failed sign-in attempts for {user} in 5 minutes",
        "Token signing key could not be loaded; using the cached key",
        "Auth service down: cannot verify any session",
    ],
    "payments": [
        "Charge {rid} captured: ${amount}",
        "Card network retry for charge {rid}",
        "Charge {rid} failed: processor timeout",
        "Payment processor unreachable for 10 minutes; every charge failing",
    ],
}
HOSTS = ["web-01", "web-02", "worker-01", "db-1", "edge-03"]
USERS = ["amira", "jonas", "li.wei", "sofia", "tom"]


def claims_rows(rng: random.Random) -> list[dict]:
    rows = []
    for i in range(1, ROWS + 1):
        claim_type = rng.choices(list(CLAIMS), weights=[0.45, 0.35, 0.2])[0]
        text = rng.choice(CLAIMS[claim_type]).format(road=rng.choice(ROADS), day=rng.choice(DAYS))
        warning = rng.random() < 0.18
        if warning:
            text = f"{text} {rng.choice(WARNING_SIGNS)}"
        amount = rng.choice([800, 1500, 2400, 4000, 6500, 9000, 14000, 22000, 35000])
        years = rng.choice([0, 1, 2, 4, 7, 12])
        reviewer = (
            "fraud review"
            if warning
            else "medical review"
            if claim_type == "injury"
            else ("standard")
        )
        if warning or amount > 20000:
            risk = 2
        elif amount > 5000 or years == 0:
            risk = 1
        else:
            risk = 0
        pay = not warning and amount <= 15000
        rows.append(
            {
                "claim_id": f"C{i:04d}",
                "claim_type": claim_type,
                "amount": amount,
                "policy_years": years,
                "description": text,
                "reviewer": reviewer,
                "risk": risk,
                "pay_in_full": "yes" if pay else "no",
            }
        )
    return rows


def logs_rows(rng: random.Random) -> list[dict]:
    rows = []
    for i in range(1, ROWS + 1):
        component = rng.choice(list(LOGS))
        severity = rng.choices([0, 1, 2, 3], weights=[0.35, 0.3, 0.22, 0.13])[0]
        fill = {
            "ms": rng.randint(40, 4000),
            "rid": f"req_{rng.randint(10000, 99999)}",
            "n": rng.randint(3, 900),
            "s": rng.randint(5, 300),
            "user": rng.choice(USERS),
            "amount": rng.randint(5, 900),
        }
        message = LOGS[component][severity].format(**fill)
        act = severity == 3 or (severity == 2 and component in ("payments", "auth"))
        rows.append(
            {
                "log_id": f"L{i:04d}",
                "host": rng.choice(HOSTS),
                "message": message,
                "component": component,
                "severity": severity,
                "act_now": "yes" if act else "no",
            }
        )
    return rows


GENERATORS = {"claims_triage": (claims_rows, 11), "server_logs": (logs_rows, 13)}


def write_sample(name: str, out_dir: Path = EXAMPLES) -> Path:
    make, seed = GENERATORS[name]
    rows = make(random.Random(seed))
    out = Path(out_dir) / f"{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return out


def main() -> None:
    for name in GENERATORS:
        write_sample(name)


if __name__ == "__main__":
    main()
