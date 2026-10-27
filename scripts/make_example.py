"""Generate examples/support_tickets.csv: 200 synthetic support tickets (deterministic)."""

from __future__ import annotations

import csv
import random
from pathlib import Path

TEMPLATES = {
    "billing": [
        ("Charged twice for {plan}", "I was billed two times this month for the {plan} plan."),
        ("Invoice {n} looks wrong", "The total on invoice {n} doesn't match our contract."),
        ("Card declined", "My card keeps getting declined when I renew {plan}."),
        ("Refund request", "We cancelled {plan} last week but were still charged."),
    ],
    "technical": [
        ("API returning 500 errors", "Since {time} our calls to the API return HTTP 500."),
        ("Dashboard won't load", "The dashboard spins forever after login since {time}."),
        ("Webhook delays", "Webhooks are arriving {n} seconds late."),
        ("SDK crash on startup", "The Python SDK crashes on import after upgrading."),
    ],
    "account": [
        ("Can't reset password", "The password reset email never arrives."),
        ("Add a teammate", "How do I invite a new teammate to our {plan} workspace?"),
        ("Change account owner", "Our admin left; please move ownership to me."),
        ("Close my account", "Please delete our account and all data."),
    ],
}
OUTAGE_SUBJECTS = {"API returning 500 errors", "Dashboard won't load"}
PLANS = ["Starter", "Pro", "Enterprise"]
TIMES = ["this morning", "an hour ago", "yesterday", "10:00 UTC"]
TIERS = ["free", "pro", "enterprise"]
URGENT_WORDS = ["Please fix ASAP.", "This is urgent.", "This is blocking our launch."]
OUT = Path(__file__).resolve().parents[1] / "examples" / "support_tickets.csv"


def main() -> None:
    rng = random.Random(7)
    rows = []
    for i in range(1, 201):
        department = rng.choices(list(TEMPLATES), weights=[0.4, 0.4, 0.2])[0]
        subject, body = rng.choice(TEMPLATES[department])
        fill = {"plan": rng.choice(PLANS), "n": rng.randint(2, 9999), "time": rng.choice(TIMES)}
        tier = rng.choice(TIERS)
        outage = subject in OUTAGE_SUBJECTS and rng.random() < 0.85
        urgent_words = rng.random() < 0.3
        text = body.format(**fill)
        if urgent_words:
            text = f"{text} {rng.choice(URGENT_WORDS)}"
        if outage or (urgent_words and tier == "enterprise"):
            urgency = 2
        elif urgent_words or department == "technical":
            urgency = 1
        else:
            urgency = 0
        rows.append(
            {
                "ticket_id": f"T{i:04d}",
                "subject": subject.format(**fill),
                "body": text,
                "customer_tier": tier,
                "department": department,
                "urgency": urgency,
                "outage": "yes" if outage else "no",
            }
        )
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
