"""Settings shared by every step of the CFPB bank-complaint benchmark: the bank, the products,
the date windows of each split, and where each step reads and writes its files."""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = HERE / "data"  # git-ignored
RAW = DATA / "raw"  # each download in its own new folder
PREPARED = DATA / "prepared"
RESULTS = HERE / "results"  # committed
ANSWERS = RESULTS / "answers"

BANK = "BANK OF AMERICA, NATIONAL ASSOCIATION"
PRODUCTS = {  # the CFPB's product name -> this benchmark's choice id
    "Checking or savings account": "checking or savings account",
    "Credit card": "credit card",
    "Money transfer, virtual currency, or money service": "money transfer",
}
OUTCOMES = {  # the CFPB's company response -> this benchmark's choice id
    "Closed with explanation": "explanation only",
    "Closed with non-monetary relief": "fix without money",
    "Closed with monetary relief": "money back",
}

# Received-date windows, inclusive. June 2026 is a buffer around current Claude models'
# June 2026 training-data cutoff; the CFPB stopped publishing narratives on 2026-08-14.
TRAIN_WINDOW = (date(2025, 1, 1), date(2026, 5, 31))
TEST_WINDOW = (date(2026, 7, 1), date(2026, 8, 14))

TRAIN_ROWS = 3000
VALIDATION_ROWS = 300
TEST_ROWS = 600
SEED = 20261103

USER_AGENT = "clef-tune-benchmark (https://github.com/hersh-gupta/clef-tune)"


def machine_env(name: str) -> str | None:
    """An environment variable from this process, else the Windows Machine or User scope (where
    the user keeps HF_TOKEN and ANTHROPIC_API_KEY); never printed."""
    value = os.environ.get(name)
    if value or sys.platform != "win32":
        return value
    import winreg

    machine = r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
    places = [
        (winreg.HKEY_LOCAL_MACHINE, machine),
        (winreg.HKEY_CURRENT_USER, "Environment"),
    ]
    for hive, key in places:
        try:
            with winreg.OpenKey(hive, key) as handle:
                return str(winreg.QueryValueEx(handle, name)[0])
        except OSError:
            continue
    return None
