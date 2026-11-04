"""Step 1: get the CFPB data.

  list       show which narrative-archive files cover the benchmark's date windows, and their sizes
  download   download those files (only with --yes, after the list has been shown and approved)
  labels     fetch each complaint's product, issue and company response from the CFPB search API

Archive files land in data/raw/<file name>/, one new folder per file; nothing in them is run.
"""

from __future__ import annotations

import argparse
import calendar
import json
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path

if __package__ in (None, ""):  # run as a script: make `benchmarks` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.cfpb.common import (
    BANK,
    PRODUCTS,
    RAW,
    TEST_WINDOW,
    TRAIN_WINDOW,
    USER_AGENT,
)

ARCHIVE_PAGE = (
    "https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/"
    "cfpb-consumer-complaint-database-narratives-archive/"
)
SEARCH_API = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
API_DIR = RAW / "api"
MONTHS = {name: number for number, name in enumerate(calendar.month_name) if name}
NAME = re.compile(
    r"CCDB_Export_(?P<n>\d+)_(?P<m1>[A-Za-z]+)_(?P<y1>\d{4})"
    r"(?:_through_(?P<m2>[A-Za-z]+)_(?P<y2>\d{4}))?\.zip$"
)


@dataclass(frozen=True)
class Archive:
    url: str
    name: str
    first: date  # first day of the first month it covers
    last: date  # last day of the last month it covers
    size: int | None = None


def _get(url: str, *, method: str = "GET", timeout: int = 120):
    request = urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310 (fixed https hosts)


def parse_archive(url: str) -> Archive | None:
    name = url.rsplit("/", 1)[-1]
    match = NAME.search(name)
    if not match:
        return None
    m1, y1 = MONTHS[match["m1"]], int(match["y1"])
    m2, y2 = (MONTHS[match["m2"]], int(match["y2"])) if match["m2"] else (m1, y1)
    last_day = calendar.monthrange(y2, m2)[1]
    return Archive(url, name, date(y1, m1, 1), date(y2, m2, last_day))


def overlaps(archive: Archive, window: tuple[date, date]) -> bool:
    return archive.first <= window[1] and archive.last >= window[0]


def needed_archives(urls: list[str]) -> list[Archive]:
    """The files covering the train or test window. If a file's period is when complaints were
    published rather than received, a complaint received at a window's end can sit in the next
    file, so the file after each window is kept too; the date filter in prepare.py decides."""
    archives = sorted((a for a in map(parse_archive, urls) if a is not None), key=lambda a: a.first)
    keep: set[str] = set()
    for window in (TRAIN_WINDOW, TEST_WINDOW):
        hits = [i for i, a in enumerate(archives) if overlaps(a, window)]
        if hits:
            keep.update(archives[i].name for i in hits)
            if hits[-1] + 1 < len(archives):
                keep.add(archives[hits[-1] + 1].name)
    return [a for a in archives if a.name in keep]


def archive_urls() -> list[str]:
    with _get(ARCHIVE_PAGE) as response:
        page = response.read().decode("utf-8", "replace")
    return sorted(set(re.findall(r'href="(https://files\.consumerfinance\.gov/[^"]+\.zip)"', page)))


def with_size(archive: Archive) -> Archive:
    with _get(archive.url, method="HEAD", timeout=60) as response:
        size = int(response.headers.get("Content-Length", 0)) or None
    return Archive(archive.url, archive.name, archive.first, archive.last, size)


def list_command() -> list[Archive]:
    archives = [with_size(a) for a in needed_archives(archive_urls())]
    total = 0
    print(f"Source: {ARCHIVE_PAGE}")
    for a in archives:
        mb = (a.size or 0) / 1e6
        total += a.size or 0
        state = "downloaded" if _target(a).exists() else ""
        print(f"  {a.name:<70} {mb:7.1f} MB  {state}")
    print(f"  {'total':<70} {total / 1e6:7.1f} MB  ({len(archives)} files)")
    return archives


def _target(archive: Archive) -> Path:
    return RAW / archive.name.removesuffix(".zip") / archive.name


def download_command(yes: bool) -> None:
    archives = list_command()
    if not yes:
        print("\nNothing downloaded. Run again with --yes once these files are approved.")
        return
    for a in archives:
        target = _target(a)
        if target.exists() and (a.size is None or target.stat().st_size == a.size):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".part")
        print(f"Downloading {a.name} ...", flush=True)
        with _get(a.url, timeout=600) as response, partial.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        if a.size is not None and partial.stat().st_size != a.size:
            raise SystemExit(f"{a.name}: got {partial.stat().st_size} bytes, expected {a.size}")
        partial.replace(target)
    print("All archive files are in", RAW)


def _months(window: tuple[date, date]) -> list[tuple[date, date]]:
    out, year, month = [], window[0].year, window[0].month
    while date(year, month, 1) <= window[1]:
        first = max(date(year, month, 1), window[0])
        last = min(date(year, month, calendar.monthrange(year, month)[1]), window[1])
        out.append((first, last))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def labels_for(first: date, last: date) -> list[dict]:
    query = [
        ("company", BANK),
        *[("product", p) for p in PRODUCTS],
        ("date_received_min", first.isoformat()),
        ("date_received_max", last.isoformat()),
        ("size", "10000"),
        ("no_aggs", "true"),
        ("sort", "created_date_asc"),
    ]
    with _get(f"{SEARCH_API}?{urllib.parse.urlencode(query)}") as response:
        found = json.load(response)
    hits = found["hits"]["hits"]
    total = found["hits"]["total"]["value"]
    if total > len(hits):  # one month never comes near 10,000 for one bank, but check
        raise SystemExit(f"{first:%Y-%m}: {total} complaints but only {len(hits)} returned")
    return [hit["_source"] for hit in hits]


def labels_command() -> None:
    API_DIR.mkdir(parents=True, exist_ok=True)
    for window in (TRAIN_WINDOW, TEST_WINDOW):
        for first, last in _months(window):
            path = API_DIR / f"labels-{first:%Y-%m}.json"
            if path.exists():
                continue
            rows = labels_for(first, last)
            path.write_text(json.dumps(rows), encoding="utf-8")
            print(f"{first:%Y-%m}: {len(rows)} complaints")
    print("Labels are in", API_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="show the archive files needed and their sizes")
    dl = sub.add_parser("download", help="download the archive files")
    dl.add_argument("--yes", action="store_true", help="the file list has been approved")
    sub.add_parser("labels", help="fetch labels from the CFPB search API")
    args = parser.parse_args()
    if args.command == "list":
        list_command()
    elif args.command == "download":
        download_command(args.yes)
    else:
        labels_command()


if __name__ == "__main__":
    main()
