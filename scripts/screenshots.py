"""Capture the web UI for review (spec §9.5): every step, the launch review, the run monitor and
the results, in light and dark, at 1440 and 1024 px. The UI runs against fake services that
replay tests/ui/fixtures/recorded_run/, so nothing touches the Hub. Writes PNGs and an
index.html contact sheet:

    uv run python scripts/screenshots.py [--out screenshots] [--widths 1440 1024]
        [--themes light dark]
"""

from __future__ import annotations

import argparse
import html
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # for tests.ui.fakes

import gradio as gr  # noqa: E402
from playwright.sync_api import Page, expect, sync_playwright  # noqa: E402

from clef_tune.prompt import counter_for_config  # noqa: E402
from clef_tune.ui import app  # noqa: E402
from clef_tune.ui.settings import Settings  # noqa: E402
from tests.helpers import CharCounter  # noqa: E402
from tests.ui.fakes import fake_services  # noqa: E402

HEIGHT = 900
SETTLE_MS = 700


def real_or_fake_counter(config):
    """The real Clef tokenizer when it can be downloaded, so token counts look right."""
    try:
        return counter_for_config(config)
    except Exception:  # offline: fall back to one token per character
        return CharCounter()


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def shoot(page: Page, out: Path, name: str, shots: list[tuple[str, Path]]) -> None:
    # Gradio dims outputs that are still updating with the class "pending".
    page.wait_for_function("() => !document.querySelector('.pending')", timeout=30_000)
    page.wait_for_timeout(SETTLE_MS)
    path = out / f"{name}.png"
    page.screenshot(path=str(path), full_page=True)
    shots.append((name, path))


def capture(page: Page, url: str, out: Path, prefix: str) -> list[tuple[str, Path]]:
    shots: list[tuple[str, Path]] = []
    page.goto(url)
    expect(page.get_by_role("button", name="Try the example")).to_be_visible()
    shoot(page, out, f"{prefix}-01-data-empty", shots)

    page.get_by_role("button", name="Try the example").click()
    expect(page.locator("#ct-columns")).to_contain_text("customer_tier")
    shoot(page, out, f"{prefix}-02-data", shots)

    page.get_by_text("Rows", exact=True).click()
    page.get_by_text("Prompt preview", exact=True).click()
    expect(page.locator("#ct-data-table")).to_be_visible()
    shoot(page, out, f"{prefix}-03-data-open", shots)

    page.get_by_role("button", name="Next: Validate →").click()
    expect(page.locator("#ct-validate")).to_contain_text("Train rows")
    shoot(page, out, f"{prefix}-04-validate", shots)

    page.get_by_role("button", name="Next: Configure →").click()
    expect(page.locator("#ct-configure")).to_contain_text("Recommended")
    shoot(page, out, f"{prefix}-05-configure", shots)

    page.get_by_text("Advanced settings", exact=True).click()
    expect(page.get_by_label("LoRA rank")).to_be_visible()
    shoot(page, out, f"{prefix}-06-configure-advanced", shots)

    page.get_by_role("button", name="Next: Launch →").click()
    page.get_by_role("tab", name="Download bundle").click()
    page.get_by_role("button", name="Prepare the bundle").click()
    expect(page.get_by_role("button", name="Download the bundle")).to_be_visible()
    shoot(page, out, f"{prefix}-07-launch-bundle", shots)

    page.get_by_role("tab", name="Hugging Face Jobs").click()
    page.get_by_label("Hugging Face token").fill("hf_example")
    page.get_by_role("button", name="Review launch").click()
    expect(page.locator("#ct-confirm")).to_contain_text("Launch for up to $")
    shoot(page, out, f"{prefix}-08-launch-review", shots)

    page.locator("#ct-runs").get_by_test_id("cell-0-1").click()  # run-a
    expect(page.locator("#ct-monitor")).to_contain_text("Step 70 of 120")
    shoot(page, out, f"{prefix}-09-monitor", shots)

    page.locator("#ct-runs").get_by_test_id("cell-1-1").click()  # run-b
    expect(page.locator("#ct-results svg").first).to_be_visible()
    shoot(page, out, f"{prefix}-10-results", shots)
    return shots


def contact_sheet(out: Path, shots: list[tuple[str, Path]]) -> Path:
    by_view: dict[str, list[tuple[str, Path]]] = {}
    for name, path in shots:
        view = name.split("-", 2)[2]
        by_view.setdefault(view, []).append((name, path))
    sections = []
    for view, items in by_view.items():
        figures = "".join(
            f'<figure><a href="{html.escape(path.name)}"><img src="{html.escape(path.name)}" '
            f'loading="lazy"></a><figcaption>{html.escape(name)}</figcaption></figure>'
            for name, path in items
        )
        sections.append(f"<section><h2>{html.escape(view)}</h2><div>{figures}</div></section>")
    page = (
        "<!doctype html><meta charset='utf-8'><title>clef-tune screenshots</title><style>"
        "body{font-family:system-ui;margin:24px;background:#f4f4f5}"
        "section div{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}"
        "img{width:100%;border:1px solid #d4d4d8;background:#fff}"
        "figcaption{font-size:12px;color:#52525b}</style>"
        "<h1>clef-tune UI screenshots</h1>" + "".join(sections)
    )
    index = out / "index.html"
    index.write_text(page, encoding="utf-8")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture the web UI for review.")
    parser.add_argument("--out", type=Path, default=ROOT / "screenshots")
    parser.add_argument("--widths", type=int, nargs="+", default=[1440, 1024])
    parser.add_argument("--themes", nargs="+", default=["light", "dark"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    services = fake_services(counter=real_or_fake_counter)[0]
    settings = Settings(space=False, max_upload_mb=200, local_training=False)
    url = app.launch(
        port=free_port(), settings=settings, services=services, prevent_thread_lock=True
    )
    shots: list[tuple[str, Path]] = []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            for width in args.widths:
                for theme in args.themes:
                    page = browser.new_page(viewport={"width": width, "height": HEIGHT})
                    shots += capture(page, f"{url}?__theme={theme}", args.out, f"{width}-{theme}")
                    page.close()
            browser.close()
    finally:
        gr.close_all()
    print(f"{len(shots)} screenshots; contact sheet: {contact_sheet(args.out, shots)}")


if __name__ == "__main__":
    main()
