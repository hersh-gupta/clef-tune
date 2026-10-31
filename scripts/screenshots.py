"""Capture the web UI for review: every screen, in light and dark, on desktop and phone widths.
The app runs against fake services that replay tests/ui/fixtures/recorded_run/, so nothing
touches the Hub. Motion is turned off (prefers-reduced-motion) so captures are still. Writes
PNGs and an index.html contact sheet:

    uv run python scripts/screenshots.py [--out screenshots] [--widths 1440 1024 390]
        [--themes light dark]
"""

from __future__ import annotations

import argparse
import html
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # for tests.ui.fakes

import uvicorn  # noqa: E402
from playwright.sync_api import Page, sync_playwright  # noqa: E402

from clef_tune.ui.server import create_app  # noqa: E402
from clef_tune.ui.settings import Settings  # noqa: E402
from tests.helpers import write_csv  # noqa: E402
from tests.ui.fakes import fake_services  # noqa: E402

HEIGHT = 900
SETTLE_MS = 400


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def serve(space: bool) -> tuple[str, uvicorn.Server]:
    """Run the app in a thread. A Space-mode app shows a newcomer's welcome."""
    if space:
        import huggingface_hub._oauth as oauth

        oauth.get_token = lambda: "hf_local_dev"
        oauth.whoami = lambda *a, **k: {
            "type": "user",
            "name": "dev",
            "fullname": "Dev",
            "email": None,
            "avatarUrl": "",
            "orgs": [],
        }
    settings = Settings(space=space, max_upload_mb=200, local_training=False)
    app = create_app(settings, fake_services()[0], secret="screenshots")
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    return f"http://127.0.0.1:{port}", server


def shoot(page: Page, out: Path, name: str, shots: list[tuple[str, Path]]) -> None:
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(SETTLE_MS)
    path = out / f"{name}.png"
    page.screenshot(path=str(path), full_page=True)
    shots.append((name, path))


def capture(page: Page, base: str, newcomer: str, out: Path, prefix: str, data: Path):
    shots: list[tuple[str, Path]] = []
    page.goto(f"{newcomer}/")
    shoot(page, out, f"{prefix}-01-welcome", shots)
    page.goto(f"{base}/")
    shoot(page, out, f"{prefix}-02-welcome-back", shots)
    page.goto(f"{base}/data")
    shoot(page, out, f"{prefix}-03-data-empty", shots)
    page.get_by_role("button", name="support tickets").click()
    page.wait_for_url("**/data")
    shoot(page, out, f"{prefix}-04-data", shots)
    page.locator(".cols .tag").nth(3).click()
    shoot(page, out, f"{prefix}-05-data-role-menu", shots)
    page.keyboard.press("Escape")
    page.goto(f"{base}/questions")
    shoot(page, out, f"{prefix}-06-questions", shots)
    page.goto(f"{base}/questions?q=urgency")
    page.get_by_role("button", name="How this question is used").hover()
    shoot(page, out, f"{prefix}-07-questions-scale-tip", shots)
    page.goto(f"{base}/settings")
    shoot(page, out, f"{prefix}-08-settings", shots)
    page.locator("[data-menu='#menu-preset']").click()
    shoot(page, out, f"{prefix}-09-settings-training", shots)
    page.keyboard.press("Escape")
    page.locator("details.adv summary").click()
    shoot(page, out, f"{prefix}-10-settings-advanced", shots)
    page.goto(f"{base}/train")
    page.wait_for_selector("#review .srows")
    shoot(page, out, f"{prefix}-11-train-ready", shots)
    page.goto(f"{base}/runs/job-a")
    shoot(page, out, f"{prefix}-12-training", shots)
    page.goto(f"{base}/runs/job-b")
    shoot(page, out, f"{prefix}-13-results", shots)
    page.locator(".qres details.more summary").first.click()
    shoot(page, out, f"{prefix}-14-results-details", shots)
    page.goto(f"{base}/runs/job-c")
    shoot(page, out, f"{prefix}-15-failed", shots)
    page.goto(f"{base}/models")
    shoot(page, out, f"{prefix}-16-your-models", shots)
    page.goto(f"{base}/predict/job-b")
    shoot(page, out, f"{prefix}-17-predict-choose", shots)
    page.locator(".drop input[type=file]").set_input_files(str(data))
    page.wait_for_selector("#predict-estimate .est")
    shoot(page, out, f"{prefix}-18-predict-ready", shots)
    page.goto(f"{base}/runs/job-p")
    shoot(page, out, f"{prefix}-19-predictions", shots)
    return shots


def contact_sheet(out: Path, shots: list[tuple[str, Path]]) -> Path:
    cards = "\n".join(
        f"<figure><figcaption>{html.escape(name)}</figcaption>"
        f'<a href="{path.name}"><img src="{path.name}" loading="lazy" alt=""></a></figure>'
        for name, path in shots
    )
    page = out / "index.html"
    page.write_text(
        "<!doctype html><meta charset=utf-8><title>clef-tune screenshots</title><style>"
        "body{font:14px system-ui;margin:24px;background:#eee}figure{display:inline-block;"
        "vertical-align:top;margin:0 16px 24px 0}img{width:360px;border:1px solid #ccc}"
        f"</style>{cards}",
        encoding="utf-8",
    )
    return page


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "screenshots")
    parser.add_argument("--widths", type=int, nargs="+", default=[1440, 1024, 390])
    parser.add_argument("--themes", nargs="+", default=["light", "dark"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    base, local_server = serve(space=False)
    newcomer, space_server = serve(space=True)
    rows = [{"subject": "Refund", "body": "Charged twice.", "customer_tier": "pro"}] * 5
    data = write_csv(Path(tempfile.mkdtemp()) / "new_tickets.csv", rows)
    shots: list[tuple[str, Path]] = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            for width in args.widths:
                for theme in args.themes:
                    context = browser.new_context(
                        viewport={"width": width, "height": HEIGHT},
                        reduced_motion="reduce",
                        is_mobile=width < 700,
                        has_touch=width < 700,
                    )
                    context.add_init_script(f"localStorage.setItem('clef-tune-theme', '{theme}')")
                    page = context.new_page()
                    shots += capture(page, base, newcomer, args.out, f"{width}-{theme}", data)
                    context.close()
            browser.close()
    finally:
        local_server.should_exit = space_server.should_exit = True
    print(f"Wrote {len(shots)} screenshots; open {contact_sheet(args.out, shots)}")


if __name__ == "__main__":
    main()
