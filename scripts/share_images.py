"""Make the images for sharing clef-tune: the 1200x630 link-preview cover (served by the app as
its og:image, and the Space's social thumbnail) and the README's screenshots. As in
screenshots.py, the app runs against fake services. With --run, the results screenshot shows
that real run's results, read from its Hub repo with this machine's Hugging Face login:

    uv run python scripts/share_images.py [--run you/clef-tune-run]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # for tests.ui.fakes

import uvicorn  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

from clef_tune.ui.server import create_app  # noqa: E402
from clef_tune.ui.settings import Settings  # noqa: E402
from scripts.screenshots import free_port, serve  # noqa: E402
from tests.ui import fakes  # noqa: E402

COVER = ROOT / "clef_tune" / "ui" / "static" / "cover.png"
IMAGES = ROOT / "docs" / "images"
SPACE_URL = "huggingface.co/spaces/hvgupta1/clef-tune"

COVER_HTML = """<!doctype html>
<html lang="en" data-theme="light"><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;500&display=swap"
  rel="stylesheet">
<link rel="stylesheet" href="{base}/static/clef-tune.css">
<style>
  html, body {{ width: 1200px; height: 630px; overflow: hidden; }}
  .cover {{ width: 1200px; height: 630px; padding: 60px 76px 52px; display: grid;
    grid-template-columns: 1fr 300px; grid-template-rows: auto 1fr auto; column-gap: 48px; }}
  .cover .brand {{ grid-column: 1 / -1; gap: 14px; }}
  .cover .mark {{ width: 18px; height: 18px; border-width: 2px; }}
  .cover .brand b {{ font-size: 17px; }}
  .cover .copy {{ align-self: center; }}
  .cover .kicker {{ font-size: 16px; margin-bottom: 22px; }}
  .cover h1 {{ font-size: 64px; font-weight: 300; line-height: 1.12; margin: 0 0 26px; }}
  .cover h1 em {{ font-style: normal; color: var(--sage-ink); }}
  .cover .rule {{ width: 72px; margin-bottom: 26px; }}
  .cover .lede {{ font-size: 21px; line-height: 1.5; max-width: none; margin: 0; }}
  .cover .ill {{ align-self: center; height: 240px; }}
  .cover .ill text {{ font-size: 9px; }}
  .cover .foot {{ grid-column: 1 / -1; margin: 0; padding-top: 18px; font-size: 15px;
    color: var(--muted); border-top: 1px solid var(--line-soft); letter-spacing: .02em; }}
</style></head>
<body><div class="cover">
  <p class="brand"><span class="mark"></span><b>clef-tune</b></p>
  <div class="copy">
    <p class="kicker">Fine-tuned decisions for everyone</p>
    <h1>Bespoke intelligence,<br><em>owned by you</em>.</h1>
    <div class="rule"></div>
    <p class="lede">Fine-tune an open decision model on your own spreadsheet,<br>and see how
      accurate it is before you use it.</p>
  </div>
  <div class="ill">{illustration}</div>
  <p class="foot">{space} &nbsp;·&nbsp; Open source</p>
</div></body></html>"""


def real_run(repo: str):
    """Fake services whose finished job (job-b) is the real run in `repo`."""
    from huggingface_hub import get_token

    from clef_tune.hub import pull_run_files

    files = pull_run_files(repo, token=get_token())

    def status(job_id, **kwargs):
        found = fakes.job_status(job_id)
        if job_id != "job-b":
            return found
        return dataclasses.replace(found, run=repo.split("/")[-1], adapter_repo=repo)

    return fakes.fake_services(job_status=status, pull_run_files=lambda repo_id, **k: files)[0]


def serve_services(services) -> tuple[str, uvicorn.Server]:
    import threading
    import time

    app = create_app(Settings(False, 200, False), services, secret="share")
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    return f"http://127.0.0.1:{port}", server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", help="A real run's Hub repo, for the results screenshot.")
    args = parser.parse_args()
    IMAGES.mkdir(parents=True, exist_ok=True)
    newcomer, first = serve(space=True)
    results, second = serve_services(real_run(args.run) if args.run else fakes.fake_services()[0])
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, reduced_motion="reduce")
        page.goto(f"{newcomer}/")
        page.wait_for_load_state("networkidle")
        illustration = page.inner_html(".ill")
        page.screenshot(path=str(IMAGES / "welcome.png"), full_page=True)

        page.goto(f"{results}/runs/job-b")
        page.wait_for_load_state("networkidle")
        top = page.locator(".hero-line").bounding_box()["y"] - 24
        end = page.locator(".qres").nth(1).bounding_box()
        page.screenshot(
            path=str(IMAGES / "results.png"),
            clip={"x": 220, "y": top, "width": 1000, "height": end["y"] + end["height"] - top},
        )

        cover = browser.new_page(viewport={"width": 1200, "height": 630}, reduced_motion="reduce")
        cover.set_content(
            COVER_HTML.format(base=newcomer, illustration=illustration, space=SPACE_URL)
        )
        cover.wait_for_load_state("networkidle")
        cover.evaluate("document.fonts.ready")
        cover.screenshot(path=str(COVER))
        browser.close()
    first.should_exit = second.should_exit = True
    for path in (COVER, IMAGES / "welcome.png", IMAGES / "results.png"):
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
