"""Drive the web UI in a real browser (opt-in: pytest -m browser; needs Playwright's chromium).
The app runs against the fake services, so nothing touches the Hub."""

import threading
import time

import pytest

from clef_tune.ui.server import create_app
from clef_tune.ui.settings import Settings
from tests.ui.fakes import fake_services

pytestmark = pytest.mark.browser
playwright = pytest.importorskip("playwright.sync_api")


@pytest.fixture(scope="module")
def base_url(monkeypatch_module):
    import socket

    import uvicorn

    monkeypatch_module.setenv("HF_TOKEN", "hf_machine")
    app = create_app(Settings(False, 200, False), fake_services()[0], secret="browser")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True


@pytest.fixture(scope="module")
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as patch:
        yield patch


@pytest.fixture
def page():
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        yield browser.new_page(reduced_motion="reduce")
        browser.close()


def no_sideways_scroll(page) -> bool:
    return page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")


def test_a_sample_goes_from_data_to_results(page, base_url):
    page.goto(f"{base_url}/data")
    page.get_by_role("button", name="support tickets").click()
    page.wait_for_url("**/data")
    title = page.locator("h1").inner_text()
    # A role change swaps the column list in place: the page itself is not reloaded.
    page.evaluate("window.__kept = true")
    page.locator(".cols .tag").nth(3).click()
    page.locator(".menu:not([hidden]) button[value=skip]").click()
    page.wait_for_function(
        "document.querySelector('.foot .summary-text').textContent.includes('2 columns read')"
    )
    assert page.evaluate("window.__kept") is True and page.locator("h1").inner_text() == title
    page.get_by_role("link", name="Continue").click()
    page.wait_for_url("**/questions")
    page.locator("#ask-box").fill("Which team owns this?")
    page.wait_for_function(
        "document.querySelector('#ask-1').textContent === 'Which team owns this?'"
    )
    page.get_by_role("link", name="Continue").click()
    page.wait_for_url("**/settings")
    assert "never cost more than" in page.locator("#settings-body").inner_text()
    page.get_by_role("link", name="Continue").click()
    page.wait_for_selector("#review .srows")
    page.get_by_role("button", name="Start training").click()
    page.wait_for_url("**/runs/job-new")
    assert "Learning from your examples" in page.content() or "Starting" in page.content()
    page.goto(f"{base_url}/runs/job-b")
    assert "24 of 30" in page.locator(".big-sentence").inner_text()


@pytest.mark.parametrize(
    "path", ["/", "/data", "/questions", "/settings", "/runs/job-b", "/runs/job-p", "/models"]
)
def test_every_page_fits_a_phone(page, base_url, path):
    page.set_viewport_size({"width": 360, "height": 740})
    page.goto(f"{base_url}/data")
    if not page.locator(".filebar").count():
        page.get_by_role("button", name="support tickets").click()
        page.wait_for_url("**/data")
    page.goto(f"{base_url}{path}")
    page.wait_for_load_state("networkidle")
    assert no_sideways_scroll(page)
