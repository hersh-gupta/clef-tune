"""Opt-in end-to-end test in a real browser (marker `browser`): the example goes from upload
to a reviewed launch and a completed run's results, against fake services. Needs Playwright's
chromium (`uv run playwright install chromium`)."""

import socket

import pytest

gr = pytest.importorskip("gradio")
sync_api = pytest.importorskip("playwright.sync_api")

from clef_tune.ui import app  # noqa: E402
from clef_tune.ui.settings import Settings  # noqa: E402
from tests.ui.fakes import fake_services  # noqa: E402

pytestmark = pytest.mark.browser


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    settings = Settings(space=False, max_upload_mb=200, local_training=False)
    url = app.launch(
        port=free_port(), settings=settings, services=fake_services()[0], prevent_thread_lock=True
    )
    yield url
    gr.close_all()


@pytest.fixture
def page(server):
    with sync_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.console_errors = []
        page.on(
            "console",
            lambda message: message.type == "error" and page.console_errors.append(message.text),
        )
        page.goto(server)
        yield page
        browser.close()


def test_the_example_goes_from_data_to_results(page):
    expect = sync_api.expect
    page.get_by_role("button", name="Load the example").click()
    expect(page.locator("#ct-data")).to_contain_text("200")
    page.locator("#ct-step-1").click()
    expect(page.locator("#ct-schema")).to_contain_text("Which team should handle this ticket?")
    page.get_by_role("button", name="Next: Validate →").click()
    expect(page.locator("#ct-validate")).to_contain_text("Train rows")
    page.get_by_role("button", name="Next: Configure →").click()
    expect(page.locator("#ct-configure")).to_contain_text("l4x1")
    page.get_by_role("button", name="Next: Launch →").click()
    page.get_by_label("Hugging Face token").fill("hf_test")
    page.get_by_role("button", name="Review launch").click()
    expect(page.locator("#ct-confirm")).to_contain_text("Launch for up to $")
    page.locator("#ct-runs").get_by_text("run-b").click()
    expect(page.locator("#ct-results svg").first).to_be_visible()
    assert page.console_errors == []
