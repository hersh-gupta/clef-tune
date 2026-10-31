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


def serve(settings: Settings):
    import socket

    import uvicorn

    app = create_app(settings, fake_services()[0], secret="browser")
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
def base_url(monkeypatch_module):
    monkeypatch_module.setenv("HF_TOKEN", "hf_machine")
    yield from serve(Settings(False, 200, False))


@pytest.fixture(scope="module")
def space_url(monkeypatch_module):
    """A Space-mode app: a visitor who hasn't signed in sees the first-time welcome. Outside a
    Space, huggingface_hub mocks sign-in from the machine's login, so stub that to stay offline."""
    import huggingface_hub._oauth as oauth

    monkeypatch_module.setattr(oauth, "get_token", lambda: "hf_local_dev")
    monkeypatch_module.setattr(
        oauth,
        "whoami",
        lambda *a, **k: {
            "type": "user",
            "name": "dev",
            "fullname": "Dev",
            "email": None,
            "avatarUrl": "",
            "orgs": [],
        },
    )
    yield from serve(Settings(True, 200, False))


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
    # Picking another question swaps the editor in place and keeps the address in step.
    page.evaluate("window.__kept = true")
    page.locator("#qlist a").nth(1).click()
    page.wait_for_url("**/questions?q=urgency")
    page.wait_for_function("document.querySelector('#ask-box').value.startsWith('How quickly')")
    assert page.evaluate("window.__kept") is True
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


def test_on_a_phone_the_file_icon_sits_centered_above_the_choose_button(page, base_url):
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"{base_url}/data")
    icon = page.locator(".drop .file-glyph").bounding_box()
    button = page.locator(".drop .choose").bounding_box()
    assert icon["y"] + icon["height"] <= button["y"]
    assert abs((icon["x"] + icon["width"] / 2) - (button["x"] + button["width"] / 2)) < 2


def test_a_tip_opens_on_hover_and_does_not_stick_open_after_a_click(page, base_url):
    page.goto(f"{base_url}/data")
    if not page.locator(".filebar").count():
        page.get_by_role("button", name="support tickets").click()
        page.wait_for_url("**/data")
    page.goto(f"{base_url}/questions")
    shown = "getComputedStyle(document.querySelector('#ask-tip')).opacity"
    button = page.get_by_role("button", name="How this question is used")
    button.hover()
    assert page.evaluate(shown) == "1"
    button.click()
    page.mouse.move(5, 5)
    assert page.evaluate(shown) == "0"
    button.focus()  # keyboard focus still opens it
    page.keyboard.press("Shift+Tab")
    page.keyboard.press("Tab")
    assert page.evaluate(shown) == "1"
    page.keyboard.press("Escape")
    assert page.evaluate(shown) == "0"


def test_an_open_tip_sits_above_the_sections_after_it(space_url):
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 760})  # motion on
        page.goto(f"{space_url}/")
        page.wait_for_timeout(2000)  # the sections' entrance animations have finished
        page.locator("button.term").hover()
        tip = page.locator("#decision-model-tip").bounding_box()
        button = page.get_by_role("link", name="Start with your data").bounding_box()
        # A point where the open tip covers the "Start with your data" button.
        point = [max(tip["x"], button["x"]) + 20, button["y"] + button["height"] / 2]
        assert tip["y"] < point[1] < tip["y"] + tip["height"]
        on_top = page.evaluate(
            "([x, y]) => !!document.elementFromPoint(x, y).closest('#decision-model-tip')", point
        )
        browser.close()
    assert on_top


def with_sample(page, base_url):
    page.goto(f"{base_url}/data")
    if not page.locator(".filebar").count():
        page.get_by_role("button", name="support tickets").click()
        page.wait_for_url("**/data")


def test_a_framed_page_offers_to_open_in_its_own_tab(page, base_url):
    page.goto(f"{base_url}/data")
    assert page.locator("[data-own-tab]").is_hidden()
    page.set_content(f'<iframe src="{base_url}/data" width="1000" height="600"></iframe>')
    link = page.frame_locator("iframe").locator("[data-own-tab]")
    link.wait_for(state="visible")
    assert link.get_attribute("target") == "_blank"
    assert link.get_attribute("href") == f"{base_url}/data"


# The sandbox huggingface.co puts on a Space's frame (seen on a live Space, 2026-10-31): it
# may open tabs, but not take over its own tab.
SPACE_SANDBOX = (
    "allow-downloads allow-forms allow-modals allow-pointer-lock allow-popups "
    "allow-popups-to-escape-sandbox allow-same-origin allow-scripts "
    "allow-storage-access-by-user-activation"
)


def framed_like_a_space(page, url: str, *, cookies: bool = True):
    """Show `url` in a frame with the sandbox huggingface.co puts on a Space. cookies=False
    shows the frame's script no cookies, which is how the page tells that the browser keeps
    none there (Safari and every iPhone browser); the browser itself still sends them."""
    if not cookies:
        page.add_init_script(
            "if (window.top !== window.self) Object.defineProperty(Document.prototype,"
            " 'cookie', {get: () => '', set: () => {}})"
        )
    page.set_content(
        f'<iframe src="{url}" sandbox="{SPACE_SANDBOX}" width="1000" height="600"></iframe>'
    )
    return page.frame_locator("iframe")


def test_sign_in_from_hugging_face_s_frame_opens_its_own_tab(page, space_url):
    link = framed_like_a_space(page, f"{space_url}/data").get_by_role(
        "link", name="Sign in with Hugging Face"
    )
    with page.expect_popup() as popup:
        link.click()
    assert popup.value.url.startswith(space_url)


def test_where_the_frame_keeps_cookies_a_sample_loads_in_place(page, space_url):
    frame = framed_like_a_space(page, f"{space_url}/")
    frame.get_by_role("button", name="support tickets").click()
    frame.locator(".filebar").wait_for()
    assert len(page.context.pages) == 1


def test_where_the_frame_keeps_no_cookies_a_sample_opens_in_its_own_tab(page, space_url):
    """Nothing set up in such a frame would stick, so the click takes its action to the app's
    own tab instead."""
    frame = framed_like_a_space(page, f"{space_url}/", cookies=False)
    with page.expect_popup() as popup:
        frame.get_by_role("button", name="support tickets").click()
    tab = popup.value
    tab.wait_for_url("**/data")
    assert tab.locator(".filebar").count() == 1


def test_a_refused_post_says_so_on_the_page(page, base_url):
    page.goto(f"{base_url}/")
    page.evaluate(
        """() => {
            const form = document.createElement("form");
            form.setAttribute("data-post", "");
            form.setAttribute("action", "/no-such-place");
            document.body.append(form);
            form.requestSubmit();
        }"""
    )
    page.locator("#page-error").wait_for(state="visible")
    assert page.locator("#page-error").inner_text()


def test_menus_and_the_prompt_panel_keep_focus_and_give_it_back(page, base_url):
    with_sample(page, base_url)
    in_menu = "!!document.activeElement.closest('.menu:not([hidden])')"
    on_trigger = "document.activeElement.matches('[data-menu=\"#role-1\"]')"
    page.locator("[data-menu='#role-1']").click()
    assert page.evaluate(in_menu)
    for key in ["Tab"] * 7 + ["Shift+Tab"] * 3:
        page.keyboard.press(key)
        assert page.evaluate(in_menu), key
    page.keyboard.press("Escape")
    assert page.evaluate(on_trigger)
    page.locator("[data-menu='#role-1']").click()
    page.locator("#role-1 button[aria-checked=true]").click()  # a choice, swapped in place
    page.wait_for_function(on_trigger)
    page.goto(f"{base_url}/questions")
    page.get_by_role("button", name="See what the model reads").click()
    in_panel = "!!document.activeElement.closest('#prompt-panel .panel')"
    page.wait_for_function(in_panel)
    for _ in range(4):
        page.keyboard.press("Tab")
        assert page.evaluate(in_panel)
    page.keyboard.press("Escape")
    assert page.evaluate("document.activeElement.matches('[data-open-panel]')")
