import pytest

gr = pytest.importorskip("gradio")

from typer.testing import CliRunner  # noqa: E402

from clef_tune import cli  # noqa: E402
from clef_tune.ui import app  # noqa: E402
from clef_tune.ui.settings import Settings  # noqa: E402
from tests.ui.fakes import fake_services  # noqa: E402


def ids(demo):
    return {getattr(block, "elem_id", None) for block in demo.blocks.values()}


@pytest.fixture
def local_hf_login(monkeypatch):
    """Off a Space, Gradio mocks the sign-in button with the machine's own HF login when the
    app is built; stand in for that login so Space mode can be built here."""
    import gradio.oauth

    monkeypatch.setattr(gradio.oauth, "get_token", lambda: "hf_test")
    user = {"type": "user", "name": "me", "fullname": "Me", "avatarUrl": ""}
    monkeypatch.setattr(gradio.oauth, "whoami", lambda *a, **k: user)


def test_the_app_builds_without_calling_any_service():
    services, calls = fake_services()
    demo = app.build_app(Settings(space=False, max_upload_mb=200, local_training=True), services)
    assert {"ct-sidebar", "ct-launch-jobs", "ct-launch-local", "ct-results"} <= ids(demo)
    assert not demo.expects_oauth and calls == []


def test_space_mode_hides_local_training_and_signs_in_with_hugging_face(local_hf_login):
    services, _ = fake_services()
    demo = app.build_app(Settings(space=True, max_upload_mb=200, local_training=True), services)
    assert "ct-launch-local" not in ids(demo) and demo.expects_oauth


def test_launch_passes_theme_css_and_the_upload_cap(monkeypatch, local_hf_login):
    seen = {}
    monkeypatch.setattr(gr.Blocks, "launch", lambda self, **kwargs: seen.update(kwargs))
    monkeypatch.setattr(app, "sweep_stale", lambda: 0)
    settings = Settings(space=True, max_upload_mb=50, local_training=False)
    app.launch(settings=settings, services=fake_services()[0])
    assert seen["max_file_size"] == "50mb" and app.STYLE in seen["css_paths"]
    assert isinstance(seen["theme"], gr.themes.Base) and app.STYLE.exists()


def test_the_ui_command_launches(monkeypatch):
    seen = {}
    monkeypatch.setattr(app, "launch", lambda **kwargs: seen.update(kwargs))
    result = CliRunner().invoke(cli.app, ["ui", "--port", "7861"])
    assert result.exit_code == 0 and seen["port"] == 7861 and seen["share"] is False
