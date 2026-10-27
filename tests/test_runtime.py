from pathlib import Path

import pytest

from clef_tune import runtime

ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_uses_the_pinned_base_image():
    first = (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()[0]
    assert first == f"FROM {runtime.BASE_IMAGE}"


def test_kernels_available_checks_every_module(monkeypatch):
    found = {"fla": object(), "causal_conv1d": None}
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda name: found.get(name))
    assert runtime.kernels_available() is False
    found["causal_conv1d"] = object()
    assert runtime.kernels_available() is True


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ("", None),
        ("abc", None),
        ("-1", None),
        ("nan", None),
        ("inf", None),
        ("0", None),
        ("2.5", 2.5),
    ],
)
def test_env_float(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(runtime.PRICE_ENV, raising=False)
    else:
        monkeypatch.setenv(runtime.PRICE_ENV, value)
    assert runtime.env_float(runtime.PRICE_ENV) == expected
