from __future__ import annotations

import pytest

from tests.torch.tiny_clef import build_tiny_release, write_tiny_project


@pytest.fixture(scope="session")
def tiny_release(tmp_path_factory):
    return build_tiny_release(tmp_path_factory.mktemp("tiny-release"))


@pytest.fixture
def tiny_project(tmp_path, tiny_release):
    def make(name="run", **overrides):
        return write_tiny_project(tmp_path / name, tiny_release, overrides)

    return make


@pytest.fixture(autouse=True)
def _cpu_device(request, monkeypatch):
    # The tiny-model recipes were verified on CPU; a CUDA torch build would otherwise pick the GPU.
    if request.node.get_closest_marker("gpu") is None:
        monkeypatch.setenv("CLEF_TUNE_DEVICE", "cpu")
