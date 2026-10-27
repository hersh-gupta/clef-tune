from __future__ import annotations

import pytest

from tests.torch.tiny_clef import build_tiny_release, write_tiny_project


@pytest.fixture(scope="session")
def tiny_release(tmp_path_factory):
    return build_tiny_release(tmp_path_factory.mktemp("tiny-release"))


@pytest.fixture(scope="session")
def adapter(tmp_path_factory, tiny_release):
    """A trained tiny adapter whose outage temperature is 2.0 (shared by predictor tests)."""
    import json

    import torch

    from clef_tune.train import train

    project = tmp_path_factory.mktemp("adapter")
    config = write_tiny_project(project, tiny_release, {"train": {"epochs": 2}})
    # Session fixtures are built before the function-scoped CLEF_TUNE_DEVICE fixture runs.
    result = train(config, device=torch.device("cpu"))
    (result.adapter_dir / "calibration.json").write_text(
        json.dumps({"outage": 2.0}), encoding="utf-8"
    )
    return result.adapter_dir


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
