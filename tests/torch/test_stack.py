import sys

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch


def test_training_stack_versions():
    import peft
    import torchvision
    import transformers
    from transformers import Qwen3_5ForConditionalGeneration  # noqa: F401

    assert torch.__version__.startswith("2.11.")
    assert torchvision.__version__.startswith("0.26.")
    assert transformers.__version__.startswith("5.10.")
    assert int(peft.__version__.split(".")[1]) >= 21


@pytest.mark.skipif(sys.platform != "win32", reason="the cu130 source applies to Windows only")
def test_windows_gets_the_cuda_build():
    assert torch.__version__.endswith("+cu130")
