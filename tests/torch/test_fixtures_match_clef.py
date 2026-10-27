"""The vendored Clef files must be byte-for-byte what the pinned Hub revision serves."""

from __future__ import annotations

from pathlib import Path

import pytest
from huggingface_hub import hf_hub_download

from clef_tune.config import PINNED_REVISIONS

pytestmark = [pytest.mark.torch, pytest.mark.network]

FIXTURES = Path(__file__).parent / "fixtures"
VENDORED = [
    "joint_schema_model.py",
    "tokenizer_config.json",
    "processor_config.json",
    "chat_template.jinja",
    "LICENSE",
]


@pytest.mark.parametrize("name", VENDORED)
def test_vendored_file_matches_the_pinned_hub_revision(name):
    downloaded = hf_hub_download(
        "Cloudflare/clef", name, revision=PINNED_REVISIONS["Cloudflare/clef"]
    )
    assert (FIXTURES / name).read_bytes() == Path(downloaded).read_bytes()
