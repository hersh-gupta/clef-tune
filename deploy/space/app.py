"""Hugging Face Space entry point for the clef-tune web UI (served by uvicorn app:app)."""

from clef_tune.ui.server import create_app

app = create_app()
