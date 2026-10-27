"""Hugging Face Hub access for adapters: create a private repo, push an adapter, pull one back.
Torch-free, so the CLI can use it without loading the training stack."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


class HubError(RuntimeError):
    """A Hub operation failed; the message says what to check."""


def ensure_repo(
    repo_id: str, *, token: str | None = None, private: bool = True, api: HfApi | None = None
) -> str:
    """Create the model repo if it is missing (private by default); return its URL."""
    api = api or HfApi()
    try:
        return str(
            api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True, token=token)
        )
    except Exception as exc:
        raise HubError(
            f"cannot create or write to {repo_id}: {exc} (check that your token has write access)"
        ) from exc


def push_adapter(
    adapter_dir: str | Path,
    repo_id: str,
    *,
    token: str | None = None,
    api: HfApi | None = None,
) -> str:
    """Upload the adapter folder to the repo (created private if missing); return the commit URL."""
    api = api or HfApi()
    ensure_repo(repo_id, token=token, api=api)
    try:
        info = api.upload_folder(
            folder_path=str(adapter_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload clef-tune adapter",
            token=token,
        )
    except Exception as exc:
        raise HubError(f"cannot upload the adapter to {repo_id}: {exc}") from exc
    return str(info)


def pull_adapter(repo_id: str, *, revision: str | None = None, token: str | None = None) -> Path:
    """Download an adapter repo (cached) and return its local folder."""
    return Path(snapshot_download(repo_id, revision=revision, token=token))
