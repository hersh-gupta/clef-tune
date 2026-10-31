"""Hugging Face Hub access for adapters: create a private repo, push an adapter, pull one back.
Torch-free, so the CLI can use it without loading the training stack."""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

WRITE_ROLES = ("write", "admin")
REPO_WRITE = "repo.write"  # the fine-grained permission to create and push to repos


class HubError(RuntimeError):
    """A Hub operation failed; the message says what to check."""


def _fine_grained_can_write(grants: Any, repo_id: str, namespace: str) -> bool | None:
    """Whether fine-grained token grants include repo.write for the repo or its namespace;
    None when the grants are not in the shape whoami-v2 documents."""
    if not isinstance(grants, dict):
        return None
    scoped, global_permissions = grants.get("scoped", []), grants.get("global", [])
    if not isinstance(scoped, list) or not isinstance(global_permissions, list):
        return None
    if REPO_WRITE in global_permissions:
        return True
    for grant in scoped:
        entity = grant.get("entity") if isinstance(grant, dict) else None
        permissions = grant.get("permissions") if isinstance(grant, dict) else None
        if not isinstance(entity, dict) or not isinstance(permissions, list):
            return None
        is_target = (entity.get("type") in ("user", "org") and entity.get("name") == namespace) or (
            entity.get("type") == "model" and entity.get("name") == repo_id
        )
        if is_target and REPO_WRITE in permissions:
            return True
    return False


def check_write_access(repo_id: str, *, token: str | None = None, api: HfApi | None = None) -> None:
    """Raise HubError unless the token can write to the repo (create_repo(exist_ok=True) succeeds
    for an existing readable repo even with a read-only token). Warn when whoami's answer
    cannot be interpreted rather than refuse a token that may well work."""
    api = api or HfApi()
    try:
        info = api.whoami(token=token)
    except Exception as exc:
        raise HubError(
            f"cannot check the Hugging Face token for {repo_id}: {exc} "
            "(log in with `hf auth login` or set HF_TOKEN to a token with write access)"
        ) from exc
    auth = info.get("auth") if isinstance(info, dict) else None
    access = auth.get("accessToken") if isinstance(auth, dict) else None
    role = access.get("role") if isinstance(access, dict) else None
    if role in WRITE_ROLES:
        return
    namespace = repo_id.split("/")[0] if "/" in repo_id else str(info.get("name"))
    if role == "read":
        allowed: bool | None = False
    elif role == "fineGrained":
        allowed = _fine_grained_can_write(access.get("fineGrained"), repo_id, namespace)
    else:
        allowed = None
    if allowed is None:
        warnings.warn(
            f"could not confirm that the Hugging Face token has write access to {repo_id}; "
            "if it does not, the push fails after training",
            stacklevel=2,
        )
    elif not allowed:
        raise HubError(
            f"the Hugging Face token has no write access to {repo_id} (role {role!r}); "
            f"use a write token, or a fine-grained token with {REPO_WRITE} for {namespace}"
        )


def ensure_repo(
    repo_id: str, *, token: str | None = None, private: bool = True, api: HfApi | None = None
) -> str:
    """Check the token can write to the repo, then create the repo if it is missing (private
    by default); return its URL."""
    api = api or HfApi()
    check_write_access(repo_id, token=token, api=api)
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


def pull_metrics(
    repo_id: str, *, token: str | None = None, download: Callable[..., str] = hf_hub_download
) -> dict:
    """An adapter repo's metrics.json (what the results view shows), without its weights."""
    try:
        path = download(repo_id, "metrics.json", token=token)
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise HubError(f"cannot read metrics.json from {repo_id}: {exc}") from exc


@dataclass
class RunFiles:
    metrics: dict
    config: dict  # train_config.yaml
    schema: dict | None  # schema.yaml, when the adapter has one


def pull_run_files(
    repo_id: str, *, token: str | None = None, download: Callable[..., str] = hf_hub_download
) -> RunFiles:
    """What the results view needs from an adapter repo: its metrics, config and schema."""
    try:
        metrics = json.loads(
            Path(download(repo_id, "metrics.json", token=token)).read_text("utf-8")
        )
        config = yaml.safe_load(
            Path(download(repo_id, "train_config.yaml", token=token)).read_text("utf-8")
        )
    except Exception as exc:
        raise HubError(f"cannot read the results of {repo_id}: {exc}") from exc
    try:
        schema = yaml.safe_load(
            Path(download(repo_id, "schema.yaml", token=token)).read_text("utf-8")
        )
    except Exception:  # an adapter without a separate schema file keeps it in the config
        schema = None
    return RunFiles(metrics, config or {}, schema)
