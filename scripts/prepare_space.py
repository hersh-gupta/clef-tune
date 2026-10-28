"""Prepare the web UI's Hugging Face Space: copy deploy/space/ into OUT_DIR and pin its
requirements.txt to an exact clef-tune commit on GitHub (which must be pushed). Then upload it:

    python scripts/prepare_space.py build/space
    hf upload <you>/clef-tune build/space . --repo-type space
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

from clef_tune.launch.source import (
    Source,
    SourceError,
    checkout_dir,
    github_json,
    installed_direct_url,
    pin_commit,
    running_commit,
    source_repo,
)

SPACE = Path(__file__).resolve().parents[1] / "deploy" / "space"


def prepare(
    out_dir: Path, ref: str | None = None, *, fetch: Callable[[str], dict] = github_json
) -> Path:
    """Copy the Space files and write a requirements.txt pinned to the commit `ref` names
    (default: the commit this checkout is at)."""
    ref = ref or running_commit(installed_direct_url(), checkout_dir())
    source = pin_commit(Source(source_repo(), ref), fetch=fetch)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in SPACE.iterdir():
        if item.is_file():
            shutil.copyfile(item, out_dir / item.name)
    requirement = f"clef-tune[ui] @ git+{source.repo_url}@{source.ref}\n"
    (out_dir / "requirements.txt").write_text(requirement, encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path, help="Folder to write the Space files to.")
    parser.add_argument("--ref", help="clef-tune commit, tag or branch; default: this checkout.")
    args = parser.parse_args()
    try:
        out = prepare(args.out_dir, args.ref)
    except SourceError as exc:
        sys.exit(f"error: {exc}")
    print(f"Space files written to {out}")
    print(f"Upload them with: hf upload <you>/clef-tune {out} . --repo-type space")


if __name__ == "__main__":
    main()
