"""Write a portable training bundle: normalized train/eval JSONL, copied images, and a
self-contained config.yaml. Bundles are what the UI downloads, what HF Jobs mounts, and what
any GPU machine trains from."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from clef_tune.config import Config
from clef_tune.data import URL_PREFIXES, Record
from clef_tune.validation import ValidationReport


def write_bundle(report: ValidationReport, config: Config, out_dir: str | Path) -> Path:
    if not report.ok:
        raise ValueError("cannot write a bundle while validation has errors")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = config.get_schema()
    copied: dict[str, str] = {}

    def row(record: Record) -> dict:
        data = record.to_json()
        if schema is not None and record.questions == schema.questions:
            del data["questions"]
        if record.images:
            data["images"] = [
                _bundle_image(path, out_dir / "images", copied) for path in record.images
            ]
        return data

    for name, records in (("train.jsonl", report.train), ("eval.jsonl", report.eval)):
        with open(out_dir / name, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(row(record), ensure_ascii=False) + "\n")

    bundled = config.to_portable_dict()
    bundled["data"] = {"train": "train.jsonl", "eval": "eval.jsonl"}
    bundled.get("output", {}).pop("dir", None)
    config_path = out_dir / "config.yaml"
    Config.model_validate(bundled).save(config_path)
    return config_path


def _bundle_image(path: str, images_dir: Path, copied: dict[str, str]) -> str:
    if path.startswith(URL_PREFIXES):
        return path
    if path not in copied:
        source = Path(path)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        target = images_dir / f"{digest}{source.suffix.lower()}"
        images_dir.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        copied[path] = f"images/{target.name}"
    return copied[path]


def zip_bundle(bundle_dir: str | Path, zip_path: str | Path) -> Path:
    base = str(Path(zip_path).with_suffix(""))
    return Path(shutil.make_archive(base, "zip", root_dir=bundle_dir))
