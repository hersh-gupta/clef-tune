"""YAML helpers using YAML 1.2 booleans: only true/false are booleans, so keys like
`yes`, `no`, `on` and `off` stay strings."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

_BOOL_TAG = "tag:yaml.org,2002:bool"


class _Loader(yaml.SafeLoader):
    pass


_Loader.yaml_implicit_resolvers = {
    first: [(tag, regexp) for tag, regexp in resolvers if tag != _BOOL_TAG]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_Loader.add_implicit_resolver(
    _BOOL_TAG,
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)


def load_yaml_text(text: str) -> Any:
    return yaml.load(text, Loader=_Loader)  # _Loader is a SafeLoader subclass


def load_yaml_file(path: str | Path) -> Any:
    return load_yaml_text(Path(path).read_text(encoding="utf-8-sig"))


def dump_yaml(data: Any) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
