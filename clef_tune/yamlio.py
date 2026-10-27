"""YAML helpers using YAML 1.2 booleans: only true/false are booleans, so keys like
`yes`, `no`, `on` and `off` stay strings. Numeric, null and date mapping keys keep their
source text. Syntax errors raise YAMLSyntaxError (a ValueError)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

_BOOL_TAG = "tag:yaml.org,2002:bool"
_STR_TAG = "tag:yaml.org,2002:str"
# Mapping keys with these tags keep their source text: `010:` stays "010" (not 8), `null:`
# stays "null" (not None), `1:30:` stays "1:30" (not 90).
_LITERAL_KEY_TAGS = {f"tag:yaml.org,2002:{name}" for name in ("int", "float", "null", "timestamp")}


class _Loader(yaml.SafeLoader):
    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        for key_node, _ in node.value:
            if isinstance(key_node, yaml.ScalarNode) and key_node.tag in _LITERAL_KEY_TAGS:
                key_node.tag = _STR_TAG
        return super().construct_mapping(node, deep=deep)


_Loader.yaml_implicit_resolvers = {
    first: [(tag, regexp) for tag, regexp in resolvers if tag != _BOOL_TAG]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_Loader.add_implicit_resolver(
    _BOOL_TAG,
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)


class YAMLSyntaxError(ValueError):
    """The text is not valid YAML."""


def load_yaml_text(text: str, source: str = "YAML") -> Any:
    try:
        return yaml.load(text, Loader=_Loader)  # _Loader is a SafeLoader subclass
    except yaml.YAMLError as exc:
        raise YAMLSyntaxError(f"{source}: invalid YAML: {exc}") from exc


def load_yaml_file(path: str | Path) -> Any:
    path = Path(path)
    return load_yaml_text(path.read_text(encoding="utf-8-sig"), source=path.name)


def dump_yaml(data: Any) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
