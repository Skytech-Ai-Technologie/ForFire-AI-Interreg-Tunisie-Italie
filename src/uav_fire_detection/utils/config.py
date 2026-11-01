# SPDX-License-Identifier: Apache-2.0
"""YAML configuration helpers with clear, actionable errors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


def load_yaml(path) -> Dict[str, Any]:
    import yaml

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError("config file not found: {}".format(p))
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping: {}".format(p))
    return data


def deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge; ``override`` wins. Returns a new dict."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def require_keys(mapping: Mapping[str, Any], keys: Iterable[str], context: str = "") -> None:
    missing = [k for k in keys if k not in mapping]
    if missing:
        where = " in {}".format(context) if context else ""
        raise KeyError("missing required key(s) {}{}".format(missing, where))
