# SPDX-License-Identifier: Apache-2.0
"""Prompt library loading and validation for the CLIP verifier.

Structure: 10 affirmative fire-context prompts, 8 confuser-negative prompts,
6 ambiguous prompts (24 total). See prompts/fire_smoke_prompts.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

CATEGORIES = ("affirmative", "confuser_negative", "ambiguous")
EXPECTED_COUNTS = {"affirmative": 10, "confuser_negative": 8, "ambiguous": 6}


@dataclass(frozen=True)
class Prompt:
    id: str
    text: str
    category: str


class PromptLibrary:
    """Validated, ordered collection of verification prompts."""

    def __init__(self, prompts: Sequence[Prompt], strict_counts: bool = True) -> None:
        prompts = list(prompts)
        if not prompts:
            raise ValueError("prompt library is empty")
        seen_ids = set()
        seen_texts = set()
        for p in prompts:
            if p.category not in CATEGORIES:
                raise ValueError(f"prompt {p.id!r} has unknown category {p.category!r}")
            if not p.text or not p.text.strip():
                raise ValueError(f"prompt {p.id!r} has empty text")
            if p.id in seen_ids:
                raise ValueError(f"duplicate prompt id {p.id!r}")
            if p.text in seen_texts:
                raise ValueError(f"duplicate prompt text {p.text!r}")
            seen_ids.add(p.id)
            seen_texts.add(p.text)

        self._prompts: List[Prompt] = prompts
        self._by_category: Dict[str, List[Prompt]] = {
            cat: [p for p in prompts if p.category == cat] for cat in CATEGORIES
        }

        if strict_counts:
            actual = {cat: len(items) for cat, items in self._by_category.items()}
            if actual != EXPECTED_COUNTS:
                raise ValueError(
                    f"prompt counts {actual} do not match the expected structure "
                    f"{EXPECTED_COUNTS}; pass strict_counts=False to override "
                    "deliberately (record the change in your run manifest)"
                )

    @property
    def prompts(self) -> List[Prompt]:
        return list(self._prompts)

    def by_category(self, category: str) -> List[Prompt]:
        if category not in CATEGORIES:
            raise KeyError(category)
        return list(self._by_category[category])

    def texts(self) -> List[str]:
        return [p.text for p in self._prompts]

    @classmethod
    def from_yaml(cls, path, strict_counts: bool = True) -> "PromptLibrary":
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "categories" not in data:
            raise ValueError(f"{path} is not a valid prompt library file")

        declared = data.get("counts_declared") or {}
        prompts: List[Prompt] = []
        for category in CATEGORIES:
            for entry in data.get("categories", {}).get(category, []) or []:
                prompts.append(
                    Prompt(
                        id=str(entry["id"]),
                        text=str(entry["text"]),
                        category=category,
                    )
                )
        if declared and declared != EXPECTED_COUNTS:
            raise ValueError(
                f"counts_declared {declared} conflicts with the expected "
                f"structure {EXPECTED_COUNTS}"
            )
        return cls(prompts, strict_counts=strict_counts)
