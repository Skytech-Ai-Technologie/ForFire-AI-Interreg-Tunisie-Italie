# SPDX-License-Identifier: Apache-2.0
"""Prompt library structure tests."""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from uav_fire_detection.verifier.prompts import (  # noqa: E402
    EXPECTED_COUNTS,
    Prompt,
    PromptLibrary,
)

ROOT = Path(__file__).resolve().parents[1]
PROMPTS_FILE = ROOT / "prompts" / "fire_smoke_prompts.yaml"

# Representative example prompts.
EXAMPLE_TEXTS = {
    "aerial view of burning forest with visible flames",
    "misty forest in early morning",
    "sunset over a lake with reflective water",
    "hazy aerial view over forest",
}


def test_library_structure():
    library = PromptLibrary.from_yaml(PROMPTS_FILE)
    assert len(library.prompts) == 24
    for category, expected in EXPECTED_COUNTS.items():
        assert len(library.by_category(category)) == expected


def test_example_texts_are_present():
    library = PromptLibrary.from_yaml(PROMPTS_FILE)
    texts = set(library.texts())
    for text in EXAMPLE_TEXTS:
        assert text in texts, "expected prompt missing: {}".format(text)


def test_ids_unique_and_categories_valid():
    library = PromptLibrary.from_yaml(PROMPTS_FILE)
    ids = [p.id for p in library.prompts]
    assert len(ids) == len(set(ids))
    assert all(p.category in EXPECTED_COUNTS for p in library.prompts)


def test_strict_counts_can_be_overridden_deliberately():
    lone = Prompt(id="X-1", text="x", category="affirmative")
    library = PromptLibrary([lone], strict_counts=False)
    assert len(library.prompts) == 1


def test_strict_counts_default_rejects_wrong_structure():
    with pytest.raises(ValueError):
        PromptLibrary([Prompt(id="X-1", text="x", category="affirmative")])
