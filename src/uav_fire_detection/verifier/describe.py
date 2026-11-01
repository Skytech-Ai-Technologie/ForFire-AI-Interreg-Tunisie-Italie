# SPDX-License-Identifier: Apache-2.0
"""Template-based scene description.

Produces a deterministic, one-sentence description from the verifier output
and detector objects. The method tag returned alongside the text is recorded
in alert payloads so the description provenance is always visible in logged
data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Sequence, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .clip_verifier import VerificationOutput
    from ..detector.detector import Detection

DESCRIPTION_METHOD = "template_v1"


def build_scene_description(
    output: "VerificationOutput",
    detections: Sequence["Detection"],
) -> str:
    """Return a one-sentence, deterministic scene description."""
    category_label = output.top_prompt.category.replace("_", " ")
    parts: List[str] = [
        'Aerial frame most resembles "{}" ({} similarity {:.2f}).'.format(
            output.top_prompt.text, category_label, output.top_prompt_score
        )
    ]
    if detections:
        summarized = ", ".join(
            "{} {:.2f}".format(d.class_name, d.confidence) for d in detections
        )
        parts.append("Detector objects: {}.".format(summarized))
    else:
        parts.append("Detector objects: none.")
    return " ".join(parts)


def description_provenance() -> Tuple[str, str]:
    """Return (method tag, provenance note)."""
    return (
        DESCRIPTION_METHOD,
        "template-generated description",
    )
