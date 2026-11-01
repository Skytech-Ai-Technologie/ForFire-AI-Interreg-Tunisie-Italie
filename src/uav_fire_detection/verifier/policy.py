# SPDX-License-Identifier: Apache-2.0
"""Stage 2 decision policy.

Rules:

* affirmative similarity  > 0.60          -> confirmed alert
* affirmative similarity in [0.45, 0.60)  -> human review
* confuser-negative similarity > 0.70     -> suppress the YOLO detection
* ambiguous prompts / unresolved cases    -> human review

Tie-breaking is a fixed, fire-safety-conservative total order: alert >
suppress > review band > default review.

This module is deliberately pure Python (no torch/open_clip imports) so the
policy is unit-testable on any machine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Decision(str, Enum):
    ALERT = "alert"
    REVIEW = "review"
    SUPPRESS = "suppress"


@dataclass(frozen=True)
class VerificationScores:
    """Aggregated per-category CLIP cosine similarities (raw cosine, [~-1, 1])."""

    affirmative: float
    confuser_negative: float
    ambiguous: float


@dataclass(frozen=True)
class PolicyConfig:
    alert_threshold: float = 0.60     # alert fires strictly above
    review_threshold: float = 0.45    # review band lower bound is inclusive
    suppress_threshold: float = 0.70  # suppression fires strictly above


@dataclass(frozen=True)
class PolicyOutcome:
    decision: Decision
    reason: str  # stable machine-readable code (see module docstring)
    scores: VerificationScores


# Reason codes (stable contract for logs and tests)
REASON_ALERT = "affirmative_above_alert_threshold"
REASON_SUPPRESS = "confuser_negative_above_suppress_threshold"
REASON_REVIEW_BAND = "affirmative_in_review_band"
REASON_REVIEW_DEFAULT = "default_review_unresolved"
REASON_INVALID_SCORES = "invalid_scores"


class VerificationPolicy:
    """Total-order decision policy with fixed tie-breaking.

    Precedence (fire-safety conservative):

    1. alert     — a strong affirmative outranks a strong confuser-negative,
                   so overlapping confuser scores can never suppress a real
                   fire;
    2. suppress  — a strong confuser-negative removes weak affirmatives
                   (the role of confuser prompts such as sunsets and mist);
    3. review band — affirmative in [review_threshold, alert_threshold);
    4. default review — anything unresolved, including ambiguous-only signal
                   and non-finite scores.
    """

    def __init__(self, config: Optional[PolicyConfig] = None) -> None:
        self.config = config or PolicyConfig()

    def decide(self, scores: VerificationScores) -> PolicyOutcome:
        cfg = self.config
        values = (scores.affirmative, scores.confuser_negative, scores.ambiguous)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            return PolicyOutcome(Decision.REVIEW, REASON_INVALID_SCORES, scores)

        if scores.affirmative > cfg.alert_threshold:
            return PolicyOutcome(Decision.ALERT, REASON_ALERT, scores)
        if scores.confuser_negative > cfg.suppress_threshold:
            return PolicyOutcome(Decision.SUPPRESS, REASON_SUPPRESS, scores)
        if scores.affirmative >= cfg.review_threshold:
            return PolicyOutcome(Decision.REVIEW, REASON_REVIEW_BAND, scores)
        return PolicyOutcome(Decision.REVIEW, REASON_REVIEW_DEFAULT, scores)
