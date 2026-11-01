# SPDX-License-Identifier: Apache-2.0
"""Decision-policy tests: thresholds, band edges, and tie-breaking."""

import math

from uav_fire_detection.verifier.policy import (
    Decision,
    PolicyConfig,
    REASON_ALERT,
    REASON_INVALID_SCORES,
    REASON_REVIEW_BAND,
    REASON_REVIEW_DEFAULT,
    REASON_SUPPRESS,
    VerificationPolicy,
    VerificationScores,
)


def scores(affirmative, confuser_negative, ambiguous=0.0):
    return VerificationScores(affirmative, confuser_negative, ambiguous)


def test_strong_affirmative_alerts_even_with_strong_confuser():
    # Precedence choice: alert outranks suppress so a real fire is never
    # suppressed by an overlapping confuser score.
    outcome = VerificationPolicy().decide(scores(0.75, 0.85))
    assert outcome.decision is Decision.ALERT
    assert outcome.reason == REASON_ALERT


def test_alert_threshold_is_strict():
    policy = VerificationPolicy()
    assert policy.decide(scores(0.6000001, 0.0)).decision is Decision.ALERT
    outcome = policy.decide(scores(0.60, 0.0))
    assert outcome.decision is Decision.REVIEW
    assert outcome.reason == REASON_REVIEW_BAND


def test_confuser_suppresses_weak_affirmative():
    outcome = VerificationPolicy().decide(scores(0.50, 0.80))
    assert outcome.decision is Decision.SUPPRESS
    assert outcome.reason == REASON_SUPPRESS


def test_suppress_threshold_is_strict():
    outcome = VerificationPolicy().decide(scores(0.10, 0.70))
    assert outcome.decision is Decision.REVIEW
    assert outcome.reason == REASON_REVIEW_DEFAULT


def test_review_band_lower_bound_is_inclusive():
    outcome = VerificationPolicy().decide(scores(0.45, 0.0))
    assert outcome.decision is Decision.REVIEW
    assert outcome.reason == REASON_REVIEW_BAND


def test_below_review_band_defaults_to_review():
    outcome = VerificationPolicy().decide(scores(0.44, 0.10))
    assert outcome.decision is Decision.REVIEW
    assert outcome.reason == REASON_REVIEW_DEFAULT


def test_nonfinite_scores_are_reviewed_not_crashing():
    outcome = VerificationPolicy().decide(scores(float("nan"), 0.0))
    assert outcome.decision is Decision.REVIEW
    assert outcome.reason == REASON_INVALID_SCORES


def test_outcome_carries_scores():
    s = scores(0.9, 0.1, 0.2)
    outcome = VerificationPolicy().decide(s)
    assert outcome.scores is s


def test_custom_thresholds():
    policy = VerificationPolicy(
        PolicyConfig(alert_threshold=0.5, review_threshold=0.2, suppress_threshold=0.9)
    )
    assert policy.decide(scores(0.55, 0.0)).decision is Decision.ALERT
    assert policy.decide(scores(0.3, 0.0)).reason == REASON_REVIEW_BAND
    assert policy.decide(scores(0.1, 0.95)).decision is Decision.SUPPRESS
