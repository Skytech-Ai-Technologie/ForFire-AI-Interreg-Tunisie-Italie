# SPDX-License-Identifier: Apache-2.0
"""Pipeline wiring tests with faked detector/verifier (no torch/ultralytics)."""

import pytest

from uav_fire_detection.detector.detector import Detection, candidate_gate
from uav_fire_detection.pipeline.pipeline import (
    DECISION_ALERT,
    DECISION_NO_DETECTION,
    DECISION_REVIEW,
    DECISION_SUPPRESS,
    SOURCE_DETECTOR_ONLY,
    SOURCE_FALLBACK,
    SOURCE_YOLO_CLIP,
    FireDetectionPipeline,
    PipelineConfig,
)
from uav_fire_detection.verifier.clip_verifier import VerificationOutput, VerificationResult
from uav_fire_detection.verifier.policy import VerificationScores
from uav_fire_detection.verifier.prompts import Prompt

TOP_PROMPT = Prompt(
    id="AFF-01",
    text="aerial view of burning forest with visible flames",
    category="affirmative",
)


def det(confidence=0.8, name="fire"):
    return Detection(class_name=name, confidence=confidence,
                     xyxy=(10.0, 10.0, 50.0, 50.0), class_id=0)


class FakeDetector:
    def __init__(self, detections):
        self._detections = list(detections)
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return list(self._detections)


class FakeVerifier:
    def __init__(self, affirmative=0.8, negative=0.1, ambiguous=0.2,
                 within_budget=True, error=None):
        self._affirmative = affirmative
        self._negative = negative
        self._ambiguous = ambiguous
        self._within_budget = within_budget
        self._error = error
        self.seen_frames = []
        self.seen_budgets = []

    def verify(self, frame, time_budget_ms=None):
        self.seen_frames.append(frame)
        self.seen_budgets.append(time_budget_ms)
        if self._error is not None:
            raise self._error
        output = VerificationOutput(
            per_prompt={"AFF-01": self._affirmative},
            category_scores=VerificationScores(self._affirmative, self._negative, self._ambiguous),
            top_prompt=TOP_PROMPT,
            top_prompt_score=self._affirmative,
        )
        return VerificationResult(output=output, elapsed_ms=10.0, within_budget=self._within_budget)


def pipeline(detections, verifier=None, config=None):
    return FireDetectionPipeline(FakeDetector(detections), verifier, config=config)


def test_no_detections_skips_verifier():
    verifier = FakeVerifier()
    result = pipeline([], verifier).process_frame("frame")
    assert result.decision == DECISION_NO_DETECTION
    assert result.gate_passed is False
    assert result.invoked_verifier is False
    assert verifier.seen_frames == []
    assert result.timings_ms["total_ms"] > 0


def test_detection_below_gate_skips_verifier():
    verifier = FakeVerifier()
    result = pipeline([det(0.44)], verifier).process_frame("frame")
    assert result.decision == DECISION_NO_DETECTION
    assert verifier.seen_frames == []


def test_gate_is_strictly_above_threshold():
    # 0.45 exactly does not pass the strict gate.
    verifier = FakeVerifier()
    result = pipeline([det(0.45)], verifier).process_frame("frame")
    assert result.decision == DECISION_NO_DETECTION


def test_alert_path_end_to_end():
    verifier = FakeVerifier(affirmative=0.8)
    result = pipeline([det(0.7), det(0.5, "smoke")], verifier).process_frame("frame")
    assert result.decision == DECISION_ALERT
    assert result.source == SOURCE_YOLO_CLIP
    assert result.gate_passed is True
    assert result.invoked_verifier is True
    assert result.reduced_confidence is False
    assert result.policy_reason == "affirmative_above_alert_threshold"
    assert "aerial view of burning forest" in result.scene_description
    assert result.scene_description_method == "template_v1"
    assert verifier.seen_budgets == [150.0]
    assert result.timings_ms["detector_ms"] > 0
    assert result.timings_ms["verifier_ms"] > 0


def test_suppress_path():
    verifier = FakeVerifier(affirmative=0.5, negative=0.9)
    result = pipeline([det(0.6)], verifier).process_frame("frame")
    assert result.decision == DECISION_SUPPRESS
    assert result.source == SOURCE_YOLO_CLIP


def test_review_band_path():
    verifier = FakeVerifier(affirmative=0.5, negative=0.2)
    result = pipeline([det(0.6)], verifier).process_frame("frame")
    assert result.decision == DECISION_REVIEW


def test_over_budget_falls_back_to_detector_only():
    verifier = FakeVerifier(within_budget=False)
    result = pipeline([det(0.7)], verifier).process_frame("frame")
    assert result.decision == DECISION_ALERT
    assert result.source == SOURCE_FALLBACK
    assert result.reduced_confidence is True
    assert result.verification is None
    assert any("time budget" in note for note in result.notes)


def test_verifier_error_falls_back():
    verifier = FakeVerifier(error=RuntimeError("CUDA OOM"))
    result = pipeline([det(0.7)], verifier).process_frame("frame")
    assert result.decision == DECISION_ALERT
    assert result.source == SOURCE_FALLBACK
    assert result.reduced_confidence is True
    assert any("verifier error" in note for note in result.notes)


def test_missing_verifier_uses_detector_only_fallback():
    result = pipeline([det(0.7)]).process_frame("frame")
    assert result.decision == DECISION_ALERT
    assert result.source == SOURCE_DETECTOR_ONLY
    assert result.reduced_confidence is True
    assert result.invoked_verifier is True


def test_fallback_can_be_configured_to_review():
    config = PipelineConfig(fallback_to_alert=False)
    result = pipeline([det(0.7)], config=config).process_frame("frame")
    assert result.decision == DECISION_REVIEW
    assert result.reduced_confidence is True


def test_full_frame_mode_passes_same_object():
    verifier = FakeVerifier()
    frame = object()
    pipeline([det(0.7)], verifier).process_frame(frame)
    assert verifier.seen_frames[0] is frame


def test_crop_mode_extracts_crop_around_best_detection():
    class FakeFrame:
        def __getitem__(self, item):
            return ("crop", item)

    verifier = FakeVerifier()
    config = PipelineConfig(input_mode="crop", crop_margin_ratio=0.25)
    frame = FakeFrame()
    result = pipeline([det(0.7)], verifier, config).process_frame(frame)
    # box (10,10,50,50) + 25% margin (10px each side) -> [0:60, 0:60]
    crop = verifier.seen_frames[0]
    assert crop is not frame
    assert crop[0] == "crop"
    key = crop[1]
    assert key[0] == slice(0, 60) and key[1] == slice(0, 60)
    assert not any("fell back" in note for note in result.notes)


def test_gate_helper_semantics():
    d = det(0.45)
    assert candidate_gate([d], 0.45, strict=True) is False
    assert candidate_gate([d], 0.45, strict=False) is True
    assert candidate_gate([det(0.44)], 0.45) is False
    assert candidate_gate([det(0.46)], 0.45) is True
