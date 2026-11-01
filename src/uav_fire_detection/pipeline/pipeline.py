# SPDX-License-Identifier: Apache-2.0
"""Conditional two-stage inference pipeline.

Control flow: YOLO first; CLIP only when at least one detection exceeds the
0.45 candidate gate; a decision policy on the verifier output; YOLO-only
fallback with a reduced-confidence flag when CLIP misses its time budget or
fails. Crop-vs-full-frame input, the budget value, and the fallback decision
are configuration (see PipelineConfig and configs/clip_verifier.yaml).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..detector.detector import Detection, candidate_gate
from ..verifier.describe import DESCRIPTION_METHOD, build_scene_description
from ..verifier.policy import Decision, VerificationPolicy
from .preprocessing import extract_primary_crop

DECISION_NO_DETECTION = "no_detection"
DECISION_ALERT = "alert"
DECISION_REVIEW = "review"
DECISION_SUPPRESS = "suppress"

SOURCE_YOLO_CLIP = "yolo_clip"
SOURCE_FALLBACK = "yolo_only_fallback"
SOURCE_DETECTOR_ONLY = "detector_only"

_DECISION_MAP = {
    Decision.ALERT: DECISION_ALERT,
    Decision.REVIEW: DECISION_REVIEW,
    Decision.SUPPRESS: DECISION_SUPPRESS,
}


@dataclass
class PipelineConfig:
    gate_threshold: float = 0.45        # Stage-2 candidate gate
    gate_strict: bool = True            # "above 0.45" -> strictly greater
    verifier_enabled: bool = True
    input_mode: str = "full_frame"      # full_frame | crop
    crop_margin_ratio: float = 0.25
    clip_time_budget_ms: float = 150.0
    # Fire-safety conservative fallback: when CLIP is unavailable/over budget,
    # the detector's decision is emitted with reduced_confidence=True instead
    # of silently dropping the frame. Set False to emit "review" instead.
    fallback_to_alert: bool = True


@dataclass
class FrameResult:
    frame_id: str
    decision: str
    source: str
    gate_passed: bool
    invoked_verifier: bool
    reduced_confidence: bool
    detections: List[Detection]
    policy_reason: Optional[str] = None
    verification: Optional[Any] = None  # verifier.VerificationOutput when CLIP ran
    scene_description: Optional[str] = None
    scene_description_method: Optional[str] = None
    timings_ms: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class FireDetectionPipeline:
    """Two-stage conditional pipeline over injected detector/verifier objects.

    The detector must expose ``detect(frame) -> List[Detection]``; the
    verifier must expose ``verify(frame, time_budget_ms=...) ->
    VerificationResult``. Injection keeps the class unit-testable with fakes
    and allows TensorRT engines to be swapped in transparently.
    """

    def __init__(self, detector, verifier=None, policy=None, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.detector = detector
        self.verifier = verifier if self.config.verifier_enabled else None
        self.policy = policy or VerificationPolicy()

    def process_frame(self, frame, frame_id: Optional[str] = None, metadata: Optional[dict] = None) -> FrameResult:
        cfg = self.config
        frame_id = frame_id or uuid.uuid4().hex
        metadata = dict(metadata or {})
        notes: List[str] = []
        timings: Dict[str, float] = {}

        t0 = time.perf_counter()
        detections = self.detector.detect(frame)
        timings["detector_ms"] = (time.perf_counter() - t0) * 1000.0

        gate_passed = candidate_gate(detections, cfg.gate_threshold, strict=cfg.gate_strict)
        if not gate_passed:
            timings["total_ms"] = (time.perf_counter() - t0) * 1000.0
            return FrameResult(
                frame_id=frame_id,
                decision=DECISION_NO_DETECTION,
                source=SOURCE_DETECTOR_ONLY,
                gate_passed=False,
                invoked_verifier=False,
                reduced_confidence=False,
                detections=detections,
                timings_ms=timings,
                notes=notes,
                metadata=metadata,
            )

        if self.verifier is None:
            return self._fallback(
                frame_id, detections, timings, notes, metadata, t0,
                note="verifier disabled or unavailable — detector-only decision",
                source=SOURCE_DETECTOR_ONLY,
            )

        verify_start = time.perf_counter()
        input_frame = frame
        if cfg.input_mode == "crop":
            crop = extract_primary_crop(frame, detections, cfg.crop_margin_ratio)
            if crop is not None:
                input_frame = crop
            else:
                notes.append("crop extraction fell back to full frame")

        try:
            vres = self.verifier.verify(input_frame, time_budget_ms=cfg.clip_time_budget_ms)
        except Exception as exc:  # noqa: BLE001 - degraded-mode fallback is intentional
            timings["verifier_ms"] = (time.perf_counter() - verify_start) * 1000.0
            return self._fallback(
                frame_id, detections, timings, notes, metadata, t0,
                note="verifier error: {}: {}".format(type(exc).__name__, exc),
                source=SOURCE_FALLBACK,
            )
        timings["verifier_ms"] = (time.perf_counter() - verify_start) * 1000.0

        if not vres.within_budget:
            return self._fallback(
                frame_id, detections, timings, notes, metadata, t0,
                note="verifier exceeded time budget ({:.1f} ms > {:.1f} ms)".format(
                    vres.elapsed_ms, cfg.clip_time_budget_ms
                ),
                source=SOURCE_FALLBACK,
            )
        if vres.output is None:
            return self._fallback(
                frame_id, detections, timings, notes, metadata, t0,
                note="verifier returned no scores",
                source=SOURCE_FALLBACK,
            )

        outcome = self.policy.decide(vres.output.category_scores)
        decision = _DECISION_MAP[outcome.decision]
        description = build_scene_description(vres.output, detections)
        timings["total_ms"] = (time.perf_counter() - t0) * 1000.0

        return FrameResult(
            frame_id=frame_id,
            decision=decision,
            source=SOURCE_YOLO_CLIP,
            gate_passed=True,
            invoked_verifier=True,
            reduced_confidence=False,
            detections=detections,
            policy_reason=outcome.reason,
            verification=vres.output,
            scene_description=description,
            scene_description_method=DESCRIPTION_METHOD,
            timings_ms=timings,
            notes=notes,
            metadata=metadata,
        )

    def _fallback(
        self, frame_id, detections, timings, notes, metadata, t0, note: str, source: str
    ) -> FrameResult:
        notes.append(note)
        decision = DECISION_ALERT if self.config.fallback_to_alert else DECISION_REVIEW
        timings["total_ms"] = (time.perf_counter() - t0) * 1000.0
        return FrameResult(
            frame_id=frame_id,
            decision=decision,
            source=source,
            gate_passed=True,
            invoked_verifier=True,
            reduced_confidence=True,
            detections=detections,
            policy_reason=None,
            verification=None,
            scene_description=None,
            scene_description_method=None,
            timings_ms=timings,
            notes=notes,
            metadata=metadata,
        )
