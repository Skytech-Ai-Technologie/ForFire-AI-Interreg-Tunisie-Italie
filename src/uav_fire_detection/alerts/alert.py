# SPDX-License-Identifier: Apache-2.0
"""Structured alert payload.

Fields: event id, UTC timestamp, decision, source, detector objects, CLIP
verification block, scene description, GPS position, gimbal pose, thumbnail
reference, timings, and notes. Bump SCHEMA_VERSION on incompatible changes.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0"
ALLOWED_DECISIONS = ("alert", "review", "suppress", "no_detection")

# Deterministic UUID namespace so identical frame content + timestamp produce
# identical event ids across runs (deduplication / reproducibility).
_EVENT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "uav-fire-detection:event-id")


@dataclass
class GeoPosition:
    """WGS84 position (degrees); altitudes in metres MSL."""

    latitude: float
    longitude: float
    altitude_m: Optional[float] = None


@dataclass
class Attitude:
    """Gimbal/UAV pose in degrees (frame convention must be documented by the
    operator, e.g. body vs gimbal)."""

    roll_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    yaw_deg: Optional[float] = None


@dataclass
class Alert:
    event_id: str
    created_utc: str
    decision: str
    source: str
    frame_id: str
    reduced_confidence: bool = False
    detections: List[Dict[str, Any]] = field(default_factory=list)
    clip: Optional[Dict[str, Any]] = None
    scene_description: Optional[str] = None
    scene_description_method: Optional[str] = None
    gps: Optional[GeoPosition] = None
    pose: Optional[Attitude] = None
    thumbnail_path: Optional[str] = None
    schema_version: str = SCHEMA_VERSION
    pipeline_version: str = ""
    notes: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        required = {
            "event_id": self.event_id,
            "created_utc": self.created_utc,
            "decision": self.decision,
            "source": self.source,
            "frame_id": self.frame_id,
        }
        missing = sorted(k for k, v in required.items() if not v)
        if missing:
            raise ValueError("alert missing required fields: {}".format(missing))
        if self.decision not in ALLOWED_DECISIONS:
            raise ValueError(
                "alert decision {!r} not in {}".format(self.decision, ALLOWED_DECISIONS)
            )
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                "alert schema_version {!r} != builder version {!r}".format(
                    self.schema_version, SCHEMA_VERSION
                )
            )

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Alert":
        data = dict(data)
        data.setdefault("extra", {})
        data["extra"].update(
            {k: data.pop(k) for k in list(data) if k not in {f.name for f in fields(cls)}}
        )
        gps = data.pop("gps", None)
        pose = data.pop("pose", None)
        if isinstance(gps, dict):
            gps = GeoPosition(**gps)
        if isinstance(pose, dict):
            pose = Attitude(**pose)
        return cls(gps=gps, pose=pose, **data)

    @classmethod
    def from_pipeline(
        cls,
        result,
        *,
        telemetry: Optional[Dict[str, Any]] = None,
        thumbnail_path: Optional[str] = None,
        event_id: Optional[str] = None,
        pipeline_version: str = "",
    ) -> "Alert":
        """Build an alert from a pipeline FrameResult.

        ``telemetry`` is an optional dict with optional ``gps`` and ``pose``
        sub-dicts (keys matching GeoPosition/Attitude fields).
        """
        from .. import __version__
        from ..pipeline.pipeline import SOURCE_DETECTOR_ONLY

        telemetry = telemetry or {}
        gps = GeoPosition(**telemetry["gps"]) if telemetry.get("gps") else None
        pose = Attitude(**telemetry["pose"]) if telemetry.get("pose") else None

        created = utc_now_iso()
        clip_payload: Optional[Dict[str, Any]] = None
        if result.verification is not None:
            v = result.verification
            clip_payload = {
                "category_scores": {
                    "affirmative": v.category_scores.affirmative,
                    "confuser_negative": v.category_scores.confuser_negative,
                    "ambiguous": v.category_scores.ambiguous,
                },
                "top_prompt_id": v.top_prompt.id,
                "top_prompt_text": v.top_prompt.text,
                "top_prompt_category": v.top_prompt.category,
                "top_prompt_score": v.top_prompt_score,
                "policy_reason": result.policy_reason,
                "verifier_elapsed_ms": result.timings_ms.get("verifier_ms"),
            }

        return cls(
            event_id=event_id or new_event_id(result.frame_id, created),
            created_utc=created,
            decision=result.decision,
            source=result.source,
            frame_id=result.frame_id,
            reduced_confidence=result.reduced_confidence,
            detections=[
                {
                    "class_name": d.class_name,
                    "confidence": round(d.confidence, 6),
                    "xyxy": [round(c, 2) for c in d.xyxy],
                }
                for d in result.detections
            ],
            clip=clip_payload,
            scene_description=result.scene_description,
            scene_description_method=result.scene_description_method,
            gps=gps,
            pose=pose,
            thumbnail_path=thumbnail_path,
            pipeline_version=pipeline_version or __version__,
            notes=list(result.notes),
            extra={
                "timings_ms": dict(result.timings_ms),
                "verifier_source": result.source,
                "detector_only_source": result.source == SOURCE_DETECTOR_ONLY,
            },
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_event_id(frame_key: str = "", created_utc: str = "") -> str:
    """Deterministic event id: uuid5 over frame content + timestamp."""
    return str(uuid.uuid5(_EVENT_NAMESPACE, "{}|{}".format(frame_key, created_utc)))


def event_id_from_frame_bytes(frame_bytes: bytes, created_utc: str) -> str:
    return new_event_id(hashlib.sha256(frame_bytes).hexdigest(), created_utc)
