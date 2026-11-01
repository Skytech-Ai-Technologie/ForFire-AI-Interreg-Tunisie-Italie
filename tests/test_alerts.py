# SPDX-License-Identifier: Apache-2.0
"""Alert schema, retry queue, and link tests."""

import json

import pytest

from uav_fire_detection.alerts import (
    Alert,
    FileAlertLink,
    MockAlertLink,
    RetryQueue,
    drain_retry_queue,
    send_with_retry,
)
from uav_fire_detection.alerts.alert import SCHEMA_VERSION, event_id_from_frame_bytes
from uav_fire_detection.detector.detector import Detection
from uav_fire_detection.pipeline.pipeline import FrameResult


def make_result(frame_id="f-1", decision="alert", source="yolo_clip"):
    return FrameResult(
        frame_id=frame_id,
        decision=decision,
        source=source,
        gate_passed=True,
        invoked_verifier=True,
        reduced_confidence=False,
        detections=[Detection("fire", 0.82, (1.0, 2.0, 3.0, 4.0), 0)],
        policy_reason="affirmative_above_alert_threshold",
        verification=None,
        scene_description="a description",
        scene_description_method="template_v1",
        timings_ms={"detector_ms": 5.0, "total_ms": 9.0},
        notes=["note-1"],
    )


def test_from_pipeline_and_json_roundtrip():
    alert = Alert.from_pipeline(
        make_result(),
        telemetry={"gps": {"latitude": 36.4, "longitude": 10.6, "altitude_m": 120.0},
                   "pose": {"yaw_deg": 90.0}},
        thumbnail_path="frames/f-1.jpg",
    )
    alert.validate()
    restored = Alert.from_dict(json.loads(alert.to_json()))
    assert restored.gps.latitude == 36.4
    assert restored.gps.altitude_m == 120.0
    assert restored.pose.yaw_deg == 90.0
    assert restored.detections[0]["class_name"] == "fire"
    assert restored.detections[0]["confidence"] == 0.82
    assert restored.scene_description_method == "template_v1"
    assert restored.schema_version == SCHEMA_VERSION
    assert restored.notes == ["note-1"]
    assert restored.extra["timings_ms"]["total_ms"] == 9.0


def test_missing_required_fields_raise():
    alert = Alert(event_id="", created_utc="2026-01-01T00:00:00Z",
                  decision="alert", source="yolo_clip", frame_id="f")
    with pytest.raises(ValueError):
        alert.validate()


def test_invalid_decision_raises():
    alert = Alert(event_id="e", created_utc="2026-01-01T00:00:00Z",
                  decision="maybe", source="yolo_clip", frame_id="f")
    with pytest.raises(ValueError):
        alert.validate()


def test_event_id_is_deterministic():
    e1 = event_id_from_frame_bytes(b"abc", "2026-01-01T00:00:00Z")
    e2 = event_id_from_frame_bytes(b"abc", "2026-01-01T00:00:00Z")
    e3 = event_id_from_frame_bytes(b"abd", "2026-01-01T00:00:00Z")
    assert e1 == e2
    assert e1 != e3


def test_retry_queue_bounded_drop_oldest(tmp_path):
    queue = RetryQueue(tmp_path / "q", max_items=2)
    for i in range(3):
        queue.enqueue("evt-{}".format(i), json.dumps({"i": i}))
    assert len(queue) == 2
    assert queue.pending() == ["evt-1", "evt-2"]


def test_retry_queue_reject_policy(tmp_path):
    queue = RetryQueue(tmp_path / "q", max_items=1, drop_policy="reject")
    queue.enqueue("evt-a", "p1")
    with pytest.raises(RuntimeError):
        queue.enqueue("evt-b", "p2")


def test_retry_queue_survives_restart(tmp_path):
    RetryQueue(tmp_path / "q").enqueue("evt-a", "p1")
    queue = RetryQueue(tmp_path / "q")  # fresh instance, same directory
    queue.enqueue("evt-b", "p2")
    assert queue.pending() == ["evt-a", "evt-b"]


def test_pop_and_ack(tmp_path):
    queue = RetryQueue(tmp_path / "q")
    queue.enqueue("evt-a", "p1")
    event_id, payload = queue.pop()
    assert event_id == "evt-a"
    assert payload == "p1"
    # pop does not remove: at-least-once semantics
    assert queue.pop()[0] == "evt-a"
    assert queue.ack(event_id) is True
    assert len(queue) == 0
    assert queue.pop() is None


def test_send_with_retry_and_drain(tmp_path):
    queue = RetryQueue(tmp_path / "q", max_items=10)
    link = MockAlertLink(fail_first=2)

    first = Alert.from_pipeline(make_result(frame_id="f-1"))
    second = Alert.from_pipeline(make_result(frame_id="f-2"))
    third = Alert.from_pipeline(make_result(frame_id="f-3"))

    assert send_with_retry(first, link, queue) is False
    assert send_with_retry(second, link, queue) is False
    assert send_with_retry(third, link, queue) is True
    assert len(queue) == 2

    sent = drain_retry_queue(queue, link)
    assert sent == 2
    assert len(queue) == 0


def test_file_alert_link_writes_jsonl(tmp_path):
    link = FileAlertLink(tmp_path / "alerts.jsonl")
    alert = Alert.from_pipeline(make_result())
    assert link.send(alert.to_json(), alert.event_id) is True
    assert link.send(alert.to_json(), alert.event_id) is True
    lines = (tmp_path / "alerts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["decision"] == "alert"
