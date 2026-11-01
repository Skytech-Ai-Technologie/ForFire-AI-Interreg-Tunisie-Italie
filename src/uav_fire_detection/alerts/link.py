# SPDX-License-Identifier: Apache-2.0
"""Alert transport abstractions.

* :class:`FileAlertLink` — append-only JSONL sink for simulations and bench
  tests;
* :class:`MavlinkAlertLink` — pymavlink STATUSTEXT fragmentation of a compact
  alert payload. STATUSTEXT carries 50 bytes per message, so this transport
  suits short alert summaries, not full-resolution imagery; production
  systems should use a dedicated data link.
* :class:`MockAlertLink` — in-memory link for tests and dry runs.

Links are lossy by nature; combine them with :class:`RetryQueue` for degraded
operation (see :func:`send_with_retry` / :func:`drain_retry_queue`).
"""

from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from .store import RetryQueue


class AlertLink(ABC):
    """A one-way alert transport. ``send`` returns True when the payload was
    accepted by the transport (not necessarily acknowledged end-to-end)."""

    @abstractmethod
    def send(self, alert_json: str, event_id: str) -> bool: ...


class FileAlertLink(AlertLink):
    """Appends each alert as one JSON line; always succeeds."""

    def __init__(self, path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def send(self, alert_json: str, event_id: str) -> bool:
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(alert_json + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        return True


class MavlinkAlertLink(AlertLink):
    """Fragments a compact alert JSON into MAVLink STATUSTEXT messages.

    Wire format per message (<= 50 bytes, MAVLink v1 STATUSTEXT size):
    ``FA|<i>/<n>|<event8>|<payload chunk>``. ``event8`` is the first 8 chars
    of the event id so fragments can be reassembled and deduplicated
    downstream. Payloads longer than ``max_payload_chars`` are truncated with
    a ``|TRUNC`` marker.
    """

    MAX_TEXT = 50  # MAVLink STATUSTEXT payload size (bytes)
    SEVERITY_INFO = 6  # MAV_SEVERITY_INFO

    def __init__(
        self,
        connection_string: str,
        source_system_id: int = 1,
        source_component_id: int = 190,  # MAV_COMPONENT: onboard computer/companion
        severity: int = SEVERITY_INFO,
        max_payload_chars: int = 800,
    ) -> None:
        self.connection_string = connection_string
        self.source_system_id = source_system_id
        self.source_component_id = source_component_id
        self.severity = severity
        self.max_payload_chars = max_payload_chars
        self._conn = None
        self._lock = threading.Lock()

    def _connect(self):
        from pymavlink import mavutil  # deferred heavy/optional import

        self._conn = mavutil.mavlink_connection(
            self.connection_string,
            source_system=self.source_system_id,
            source_component=self.source_component_id,
        )

    @staticmethod
    def _fragment(payload: str, event8: str) -> List[str]:
        header_template = "FA|{:02d}/{}|{}|"
        # Chunk size = 50 minus the widest possible header (assume n <= 99).
        probe = header_template.format(99, 99, event8)
        chunk_size = MavlinkAlertLink.MAX_TEXT - len(probe)
        chunks = [payload[i : i + chunk_size] for i in range(0, len(payload), chunk_size)]
        n = max(1, len(chunks))
        return [header_template.format(i, n, event8) + chunk for i, chunk in enumerate(chunks, 1)]

    def send(self, alert_json: str, event_id: str) -> bool:
        with self._lock:
            try:
                if self._conn is None:
                    self._connect()
                payload = alert_json
                if len(payload) > self.max_payload_chars:
                    payload = payload[: self.max_payload_chars - 6] + "|TRUNC"
                event8 = event_id.replace("-", "")[:8]
                for text in self._fragment(payload, event8):
                    self._conn.mav.statustext_send(self.severity, text.encode("utf-8")[: self.MAX_TEXT])
                return True
            except Exception:  # noqa: BLE001 - transport errors -> retry queue
                return False


class MockAlertLink(AlertLink):
    """In-memory link for tests and dry runs; configurable failure mode."""

    def __init__(self, fail_first: int = 0) -> None:
        self.calls: List[str] = []
        self._fail_first = fail_first

    def send(self, alert_json: str, event_id: str) -> bool:
        if self._fail_first > 0:
            self._fail_first -= 1
            self.calls.append(event_id)
            return False
        self.calls.append(event_id)
        return True


def send_with_retry(alert, link: AlertLink, queue: RetryQueue) -> bool:
    """Send one alert; on failure, persist it to the retry queue.

    Returns the link's success flag. At-least-once semantics: an alert that
    reached the link but whose ack was lost may be re-sent by a later drain.
    """
    payload = alert.to_json() if hasattr(alert, "to_json") else str(alert)
    ok = link.send(payload, alert.event_id)
    if not ok:
        queue.enqueue(alert.event_id, payload)
    return ok


def drain_retry_queue(queue: RetryQueue, link: AlertLink, max_sends: Optional[int] = None) -> int:
    """Re-send queued alerts oldest-first until the link fails or the queue
    empties. Only acked (successfully sent) items are removed. Returns the
    number of alerts confirmed sent."""
    sent = 0
    while True:
        if max_sends is not None and sent >= max_sends:
            return sent
        item = queue.pop()
        if item is None:
            return sent
        event_id, payload = item
        if link.send(payload, event_id):
            queue.ack(event_id)
            sent += 1
        else:
            return sent
