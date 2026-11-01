# SPDX-License-Identifier: Apache-2.0
"""Onboard logging and bounded retry storage.

Full-resolution frames are written as JPEG under ``<root>/<UTC-date>/``.
Alerts that could not be sent are retained in a bounded, NVMe-backed retry
queue with a configurable capacity and drop policy (default: drop-oldest)
and retried after link recovery with explicit acknowledgement.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from .alert import utc_now_iso

DROP_POLICIES = ("drop_oldest", "reject")


class ImageLogStore:
    """Writes full-resolution frames as JPEG under ``<root>/<UTC-date>/``.

    Uses ``cv2.imencode`` + atomic rename so a power loss never leaves a
    half-written image. Frame data stays onboard; only alert summaries travel
    over the telemetry link.
    """

    def __init__(self, root, jpg_quality: int = 90) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.jpg_quality = int(jpg_quality)

    def save_frame(self, frame, frame_id: str, created_utc: Optional[str] = None) -> str:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError("opencv-python is required to encode frames") from exc

        day = (created_utc or utc_now_iso())[:10]
        directory = self.root / day
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "{}.jpg".format(frame_id)

        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpg_quality]
        )
        if not ok:
            raise IOError("cv2.imencode failed for frame {}".format(frame_id))
        tmp = path.with_suffix(".jpg.tmp")
        tmp.write_bytes(buf.tobytes())
        os.replace(str(tmp), str(path))
        return str(path)


class RetryQueue:
    """Filesystem FIFO with a hard item bound; survives restarts.

    Entries are ``<utc-ms>-<seq>-<event_id>.json`` files, so ordering is
    creation order and state is the directory itself. ``pop`` peeks the
    oldest entry; ``ack`` deletes it after a confirmed send, so a crash
    between send and ack at worst duplicates an alert (at-least-once
    semantics).
    """

    def __init__(self, directory, max_items: int = 500, drop_policy: str = "drop_oldest") -> None:
        if drop_policy not in DROP_POLICIES:
            raise ValueError("drop_policy must be one of {}".format(DROP_POLICIES))
        if max_items < 1:
            raise ValueError("max_items must be >= 1")
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_items = int(max_items)
        self.drop_policy = drop_policy
        self._lock = threading.Lock()
        self._seq = 0

    def _entries(self) -> List[Path]:
        return sorted(self.dir.glob("*.json"))

    @staticmethod
    def _event_id_of(path: Path) -> str:
        # filename layout: <utc-ms>-<seq>-<event_id>.json (event ids are hex)
        return path.stem.split("-", 2)[2]

    def pending(self) -> List[str]:
        with self._lock:
            return [self._event_id_of(p) for p in self._entries()]

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries())

    def enqueue(self, event_id: str, payload: str) -> str:
        with self._lock:
            entries = self._entries()
            if len(entries) >= self.max_items:
                if self.drop_policy == "reject":
                    raise RuntimeError("retry queue full (max_items={})".format(self.max_items))
                # Documented default: drop oldest to make room (bounded ring).
                for victim in entries[: len(entries) - self.max_items + 1]:
                    victim.unlink()

            self._seq += 1
            stamp = int(time.time() * 1000)
            final = self.dir / "{}-{:06d}-{}.json".format(stamp, self._seq, event_id)
            tmp = self.dir / ".tmp-{}-{}-{}.json".format(os.getpid(), threading.get_ident(), uuid.uuid4().hex)
            tmp.write_text(payload, encoding="utf-8")
            os.replace(str(tmp), str(final))
        return event_id

    def pop(self) -> Optional[Tuple[str, str]]:
        """Return the oldest (event_id, payload) without removing it."""
        with self._lock:
            entries = self._entries()
            if not entries:
                return None
            oldest = entries[0]
            return self._event_id_of(oldest), oldest.read_text(encoding="utf-8")

    def ack(self, event_id: str) -> bool:
        with self._lock:
            removed = False
            for p in self._entries():
                if self._event_id_of(p) == event_id:
                    p.unlink()
                    removed = True
            return removed
