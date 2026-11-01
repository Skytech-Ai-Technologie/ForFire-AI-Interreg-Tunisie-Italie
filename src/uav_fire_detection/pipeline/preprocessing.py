# SPDX-License-Identifier: Apache-2.0
"""Frame preparation utilities.

The pipeline passes the full camera frame to YOLO and lets Ultralytics
letterbox/resize to the configured inference size; OpenCLIP's own preprocess
(resize + centre crop) handles verifier input. Optionally, a crop around the
highest-confidence detection is extracted before CLIP
(``input_mode: crop`` in configs/clip_verifier.yaml, off by default).

Functions here work on any HxWx3 slice-able object (numpy arrays in
production); no numpy import is needed at module scope.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..detector.detector import Detection


def extract_primary_crop(
    frame,
    detections: Sequence[Detection],
    margin_ratio: float = 0.25,
):
    """Crop ``frame`` around the highest-confidence detection.

    The box is expanded by ``margin_ratio`` of its own size on every side and
    clamped to the frame bounds when the frame exposes ``.shape``. Returns
    None when no usable box remains, so the caller can fall back to the full
    frame.
    """
    if not detections:
        return None

    best = max(detections, key=lambda d: d.confidence)
    x1, y1, x2, y2 = best.xyxy
    if x2 <= x1 or y2 <= y1:
        return None

    pad_x = (x2 - x1) * margin_ratio
    pad_y = (y2 - y1) * margin_ratio
    cx1, cy1, cx2, cy2 = x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y

    if hasattr(frame, "shape") and getattr(frame, "ndim", 0) >= 2:
        height, width = frame.shape[0], frame.shape[1]
        cx1 = max(0.0, cx1)
        cy1 = max(0.0, cy1)
        cx2 = min(float(width), cx2)
        cy2 = min(float(height), cy2)

    cx1, cy1, cx2, cy2 = int(round(cx1)), int(round(cy1)), int(round(cx2)), int(round(cy2))
    if cx2 - cx1 < 2 or cy2 - cy1 < 2:
        return None
    return frame[cy1:cy2, cx1:cx2]
