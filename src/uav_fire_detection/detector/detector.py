# SPDX-License-Identifier: Apache-2.0
"""Stage 1 — Ultralytics YOLOv8s fire/smoke detector wrapper.

Fine-tuned from COCO-pretrained weights for the two classes ``fire`` and
``smoke`` (11.2M parameters). A frame is forwarded to Stage 2 when at least
one detection confidence is above 0.45 (see :func:`candidate_gate`). Heavy
imports (ultralytics/torch) are deferred to :class:`YoloDetector` construction
so the rest of the package stays importable on machines without them (e.g.
unit-test CI on CPU-only runners).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

FIRE_CLASS_ID = 0
SMOKE_CLASS_ID = 1
DEFAULT_CLASS_NAMES = {0: "fire", 1: "smoke"}
ALLOWED_CLASS_NAMES = ("fire", "smoke")


@dataclass(frozen=True)
class Detection:
    """One detector output box."""

    class_name: str
    confidence: float
    xyxy: Tuple[float, float, float, float]
    class_id: int


def candidate_gate(detections: Sequence[Detection], threshold: float, strict: bool = True) -> bool:
    """Stage-2 gate: True when at least one detection confidence exceeds the
    threshold. ``strict=True`` (default) uses ``>`` ("above 0.45"); set
    ``strict=False`` for ``>=`` semantics."""
    if strict:
        return any(d.confidence > threshold for d in detections)
    return any(d.confidence >= threshold for d in detections)


class YoloDetector:
    """Thin wrapper around ``ultralytics.YOLO``.

    Accepts ``.pt`` (PyTorch) and ``.engine`` (TensorRT) weights
    interchangeably through Ultralytics' own loading. Only ``fire``/``smoke``
    outputs are retained when the model exposes those class names; foreign
    class names (e.g. raw COCO weights) are kept and flagged via
    :attr:`keeps_foreign_classes` so callers never silently mistake a
    COCO-pretrained model for a fine-tuned one.
    """

    def __init__(
        self,
        weights: str,
        conf_threshold: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: Optional[str] = None,
        class_names: Optional[Dict[int, str]] = None,
    ) -> None:
        try:
            from ultralytics import YOLO  # deferred heavy import
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ImportError(
                "ultralytics is required for YoloDetector; "
                "install with: pip install 'uav-fire-detection[detector]'"
            ) from exc

        self.model = YOLO(weights)
        self.conf_threshold = float(conf_threshold)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device = device

        names = dict(class_names) if class_names else dict(getattr(self.model, "names", {}) or DEFAULT_CLASS_NAMES)
        self.names: Dict[int, str] = names
        self.keeps_foreign_classes = not set(names.values()) <= set(ALLOWED_CLASS_NAMES)

    def detect(self, frame) -> List[Detection]:
        """Run inference on one BGR frame (HxWx3, OpenCV convention)."""
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        boxes = results[0].boxes
        out: List[Detection] = []
        if boxes is None:
            return out
        confidences = boxes.conf.tolist()
        class_ids = boxes.cls.tolist()
        xyxy = boxes.xyxy.tolist()
        for conf, cls_id, box in zip(confidences, class_ids, xyxy):
            cls_id = int(cls_id)
            name = self.names.get(cls_id, str(cls_id))
            if self.keeps_foreign_classes and name not in ALLOWED_CLASS_NAMES:
                continue
            out.append(
                Detection(
                    class_name=name,
                    confidence=float(conf),
                    xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    class_id=cls_id,
                )
            )
        return out
