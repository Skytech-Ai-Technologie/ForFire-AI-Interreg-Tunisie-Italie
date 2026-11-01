#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Export the detector to TensorRT FP32/FP16/INT8 engines.

INT8 uses Ultralytics' built-in calibrator over the dataset yaml provided via
``--data``; export arguments and engine hashes are recorded as metadata for
auditability. Measure latencies with scripts/benchmark_jetson.py on the
target board.

Usage::

    python scripts/export_tensorrt.py --weights runs/detector/train/weights/best.pt \
        --precisions fp32 fp16 int8 --data runs/detector/train/datasets/dataset.yaml --device 0
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from uav_fire_detection.utils.environment import collect  # noqa: E402
from uav_fire_detection.utils.log import get_logger  # noqa: E402

logger = get_logger("export_tensorrt")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--precisions", nargs="+", default=["fp32", "fp16", "int8"],
                        choices=["fp32", "fp16", "int8"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workspace", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--data", default=None,
                        help="dataset yaml; REQUIRED for int8 (calibration source)")
    parser.add_argument("--out-dir", default="models/engines")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        logger.error("weights not found: %s", weights)
        return 1
    if "int8" in args.precisions and not args.data:
        parser.error("--data is required for int8 (calibration dataset yaml)")

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics is required: pip install 'uav-fire-detection[detector]'")
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for precision in args.precisions:
        logger.info("exporting %s engine", precision.upper())
        export_kwargs = {
            "format": "engine",
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "workspace": args.workspace,
            "simplify": True,
            "dynamic": False,
            "half": precision == "fp16",
            "int8": precision == "int8",
            "verbose": False,
        }
        if precision == "int8":
            export_kwargs["data"] = args.data
        engine_path = YOLO(str(weights)).export(**export_kwargs)
        engine_path = Path(engine_path)
        meta = {
            "precision": precision,
            "source_weights": str(weights),
            "engine_path": str(engine_path),
            "engine_sha256": _sha256(engine_path),
            "engine_size_bytes": engine_path.stat().st_size,
            "export_args": {k: (str(v) if isinstance(v, Path) else v) for k, v in export_kwargs.items()},
            "calibration_note": (
                "Ultralytics built-in INT8 calibrator over the dataset yaml "
                "provided via --data"
                if precision == "int8"
                else None
            ),
            "environment": collect(repo_dir=REPO_ROOT),
        }
        meta_path = out_dir / "engine_{}.meta.json".format(precision)
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str), encoding="utf-8")
        results[precision] = {"engine": str(engine_path), "meta": str(meta_path)}
        logger.info("wrote %s -> %s", engine_path, meta_path)

    logger.info("done: %s", json.dumps(results, indent=2))
    logger.info("measure real latencies with scripts/benchmark_jetson.py on the target board")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
