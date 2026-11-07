#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Benchmark pipeline latency/throughput.

Measures per-stage and end-to-end latency distributions (median/p95/p99)
after warm-up, CLIP invocation rate, decision mix, and throughput. On Jetson
it additionally captures raw ``tegrastats`` output (thermal/power context).

Frames from a directory source are preloaded to RAM to exclude decode I/O;
video/CSI sources include decode in the measurement (stated in the report).

Usage::

    python scripts/benchmark_jetson.py --weights models/engines/best_int8.engine \
        --source data/sample_frames --iters 500 --warmup 50
    python scripts/benchmark_jetson.py --weights runs/detector/train/weights/best.pt \
        --source /dev/video0 --source-type video
"""

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from uav_fire_detection import __version__  # noqa: E402
from uav_fire_detection.alerts import Alert, FileAlertLink, RetryQueue, send_with_retry  # noqa: E402
from uav_fire_detection.pipeline.factory import build_pipeline  # noqa: E402
from uav_fire_detection.utils.environment import collect  # noqa: E402
from uav_fire_detection.utils.imaging import imread_bgr  # noqa: E402
from uav_fire_detection.utils.log import get_logger  # noqa: E402
from uav_fire_detection.utils.timing import summarize_latencies  # noqa: E402

logger = get_logger("benchmark_jetson")

BENCHMARKS_DIR = REPO_ROOT / "benchmarks"


def load_reference_benchmarks() -> dict:
    """Load the reference benchmark tables shipped in benchmarks/."""
    from uav_fire_detection.utils.benchmarks import load_csv

    tables = (
        "inference_latency_int8.csv",
        "quantization_comparison.csv",
        "end_to_end_alert_latency.csv",
    )
    out = {}
    for name in tables:
        path = BENCHMARKS_DIR / name
        out[name.replace(".csv", "")] = load_csv(path) if path.exists() else None
    return out

IMX219_GSTREAMER = (
    "nvarguscamerasrc sensor-id={sensor_id} ! "
    "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! "
    "appsink drop=1"
)


def frame_source(source: str, source_type: str, limit: int):
    """Yield (frame, name) pairs. Directory sources are preloaded to RAM."""
    if source_type == "dir":
        path = Path(source)
        files = sorted(
            p for p in path.rglob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        )
        if not files:
            raise FileNotFoundError("no images under {}".format(path))
        logger.info("preloading %d frames from %s", len(files), path)
        frames = [(imread_bgr(p), p.name) for p in files]
        i = 0
        while i < limit:
            yield frames[i % len(frames)]
            i += 1
    elif source_type == "video":
        import cv2

        while True:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise IOError("cannot open video source: {}".format(source))
            yielded_here = 0
            while yielded_here < limit:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame, "video_frame_{:06d}".format(yielded_here)
                yielded_here += 1
            cap.release()
            limit -= yielded_here
            if limit <= 0:
                return
    elif source_type == "csi":
        import cv2

        pipeline_str = IMX219_GSTREAMER.format(sensor_id=int(source))
        cap = cv2.VideoCapture(pipeline_str, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            raise IOError("cannot open CSI camera (Jetson + nvarguscamerasrc required)")
        try:
            for i in range(limit):
                ok, frame = cap.read()
                if not ok:
                    raise IOError("CSI camera read failed at frame {}".format(i))
                yield frame, "csi_frame_{:06d}".format(i)
        finally:
            cap.release()
    else:
        raise ValueError("unknown source type: {}".format(source_type))


class TegraStatsCapture:
    """Captures raw ``tegrastats`` lines in a background thread (Jetson only)."""

    def __init__(self) -> None:
        self.lines = []
        self._proc = None
        self._thread = None

    def __enter__(self):
        exe = shutil.which("tegrastats")
        if exe is None:
            logger.info("tegrastats not found; thermal/power context not captured")
            return self
        self._proc = subprocess.Popen([exe], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

        def _reader():
            assert self._proc and self._proc.stdout
            for line in self._proc.stdout:
                self.lines.append(line.rstrip("\n"))

        self._thread = threading.Thread(target=_reader, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover
                self._proc.kill()
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True, help="image dir, video path/device, or CSI sensor index")
    parser.add_argument("--source-type", choices=["auto", "dir", "video", "csi"], default="auto")
    parser.add_argument("--detector-config", default="configs/detector.yaml")
    parser.add_argument("--clip-config", default="configs/clip_verifier.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--disable-verifier", action="store_true", help="detector-only benchmark")
    parser.add_argument("--save-alerts", action="store_true",
                        help="route alert decisions through FileAlertLink + RetryQueue (degraded-ops exercise)")
    parser.add_argument("--out", default="reports/benchmark")
    args = parser.parse_args()

    source_type = args.source_type
    if source_type == "auto":
        source_type = "csi" if str(args.source).isdigit() else (
            "dir" if Path(args.source).is_dir() else "video"
        )
    logger.info("source: %s (%s), %d iters after %d warmup", args.source, source_type, args.iters, args.warmup)

    pipeline = build_pipeline(
        REPO_ROOT / args.detector_config if not Path(args.detector_config).is_absolute() else Path(args.detector_config),
        (REPO_ROOT / args.clip_config) if (not Path(args.clip_config).is_absolute() and not args.disable_verifier) else None,
        weights=args.weights,
        device=args.device,
        include_verifier=not args.disable_verifier,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Warm-up (cold-start allocations, engine build, caches).
    for frame, _ in frame_source(args.source, source_type, args.warmup):
        pipeline.process_frame(frame)

    wall, detector_ms, verifier_ms, total_ms = [], [], [], []
    decisions = Counter()
    sources = Counter()
    gate_count = 0
    fallback_count = 0
    alert_link = FileAlertLink(out_dir / "alerts.jsonl") if args.save_alerts else None
    retry_queue = RetryQueue(out_dir / "retry_queue") if args.save_alerts else None

    with TegraStatsCapture() as tegrastats:
        for frame, name in frame_source(args.source, source_type, args.iters):
            start = time.perf_counter()
            result = pipeline.process_frame(frame, frame_id=name)
            wall.append((time.perf_counter() - start) * 1000.0)
            detector_ms.append(result.timings_ms.get("detector_ms"))
            verifier_ms.append(result.timings_ms.get("verifier_ms"))
            total_ms.append(result.timings_ms.get("total_ms"))
            decisions[result.decision] += 1
            sources[result.source] += 1
            gate_count += int(result.gate_passed)
            fallback_count += int(result.source == "yolo_only_fallback")
            if alert_link is not None and result.decision == "alert":
                send_with_retry(Alert.from_pipeline(result), alert_link, retry_queue)

    if tegrastats.lines:
        (out_dir / "tegrastats.log").write_text("\n".join(tegrastats.lines) + "\n", encoding="utf-8")

    wall_summary = summarize_latencies(wall)

    reference_benchmarks = load_reference_benchmarks()
    if reference_benchmarks.get("inference_latency_int8") and wall_summary.get("p50_ms"):
        from uav_fire_detection.utils.benchmarks import find_row

        detector_reference = find_row(
            reference_benchmarks["inference_latency_int8"], component="yolov8s_detector"
        )
        if detector_reference:
            logger.info(
                "measured detector p50 %.1f ms vs benchmark %s ms",
                wall_summary["p50_ms"],
                detector_reference["latency_ms"],
            )

    report = {
        "pipeline_version": __version__,
        "weights": args.weights,
        "source": args.source,
        "source_type": source_type,
        "verifier_enabled": pipeline.verifier is not None,
        "warmup_iters": args.warmup,
        "measured_iters": len(wall),
        "latency": {
            "wall": wall_summary,
            "detector_stage": summarize_latencies([v for v in detector_ms if v is not None]),
            "verifier_stage_when_invoked": summarize_latencies([v for v in verifier_ms if v is not None]),
            "total_pipeline": summarize_latencies([v for v in total_ms if v is not None]),
            "throughput_fps_median": 1000.0 / wall_summary["p50_ms"] if wall_summary.get("p50_ms") else None,
        },
        "clip_invocation_rate": gate_count / max(1, len(wall)),
        "yolo_only_fallback_frames": fallback_count,
        "decision_counts": dict(decisions),
        "source_counts": dict(sources),
        "tegrastats_captured": bool(tegrastats.lines),
        "reference_benchmarks": reference_benchmarks,
        "environment": collect(repo_dir=REPO_ROOT),
    }

    json_path = out_dir / "benchmark.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    logger.info("summary:\n%s", json.dumps(report["latency"], indent=2, default=str))
    logger.info("decision mix: %s | invocation rate: %.1f%%",
                dict(decisions), 100 * report["clip_invocation_rate"])
    logger.info("wrote %s", json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
