#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Evaluate detector and full pipeline on a frozen split.

Computes:

* Ultralytics detector validation metrics (mAP@0.5, per-class P/R) via
  ``model.val``;
* frame-level alert metrics under the configured labeling rule (positive
  frame = at least one fire/smoke instance): TP/FP/FN, precision/recall/F1
  for the detector-only baseline (gate level) and for the full pipeline,
  plus the FP/FN deltas introduced by CLIP suppression;
* CLIP invocation rate, review/suppress/fallback counts, and latency
  distributions (median/p95/p99).

Everything is written to JSON + Markdown with environment capture.

Usage::

    python scripts/evaluate.py --weights runs/detector/train/weights/best.pt --device 0
    python scripts/evaluate.py --mode detector_only --max-images 100
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from uav_fire_detection import __version__  # noqa: E402
from uav_fire_detection.pipeline.factory import build_pipeline  # noqa: E402
from uav_fire_detection.utils.config import load_yaml  # noqa: E402
from uav_fire_detection.utils.dataset import build_ultralytics_dataset_yaml, read_split_rows  # noqa: E402
from uav_fire_detection.utils.environment import collect  # noqa: E402
from uav_fire_detection.utils.imaging import imread_bgr  # noqa: E402
from uav_fire_detection.utils.log import get_logger  # noqa: E402
from uav_fire_detection.utils.timing import summarize_latencies  # noqa: E402

logger = get_logger("evaluate")

# Reference results reported in the paper (Jabli et al.), for comparison.
REFERENCE_RESULTS = {
    "accuracy": 0.947, "precision": 0.931, "recall": 0.954, "f1": 0.942,
    "map50": 0.935, "throughput_fps": 26.2, "e2e_latency_ms": 38.1,
    "yolo_int8_latency_ms": 18.4, "clip_latency_ms": 74.6,
    "invocation_rate": "8-12%", "fp_yolo_only": 147, "fp_yolo_clip": 116,
    "fn_yolo_only": 77, "fn_yolo_clip": 89,
}


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)


def confusion(records, predicate):
    tp = fp = fn = tn = 0
    for r in records:
        pred = predicate(r)
        label = r["label_positive"]
        if pred and label:
            tp += 1
        elif pred and not label:
            fp += 1
        elif not pred and label:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1}


def bootstrap_ci(records, predicate, key, n_iterations, seed):
    """Percentile bootstrap CIs for precision/recall/f1 (numpy if available)."""
    try:
        import numpy as np
    except ImportError:
        return None
    rng = np.random.default_rng(seed)
    n = len(records)
    if n == 0:
        return None
    labels = np.array([r["label_positive"] for r in records])
    preds = np.array([predicate(r) for r in records])
    out = {k: [] for k in ("precision", "recall", "f1")}
    for _ in range(n_iterations):
        idx = rng.integers(0, n, n)
        b_labels, b_preds = labels[idx], preds[idx]
        tp = int(((b_preds) & (b_labels)).sum())
        fp = int(((b_preds) & (~b_labels)).sum())
        fn = int(((~b_preds) & (b_labels)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out["precision"].append(p)
        out["recall"].append(r)
        out["f1"].append(f)
    return {
        key: {
            metric: [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
            for metric, vals in out.items()
        }
    }


def run_detector_validation(weights, split_manifest, dataset_root, split, imgsz, out_dir):
    """Ultralytics val on the frozen split; returns results_dict or an error."""
    try:
        from ultralytics import YOLO

        ds = build_ultralytics_dataset_yaml(
            split_manifest, out_dir=out_dir / "datasets",
            dataset_root=dataset_root, splits=("train", "val", split),
        )
        metrics = YOLO(str(weights)).val(
            data=ds["yaml_path"], split=split, imgsz=imgsz, verbose=False
        )
        return {k: float(v) for k, v in metrics.results_dict.items()}
    except Exception as exc:  # noqa: BLE001 - report, don't crash the whole eval
        return {"error": "{}: {}".format(type(exc).__name__, exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--detector-config", default="configs/detector.yaml")
    parser.add_argument("--clip-config", default="configs/clip_verifier.yaml")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--mode", choices=["both", "pipeline", "detector_only"], default="both")
    parser.add_argument("--split", default=None, help="override evaluation.yaml data.split")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    ev = load_yaml(_resolve(args.config))
    data = ev["data"]
    split_manifest = _resolve(data["split_manifest"])
    dataset_root = _resolve(data["dataset_root"])
    split = args.split or data.get("split", "test")

    rows = [r for r in read_split_rows(split_manifest) if r["split"] == split]
    if not rows:
        logger.error("no rows for split %r in %s", split, split_manifest)
        return 1
    if args.max_images:
        rows = rows[: args.max_images]
    logger.info("evaluating %d frames from split %r", len(rows), split)

    weights = args.weights or _resolve(load_yaml(_resolve(args.detector_config))["inference"]["weights"])
    include_verifier = args.mode in ("pipeline", "both")
    pipeline = build_pipeline(
        _resolve(args.detector_config),
        _resolve(args.clip_config) if include_verifier else None,
        weights=str(weights),
        device=args.device,
        include_verifier=include_verifier,
    )

    out_dir = Path(args.out_dir) if args.out_dir else _resolve(ev["outputs"]["directory"])
    out_dir.mkdir(parents=True, exist_ok=True)

    detector_metrics = None
    if args.mode in ("detector_only", "both"):
        logger.info("running Ultralytics validation on split %r ...", split)
        detector_metrics = run_detector_validation(
            weights, split_manifest, dataset_root, split,
            int(load_yaml(_resolve(args.detector_config))["inference"]["imgsz"]), out_dir,
        )

    logger.info("running frame-level pipeline evaluation ...")
    records = []
    for i, row in enumerate(rows):
        image_path = dataset_root / "images" / row["image_path"]
        try:
            frame = imread_bgr(image_path)
        except (IOError, OSError) as exc:
            logger.warning("skipping unreadable image %s: %s", image_path, exc)
            continue
        start = time.perf_counter()
        result = pipeline.process_frame(frame, frame_id=row["image_path"])
        wall_ms = (time.perf_counter() - start) * 1000.0
        records.append(
            {
                "image_path": row["image_path"],
                "label_positive": (int(row.get("n_fire") or 0) + int(row.get("n_smoke") or 0)) > 0,
                "decision": result.decision,
                "source": result.source,
                "gate_passed": result.gate_passed,
                "reduced_confidence": result.reduced_confidence,
                "wall_ms": wall_ms,
                "detector_ms": result.timings_ms.get("detector_ms"),
                "verifier_ms": result.timings_ms.get("verifier_ms"),
                "total_ms": result.timings_ms.get("total_ms"),
            }
        )
        if (i + 1) % 100 == 0:
            logger.info("  %d / %d frames", i + 1, len(rows))

    if not records:
        logger.error("no frames evaluated; nothing to report")
        return 1

    bootstrap_cfg = ev.get("bootstrap", {})
    alert_predicate = lambda r: r["decision"] == "alert"  # noqa: E731
    review_positive_predicate = lambda r: r["decision"] in ("alert", "review")  # noqa: E731
    baseline_predicate = lambda r: r["gate_passed"]  # noqa: E731

    outcomes = {
        "detector_only_baseline": confusion(records, baseline_predicate),
        "pipeline_alert_level": confusion(records, alert_predicate),
        "pipeline_alert_plus_review": confusion(records, review_positive_predicate),
    }
    decisions = Counter(r["decision"] for r in records)
    sources = Counter(r["source"] for r in records)
    invocation_rate = sum(1 for r in records if r["gate_passed"]) / len(records)
    fallback_count = sum(1 for r in records if r["source"] == "yolo_only_fallback")

    for name, predicate, key in (
        ("pipeline_alert_level", alert_predicate, "alert"),
        ("detector_only_baseline", baseline_predicate, "baseline"),
    ):
        ci = bootstrap_ci(
            records, predicate, key,
            int(bootstrap_cfg.get("n_iterations", 1000)), int(bootstrap_cfg.get("seed", 0)),
        )
        if ci:
            outcomes[name]["bootstrap_ci_95"] = ci[key]

    latency = {
        "wall": summarize_latencies([r["wall_ms"] for r in records]),
        "detector_stage": summarize_latencies(
            [r["detector_ms"] for r in records if r["detector_ms"] is not None]
        ),
        "verifier_stage_when_invoked": summarize_latencies(
            [r["verifier_ms"] for r in records if r["verifier_ms"] is not None]
        ),
        "total_pipeline": summarize_latencies(
            [r["total_ms"] for r in records if r["total_ms"] is not None]
        ),
    }
    latency["throughput_fps_median"] = (
        1000.0 / latency["wall"]["p50_ms"] if latency["wall"].get("p50_ms") else None
    )

    report = {
        "pipeline_version": __version__,
        "mode": args.mode,
        "split": split,
        "n_frames": len(records),
        "weights": str(weights),
        "detector_metrics_ultralytics_val": detector_metrics,
        "frame_level_outcomes": outcomes,
        "decision_counts": dict(decisions),
        "source_counts": dict(sources),
        "clip_invocation_rate": invocation_rate,
        "yolo_only_fallback_frames": fallback_count,
        "false_negative_delta_vs_baseline": (
            outcomes["pipeline_alert_level"]["fn"] - outcomes["detector_only_baseline"]["fn"]
        ),
        "false_positive_delta_vs_baseline": (
            outcomes["pipeline_alert_level"]["fp"] - outcomes["detector_only_baseline"]["fp"]
        ),
        "latency": latency,
        "reference_results": REFERENCE_RESULTS,
        "environment": collect(repo_dir=REPO_ROOT),
    }

    json_path = out_dir / "evaluation_{}.json".format(split)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    md_path = out_dir / "evaluation_{}.md".format(split)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    logger.info("wrote %s and %s", json_path, md_path)
    return 0


def render_markdown(report: dict) -> str:
    base = outcomes = report["frame_level_outcomes"]
    lines = [
        "# Evaluation results",
        "",
        "- Mode: `{}` — split: `{}` — frames: {}".format(
            report["mode"], report["split"], report["n_frames"]
        ),
        "- Environment: Python {} on {}".format(
            report["environment"]["python"], report["environment"]["platform"]
        ),
        "",
        "## Frame-level outcomes",
        "",
        "| Scope | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, label in (
        ("detector_only_baseline", "Detector-only baseline (gate level)"),
        ("pipeline_alert_level", "Pipeline (alert = positive)"),
        ("pipeline_alert_plus_review", "Pipeline (alert+review = positive)"),
    ):
        o = base[name]
        lines.append(
            "| {} | {} | {} | {} | {:.3f} | {:.3f} | {:.3f} |".format(
                label, o["tp"], o["fp"], o["fn"], o["precision"], o["recall"], o["f1"]
            )
        )
    lines += [
        "",
        "- CLIP invocation rate: {:.1%}".format(report["clip_invocation_rate"]),
        "- YOLO-only fallback frames: {} (reduced-confidence)".format(report["yolo_only_fallback_frames"]),
        "- FP delta vs baseline: {:+d}".format(report["false_positive_delta_vs_baseline"]),
        "- FN delta vs baseline: {:+d}".format(report["false_negative_delta_vs_baseline"]),
        "",
        "## Latency (ms, wall clock on this machine)",
        "",
        "```json",
        json.dumps(report["latency"], indent=2, default=str),
        "```",
        "",
        "## Reference results (reported in the paper)",
        "",
        "```json",
        json.dumps(report["reference_results"], indent=2),
        "```",
        "",
        "## Detector validation (Ultralytics, this split)",
        "",
        "```json",
        json.dumps(report["detector_metrics_ultralytics_val"], indent=2, default=str),
        "```",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
