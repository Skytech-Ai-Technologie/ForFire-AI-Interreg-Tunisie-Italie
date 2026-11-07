# Results

Status: pending evaluation — run the scripts below and fill in the measured
columns. Reference columns come from the CSV tables in [`benchmarks/`](../benchmarks).

## 1. Environment (from `scripts/capture_environment.py`)

| Item | Value |
|---|---|
| Date | _(fill)_ |
| Host / board | _(fill)_ |
| JetPack / L4T | _(fill)_ |
| CUDA / TensorRT | _(fill)_ |
| Python | _(fill)_ |
| ultralytics / open_clip_torch | _(fill)_ |
| torch | _(fill)_ |
| Container digest | _(fill)_ |
| Git commit | _(fill)_ |

## 2. Detector metrics (from `scripts/evaluate.py`)

| Metric | Measured | Reference (`benchmarks/detection_performance.csv`, yolov8s_clip) |
|---|---|---:|
| Accuracy | _(fill)_ | 94.7% |
| Precision | _(fill)_ | 93.1% |
| Recall | _(fill)_ | 95.4% |
| F1 | _(fill)_ | 0.942 |
| mAP@0.5 | _(fill)_ | 93.5% |

## 3. Frame-level pipeline outcomes (from `scripts/evaluate.py`)

| Scope | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Detector-only baseline (gate level) | _(fill)_ | _(fill)_ | _(fill)_ | | | |
| Pipeline (alert = positive) | _(fill)_ | | | | | |
| Pipeline (alert+review = positive) | _(fill)_ | | | | | |

- FP / FN reference (`benchmarks/two_stage_vs_single_stage.csv`):
  yolov8s_only 147 / 77 → yolov8s_clip 116 / 89.
- CLIP invocation reference (`benchmarks/detection_threshold_sweep.csv`):
  threshold 0.45 → 10%.
- YOLO-only fallback frames: _(fill)_.

## 4. Latency (from `scripts/benchmark_jetson.py`, median/p95/p99, after warm-up)

| Stage | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---:|---:|---:|
| Detector (INT8 engine) | | | |
| CLIP when invoked (FP16) | | | |
| End-to-end, mixed pipeline | | | |
| End-to-end, detector-only | | | |

Reference latency profile (`benchmarks/inference_latency_int8.csv`):
preprocessing 2.1 ms; YOLOv8s detector 18.4 ms (54.3 FPS); CLIP when invoked
74.6 ms (13.4 FPS); detector-only total 21.2 ms (47.2 FPS). End-to-end alert
breakdown (`benchmarks/end_to_end_alert_latency.csv`): total 109.3 ms.

## 5. Field/telemetry behavior (if flight-tested)

GPS/telemetry accuracy, radio packet loss, queue overflow counts,
operator-review outcomes, altitude/speed/illumination context: _(fill)_.
Operational references: `benchmarks/flight_parameters.csv`,
`benchmarks/coverage_time.csv`, `benchmarks/power_budget.csv`,
`benchmarks/environmental_conditions.csv`.

## 6. Notes and deviations

Record any deviation from the reference benchmarks here (prompt
substitutions, calibration recipe, split differences) with pointers to the
configuration files in which each choice is set.
