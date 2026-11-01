# Results

Status: pending evaluation — run the scripts below and fill in the tables.

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

| Metric | Measured | Reference (paper) |
|---|---|---:|
| mAP@0.5 | _(fill)_ | 0.935 |
| Precision | _(fill)_ | 0.931 |
| Recall | _(fill)_ | 0.954 |
| F1 | _(fill)_ | 0.942 |

## 3. Frame-level pipeline outcomes (from `scripts/evaluate.py`)

| Scope | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Detector-only baseline (gate level) | _(fill)_ | _(fill)_ | _(fill)_ | | | |
| Pipeline (alert = positive) | _(fill)_ | | | | | |
| Pipeline (alert+review = positive) | _(fill)_ | | | | | |

- FP delta vs baseline: _(fill)_ — reference: 147 → 116.
- FN delta vs baseline: _(fill)_ — reference: 77 → 89.
- CLIP invocation rate: _(fill)_ — reference: ~8–12%.
- YOLO-only fallback frames: _(fill)_.

## 4. Latency (from `scripts/benchmark_jetson.py`, median/p95/p99, after warm-up)

| Stage | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---:|---:|---:|
| Detector (INT8 engine) | | | |
| CLIP when invoked (FP16) | | | |
| End-to-end, mixed pipeline | | | |
| End-to-end, detector-only | | | |

Reference figures from the paper: detector INT8 18.4 ms; CLIP 74.6 ms when
invoked; end-to-end with conditional VLM 38.1 ms; throughput 26.2 FPS.

## 5. Field/telemetry behavior (if flight-tested)

GPS/telemetry accuracy, radio packet loss, queue overflow counts,
operator-review outcomes, altitude/speed/illumination context: _(fill)_.

## 6. Notes and deviations

Record any deviation from the reference configuration here (prompt
substitutions, calibration recipe, split differences) with pointers to the
configuration files in which each choice is set.
