# Benchmarks

Reference benchmark tables for the two-stage YOLOv8s + OpenCLIP fire-detection
pipeline (Jetson Orin Nano). One CSV table per file; `scripts/evaluate.py` and
`scripts/benchmark_jetson.py` load these tables and report them alongside
measured results.

| File | Content |
|---|---|
| `detection_performance.csv` | Accuracy / precision / recall / F1 / mAP@0.5 for YOLOv8n, YOLOv8s and YOLOv8s+CLIP |
| `two_stage_vs_single_stage.csv` | FP / FN / F1 / FPS / precision for detector-only vs two-stage at matched threshold 0.45 |
| `inference_latency_int8.csv` | Per-stage latency profile (preprocess, detector, CLIP when invoked, detector-only total) |
| `quantization_comparison.csv` | FP32 / FP16 / INT8 latency, FPS, accuracy drop, memory |
| `environmental_conditions.csv` | F1 per operating condition (clear flame … wind-driven smoke) |
| `detection_threshold_sweep.csv` | FP / FN / FPS / VLM-invocation rate vs detection threshold 0.30–0.60 |
| `prompt_library_size.csv` | Per-category VLM accuracy vs prompt-library size 6–24 |
| `yolov8_variants.csv` | YOLOv8 n/s/m/l parameters, mAP, latency, FPS, power |
| `flight_parameters.csv` | Altitude vs detection range, GSD, F1, patrol speed |
| `coverage_time.csv` | Swath and patrol time per 100 / 1,000 ha |
| `end_to_end_alert_latency.csv` | Stage-by-stage alert latency (capture → radio TX) |
| `power_budget.csv` | Hover power budget per component |
| `false_positive_categories.csv` | Residual false-positive categories |
| `false_negative_causes.csv` | False-negative cause categories |
| `dataset_composition.csv` | Per-biome images / fire / smoke instance counts |
| `augmentation.csv` | Training augmentation operations (probability, magnitude) |
| `hardware.csv` | Reference platform specifications |
| `software_stack.csv` | Reference software stack versions |

Load them in Python:

```python
from uav_fire_detection.utils.benchmarks import load_csv, find_row

rows = load_csv("benchmarks/detection_performance.csv")
row = find_row(rows, model="yolov8s_clip")   # 94.7 / 93.1 / 95.4 / 0.942 / 93.5
```

Measured results belong in `reports/results.md` (filled from
`scripts/evaluate.py` and `scripts/benchmark_jetson.py` outputs), never in
these reference tables.
