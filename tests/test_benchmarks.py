# SPDX-License-Identifier: Apache-2.0
"""Reference benchmark tables (benchmarks/*.csv): parse + key rows."""

from pathlib import Path

import pytest

from uav_fire_detection.utils.benchmarks import find_row, load_csv

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"

EXPECTED_FILES = (
    "detection_performance.csv",
    "two_stage_vs_single_stage.csv",
    "inference_latency_int8.csv",
    "quantization_comparison.csv",
    "environmental_conditions.csv",
    "detection_threshold_sweep.csv",
    "prompt_library_size.csv",
    "yolov8_variants.csv",
    "flight_parameters.csv",
    "coverage_time.csv",
    "end_to_end_alert_latency.csv",
    "power_budget.csv",
    "false_positive_categories.csv",
    "false_negative_causes.csv",
    "dataset_composition.csv",
    "augmentation.csv",
    "hardware.csv",
    "software_stack.csv",
)


def test_all_benchmark_tables_parse():
    for name in EXPECTED_FILES:
        rows = load_csv(BENCHMARKS / name)
        assert rows, "empty benchmark table: {}".format(name)


def test_detection_performance_rows():
    rows = load_csv(BENCHMARKS / "detection_performance.csv")
    two_stage = find_row(rows, model="yolov8s_clip")
    assert two_stage["accuracy_pct"] == "94.7"
    assert two_stage["precision_pct"] == "93.1"
    assert two_stage["recall_pct"] == "95.4"
    assert two_stage["f1"] == "0.942"
    assert two_stage["map50_pct"] == "93.5"
    single_stage = find_row(rows, model="yolov8s")
    assert single_stage["map50_pct"] == "91.7"


def test_two_stage_vs_single_stage_rows():
    rows = load_csv(BENCHMARKS / "two_stage_vs_single_stage.csv")
    baseline = find_row(rows, system="yolov8s_only")
    ours = find_row(rows, system="yolov8s_clip")
    assert baseline["fp"] == "147" and baseline["fn"] == "77"
    assert ours["fp"] == "116" and ours["fn"] == "89"
    assert ours["f1"] == "0.942"


def test_latency_profile_rows():
    rows = load_csv(BENCHMARKS / "inference_latency_int8.csv")
    detector = find_row(rows, component="yolov8s_detector")
    clip = find_row(rows, component="clip_vlm_when_invoked")
    total = find_row(rows, component="detector_only_total")
    assert detector["latency_ms"] == "18.4"
    assert clip["latency_ms"] == "74.6"
    assert total["latency_ms"] == "21.2"


def test_threshold_sweep_operating_point():
    rows = load_csv(BENCHMARKS / "detection_threshold_sweep.csv")
    operating_point = find_row(rows, threshold="0.45")
    assert operating_point["fp"] == "116"
    assert operating_point["fn"] == "89"
    assert operating_point["vlm_invocation_pct"] == "10"


def test_end_to_end_alert_latency_total():
    rows = load_csv(BENCHMARKS / "end_to_end_alert_latency.csv")
    total = find_row(rows, stage="total")
    assert total["latency_ms"] == "109.3"
    stages = [r for r in rows if r["stage"] != "total"]
    assert abs(sum(float(r["latency_ms"]) for r in stages) - 109.3) < 0.05


def test_power_budget_total():
    rows = load_csv(BENCHMARKS / "power_budget.csv")
    total = find_row(rows, component="total")
    assert total["power_w"] == "309.5"
    components = [r for r in rows if r["component"] != "total"]
    assert abs(sum(float(r["power_w"]) for r in components) - 309.5) < 0.05


def test_missing_table_raises():
    with pytest.raises(FileNotFoundError):
        load_csv(BENCHMARKS / "does_not_exist.csv")
