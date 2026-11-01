# SPDX-License-Identifier: Apache-2.0
"""Configuration invariants: reference values must not silently drift."""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]


def load(rel):
    return yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))


def test_clip_policy_thresholds():
    policy = load("configs/clip_verifier.yaml")["policy"]
    assert policy["alert_threshold"] == 0.60
    assert policy["review_threshold"] == 0.45
    assert policy["suppress_threshold"] == 0.70


def test_clip_verification_defaults():
    cfg = load("configs/clip_verifier.yaml")
    assert cfg["verification"]["on_budget_exceeded"] == "yolo_only_fallback"
    assert cfg["model"]["model_name"] == "ViT-B-32"
    assert cfg["model"]["precision"] == "fp16"
    assert cfg["prompts_file"] == "prompts/fire_smoke_prompts.yaml"


def test_detector_training_settings():
    cfg = load("configs/detector.yaml")
    assert cfg["model"]["architecture"] == "yolov8s"
    assert cfg["model"]["classes"] == ["fire", "smoke"]
    training = cfg["training"]
    assert training["epochs"] == 120
    assert training["early_stopping"]["patience"] == 15
    assert training["optimizer"] == "SGD"
    assert training["lr0"] == 0.01
    assert training["lr_schedule"] == "cosine"
    assert training["momentum"] == 0.937
    assert training["weight_decay"] == 5.0e-4
    assert training["fire_only_oversampling"] == 2.0


def test_detector_gate_threshold():
    inference = load("configs/detector.yaml")["inference"]
    assert inference["conf_threshold"] == 0.45


def test_training_runtime_and_augmentation_defaults():
    training = load("configs/detector.yaml")["training"]
    assert training["imgsz"] == 640
    assert training["seed"] == 0
    augmentation = training["augmentation"]
    assert augmentation["fliplr"] == 0.5
    assert augmentation["degrees"] == 15.0
    assert augmentation["hsv_h"] == 0.08
    assert augmentation["hsv_s"] == 0.12
    assert augmentation["hsv_v"] == 0.10
    assert augmentation["mosaic"] == 0.25


def test_evaluation_protocol_keys():
    cfg = load("configs/evaluation.yaml")
    assert cfg["frame_label_rule"] == "positive_if_any_fire_or_smoke_instance"
    assert cfg["alert_metrics"]["predicted_positive_decisions"] == ["alert"]
    assert 50 in cfg["latency"]["percentiles"]
    assert 95 in cfg["latency"]["percentiles"]
    assert 99 in cfg["latency"]["percentiles"]
