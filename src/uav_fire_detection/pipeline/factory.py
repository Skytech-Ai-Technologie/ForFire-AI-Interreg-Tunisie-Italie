# SPDX-License-Identifier: Apache-2.0
"""Assemble a :class:`FireDetectionPipeline` from the repository YAML configs.

Shared by scripts/evaluate.py and scripts/benchmark_jetson.py so both build
the identical stack from identical configuration. Heavy imports happen inside
``build_pipeline`` so importing this module stays cheap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def _resolve_config_path(raw: str, config_dir: Path) -> Path:
    """Resolve a config-relative path: absolute, then CWD, then repo-style
    (relative to the configs' parent directory)."""
    p = Path(raw)
    if p.is_absolute():
        return p
    candidates = [Path.cwd() / p, config_dir.parent / p]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def build_pipeline(
    detector_config,
    clip_config=None,
    weights: Optional[str] = None,
    device: Optional[str] = None,
    include_verifier: bool = True,
):
    """Build the two-stage pipeline from detector (+ optional clip) configs.

    With ``include_verifier=False`` (or when ``clip_config`` is omitted) the
    pipeline runs detector-only with the documented reduced-confidence
    fallback behaviour.
    """
    from ..detector.detector import YoloDetector
    from ..utils.config import load_yaml
    from ..verifier.policy import PolicyConfig, VerificationPolicy
    from .pipeline import FireDetectionPipeline, PipelineConfig

    detector_cfg = load_yaml(detector_config)
    inference = detector_cfg["inference"]
    weights_path = str(weights or inference["weights"])
    detector = YoloDetector(
        weights_path,
        conf_threshold=float(inference.get("predict_conf", 0.25)),
        iou=float(inference.get("iou", 0.45)),
        imgsz=int(inference.get("imgsz", 640)),
        device=device,
    )

    clip_cfg = None
    verifier = None
    if include_verifier and clip_config is not None:
        from ..verifier.clip_verifier import ClipVerifier
        from ..verifier.prompts import PromptLibrary

        clip_cfg = load_yaml(clip_config)
        config_dir = Path(clip_config).resolve().parent
        prompts_path = _resolve_config_path(clip_cfg["prompts_file"], config_dir)
        model_cfg = clip_cfg["model"]
        verifier = ClipVerifier(
            PromptLibrary.from_yaml(prompts_path),
            model_name=str(model_cfg["model_name"]),
            pretrained=str(model_cfg["pretrained"]),
            fp16=str(model_cfg.get("precision", "fp16")).lower() == "fp16",
            device=device,
        )

    policy_cfg = (clip_cfg or {}).get("policy", {})
    policy = VerificationPolicy(
        PolicyConfig(
            alert_threshold=float(policy_cfg.get("alert_threshold", 0.60)),
            review_threshold=float(policy_cfg.get("review_threshold", 0.45)),
            suppress_threshold=float(policy_cfg.get("suppress_threshold", 0.70)),
        )
    )

    ver_cfg = (clip_cfg or {}).get("verification", {})
    pipeline_cfg = PipelineConfig(
        gate_threshold=float(inference.get("conf_threshold", 0.45)),
        gate_strict=True,
        verifier_enabled=verifier is not None,
        input_mode=str(ver_cfg.get("input_mode", "full_frame")),
        crop_margin_ratio=float(ver_cfg.get("crop_margin_ratio", 0.25)),
        clip_time_budget_ms=float(ver_cfg.get("time_budget_ms", 150.0)),
    )
    return FireDetectionPipeline(detector, verifier, policy, pipeline_cfg)
