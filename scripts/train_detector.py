#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Train the fire/smoke YOLOv8s detector.

Applies the training configuration from configs/detector.yaml and writes a
``run_manifest.json`` (config snapshot, seed posture, manifest hashes,
package versions, git commit) next to the run.

Usage::

    python scripts/train_detector.py --dataset-root data/processed --device 0
    python scripts/train_detector.py --dry-run   # print resolved arguments
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from uav_fire_detection.utils.config import load_yaml  # noqa: E402
from uav_fire_detection.utils.dataset import build_ultralytics_dataset_yaml  # noqa: E402
from uav_fire_detection.utils.environment import collect  # noqa: E402
from uav_fire_detection.utils.log import get_logger  # noqa: E402
from uav_fire_detection.utils.seed import set_seed  # noqa: E402

logger = get_logger("train_detector")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/detector.yaml")
    parser.add_argument("--split-manifest", default="manifests/split_manifest.csv")
    parser.add_argument("--dataset-root", default="data/processed")
    parser.add_argument("--output", default="runs/detector")
    parser.add_argument("--name", default="train")
    parser.add_argument("--device", default=None, help="e.g. 0 or cpu (default: auto)")
    parser.add_argument("--dry-run", action="store_true", help="print resolved arguments and exit")
    args = parser.parse_args()

    cfg = load_yaml(_resolve(args.config))
    training = cfg["training"]

    dataset_root = _resolve(args.dataset_root)
    split_manifest = _resolve(args.split_manifest)
    output_dir = _resolve(args.output) / args.name
    ds = build_ultralytics_dataset_yaml(
        split_manifest,
        out_dir=output_dir / "datasets",
        dataset_root=dataset_root,
        oversample_fire_only_ratio=float(training["fire_only_oversampling"]),
    )
    logger.info("dataset list stats: %s", json.dumps(ds["stats"]))

    seed_info = set_seed(int(training["seed"]), deterministic=bool(training["deterministic"]))

    augmentation = training.get("augmentation", {})
    train_kwargs = {
        "data": ds["yaml_path"],
        "epochs": int(training["epochs"]),
        "patience": int(training["early_stopping"]["patience"]),
        "optimizer": str(training["optimizer"]),
        "lr0": float(training["lr0"]),
        "cos_lr": str(training["lr_schedule"]).lower() == "cosine",
        "lrf": float(training["lrf"]),
        "warmup_epochs": float(training["warmup_epochs"]),
        "momentum": float(training["momentum"]),
        "weight_decay": float(training["weight_decay"]),
        "imgsz": int(training["imgsz"]),
        "batch": int(training["batch"]),
        "seed": int(training["seed"]),
        "deterministic": bool(training["deterministic"]),
        "workers": int(training["workers"]),
        "fliplr": float(augmentation.get("fliplr", 0.5)),
        "degrees": float(augmentation.get("degrees", 15.0)),
        "hsv_h": float(augmentation.get("hsv_h", 0.08)),
        "hsv_s": float(augmentation.get("hsv_s", 0.12)),
        "hsv_v": float(augmentation.get("hsv_v", 0.10)),
        "mosaic": float(augmentation.get("mosaic", 0.25)),
        "close_mosaic": int(augmentation.get("close_mosaic", 10)),
        "project": str(output_dir.parent),
        "name": args.name,
        "exist_ok": True,
        "device": args.device,
    }

    logger.info("resolved training arguments:\n%s", json.dumps(train_kwargs, indent=2, default=str))
    if args.dry_run:
        logger.info("dry run: not training")
        return 0

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics is required: pip install 'uav-fire-detection[detector]'")
        return 2

    model = YOLO(cfg["model"]["pretrained_weights"])
    model.train(**train_kwargs)

    save_dir = Path(getattr(model.trainer, "save_dir", output_dir))
    best = save_dir / "weights" / "best.pt"
    run_manifest = {
        "config_snapshot": cfg,
        "train_kwargs": {k: str(v) for k, v in train_kwargs.items()},
        "dataset": {
            "yaml": ds["yaml_path"],
            "split_manifest": str(split_manifest),
            "split_manifest_sha256": _sha256(split_manifest) if split_manifest.exists() else None,
            "stats": ds["stats"],
        },
        "seed": seed_info,
        "environment": collect(repo_dir=REPO_ROOT),
        "best_weights": str(best) if best.exists() else None,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    logger.info("training complete; best weights: %s", best if best.exists() else "(not found)")
    logger.info("run manifest: %s", output_dir / "run_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
