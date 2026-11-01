#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Build dataset + split manifests.

Walks ``<dataset-root>/images/<source>/<session>/<image>`` with mirrored YOLO
labels, computes sha256/size/instance counts, and writes stratified
80/15/5 splits (whole sessions kept together by default to avoid
adjacent-frame leakage). Deterministic for a given seed and file set.

Usage::

    python scripts/build_manifests.py --dataset-root data/processed --seed 0
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from uav_fire_detection.utils.log import get_logger  # noqa: E402
from uav_fire_detection.utils.manifests import build_manifests  # noqa: E402

logger = get_logger("build_manifests")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-root", default="data/processed")
    parser.add_argument("--labels-root", default=None,
                        help="default: <dataset-root>/labels")
    parser.add_argument("--out-dir", default="manifests")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-group-by-session", action="store_true",
                        help="split per image instead of per session (risks leakage)")
    args = parser.parse_args()

    stats = build_manifests(
        dataset_root=args.dataset_root,
        labels_root=args.labels_root,
        out_dir=args.out_dir,
        seed=args.seed,
        group_by_session=not args.no_group_by_session,
    )
    logger.info("manifest stats:\n%s", json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
