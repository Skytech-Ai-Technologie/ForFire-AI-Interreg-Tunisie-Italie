#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Capture an environment snapshot.

Records Python/platform/JetPack/CUDA/package versions, container detection,
and git commit into a JSON file so every reported result can name its
environment.

Usage::

    python scripts/capture_environment.py --out reports/environment.json
    python scripts/capture_environment.py --print-only   # Docker sanity check
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from uav_fire_detection.utils.environment import collect, write_snapshot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="reports/environment.json")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    if args.print_only:
        print(json.dumps(collect(repo_dir=REPO_ROOT), indent=2, sort_keys=True, default=str))
        return 0

    info = write_snapshot(args.out, repo_dir=REPO_ROOT)
    print(json.dumps(info, indent=2, sort_keys=True, default=str))
    print("written: {}".format(args.out), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
